"""Tests for the scout module."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import requests

from adam_core import _rust_native
from adam_core._rust.arrow import table_from_record_batch
from adam_core.observations.obs80 import ScoutObservations
from adam_core.orbits.variants import VariantOrbits

from ..scout import (
    ScoutObjectNotFoundError,
    ScoutObjectSummary,
    ScoutOrbit,
    ScoutResponseError,
    ScoutServiceUnavailableError,
    _request_scout_json,
    query_scout,
    query_scout_observations,
    scout_orbits_to_variant_orbits,
)

SCOUT_DATA = Path(__file__).parent / "testdata" / "scout"


def test_scout_request_retries_transient_failure() -> None:
    class Response:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

        def json(self) -> dict:
            return {"ok": True}

    responses = iter([Response(502), Response(200)])
    payload = _request_scout_json(
        http_get=lambda *args, **kwargs: next(responses), retry_delay_s=0
    )
    assert payload == {"ok": True}


def _orbit_payload() -> dict:
    return {
        "signature": {"version": "1.3", "source": "NASA/JPL Scout API"},
        "orbits": {
            "data": [
                [
                    0,
                    "2457581.871499164",
                    "0.3357123709445450",
                    "0.9083681207232809",
                    "2457636.871738402",
                    "111.41193497296813",
                    "244.46138666195648",
                    "16.61087545506555",
                    "24.694617",
                    None,
                    None,
                    "0.0321364995",
                    None,
                    "1.0e99",
                    0,
                ]
            ]
        },
    }


def test_query_scout_accepts_scalar_object_id() -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return _orbit_payload()

    def get(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    variants = query_scout("A11EpSe", http_get=get)

    assert len(variants) == 1
    assert variants.object_id.to_pylist() == ["A11EpSe"]
    assert calls[0][1]["params"] == {"tdes": "A11EpSe", "orbits": "1"}
    np.testing.assert_allclose(
        variants.coordinates.values[0],
        [
            0.3814075769793518,
            -0.990884410762772,
            0.0019868502438316893,
            0.014258742327653812,
            0.010562800146262947,
            -0.005110445147902333,
        ],
        rtol=0.0,
        atol=2e-15,
    )


def test_query_scout_maps_error_payload_to_structured_not_found() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"error": "specified object was not found"}

    with pytest.raises(ScoutObjectNotFoundError) as caught:
        query_scout("vanished", http_get=lambda *args, **kwargs: Response())

    assert caught.value.object_id == "vanished"
    assert caught.value.error_type == "not_found"
    assert caught.value.http_status == 404
    assert caught.value.retryable is False


def test_query_scout_maps_transient_error_payload_to_503() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"error": "service temporarily unavailable; try again"}

    with pytest.raises(ScoutServiceUnavailableError) as caught:
        query_scout("A11EpSe", http_get=lambda *args, **kwargs: Response())

    assert caught.value.object_id == "A11EpSe"
    assert caught.value.http_status == 503
    assert caught.value.retryable is True


def test_query_scout_maps_exhausted_transient_http_to_503() -> None:
    calls = 0

    class Response:
        status_code = 503

        def raise_for_status(self) -> None:
            raise requests.HTTPError(response=self)

    def get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response()

    with pytest.raises(ScoutServiceUnavailableError) as caught:
        query_scout(
            "A11EpSe",
            http_get=get,
            max_attempts=2,
            retry_delay_s=0,
        )

    assert calls == 2
    assert caught.value.object_id == "A11EpSe"
    assert caught.value.error_type == "service_unavailable"
    assert caught.value.http_status == 503
    assert caught.value.upstream_status == 503
    assert caught.value.retryable is True


def test_query_scout_rejects_missing_orbits_and_signature() -> None:
    class Response:
        def __init__(self, payload: dict):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.payload

    with pytest.raises(ScoutResponseError, match="signature"):
        query_scout(
            "A11EpSe",
            http_get=lambda *args, **kwargs: Response({"orbits": {"data": []}}),
        )
    with pytest.raises(ScoutResponseError, match="orbits.data"):
        query_scout(
            "A11EpSe",
            http_get=lambda *args, **kwargs: Response(
                {
                    "signature": {"version": "1.3"},
                    "orbits": {},
                }
            ),
        )
    with pytest.raises(ScoutResponseError, match="invalid orbit samples"):
        query_scout(
            "A11EpSe",
            http_get=lambda *args, **kwargs: Response(
                {
                    "signature": {"version": "1.3"},
                    "orbits": {"data": [["not", "a", "ScoutOrbit"]]},
                }
            ),
        )


def test_query_scout_observations_uses_file_membership() -> None:
    lines = [
        "     A11EpSe*0C2026 07 08.17725719 41 24.185-30 19 19.42         19.35oVNEOCPW68",
        "     A11EpSe KC2026 07 14.53636 19 37 22.30 -29 16 44.5          19.0 GVNEOCPE23",
    ]

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "objectName": "A11EpSe",
                # Deliberately differs: file=mpc, not nObs, is authoritative.
                "nObs": 1,
                "lastRun": "2026-07-14 13:33",
                "signature": {
                    "version": "1.3",
                    "source": "NASA/JPL Scout API",
                },
                "fileMPC": "\n".join(lines) + "\n",
            }

    calls = []

    def get(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    observations = query_scout_observations(["A11EpSe"], http_get=get)

    assert len(observations) == 2
    assert observations.object_id.to_pylist() == ["A11EpSe", "A11EpSe"]
    assert observations.declared_n_obs.to_pylist() == [1, 1]
    assert observations.snapshot_observation_count.to_pylist() == [2, 2]
    assert observations.observation_index.to_pylist() == [0, 1]
    assert observations.observation.designation.to_pylist() == [
        "A11EpSe",
        "A11EpSe",
    ]
    assert observations.observation.time.scale == "utc"
    assert len(set(observations.snapshot_sha256.to_pylist())) == 1
    assert observations.signature_version.to_pylist() == ["1.3", "1.3"]
    assert calls[0][1]["params"] == {"tdes": "A11EpSe", "file": "mpc"}


def test_scout_orbits_to_variant_orbits():
    """Test that scout orbits are correctly converted to variant orbits."""
    # Create a mock scout orbits table
    scout_data = {
        "idx": [0, 1],
        "epoch": ["60000.0", "60000.0"],
        "ec": ["0.5", "0.51"],
        "qr": ["1.0", "1.01"],
        "tp": ["59000.0", "59000.0"],
        "om": ["10.0", "10.1"],
        "w": ["50.0", "50.1"],
        "inc": ["10.0", "10.1"],
        "H": ["20.0", "20.0"],
        "dca": ["0.1", "0.1"],
        "tca": ["0.1", "0.1"],
        "moid": ["0.1", "0.1"],
        "vinf": ["0.1", "0.1"],
        "geoEcc": ["0.1", "0.1"],
        "impFlag": [0, 0],
    }
    scout_orbits = ScoutOrbit.from_kwargs(**scout_data)

    # Convert to variant orbits
    variant_orbits = scout_orbits_to_variant_orbits("2024AA", scout_orbits)

    # Check that the output has the expected structure
    assert len(variant_orbits) == len(scout_orbits)
    assert variant_orbits.coordinates.frame == "ecliptic"
    assert pc.all(pc.equal(variant_orbits.coordinates.origin.code, "SUN")).as_py()

    # Check that the object IDs are correct
    assert variant_orbits.object_id.to_pylist() == ["2024AA", "2024AA"]

    # Check that the orbit IDs are correct
    assert variant_orbits.orbit_id.to_pylist() == ["0", "1"]

    # Check that the variant IDs are unique
    assert len(pc.unique(variant_orbits.variant_id)) == len(scout_orbits)

    # Check that the time is correct
    np.testing.assert_array_equal(
        variant_orbits.coordinates.time.jd(), pc.cast(scout_orbits.epoch, pa.float64())
    )


def test_recorded_scout_summary_and_orbits_are_rust_owned():
    summary_payload = (SCOUT_DATA / "summary.json").read_text()
    summary = table_from_record_batch(
        ScoutObjectSummary,
        _rust_native.get_scout_objects_arrow(summary_payload),
    )
    assert len(summary) > 0
    object_id = summary.objectName[0].as_py()

    variants = table_from_record_batch(
        VariantOrbits,
        _rust_native.query_scout_arrow(
            [object_id], [(SCOUT_DATA / "orbits.json").read_text()]
        ),
    )
    assert len(variants) == 1000
    assert set(variants.object_id.to_pylist()) == {object_id}
    assert np.all(np.isfinite(variants.coordinates.values))


def _recorded_scout_observations_payload() -> str:
    lines = [
        "     A11EpSe*0C2026 07 08.17725719 41 24.185-30 19 19.42         19.35oVNEOCPW68",
        "     A11EpSe KC2026 07 14.53636 19 37 22.30 -29 16 44.5          19.0 GVNEOCPE23",
    ]
    return json.dumps(
        {
            "objectName": "A11EpSe",
            "nObs": 1,
            "lastRun": "2026-07-14 13:33",
            "signature": {"version": "1.3", "source": "NASA/JPL Scout API"},
            "fileMPC": "\n".join(lines) + "\n",
        }
    )


def test_default_public_scout_paths_are_one_native_crossing(monkeypatch):
    orbit_payload = _orbit_payload()
    orbit_payload["orbits"]["fields"] = list(ScoutOrbit.schema.names)
    orbit_batch = _rust_native.query_scout_arrow(
        ["A11EpSe"], [json.dumps(orbit_payload)]
    )
    observation_batch = _rust_native.query_scout_observations_arrow(
        ["A11EpSe"], [_recorded_scout_observations_payload()]
    )
    calls = []

    def query_orbits(*args):
        calls.append(("orbits", args))
        return orbit_batch

    def query_observations(*args):
        calls.append(("observations", args))
        return observation_batch

    monkeypatch.setattr(_rust_native, "query_scout_arrow", query_orbits)
    monkeypatch.setattr(
        _rust_native, "query_scout_observations_arrow", query_observations
    )

    variants = query_scout("A11EpSe")
    observations = query_scout_observations("A11EpSe")
    assert len(variants) == 1
    assert len(observations) == 2
    assert [kind for kind, _ in calls] == ["orbits", "observations"]
    assert calls[0][1][0] == ["A11EpSe"]
    assert calls[1][1][0] == ["A11EpSe"]


def test_recorded_scout_observations_are_rust_owned():
    observations = table_from_record_batch(
        ScoutObservations,
        _rust_native.query_scout_observations_arrow(
            ["A11EpSe"], [_recorded_scout_observations_payload()]
        ),
    )
    assert len(observations) == 2
    assert observations.observation.designation.to_pylist() == [
        "A11EpSe",
        "A11EpSe",
    ]
    assert observations.snapshot_observation_count.to_pylist() == [2, 2]
    assert observations.observation.time.scale == "utc"


def test_scout_products_have_rust_owned_timing():
    for kind, payload in [
        (
            "neocc",
            (Path(__file__).parent / "testdata" / "neocc" / "2024YR4.ke1").read_text(),
        ),
        ("scout-summary", (SCOUT_DATA / "summary.json").read_text()),
        ("scout", (SCOUT_DATA / "orbits.json").read_text()),
        ("scout-observations", _recorded_scout_observations_payload()),
    ]:
        samples = np.asarray(
            _rust_native.benchmark_query_client_processing(kind, [payload], 2, 2, 1)
        )
        assert samples.shape == (2, 2)
        assert np.all(samples > 0.0)
