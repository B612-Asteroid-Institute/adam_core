"""Tests for the scout module."""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import requests

from ..scout import (
    ScoutOrbit,
    _request_scout_json,
    query_scout_observations,
    scout_orbits_to_variant_orbits,
)


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
