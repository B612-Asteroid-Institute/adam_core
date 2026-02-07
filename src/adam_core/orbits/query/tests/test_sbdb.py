import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
from astroquery.jplsbdb import SBDB

from ..sbdb import NotFoundError, _convert_SBDB_covariances, query_sbdb, query_sbdb_new


def test__convert_SBDB_covariances():
    sbdb_format = np.array(
        [
            ["e_e", "e_q", "e_tp", "e_raan", "e_ap", "e_i"],
            ["q_e", "q_q", "q_tp", "q_raan", "q_ap", "q_i"],
            ["tp_e", "tp_q", "tp_tp", "tp_raan", "tp_ap", "tp_i"],
            ["raan_e", "raan_q", "raan_tp", "raan_raan", "raan_ap", "raan_i"],
            ["ap_e", "ap_q", "ap_tp", "ap_raan", "ap_ap", "ap_i"],
            ["i_e", "i_q", "i_tp", "i_raan", "i_ap", "i_i"],
        ]
    )
    adam_core_format = np.array(
        [
            ["q_q", "q_e", "q_i", "q_raan", "q_ap", "q_tp"],
            ["e_q", "e_e", "e_i", "e_raan", "e_ap", "e_tp"],
            ["i_q", "i_e", "i_i", "i_raan", "i_ap", "i_tp"],
            ["raan_q", "raan_e", "raan_i", "raan_raan", "raan_ap", "raan_tp"],
            ["ap_q", "ap_e", "ap_i", "ap_raan", "ap_ap", "ap_tp"],
            ["tp_q", "tp_e", "tp_i", "tp_raan", "tp_ap", "tp_tp"],
        ]
    )
    sbdb_format = np.array([sbdb_format])
    adam_core_format = np.array([adam_core_format])

    cometary_covariances = _convert_SBDB_covariances(sbdb_format)
    # Some of the symmetric terms will be in reverse format: for example
    # q_e will be e_q where it should be q_e. So for places where
    # the order is reversed, let's try to flip them around and then lets
    # check for equality with the expected adam format.

    flip_mask = np.where(cometary_covariances != adam_core_format)
    for i, j, k in zip(*flip_mask):
        cometary_covariances[i, j, k] = "_".join(
            cometary_covariances[i, j, k].split("_")[::-1]
        )

    npt.assert_equal(cometary_covariances, adam_core_format)


def test_query_sbdb_for_ceres():
    with mock_sbdb_query("Ceres.json") as mock:
        result = query_sbdb(["Ceres"])
        mock.assert_called_once()

    assert len(result) == 1


def test_query_sbdb_for_2001vb():
    with mock_sbdb_query("2001VB.json") as mock:
        result = query_sbdb(["2001VB"])
        mock.assert_called_once()

    assert len(result) == 1


def test_query_sbdb_for_54509():
    # 54509 has a 7x7 covariance matrix so lets test that we
    # correctly convert it to the 6x6 format
    with mock_sbdb_query("54509.json") as mock:
        assert mock.return_value["orbit"]["covariance"]["data"].shape == (7, 7)
        assert mock.return_value["orbit"]["covariance"]["labels"] == [
            "e",
            "q",
            "tp",
            "node",
            "peri",
            "i",
            "A2",
        ]
        result = query_sbdb(["54509"])
        mock.assert_called_once()

    assert len(result) == 1
    assert result.coordinates.covariance.to_matrix().shape == (1, 6, 6)
    assert not np.isnan(result.coordinates.covariance.to_matrix()).all()


def test_query_sbdb_for_missing_value():
    with pytest.raises(NotFoundError):
        with mock_sbdb_query("missing.json"):
            query_sbdb(["missing"])


def _load_sbdb_fixture_payload(response_file: str) -> dict:
    resp_path = os.path.join(
        os.path.dirname(__file__), "testdata", "sbdb", response_file
    )
    with open(resp_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _assert_orbits_equivalent(a, b) -> None:
    assert len(a) == len(b)
    assert a.orbit_id.to_pylist() == b.orbit_id.to_pylist()
    assert a.object_id.to_pylist() == b.object_id.to_pylist()

    assert a.coordinates.frame == b.coordinates.frame
    assert (
        a.coordinates.origin.code.to_pylist() == b.coordinates.origin.code.to_pylist()
    )

    np.testing.assert_array_equal(
        a.coordinates.time.days.to_numpy(zero_copy_only=False),
        b.coordinates.time.days.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_array_equal(
        a.coordinates.time.nanos.to_numpy(zero_copy_only=False),
        b.coordinates.time.nanos.to_numpy(zero_copy_only=False),
    )

    for field in ["x", "y", "z", "vx", "vy", "vz"]:
        np.testing.assert_allclose(
            getattr(a.coordinates, field).to_numpy(zero_copy_only=False),
            getattr(b.coordinates, field).to_numpy(zero_copy_only=False),
            rtol=0.0,
            atol=0.0,
        )

    np.testing.assert_allclose(
        a.coordinates.covariance.to_matrix(),
        b.coordinates.covariance.to_matrix(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_query_sbdb_new_matches_query_sbdb_for_multiple_objects() -> None:
    payloads = {
        "Ceres": _load_sbdb_fixture_payload("Ceres.json"),
        "2001VB": _load_sbdb_fixture_payload("2001VB.json"),
        "54509": _load_sbdb_fixture_payload("54509.json"),
    }

    def legacy_side_effect(obj_id: str, *args, **kwargs):
        return SBDB()._process_data(OrderedDict(payloads[obj_id]))

    def new_side_effect(object_id: str, *, timeout_s: float, max_attempts: int) -> dict:
        return payloads[object_id]

    with patch("adam_core.orbits.query.sbdb.SBDB.query") as mock_legacy:
        mock_legacy.side_effect = legacy_side_effect
        legacy = query_sbdb(["Ceres", "2001VB", "54509"])

    with patch("adam_core.orbits.query.sbdb._sbdb_api_get_json") as mock_new:
        mock_new.side_effect = new_side_effect
        new = query_sbdb_new(
            ["Ceres", "2001VB", "54509"],
            max_concurrent_requests=2,
            timeout_s=1.0,
            max_attempts=1,
        )

    _assert_orbits_equivalent(legacy, new)


def test_query_sbdb_new_missing_raises() -> None:
    payload = _load_sbdb_fixture_payload("missing.json")

    def new_side_effect(object_id: str, *, timeout_s: float, max_attempts: int) -> dict:
        return payload

    with patch("adam_core.orbits.query.sbdb._sbdb_api_get_json") as mock_new:
        mock_new.side_effect = new_side_effect
        with pytest.raises(NotFoundError):
            query_sbdb_new(["missing"], timeout_s=1.0, max_attempts=1)


def test_query_sbdb_new_allow_missing_filters_missing() -> None:
    payload_missing = _load_sbdb_fixture_payload("missing.json")
    payload_ceres = _load_sbdb_fixture_payload("Ceres.json")

    def new_side_effect(object_id: str, *, timeout_s: float, max_attempts: int) -> dict:
        if object_id == "Ceres":
            return payload_ceres
        return payload_missing

    with patch("adam_core.orbits.query.sbdb._sbdb_api_get_json") as mock_new:
        mock_new.side_effect = new_side_effect
        orbits = query_sbdb_new(
            ["missing", "Ceres"],
            allow_missing=True,
            orbit_id_from_input=True,
            timeout_s=1.0,
            max_attempts=1,
        )

    assert len(orbits) == 1
    assert orbits.orbit_id.to_pylist() == ["Ceres"]


def test_query_sbdb_new_allow_missing_all_missing_returns_empty() -> None:
    payload_missing = _load_sbdb_fixture_payload("missing.json")

    def new_side_effect(object_id: str, *, timeout_s: float, max_attempts: int) -> dict:
        return payload_missing

    with patch("adam_core.orbits.query.sbdb._sbdb_api_get_json") as mock_new:
        mock_new.side_effect = new_side_effect
        orbits = query_sbdb_new(
            ["missing"],
            allow_missing=True,
            orbit_id_from_input=True,
            timeout_s=1.0,
            max_attempts=1,
        )

    assert len(orbits) == 0


def test_query_sbdb_new_fallback_covariance_handles_missing_sigma() -> None:
    # Some SBDB payloads omit per-element sigmas when no covariance matrix is provided.
    # We should still return an orbit (with NaNs in the covariance fallback) rather than raising.
    payload = {
        "object": {"fullname": "Test Object"},
        "orbit": {
            "epoch": 2459000.5,
            "elements": [
                {"name": "q", "value": 1.0, "sigma": 0.01},
                {"name": "e", "value": 0.1},  # sigma intentionally missing
                {"name": "tp", "value": 2459000.1, "sigma": 0.1},
                {"name": "om", "value": 80.0, "sigma": 0.1},
                {"name": "w", "value": 30.0, "sigma": 0.1},
                {"name": "i", "value": 10.0, "sigma": 0.1},
            ],
        },
    }

    def new_side_effect(object_id: str, *, timeout_s: float, max_attempts: int) -> dict:
        return payload

    with patch("adam_core.orbits.query.sbdb._sbdb_api_get_json") as mock_new:
        mock_new.side_effect = new_side_effect
        orbits = query_sbdb_new(["missing_sigma"], timeout_s=1.0, max_attempts=1)

    assert len(orbits) == 1
    cov = orbits.coordinates.covariance.to_matrix()
    assert cov.shape == (1, 6, 6)
    assert np.isnan(cov).any()


@contextmanager
def mock_sbdb_query(response_file: str):
    with patch("adam_core.orbits.query.sbdb.SBDB.query") as mock_sbdb_query:
        resp_path = os.path.join(
            os.path.dirname(__file__), "testdata", "sbdb", response_file
        )
        response_dict = json.load(open(resp_path, "r"))
        response_value = SBDB()._process_data(OrderedDict(response_dict))
        mock_sbdb_query.return_value = response_value
        yield mock_sbdb_query
