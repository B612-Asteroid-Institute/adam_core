import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
from astroquery.jplsbdb import SBDB

from ..sbdb import (
    NotFoundError,
    _convert_SBDB_covariances,
    _orbits_from_sbdb_payloads,
    _physical_parameters_from_sbdb,
    _sbdb_phys_par_from_payload,
    query_sbdb,
    query_sbdb_new,
)


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


def test__sbdb_phys_par_from_payload_empty() -> None:
    out = _sbdb_phys_par_from_payload({})
    assert out == (None, None, None, None)
    out = _sbdb_phys_par_from_payload({"phys_par": []})
    assert out == (None, None, None, None)


def test__sbdb_phys_par_from_payload_H_only() -> None:
    payload = {"phys_par": [{"name": "H", "value": "19.5"}]}
    out = _sbdb_phys_par_from_payload(payload)
    assert out[0] == 19.5
    assert out[1] is None
    assert out[2] is None
    assert out[3] is None


def test__sbdb_phys_par_from_payload_H_mag_fallback() -> None:
    payload = {"phys_par": [{"name": "H_mag", "value": "20.0"}]}
    out = _sbdb_phys_par_from_payload(payload)
    assert out[0] == 20.0
    assert out[2] is None


def test__sbdb_phys_par_from_payload_H_G_with_sigmas() -> None:
    payload = {
        "phys_par": [
            {"name": "H", "value": "18.2", "sigma": "0.3"},
            {"name": "G", "value": "0.15", "sigma": "0.02"},
        ]
    }
    out = _sbdb_phys_par_from_payload(payload)
    assert out == (18.2, 0.3, 0.15, 0.02)


def test__physical_parameters_from_sbdb_empty() -> None:
    tbl = _physical_parameters_from_sbdb([])
    assert len(tbl) == 0


def test__physical_parameters_from_sbdb_one_row_all_none() -> None:
    tbl = _physical_parameters_from_sbdb([(None, None, None, None)])
    assert len(tbl) == 1
    assert np.isnan(tbl.H_v[0].as_py())
    assert np.isnan(tbl.H_v_sigma[0].as_py())
    assert np.isnan(tbl.G[0].as_py())
    assert np.isnan(tbl.G_sigma[0].as_py())


def test__physical_parameters_from_sbdb_one_row_values() -> None:
    tbl = _physical_parameters_from_sbdb([(10.5, 0.2, 0.15, None)])
    assert len(tbl) == 1
    assert tbl.H_v[0].as_py() == 10.5
    assert tbl.H_v_sigma[0].as_py() == 0.2
    assert tbl.G[0].as_py() == 0.15
    assert np.isnan(tbl.G_sigma[0].as_py())


def test__physical_parameters_from_sbdb_two_rows() -> None:
    tbl = _physical_parameters_from_sbdb(
        [(10.0, 0.1, 0.15, None), (20.0, None, 0.25, 0.05)]
    )
    assert len(tbl) == 2
    assert tbl.H_v[0].as_py() == 10.0 and tbl.H_v[1].as_py() == 20.0
    assert tbl.H_v_sigma[0].as_py() == 0.1 and np.isnan(tbl.H_v_sigma[1].as_py())
    assert tbl.G[0].as_py() == 0.15 and tbl.G[1].as_py() == 0.25
    assert np.isnan(tbl.G_sigma[0].as_py()) and tbl.G_sigma[1].as_py() == 0.05


def test_query_sbdb_new_physical_parameters_from_phys_par() -> None:
    # SBDB returns H, G (and optional sigmas) when phys-par=1; V-band per JPL/MPC convention.
    payload = _load_sbdb_fixture_payload("2001VB.json")
    payload["phys_par"] = [
        {"name": "H", "value": "18.2", "sigma": "0.3"},
        {"name": "G", "value": "0.15", "sigma": None},
    ]

    def new_side_effect(object_id: str, *, timeout_s: float, max_attempts: int) -> dict:
        return payload

    with patch("adam_core.orbits.query.sbdb._sbdb_api_get_json") as mock_new:
        mock_new.side_effect = new_side_effect
        orbits = query_sbdb_new(["2001VB"], timeout_s=1.0, max_attempts=1)

    assert len(orbits) == 1
    assert orbits.physical_parameters is not None
    assert orbits.physical_parameters.H_v[0].as_py() == 18.2
    assert orbits.physical_parameters.H_v_sigma[0].as_py() == 0.3
    assert orbits.physical_parameters.G[0].as_py() == 0.15
    assert np.isnan(orbits.physical_parameters.G_sigma[0].as_py())


def test_query_sbdb_new_physical_parameters_accepts_H_mag() -> None:
    # Some SBDB sources use name "H_mag" instead of "H"; both are accepted.
    payload = _load_sbdb_fixture_payload("2001VB.json")
    payload["phys_par"] = [
        {"name": "H_mag", "value": "20.5"},
        {"name": "G", "value": "0.25"},
    ]

    def new_side_effect(object_id: str, *, timeout_s: float, max_attempts: int) -> dict:
        return payload

    with patch("adam_core.orbits.query.sbdb._sbdb_api_get_json") as mock_new:
        mock_new.side_effect = new_side_effect
        orbits = query_sbdb_new(["2001VB"], timeout_s=1.0, max_attempts=1)

    assert orbits.physical_parameters.H_v[0].as_py() == 20.5
    assert orbits.physical_parameters.G[0].as_py() == 0.25


def test_query_sbdb_new_physical_parameters_missing_phys_par_fills_nan() -> None:
    # Payload without phys_par (or without H) yields NaN physical parameters.
    payload = _load_sbdb_fixture_payload("2001VB.json")
    assert "phys_par" not in payload

    def new_side_effect(object_id: str, *, timeout_s: float, max_attempts: int) -> dict:
        return payload

    with patch("adam_core.orbits.query.sbdb._sbdb_api_get_json") as mock_new:
        mock_new.side_effect = new_side_effect
        orbits = query_sbdb_new(["2001VB"], timeout_s=1.0, max_attempts=1)

    assert orbits.physical_parameters is not None
    assert np.isnan(orbits.physical_parameters.H_v[0].as_py())
    assert np.isnan(orbits.physical_parameters.G[0].as_py())


def test_real_sbdb_payloads_parse_without_error() -> None:
    # Parse every *_phys.json in testdata; ensures real API response shapes don't break us.
    sbdb_dir = os.path.join(os.path.dirname(__file__), "testdata", "sbdb")
    for name in sorted(os.listdir(sbdb_dir)):
        if not name.endswith("_phys.json"):
            continue
        payload = _load_sbdb_fixture_payload(name)
        if "object" not in payload or "orbit" not in payload:
            continue
        obj_id = str(payload["object"].get("fullname", payload["object"].get("des", name)))
        orbits = _orbits_from_sbdb_payloads([obj_id], [payload])
        assert len(orbits) == 1
        assert orbits.coordinates is not None
        assert orbits.physical_parameters is not None


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
