"""Offline parity/shape gates for Rust-backed W10 query parsers (bead
personal-cmy.29). Live HTTP remains opt-in; these tests exercise recorded or
synthetic payload parsing behind the canonical public Python helpers."""

import json
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from astroquery.jplsbdb import SBDB

from adam_core.observers import Observers
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.query.horizons import (
    query_horizons,
    query_horizons_ephemeris,
)
from adam_core.orbits.query.neocc import _parse_oef
from adam_core.orbits.query.sbdb import (
    _orbits_from_sbdb_payloads,
    _orbits_from_sbdb_results,
)
from adam_core.time import Timestamp

TESTDATA = Path(__file__).parent / "testdata"


def _assert_orbits_bitwise_equal(a, b):
    assert len(a) == len(b)
    assert a.orbit_id.to_pylist() == b.orbit_id.to_pylist()
    assert a.object_id.to_pylist() == b.object_id.to_pylist()
    assert a.coordinates.frame == b.coordinates.frame
    assert (
        a.coordinates.origin.code.to_pylist() == b.coordinates.origin.code.to_pylist()
    )
    assert a.coordinates.time.days.to_pylist() == b.coordinates.time.days.to_pylist()
    assert a.coordinates.time.nanos.to_pylist() == b.coordinates.time.nanos.to_pylist()
    np.testing.assert_array_equal(a.coordinates.values, b.coordinates.values)
    np.testing.assert_array_equal(
        np.nan_to_num(a.coordinates.covariance.to_matrix(), nan=0.0),
        np.nan_to_num(b.coordinates.covariance.to_matrix(), nan=0.0),
    )


def test_neocc_rust_parser_matches_recorded_oef_fixture():
    text = (TESTDATA / "neocc" / "2024YR4.ke1").read_text()
    parsed = _parse_oef(text)
    assert parsed["object_id"] == "2024YR4"
    assert parsed["header"]["refsys"] == "ECLM J2000"
    assert parsed["time_system"] == "TDT"
    assert parsed["elements"]["a"] == 2.5158372264739257
    assert parsed["magnitude"] == {"H": 24.047, "G": 0.150}
    assert parsed["covariance"].shape == (6, 6)
    assert parsed["correlation"].shape == (6, 6)
    np.testing.assert_array_equal(parsed["covariance"], parsed["covariance"].T)


def test_sbdb_rust_normalizer_matches_legacy_astroquery_processed_payloads():
    ids = ["Ceres", "2001VB", "54509"]
    payloads = []
    processed = []
    for obj_id in ids:
        payload = json.loads((TESTDATA / "sbdb" / f"{obj_id}.json").read_text())
        payloads.append(payload)
        processed.append(SBDB()._process_data(OrderedDict(payload)))

    legacy = _orbits_from_sbdb_results(ids, processed)
    rust_backed = _orbits_from_sbdb_payloads(ids, payloads)
    _assert_orbits_bitwise_equal(rust_backed, legacy)


def _fake_vectors(*args, **kwargs):
    return pd.DataFrame(
        {
            "orbit_id": ["00000", "00000"],
            "targetname": ["Test Object", "Test Object"],
            "datetime_jd": [2460000.5, 2460001.5],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
            "vx": [0.1, 0.2],
            "vy": [0.3, 0.4],
            "vz": [0.5, 0.6],
        }
    )


def _fake_elements(*args, **kwargs):
    return pd.DataFrame(
        {
            "orbit_id": ["00000", "00000"],
            "targetname": ["Test Object", "Test Object"],
            "datetime_jd": [2460000.5, 2460001.5],
            "a": [1.0, 1.1],
            "e": [0.1, 0.2],
            "incl": [3.0, 4.0],
            "Omega": [5.0, 6.0],
            "w": [7.0, 8.0],
            "M": [9.0, 10.0],
            "q": [0.9, 0.95],
            "Tp_jd": [2459000.5, 2459001.5],
        }
    )


def test_horizons_rust_normalizers_build_cartesian_keplerian_cometary_orbits():
    times = Timestamp.from_jd([2460000.5, 2460001.5], scale="tdb")
    with patch(
        "adam_core.orbits.query.horizons._get_horizons_vectors",
        side_effect=_fake_vectors,
    ):
        cart = query_horizons(["Test"], times, coordinate_type="cartesian")
    assert cart.orbit_id.to_pylist() == ["00000", "00000"]
    assert cart.object_id.to_pylist() == ["Test Object", "Test Object"]
    np.testing.assert_array_equal(cart.coordinates.values[0], [1, 3, 5, 0.1, 0.3, 0.5])

    with patch(
        "adam_core.orbits.query.horizons._get_horizons_elements",
        side_effect=_fake_elements,
    ):
        kep = query_horizons(["Test"], times, coordinate_type="keplerian")
        com = query_horizons(["Test"], times, coordinate_type="cometary")
    assert len(kep) == 2
    assert len(com) == 2
    assert kep.object_id.to_pylist() == ["Test Object", "Test Object"]
    assert com.object_id.to_pylist() == ["Test Object", "Test Object"]


def test_horizons_ephemeris_rust_normalizer_builds_ephemeris():
    df = pd.DataFrame(
        {
            "orbit_id": ["00000"],
            "targetname": ["Test Object"],
            "datetime_jd": [2460000.5],
            "observatory_code": ["500"],
            "lighttime": [144.0],
            "alpha": [12.5],
            "RA": [123.4],
            "DEC": [-45.6],
        }
    )
    observers = Observers.from_code("500", Timestamp.from_jd([2460000.5], scale="utc"))
    with patch(
        "adam_core.orbits.query.horizons._get_horizons_ephemeris", return_value=df
    ):
        eph = query_horizons_ephemeris(["Test"], observers)
    assert isinstance(eph, Ephemeris)
    assert eph.orbit_id.to_pylist() == ["00000"]
    assert eph.object_id.to_pylist() == ["Test Object"]
    assert eph.light_time[0].as_py() == 0.1
    assert eph.coordinates.origin.code.to_pylist() == ["500"]
