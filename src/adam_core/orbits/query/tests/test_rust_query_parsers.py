"""Offline parity/shape gates for Rust-backed W10 query parsers (bead
personal-cmy.29). Live HTTP remains opt-in; these tests exercise recorded or
synthetic payload parsing behind the canonical public Python helpers."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from astroquery.jplsbdb import SBDB

from adam_core.orbits.query.neocc import _parse_oef
from adam_core.orbits.query.sbdb import (
    _orbits_from_sbdb_payloads,
    _orbits_from_sbdb_results,
)

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
