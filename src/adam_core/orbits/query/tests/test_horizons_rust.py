from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from adam_core import _rust_native
from adam_core._rust.arrow import table_from_record_batch
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.orbits import Orbits

DATA = Path(__file__).parent / "data" / "horizons"


def _time_batch(days=(60310,), nanos=(0,)):
    return pa.record_batch(
        [pa.array(days, type=pa.int64()), pa.array(nanos, type=pa.int64())],
        names=["days", "nanos"],
    )


@pytest.mark.parametrize(
    ("coordinate_type", "fixture"),
    [
        ("cartesian", "vectors_bennu_20240101.txt"),
        ("keplerian", "elements_bennu_20240101.txt"),
        ("cometary", "elements_bennu_20240101.txt"),
    ],
)
def test_recorded_horizons_orbits_are_parsed_and_assembled_in_rust(
    coordinate_type, fixture
):
    batch = _rust_native.query_horizons_arrow(
        ["101955"],
        _time_batch(),
        "tdb",
        coordinate_type,
        "@sun",
        "geometric",
        "smallbody",
        [(DATA / fixture).read_text()],
    )
    orbits = table_from_record_batch(Orbits, batch)

    assert orbits.orbit_id.to_pylist() == ["00000"]
    assert orbits.object_id.to_pylist() == ["101955 Bennu (1999 RQ36) (210195"]
    assert orbits.coordinates.time.mjd().to_pylist() == pytest.approx([60310.0])
    expected = {
        "cartesian": [
            -0.8949052809876745,
            0.5353223139378872,
            0.05982667219505702,
            -0.01167459005576071,
            -0.01287494949256736,
            -0.001317155627348354,
        ],
        "keplerian": [
            -0.8949052809876745,
            0.5353223139378871,
            0.059826672195056996,
            -0.011674590055760712,
            -0.012874949492567362,
            -0.0013171556273483544,
        ],
        "cometary": [
            -0.8949052809833884,
            0.535322313942614,
            0.059826672195540574,
            -0.011674590055846031,
            -0.01287494949251633,
            -0.001317155627342651,
        ],
    }
    np.testing.assert_allclose(
        orbits.coordinates.values,
        [expected[coordinate_type]],
        rtol=0.0,
        atol=5e-16,
    )


def test_recorded_horizons_ephemeris_preserves_nulls_units_and_schema():
    observer_batch = pa.record_batch(
        [
            pa.array(["500"], type=pa.large_string()),
            pa.array([60310], type=pa.int64()),
            pa.array([0], type=pa.int64()),
        ],
        names=["code", "days", "nanos"],
    )
    batch = _rust_native.query_horizons_ephemeris_arrow(
        ["101955"],
        observer_batch,
        "utc",
        [(DATA / "ephemerides_bennu_20240101.txt").read_text()],
    )
    ephemeris = table_from_record_batch(Ephemeris, batch)

    assert ephemeris.orbit_id.to_pylist() == ["00000"]
    assert ephemeris.object_id.to_pylist() == ["101955 Bennu (1999 RQ36) (210195"]
    assert ephemeris.coordinates.lon.to_pylist() == pytest.approx([210.058339162])
    assert ephemeris.coordinates.lat.to_pylist() == pytest.approx([-7.950578094])
    assert ephemeris.coordinates.rho.to_pylist() == [None]
    assert ephemeris.coordinates.vrho.to_pylist() == [None]
    assert ephemeris.coordinates.origin.code.to_pylist() == ["500"]
    assert ephemeris.light_time.to_pylist() == pytest.approx([7.07287679 / 1440.0])
    assert ephemeris.alpha.to_pylist() == pytest.approx([61.5181])


def test_horizons_recorded_processing_has_rust_owned_timing():
    for kind, fixture in [
        ("vectors", "vectors_bennu_20240101.txt"),
        ("elements", "elements_bennu_20240101.txt"),
        ("ephemerides", "ephemerides_bennu_20240101.txt"),
    ]:
        samples = np.asarray(
            _rust_native.benchmark_horizons_response_processing(
                kind, [(DATA / fixture).read_text()], 2, 2, 1
            )
        )
        assert samples.shape == (2, 2)
        assert np.all(samples > 0.0)


def test_horizons_validation_happens_before_http():
    with pytest.raises(
        ValueError,
        match="coordinate_type should be one of",
    ):
        _rust_native.query_horizons_arrow(
            ["101955"],
            _time_batch(),
            "tdb",
            "invalid",
            "@sun",
            "geometric",
            None,
            [],
        )
    with pytest.raises(AssertionError, match="Must have at least one time"):
        _rust_native.query_horizons_arrow(
            ["101955"],
            _time_batch((), ()),
            "tdb",
            "cartesian",
            "@sun",
            "geometric",
            None,
            [],
        )
