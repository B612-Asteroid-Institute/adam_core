"""Global-MOID correctness and invariance tests.

JPL values are full-precision SBDB API fixtures fetched 2026-11-05.  Small
sub-micro-au offsets reflect JPL's Earth-orbit convention versus the DE440
geocentric osculating conic used by ``calculate_perturber_moids``.
"""

import inspect

import numpy as np
import pytest

from adam_core import _rust_native
from adam_core.coordinates import KeplerianCoordinates, Origin, OriginCodes
from adam_core.dynamics.moid import calculate_moid, calculate_perturber_moids
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

MU_SUN = 0.00029591220828559104


def _states(rows):
    rows = np.asarray(rows, dtype=np.float64)
    n = len(rows)
    coordinates = KeplerianCoordinates.from_kwargs(
        a=rows[:, 0],
        e=rows[:, 1],
        i=rows[:, 2],
        raan=rows[:, 3],
        ap=rows[:, 4],
        M=rows[:, 5],
        time=Timestamp.from_mjd(np.full(n, 60000.0), scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"] * n),
        frame="ecliptic",
    ).to_cartesian()
    return np.ascontiguousarray(coordinates.values, dtype=np.float64)


def test_moid_contract_has_no_obsolete_controls():
    from adam_core._rust import api as rust_api

    assert tuple(inspect.signature(calculate_moid).parameters) == (
        "primary_ellipse",
        "secondary_ellipse",
    )
    assert tuple(inspect.signature(calculate_perturber_moids).parameters) == (
        "orbits",
        "perturber",
    )
    assert tuple(inspect.signature(rust_api.calculate_moid_numpy).parameters) == (
        "primary_orbit",
        "secondary_orbit",
        "mu",
    )
    assert tuple(inspect.signature(rust_api.calculate_moid_batch_numpy).parameters) == (
        "primary_orbits",
        "secondary_orbits",
        "mus",
    )
    assert tuple(inspect.signature(_rust_native.calculate_moid_numpy).parameters) == (
        "primary_orbit",
        "secondary_orbit",
        "mu",
    )
    assert tuple(
        inspect.signature(_rust_native.calculate_moid_batch_numpy).parameters
    ) == (
        "primary_orbits",
        "secondary_orbits",
        "mus",
    )
    assert tuple(
        inspect.signature(rust_api.calculate_perturber_moids_arrow).parameters
    ) == (
        "orbit_batch",
        "perturber_codes",
    )
    assert tuple(
        inspect.signature(rust_api.calculate_perturber_moids_native).parameters
    ) == (
        "primary",
        "mus",
        "time_scale",
        "time_days",
        "time_nanos",
        "perturber_codes",
        "frame",
        "origin_code",
    )
    assert tuple(
        inspect.signature(_rust_native.calculate_perturber_moids_arrow).parameters
    ) == (
        "orbit_batch",
        "perturber_codes",
    )
    assert tuple(
        inspect.signature(_rust_native.calculate_perturber_moids_native).parameters
    ) == (
        "primary",
        "mus",
        "time_scale",
        "time_days",
        "time_nanos",
        "perturber_codes",
        "frame",
        "origin_code",
    )
    assert tuple(
        inspect.signature(_rust_native.benchmark_calculate_moid_numpy).parameters
    ) == (
        "primary_orbits",
        "secondary_orbits",
        "mus",
        "reps",
        "trials",
        "warmup_reps",
    )
    assert tuple(
        inspect.signature(_rust_native.benchmark_calculate_moid_batch_numpy).parameters
    ) == (
        "primary_orbits",
        "secondary_orbits",
        "mus",
        "reps",
        "trials",
        "warmup_reps",
    )
    assert tuple(
        inspect.signature(
            _rust_native.benchmark_calculate_perturber_moids_arrow
        ).parameters
    ) == (
        "orbit_batch",
        "perturber_codes",
        "reps",
        "trials",
        "warmup_reps",
    )

    state = _states([[1.0, 0.1, 3.0, 4.0, 5.0, 6.0]])[0]
    with pytest.raises(TypeError):
        rust_api.calculate_moid_numpy(state, state, MU_SUN, max_iter=100)
    with pytest.raises(TypeError):
        _rust_native.calculate_moid_numpy(state, state, MU_SUN, 100, 1e-10)
    with pytest.raises(TypeError):
        calculate_perturber_moids(None, OriginCodes.EARTH, chunk_size=1)


def test_global_moid_scalar_batch_symmetry_rotation_and_repeatability():
    first = _states(
        [
            [1.70, 0.31, 22.0, 41.0, 73.0, 13.0],
            [0.92, 0.19, 3.3, 203.0, 126.0, 175.0],
            [2.52, 0.66, 3.4, 271.0, 134.0, 139.0],
            [1.00, 0.00, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    second = _states(
        [
            [1.00, 0.0167, 0.2, 11.0, 102.9, 251.0],
            [1.63, 0.48, 31.0, 73.0, 260.0, 3.0],
            [0.83, 0.22, 81.0, 7.0, 12.0, 310.0],
            [1.50, 0.00, 73.0, 41.0, 80.0, 220.0],
        ]
    )
    mus = np.full(len(first), MU_SUN)

    expected, witness = _rust_native.calculate_moid_batch_numpy(first, second, mus)
    swapped, _ = _rust_native.calculate_moid_batch_numpy(second, first, mus)
    scalar = np.array(
        [
            _rust_native.calculate_moid_numpy(p, s, MU_SUN)[0]
            for p, s in zip(first, second)
        ]
    )

    np.testing.assert_array_equal(expected, swapped)
    np.testing.assert_array_equal(expected, scalar)
    assert np.all(np.isfinite(witness))
    assert expected[-1] == pytest.approx(0.5, abs=1e-11)

    # Arbitrary proper 3-D common-frame rotation, not merely a z-axis change.
    rotation, _ = np.linalg.qr(
        np.array([[0.2, 0.3, 0.9], [0.8, 0.1, 0.4], [0.5, 0.7, 0.2]])
    )
    rotation *= np.linalg.det(rotation)
    first_rotated = first.copy()
    second_rotated = second.copy()
    first_rotated[:, :3] @= rotation
    first_rotated[:, 3:] @= rotation
    second_rotated[:, :3] @= rotation
    second_rotated[:, 3:] @= rotation
    rotated, _ = _rust_native.calculate_moid_batch_numpy(
        first_rotated, second_rotated, mus
    )
    np.testing.assert_allclose(rotated, expected, rtol=0.0, atol=5e-11)

    for _ in range(10):
        repeated, repeated_witness = _rust_native.calculate_moid_batch_numpy(
            first, second, mus
        )
        np.testing.assert_array_equal(repeated, expected)
        np.testing.assert_array_equal(repeated_witness, witness)


def test_regression_full_secondary_orientation_is_preserved():
    # Discovery-epoch N0039b4d/Earth state pair.  The former kernel returned
    # 0.6445 au after discarding the Earth's orbital orientation; the
    # independently refined global value is 0.0165564227952 au.
    asteroid = np.array(
        [
            -0.8805909973611534,
            0.5615377914632823,
            -0.12148378350063478,
            -0.004512629923099136,
            -0.013820053438607916,
            0.012915610447333024,
        ]
    )
    earth = np.array(
        [
            -0.8039837945074675,
            0.5732704136045476,
            -3.537783275595139e-05,
            -0.010272228480035098,
            -0.014077738607485912,
            5.057570863716226e-07,
        ]
    )
    moid, _ = _rust_native.calculate_moid_numpy(asteroid, earth, MU_SUN)
    assert moid == pytest.approx(0.01655642279523, abs=5e-12)
    assert moid <= np.linalg.norm(asteroid[:3] - earth[:3])


def test_invalid_conics_are_explicit_not_bogus_finite_moids():
    valid = _states([[1.0, 0.1, 3.0, 4.0, 5.0, 6.0]])[0]
    invalid = np.full(6, np.nan)
    with pytest.raises(ValueError, match="finite"):
        _rust_native.calculate_moid_numpy(invalid, valid, MU_SUN)

    hyperbolic = _states([[-2.0, 1.5, 20.0, 30.0, 40.0, 10.0]])[0]
    with pytest.raises(ValueError, match="hyperbolic"):
        _rust_native.calculate_moid_numpy(hyperbolic, valid, MU_SUN)

    with pytest.raises(
        ValueError,
        match="global MOID solver failed for row 0: MOID inputs must be finite",
    ):
        _rust_native.calculate_moid_batch_numpy(
            np.ascontiguousarray([invalid]),
            np.ascontiguousarray([valid]),
            np.array([MU_SUN]),
        )


# des, epoch MJD TDB, a, e, i, raan, argument of periapsis, M, JPL Earth MOID
JPL_SBDB_FIXTURES = [
    (
        "101955",
        55562.0,
        1.126391025894812,
        0.2037450762416414,
        6.03494377024794,
        2.06086619569642,
        66.22306084084298,
        101.703952002457,
        0.0032228,
    ),
    (
        "1036",
        61200.0,
        2.664068815977234,
        0.5335006672216854,
        26.68765340379128,
        215.3977394974259,
        132.5072976556281,
        142.9653116916981,
        0.341955,
    ),
    (
        "1221",
        61200.0,
        1.919716404464349,
        0.4347308670682633,
        11.87012985263682,
        171.2356175438618,
        26.75303780543283,
        133.9760950671647,
        0.10902,
    ),
    (
        "2024 YR4",
        61200.0,
        2.515989042462611,
        0.6613330031901985,
        3.408056116054586,
        271.3669515560151,
        134.3664504369954,
        139.2118467324958,
        0.00274808,
    ),
    (
        "433",
        61200.0,
        1.458243716760167,
        0.2228779627700761,
        10.82854410314273,
        304.2679713350896,
        178.9181319135911,
        62.51145501986792,
        0.148798,
    ),
    (
        "99942",
        61200.0,
        0.9223592206975018,
        0.1911492279663492,
        3.340996879880978,
        203.8936514240762,
        126.6795706895841,
        175.3304026592739,
        0.000107917,
    ),
]


def test_jpl_sbdb_earth_moid_fixtures():
    from adam_core.utils.spice import setup_SPICE

    setup_SPICE()
    fields = list(zip(*JPL_SBDB_FIXTURES))
    coordinates = KeplerianCoordinates.from_kwargs(
        a=fields[2],
        e=fields[3],
        i=fields[4],
        raan=fields[5],
        ap=fields[6],
        M=fields[7],
        time=Timestamp.from_mjd(fields[1], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"] * len(JPL_SBDB_FIXTURES)),
        frame="ecliptic",
    ).to_cartesian()
    orbits = Orbits.from_kwargs(orbit_id=fields[0], coordinates=coordinates)
    result = calculate_perturber_moids(orbits, OriginCodes.EARTH)
    actual = result.moid.to_numpy(zero_copy_only=False)
    expected = np.asarray(fields[8])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-7)
