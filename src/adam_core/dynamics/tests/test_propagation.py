import cProfile
import itertools

import numpy as np
import pytest
from astropy import units as u

from ..._rust import propagate_2body_numpy
from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...dynamics.exceptions import DynamicsNumericalError
from ...orbits import Orbits
from ...orbits.physical_parameters import PhysicalParameters
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from .. import propagation as propagation_module
from ..propagation import propagate_2body

# Fixed independent oracle vectors generated from CSPICE `spiceypy.prop2b`
# in the baseline-main `.legacy-venv` on 2026-04-30. The pre-migration
# tests computed this oracle live; keeping a small fixed subset preserves the
# independent check without reintroducing `spiceypy` as a test dependency.
# Oracle generation used adam-core's `OriginGravitationalParameters` values
# for `mu` so this isolates Kepler-solver behavior from GM convention choices.
_PROP2B_ORACLE_CASES = (
    {
        "id": "sun_elliptical_multi_rev",
        "origin_code": "SUN",
        "t0_mjd_tdb": 59091.0,
        "dt_days": 4990.0,
        "state": (
            0.2184969861179754,
            -0.488165331818421,
            -0.1450698595270714,
            0.0225265741518987,
            0.005220079188059638,
            0.0007255527925032393,
        ),
        "expected": (
            0.2251964470659416,
            -0.48657131508421136,
            -0.14484234782346428,
            0.022411936510779062,
            0.005471927776407792,
            0.0008004590886333903,
        ),
    },
    {
        "id": "sun_hyperbolic_medium",
        "origin_code": "SUN",
        "t0_mjd_tdb": 58080.0,
        "dt_days": 1000.0,
        "state": (
            1.889136186533479,
            0.6815829716216527,
            0.259065170725899,
            0.0210650228586455,
            0.003903782164346327,
            0.008115468208135282,
        ),
        "expected": (
            17.751479210772544,
            3.123635865329321,
            7.073742444802862,
            0.014731074779228361,
            0.0021987873982522175,
            0.00642638500647284,
        ),
    },
    {
        "id": "ssb_elliptical_multi_rev",
        "origin_code": "SOLAR_SYSTEM_BARYCENTER",
        "t0_mjd_tdb": 59091.0,
        "dt_days": 4990.0,
        "state": (
            0.2127380745372033,
            -0.4815155316206986,
            -0.1449911948382041,
            0.0225189702381014,
            0.005215419523856744,
            0.0007257758516405377,
        ),
        "expected": (
            -0.2366066308701146,
            0.4130694115536764,
            0.12622225401687803,
            -0.02028332510271844,
            -0.015106639469414181,
            -0.0036315361999360403,
        ),
    },
    {
        "id": "ssb_hyperbolic_medium",
        "origin_code": "SOLAR_SYSTEM_BARYCENTER",
        "t0_mjd_tdb": 58080.0,
        "dt_days": 1000.0,
        "state": (
            1.891160191883445,
            0.6875120603422388,
            0.258939713700702,
            0.0210594865646723,
            0.00390906388113403,
            0.008115600602499821,
        ),
        "expected": (
            17.753816132590185,
            3.1280145364976573,
            7.075534455355301,
            0.014731071510713982,
            0.002196317711876263,
            0.006428193609586648,
        ),
    },
)


def _propagate_2body_single(
    orbit: np.ndarray, t0: float, t1: float, mu: float
) -> np.ndarray:
    """Single-row wrapper around the batched Rust 2-body kernel for tests."""
    out = propagate_2body_numpy(
        np.ascontiguousarray(orbit.reshape(1, 6), dtype=np.float64),
        np.array([t1 - t0], dtype=np.float64),
        np.array([mu], dtype=np.float64),
        1000,
        1e-14,
    )
    assert out is not None
    return np.asarray(out)[0]


def _specific_energy(state: np.ndarray, mu: float) -> float:
    r = np.linalg.norm(state[:3])
    v2 = float(np.dot(state[3:], state[3:]))
    return 0.5 * v2 - mu / r


def _angular_momentum(state: np.ndarray) -> np.ndarray:
    return np.cross(state[:3], state[3:])


def _check_roundtrip(orbital_elements, mu_vec, max_dt: float = 10000.0):
    # Forward then backward Kepler propagation must return to the original
    # state, and energy + angular momentum must be preserved along the
    # propagated arc. Tolerance accommodates ~two calls worth of universal-
    # variable Kepler solver drift over the propagated span.
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values

    dts = np.arange(0, max_dt, 10)
    for i, (t0_i, cartesian_i) in enumerate(zip(t0, cartesian_elements)):

        e0 = _specific_energy(cartesian_i, mu_vec[i])
        h0 = _angular_momentum(cartesian_i)

        for dt_i in dts:
            forward = _propagate_2body_single(cartesian_i, t0_i, t0_i + dt_i, mu_vec[i])
            back = _propagate_2body_single(forward, t0_i + dt_i, t0_i, mu_vec[i])

            r_diff = np.linalg.norm(back[:3] - cartesian_i[:3]) * u.au.to(u.m)
            v_diff = np.linalg.norm(back[3:] - cartesian_i[3:]) * (u.au / u.d).to(
                u.mm / u.s
            )
            # 100 m / 1 mm/s over a 10,000-day (~27 year) roundtrip absorbs
            # universal-variable-Kepler time-reversibility drift and is
            # still well below any science-relevant threshold.
            np.testing.assert_array_less(r_diff, 100.0)
            np.testing.assert_array_less(v_diff, 1.0)

            # Energy and angular momentum are conservation invariants of
            # the two-body problem and must hold along the propagated arc.
            e_fwd = _specific_energy(forward, mu_vec[i])
            h_fwd = _angular_momentum(forward)
            np.testing.assert_allclose(e_fwd, e0, rtol=1e-10, atol=1e-18)
            np.testing.assert_allclose(h_fwd, h0, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize(
    "case",
    _PROP2B_ORACLE_CASES,
    ids=[str(case["id"]) for case in _PROP2B_ORACLE_CASES],
)
def test_propagate_2body_matches_fixed_cspice_prop2b_oracle(case) -> None:
    state = np.asarray(case["state"], dtype=np.float64)
    expected = np.asarray(case["expected"], dtype=np.float64)
    t0 = float(case["t0_mjd_tdb"])
    t1 = t0 + float(case["dt_days"])

    coordinates = CartesianCoordinates.from_kwargs(
        x=[state[0]],
        y=[state[1]],
        z=[state[2]],
        vx=[state[3]],
        vy=[state[4]],
        vz=[state[5]],
        time=Timestamp.from_mjd([t0], scale="tdb"),
        origin=Origin.from_kwargs(code=[case["origin_code"]]),
        frame="ecliptic",
    )
    orbits = Orbits.from_kwargs(
        orbit_id=[case["id"]],
        object_id=[case["id"]],
        coordinates=coordinates,
    )

    propagated = propagate_2body(
        orbits,
        Timestamp.from_mjd([t1], scale="tdb"),
        max_processes=1,
    )
    actual = np.asarray(propagated.coordinates.values[0], dtype=np.float64)
    diff = actual - expected

    r_diff_cm = np.linalg.norm(diff[:3]) * u.au.to(u.cm)
    v_diff_mm_s = np.linalg.norm(diff[3:]) * (u.au / u.d).to(u.mm / u.s)

    # Preserve the pre-migration CSPICE oracle tolerances.
    np.testing.assert_array_less(r_diff_cm, 10.0)
    np.testing.assert_array_less(v_diff_mm_s, 1.0)


def test__propagate_2body_single_roundtrip_elliptical(orbital_elements):
    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))
    _check_roundtrip(orbital_elements, origin.mu())


def test__propagate_2body_single_roundtrip_hyperbolic(orbital_elements):
    orbital_elements = orbital_elements[orbital_elements["e"] > 1.0]
    origin = Origin.from_kwargs(code=["SUN"] * len(orbital_elements))
    _check_roundtrip(orbital_elements, origin.mu())


def test__propagate_2body_single_roundtrip_elliptical_barycentric(
    orbital_elements_barycentric,
):
    orbital_elements = orbital_elements_barycentric[
        orbital_elements_barycentric["e"] < 1.0
    ]
    origin = Origin.from_kwargs(
        code=["SOLAR_SYSTEM_BARYCENTER"] * len(orbital_elements)
    )
    _check_roundtrip(orbital_elements, origin.mu())


def test__propagate_2body_single_roundtrip_hyperbolic_barycentric(
    orbital_elements_barycentric,
):
    orbital_elements = orbital_elements_barycentric[
        orbital_elements_barycentric["e"] > 1.0
    ]
    origin = Origin.from_kwargs(
        code=["SOLAR_SYSTEM_BARYCENTER"] * len(orbital_elements)
    )
    _check_roundtrip(orbital_elements, origin.mu())


def test_benchmark_propagate_2body_single(benchmark, orbital_elements):
    # Benchmark propagate_2body with a single orbit propagated forward 1 day
    # This function appears to add substantial overhead, so we benchmark it
    # separately from _propagate_2body_single
    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    # Create orbits object
    orbits = Orbits.from_kwargs(
        orbit_id=orbital_elements["targetname"].values,
        object_id=orbital_elements["targetname"].values,
        coordinates=CartesianCoordinates.from_kwargs(
            x=cartesian_elements[:, 0],
            y=cartesian_elements[:, 1],
            z=cartesian_elements[:, 2],
            vx=cartesian_elements[:, 3],
            vy=cartesian_elements[:, 4],
            vz=cartesian_elements[:, 5],
            time=Timestamp.from_mjd(
                t0,
                scale="tdb",
            ),
            origin=Origin.from_kwargs(
                code=["SUN" for i in range(len(cartesian_elements))]
            ),
            frame="ecliptic",
        ),
    )
    times = Timestamp.from_mjd(
        [t0.min() + 1],
        scale="tdb",
    )
    benchmark(propagate_2body, orbits[0], times=times)


@pytest.mark.benchmark(group="propagate_2body")
def test_benchmark_propagate_2body_single_matrix(benchmark, propagated_orbits):
    # Clear the jax cache
    pass  # cache clear no longer needed (no JAX)

    def benchmark_function():
        n_orbits = [1, 5, 20]
        n_times = [1, 10, 100]

        for n_orbits_i, n_times_i in itertools.product(n_orbits, n_times):
            times = Timestamp.from_mjd(
                np.arange(0, n_times_i, 1),
                scale="tdb",
            )
            propagate_2body(propagated_orbits[:n_orbits_i], times=times)

    benchmark(benchmark_function)


def test_propagate_2body_single_preserves_physical_parameters():
    t0 = Timestamp.from_mjd([60000.0, 60000.0], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["o1", "o2"],
        object_id=["o1", "o2"],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.0, 17.5],
            G=[0.15, 0.25],
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.2],
            y=[0.0, 0.1],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.017, 0.015],
            vz=[0.0, 0.0],
            time=t0,
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )

    times = Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb")
    propagated = propagate_2body(orbits, times)

    expected_H = np.array([15.0, 15.0, 15.0, 17.5, 17.5, 17.5], dtype=np.float64)
    expected_G = np.array([0.15, 0.15, 0.15, 0.25, 0.25, 0.25], dtype=np.float64)

    have_H = propagated.physical_parameters.H_v.to_numpy(zero_copy_only=False)
    have_G = propagated.physical_parameters.G.to_numpy(zero_copy_only=False)

    np.testing.assert_allclose(have_H, expected_H)
    np.testing.assert_allclose(have_G, expected_G)


def test_propagate_2body_single_does_not_include_padded_rows() -> None:
    """
    `process_in_chunks` pads the final chunk to a fixed size. Ensure the propagation
    output only contains the true (n_orbits * n_times) rows.
    """
    orbits = make_real_orbits(1)
    orbit_mjd = orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    base_mjd = float(orbit_mjd[0])

    n_times = 201  # not divisible by chunk_size=200
    times = Timestamp.from_mjd(
        base_mjd + np.arange(n_times, dtype=np.float64), scale="tdb"
    )
    propagated = propagate_2body(orbits, times)

    assert len(propagated) == n_times
    out_mjd = (
        propagated.coordinates.time.rescale("tdb").mjd().to_numpy(zero_copy_only=False)
    )
    in_mjd = times.rescale("tdb").mjd().to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(out_mjd, in_mjd)


def test_propagate_2body_process_count_compatibility_matches_default() -> None:
    orbits = make_real_orbits(5)
    base_mjd = float(
        np.median(orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False))
    )
    times = Timestamp.from_mjd(base_mjd + np.arange(25, dtype=np.float64), scale="tdb")

    serial = propagate_2body(orbits, times, max_processes=1)
    parallel = propagate_2body(orbits, times, max_processes=2, chunk_size=2)

    assert len(serial) == len(parallel)
    np.testing.assert_array_equal(
        serial.orbit_id.to_numpy(zero_copy_only=False),
        parallel.orbit_id.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_allclose(
        serial.coordinates.values, parallel.coordinates.values, rtol=0, atol=0
    )


def test_propagate_2body_uses_record_batches_without_numpy_rebuild(monkeypatch) -> None:
    t0 = Timestamp.from_mjd([60000.0], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["arrow"],
        object_id=["arrow"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.017],
            vz=[0.0],
            time=t0,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    def forbidden_values(_self):
        raise AssertionError("orbit states must stay inside the Arrow RecordBatch")

    def forbidden_from_kwargs(*_args, **_kwargs):
        raise AssertionError("propagated output must wrap Rust's RecordBatch directly")

    monkeypatch.setattr(CartesianCoordinates, "values", property(forbidden_values))
    monkeypatch.setattr(Orbits, "from_kwargs", forbidden_from_kwargs)

    propagated = propagate_2body(
        orbits,
        Timestamp.from_mjd([60001.0], scale="tdb"),
        max_processes=1,
    )
    assert len(propagated) == 1
    assert propagated.orbit_id.to_pylist() == ["arrow"]


def test_propagate_2body_single_failfast_nonfinite_input() -> None:
    t0 = Timestamp.from_mjd([60000.0], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["bad_orbit"],
        object_id=["bad_object"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[np.nan],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.017],
            vz=[0.0],
            time=t0,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    times = Timestamp.from_mjd([60001.0], scale="tdb")

    with pytest.raises(DynamicsNumericalError, match="non_finite_input_state"):
        propagate_2body(orbits, times, max_processes=1)


def test_propagate_2body_single_failfast_nonfinite_output(monkeypatch) -> None:
    t0 = Timestamp.from_mjd([60000.0], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["bad_orbit"],
        object_id=["bad_object"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.017],
            vz=[0.0],
            time=t0,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    times = Timestamp.from_mjd([60001.0], scale="tdb")

    def _nan_rust(*_args, **_kwargs):
        raise RuntimeError(
            "propagation row failure: reason=non_finite_output_state; "
            "output_row=0; input_orbit_index=0; input_time_index=0"
        )

    monkeypatch.setattr(propagation_module, "propagate_orbits_arrow", _nan_rust)

    with pytest.raises(DynamicsNumericalError, match="non_finite_output_state"):
        propagate_2body(orbits, times, max_processes=1)


@pytest.mark.profile
def test_profile_propagate_2body_single_matrix(propagated_orbits, tmp_path):
    """Profile the propagate_2body function with different combinations of orbits and times.
    Results are saved to a stats file that can be visualized with snakeviz."""
    # Clear the jax cache
    pass  # cache clear no longer needed (no JAX)

    # Create profiler
    profiler = cProfile.Profile(subcalls=True, builtins=True)
    profiler.bias = 0
    # Run profiling
    profiler.enable()
    n_orbits = [1, 5, 20]
    n_times = [1, 10, 100]
    for n_orbits_i, n_times_i in itertools.product(n_orbits, n_times):
        times = Timestamp.from_mjd(
            np.arange(0, n_times_i, 1),
            scale="tdb",
        )
        propagate_2body(propagated_orbits[:n_orbits_i], times=times)
    profiler.disable()

    # Save and print results
    stats_file = tmp_path / "precovery_profile.prof"
    profiler.dump_stats(stats_file)
    print(f"Run 'snakeviz {stats_file}' to view the profile results.")
