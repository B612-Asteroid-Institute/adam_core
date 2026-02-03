import cProfile
import itertools

import jax
import numpy as np
import pytest
import spiceypy as sp
from astropy import units as u

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...orbits import Orbits
from ...orbits.physical_parameters import PhysicalParameters
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from ..propagation import _propagate_2body, _propagate_2body_vmap, propagate_2body

RAY_INSTALLED = False
try:
    import ray

    RAY_INSTALLED = True
except ImportError:
    pass


def test__propagate_2body_against_spice_elliptical(orbital_elements):
    # Test propagation against SPICE for elliptical orbits.

    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN" for i in range(len(cartesian_elements))])
    mu = origin.mu()

    dts = np.arange(0, 10000, 10)
    for i, (t0_i, cartesian_i) in enumerate(zip(t0, cartesian_elements)):

        spice_propagated = np.empty((len(dts), 6))
        cartesian_propagated = np.empty((len(dts), 6))

        for j, dt_i in enumerate(dts):

            spice_propagated[j] = sp.prop2b(
                mu[i], np.ascontiguousarray(cartesian_i), dt_i
            )
            cartesian_propagated[j] = _propagate_2body(
                cartesian_i, t0_i, t0_i + dt_i, mu[i]
            )

        # Calculate difference between SPICE and adam_core
        diff = cartesian_propagated - spice_propagated

        # Calculate offset in position in cm
        r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
        # Calculate offset in velocity in mm/s
        v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

        # Assert positions are to within 10 cm
        np.testing.assert_array_less(r_diff, 10)
        # Assert velocities are to within 1 mm/s
        np.testing.assert_array_less(v_diff, 1)


def test__propagate_2body_against_spice_hyperbolic(orbital_elements):
    # Test propagation against SPICE for hyperbolic orbits.

    orbital_elements = orbital_elements[orbital_elements["e"] > 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN" for i in range(len(cartesian_elements))])
    mu = origin.mu()

    dts = np.arange(0, 10000, 10)
    for i, (t0_i, cartesian_i) in enumerate(zip(t0, cartesian_elements)):

        spice_propagated = np.empty((len(dts), 6))
        cartesian_propagated = np.empty((len(dts), 6))

        for j, dt_i in enumerate(dts):

            spice_propagated[j] = sp.prop2b(
                mu[i], np.ascontiguousarray(cartesian_i), dt_i
            )
            cartesian_propagated[j] = _propagate_2body(
                cartesian_i, t0_i, t0_i + dt_i, mu[i]
            )

        # Calculate difference between SPICE and adam_core
        diff = cartesian_propagated - spice_propagated

        # Calculate offset in position in cm
        r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
        # Calculate offset in velocity in mm/s
        v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

        # Assert positions are to within 10 cm
        np.testing.assert_array_less(r_diff, 10)
        # Assert velocities are to within 1 mm/s
        np.testing.assert_array_less(v_diff, 1)


def test_propagate_2body_against_spice_elliptical(orbital_elements):
    # Test propagation against SPICE for elliptical orbits.

    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN" for i in range(len(cartesian_elements))])
    mu = origin.mu()

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
            origin=origin,
            frame="ecliptic",
        ),
    )

    # Set propagation times (same for all orbits)
    times = Timestamp.from_mjd(
        t0.min() + np.arange(0, 10000, 10),
        scale="tdb",
    )

    # Propagate orbits with SPICE and accumulate results
    spice_propagated = []
    times_mjd = times.mjd()
    for i, cartesian_i in enumerate(cartesian_elements):

        # Calculate dts from t0 for this orbit (each orbit's t0 is different)
        # but the final times we are propagating to are the same for all orbits
        dts = times_mjd - t0[i]
        spice_propagated_i = np.empty((len(dts), 6))
        for j, dt_i in enumerate(dts):
            spice_propagated_i[j] = sp.prop2b(
                mu[i], np.ascontiguousarray(cartesian_i), dt_i
            )

        spice_propagated.append(spice_propagated_i)

    spice_propagated = np.vstack(spice_propagated)

    # Propagate orbits with adam_core
    orbits_propagated = propagate_2body(orbits, times)

    # Calculate difference between SPICE and adam_core
    diff = orbits_propagated.coordinates.values - spice_propagated

    # Calculate offset in position in cm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 10 cm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 1 mm/s
    np.testing.assert_array_less(v_diff, 1)


def test_propagate_2body_against_spice_hyperbolic(orbital_elements):
    # Test propagation against SPICE for hyperbolic orbits.
    orbital_elements = orbital_elements[orbital_elements["e"] > 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN" for i in range(len(cartesian_elements))])
    mu = origin.mu()

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
            origin=origin,
            frame="ecliptic",
        ),
    )

    # Set propagation times (same for all orbits)
    times = Timestamp.from_mjd(
        t0.min() + np.arange(0, 10000, 10),
        scale="tdb",
    )

    # Propagate orbits with SPICE and accumulate results
    spice_propagated = []
    times_mjd = times.mjd()
    for i, cartesian_i in enumerate(cartesian_elements):

        # Calculate dts from t0 for this orbit (each orbit's t0 is different)
        # but the final times we are propagating to are the same for all orbits
        dts = times_mjd - t0[i]
        spice_propagated_i = np.empty((len(dts), 6))
        for j, dt_i in enumerate(dts):
            spice_propagated_i[j] = sp.prop2b(
                mu[i], np.ascontiguousarray(cartesian_i), dt_i
            )

        spice_propagated.append(spice_propagated_i)

    spice_propagated = np.vstack(spice_propagated)

    # Propagate orbits with adam_core
    orbits_propagated = propagate_2body(orbits, times)

    # Calculate difference between SPICE and adam_core
    diff = orbits_propagated.coordinates.values - spice_propagated

    # Calculate offset in position in cm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 10 cm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 1 mm/s
    np.testing.assert_array_less(v_diff, 1)


def test_propagate_2body_against_spice_elliptical_barycentric(
    orbital_elements_barycentric,
):
    # Test propagation against SPICE for elliptical orbits.
    orbital_elements = orbital_elements_barycentric[
        orbital_elements_barycentric["e"] < 1.0
    ]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(
        code=["SOLAR_SYSTEM_BARYCENTER" for i in range(len(cartesian_elements))]
    )
    mu = origin.mu()

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
            origin=origin,
            frame="ecliptic",
        ),
    )

    # Set propagation times (same for all orbits)
    times = Timestamp.from_mjd(
        t0.min() + np.arange(0, 10000, 10),
        scale="tdb",
    )

    # Propagate orbits with SPICE and accumulate results
    spice_propagated = []
    times_mjd = times.mjd()
    for i, cartesian_i in enumerate(cartesian_elements):

        # Calculate dts from t0 for this orbit (each orbit's t0 is different)
        # but the final times we are propagating to are the same for all orbits
        dts = times_mjd - t0[i]
        spice_propagated_i = np.empty((len(dts), 6))
        for j, dt_i in enumerate(dts):
            spice_propagated_i[j] = sp.prop2b(
                mu[i], np.ascontiguousarray(cartesian_i), dt_i
            )

        spice_propagated.append(spice_propagated_i)

    spice_propagated = np.vstack(spice_propagated)

    # Propagate orbits with adam_core
    orbits_propagated = propagate_2body(orbits, times)

    # Calculate difference between SPICE and adam_core
    diff = orbits_propagated.coordinates.values - spice_propagated

    # Calculate offset in position in cm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 10 cm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 1 mm/s
    np.testing.assert_array_less(v_diff, 1)


def test_propagate_2body_against_spice_hyperbolic_barycentric(
    orbital_elements_barycentric,
):
    # Test propagation against SPICE for hyperbolic orbits.

    orbital_elements = orbital_elements_barycentric[
        orbital_elements_barycentric["e"] > 1.0
    ]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(
        code=["SOLAR_SYSTEM_BARYCENTER" for i in range(len(cartesian_elements))]
    )
    mu = origin.mu()

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
            origin=origin,
            frame="ecliptic",
        ),
    )

    # Set propagation times (same for all orbits)
    times = Timestamp.from_mjd(
        t0.min() + np.arange(0, 10000, 10),
        scale="tdb",
    )

    # Propagate orbits with SPICE and accumulate results
    spice_propagated = []
    times_mjd = times.mjd()
    for i, cartesian_i in enumerate(cartesian_elements):

        # Calculate dts from t0 for this orbit (each orbit's t0 is different)
        # but the final times we are propagating to are the same for all orbits
        dts = times_mjd - t0[i]
        spice_propagated_i = np.empty((len(dts), 6))
        for j, dt_i in enumerate(dts):
            spice_propagated_i[j] = sp.prop2b(
                mu[i], np.ascontiguousarray(cartesian_i), dt_i
            )

        spice_propagated.append(spice_propagated_i)

    spice_propagated = np.vstack(spice_propagated)

    # Propagate orbits with adam_core
    orbits_propagated = propagate_2body(orbits, times)

    # Calculate difference between SPICE and adam_core
    diff = orbits_propagated.coordinates.values - spice_propagated

    # Calculate offset in position in cm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 10 cm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 1 mm/s
    np.testing.assert_array_less(v_diff, 1)


def test_benchmark__propagate_2body(benchmark, orbital_elements):
    # Benchmark _propagate_2body with a single orbit propagated forward 1 day
    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN" for i in range(len(orbital_elements))])
    mu = origin.mu()

    benchmark(_propagate_2body, cartesian_elements[0], t0[0], t0[0] + 1, mu[0])


def test_benchmark__propagate_2body_vmap(benchmark, orbital_elements):
    # Benchmark the vectorized map version of _propagate_2body with a single
    # orbit propagated forward 1 day
    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    origin = Origin.from_kwargs(code=["SUN" for i in range(len(orbital_elements))])
    mu = origin.mu()

    benchmark(
        _propagate_2body_vmap,
        cartesian_elements[:1],
        t0[:1],
        t0[:1] + 1,
        mu[:1],
        1000,
        1e-14,
    )


def test_benchmark_propagate_2body(benchmark, orbital_elements):
    # Benchmark propagate_2body with a single orbit propagated forward 1 day
    # This function appears to add substantial overhead, so we benchmark it
    # separately from _propagate_2body
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
def test_benchmark_propagate_2body_matrix(benchmark, propagated_orbits):
    # Clear the jax cache
    jax.clear_caches()

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


def test_propagate_2body_preserves_physical_parameters():
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


def test_propagate_2body_does_not_include_padded_rows() -> None:
    """
    `process_in_chunks` pads the final chunk to a fixed size. Ensure the propagation
    output only contains the true (n_orbits * n_times) rows.
    """
    orbits = make_real_orbits(1)
    orbit_mjd = orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    base_mjd = float(orbit_mjd[0])

    n_times = 201  # not divisible by chunk_size=200
    times = Timestamp.from_mjd(base_mjd + np.arange(n_times, dtype=np.float64), scale="tdb")
    propagated = propagate_2body(orbits, times)

    assert len(propagated) == n_times
    out_mjd = propagated.coordinates.time.rescale("tdb").mjd().to_numpy(zero_copy_only=False)
    in_mjd = times.rescale("tdb").mjd().to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(out_mjd, in_mjd)


@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
def test_propagate_2body_ray_matches_serial() -> None:
    # Ensure a clean local ray runtime for this test.
    if ray.is_initialized():  # type: ignore[name-defined]
        ray.shutdown()  # type: ignore[name-defined]
    ray.init(num_cpus=2, include_dashboard=False)  # type: ignore[name-defined]

    orbits = make_real_orbits(5)
    base_mjd = float(np.median(orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False)))
    times = Timestamp.from_mjd(base_mjd + np.arange(25, dtype=np.float64), scale="tdb")

    serial = propagate_2body(orbits, times, max_processes=1)
    parallel = propagate_2body(orbits, times, max_processes=2, chunk_size=2)

    assert len(serial) == len(parallel)
    np.testing.assert_array_equal(
        serial.orbit_id.to_numpy(zero_copy_only=False),
        parallel.orbit_id.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_allclose(serial.coordinates.values, parallel.coordinates.values, rtol=0, atol=0)

    ray.shutdown()  # type: ignore[name-defined]



@pytest.mark.profile
def test_profile_propagate_2body_matrix(propagated_orbits, tmp_path):
    """Profile the propagate_2body function with different combinations of orbits and times.
    Results are saved to a stats file that can be visualized with snakeviz."""
    # Clear the jax cache
    jax.clear_caches()

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
