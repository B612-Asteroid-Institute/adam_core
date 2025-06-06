import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import quivr as qv
from adam_assist import ASSISTPropagator

from adam_core.ray_cluster import initialize_use_ray

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin, OriginCodes
from ...coordinates.transform import transform_coordinates
from ...observers.observers import Observers
from ...orbits.ephemeris import Ephemeris
from ...orbits.orbits import Orbits
from ...orbits.variants import VariantOrbits
from ...time.time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from ..propagator import EphemerisMixin, Propagator


class MockPropagator(Propagator, EphemerisMixin):

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # MockPropagator propagates orbits by just setting the time of the orbits.
    def _propagate_orbits(self, orbits: Orbits, times: Timestamp) -> Orbits:
        all_times = []
        for t in times:
            repeated_time = qv.concatenate([t] * len(orbits))
            orbits.coordinates.time = repeated_time
            all_times.append(orbits)
        all_times = qv.concatenate(all_times)

        # Artifically change origin to test that it is preserved in the final output
        output = all_times.set_column(
            "coordinates",
            transform_coordinates(
                all_times.coordinates,
                origin_out=OriginCodes["SATURN_BARYCENTER"],
                frame_out="equatorial",
            ),
        )

        return output

    # MockPropagator generated ephemeris by just subtracting the state from
    # the state of the observers
    def _generate_ephemeris(self, orbits: Orbits, observers: Observers) -> Ephemeris:
        ephemeris_list = []
        observer_coordinates = observers.coordinates.values
        for orbit in orbits:
            topocentric_state = orbit.coordinates.values - observer_coordinates
            coords = CartesianCoordinates.from_kwargs(
                x=topocentric_state[:, 0],
                y=topocentric_state[:, 1],
                z=topocentric_state[:, 2],
                vx=topocentric_state[:, 3],
                vy=topocentric_state[:, 4],
                vz=topocentric_state[:, 5],
                time=observers.coordinates.time,
                origin=Origin.from_kwargs(code=observers.code),
                frame="ecliptic",
            )

            ephemeris_i = Ephemeris.from_kwargs(
                orbit_id=pa.array(np.full(len(coords), orbit.orbit_id[0].as_py())),
                object_id=pa.array(np.full(len(coords), orbit.object_id[0].as_py())),
                coordinates=coords.to_spherical(),
                aberrated_coordinates=coords,
                light_time=np.full(len(coords), 0.0),
            )
            ephemeris_list.append(ephemeris_i)

        return qv.concatenate(ephemeris_list)


def test_propagator_single_worker():
    orbits = make_real_orbits(10)
    times = Timestamp.from_iso8601(["2020-01-01T00:00:00", "2020-01-01T00:00:01"])

    prop = MockPropagator()
    have = prop.propagate_orbits(orbits, times, max_processes=1)

    assert len(have) == len(orbits) * len(times)

    observers = Observers.from_code("X05", times)
    have = prop.generate_ephemeris(orbits, observers)

    assert len(have) == len(orbits) * len(times)


RAY_INSTALLED = False
try:
    import ray

    RAY_INSTALLED = True

except ImportError:
    pass


def test_propagator_multiple_workers_ray():
    orbits = make_real_orbits(10)
    times = Timestamp.from_iso8601(["2020-01-01T00:00:00", "2020-01-01T00:00:01"])

    prop = MockPropagator()
    have = prop.propagate_orbits(orbits, times, max_processes=4)

    assert len(have) == len(orbits) * len(times)

    observers = Observers.from_code("X05", times)
    have = prop.generate_ephemeris(orbits, observers, max_processes=4)

    assert len(have) == len(orbits) * len(times)

    # Now lets put times and orbits into the object store first
    orbits_ref = ray.put(orbits)
    time_ref = ray.put(times)

    have = prop.propagate_orbits(orbits_ref, time_ref, max_processes=4)

    assert len(have) == len(orbits) * len(times)

    observers_ref = ray.put(observers)
    have = prop.generate_ephemeris(orbits_ref, observers_ref, max_processes=4)

    assert len(have) == len(orbits) * len(times)


def test_propagate_different_origins():
    """
    Test that we are returning propagated orbits with their original origins
    """
    orbits = Orbits.from_kwargs(
        orbit_id=["1", "2"],
        object_id=["1", "2"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1, 1],
            y=[1, 1],
            z=[1, 1],
            vx=[1, 1],
            vy=[1, 1],
            vz=[1, 1],
            time=Timestamp.from_mjd([60000, 60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(
                code=["SOLAR_SYSTEM_BARYCENTER", "EARTH_MOON_BARYCENTER"]
            ),
        ),
    )

    prop = MockPropagator()
    propagated_orbits = prop.propagate_orbits(
        orbits, Timestamp.from_mjd([60001, 60002, 60003], scale="tdb")
    )
    orbit_one_results = propagated_orbits.select("orbit_id", "1")
    orbit_two_results = propagated_orbits.select("orbit_id", "2")
    # Assert that the origin codes for each set of results is unique
    # and that it matches the original input
    assert len(orbit_one_results.coordinates.origin.code.unique()) == 1
    assert (
        orbit_one_results.coordinates.origin.code.unique()[0].as_py()
        == "SOLAR_SYSTEM_BARYCENTER"
    )
    assert len(orbit_two_results.coordinates.origin.code.unique()) == 1
    assert (
        orbit_two_results.coordinates.origin.code.unique()[0].as_py()
        == "EARTH_MOON_BARYCENTER"
    )


def test_light_time_distance_threshold():
    """
    Test that _add_light_time raises a ValueError when an object gets too far from the observer.
    """
    # Create a single orbit with very high velocity
    orbit = Orbits.from_kwargs(
        orbit_id=["1"],
        object_id=["1"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1],  # Start at origin
            y=[1],
            z=[1],
            vx=[1e9],  # Very high velocity (will quickly exceed our limits)
            vy=[0],
            vz=[0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"]),
        ),
    )

    # Create an observer at a fixed position
    observer = Observers.from_kwargs(
        code=["500"],  # Arbitrary observer code
        coordinates=CartesianCoordinates.from_kwargs(
            x=[0],
            y=[0],
            z=[0],
            vx=[0],
            vy=[0],
            vz=[0],
            time=Timestamp.from_mjd([60020], scale="tdb"),  # 1 day later
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"]),
        ),
    )

    prop = MockPropagator()

    # The object will have moved ~86400 * 1e6 AU in one day, which should trigger the threshold
    with pytest.raises(
        ValueError,
        match="Distance from observer is NaN or too large and propagation will break.",
    ):
        prop._add_light_time(orbit, observer)


@pytest.mark.parametrize("max_processes", [1, 4])
@pytest.mark.parametrize("input_time_scale", ["utc", "tdb"])
def test_generate_ephemeris_unordered_observers(max_processes, input_time_scale):
    """
    Test that ephemeris generation works correctly even when observers
    are not ordered by time, verifying that physical positions don't get mixed up
    between different objects.
    """
    # Create two distinct orbits - one near Earth and one near Mars
    orbits = Orbits.from_kwargs(
        orbit_id=["far", "near"],
        object_id=["far", "near"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[3.0, 1],  # Pick something far from both observers (earth) and sun
            y=[3.0, 1],
            z=[3.0, 1],
            vx=[0.1, 0.1],
            vy=[0.0, 0.0],  # Different orbital velocities
            vz=[0.0, 0.0],
            time=Timestamp.from_mjd([59000, 59000], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )
    # Create observers with deliberately unordered times
    times = Timestamp.from_mjd(
        [59005, 59004, 59005, 59004, 59006, 59006], scale=input_time_scale
    )
    codes = ["500", "500", "X05", "X05", "X05", "500"]  # Mix of observatory codes
    observers = Observers.from_codes(codes, times)

    propagator = ASSISTPropagator()
    ephemeris = propagator.generate_ephemeris(
        orbits, observers, max_processes=max_processes
    )

    # Basic ordering checks
    assert len(ephemeris) == 12  # 2 objects × 6 times

    # Verify that coordinates.time - aberrated_coordinates.time is equal to light_time
    time_difference_days, time_difference_nanos = ephemeris.coordinates.time.rescale(
        "tdb"
    ).difference(ephemeris.aberrated_coordinates.time)
    fractional_days = pc.divide(time_difference_nanos, 86400 * 1e9)
    time_difference = pc.add(time_difference_days, fractional_days)
    np.testing.assert_allclose(
        time_difference.to_numpy(zero_copy_only=False),
        ephemeris.light_time.to_numpy(zero_copy_only=False),
        atol=1e-6,
    )

    # Verify that the near-Earth object is consistently closer than the Mars object
    far_ephem = ephemeris.select("orbit_id", "far")
    near_ephem = ephemeris.select("orbit_id", "near")

    # make sure the far object is always further than the near object
    far_positions = np.linalg.norm(
        far_ephem.aberrated_coordinates.values[:, :3], axis=1
    )
    near_positions = np.linalg.norm(
        near_ephem.aberrated_coordinates.values[:, :3], axis=1
    )
    assert np.all(
        far_positions > near_positions
    ), "Far aberrated positions should be consistently further than near positions"

    # Verify observer codes match the expected order after sorting for each object
    sorted_observers = observers.sort_by(
        ["coordinates.time.days", "coordinates.time.nanos", "code"]
    )
    for orbit_id in ["far", "near"]:
        orbit_ephem = ephemeris.select("orbit_id", orbit_id)
        np.testing.assert_array_equal(
            orbit_ephem.coordinates.origin.code.to_numpy(zero_copy_only=False),
            sorted_observers.code.to_numpy(zero_copy_only=False),
        )

    # Link back to observers to verify correct correspondence
    linkage = ephemeris.link_to_observers(observers)
    assert len(linkage.all_unique_values) == len(observers)

    # Verify that the returned ephemeris is in UTC
    assert ephemeris.coordinates.time.scale == "utc"


@pytest.mark.parametrize("input_time_scale", ["utc", "tdb"])
def test_generate_ephemeris_variant_orbits(input_time_scale):
    """Test that ephemeris generation works correctly with variant orbits and respects covariance flag."""

    # Create base orbits
    base_orbits = Orbits.from_kwargs(
        orbit_id=["test1", "test2"],
        object_id=["test1", "test2"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0],
            y=[1.0, 2.0],
            z=[1.0, 2.0],
            vx=[0.1, 0.2],
            vy=[0.1, 0.2],
            vz=[0.1, 0.2],
            time=Timestamp.from_mjd([60000, 60000]),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )

    # Create variant orbits
    variant_orbits = VariantOrbits.from_kwargs(
        orbit_id=["test1", "test1", "test2", "test2"],
        variant_id=["0", "1", "0", "1"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.1, 2.0, 2.1],
            y=[1.0, 1.1, 2.0, 2.1],
            z=[1.0, 1.1, 2.0, 2.1],
            vx=[0.1, 0.11, 0.2, 0.21],
            vy=[0.1, 0.11, 0.2, 0.21],
            vz=[0.1, 0.11, 0.2, 0.21],
            time=Timestamp.from_mjd([60000, 60000, 60000, 60000]),
            origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN", "SUN"]),
            frame="ecliptic",
        ),
    )

    # Create observers
    times = Timestamp.from_mjd([60001, 60002], scale=input_time_scale)
    observers = Observers.from_code("500", times)

    prop = MockPropagator()

    # Test with variant orbits - should work
    ephemeris = prop.generate_ephemeris(variant_orbits, observers, covariance=False)
    assert len(ephemeris) == len(variant_orbits) * len(times)

    # Test with variant orbits and covariance=True - should raise assertion error
    with pytest.raises(
        AssertionError, match="Covariance is not supported for VariantOrbits"
    ):
        prop.generate_ephemeris(variant_orbits, observers, covariance=True)

    # Test with regular orbits and covariance=True - should work
    ephemeris = prop.generate_ephemeris(base_orbits, observers, covariance=True)
    assert len(ephemeris) == len(base_orbits) * len(times)

    # Verify that the returned ephemeris is in UTC
    assert ephemeris.coordinates.time.scale == "utc"


@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
def test_generate_ephemeris_performance_benchmark():
    """
    Benchmark test to ensure generate_ephemeris performance with multiprocessing
    is reasonable and doesn't degrade significantly.

    This test compares single-process vs multi-process performance to ensure
    multiprocessing provides a benefit rather than a penalty.
    """
    # Create a moderately sized test case
    orbits = make_real_orbits(10)
    times = Timestamp.from_mjd(np.arange(60001, 60005), scale="tdb")
    observers = Observers.from_code("500", times)

    prop = ASSISTPropagator()
    initialize_use_ray(num_cpus=4)

    # Benchmark single process
    start_time = time.time()
    ephemeris_single = prop.generate_ephemeris(
        orbits,
        observers,
        covariance=True,
        num_samples=10,
        max_processes=1,
        chunk_size=1,
        seed=42,
    )
    single_process_time = time.time() - start_time

    # Benchmark multiple processes
    start_time = time.time()
    ephemeris_multi = prop.generate_ephemeris(
        orbits,
        observers,
        covariance=True,
        num_samples=10,
        max_processes=4,
        chunk_size=1,
        seed=42,
    )
    multi_process_time = time.time() - start_time

    # Verify results are identical
    assert len(ephemeris_single) == len(ephemeris_multi)
    assert len(ephemeris_single) == len(orbits) * len(times)

    # Performance check: multiprocessing shouldn't be more than 1x slower
    # (allowing for overhead, but catching catastrophic performance regressions)
    max_acceptable_ratio = 0.9
    actual_ratio = (
        multi_process_time / single_process_time
        if single_process_time > 0
        else float("inf")
    )
    assert actual_ratio < max_acceptable_ratio, (
        f"Multiprocessing performance is {actual_ratio:.2f}x slower than single process "
        f"({multi_process_time:.3f}s vs {single_process_time:.3f}s). "
        f"This suggests a performance regression in the multiprocessing code."
    )
