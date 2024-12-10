import numpy as np
import pyarrow as pa
import quivr as qv

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin, OriginCodes
from ...coordinates.transform import transform_coordinates
from ...observers.observers import Observers
from ...orbits.ephemeris import Ephemeris
from ...orbits.orbits import Orbits
from ...time.time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from ..propagator import EphemerisMixin, Propagator


class MockPropagator(Propagator, EphemerisMixin):
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
