from astropy.time import Time
from ...orbits import Orbits
from ... import coordinates
from ...utils.helpers.orbits import make_real_orbits
from .. import Propagator

import quivr
import pyarrow as pa


class MockPropagator(Propagator):
    # MockPropagator propagates orbits by just setting the time of the orbits.
    def _propagate_orbits(self, orbits: Orbits, times: Time) -> Orbits:
        all_times = []
        for t in times:
            repeated_time = Time([t] * len(orbits))
            orbits.coordinates.time = coordinates.Times.from_astropy(
                repeated_time
            ).to_structarray()
            all_times.append(orbits)

        return quivr.concatenate(all_times)

    def _generate_ephemeris(self, orbits: Orbits, observers):
        raise NotImplementedError("not impleemented")


def test_propagator_single_worker():
    orbits = make_real_orbits(10)
    times = Time(["2020-01-01T00:00:00", "2020-01-01T00:00:01"])

    prop = MockPropagator()
    have = prop.propagate_orbits(orbits, times, num_jobs=1)

    assert len(have) == len(orbits) * len(times)


def test_propagator_multiple_workers():
    orbits = make_real_orbits(10)
    times = Time(["2020-01-01T00:00:00", "2020-01-01T00:00:01"])

    prop = MockPropagator()
    have = prop.propagate_orbits(orbits, times, num_jobs=4)

    assert len(have) == len(orbits) * len(times)
