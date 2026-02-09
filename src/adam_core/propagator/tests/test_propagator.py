import time
from typing import Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import quivr as qv
import ray
from adam_assist import ASSISTPropagator

from adam_core.ray_cluster import initialize_use_ray

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin, OriginCodes
from ...coordinates.transform import transform_coordinates
from ...observations.exposures import Exposures
from ...observers.observers import Observers
from ...orbits.ephemeris import Ephemeris
from ...orbits.orbits import Orbits
from ...orbits.physical_parameters import PhysicalParameters
from ...orbits.variants import VariantEphemeris, VariantOrbits
from ...photometry import calculate_phase_angle
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

            # For photometry, store a heliocentric (emission-time) state; for this mock
            # ephemeris generator, assume zero light-time so emission == observation.
            origin_code = orbit.coordinates.origin.code[0].as_py()
            aberrated = CartesianCoordinates.from_kwargs(
                x=np.full(len(coords), orbit.coordinates.x[0].as_py()),
                y=np.full(len(coords), orbit.coordinates.y[0].as_py()),
                z=np.full(len(coords), orbit.coordinates.z[0].as_py()),
                vx=np.full(len(coords), orbit.coordinates.vx[0].as_py()),
                vy=np.full(len(coords), orbit.coordinates.vy[0].as_py()),
                vz=np.full(len(coords), orbit.coordinates.vz[0].as_py()),
                time=observers.coordinates.time,
                origin=Origin.from_kwargs(code=np.full(len(coords), origin_code)),
                frame="ecliptic",
            )

            ephemeris_i = Ephemeris.from_kwargs(
                orbit_id=pa.array(np.full(len(coords), orbit.orbit_id[0].as_py())),
                object_id=pa.array(np.full(len(coords), orbit.object_id[0].as_py())),
                coordinates=coords.to_spherical(),
                aberrated_coordinates=aberrated,
                light_time=np.full(len(coords), 0.0),
            )
            ephemeris_list.append(ephemeris_i)

        return qv.concatenate(ephemeris_list)


class TimeMajorVariantPropagator(Propagator, EphemerisMixin):
    """
    Propagator that expands (variant_orbits, times) in time-major order:
      [all variants @ t0, all variants @ t1, ...]

    This intentionally exercises the ordering assumptions inside generic ephemeris generation.
    """

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _propagate_orbits(
        self, orbits: VariantOrbits, times: Timestamp
    ) -> VariantOrbits:
        out = []
        for t in times:
            repeated_time = qv.concatenate([t] * len(orbits))
            out.append(orbits.set_column("coordinates.time", repeated_time))
        return qv.concatenate(out)


class VariantAwareMockPropagator(Propagator, EphemerisMixin):
    """
    Propagator that preserves VariantOrbits through propagation and does not
    override _generate_ephemeris, so EphemerisMixin._generate_ephemeris runs
    (time-major sort and np.tile(weights, len(observers))).
    """

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _propagate_orbits(
        self,
        orbits: Union[Orbits, VariantOrbits],
        times: Timestamp,
    ) -> Union[Orbits, VariantOrbits]:
        out = []
        for t in times:
            repeated_time = qv.concatenate([t] * len(orbits))
            out.append(orbits.set_column("coordinates.time", repeated_time))
        return qv.concatenate(out)


def test_propagator_single_worker():
    orbits = make_real_orbits(10)
    times = Timestamp.from_iso8601(["2020-01-01T00:00:00", "2020-01-01T00:00:01"])

    prop = MockPropagator()
    have = prop.propagate_orbits(orbits, times, max_processes=1)

    assert len(have) == len(orbits) * len(times)

    observers = Observers.from_code("X05", times)
    have = prop.generate_ephemeris(orbits, observers)

    assert len(have) == len(orbits) * len(times)


def test_generate_ephemeris_variant_orbits_pairs_observers_consistently():
    # Two identical variants, three distinct observer states at distinct times.
    # If variant propagation ordering and observer tiling are mismatched, the ephemeris
    # at a single requested time will show an artificial spread across variants.
    times = Timestamp.from_iso8601(
        [
            "2020-01-01T00:00:00Z",
            "2020-01-01T00:00:01Z",
            "2020-01-01T00:00:02Z",
        ]
    )
    orbit_epoch = times[:1]
    orbit_epoch_repeated = qv.concatenate([orbit_epoch] * 2)

    variants = VariantOrbits.from_kwargs(
        orbit_id=["o1", "o1"],
        variant_id=["v0", "v1"],
        object_id=["o1", "o1"],
        weights=[0.5, 0.5],
        weights_cov=[0.0, 0.0],
        coordinates=CartesianCoordinates.from_kwargs(
            # Identical variants: any spread at a fixed observer is a bug.
            x=[2.0, 2.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.0, 0.0],
            vz=[0.0, 0.0],
            time=orbit_epoch_repeated,
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )

    observers = Observers.from_kwargs(
        code=["X05", "X05", "X05"],
        coordinates=CartesianCoordinates.from_kwargs(
            # Deliberately distinct observer positions.
            x=[1.0, 0.0, -1.0],
            y=[0.0, 1.0, 0.0],
            z=[0.0, 0.0, 0.0],
            vx=[0.0, 0.0, 0.0],
            vy=[0.0, 0.0, 0.0],
            vz=[0.0, 0.0, 0.0],
            time=times,
            origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
            frame="ecliptic",
        ),
    )

    prop = TimeMajorVariantPropagator()
    eph = prop.generate_ephemeris(
        variants, observers, max_processes=1, predict_magnitudes=False
    )

    # Grab ephemeris rows at the first observer time (should be exactly 2 rows: v0, v1).
    t0 = times[:1]
    m = eph.coordinates.time.equals(t0, precision="ns")
    i = np.nonzero(m.to_numpy(zero_copy_only=False))[0]
    assert i.size == 2

    lon = eph.coordinates.lon.to_numpy(zero_copy_only=False)[i]
    lat = eph.coordinates.lat.to_numpy(zero_copy_only=False)[i]
    rho = eph.coordinates.rho.to_numpy(zero_copy_only=False)[i]

    # Identical variants at a fixed observer/time must produce identical ephemerides.
    assert float(np.max(lon) - np.min(lon)) < 1e-8
    assert float(np.max(lat) - np.min(lat)) < 1e-8
    assert float(np.max(rho) - np.min(rho)) < 1e-10


def test_generate_ephemeris_predicted_magnitudes_default_on():
    # Simple opposition geometry: object at 2 AU on +x, observer at 1 AU on +x.
    time = Timestamp.from_mjd([60000], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["o1"],
        object_id=["o1"],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[15.0], G=[0.15]),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    prop = MockPropagator()
    eph = prop.generate_ephemeris(orbits, observers, max_processes=1)

    assert not pc.all(pc.is_null(eph.predicted_magnitude_v)).as_py()

    expected = 15.0 + 5.0 * np.log10(2.0 * 1.0)
    have = eph.predicted_magnitude_v.to_numpy(zero_copy_only=False)[0]
    assert have == pytest.approx(expected, abs=1e-8)


def test_generate_ephemeris_phase_angle_default_off() -> None:
    # Simple opposition geometry: object at 2 AU on +x, observer at 1 AU on +x.
    time = Timestamp.from_mjd([60000], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["o1"],
        object_id=["o1"],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[15.0], G=[0.15]),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    prop = MockPropagator()
    eph = prop.generate_ephemeris(orbits, observers, max_processes=1)

    assert pc.all(pc.is_null(eph.alpha)).as_py()


def test_generate_ephemeris_phase_angle_enabled() -> None:
    # Simple opposition geometry: object at 2 AU on +x, observer at 1 AU on +x.
    time = Timestamp.from_mjd([60000], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["o1"],
        object_id=["o1"],
        # Phase angle should not depend on photometric parameters.
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[None], G=[None]),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    prop = MockPropagator()
    eph = prop.generate_ephemeris(
        orbits,
        observers,
        max_processes=1,
        predict_magnitudes=False,
        predict_phase_angle=True,
    )

    assert not pc.all(pc.is_null(eph.alpha)).as_py()
    have = eph.alpha.to_numpy(zero_copy_only=False)[0]
    assert have == pytest.approx(0.0, abs=1e-10)


def test_generate_ephemeris_phase_angle_90deg() -> None:
    # Quadrature: object at (1,0,0), observer at (1,1,0) -> phase angle 90°.
    time = Timestamp.from_mjd([60000], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["o1"],
        object_id=["o1"],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[None], G=[None]),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[1.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    prop = MockPropagator()
    eph = prop.generate_ephemeris(
        orbits,
        observers,
        max_processes=1,
        predict_magnitudes=False,
        predict_phase_angle=True,
    )
    assert not pc.all(pc.is_null(eph.alpha)).as_py()
    have = eph.alpha.to_numpy(zero_copy_only=False)[0]
    assert have == pytest.approx(90.0, abs=1e-10)


def test_generate_ephemeris_phase_angle_180deg() -> None:
    # Conjunction: Sun -> object -> observer on same ray; phase angle 180°.
    # Object at (2,0,0), observer at (3,0,0).
    time = Timestamp.from_mjd([60000], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["o1"],
        object_id=["o1"],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[None], G=[None]),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[3.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    prop = MockPropagator()
    eph = prop.generate_ephemeris(
        orbits,
        observers,
        max_processes=1,
        predict_magnitudes=False,
        predict_phase_angle=True,
    )
    assert not pc.all(pc.is_null(eph.alpha)).as_py()
    have = eph.alpha.to_numpy(zero_copy_only=False)[0]
    assert have == pytest.approx(180.0, abs=1e-10)


def test_generate_ephemeris_phase_angle_matches_calculate_phase_angle() -> None:
    # Cross-check: eph.alpha should match calculate_phase_angle for the same geometry.
    # Single orbit and observer so ephemeris has one row and expected has one value.
    time = Timestamp.from_mjd([60000], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["o1"],
        object_id=["o1"],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[None], G=[None]),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    expected = calculate_phase_angle(orbits.coordinates, observers)
    expected = np.asarray(expected, dtype=np.float64)

    prop = MockPropagator()
    eph = prop.generate_ephemeris(
        orbits,
        observers,
        max_processes=1,
        predict_magnitudes=False,
        predict_phase_angle=True,
    )
    have = eph.alpha.to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(have, expected, rtol=0.0, atol=1e-10)


def test_generate_ephemeris_phase_angle_and_magnitude_both_enabled() -> None:
    # Both predict_magnitudes and predict_phase_angle True; both columns populated.
    time = Timestamp.from_mjd([60000], scale="tdb")
    orbits = Orbits.from_kwargs(
        orbit_id=["o1"],
        object_id=["o1"],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[15.0], G=[0.15]),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    prop = MockPropagator()
    eph = prop.generate_ephemeris(
        orbits,
        observers,
        max_processes=1,
        predict_magnitudes=True,
        predict_phase_angle=True,
    )
    assert not pc.all(pc.is_null(eph.alpha)).as_py()
    assert not pc.all(pc.is_null(eph.predicted_magnitude_v)).as_py()
    assert eph.alpha.to_numpy(zero_copy_only=False)[0] == pytest.approx(0.0, abs=1e-10)


def test_generate_ephemeris_for_exposures_returns_v_magnitudes():
    time = Timestamp.from_mjd([60000, 60001], scale="tdb")
    exposures = Exposures.from_kwargs(
        id=["e1", "e2"],
        start_time=time,
        duration=[30.0, 30.0],
        filter=["g", "r"],
        observatory_code=["500", "500"],
        seeing=[None, None],
        depth_5sigma=[None, None],
    )

    orbits = Orbits.from_kwargs(
        orbit_id=["o1"],
        object_id=["o1"],
        physical_parameters=PhysicalParameters.from_kwargs(H_v=[15.0], G=[0.15]),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    prop = MockPropagator()
    observers = exposures.observers()
    eph = prop.generate_ephemeris(orbits, observers, predict_magnitudes=True)
    assert len(eph) == len(exposures)

    assert not pc.all(pc.is_null(eph.predicted_magnitude_v)).as_py()


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


def test_propagate_orbits_multiple_workers_ray_variant_orbits_input():
    """
    Regression test: VariantOrbits should be supported as an input to propagate_orbits
    under the ray parallel dispatcher.
    """
    base = make_real_orbits(4)
    variants = VariantOrbits.create(base, method="sigma-point")
    times = Timestamp.from_iso8601(["2020-01-01T00:00:00", "2020-01-01T00:00:01"])

    prop = MockPropagator()
    have = prop.propagate_orbits(variants, times, max_processes=2)
    assert isinstance(have, VariantOrbits)
    assert len(have) == len(variants) * len(times)


def test_propagate_orbits_ordering_orbit_id_then_time():
    """Propagated Orbits are sorted by (orbit_id, time) for observer-tiling alignment."""
    orbits = make_real_orbits(2)
    times = Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb")
    prop = MockPropagator()
    have = prop.propagate_orbits(orbits, times, max_processes=1)
    assert len(have) == 6
    oid = np.array([x.as_py() for x in have.orbit_id])
    days = have.coordinates.time.days.to_numpy(zero_copy_only=False)
    nanos = have.coordinates.time.nanos.to_numpy(zero_copy_only=False)
    for i in range(len(have) - 1):
        o_cur, o_nxt = oid[i], oid[i + 1]
        t_cur = (int(days[i]), int(nanos[i]))
        t_nxt = (int(days[i + 1]), int(nanos[i + 1]))
        assert (o_cur, t_cur) <= (o_nxt, t_nxt), "expected (orbit_id, time) order"


def test_propagate_orbits_ordering_orbit_id_then_variant_id_then_time():
    """Propagated VariantOrbits are sorted by (orbit_id, variant_id, time) for observer tiling."""
    base = make_real_orbits(1)
    variants = VariantOrbits.create(base, method="sigma-point")
    times = Timestamp.from_mjd([60000.0, 60001.0], scale="tdb")
    prop = MockPropagator()
    have = prop.propagate_orbits(variants, times, max_processes=1)
    assert isinstance(have, VariantOrbits)
    oid = np.array([x.as_py() for x in have.orbit_id])
    vid = np.array([x.as_py() for x in have.variant_id])
    days = have.coordinates.time.days.to_numpy(zero_copy_only=False)
    nanos = have.coordinates.time.nanos.to_numpy(zero_copy_only=False)
    for i in range(len(have) - 1):
        key_cur = (oid[i], vid[i], int(days[i]), int(nanos[i]))
        key_nxt = (oid[i + 1], vid[i + 1], int(days[i + 1]), int(nanos[i + 1]))
        assert key_cur <= key_nxt


def test_generate_ephemeris_variant_ordering_and_weights_aligned():
    """
    VariantEphemeris has same order as propagate_orbits: (orbit_id, variant_id, time).
    Weights/weights_cov align with variant so collapse() gets correct per-variant weights.
    """
    times = Timestamp.from_iso8601(
        ["2020-01-01T00:00:00Z", "2020-01-01T00:00:01Z"],
    )
    orbit_epoch = times[:1]
    orbit_epoch_x2 = qv.concatenate([orbit_epoch] * 2)
    # Two variants with distinct weights so we can check alignment
    w0, w1 = 0.2, 0.8
    wc0, wc1 = 0.1, 0.9
    variants = VariantOrbits.from_kwargs(
        orbit_id=["o1", "o1"],
        variant_id=["v0", "v1"],
        object_id=["o1", "o1"],
        weights=[w0, w1],
        weights_cov=[wc0, wc1],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0, 2.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.0, 0.0],
            vz=[0.0, 0.0],
            time=orbit_epoch_x2,
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_code("500", times)
    prop = VariantAwareMockPropagator()
    eph = prop.generate_ephemeris(
        variants, observers, max_processes=1, predict_magnitudes=False
    )
    assert isinstance(eph, VariantEphemeris)
    assert len(eph) == 4  # 2 variants * 2 times

    # Same order as propagate_orbits: (orbit_id, variant_id, time). Weights aligned per variant.
    weights = eph.weights.to_numpy(zero_copy_only=False)
    weights_cov = eph.weights_cov.to_numpy(zero_copy_only=False)
    # First two rows = v0 @ t0, t1; next two = v1 @ t0, t1
    np.testing.assert_array_almost_equal(weights[:2], [w0, w0])
    np.testing.assert_array_almost_equal(weights_cov[:2], [wc0, wc0])
    np.testing.assert_array_almost_equal(weights[2:4], [w1, w1])
    np.testing.assert_array_almost_equal(weights_cov[2:4], [wc1, wc1])

    # Group by time using apply_mask: each time slice has 2 rows (v0, v1)
    for t_idx in range(len(times)):
        t_slice = times[t_idx : t_idx + 1]
        mask = eph.coordinates.time.equals(t_slice, precision="ns")
        group = eph.apply_mask(mask)
        assert len(group) == 2
        assert list(np.unique(np.array([x.as_py() for x in group.variant_id]))) == [
            "v0",
            "v1",
        ]
        np.testing.assert_array_almost_equal(
            group.weights.to_numpy(zero_copy_only=False), [w0, w1]
        )


def test_variant_ephemeris_weights_follow_variant_id_every_row():
    """
    For every row in VariantEphemeris, weight and weights_cov must equal the input
    weight for that (orbit_id, variant_id). Validates np.repeat alignment with no mixup.
    """
    times = Timestamp.from_iso8601(
        ["2020-01-01T00:00:00Z", "2020-01-01T00:00:01Z", "2020-01-01T00:00:02Z"],
    )
    orbit_epoch = times[:1]
    # Two orbits, two variants each -> 4 rows
    variants = VariantOrbits.from_kwargs(
        orbit_id=["o1", "o1", "o2", "o2"],
        variant_id=["v0", "v1", "v0", "v1"],
        object_id=["o1", "o1", "o2", "o2"],
        weights=[0.1, 0.9, 0.3, 0.7],
        weights_cov=[0.05, 0.95, 0.25, 0.75],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0, 2.0, 3.0, 3.0],
            y=[0.0, 0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0, 0.0],
            vx=[0.0] * 4,
            vy=[0.0] * 4,
            vz=[0.0] * 4,
            time=qv.concatenate([orbit_epoch] * 4),
            origin=Origin.from_kwargs(code=["SUN"] * 4),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_code("500", times)
    prop = VariantAwareMockPropagator()
    eph = prop.generate_ephemeris(
        variants, observers, max_processes=1, predict_magnitudes=False
    )
    assert isinstance(eph, VariantEphemeris)
    assert len(eph) == 4 * 3  # 4 variant rows * 3 times

    # Build expected (orbit_id, variant_id) -> (weight, weight_cov) from input
    expected = {
        ("o1", "v0"): (0.1, 0.05),
        ("o1", "v1"): (0.9, 0.95),
        ("o2", "v0"): (0.3, 0.25),
        ("o2", "v1"): (0.7, 0.75),
    }
    oid = np.array([x.as_py() for x in eph.orbit_id])
    vid = np.array([x.as_py() for x in eph.variant_id])
    weights = eph.weights.to_numpy(zero_copy_only=False)
    weights_cov = eph.weights_cov.to_numpy(zero_copy_only=False)
    for i in range(len(eph)):
        key = (oid[i], vid[i])
        assert key in expected, f"row {i}: unexpected (orbit_id, variant_id) {key}"
        exp_w, exp_wc = expected[key]
        assert weights[i] == pytest.approx(
            exp_w
        ), f"row {i}: weight for {key} should be {exp_w}, got {weights[i]}"
        assert weights_cov[i] == pytest.approx(
            exp_wc
        ), f"row {i}: weights_cov for {key} should be {exp_wc}, got {weights_cov[i]}"


def test_generate_ephemeris_variant_ephemeris_light_time_populated() -> None:
    """VariantEphemeris.light_time is populated (non-null, finite) when using EphemerisMixin."""
    times = Timestamp.from_iso8601(
        ["2020-01-01T00:00:00Z", "2020-01-01T00:00:01Z"],
    )
    orbit_epoch = times[:1]
    variants = VariantOrbits.from_kwargs(
        orbit_id=["o1", "o1"],
        variant_id=["v0", "v1"],
        object_id=["o1", "o1"],
        weights=[0.5, 0.5],
        weights_cov=[0.0, 0.0],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0, 2.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.0, 0.0],
            vz=[0.0, 0.0],
            time=qv.concatenate([orbit_epoch] * 2),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_code("500", times)
    prop = VariantAwareMockPropagator()
    eph = prop.generate_ephemeris(
        variants, observers, max_processes=1, predict_magnitudes=False
    )
    assert isinstance(eph, VariantEphemeris)
    assert not pc.all(pc.is_null(eph.light_time)).as_py()
    lt = eph.light_time.to_numpy(zero_copy_only=False)
    assert np.all(np.isfinite(lt))
    assert np.all(lt >= 0.0)


def test_generate_ephemeris_variant_ephemeris_phase_angle_populated() -> None:
    """VariantEphemeris.alpha is populated when predict_phase_angle=True (opposition -> 0°)."""
    time = Timestamp.from_mjd([60000], scale="tdb")
    orbit_epoch = time
    variants = VariantOrbits.from_kwargs(
        orbit_id=["o1", "o1"],
        variant_id=["v0", "v1"],
        object_id=["o1", "o1"],
        weights=[0.5, 0.5],
        weights_cov=[0.0, 0.0],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[None, None], G=[None, None]
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0, 2.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.0, 0.0],
            vz=[0.0, 0.0],
            time=qv.concatenate([orbit_epoch] * 2),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    prop = VariantAwareMockPropagator()
    eph = prop.generate_ephemeris(
        variants,
        observers,
        max_processes=1,
        predict_magnitudes=False,
        predict_phase_angle=True,
    )
    assert isinstance(eph, VariantEphemeris)
    assert not pc.all(pc.is_null(eph.alpha)).as_py()
    alpha = eph.alpha.to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(alpha, 0.0, atol=1e-6)


def test_generate_ephemeris_variant_ephemeris_predicted_magnitude_populated() -> None:
    """VariantEphemeris.predicted_magnitude_v is populated when predict_magnitudes=True and H,G set."""
    time = Timestamp.from_mjd([60000], scale="tdb")
    orbit_epoch = time
    variants = VariantOrbits.from_kwargs(
        orbit_id=["o1", "o1"],
        variant_id=["v0", "v1"],
        object_id=["o1", "o1"],
        weights=[0.5, 0.5],
        weights_cov=[0.0, 0.0],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.0, 15.0], G=[0.15, 0.15]
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0, 2.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.0, 0.0],
            vz=[0.0, 0.0],
            time=qv.concatenate([orbit_epoch] * 2),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_kwargs(
        code=["500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=time,
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    prop = VariantAwareMockPropagator()
    eph = prop.generate_ephemeris(
        variants,
        observers,
        max_processes=1,
        predict_magnitudes=True,
        predict_phase_angle=False,
    )
    assert isinstance(eph, VariantEphemeris)
    assert not pc.all(pc.is_null(eph.predicted_magnitude_v)).as_py()
    expected = 15.0 + 5.0 * np.log10(2.0 * 1.0)
    mags = eph.predicted_magnitude_v.to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(mags, expected, atol=1e-8)


def test_variant_ephemeris_ordering_multiple_orbits_two_times():
    """
    With 2 orbits × 2 variants × 2 times, rows must be (orbit_id, variant_id, time)
    and weights must repeat per variant across times (np.repeat pattern).
    """
    times = Timestamp.from_iso8601(
        ["2020-01-01T00:00:00Z", "2020-01-01T00:00:01Z"],
    )
    orbit_epoch = times[:1]
    variants = VariantOrbits.from_kwargs(
        orbit_id=["o1", "o1", "o2", "o2"],
        variant_id=["v0", "v1", "v0", "v1"],
        object_id=["o1", "o1", "o2", "o2"],
        weights=[0.2, 0.8, 0.4, 0.6],
        weights_cov=[0.0] * 4,
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0, 2.0, 3.0, 3.0],
            y=[0.0, 0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0, 0.0],
            vx=[0.0] * 4,
            vy=[0.0] * 4,
            vz=[0.0] * 4,
            time=qv.concatenate([orbit_epoch] * 4),
            origin=Origin.from_kwargs(code=["SUN"] * 4),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_code("500", times)
    prop = VariantAwareMockPropagator()
    eph = prop.generate_ephemeris(
        variants, observers, max_processes=1, predict_magnitudes=False
    )
    # Order: o1,v0,t0; o1,v0,t1; o1,v1,t0; o1,v1,t1; o2,v0,t0; o2,v0,t1; o2,v1,t0; o2,v1,t1
    assert len(eph) == 8
    oid = np.array([x.as_py() for x in eph.orbit_id])
    vid = np.array([x.as_py() for x in eph.variant_id])
    want_order = [
        ("o1", "v0"),
        ("o1", "v0"),
        ("o1", "v1"),
        ("o1", "v1"),
        ("o2", "v0"),
        ("o2", "v0"),
        ("o2", "v1"),
        ("o2", "v1"),
    ]
    for i in range(8):
        assert (oid[i], vid[i]) == want_order[
            i
        ], f"row {i}: expected {want_order[i]}, got ({oid[i]}, {vid[i]})"
    want_weights = [0.2, 0.2, 0.8, 0.8, 0.4, 0.4, 0.6, 0.6]
    np.testing.assert_array_almost_equal(
        eph.weights.to_numpy(zero_copy_only=False), want_weights
    )


def test_variant_ephemeris_input_shuffled_still_correct_weights():
    """
    Input VariantOrbits in time-major order (or any order); output must still have
    weights aligned to variant_id. Validates internal sort + np.repeat.
    """
    times = Timestamp.from_iso8601(
        ["2020-01-01T00:00:00Z", "2020-01-01T00:00:01Z"],
    )
    orbit_epoch = times[:1]
    # Build with v1 first, v0 second (reversed variant order). Internal sort by
    # (orbit_id, variant_id, time) + np.repeat must still assign 0.2 to v0, 0.8 to v1.
    orbit_epoch_x2 = qv.concatenate([orbit_epoch] * 2)
    variants = VariantOrbits.from_kwargs(
        orbit_id=["o1", "o1"],
        variant_id=["v1", "v0"],  # reversed order
        object_id=["o1", "o1"],
        weights=[0.8, 0.2],  # v1=0.8, v0=0.2
        weights_cov=[0.9, 0.1],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0, 2.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.0, 0.0],
            vz=[0.0, 0.0],
            time=orbit_epoch_x2,
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )
    # Sort input to time-major: same epoch so still (v1, v0). Then we propagate;
    # internal sort in _generate_ephemeris sorts by (orbit_id, variant_id, time) so
    # we get (o1, v0, t0), (o1, v0, t1), (o1, v1, t0), (o1, v1, t1). Weights from
    # orbits_sorted (variant-major) are [0.2, 0.8], np.repeat(., 2) = [0.2, 0.2, 0.8, 0.8].
    observers = Observers.from_code("500", times)
    prop = VariantAwareMockPropagator()
    eph = prop.generate_ephemeris(
        variants, observers, max_processes=1, predict_magnitudes=False
    )
    assert len(eph) == 4
    # Expect v0 weight 0.2, v1 weight 0.8 on the right rows
    vid = np.array([x.as_py() for x in eph.variant_id])
    weights = eph.weights.to_numpy(zero_copy_only=False)
    for i in range(4):
        if vid[i] == "v0":
            assert weights[i] == pytest.approx(
                0.2
            ), f"row {i} v0 should have weight 0.2"
        else:
            assert weights[i] == pytest.approx(
                0.8
            ), f"row {i} v1 should have weight 0.8"


def test_propagate_orbits_times_sorted_before_backend():
    """Pass unsorted times; output must still be (orbit_id, time) with times chronological per orbit."""
    orbits = make_real_orbits(2)
    # Reverse time order
    times_asc = Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb")
    times_reversed = Timestamp.from_mjd([60002.0, 60001.0, 60000.0], scale="tdb")
    prop = MockPropagator()
    have_asc = prop.propagate_orbits(orbits, times_asc, max_processes=1)
    have_rev = prop.propagate_orbits(orbits, times_reversed, max_processes=1)
    assert len(have_asc) == len(have_rev) == 6
    # Both outputs must be (orbit_id, time) ordered with times chronological within orbit
    for have in (have_asc, have_rev):
        oid = np.array([x.as_py() for x in have.orbit_id])
        days = have.coordinates.time.days.to_numpy(zero_copy_only=False)
        nanos = have.coordinates.time.nanos.to_numpy(zero_copy_only=False)
        for i in range(len(have) - 1):
            o_cur, o_nxt = oid[i], oid[i + 1]
            t_cur = (int(days[i]), int(nanos[i]))
            t_nxt = (int(days[i + 1]), int(nanos[i + 1]))
            assert (o_cur, t_cur) <= (o_nxt, t_nxt)
    # Same orbit_id and time sets (order of rows may differ by orbit)
    np.testing.assert_array_equal(
        np.sort(have_asc.coordinates.time.mjd().to_numpy(zero_copy_only=False)),
        np.sort(have_rev.coordinates.time.mjd().to_numpy(zero_copy_only=False)),
    )


def test_propagate_orbits_ray_concatenation_order():
    """
    With max_processes > 1, results are collected in completion order then sorted.
    Assert final order is (orbit_id, time) for Orbits and (orbit_id, variant_id, time) for VariantOrbits.
    """
    if ray.is_initialized():
        ray.shutdown()
    orbits = make_real_orbits(3)
    times = Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb")
    prop = MockPropagator()

    have_serial = prop.propagate_orbits(orbits, times, max_processes=1)
    have_ray = prop.propagate_orbits(orbits, times, max_processes=2, chunk_size=2)

    assert len(have_serial) == len(have_ray) == 9
    for have in (have_serial, have_ray):
        oid = np.array([x.as_py() for x in have.orbit_id])
        days = have.coordinates.time.days.to_numpy(zero_copy_only=False)
        nanos = have.coordinates.time.nanos.to_numpy(zero_copy_only=False)
        for i in range(len(have) - 1):
            o_cur, o_nxt = oid[i], oid[i + 1]
            t_cur = (int(days[i]), int(nanos[i]))
            t_nxt = (int(days[i + 1]), int(nanos[i + 1]))
            assert (o_cur, t_cur) <= (
                o_nxt,
                t_nxt,
            ), "expected (orbit_id, time) order after concatenation"

    np.testing.assert_array_equal(
        have_serial.orbit_id.to_numpy(zero_copy_only=False),
        have_ray.orbit_id.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_allclose(
        have_serial.coordinates.values,
        have_ray.coordinates.values,
        rtol=0,
        atol=1e-15,
    )

    # VariantOrbits with multiple chunks
    base = make_real_orbits(2)
    variants = VariantOrbits.create(base, method="sigma-point")
    times_v = Timestamp.from_mjd([60000.0, 60001.0], scale="tdb")
    have_v_serial = prop.propagate_orbits(variants, times_v, max_processes=1)
    have_v_ray = prop.propagate_orbits(variants, times_v, max_processes=2, chunk_size=5)

    assert len(have_v_serial) == len(have_v_ray)
    for have in (have_v_serial, have_v_ray):
        oid = np.array([x.as_py() for x in have.orbit_id])
        vid = np.array([x.as_py() for x in have.variant_id])
        days = have.coordinates.time.days.to_numpy(zero_copy_only=False)
        nanos = have.coordinates.time.nanos.to_numpy(zero_copy_only=False)
        for i in range(len(have) - 1):
            key_cur = (oid[i], vid[i], int(days[i]), int(nanos[i]))
            key_nxt = (oid[i + 1], vid[i + 1], int(days[i + 1]), int(nanos[i + 1]))
            assert (
                key_cur <= key_nxt
            ), "expected (orbit_id, variant_id, time) order after concatenation"


def test_generate_ephemeris_ray_concatenation_order_and_weights():
    """
    With max_processes > 1, ephemeris chunks are collected in completion order then sorted.
    Assert final order and weights match single-process (no mixup after concatenation).
    """
    if ray.is_initialized():
        ray.shutdown()
    times = Timestamp.from_iso8601(
        ["2020-01-01T00:00:00Z", "2020-01-01T00:00:01Z", "2020-01-01T00:00:02Z"],
    )
    orbit_epoch = times[:1]
    variants = VariantOrbits.from_kwargs(
        orbit_id=["o1", "o1", "o2", "o2"],
        variant_id=["v0", "v1", "v0", "v1"],
        object_id=["o1", "o1", "o2", "o2"],
        weights=[0.1, 0.9, 0.3, 0.7],
        weights_cov=[0.05, 0.95, 0.25, 0.75],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[2.0, 2.0, 3.0, 3.0],
            y=[0.0, 0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0, 0.0],
            vx=[0.0] * 4,
            vy=[0.0] * 4,
            vz=[0.0] * 4,
            time=qv.concatenate([orbit_epoch] * 4),
            origin=Origin.from_kwargs(code=["SUN"] * 4),
            frame="ecliptic",
        ),
    )
    observers = Observers.from_code("500", times)
    prop = VariantAwareMockPropagator()

    eph_serial = prop.generate_ephemeris(
        variants, observers, max_processes=1, predict_magnitudes=False
    )
    eph_ray = prop.generate_ephemeris(
        variants, observers, max_processes=2, chunk_size=2, predict_magnitudes=False
    )

    assert len(eph_serial) == len(eph_ray) == 12
    np.testing.assert_array_equal(
        eph_serial.orbit_id.to_numpy(zero_copy_only=False),
        eph_ray.orbit_id.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_array_equal(
        eph_serial.variant_id.to_numpy(zero_copy_only=False),
        eph_ray.variant_id.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_allclose(
        eph_serial.weights.to_numpy(zero_copy_only=False),
        eph_ray.weights.to_numpy(zero_copy_only=False),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        eph_serial.weights_cov.to_numpy(zero_copy_only=False),
        eph_ray.weights_cov.to_numpy(zero_copy_only=False),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        eph_serial.coordinates.values, eph_ray.coordinates.values, rtol=0, atol=0
    )


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
    Test that light-time computation remains stable (no overflow/NaNs)
    even for extremely large relative motion between object and observer.
    """
    # Create a single orbit with very high velocity that will drive light-time
    # computation into a pathological regime.
    orbit = Orbits.from_kwargs(
        orbit_id=["1"],
        object_id=["1"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1],  # Start at origin
            y=[1],
            z=[1],
            vx=[1e9],  # Extremely high velocity (unphysical)
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

    class LightTimePropagator(Propagator, EphemerisMixin):
        # Use the generic ephemeris implementation from EphemerisMixin;
        # this minimal propagator just tiles the orbit to the requested times.
        def _propagate_orbits(self, orbits: Orbits, times: Timestamp) -> Orbits:
            all_times = []
            for t in times:
                repeated_time = qv.concatenate([t] * len(orbits))
                orbits.coordinates.time = repeated_time
                all_times.append(orbits)
            return qv.concatenate(all_times)

    prop = LightTimePropagator()

    # With such an extreme configuration, light-time should be detected as invalid
    # and a ValueError should be raised.
    with pytest.raises(
        ValueError,
        match="Light travel time is NaN or too large and propagation will break.",
    ):
        prop.generate_ephemeris(orbit, observer, max_processes=1)


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

    # Verify that coordinates.time - aberrated_coordinates.time is equal to light_time.
    # Rescale both to the same time scale before taking the difference.
    coords_tdb = ephemeris.coordinates.time.rescale("tdb")
    aberr_tdb = ephemeris.aberrated_coordinates.time.rescale("tdb")
    time_difference_days, time_difference_nanos = coords_tdb.difference(aberr_tdb)
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


def test_generate_ephemeris_performance_benchmark():
    """
    Benchmark test to ensure generate_ephemeris performance with multiprocessing
    is reasonable and doesn't degrade significantly.

    This test compares single-process vs multi-process performance to ensure
    multiprocessing provides a benefit rather than a penalty.
    """
    # Create a moderately sized test case
    orbits = make_real_orbits(10)
    times = Timestamp.from_mjd(np.arange(60001, 60101), scale="tdb")
    observers = Observers.from_code("500", times)

    prop = ASSISTPropagator()
    initialize_use_ray(num_cpus=4)

    # Benchmark single process
    start_time = time.time()
    ephemeris_single = prop.generate_ephemeris(
        orbits,
        observers,
        covariance=True,
        num_samples=100,
        max_processes=1,
        chunk_size=100,
        seed=42,
    )
    single_process_time = time.time() - start_time

    # Benchmark multiple processes
    start_time = time.time()
    ephemeris_multi = prop.generate_ephemeris(
        orbits,
        observers,
        covariance=True,
        num_samples=100,
        max_processes=4,
        chunk_size=100,
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
