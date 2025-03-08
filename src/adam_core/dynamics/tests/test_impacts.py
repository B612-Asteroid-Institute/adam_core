import numpy as np
import pyarrow.compute as pc

from ...coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    OriginCodes,
    SphericalCoordinates,
    transform_coordinates,
)
from ...orbits import Orbits, VariantOrbits
from ...propagator import Propagator
from ...time import Timestamp
from ..impacts import (
    CollisionConditions,
    CollisionEvent,
    ImpactMixin,
    ImpactProbabilities,
    calculate_impact_probabilities,
    calculate_impacts,
    calculate_mahalanobis_distance,
)


class MockImpactPropagator(Propagator, ImpactMixin):
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _propagate_orbits(self, orbits: Orbits, times: Timestamp) -> Orbits:
        return orbits

    def _detect_collisions(
        self, orbits: Orbits, num_days: float, conditions: CollisionConditions
    ) -> Orbits:
        # Artificially set the orbits.coordinates.times to the end time
        # except for the orbits who impacted
        new_times = orbits.coordinates.time.add_days(num_days)
        orbits = orbits.set_column("coordinates.time", new_times)

        # Do a transform away from the input origin and frame
        orbits_transformed = orbits.set_column(
            "coordinates",
            transform_coordinates(
                orbits.coordinates,
                representation_out=CartesianCoordinates,
                origin_out=OriginCodes.EARTH,
                frame_out="equatorial",
            ),
        )

        # Pick random orbit to impact
        variant = orbits_transformed[0]
        impact = CollisionEvent.from_kwargs(
            orbit_id=variant.orbit_id,
            variant_id=variant.variant_id,
            coordinates=CartesianCoordinates.from_kwargs(
                x=variant.coordinates.x,
                y=variant.coordinates.y,
                z=variant.coordinates.z,
                vx=variant.coordinates.vx,
                vy=variant.coordinates.vy,
                vz=variant.coordinates.vz,
                time=variant.coordinates.time,
                origin=variant.coordinates.origin,
                frame=variant.coordinates.frame,
            ),
            condition_id=["1"],
            collision_coordinates=transform_coordinates(
                variant.coordinates,
                representation_out=SphericalCoordinates,
                origin_out=OriginCodes.EARTH,
                frame_out="itrf93",
            ),
            collision_object=Origin.from_kwargs(code=["EARTH"]),
            stopping_condition=[False],
        )

        return variant, impact


def test_calculate_impacts():
    """
    Tests the i/o of calculate_impacts
    """
    orbits = Orbits.from_kwargs(
        orbit_id=["1", "2", "3"],
        object_id=["1", "2", "3"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0, 3.0],
            y=[1.0, 2.0, 3.0],
            z=[1.0, 2.0, 3.0],
            vx=[1.0, 2.0, 3.0],
            vy=[1.0, 2.0, 3.0],
            vz=[1.0, 2.0, 3.0],
            time=Timestamp.from_iso8601(["2020-01-01T00:00:00"] * 3),
            origin=Origin.from_kwargs(code=["SUN"] * 3),
            frame="ecliptic",
            covariance=CoordinateCovariances.from_sigmas(
                np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ),
        ),
    )

    prop = MockImpactPropagator()
    variants, impacts = calculate_impacts(
        orbits, 10, prop, num_samples=1000, processes=1, seed=42
    )
    assert pc.all(
        pc.equal(
            variants.coordinates.time.mjd().to_pylist(),
            orbits[0].coordinates.time.add_days(10).mjd().to_pylist()[0],
        )
    )
    assert pc.all(
        pc.equal(
            variants.coordinates.origin.code.unique(),
            orbits.coordinates.origin.code.unique(),
        )
    ).as_py()
    assert variants.coordinates.frame == orbits.coordinates.frame


def test_calculate_impact_probabilities():
    """
    Tests the i/o of calculate_impact_probabilities
    """
    variants = VariantOrbits.from_kwargs(
        orbit_id=["1", "1", "1", "2", "2", "2", "3", "3", "3"],
        object_id=["1", "1", "1", "2", "2", "2", "3", "3", "3"],
        variant_id=["1", "2", "3", "1", "2", "3", "1", "2", "3"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0, 3.0] * 3,
            y=[1.0, 2.0, 3.0] * 3,
            z=[1.0, 2.0, 3.0] * 3,
            vx=[1.0, 2.0, 3.0] * 3,
            vy=[1.0, 2.0, 3.0] * 3,
            vz=[1.0, 2.0, 3.0] * 3,
            time=Timestamp.from_iso8601(["2020-01-01T00:00:00"] * 9),
            origin=Origin.from_kwargs(code=["SUN"] * 9),
            frame="ecliptic",
        ),
    )

    impacts = CollisionEvent.from_kwargs(
        orbit_id=["1", "2", "2"],
        variant_id=["1", "1", "2"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0, 3.0],
            y=[1.0, 2.0, 3.0],
            z=[1.0, 2.0, 3.0],
            vx=[1.0, 2.0, 3.0],
            vy=[1.0, 2.0, 3.0],
            vz=[1.0, 2.0, 3.0],
            time=Timestamp.from_kwargs(
                days=[59200, 59200, 59200],
                nanos=[0, 0, 43200 * 1e9],
                scale="utc",
            ),
            origin=Origin.from_kwargs(code=["SUN"] * 3),
            frame="ecliptic",
        ),
        condition_id=["1", "1", "1"],
        collision_object=Origin.from_kwargs(code=["EARTH", "EARTH", "EARTH"]),
        collision_coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.0, 1.0],
            lon=[1.0, 2.0, 3.0],
            lat=[1.0, 2.0, 3.0],
            time=Timestamp.from_kwargs(
                days=[59200, 59200, 59200],
                nanos=[0, 0, 43200 * 1e9],
                scale="utc",
            ),
        ),
        stopping_condition=[False, False, False],
    )

    impact_conditions = CollisionConditions.from_kwargs(
        condition_id=["1"],
        collision_object=Origin.from_kwargs(code=["EARTH"]),
        collision_distance=[1.0],
        stopping_condition=[False],
    )

    ip = calculate_impact_probabilities(variants, impacts, conditions=impact_conditions)

    desired = ImpactProbabilities.from_kwargs(
        condition_id=["1", "1", "1"],
        orbit_id=["1", "2", "3"],
        impacts=[1, 2, 0],
        variants=[3, 3, 3],
        cumulative_probability=[1 / 3, 2 / 3, 0.0],
        mean_impact_time=Timestamp.from_kwargs(
            days=[59200, 59200, None],
            nanos=[0, 43200 * 1e9 / 2, None],
            scale="utc",
            permit_nulls=True,
        ),
        stddev_impact_time=[0.0, 0.25, None],
        minimum_impact_time=Timestamp.from_kwargs(
            days=[59200, 59200, None],
            nanos=[0, 0, None],
            scale="utc",
            permit_nulls=True,
        ),
        maximum_impact_time=Timestamp.from_kwargs(
            days=[59200, 59200, None],
            nanos=[0, 43200 * 1e9, None],
            scale="utc",
            permit_nulls=True,
        ),
    )

    assert ip == desired


def test_calculate_mahalanobis_distance():
    """ """
    observed_orbit = Orbits.from_kwargs(
        orbit_id=["1", "1", "1"],
        object_id=["1", "1", "1"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0, 3.0],
            y=[1.0, 2.0, 3.0],
            z=[1.0, 2.0, 3.0],
            vx=[1.0, 2.0, 3.0],
            vy=[1.0, 2.0, 3.0],
            vz=[1.0, 2.0, 3.0],
            time=Timestamp.from_iso8601(["2020-01-01T00:00:00"] * 3),
            origin=Origin.from_kwargs(code=["SUN"] * 3),
            frame="ecliptic",
            covariance=CoordinateCovariances.from_sigmas(
                np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ),
        ),
    )

    predicted_orbit = Orbits.from_kwargs(
        orbit_id=["1", "1", "1"],
        object_id=["1", "1", "1"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0, 3.0],
            y=[1.0, 2.0, 3.0],
            z=[1.0, 2.0, 3.0],
            vx=[1.0, 2.0, 3.0],
            vy=[1.0, 2.0, 3.0],
            vz=[1.0, 2.0, 3.0],
            time=Timestamp.from_iso8601(["2020-01-01T00:00:00"] * 3),
            origin=Origin.from_kwargs(code=["SUN"] * 3),
            frame="ecliptic",
        ),
    )

    md = calculate_mahalanobis_distance(observed_orbit, predicted_orbit)
    assert np.all(md == 0.0)

    less_perfect_observed_orbit = Orbits.from_kwargs(
        orbit_id=["1", "1", "1"],
        object_id=["1", "1", "1"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.1, 2.1, 3.1],
            y=[1.1, 2.1, 3.1],
            z=[1.1, 2.1, 3.1],
            vx=[1.1, 2.1, 3.1],
            vy=[1.1, 2.1, 3.1],
            vz=[1.1, 2.1, 3.1],
            time=Timestamp.from_iso8601(["2020-01-01T00:00:00"] * 3),
            origin=Origin.from_kwargs(code=["SUN"] * 3),
            frame="ecliptic",
            covariance=CoordinateCovariances.from_sigmas(
                np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                )
            ),
        ),
    )

    md = calculate_mahalanobis_distance(less_perfect_observed_orbit, predicted_orbit)
    np.testing.assert_almost_equal(md, np.array([0.24494897, 0.24494897, 0.24494897]))
