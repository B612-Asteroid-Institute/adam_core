import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import quivr as qv

from adam_core.constants import KM_P_AU
from adam_core.constants import Constants as c
from adam_core.coordinates import CartesianCoordinates, Origin

from ..coordinates.residuals import Residuals
from ..coordinates.spherical import SphericalCoordinates
from ..orbits import Orbits
from ..orbits.variants import VariantOrbits
from ..propagator import Propagator
from ..propagator.propagator import OrbitType
from ..time import Timestamp

logger = logging.getLogger(__name__)

C = c.C

# Use the Earth's equatorial radius as used in DE4XX ephemerides
# adam_core defines it in au but we need it in km
EARTH_RADIUS_KM = c.R_EARTH_EQUATORIAL * KM_P_AU


class ImpactProbabilities(qv.Table):
    condition_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()
    impacts = qv.Int64Column()
    variants = qv.Int64Column()
    cumulative_probability = qv.Float64Column()
    mean_impact_time = Timestamp.as_column(nullable=True)
    stddev_impact_time = qv.Float64Column(nullable=True)
    minimum_impact_time = Timestamp.as_column(nullable=True)
    maximum_impact_time = Timestamp.as_column(nullable=True)


class CollisionConditions(qv.Table):
    #: Unique identifier for the condition
    condition_id = qv.LargeStringColumn()
    #: Name of the object with which to detect collisions
    collision_object = Origin.as_column()
    #: Distance from the object at which to detect collisions (in km)
    collision_distance = qv.Float64Column()
    #: Whether to stop propagation after a collision
    stopping_condition = qv.BooleanColumn()

    @classmethod
    def default(cls) -> "CollisionConditions":
        from adam_core import _rust_native

        condition_id, object_code, distance, stopping = (
            _rust_native.default_collision_condition_values()
        )
        return cls.from_kwargs(
            condition_id=[condition_id],
            collision_object=Origin.from_kwargs(code=[object_code]),
            collision_distance=[distance],
            stopping_condition=[stopping],
        )


class CollisionEvent(qv.Table):
    #: Unique identifier for the orbit
    orbit_id = qv.LargeStringColumn()
    #: Unique identifier for the variant
    variant_id = qv.LargeStringColumn(nullable=True)
    #: Cartesian coordinates of the colliding variant
    coordinates = CartesianCoordinates.as_column()
    #: Unique identifier for the condition
    condition_id = qv.LargeStringColumn()
    #: Name of the object with which collisions were detected
    collision_object = Origin.as_column()
    #: Spherical coordinates of the impact in the body-centered frame (does not
    #: have to be a body-fixed frame)
    collision_coordinates = SphericalCoordinates.as_column()
    #: Whether the propagation was stopped after a collision
    stopping_condition = qv.BooleanColumn()

    def preview(self) -> None:
        """
        Plot the risk corridor for the given impacts.
        """
        from .plots import plot_risk_corridor

        fig = plot_risk_corridor(self, title="Risk Corridor")
        fig.show()


class ImpactMixin(ABC):
    """Collision-detection contract for propagators.

    Concrete (Rust-backed) propagators implement :meth:`detect_collisions` as a
    single Python->Rust crossing (rayon parallelism lives in the backend).
    adam_core provides only the abstract contract -- no Python/Ray composition.
    Callers sample variants (via :func:`calculate_impacts`) before calling this.
    """

    @abstractmethod
    def detect_collisions(
        self,
        orbits: OrbitType,
        num_days: int,
        conditions: Optional[CollisionConditions] = None,
        max_processes: Optional[int] = 1,
        chunk_size: Optional[int] = 100,
    ) -> Tuple[OrbitType, CollisionConditions]:
        """
        Detect collisions for each orbit in orbits after num_days.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            Orbits for which to detect impacts.
        num_days : int
            Number of days after which to detect impacts.
        conditions : `~adam_core.orbits.earth_impacts.CollisionConditions`
            Conditions for detecting collisions, including:
            - condition_id: Unique identifier for the condition.
            - collision_object_name: Name of the object with which to detect collisions.
            - collision_distance: Distance from the object at which to detect collisions.
            - stopping_condition: Whether to stop propagation after a collision.
        max_processes : int or None, optional
            Maximum number of processes to launch. If None then the number of
            processes will be equal to the number of cores on the machine. If 1
            then no multiprocessing will be used.

        Returns
        -------
        propagated : `~adam_core.orbits.OrbitType`
            The input orbits propagated to the end of simulation.
        impacts : `~adam_core.orbits.earth_impacts.CollisionEvent`
            Impacts/collisions detected for the orbits. Includes:
            - orbit_id: Unique identifier for the orbit.
            - distance: Distance from the collision object.
            - coordinates: Cartesian coordinates of the impact.
            - variant_id: Unique identifier for the variant.
            - condition_id: Unique identifier for the condition.
            - collision_object_name: Name of the object with which collisions were detected.
            - collision_distance: Distance from the object at which collisions were detected.
            - stopping_condition: Whether the propagation was stopped after a collision.
        """
        ...


def calculate_impacts(
    orbits: Orbits,
    num_days: int,
    propagator: Propagator,
    num_samples: int = 1000,
    processes: Optional[int] = None,
    seed: Optional[int] = None,
    conditions: Optional[CollisionConditions] = None,
) -> Tuple[OrbitType, CollisionEvent]:
    """
    Calculate the impacts for each variant orbit generated from the input orbits.

    Parameters
    ----------
    orbits : `~adam_core.orbits.orbits.Orbits`
        Orbits for which to calculate impact probabilities.
    num_days : int
        Number of days to propagate the orbits.
    propagator : `~adam_core.propagator.propagator.Propagator`
        Propagator to use for orbit propagation.
    num_samples : int, optional
        Number of samples to take over the period, by default 1000.
    processes : int, optional
        Number of processes to use for parallelization, by default all available.
    seed : int, optional
        Seed for random number generation, by default None.

    Returns
    -------

    """
    assert issubclass(
        propagator.__class__, ImpactMixin
    ), "Propagator must support impact detection."
    logger.info("Generating variants...")
    variants = VariantOrbits.create(
        orbits, method="monte-carlo", num_samples=num_samples, seed=seed
    )
    if conditions is None:
        conditions = CollisionConditions.default()
    logger.info("Detecting impacts...")
    results, collisions = propagator.detect_collisions(
        variants,
        num_days,
        conditions=conditions,
        max_processes=processes,
    )

    return results, collisions


def calculate_impact_probabilities(
    variants: VariantOrbits,
    collision_events: CollisionEvent,
    conditions: Optional[CollisionConditions] = None,
) -> ImpactProbabilities:
    """
    Calculate the impact probabilities for each variant orbit generated from the input orbits.
    Parameters
    ----------
    variants : `~adam_core.orbits.variants.VariantOrbits`
        Variant orbits for which to calculate impact probabilities.
    impacts : `~adam_core.orbits.impacts.Impacts`
        Impacts for the variant orbits.
    Returns
    -------
    impact_probabilities : `~adam_core.orbits.impact_probabilities.ImpactProbabilities`
        Impact probabilities for the variant orbits.
    """

    if conditions is None:
        conditions = CollisionConditions.default()
    if len(variants) == 0:
        return ImpactProbabilities.empty()

    from adam_core import _rust_native

    collision_time = collision_events.coordinates.time
    scale = (
        collision_time.scale
        if len(collision_events) > 0
        else variants.coordinates.time.scale
    )
    stats = _rust_native.impact_probability_stats_numpy(
        variants.orbit_id.to_pylist(),
        collision_events.orbit_id.to_pylist(),
        collision_events.condition_id.to_pylist(),
        np.ascontiguousarray(
            collision_time.days.to_numpy(zero_copy_only=False), dtype=np.int64
        ),
        np.ascontiguousarray(
            collision_time.nanos.to_numpy(zero_copy_only=False), dtype=np.int64
        ),
        scale,
        conditions.condition_id.to_pylist(),
    )

    mean_time = Timestamp.from_kwargs(
        days=stats["mean_days"],
        nanos=stats["mean_nanos"],
        scale=scale,
        permit_nulls=True,
    )
    minimum_time = Timestamp.from_kwargs(
        days=stats["minimum_days"],
        nanos=stats["minimum_nanos"],
        scale=scale,
        permit_nulls=True,
    )
    maximum_time = Timestamp.from_kwargs(
        days=stats["maximum_days"],
        nanos=stats["maximum_nanos"],
        scale=scale,
        permit_nulls=True,
    )
    return ImpactProbabilities.from_kwargs(
        condition_id=stats["condition_id"],
        orbit_id=stats["orbit_id"],
        impacts=stats["impacts"],
        variants=stats["variants"],
        cumulative_probability=stats["cumulative_probability"],
        mean_impact_time=mean_time,
        stddev_impact_time=stats["stddev_impact_time"],
        minimum_impact_time=minimum_time,
        maximum_impact_time=maximum_time,
    )


def link_impacting_variants(variants, impacts):
    """
    Link variants to the orbits from which they were generated.
    Parameters
    ----------
    orbits : `~adam_core.orbits.orbits.Orbits`
        Orbits from which the variants were generated.
    Returns
    -------
    linkage : `~quivr.MultiKeyLinkage[Orbits, VariantOrbits]`
        Linkage between variants and orbits.
    """

    return qv.MultiKeyLinkage(
        impacts,
        variants,
        left_keys={
            "orbit_id": impacts.orbit_id,
            "variant_id": impacts.variant_id,
        },
        right_keys={
            "orbit_id": variants.orbit_id,
            "variant_id": variants.variant_id,
        },
    )


def calculate_mahalanobis_distance(
    observed_orbit: OrbitType, predicted_orbit: OrbitType
) -> npt.NDArray[np.float64]:
    """
    Calculate the Mahalanobis distance between an observed orbit and a predicted orbit.
    Parameters
    ----------
    observed_orbit : `~adam_core.coordinates.CartesianCoordinates`
        Observed orbit.
    observed_covariance : `ndarray`
        Covariance of the observed orbit.
    predicted_orbit : `~adam_core.coordinates.CartesianCoordinates`
        Predicted orbit.
    Returns
    -------
    mahalanobis_distance : `ndarray` (N)
        Mahalanobis distance between the observed and predicted orbits.
    """

    residuals = Residuals.calculate(
        observed_orbit.coordinates, predicted_orbit.coordinates
    )
    from adam_core import _rust_native

    return np.asarray(
        _rust_native.sqrt_values_numpy(
            np.ascontiguousarray(
                residuals.chi2.to_numpy(zero_copy_only=False), dtype=np.float64
            )
        ),
        dtype=np.float64,
    )
