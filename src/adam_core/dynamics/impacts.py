import logging
from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

from adam_core.constants import KM_P_AU
from adam_core.constants import Constants as c
from adam_core.coordinates import CartesianCoordinates
from adam_core.ray_cluster import initialize_use_ray

from ..coordinates.residuals import Residuals
from ..orbits import Orbits
from ..orbits.variants import VariantOrbits
from ..propagator import Propagator
from ..propagator.propagator import OrbitType
from ..propagator.utils import _iterate_chunks
from ..time import Timestamp

logger = logging.getLogger(__name__)

C = c.C

# Use the Earth's equatorial radius as used in DE4XX ephemerides
# adam_core defines it in au but we need it in km
EARTH_RADIUS_KM = c.R_EARTH_EQUATORIAL * KM_P_AU

RAY_INSTALLED = False
try:
    import ray
    from ray import ObjectRef

    RAY_INSTALLED = True
except ImportError:
    pass


if RAY_INSTALLED:

    @ray.remote
    def impact_worker_ray(idx_chunk, orbits, propagator_class, num_days):
        prop = propagator_class()
        orbits_chunk = orbits.take(idx_chunk)
        variants, impacts = prop._detect_collisions(orbits_chunk, num_days)
        return variants, impacts



class ImpactProbabilities(qv.Table):
    condition_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()
    impacts = qv.Int64Column()
    variants = qv.Int64Column()
    cumulative_probability = qv.Float64Column()
    minimum_impact_time = Timestamp.as_column(nullable=True)
    maximum_impact_time = Timestamp.as_column(nullable=True)


class CollisionConditions(qv.Table):
    condition_id = qv.LargeStringColumn()
    collision_object_name = qv.LargeStringColumn()
    collision_distance = qv.Float64Column()
    stopping_condition = qv.BooleanColumn()


class CollisionEvent(qv.Table):
    orbit_id = qv.LargeStringColumn()
    distance = qv.Float64Column()  # Distance from collision object in km
    coordinates = CartesianCoordinates.as_column()
    variant_id = qv.LargeStringColumn(nullable=True)
    condition_id = qv.LargeStringColumn()
    collision_object_name = qv.LargeStringColumn()
    collision_distance = qv.Float64Column()
    stopping_condition = qv.BooleanColumn()


class ImpactMixin:
    """
    `~adam_core.propagator.Propagator` mixin with signature for detecting Earth impacts.
    Subclasses should implement the _detect_collisions method.
    """


    @abstractmethod
    def _detect_collisions(
        self,
        orbits: Orbits,
        num_days: float,
        conditions: Optional[CollisionConditions] = None,
    ) -> Tuple[OrbitType, CollisionEvent]:
        """
        Detect collisions for the given orbits.

        THIS FUNCTION SHOULD NOT BE OVERRIDDEN BY THE USER.
        """
        pass

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
        if max_processes is None or max_processes > 1:
            impact_list: List[CollisionConditions] = []
            propagated_list: List[OrbitType] = []

            if RAY_INSTALLED is False:
                raise ImportError(
                    "Ray must be installed to use the ray parallel backend"
                )

            initialize_use_ray(num_cpus=max_processes)

            # Add orbits to object store if
            # they haven't already been added
            if not isinstance(orbits, ObjectRef):
                orbits_ref = ray.put(orbits)
            else:
                orbits_ref = orbits
                # We need to dereference the orbits ObjectRef so we can
                # check its length for chunking and determine
                # if we need to propagate variants
                orbits = ray.get(orbits_ref)

            # Create futures
            futures = []
            idx = np.arange(0, len(orbits))
            for idx_chunk in _iterate_chunks(idx, chunk_size):
                futures.append(
                    impact_worker_ray.remote(
                        idx_chunk, orbits_ref, self.__class__, num_days
                    )
                )

            # Get results as they finish (we sort later)
            unfinished = futures
            while unfinished:
                finished, unfinished = ray.wait(unfinished, num_returns=1)
                (propagated, impacts) = ray.get(finished[0])
                propagated_list.append(propagated)
                impact_list.append(impacts)

            propagated = qv.concatenate(propagated_list)
            impacts = qv.concatenate(impact_list)

        else:
            propagated, impacts = self._detect_collisions(
                orbits, num_days, conditions=conditions
            )

        return propagated, impacts


def calculate_impacts(
    orbits: Orbits,
    num_days: int,
    propagator: Propagator,
    num_samples: int = 1000,
    processes: Optional[int] = None,
    seed: Optional[int] = None,
    impact_conditions: Optional[CollisionConditions] = None,
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
    if impact_conditions is None:
        impact_conditions = CollisionConditions.from_kwargs(
            condition_id=["Default"],
            collision_object_name=["Earth"],
            collision_distance=[EARTH_RADIUS_KM],
            stopping_condition=[True],
        )
    logger.info("Detecting impacts...")
    results, collisions = propagator.detect_collisions(
        variants,
        num_days,
        impact_conditions,
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
        conditions = CollisionConditions.from_kwargs(
            condition_id=["Default"],
            collision_object_name=["Earth"],
            collision_distance=[EARTH_RADIUS_KM],
            stopping_condition=[True],
        )

    # Loop through the unique set of orbit_ids within variants using quivr
    unique_orbits = pc.unique(variants.orbit_id).to_pylist()

    impact_probabilities = None

    for orbit_id in unique_orbits:
        variant_masked = variants.select("orbit_id", orbit_id)
        variant_count = len(variant_masked)

        impacts_masked = collision_events.select("orbit_id", orbit_id)

        for unique_condition in conditions:
            impacts_per_condition = impacts_masked.select(
                "condition_id", unique_condition.condition_id[0]
            )
            impact_count = len(impacts_per_condition)

            ip = ImpactProbabilities.from_kwargs(
                condition_id=[unique_condition.condition_id[0]],
                orbit_id=[orbit_id],
                impacts=[impact_count],
                variants=[variant_count],
                cumulative_probability=[impact_count / variant_count],
            )

            if impact_probabilities is None:
                impact_probabilities = ip
            else:
                impact_probabilities = qv.concatenate([impact_probabilities, ip])

    return impact_probabilities


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
    mahalanobis_distance = np.sqrt(residuals.chi2.to_numpy(zero_copy_only=False))
    return mahalanobis_distance
