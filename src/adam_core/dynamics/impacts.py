import logging
from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

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
        variants, impacts = prop._detect_impacts(orbits_chunk, num_days)
        return variants, impacts


class EarthImpacts(qv.Table):
    orbit_id = qv.StringColumn()
    # Distance from earth center in km
    distance = qv.Float64Column()
    coordinates = CartesianCoordinates.as_column()
    variant_id = qv.LargeStringColumn(nullable=True)


class ImpactProbabilities(qv.Table):
    orbit_id = qv.LargeStringColumn()
    impacts = qv.Int64Column()
    variants = qv.Int64Column()
    cumulative_probability = qv.Float64Column()
    mean_impact_time = Timestamp.as_column(nullable=True)
    stddev_impact_time = qv.Float64Column(nullable=True)
    minimum_impact_time = Timestamp.as_column(nullable=True)
    maximum_impact_time = Timestamp.as_column(nullable=True)


class ImpactMixin:
    """
    `~adam_core.propagator.Propagator` mixin with signature for detecting Earth impacts.
    Subclasses should implement the _detect_impacts method.
    """

    @abstractmethod
    def _detect_impacts(
        self, orbits: Orbits, num_days: float
    ) -> Tuple[OrbitType, EarthImpacts]:
        """
        Detect impacts for the given orbits.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        """
        pass

    def detect_impacts(
        self,
        orbits: OrbitType,
        num_days: int,
        max_processes: Optional[int] = 1,
        chunk_size: int = 100,
    ) -> Tuple[OrbitType, EarthImpacts]:
        """
        Detect impacts for each orbit in orbits after num_days.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            Orbits for which to detect impacts.
        num_days : int
            Number of days after which to detect impacts.
        max_processes : int or None, optional
            Maximum number of processes to launch. If None then the number of
            processes will be equal to the number of cores on the machine. If 1
            then no multiprocessing will be used.

        Returns
        -------
        propagated : `~adam_core.orbits.OrbitType`
            The input orbits propagated to the end of simulation.
        impacts : `~adam_core.orbits.earth_impacts.EarthImpacts`
            Impacts detected for the orbits.
        """
        if max_processes is None or max_processes > 1:
            impact_list: List[EarthImpacts] = []
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
            propagated, impacts = self._detect_impacts(orbits, num_days)

        return propagated, impacts


def calculate_impacts(
    orbits: Orbits,
    num_days: int,
    propagator: Propagator,
    num_samples: int = 1000,
    processes: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[Orbits, EarthImpacts]:
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
    logger.info("Detecting impacts...")
    results, impacts = propagator.detect_impacts(
        variants,
        num_days,
        max_processes=processes,
    )

    return results, impacts


def calculate_impact_probabilities(
    variants: VariantOrbits, impacts: EarthImpacts
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

    # Loop through the unique set of orbit_ids within variants using quivr
    unique_orbits = pc.unique(variants.orbit_id).to_pylist()

    earth_impact_probabilities = ImpactProbabilities.empty()

    for orbit_id in unique_orbits:
        # mask = pc.equal(variants.orbit_id, orbit_id)
        variant_masked = variants.select("orbit_id", orbit_id)
        variant_count = len(variant_masked)

        impacts_masked = impacts.select("orbit_id", orbit_id)
        impact_count = len(impacts_masked)

        if len(impacts_masked) > 0:
            impact_mjds = impacts_masked.coordinates.time.mjd().to_numpy(
                zero_copy_only=False
            )
            mean_mjd = Timestamp.from_mjd(
                [np.mean(impact_mjds)], scale=impacts_masked.coordinates.time.scale
            )
            stddev = np.std(impact_mjds)
            min_mjd = impacts_masked.coordinates.time.min()
            max_mjd = impacts_masked.coordinates.time.max()
        else:
            mean_mjd = Timestamp.nulls(1, scale=impacts_masked.coordinates.time.scale)
            stddev = None
            min_mjd = Timestamp.nulls(1, scale=impacts_masked.coordinates.time.scale)
            max_mjd = Timestamp.nulls(1, scale=impacts_masked.coordinates.time.scale)

        ip = ImpactProbabilities.from_kwargs(
            orbit_id=[orbit_id],
            impacts=[impact_count],
            variants=[variant_count],
            cumulative_probability=[impact_count / variant_count],
            mean_impact_time=mean_mjd,
            stddev_impact_time=[stddev],
            minimum_impact_time=min_mjd,
            maximum_impact_time=max_mjd,
        )

        earth_impact_probabilities = qv.concatenate([earth_impact_probabilities, ip])

    return earth_impact_probabilities


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
