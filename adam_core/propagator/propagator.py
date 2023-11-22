import concurrent.futures
import logging
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

import numpy as np
import numpy.typing as npt
import quivr as qv

from adam_core.ray_cluster import initialize_use_ray

from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..orbits.variants import VariantEphemeris, VariantOrbits
from ..time import Timestamp
from .utils import _iterate_chunks

logger = logging.getLogger(__name__)

RAY_INSTALLED = False
try:
    import ray
    from ray import ObjectRef

    RAY_INSTALLED = True
except ImportError:
    pass

TimestampType = Union[Timestamp, ObjectRef]
OrbitType = Union[Orbits, VariantOrbits, ObjectRef]
EphemerisType = Union[Ephemeris, VariantOrbits, ObjectRef]
ObserverType = Union[Observers, ObjectRef]


def propagation_worker(
    orbits: Union[Orbits, VariantOrbits],
    times: Timestamp,
    propagator: Type["Propagator"],
    **kwargs,
) -> Union[Orbits, VariantOrbits]:
    prop = propagator(**kwargs)
    propagated = prop._propagate_orbits(orbits, times)
    return propagated


def ephemeris_worker(
    orbits: Union[Orbits, VariantOrbits],
    observers: Observers,
    propagator: Type["Propagator"],
    **kwargs,
) -> Union[Ephemeris, VariantOrbits]:
    prop = propagator(**kwargs)
    ephemeris = prop._generate_ephemeris(orbits, observers)
    return ephemeris


if RAY_INSTALLED:

    @ray.remote
    def propagation_worker_ray(
        idx: npt.NDArray[np.int64],
        orbits: OrbitType,
        times: OrbitType,
        propagator: Type["Propagator"],
        **kwargs,
    ) -> OrbitType:
        prop = propagator(**kwargs)
        orbits_chunk = orbits.take(idx)
        propagated = prop._propagate_orbits(orbits_chunk, times)
        return propagated

    @ray.remote
    def ephemeris_worker_ray(
        idx: npt.NDArray[np.int64],
        orbits: OrbitType,
        observers: ObserverType,
        propagator: Type["Propagator"],
        **kwargs,
    ) -> EphemerisType:
        prop = propagator(**kwargs)
        orbits_chunk = orbits.take(idx)
        ephemeris = prop._generate_ephemeris(orbits_chunk, observers)
        return ephemeris


class Propagator(ABC):
    """
    Class for propagating orbits and generating ephemerides. Subclasses
    should implement the _propagate_orbits and _generate_ephemeris methods.

    Important: subclasses should be pickleable! As this class uses multiprocessing
    to parallelize propagation and ephemeris generation. This means that
    subclasses should not have any unpickleable attributes.


    """

    @abstractmethod
    def _propagate_orbits(self, orbits: OrbitType, times: TimestampType) -> OrbitType:
        """
        Propagate orbits to times.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        """
        pass

    def propagate_orbits(
        self,
        orbits: OrbitType,
        times: TimestampType,
        covariance: bool = False,
        covariance_method: Literal[
            "auto", "sigma-point", "monte-carlo"
        ] = "monte-carlo",
        num_samples: int = 1000,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
        parallel_backend: Literal["cf", "ray"] = "ray",
    ) -> Orbits:
        """
        Propagate each orbit in orbits to each time in times.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            Orbits to propagate.
        times : Timestamp (M)
            Times to which to propagate orbits.
        covariance : bool, optional
            Propagate the covariance matrices of the orbits. This is done by sampling the
            orbits from their covariance matrices and propagating each sample. The covariance
            of the propagated orbits is then the covariance of the samples.
        covariance_method : {'sigma-point', 'monte-carlo', 'auto'}, optional
            The method to use for sampling the covariance matrix. If 'auto' is selected then the method
            will be automatically selected based on the covariance matrix. The default is 'monte-carlo'.
        num_samples : int, optional
            The number of samples to draw when sampling with monte-carlo.
        chunk_size : int, optional
            Number of orbits to send to each job.
        max_processes : int or None, optional
            Maximum number of processes to launch. If None then the number of
            processes will be equal to the number of cores on the machine. If 1
            then no multiprocessing will be used. If "ray" is the parallel_backend and a ray instance
            is initialized already then this argument is ignored.
        parallel_backend : {'cf', 'ray'}, optional
            The parallel backend to use. 'cf' uses concurrent.futures and 'ray' uses ray. The default is 'cf'.
            To use ray, ray must be installed.

        Returns
        -------
        propagated : `~adam_core.orbits.orbits.Orbits`
            Propagated orbits.
        """
        if max_processes is None or max_processes > 1:
            propagated_list: List[Orbits] = []
            variants_list: List[VariantOrbits] = []

            if parallel_backend == "cf":
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_processes
                ) as executor:
                    # Add orbits to propagate to futures
                    futures = []
                    for orbit_chunk in _iterate_chunks(orbits, chunk_size):
                        futures.append(
                            executor.submit(
                                propagation_worker,
                                orbit_chunk,
                                times,
                                self.__class__,
                                **self.__dict__,
                            )
                        )

                    # Add variants to propagate to futures
                    if (
                        covariance is True
                        and not orbits.coordinates.covariance.is_all_nan()
                    ):
                        variants = VariantOrbits.create(
                            orbits, method=covariance_method, num_samples=num_samples
                        )
                        for variant_chunk in _iterate_chunks(variants, chunk_size):
                            futures.append(
                                executor.submit(
                                    propagation_worker,
                                    variant_chunk,
                                    times,
                                    self.__class__,
                                    **self.__dict__,
                                )
                            )

                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if isinstance(result, Orbits):
                            propagated_list.append(result)
                        elif isinstance(result, VariantOrbits):
                            variants_list.append(result)
                        else:
                            raise ValueError(
                                f"Unexpected result type from propagation worker: {type(result)}"
                            )

            elif parallel_backend == "ray":
                if RAY_INSTALLED is False:
                    raise ImportError(
                        "Ray must be installed to use the ray parallel backend"
                    )

                initialize_use_ray(num_cpus=max_processes)

                # Add orbits and times to object store if
                # they haven't already been added
                if not isinstance(times, ObjectRef):
                    times_ref = ray.put(times)
                else:
                    times_ref = times

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
                        propagation_worker_ray.remote(
                            idx_chunk,
                            orbits_ref,
                            times_ref,
                            self.__class__,
                            **self.__dict__,
                        )
                    )

                # Add variants to propagate to futures
                if (
                    covariance is True
                    and not orbits.coordinates.covariance.is_all_nan()
                ):
                    variants = VariantOrbits.create(
                        orbits, method=covariance_method, num_samples=num_samples
                    )
                    variants_ref = ray.put(variants)

                    idx = np.arange(0, len(variants))
                    for variant_chunk in _iterate_chunks(idx, chunk_size):
                        idx_chunk = ray.put(variant_chunk)
                        futures.append(
                            propagation_worker_ray.remote(
                                idx_chunk,
                                variants_ref,
                                times_ref,
                                self.__class__,
                                **self.__dict__,
                            )
                        )

                # Get results as they finish (we sort later)
                unfinished = futures
                while unfinished:
                    finished, unfinished = ray.wait(unfinished, num_returns=1)
                    result = ray.get(finished[0])
                    if isinstance(result, Orbits):
                        propagated_list.append(result)
                    elif isinstance(result, VariantOrbits):
                        variants_list.append(result)
                    else:
                        raise ValueError(
                            f"Unexpected result type from propagation worker: {type(result)}"
                        )

            else:
                raise ValueError(f"Unknown parallel backend: {parallel_backend}")

            # Concatenate propagated orbits
            propagated = qv.concatenate(propagated_list)
            if len(variants_list) > 0:
                propagated_variants = qv.concatenate(variants_list)
            else:
                propagated_variants = None

        else:
            propagated = self._propagate_orbits(orbits, times)

            if covariance is True and not orbits.coordinates.covariance.is_all_nan():
                variants = VariantOrbits.create(
                    orbits, method=covariance_method, num_samples=num_samples
                )
                propagated_variants = self._propagate_orbits(variants, times)
            else:
                propagated_variants = None

        if propagated_variants is not None:
            propagated = propagated_variants.collapse(propagated)

        return propagated.sort_by(
            ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"]
        )

    @abstractmethod
    def _generate_ephemeris(
        self, orbits: EphemerisType, observers: ObserverType
    ) -> EphemerisType:
        """
        Generate ephemerides for the given orbits as observed by
        the observers.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        """
        pass

    def generate_ephemeris(
        self,
        orbits: OrbitType,
        observers: ObserverType,
        covariance: bool = False,
        covariance_method: Literal[
            "auto", "sigma-point", "monte-carlo"
        ] = "monte-carlo",
        num_samples: int = 1000,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
        parallel_backend: Literal["cf", "ray"] = "cf",
    ) -> Ephemeris:
        """
        Generate ephemerides for each orbit in orbits as observed by each observer
        in observers.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            Orbits for which to generate ephemerides.
        observers : `~adam_core.observers.observers.Observers` (M)
            Observers for which to generate the ephemerides of each
            orbit.
        covariance: bool, optional
            Propagate the covariance matrices of the orbits. This is done by sampling the
            orbits from their covariance matrices and propagating each sample and for each
            sample also generating ephemerides. The covariance
            of the ephemerides is then the covariance of the samples.
        covariance_method : {'sigma-point', 'monte-carlo', 'auto'}, optional
            The method to use for sampling the covariance matrix. If 'auto' is selected then the method
            will be automatically selected based on the covariance matrix. The default is 'monte-carlo'.
        num_samples : int, optional
            The number of samples to draw when sampling with monte-carlo.
        chunk_size : int, optional
            Number of orbits to send to each job.
        max_processes : int or None, optional
            Number of processes to launch. If None then the number of
            processes will be equal to the number of cores on the machine. If 1
            then no multiprocessing will be used. If "ray" is the parallel_backend and a ray instance
            is initialized already then this argument is ignored.
        parallel_backend : {'cf', 'ray'}, optional
            The parallel backend to use. 'cf' uses concurrent.futures and 'ray' uses ray. The default is 'cf'.
            To use ray, ray must be installed.

        Returns
        -------
        ephemeris : `~adam_core.orbits.ephemeris.Ephemeris` (M)
            Predicted ephemerides for each orbit observed by each
            observer.
        """
        # Check if we need to propagate orbit variants so we can propagate covariance
        # matrices

        if max_processes is None or max_processes > 1:
            ephemeris_list: List[Ephemeris] = []
            variants_list: List[VariantEphemeris] = []

            if parallel_backend == "cf":
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_processes
                ) as executor:
                    # Add orbits to propagate to futures
                    futures = []
                    for orbit_chunk in _iterate_chunks(orbits, chunk_size):
                        futures.append(
                            executor.submit(
                                ephemeris_worker,
                                orbit_chunk,
                                observers,
                                self.__class__,
                                **self.__dict__,
                            )
                        )

                    # Add variants to propagate to futures
                    if (
                        covariance is True
                        and not orbits.coordinates.covariance.is_all_nan()
                    ):
                        variants = VariantOrbits.create(
                            orbits, method=covariance_method, num_samples=num_samples
                        )
                        for variant_chunk in _iterate_chunks(variants, chunk_size):
                            futures.append(
                                executor.submit(
                                    ephemeris_worker,
                                    variant_chunk,
                                    observers,
                                    self.__class__,
                                    **self.__dict__,
                                )
                            )

                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if isinstance(result, Ephemeris):
                            ephemeris_list.append(result)
                        elif isinstance(result, VariantEphemeris):
                            variants_list.append(result)
                        else:
                            raise ValueError(
                                f"Unexpected result type from ephemeris worker: {type(result)}"
                            )
            elif parallel_backend == "ray":
                if RAY_INSTALLED is False:
                    raise ImportError(
                        "Ray must be installed to use the ray parallel backend"
                    )

                initialize_use_ray(num_cpus=max_processes)

                # Add orbits and observers to object store if
                # they haven't already been added
                if not isinstance(observers, ObjectRef):
                    observers_ref = ray.put(observers)
                else:
                    observers_ref = observers

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
                        ephemeris_worker_ray.remote(
                            idx_chunk,
                            orbits_ref,
                            observers_ref,
                            self.__class__,
                            **self.__dict__,
                        )
                    )

                # Add variants to futures (if we have any)
                if (
                    covariance is True
                    and not orbits.coordinates.covariance.is_all_nan()
                ):
                    variants = VariantOrbits.create(
                        orbits, method=covariance_method, num_samples=num_samples
                    )

                    # Add variants to object store
                    variants_ref = ray.put(variants)

                    idx = np.arange(0, len(variants))
                    for variant_chunk in _iterate_chunks(idx, chunk_size):
                        idx_chunk = ray.put(variant_chunk)
                        futures.append(
                            ephemeris_worker_ray.remote(
                                idx_chunk,
                                variants_ref,
                                observers_ref,
                                self.__class__,
                                **self.__dict__,
                            )
                        )

                # Get results as they finish (we sort later)
                unfinished = futures
                while unfinished:
                    finished, unfinished = ray.wait(unfinished, num_returns=1)
                    result = ray.get(finished[0])
                    if isinstance(result, Ephemeris):
                        ephemeris_list.append(result)
                    elif isinstance(result, VariantEphemeris):
                        variants_list.append(result)
                    else:
                        raise ValueError(
                            f"Unexpected result type from ephemeris worker: {type(result)}"
                        )

            else:
                raise ValueError(f"Unknown parallel backend: {parallel_backend}")

            ephemeris = qv.concatenate(ephemeris_list)
            if len(variants_list) > 0:
                ephemeris_variants = qv.concatenate(variants_list)
            else:
                ephemeris_variants = None

        else:
            ephemeris = self._generate_ephemeris(orbits, observers)

            if covariance is True and not orbits.coordinates.covariance.is_all_nan():
                variants = VariantOrbits.create(
                    orbits, method=covariance_method, num_samples=num_samples
                )
                ephemeris_variants = self._generate_ephemeris(variants, observers)
            else:
                ephemeris_variants = None

        if ephemeris_variants is not None:
            ephemeris = ephemeris_variants.collapse(ephemeris)

        return ephemeris.sort_by(
            [
                "orbit_id",
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )
