import concurrent.futures
import logging
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import quivr as qv

from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..orbits.variants import VariantEphemeris, VariantOrbits
from ..time import Timestamp
from .utils import _iterate_chunks

logger = logging.getLogger(__name__)

OrbitType = Union[Orbits, VariantOrbits]
EphemerisType = Union[Ephemeris, VariantOrbits]


def propagation_worker(
    orbits: OrbitType, times: Timestamp, propagator: "Propagator"
) -> Orbits:
    propagated = propagator._propagate_orbits(orbits, times)
    return propagated


def ephemeris_worker(
    orbits: Orbits, observers: Observers, propagator: "Propagator"
) -> Ephemeris:
    ephemeris = propagator._generate_ephemeris(orbits, observers)
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
    def _propagate_orbits(self, orbits: OrbitType, times: Timestamp) -> OrbitType:
        """
        Propagate orbits to times.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        """
        pass

    def propagate_orbits(
        self,
        orbits: Orbits,
        times: Timestamp,
        covariance: bool = False,
        covariance_method: Literal[
            "auto", "sigma-point", "monte-carlo"
        ] = "monte-carlo",
        num_samples: int = 1000,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
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
            then no multiprocessing will be used.

        Returns
        -------
        propagated : `~adam_core.orbits.orbits.Orbits`
            Propagated orbits.
        """
        # Check if we need to propagate orbit variants so we can propagate covariance
        # matrices
        if not orbits.coordinates.covariance.is_all_nan() and covariance is True:
            variants = VariantOrbits.create(
                orbits, method=covariance_method, num_samples=num_samples
            )
        else:
            variants = None

        if max_processes is None or max_processes > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_processes
            ) as executor:

                # Add orbits to propagate to futures
                futures = []
                for orbit_chunk in _iterate_chunks(orbits, chunk_size):
                    futures.append(
                        executor.submit(propagation_worker, orbit_chunk, times, self)
                    )

                # Add variants to propagate to futures
                if variants is not None:
                    for variant_chunk in _iterate_chunks(variants, chunk_size):
                        futures.append(
                            executor.submit(
                                propagation_worker, variant_chunk, times, self
                            )
                        )

                propagated_list: List[Orbits] = []
                variants_list: List[VariantOrbits] = []
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

            propagated = qv.concatenate(propagated_list)
            if len(variants_list) > 0:
                propagated_variants = qv.concatenate(variants_list)
            else:
                propagated_variants = None

        else:
            propagated = self._propagate_orbits(orbits, times)

            if variants is not None:
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
        self, orbits: Orbits, observers: Observers
    ) -> EphemerisType:
        """
        Generate ephemerides for the given orbits as observed by
        the observers.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        """
        pass

    def generate_ephemeris(
        self,
        orbits: Orbits,
        observers: Observers,
        covariance: bool = False,
        covariance_method: Literal[
            "auto", "sigma-point", "monte-carlo"
        ] = "monte-carlo",
        num_samples: int = 1000,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
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
            then no multiprocessing will be used.

        Returns
        -------
        ephemeris : `~adam_core.orbits.ephemeris.Ephemeris` (M)
            Predicted ephemerides for each orbit observed by each
            observer.
        """
        # Check if we need to propagate orbit variants so we can propagate covariance
        # matrices
        if not orbits.coordinates.covariance.is_all_nan() and covariance is True:
            variants = VariantOrbits.create(
                orbits, method=covariance_method, num_samples=num_samples
            )
        else:
            variants = None

        if max_processes is None or max_processes > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_processes
            ) as executor:

                # Add orbits to propagate to futures
                futures = []
                for orbit_chunk in _iterate_chunks(orbits, chunk_size):
                    futures.append(
                        executor.submit(ephemeris_worker, orbit_chunk, observers, self)
                    )

                # Add variants to propagate to futures
                if variants is not None:
                    for variant_chunk in _iterate_chunks(variants, chunk_size):
                        futures.append(
                            executor.submit(
                                ephemeris_worker, variant_chunk, observers, self
                            )
                        )

                ephemeris_list: List[Ephemeris] = []
                variants_list: List[VariantEphemeris] = []
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

            ephemeris = qv.concatenate(ephemeris_list)
            if len(variants_list) > 0:
                ephemeris_variants = qv.concatenate(variants_list)
            else:
                ephemeris_variants = None

        else:
            ephemeris = self._generate_ephemeris(orbits, observers)

            if variants is not None:
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
