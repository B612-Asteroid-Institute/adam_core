import logging
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

from ..constants import Constants as c
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import transform_coordinates
from ..dynamics import propagate_2body
from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..orbits.variants import VariantEphemeris, VariantOrbits
from ..ray_cluster import initialize_use_ray
from ..time import Timestamp
from .utils import _iterate_chunks

logger = logging.getLogger(__name__)

C = c.C

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


class EphemerisMixin:
    """
    Mixin with signature for generating ephemerides.
    Subclasses should implement the _generate_ephemeris method.
    """

    def _add_light_time(
        self,
        orbits,
        observers,
        lt_tol: float = 1e-12,
        max_iter: int = 10,
    ) -> Tuple[Orbits, np.ndarray]:
        orbits_aberrated = Orbits.empty()
        lts = np.zeros(len(orbits))
        for i, (orbit, observer) in enumerate(zip(orbits, observers)):
            # Set the running variables
            lt_prev = 0
            dlt = float("inf")
            orbit_i = orbit
            lt = 0

            # Extract the observer's position which remains
            # constant for all iterations
            observer_position = observer.coordinates.r

            # Calculate the orbit's current epoch (the epoch from which
            # the light travel time will be calculated)
            t0 = orbit_i.coordinates.time.rescale("tdb").mjd()[0].as_py()

            iterations = 0
            while dlt > lt_tol and iterations < max_iter:
                iterations += 1

                # Calculate the topocentric distance
                rho = np.linalg.norm(orbit_i.coordinates.r - observer_position)
                # If rho becomes too large, we are probably simulating a close encounter
                # and our propagation will break
                if np.isnan(rho) or rho > 1e12:
                    raise ValueError(
                        "Distance from observer is NaN or too large and propagation will break."
                    )

                # Calculate the light travel time
                lt = rho / C

                # Calculate the change in light travel time since the previous iteration
                dlt = np.abs(lt - lt_prev)

                if np.isnan(lt) or lt > 1e12:
                    raise ValueError(
                        "Light travel time is NaN or too large and propagation will break."
                    )

                # Calculate the new epoch and propagate the initial orbit to that epoch
                # Should be sufficient to use 2body propagation for this
                orbit_i = propagate_2body(
                    orbit, Timestamp.from_mjd([t0 - lt], scale="tdb")
                )

                # Update the previous light travel time to this iteration's light travel time
                lt_prev = lt

            orbits_aberrated = qv.concatenate([orbits_aberrated, orbit_i])
            lts[i] = lt

        return orbits_aberrated, lts

    def _generate_ephemeris(
        self, orbits: OrbitType, observers: ObserverType
    ) -> EphemerisType:
        """
        A generic ephemeris implementation, which can be used or overridden by subclasses.
        """

        if isinstance(orbits, Orbits):
            ephemeris_total = Ephemeris.empty()
        elif isinstance(orbits, VariantOrbits):
            ephemeris_total = VariantEphemeris.empty()

        # Sort observers by time and code to ensure consistent ordering
        # As further propagation will order by time as well
        observers = observers.sort_by(
            ["coordinates.time.days", "coordinates.time.nanos", "code"]
        )

        observers_barycentric = observers.set_column(
            "coordinates",
            transform_coordinates(
                observers.coordinates,
                CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
            ),
        )

        for orbit in orbits:
            # Propagate orbits to sorted observer times
            propagated_orbits = self.propagate_orbits(orbit, observers.coordinates.time)

            # Transform both the orbits and observers to the barycenter if they are not already.
            propagated_orbits_barycentric = propagated_orbits.set_column(
                "coordinates",
                transform_coordinates(
                    propagated_orbits.coordinates,
                    CartesianCoordinates,
                    frame_out="ecliptic",
                    origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
                ),
            )

            num_orbits = len(propagated_orbits_barycentric.orbit_id.unique())

            observer_codes = np.tile(
                observers.code.to_numpy(zero_copy_only=False), num_orbits
            )

            propagated_orbits_aberrated, light_time = self._add_light_time(
                propagated_orbits_barycentric,
                observers_barycentric,
                lt_tol=1e-12,
            )

            topocentric_state = (
                propagated_orbits_aberrated.coordinates.values
                - observers_barycentric.coordinates.values
            )
            topocentric_coordinates = CartesianCoordinates.from_kwargs(
                x=topocentric_state[:, 0],
                y=topocentric_state[:, 1],
                z=topocentric_state[:, 2],
                vx=topocentric_state[:, 3],
                vy=topocentric_state[:, 4],
                vz=topocentric_state[:, 5],
                covariance=None,
                # The ephemeris times are at the point of the observer,
                # not the aberrated orbit
                time=observers.coordinates.time,
                origin=Origin.from_kwargs(code=observer_codes),
                frame="ecliptic",
            )

            spherical_coordinates = SphericalCoordinates.from_cartesian(
                topocentric_coordinates
            )

            light_time = np.array(light_time)

            spherical_coordinates = transform_coordinates(
                spherical_coordinates, SphericalCoordinates, frame_out="equatorial"
            )

            # Ephemeris are generally compared in UTC, so rescale the time
            spherical_coordinates = spherical_coordinates.set_column(
                "time",
                spherical_coordinates.time.rescale("utc"),
            )

            if isinstance(orbit, Orbits):
                ephemeris = Ephemeris.from_kwargs(
                    orbit_id=propagated_orbits_barycentric.orbit_id,
                    object_id=propagated_orbits_barycentric.object_id,
                    coordinates=spherical_coordinates,
                    light_time=light_time,
                    aberrated_coordinates=propagated_orbits_aberrated.coordinates,
                )
            elif isinstance(orbit, VariantOrbits):
                ephemeris = VariantEphemeris.from_kwargs(
                    orbit_id=propagated_orbits_barycentric.orbit_id,
                    object_id=propagated_orbits_barycentric.object_id,
                    coordinates=spherical_coordinates,
                    weights=np.repeat(orbit.weights[0], len(observers)),
                    weights_cov=np.repeat(orbit.weights_cov[0], len(observers)),
                )

            ephemeris_total = qv.concatenate([ephemeris_total, ephemeris])

        ephemeris_total = ephemeris_total.sort_by(
            [
                "orbit_id",
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        return ephemeris_total

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
        seed: Optional[int] = None,
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
            if covariance is True and not orbits.coordinates.covariance.is_all_nan():
                variants = VariantOrbits.create(
                    orbits,
                    method=covariance_method,
                    num_samples=num_samples,
                    seed=seed,
                )

                # Add variants to object store
                variants_ref = ray.put(variants)

                idx = np.arange(0, len(variants))
                for variant_chunk_idx in _iterate_chunks(idx, chunk_size):
                    futures.append(
                        ephemeris_worker_ray.remote(
                            variant_chunk_idx,
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

            ephemeris = qv.concatenate(ephemeris_list)
            if len(variants_list) > 0:
                ephemeris_variants = qv.concatenate(variants_list)
            else:
                ephemeris_variants = None

        else:
            ephemeris = self._generate_ephemeris(orbits, observers)

            if covariance is True and not orbits.coordinates.covariance.is_all_nan():
                variants = VariantOrbits.create(
                    orbits,
                    method=covariance_method,
                    num_samples=num_samples,
                    seed=seed,
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


class Propagator(ABC, EphemerisMixin):
    """
    Abstract class for propagating orbits and related functions.

    Subclasses should implement the _propagate_orbits.
    For additional functions, subclasses can add abstract mixins.

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
        orbits: Union[OrbitType, ObjectRef],
        times: Union[TimestampType, ObjectRef],
        covariance: bool = False,
        covariance_method: Literal[
            "auto", "sigma-point", "monte-carlo"
        ] = "monte-carlo",
        num_samples: int = 1000,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> Orbits:
        """
        Propagate each orbit in orbits to each time in times.

        This method handles parallelization of the propagation of the orbits.
        Subclasses may override this method to modify parallelization behavior.

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

        Returns
        -------
        propagated : `~adam_core.orbits.orbits.Orbits`
            Propagated orbits.
        """

        if max_processes is None or max_processes > 1:
            propagated_list: List[Orbits] = []
            variants_list: List[VariantOrbits] = []

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
                times = ray.get(times_ref)

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
            if covariance is True and not orbits.coordinates.covariance.is_all_nan():
                variants = VariantOrbits.create(
                    orbits,
                    method=covariance_method,
                    num_samples=num_samples,
                    seed=seed,
                )

                variants_ref = ray.put(variants)

                idx = np.arange(0, len(variants))
                for variant_chunk_idx in _iterate_chunks(idx, chunk_size):
                    futures.append(
                        propagation_worker_ray.remote(
                            variant_chunk_idx,
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

            # Concatenate propagated orbits
            propagated = qv.concatenate(propagated_list)
            if len(variants_list) > 0:
                propagated_variants = qv.concatenate(variants_list)
                # sort by variant_id and time
                propagated_variants = propagated_variants.sort_by(
                    ["variant_id", "coordinates.time.days", "coordinates.time.nanos"]
                )
            else:
                propagated_variants = None

        else:
            propagated = self._propagate_orbits(orbits, times)

            if covariance is True and not orbits.coordinates.covariance.is_all_nan():
                variants = VariantOrbits.create(
                    orbits,
                    method=covariance_method,
                    num_samples=num_samples,
                    seed=seed,
                )

                propagated_variants = self._propagate_orbits(variants, times)
                # sort by variant_id and time
                propagated_variants = propagated_variants.sort_by(
                    ["variant_id", "coordinates.time.days", "coordinates.time.nanos"]
                )
            else:
                propagated_variants = None

        if propagated_variants is not None:
            propagated = propagated_variants.collapse(propagated)

        # Preserve the time scale of the requested times
        propagated = propagated.set_column(
            "coordinates.time",
            propagated.coordinates.time.rescale(times.scale),
        )

        # Return the results with the original origin and frame
        # Preserve the original output origin for the input orbits
        # by orbit id
        final_results = None
        unique_origins = pc.unique(orbits.coordinates.origin.code)
        for origin_code in unique_origins:
            origin_orbits = orbits.select("coordinates.origin.code", origin_code)
            result_origin_orbits = propagated.apply_mask(
                pc.is_in(propagated.orbit_id, origin_orbits.orbit_id)
            )
            partial_results = result_origin_orbits.set_column(
                "coordinates",
                transform_coordinates(
                    result_origin_orbits.coordinates,
                    origin_out=OriginCodes[origin_code.as_py()],
                    frame_out=orbits.coordinates.frame,
                ),
            )
            if final_results is None:
                final_results = partial_results
            else:
                final_results = qv.concatenate([final_results, partial_results])

        return final_results.sort_by(
            ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"]
        )
