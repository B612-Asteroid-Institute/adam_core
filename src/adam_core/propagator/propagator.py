import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
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
from ..utils.iter import _iterate_chunks
from .types import EphemerisType, ObserverType, OrbitType, TimestampType
from .utils import ensure_input_origin_and_frame, ensure_input_time_scale

logger = logging.getLogger(__name__)

C = c.C

RAY_INSTALLED = False
try:
    import ray
    from ray import ObjectRef

    RAY_INSTALLED = True
except ImportError:
    pass


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
        propagator: "Propagator",
    ) -> OrbitType:
        orbits_chunk = orbits.take(idx)
        propagated = propagator._propagate_orbits(orbits_chunk, times)
        return propagated

    @ray.remote
    def ephemeris_worker_ray(
        idx: npt.NDArray[np.int64],
        orbits: OrbitType,
        observers: ObserverType,
        propagator: "Propagator",
    ) -> EphemerisType:
        orbits_chunk = orbits.take(idx)
        ephemeris = propagator._generate_ephemeris(orbits_chunk, observers)
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
        # If sending in VariantOrbits, we make sure not to run covariance
        assert (covariance is False) or (
            isinstance(orbits, Orbits)
        ), "Covariance is not supported for VariantOrbits"

        # Check if we need to propagate orbit variants so we can propagate covariance
        # matrices
        ephemeris: Ephemeris = Ephemeris.empty()
        variant_ephemeris: VariantEphemeris = VariantEphemeris.empty()

        variants = VariantOrbits.empty()
        if covariance is True and not orbits.coordinates.covariance.is_all_nan():
            variants = VariantOrbits.create(
                orbits,
                method=covariance_method,
                num_samples=num_samples,
                seed=seed,
            )

        if max_processes is None:
            max_processes = mp.cpu_count()

        if max_processes > 1:

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
            futures_inputs = []
            idx = np.arange(0, len(orbits))
            for idx_chunk in _iterate_chunks(idx, chunk_size):
                futures_inputs.append(
                    (
                        idx_chunk,
                        orbits_ref,
                        observers_ref,
                        self,
                    )
                )

            # Add variants to propagate to futures inputs
            if covariance is True and len(variants) > 0:
                variants_ref = ray.put(variants)

                idx = np.arange(0, len(variants))
                for variant_chunk_idx in _iterate_chunks(idx, chunk_size):
                    futures_inputs.append(
                        (
                            variant_chunk_idx,
                            variants_ref,
                            observers_ref,
                            self,
                        )
                    )

            # Get results as they finish (we sort later)
            futures = []
            for future_input in futures_inputs:
                futures.append(ephemeris_worker_ray.remote(*future_input))

                if len(futures) >= max_processes * 1.5:
                    finished, futures = ray.wait(futures, num_returns=1)
                    result = ray.get(finished[0])
                    if isinstance(result, Ephemeris):
                        ephemeris = qv.concatenate([ephemeris, result])
                    elif isinstance(result, VariantEphemeris):
                        variant_ephemeris = qv.concatenate([variant_ephemeris, result])
                    else:
                        raise ValueError(
                            f"Unexpected result type from ephemeris worker: {type(result)}"
                        )

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                if isinstance(result, Ephemeris):
                    ephemeris = qv.concatenate([ephemeris, result])
                elif isinstance(result, VariantEphemeris):
                    variant_ephemeris = qv.concatenate([variant_ephemeris, result])
                else:
                    raise ValueError(
                        f"Unexpected result type from ephemeris worker: {type(result)}"
                    )

        else:
            results = self._generate_ephemeris(orbits, observers)
            if isinstance(results, Ephemeris):
                ephemeris = results
            elif isinstance(results, VariantEphemeris):
                variant_ephemeris = results
            else:
                raise ValueError(
                    f"Unexpected result type from generate_ephemeris: {type(results)}"
                )
            if covariance is True and len(variants) > 0:
                variant_ephemeris = self._generate_ephemeris(variants, observers)

        if covariance is False and len(variant_ephemeris) > 0:
            # If we decide that we do not need to guarantee that the time scale is in UTC
            # then we may want to call:
            # if isinstance(observers, ray.ObjectRef):
            #     variant_ephemeris = ensure_input_time_scale(
            #         variant_ephemeris, ray.get(observers).coordinates.time
            #     )
            # else:
            #     variant_ephemeris = ensure_input_time_scale(
            #         variant_ephemeris, observers.coordinates.time
            #     )
            variant_ephemeris = variant_ephemeris.set_column(
                "coordinates.time",
                variant_ephemeris.coordinates.time.rescale("utc"),
            )

            # We were given VariantOrbits as an input, so return VariantEphemeris
            return variant_ephemeris.sort_by(
                [
                    "orbit_id",
                    "variant_id",
                    "coordinates.time.days",
                    "coordinates.time.nanos",
                    "coordinates.origin.code",
                ]
            )

        if covariance is True and len(variant_ephemeris) > 0:
            ephemeris = variant_ephemeris.collapse(ephemeris)

        # Same note as above.
        # if isinstance(observers, ray.ObjectRef):
        #     ephemeris = ensure_input_time_scale(
        #         ephemeris, ray.get(observers).coordinates.time
        #     )
        # else:
        #     ephemeris = ensure_input_time_scale(
        #         ephemeris, observers.coordinates.time
        #     )
        ephemeris = ephemeris.set_column(
            "coordinates.time",
            ephemeris.coordinates.time.rescale("utc"),
        )

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

    def __getstate__(self):
        """
        Get the state of the propagator.

        Subclasses need to define what is picklable for multiprocessing.

        e.g.

        def __getstate__(self):
            state = self.__dict__.copy()
            state.pop("_stateful_attribute_that_is_not_pickleable")
            return state
        """
        raise NotImplementedError(
            "Propagator must implement __getstate__ for multiprocessing serialization.\n"
            "Example implementation: \n"
            "def __getstate__(self):\n"
            "    state = self.__dict__.copy()\n"
            "    state.pop('_stateful_attribute_that_is_not_pickleable')\n"
            "    return state"
        )

    def __setstate__(self, state):
        """
        Set the state of the propagator.

        Subclasses need to define what is unpicklable for multiprocessing.

        e.g.

        def __setstate__(self, state):
            self.__dict__.update(state)
            self._stateful_attribute_that_is_not_pickleable = None
        """
        raise NotImplementedError(
            "Propagator must implement __setstate__ for multiprocessing serialization.\n"
            "Example implementation: \n"
            "def __setstate__(self, state):\n"
            "    self.__dict__.update(state)\n"
            "    self._stateful_attribute_that_is_not_pickleable = None"
        )

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
        if max_processes is None:
            max_processes = mp.cpu_count()

        if max_processes > 1:
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

            # Create futures inputs
            futures_inputs = []
            idx = np.arange(0, len(orbits))
            for idx_chunk in _iterate_chunks(idx, chunk_size):
                futures_inputs.append(
                    (
                        idx_chunk,
                        orbits_ref,
                        times_ref,
                        self,
                    )
                )

            # Add variants to propagate to futures inputs
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
                    futures_inputs.append(
                        (
                            variant_chunk_idx,
                            variants_ref,
                            times_ref,
                            self,
                        )
                    )

            # Submit and process jobs with queuing
            futures = []
            for future_input in futures_inputs:
                futures.append(propagation_worker_ray.remote(*future_input))

                if len(futures) >= max_processes * 1.5:
                    finished, futures = ray.wait(futures, num_returns=1)
                    result = ray.get(finished[0])
                    if isinstance(result, Orbits):
                        propagated_list.append(result)
                    elif isinstance(result, VariantOrbits):
                        variants_list.append(result)
                    else:
                        raise ValueError(
                            f"Unexpected result type from propagation worker: {type(result)}"
                        )

            # Process remaining futures
            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
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
        propagated = ensure_input_time_scale(propagated, times)

        # Return the results with the original origin and frame
        # Preserve the original output origin for the input orbits
        # by orbit id
        propagated = ensure_input_origin_and_frame(orbits, propagated)

        return propagated.sort_by(
            ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"]
        )
