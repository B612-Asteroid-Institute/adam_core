import logging
import concurrent.futures
from abc import ABC, abstractmethod
from itertools import repeat
from typing import Optional

from astropy.time import Time
from quivr.concat import concatenate

from ..orbits import Orbits
from .utils import _iterate_chunks, sort_propagated_orbits

logger = logging.getLogger(__name__)


def propagation_worker(orbits: Orbits, times: Time, propagator: "Propagator") -> Orbits:
    propagated = propagator._propagate_orbits(orbits, times)
    return propagated


def ephemeris_worker(orbits: Orbits, observers, propagator: "Propagator"):
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
    def _propagate_orbits(self, orbits: Orbits, times: Time) -> Orbits:
        """
        Propagate orbits to times.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        """
        pass

    def propagate_orbits(
        self,
        orbits: Orbits,
        times: Time,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
    ) -> Orbits:
        """
        Propagate each orbit in orbits to each time in times.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            Orbits to propagate.
        times : `~astropy.time.core.Time` (M)
            Times to which to propagate orbits.
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
        if max_processes is None or max_processes > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
                futures = []
                for orbit_chunk in _iterate_chunks(orbits, chunk_size):
                    futures.append(executor.submit(propagation_worker, orbit_chunk, times, self))

                propagated_list = []
                for future in concurrent.futures.as_completed(futures):
                    propagated_list.append(future.result())

            propagated = concatenate(propagated_list)
        else:
            propagated = self._propagate_orbits(orbits, times)

        propagated = sort_propagated_orbits(propagated)

        return propagated

    @abstractmethod
    def _generate_ephemeris(self, orbits: Orbits, observers):
        """
        Generate ephemerides for the given orbits as observed by
        the observers.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        """
        pass

    def generate_ephemeris(
        self,
        orbits: Orbits,
        observers,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
    ):
        """
        Generate ephemerides for each orbit in orbits as observed by each observer
        in observers.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            Orbits for which to generate ephemerides.
        observers : (M)
            Observers for which to generate the ephemerides of each
            orbit.
        chunk_size : int, optional
            Number of orbits to send to each job.
        max_processes : int or None, optional
            Number of processes to launch. If None then the number of
            processes will be equal to the number of cores on the machine. If 1
            then no multiprocessing will be used.

        Returns
        -------
        ephemeris : (N * M)
            Predicted ephemerides for each orbit observed by each
            observer.

        TODO: Add ephemeris class
        TODO: Add an observers class
        """
        if max_processes is None or max_processes > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
                futures = []
                for orbit_chunk in _iterate_chunks(orbits, chunk_size):
                    futures.append(executor.submit(ephemeris_worker, orbit_chunk, observers, self))

                ephemeris_list = []
                for future in concurrent.futures.as_completed(futures):
                    ephemeris_list.append(future.result())
            ephemeris = concatenate(ephemeris_list)

        else:
            ephemeris = self._generate_ephemeris(orbits, observers)

        ephemeris.sort_values(
            by=["orbit_ids", "origin", "times"],
            inplace=True,
        )
        return ephemeris
