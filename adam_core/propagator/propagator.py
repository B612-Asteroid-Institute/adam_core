import logging
import multiprocessing as mp
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
        num_jobs: Optional[int] = 1,
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
        num_jobs : int or None, optional
            Number of jobs to launch. If None then the number of
            jobs will be equal to the number of cores on the machine. If 1
            then no multiprocessing will be used.

        Returns
        -------
        propagated : `~adam_core.orbits.orbits.Orbits`
            Propagated orbits.
        """
        if num_jobs is None or num_jobs > 1:
            orbits_split = [
                orbit_chunk for orbit_chunk in _iterate_chunks(orbits, chunk_size)
            ]

            p = mp.Pool(
                processes=num_jobs,
            )

            propagated_list = p.starmap(
                propagation_worker,
                zip(
                    orbits_split,
                    repeat(times),
                    repeat(self),
                ),
            )
            p.close()
            p.join()

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
        num_jobs: Optional[int] = 1,
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
        num_jobs : int or None, optional
            Number of jobs to launch. If None then the number of
            jobs will be equal to the number of cores on the machine. If 1
            then no multiprocessing will be used.

        Returns
        -------
        ephemeris : (N * M)
            Predicted ephemerides for each orbit observed by each
            observer.

        TODO: Add ephemeris class
        TODO: Add an observers class
        """
        if num_jobs is None or num_jobs > 1:
            orbits_split = [
                orbit_chunk for orbit_chunk in _iterate_chunks(orbits, chunk_size)
            ]

            p = mp.Pool(
                processes=num_jobs,
            )

            ephemeris_list = p.starmap(
                ephemeris_worker,
                zip(
                    orbits_split,
                    repeat(observers),
                    repeat(self),
                ),
            )
            p.close()
            p.join()

            ephemeris = concatenate(ephemeris_list)

        else:
            ephemeris = self._generate_ephemeris(orbits, observers)

        ephemeris.sort_values(
            by=["orbit_ids", "origin", "times"],
            inplace=True,
        )
        return ephemeris
