import copy
import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from astropy.time import Time

from ..orbits import Orbits
from ..utils.indexable import concatenate
from ..utils.multiprocessing import _check_parallel

logger = logging.getLogger(__name__)


def propagation_worker(orbits: Orbits, times: Time, backend: "Backend") -> Orbits:
    propagated = backend._propagate_orbits(orbits, times)
    return propagated


def ephemeris_worker(orbits: Orbits, observers, backend: "Backend"):
    ephemeris = backend._generate_ephemeris(orbits, observers)
    return ephemeris


def orbit_determination_worker(observations, backend: "Backend"):
    orbits = backend._orbit_determination(observations)
    return orbits


class Backend(ABC):
    def __init__(self, name: str = "Backend", **kwargs):
        self.__dict__.update(kwargs)
        self.name = name
        return

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
        num_jobs: int = 1,
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
        num_jobs : int or "auto", optional
            Number of jobs to launch. If "auto" then the number of
            jobs will be equal to the number of cores on the machine.

        Returns
        -------
        propagated : `~adam_core.orbits.orbits.Orbits`
            Propagated orbits.
        """
        parallel, num_workers = _check_parallel(num_jobs)
        if parallel:
            orbits_split = list(orbits.yield_chunks(chunk_size))
            times_duplicated = [copy.copy(times) for i in range(len(orbits_split))]
            backend_duplicated = [copy.copy(self) for i in range(len(orbits_split))]

            p = mp.Pool(
                processes=num_workers,
            )

            propagated_list = p.starmap(
                propagation_worker,
                zip(
                    orbits_split,
                    times_duplicated,
                    backend_duplicated,
                ),
            )
            p.join()
            p.close()

            propagated = concatenate(propagated_list)
        else:
            propagated = self._propagate_orbits(orbits, times)

        propagated.sort_values(
            by=["orbit_ids", "times"], ascending=[True, True], inplace=True
        )

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
        num_jobs: Union[str, int] = 1,
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
        num_jobs : int, optional
            Number of jobs to launch. If "auto" then the number of
            jobs will be equal to the number of cores on the machine.

        Returns
        -------
        ephemeris : (N * M)
            Predicted ephemerides for each orbit observed by each
            observer.
        """
        parallel, num_workers = _check_parallel(num_jobs)
        if parallel:
            orbits_split = list(orbits.yield_chunks(chunk_size))
            observers_duplicated = [
                copy.copy(observers) for i in range(len(orbits_split))
            ]
            backend_duplicated = [copy.copy(self) for i in range(len(orbits_split))]

            p = mp.Pool(
                processes=num_workers,
            )

            ephemeris_list = p.starmap(
                ephemeris_worker,
                zip(
                    orbits_split,
                    observers_duplicated,
                    backend_duplicated,
                ),
            )
            p.join()
            p.close()

            ephemeris = concatenate(ephemeris_list)

        else:
            ephemeris = self._generate_ephemeris(orbits, observers)

        ephemeris.sort_values(
            by=["orbit_ids", "origin", "times"],
            inplace=True,
        )
        return ephemeris

    @abstractmethod
    def _orbit_determination(self):
        """
        Run orbit determination on the input observations.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        """
        pass

    def orbit_determination(
        self, observations, chunk_size=10, num_jobs=1, parallel_backend="mp"
    ):
        """
        Run orbit determination on the input observations. These observations
        must at least contain the following columns:

        """
        unique_objs = observations["obj_id"].unique()
        observations_split = [
            observations[
                observations["obj_id"].isin(unique_objs[i : i + chunk_size])
            ].copy()
            for i in range(0, len(unique_objs), chunk_size)
        ]
        backend_duplicated = [copy.copy(self) for i in range(len(observations_split))]

        parallel, num_workers = _check_parallel(num_jobs, parallel_backend)
        if parallel:
            p = mp.Pool(
                processes=num_workers,
            )

            od_orbits_dfs = p.starmap(
                orbit_determination_worker,
                zip(
                    observations_split,
                    backend_duplicated,
                ),
            )
            p.join()
            p.close()

            od_orbits = pd.concat(od_orbits_dfs, ignore_index=True)

        else:
            od_orbits = self._orbit_determination(observations)

        return od_orbits
