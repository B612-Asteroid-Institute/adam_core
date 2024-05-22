import logging
import uuid
from typing import Iterable, Tuple

import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from .classification import calc_orbit_class

logger = logging.getLogger(__name__)


class Orbits(qv.Table):

    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()

    def group_by_orbit_id(self) -> Iterable[Tuple[str, "Orbits"]]:
        """
        Group orbits by orbit ID and yield them.

        Yields
        ------
        orbit_id : str
            Orbit ID.
        orbits : `~adam_core.orbits.orbits.Orbits`
            Orbits belonging to this orbit ID.
        """
        unique_orbit_ids = self.orbit_id.unique()
        for orbit_id in unique_orbit_ids:
            mask = pc.equal(self.orbit_id, orbit_id)
            yield orbit_id, self.apply_mask(mask)

    def dynamical_class(self) -> npt.NDArray[str]:
        """
        Compute dynamical classes of orbits. Currently
        limited to asteroid dynamical classes.

        Returns
        -------
        dynamical_classes : `~numpy.ndarray`
            Dynamical classes of orbits.
        """
        keplerian = self.coordinates.to_keplerian()
        return calc_orbit_class(keplerian)
