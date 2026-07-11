import logging
import uuid
from typing import TYPE_CHECKING, Iterable, Tuple

import numpy as np
import numpy.typing as npt
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from .physical_parameters import PhysicalParameters

if TYPE_CHECKING:
    from ..propagator import Propagator

logger = logging.getLogger(__name__)


class Orbits(qv.Table):

    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    physical_parameters = PhysicalParameters.as_column(nullable=True)

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
        from adam_core import _rust_native

        from .arrow_bridge import orbits_from_record_batch, orbits_to_record_batch

        grouped = _rust_native.group_by_orbit_id_arrow(orbits_to_record_batch(self))
        for orbit_id, batch in grouped:
            yield str(orbit_id), orbits_from_record_batch(batch)

    def dynamical_class(self) -> npt.NDArray[str]:
        """
        Compute dynamical classes of orbits. Currently
        limited to asteroid dynamical classes.

        Returns
        -------
        dynamical_classes : `~numpy.ndarray`
            Dynamical classes of orbits.
        """
        from adam_core import _rust_native

        from .arrow_bridge import orbits_to_record_batch

        classes = _rust_native.dynamical_class_arrow(orbits_to_record_batch(self))
        return np.asarray(classes, dtype=str)

    def preview(self, propagator: "Propagator") -> None:
        """
        For a single orbit, render a plotly plot of the orbit.
        """
        from .plots import plot_orbit

        fig = plot_orbit(self, propagator)
        fig.show()
