import logging
import uuid
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Tuple, TypeVar, cast

import numpy as np
import numpy.typing as npt
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.cometary import CometaryCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.keplerian import KeplerianCoordinates
from ..coordinates.origin import OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import transform_coordinates
from .non_gravitational_parameters import NonGravitationalParameters
from .physical_parameters import PhysicalParameters

if TYPE_CHECKING:
    from ..propagator import Propagator

logger = logging.getLogger(__name__)

CoordinateType = TypeVar(
    "CoordinateType",
    CartesianCoordinates,
    KeplerianCoordinates,
    CometaryCoordinates,
    SphericalCoordinates,
)


class Orbits(qv.Table):

    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    physical_parameters = PhysicalParameters.as_column(nullable=True)
    non_gravitational_parameters = NonGravitationalParameters.as_column(nullable=True)

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

    def has_non_gravitational_parameters(self) -> bool:
        """Return whether any orbit carries a non-zero A1/A2/A3 value."""
        return self.non_gravitational_parameters.has_values()

    def has_non_gravitational_solution(self) -> bool:
        """Return whether values or an extended covariance encode a solution."""
        return (
            self.has_non_gravitational_parameters()
            or self.coordinates.covariance.has_nongrav_block()
        )

    def without_non_gravitational_parameters(self) -> "Orbits":
        """Strip non-grav values and reduce extended covariance to its 6D block."""
        orbits = self.set_column(
            "non_gravitational_parameters",
            NonGravitationalParameters.nulls(len(self)),
        )
        if orbits.coordinates.covariance.has_nongrav_block():
            orbits = orbits.set_column(
                "coordinates.covariance",
                CoordinateCovariances.from_matrix(
                    orbits.coordinates.covariance.to_matrix()
                ),
            )
        return orbits

    def coordinates_to(
        self,
        representation_out: type[CoordinateType],
        *,
        frame_out: Optional[Literal["ecliptic", "equatorial", "itrf93"]] = None,
        origin_out: Optional[OriginCodes] = None,
    ) -> CoordinateType:
        """Transform coordinates while preserving the complete covariance."""
        return cast(
            CoordinateType,
            transform_coordinates(
                self.coordinates,
                representation_out=representation_out,
                frame_out=frame_out,
                origin_out=origin_out,
            ),
        )

    def to_keplerian(self) -> KeplerianCoordinates:
        return self.coordinates_to(KeplerianCoordinates)

    def to_cometary(self) -> CometaryCoordinates:
        return self.coordinates_to(CometaryCoordinates)

    def to_spherical(self) -> SphericalCoordinates:
        return self.coordinates_to(SphericalCoordinates)

    def preview(self, propagator: "Propagator") -> None:
        """
        For a single orbit, render a plotly plot of the orbit.
        """
        from .plots import plot_orbit

        fig = plot_orbit(self, propagator)
        fig.show()
