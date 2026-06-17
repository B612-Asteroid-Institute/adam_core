import logging
import uuid
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Tuple

import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.cometary import CometaryCoordinates
from ..coordinates.keplerian import KeplerianCoordinates
from ..coordinates.origin import OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import transform_coordinates
from .classification import calc_orbit_class
from .non_gravitational_parameters import NonGravitationalParameters
from .physical_parameters import PhysicalParameters
from .solved_state_covariances import SolvedStateCovariances

if TYPE_CHECKING:
    from ..propagator import Propagator

logger = logging.getLogger(__name__)


class Orbits(qv.Table):

    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    physical_parameters = PhysicalParameters.as_column(nullable=True)
    non_gravitational_parameters = NonGravitationalParameters.as_column(nullable=True)
    solved_state_covariance = SolvedStateCovariances.as_column(nullable=True)

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

    def has_non_gravitational_parameters(self) -> bool:
        """
        Return True if any orbit carries a non-zero non-gravitational parameter value.

        Parameters that are explicitly solved to zero are treated as absent: they
        exert no force, so a gravity-only propagation of such an orbit is still exact.
        """
        return self.non_gravitational_parameters.has_values()

    def without_non_gravitational_parameters(self) -> "Orbits":
        return self.set_column(
            "non_gravitational_parameters",
            NonGravitationalParameters.nulls(len(self)),
        ).set_column(
            "solved_state_covariance",
            SolvedStateCovariances.nulls(len(self)),
        )

    def with_non_gravitational_parameters(self, enabled: bool = True) -> "Orbits":
        """
        Return self unchanged if `enabled` is True, otherwise a copy with the
        non-gravitational parameter and solved-state covariance columns nulled.
        """
        if enabled:
            return self
        return self.without_non_gravitational_parameters()

    def coordinates_to(
        self,
        representation_out: type[
            CartesianCoordinates
            | KeplerianCoordinates
            | CometaryCoordinates
            | SphericalCoordinates
        ],
        *,
        frame_out: Optional[Literal["ecliptic", "equatorial", "itrf93"]] = None,
        origin_out: Optional[OriginCodes] = None,
    ):
        """
        Transform this orbit's coordinates to another representation, frame,
        and/or origin. The solved-state covariances are not transformed; use
        `solved_state_covariance_to` for that.
        """
        return transform_coordinates(
            self.coordinates,
            representation_out=representation_out,
            frame_out=frame_out,
            origin_out=origin_out,
        )

    def solved_state_covariance_to(
        self,
        representation_out: type[
            CartesianCoordinates
            | KeplerianCoordinates
            | CometaryCoordinates
            | SphericalCoordinates
        ],
        *,
        frame_out: Optional[Literal["ecliptic", "equatorial", "itrf93"]] = None,
        origin_out: Optional[OriginCodes] = None,
    ) -> SolvedStateCovariances:
        """
        Transform the solved-state covariances to another coordinate
        representation, frame, and/or origin. The leading 6x6 orbital block is
        transformed alongside the coordinates; extra solved parameters and
        their cross-covariances are preserved. Rows without a solved-state
        covariance remain null.
        """
        _, solved = transform_coordinates(
            self.coordinates,
            representation_out=representation_out,
            frame_out=frame_out,
            origin_out=origin_out,
            solved_state_covariances=self.solved_state_covariance,
        )
        return solved

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
