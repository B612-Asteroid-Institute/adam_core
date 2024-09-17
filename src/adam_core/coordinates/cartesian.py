from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import quivr as qv

from ..time import Timestamp
from . import cometary, keplerian, spherical
from .covariances import CoordinateCovariances
from .origin import Origin

__all__ = ["CartesianCoordinates"]

COVARIANCE_ROTATION_TOLERANCE = 1e-25
logger = logging.getLogger(__name__)


class CartesianCoordinates(qv.Table):
    """Represents coordinates in Cartesian space."""

    #: x coordinates in AU
    x = qv.Float64Column(nullable=True)

    #: y coordinates in AU
    y = qv.Float64Column(nullable=True)

    #: z coordinates in AU
    z = qv.Float64Column(nullable=True)

    #: x velocity in AU/day
    vx = qv.Float64Column(nullable=True)

    #: y velocity in AU/day
    vy = qv.Float64Column(nullable=True)

    #: z velocity in AU/day
    vz = qv.Float64Column(nullable=True)

    #: The instant at which the coordinates are valid
    time = Timestamp.as_column(nullable=True)

    #: Covariance matrix for the x, y, z, vx, vy, and vz values
    covariance = CoordinateCovariances.as_column(nullable=True)

    #: Center of the coordinate system
    origin = Origin.as_column()

    #: Frame of the coordinate system - 'ecliptic' or 'equatorial' or 'unspecified'.
    frame = qv.StringAttribute(default="unspecified")

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """The x, y, z, vx, vy, and vz columns all in one 6-N matrix
        of numpy float64 values. Nulls are converted to NaNs.

        """
        return np.array(self.table.select(["x", "y", "z", "vx", "vy", "vz"]))

    @property
    def r(self) -> npt.NDArray[np.float64]:
        """
        Position vector.
        """
        return np.array(self.table.select(["x", "y", "z"]))

    @property
    def r_mag(self) -> npt.NDArray[np.float64]:
        """
        Magnitude of the position vector.
        """
        return np.linalg.norm(self.r, axis=1)

    @property
    def r_hat(self) -> npt.NDArray[np.float64]:
        """
        Unit vector in the direction of the position vector.
        """
        return self.r / self.r_mag[:, None]

    @property
    def v(self) -> npt.NDArray[np.float64]:
        """
        Velocity vector.
        """
        return np.array(self.table.select(["vx", "vy", "vz"]))

    @property
    def v_mag(self) -> npt.NDArray[np.float64]:
        """
        Magnitude of the velocity vector.
        """
        return np.linalg.norm(self.v, axis=1)

    @property
    def v_hat(self) -> npt.NDArray[np.float64]:
        """
        Unit vector in the direction of the velocity vector.
        """
        return self.v / self.v_mag[:, None]

    @property
    def sigma_x(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainty in the X-coordinate.
        """
        return self.covariance.sigmas[:, 0]

    @property
    def sigma_y(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainty in the Y-coordinate.
        """
        return self.covariance.sigmas[:, 1]

    @property
    def sigma_z(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainty in the Z-coordinate.
        """
        return self.covariance.sigmas[:, 2]

    @property
    def sigma_vx(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainty in the X-coordinate velocity.
        """
        return self.covariance.sigmas[:, 3]

    @property
    def sigma_vy(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainty in the Y-coordinate velocity.
        """
        return self.covariance.sigmas[:, 4]

    @property
    def sigma_vz(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainty in the Z-coordinate velocity.
        """
        return self.covariance.sigmas[:, 5]

    @property
    def sigma_r(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainties in the position vector.
        """
        return self.covariance.sigmas[:, 0:3]

    @property
    def sigma_r_mag(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainty in the position.
        """
        return np.sqrt(np.sum(self.covariance.sigmas[:, 0:3] ** 2, axis=1))

    @property
    def sigma_v(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainties in the velocity vector.
        """
        return self.covariance.sigmas[:, 3:6]

    @property
    def sigma_v_mag(self) -> npt.NDArray[np.float64]:
        """
        1-sigma uncertainty in the velocity vector.
        """
        return np.sqrt(np.sum(self.covariance.sigmas[:, 3:6] ** 2, axis=1))

    @property
    def h(self) -> npt.NDArray[np.float64]:
        """
        Specific angular momentum vector.
        """
        return np.cross(self.r, self.v)

    @property
    def h_mag(self) -> npt.NDArray[np.float64]:
        """
        Magnitude of the specific angular momentum vector.
        """
        return np.linalg.norm(self.h, axis=1)

    def rotate(
        self, rotation_matrix: npt.NDArray[np.float64], frame_out: str
    ) -> CartesianCoordinates:
        """
        Rotate Cartesian coordinates and their covariances by the given rotation matrix.

        Covariance matrices are also rotated. Rotations will sometimes result
        in covariance matrix elements very near zero but not exactly zero. Any
        elements that are smaller than +-1e-25 are rounded down to 0.

        Parameters
        ----------
        matrix:
            6x6 rotation matrix.
        frame_out: str
            Name of the frame to which coordinates are being rotated.

        Returns
        -------
            Rotated Cartesian coordinates and their covariances.
        """
        # Extract coordinate values into a masked array and mask NaNss
        masked_coords = np.ma.masked_array(self.values, fill_value=np.nan)
        masked_coords.mask = np.isnan(masked_coords.data)

        # Rotate coordinates
        coords_rotated = np.ma.dot(masked_coords, rotation_matrix.T, strict=False)

        # Extract covariances
        masked_covariances = np.ma.masked_array(
            self.covariance.to_matrix(), fill_value=0.0
        )
        masked_covariances.mask = np.isnan(masked_covariances.data)

        # Rotate covariances
        covariances_rotated = (
            rotation_matrix @ masked_covariances.filled() @ rotation_matrix.T
        )
        # Reset the mask to the original mask
        covariances_rotated[masked_covariances.mask] = np.nan

        # Check if any covariance elements are near zero, if so set them to zero
        near_zero = len(
            covariances_rotated[
                np.abs(covariances_rotated) < COVARIANCE_ROTATION_TOLERANCE
            ]
        )
        if near_zero > 0:
            logger.debug(
                f"{near_zero} covariance elements are within {COVARIANCE_ROTATION_TOLERANCE:.0e}"
                " of zero after rotation, setting these elements to 0."
            )
            covariances_rotated = np.where(
                np.abs(covariances_rotated) < COVARIANCE_ROTATION_TOLERANCE,
                0,
                covariances_rotated,
            )

        coords = self.from_kwargs(
            x=coords_rotated[:, 0],
            y=coords_rotated[:, 1],
            z=coords_rotated[:, 2],
            vx=coords_rotated[:, 3],
            vy=coords_rotated[:, 4],
            vz=coords_rotated[:, 5],
            time=self.time,
            covariance=CoordinateCovariances.from_matrix(covariances_rotated),
            origin=self.origin,
            frame=frame_out,
        )
        return coords

    def translate(
        self, vector: npt.NDArray[np.float64], origin_out: str
    ) -> CartesianCoordinates:
        """
        Translate Cartesian coordinates by the given vector.

        Parameters
        ----------

        vector:
            6x1 or 6xN translation vector. If a 6x1 vector is given, it is applied to all coordinates.
            If an 6xN array of vectors is given, each vector is applied to the corresponding coordinate.
        origin_out:
            Name of the origin to which coordinates are being translated.

        Returns
        -------
            Translated Cartesian coordinates and their covariances.
        """
        N = len(self)
        if vector.shape == (6,):
            vector_ = vector.reshape(1, 6)
        elif vector.shape == (N, 6):
            vector_ = vector
        else:
            raise ValueError(f"Expected vector to have shape (6,) or ({N}, 6).")

        # Extract coordinate values into a masked array and mask NaNss
        masked_coords = np.ma.masked_array(self.values, fill_value=np.nan)
        masked_coords.mask = np.isnan(masked_coords.data)

        # Translate coordinates
        coords_translated = (masked_coords + vector_).filled()

        coords = self.from_kwargs(
            x=coords_translated[:, 0],
            y=coords_translated[:, 1],
            z=coords_translated[:, 2],
            vx=coords_translated[:, 3],
            vy=coords_translated[:, 4],
            vz=coords_translated[:, 5],
            time=self.time,
            covariance=self.covariance,
            origin=Origin.from_kwargs(code=[origin_out] * len(self)),
            frame=self.frame,
        )
        return coords

    def to_cometary(self) -> cometary.CometaryCoordinates:
        """
        Converts the Cartesian coordinates to the cometary parameterization.
        """
        return cometary.CometaryCoordinates.from_cartesian(self)

    @classmethod
    def from_cometary(
        cls, cometary: cometary.CometaryCoordinates
    ) -> CartesianCoordinates:
        """
        Constructs CartesianCoordinates from the cometary parameterization.
        """
        return cometary.to_cartesian()

    def to_keplerian(self) -> keplerian.KeplerianCoordinates:
        """
        Converts the Cartesian coordinates to the Keplerian parameterization.
        """
        return keplerian.KeplerianCoordinates.from_cartesian(self)

    @classmethod
    def from_keplerian(
        cls, keplerian: keplerian.KeplerianCoordinates
    ) -> CartesianCoordinates:
        """
        Constructs CartesianCoordinates from the Keplerian parameterization.
        """
        return keplerian.to_cartesian()

    def to_spherical(self) -> spherical.SphericalCoordinates:
        """
        Converts the Cartesian coordinates to the spherical parameterization.
        """
        return spherical.SphericalCoordinates.from_cartesian(self)

    @classmethod
    def from_spherical(
        cls, spherical: spherical.SphericalCoordinates
    ) -> CartesianCoordinates:
        """
        Constructs CartesianCoordinates from the spherical parameterization.
        """
        return spherical.to_cartesian()
