import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from astropy import units as u
from quivr import Float64Field, StringAttribute, Table

from .covariances import CoordinateCovariances
from .io import coords_from_dataframe, coords_to_dataframe
from .origin import Origin
from .times import Times

if TYPE_CHECKING:
    from .cometary import CometaryCoordinates
    from .keplerian import KeplerianCoordinates
    from .spherical import SphericalCoordinates

__all__ = ["CartesianCoordinates", "CARTESIAN_COLS", "CARTESIAN_UNITS"]

CARTESIAN_COLS = {}
CARTESIAN_UNITS = {}
for i in ["x", "y", "z"]:
    CARTESIAN_COLS[i] = i
    CARTESIAN_UNITS[i] = u.au
for i in ["vx", "vy", "vz"]:
    CARTESIAN_COLS[i] = i
    CARTESIAN_UNITS[i] = u.au / u.d

COVARIANCE_ROTATION_TOLERANCE = 1e-25
logger = logging.getLogger(__name__)


class CartesianCoordinates(Table):

    x = Float64Field(nullable=True)
    y = Float64Field(nullable=True)
    z = Float64Field(nullable=True)
    vx = Float64Field(nullable=True)
    vy = Float64Field(nullable=True)
    vz = Float64Field(nullable=True)
    times = Times.as_field(nullable=True)
    covariances = CoordinateCovariances.as_field(nullable=True)
    origin = Origin.as_field(nullable=False)
    frame = StringAttribute()

    @property
    def values(self) -> np.ndarray:
        return np.array(self.table.select(["x", "y", "z", "vx", "vy", "vz"])).T

    @property
    def r(self) -> np.ndarray:
        """
        Position vector.
        """
        return np.array(self.table.select(["x", "y", "z"]))

    @property
    def r_mag(self) -> np.ndarray:
        """
        Magnitude of the position vector.
        """
        return np.linalg.norm(self.r, axis=1)

    @property
    def r_hat(self) -> np.ndarray:
        """
        Unit vector in the direction of the position vector.
        """
        return self.r / self.r_mag[:, None]

    @property
    def v(self) -> np.ndarray:
        """
        Velocity vector.
        """
        return np.array(self.table.select(["vx", "vy", "vz"]))

    @property
    def v_mag(self) -> np.ndarray:
        """
        Magnitude of the velocity vector.
        """
        return np.linalg.norm(self.v, axis=1)

    @property
    def v_hat(self) -> np.ndarray:
        """
        Unit vector in the direction of the velocity vector.
        """
        return self.v / self.v_mag[:, None]

    @property
    def sigma_x(self) -> np.ndarray:
        """
        1-sigma uncertainty in the X-coordinate.
        """
        return self.covariances.sigmas[:, 0]

    @property
    def sigma_y(self) -> np.ndarray:
        """
        1-sigma uncertainty in the Y-coordinate.
        """
        return self.covariances.sigmas[:, 1]

    @property
    def sigma_z(self) -> np.ndarray:
        """
        1-sigma uncertainty in the Z-coordinate.
        """
        return self.covariances.sigmas[:, 2]

    @property
    def sigma_vx(self) -> np.ndarray:
        """
        1-sigma uncertainty in the X-coordinate velocity.
        """
        return self.covariances.sigmas[:, 3]

    @property
    def sigma_vy(self) -> np.ndarray:
        """
        1-sigma uncertainty in the Y-coordinate velocity.
        """
        return self.covariances.sigmas[:, 4]

    @property
    def sigma_vz(self) -> np.ndarray:
        """
        1-sigma uncertainty in the Z-coordinate velocity.
        """
        return self.covariances.sigmas[:, 5]

    @property
    def sigma_r(self) -> np.ndarray:
        """
        1-sigma uncertainties in the position vector.
        """
        return np.sqrt(self.covariances.sigmas[:, 0:3])

    @property
    def sigma_r_mag(self) -> np.ndarray:
        """
        1-sigma uncertainty in the position.
        """
        return np.sqrt(np.sum(self.covariances.sigmas[:, 0:3] ** 2, axis=1))

    @property
    def sigma_v(self) -> np.ndarray:
        """
        1-sigma uncertainties in the velocity vector.
        """
        return np.sqrt(self.covariances.sigmas[:, 3:6])

    @property
    def sigma_v_mag(self) -> np.ndarray:
        """
        1-sigma uncertainty in the velocity vector.
        """
        return np.sqrt(np.sum(self.covariances.sigmas[:, 3:6] ** 2, axis=1))

    def rotate(
        self, rotation_matrix: np.ndarray, frame_out: str
    ) -> "CartesianCoordinates":
        """
        Rotate Cartesian coordinates and their covariances by the given rotation matrix.

        Covariance matrices are also rotated. Rotations will sometimes result
        in covariance matrix elements very near zero but not exactly zero. Any
        elements that are smaller than +-1e-25 are rounded down to 0.

        Parameters
        ----------
        matrix : `~numpy.ndarray` (6, 6)
            Rotation matrix.
        frame_out : str
            Name of the frame to which coordinates are being rotated.

        Returns
        -------
        CartesianCoordinates : `~adam_core.coordinates.cartesian.CartesianCoordinates`
            Rotated Cartesian coordinates and their covariances.
        """
        # Extract coordinate values into a masked array and mask NaNss
        masked_coords = np.ma.masked_array(self.values, fill_value=np.NaN)
        masked_coords.mask = np.isnan(masked_coords.data)

        # Rotate coordinates
        coords_rotated = np.ma.dot(masked_coords, rotation_matrix.T, strict=False)

        # Extract covariances
        masked_covariances = np.ma.masked_array(
            self.covariances.to_matrix(), fill_value=0.0
        )
        masked_covariances.mask = np.isnan(masked_covariances.data)

        # Rotate covariances
        covariances_rotated = (
            rotation_matrix @ masked_covariances.filled() @ rotation_matrix.T
        )
        # Reset the mask to the original mask
        covariances_rotated[masked_covariances.mask] = np.NaN

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
            times=self.times,
            covariances=CoordinateCovariances.from_matrix(covariances_rotated),
            origin=self.origin,
            frame=frame_out,
        )
        return coords

    def translate(self, vector: np.ndarray, origin_out: str) -> "CartesianCoordinates":
        """
        Translate Cartesian coordinates by the given vector.

        Parameters
        ----------
        vector : `~numpy.ndarray` (6,) or (N, 6)
            Translation vector. If a single vector is given, it is applied to all coordinates.
            If an array of vectors is given, each vector is applied to the corresponding coordinate.
        origin_out : str
            Name of the origin to which coordinates are being translated.

        Returns
        -------
        CartesianCoordinates : `~adam_core.coordinates.cartesian.CartesianCoordinates`
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
        masked_coords = np.ma.masked_array(self.values, fill_value=np.NaN)
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
            times=self.times,
            covariances=self.covariances,
            origin=Origin.from_kwargs(code=[origin_out] * len(self)),
            frame=self.frame,
        )
        return coords

    def to_cometary(self) -> "CometaryCoordinates":
        from .cometary import CometaryCoordinates

        return CometaryCoordinates.from_cartesian(self)

    @classmethod
    def from_cometary(cls, cometary: "CometaryCoordinates") -> "CartesianCoordinates":
        return cometary.to_cartesian()

    def to_keplerian(self) -> "KeplerianCoordinates":
        from .keplerian import KeplerianCoordinates

        return KeplerianCoordinates.from_cartesian(self)

    @classmethod
    def from_keplerian(
        cls, keplerian: "KeplerianCoordinates"
    ) -> "CartesianCoordinates":
        return keplerian.to_cartesian()

    def to_spherical(self) -> "SphericalCoordinates":
        from .spherical import SphericalCoordinates

        return SphericalCoordinates.from_cartesian(self)

    @classmethod
    def from_spherical(
        cls, spherical: "SphericalCoordinates"
    ) -> "CartesianCoordinates":
        return spherical.to_cartesian()

    def to_dataframe(
        self, sigmas: bool = False, covariances: bool = True
    ) -> pd.DataFrame:
        """
        Convert coordinates to a pandas DataFrame.

        Parameters
        ----------
        sigmas : bool, optional
            If True, include 1-sigma uncertainties in the DataFrame.
        covariances : bool, optional
            If True, include covariance matrices in the DataFrame. Covariance matrices
            will be split into 21 columns, with the lower triangular elements stored.

        Returns
        -------
        df : `~pandas.Dataframe`
            DataFrame containing coordinates.
        """
        return coords_to_dataframe(
            self,
            ["x", "y", "z", "vx", "vy", "vz"],
            sigmas=sigmas,
            covariances=covariances,
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, frame: Literal["ecliptic", "equatorial"]
    ) -> "CartesianCoordinates":
        """
        Create coordinates from a pandas DataFrame.

        Parameters
        ----------
        df : `~pandas.Dataframe`
            DataFrame containing coordinates.
        frame : {"ecliptic", "equatorial"}
            Frame in which coordinates are defined.

        Returns
        -------
        coords : `~adam_core.coordinates.cartesian.CartesianCoordinates`
            Cartesian coordinates.
        """
        return coords_from_dataframe(
            cls, df, coord_names=["x", "y", "z", "vx", "vy", "vz"], frame=frame
        )
