import logging
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time

from .coordinates import Coordinates

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


class CartesianCoordinates(Coordinates):
    def __init__(
        self,
        x: Optional[Union[int, float, np.ndarray]] = None,
        y: Optional[Union[int, float, np.ndarray]] = None,
        z: Optional[Union[int, float, np.ndarray]] = None,
        vx: Optional[Union[int, float, np.ndarray]] = None,
        vy: Optional[Union[int, float, np.ndarray]] = None,
        vz: Optional[Union[int, float, np.ndarray]] = None,
        times: Optional[Time] = None,
        covariances: Optional[np.ndarray] = None,
        sigma_x: Optional[np.ndarray] = None,
        sigma_y: Optional[np.ndarray] = None,
        sigma_z: Optional[np.ndarray] = None,
        sigma_vx: Optional[np.ndarray] = None,
        sigma_vy: Optional[np.ndarray] = None,
        sigma_vz: Optional[np.ndarray] = None,
        origin: str = "heliocenter",
        frame: str = "ecliptic",
        names: dict = CARTESIAN_COLS,
        units: dict = CARTESIAN_UNITS,
    ):
        """

        Parameters
        ----------
        x : `~numpy.ndarray` (N)
            X-coordinate.
        y : `~numpy.ndarray` (N)
            Y-coordinate.
        z : `~numpy.ndarray` (N)
            Z-coordinate.
        vx : `~numpy.ndarray` (N)
            X-coordinate velocity.
        vy : `~numpy.ndarray` (N)
            Y-coordinate velocity.
        vz : `~numpy.ndarray` (N)
            Z-coordinate velocity.
        """
        sigmas = (sigma_x, sigma_y, sigma_z, sigma_vx, sigma_vy, sigma_vz)
        Coordinates.__init__(
            self,
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            covariances=covariances,
            sigmas=sigmas,
            times=times,
            origin=origin,
            frame=frame,
            names=names,
            units=units,
        )
        return

    @property
    def x(self):
        """
        X-coordinate
        """
        return self._values[:, 0]

    @x.setter
    def x(self, value):
        self._values[:, 0] = value
        self._values[:, 0].mask = False

    @x.deleter
    def x(self):
        self._values[:, 0] = np.nan
        self._values[:, 0].mask = True

    @property
    def y(self):
        """
        Y-coordinate
        """
        return self._values[:, 1]

    @y.setter
    def y(self, value):
        self._values[:, 1] = value
        self._values[:, 1].mask = False

    @y.deleter
    def y(self):
        self._values[:, 1] = np.nan
        self._values[:, 1].mask = True

    @property
    def z(self):
        """
        Z-coordinate
        """
        return self._values[:, 2]

    @z.setter
    def z(self, value):
        self._values[:, 2] = value
        self._values[:, 2].mask = False

    @z.deleter
    def z(self):
        self._values[:, 2] = np.nan
        self._values[:, 2].mask = True

    @property
    def vx(self):
        """
        X-coordinate velocity
        """
        return self._values[:, 3]

    @vx.setter
    def vx(self, value):
        self._values[:, 3] = value
        self._values[:, 3].mask = False

    @vx.deleter
    def vx(self):
        self._values[:, 3] = np.nan
        self._values[:, 3].mask = True

    @property
    def vy(self):
        """
        Y-coordinate velocity
        """
        return self._values[:, 4]

    @vy.setter
    def vy(self, value):
        self._values[:, 4] = value
        self._values[:, 4].mask = False

    @vy.deleter
    def vy(self):
        self._values[:, 4] = np.nan
        self._values[:, 4].mask = True

    @property
    def vz(self):
        """
        Z-coordinate velocity
        """
        return self._values[:, 5]

    @vz.setter
    def vz(self, value):
        self._values[:, 5] = value
        self._values[:, 5].mask = False

    @vz.deleter
    def vz(self):
        self._values[:, 5] = np.nan
        self._values[:, 5].mask = True

    @property
    def sigma_x(self):
        """
        1-sigma uncertainty in the X-coordinate
        """
        return self.sigmas[:, 0]

    @sigma_x.setter
    def sigma_x(self, value):
        self._covariances[:, 0, 0] = value**2
        self._covariances[:, 0, 0].mask = False

    @sigma_x.deleter
    def sigma_x(self):
        self._covariances[:, 0, 0] = np.nan
        self._covariances[:, 0, 0].mask = True

    @property
    def sigma_y(self):
        """
        1-sigma uncertainty in the Y-coordinate
        """
        return self.sigmas[:, 1]

    @sigma_y.setter
    def sigma_y(self, value):
        self._covariances[:, 1, 1] = value**2
        self._covariances[:, 1, 1].mask = False

    @sigma_y.deleter
    def sigma_y(self):
        self._covariances[:, 1, 1] = np.nan
        self._covariances[:, 1, 1].mask = True

    @property
    def sigma_z(self):
        """
        1-sigma uncertainty in the Z-coordinate
        """
        return self.sigmas[:, 2]

    @sigma_z.setter
    def sigma_z(self, value):
        self._covariances[:, 2, 2] = value**2
        self._covariances[:, 2, 2].mask = False

    @sigma_z.deleter
    def sigma_z(self):
        self._covariances[:, 2, 2] = np.nan
        self._covariances[:, 2, 2].mask = True

    @property
    def sigma_vx(self):
        """
        1-sigma uncertainty in the X-coordinate velocity
        """
        return self.sigmas[:, 3]

    @sigma_vx.setter
    def sigma_vx(self, value):
        self._covariances[:, 3, 3] = value**2
        self._covariances[:, 3, 3].mask = False

    @sigma_vx.deleter
    def sigma_vx(self):
        self._covariances[:, 3, 3] = np.nan
        self._covariances[:, 3, 3].mask = True

    @property
    def sigma_vy(self):
        """
        1-sigma uncertainty in the Y-coordinate velocity
        """
        return self.sigmas[:, 4]

    @sigma_vy.setter
    def sigma_vy(self, value):
        self._covariances[:, 4, 4] = value**2
        self._covariances[:, 4, 4].mask = False

    @sigma_vy.deleter
    def sigma_vy(self):
        self._covariances[:, 4, 4] = np.nan
        self._covariances[:, 4, 4].mask = True

    @property
    def sigma_vz(self):
        """
        1-sigma uncertainty in the Z-coordinate velocity
        """
        return self.sigmas[:, 5]

    @sigma_vz.setter
    def sigma_vz(self, value):
        self._covariances[:, 5, 5] = value**2
        self._covariances[:, 5, 5].mask = False

    @sigma_vz.deleter
    def sigma_vz(self):
        self._covariances[:, 5, 5] = np.nan
        self._covariances[:, 5, 5].mask = True

    @property
    def r(self):
        """
        Position vector
        """
        return self._values[:, 0:3]

    @r.setter
    def r(self, value):
        self._values[:, 0:3] = value
        self._values[:, 0:3].mask = False

    @r.deleter
    def r(self):
        self._values[:, 0:3] = np.nan
        self._values[:, 0:3].mask = True

    @property
    def v(self):
        """
        Velocity vector
        """
        return self._values[:, 3:6]

    @v.setter
    def v(self, value):
        self._values[:, 3:6] = value
        self._values[:, 3:6].mask = False

    @v.deleter
    def v(self):
        self._values[:, 3:6] = np.nan
        self._values[:, 3:6].mask = True

    @property
    def sigma_r_mag(self):
        """
        1-sigma uncertainty in the position
        """
        return np.sqrt(np.sum(self.sigmas.filled()[:, 0:3] ** 2, axis=1))

    @sigma_r_mag.setter
    def sigma_r_mag(self, value):
        err = (
            "Cannot set 1-sigma uncertainty in position as it is"
            " derived from the 1-sigma uncertainties in x, y, and z."
        )
        raise ValueError(err)

    @sigma_r_mag.deleter
    def sigma_r_mag(self):
        err = (
            "Cannot delete the 1-sigma uncertainty in position as it is"
            " derived from the 1-sigma uncertainties in x, y, and z."
        )
        raise ValueError(err)

    @property
    def sigma_v_mag(self):
        """
        1-sigma uncertainty in the velocity
        """
        return np.sqrt(np.sum(self.sigmas.filled()[:, 3:6] ** 2, axis=1))

    @sigma_v_mag.setter
    def sigma_v_mag(self, value):
        err = (
            "Cannot set the 1-sigma uncertainty in velocity as it is"
            " derived from the 1-sigma uncertainties in vx, vy, and vz."
        )
        raise ValueError(err)

    @sigma_v_mag.deleter
    def sigma_v_mag(self):
        err = (
            "Cannot delete the 1-sigma uncertainty in velocity as it is"
            " derived from the 1-sigma uncertainties in vx, vy, and vz."
        )
        raise ValueError(err)

    @property
    def r_mag(self):
        """
        Magnitude of the position vector
        """
        return np.linalg.norm(self.r.filled(), axis=1)

    @r_mag.setter
    def r_mag(self, value):
        err = (
            "Cannot set the magnitude of the position vector as it is"
            " derived from the position vector."
        )
        raise ValueError(err)

    @r_mag.deleter
    def r_mag(self):
        err = (
            "Cannot delete the magnitude of the position vector as it is"
            "derived from the position vector."
        )
        raise ValueError(err)

    @property
    def v_mag(self):
        """
        Magnitude of the velocity vector
        """
        return np.linalg.norm(self.v.filled(), axis=1)

    @v_mag.setter
    def v_mag(self, value):
        err = "Cannot set the magnitude of the velocity vector as it is derived from the velocity vector."
        raise ValueError(err)

    @v_mag.deleter
    def v_mag(self):
        err = "Cannot delete the magnitude of the velocity vector as it is derived from the velocity vector."
        raise ValueError(err)

    @property
    def r_hat(self):
        """
        Unit vector in the direction of the position vector
        """
        return self.r.filled() / self.r_mag.reshape(-1, 1)

    @r_hat.setter
    def r_hat(self, value):
        err = (
            "Cannot set the unit vector in the direction of the position vector as it is"
            " derived from the position vector."
        )
        raise ValueError(err)

    @r_hat.deleter
    def r_hat(self):
        err = (
            "Cannot delete the unit vector in the direction of the position vector as it is"
            " derived from the position vector."
        )
        raise ValueError(err)

    @property
    def v_hat(self):
        """
        Unit vector in the direction of the velocity vector
        """
        return self.v.filled() / self.v_mag.reshape(-1, 1)

    @v_hat.setter
    def v_hat(self, value):
        err = (
            "Cannot set the unit vector in the direction of the velocity vector as it is"
            " derived from the velocity vector."
        )
        raise ValueError(err)

    @v_hat.deleter
    def v_hat(self):
        err = (
            "Cannot delete the unit vector in the direction of the velocity vector as it is"
            " derived from the velocity vector."
        )
        raise ValueError(err)

    def to_cartesian(self):
        return self

    @classmethod
    def from_cartesian(
        cls, cartesian: "CartesianCoordinates"
    ) -> "CartesianCoordinates":
        return cartesian

    def rotate(self, matrix: np.ndarray, frame_out: str) -> "CartesianCoordinates":
        """
        Rotate Cartesian coordinates and their covariances by the
        given rotation matrix. A copy is made of the coordinates and a new
        instance of the CartesianCoordinates class is returned.

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
        CartesianCoordinates : `~thor.coordinates.cartesian.CartesianCoordinates`
            Rotated Cartesian coordinates and their covariances.
        """
        coords_rotated = deepcopy(np.ma.dot(self._values, matrix.T))
        coords_rotated[self._values.mask] = np.NaN

        if self._covariances is not None:
            covariances_rotated = deepcopy(matrix @ self._covariances @ matrix.T)
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

        else:
            covariances_rotated = None

        data = {}
        data["x"] = coords_rotated[:, 0]
        data["y"] = coords_rotated[:, 1]
        data["z"] = coords_rotated[:, 2]
        data["vx"] = coords_rotated[:, 3]
        data["vy"] = coords_rotated[:, 4]
        data["vz"] = coords_rotated[:, 5]
        data["times"] = deepcopy(self.times)
        data["covariances"] = covariances_rotated
        data["origin"] = deepcopy(self.origin)
        data["frame"] = deepcopy(frame_out)
        data["units"] = deepcopy(self.units)
        data["names"] = deepcopy(self.names)
        return CartesianCoordinates(**data)

    def translate(
        self, vectors: Union[np.ndarray, np.ma.masked_array], origin_out: str
    ) -> "CartesianCoordinates":
        """
        Translate CartesianCoordinates by the given coordinate vector(s).
        A copy is made of the coordinates and a new instance of the
        CartesianCoordinates class is returned.

        Translation will only be applied to those coordinates that do not already
        have the desired origin (self.origin != origin_out).

        Parameters
        ----------
        vectors : {`~numpy.ndarray`, `~numpy.ma.masked_array`} (N, 6), (1, 6) or (6)
            Translation vector(s) for each coordinate or a single vector with which
            to translate all coordinates.
        origin_out : str
            Name of the origin to which coordinates are being translated.

        Returns
        -------
        CartesianCoordinates : `~thor.coordinates.cartesian.CartesianCoordinates`
            Translated Cartesian coordinates and their covariances.

        Raises
        ------
        ValueError: If vectors does not have shape (N, 6), (1, 6), or (6)
        TypeError: If vectors is not a `~numpy.ndarray` or a `~numpy.ma.masked_array`
        """
        if not isinstance(vectors, (np.ndarray, np.ma.masked_array)):
            err = "coords should be one of {`~numpy.ndarray`, `~numpy.ma.masked_array`}"
            raise TypeError(err)

        if len(vectors.shape) == 2:
            N, D = vectors.shape
        elif len(vectors.shape) == 1:
            N, D = vectors.shape[0], None
        else:
            err = (
                f"vectors should be 2D or 1D, instead vectors is {len(vectors.shape)}D."
            )
            raise ValueError(err)

        N_self, D_self = self.values.shape
        if (N != len(self) and N != 1) and (D is None and N != D_self):
            err = (
                f"Translation vector(s) should have shape ({N_self}, {D_self}),"
                f" (1, {D_self}) or ({D_self},).\n"
                f"Given translation vector(s) has shape {vectors.shape}."
            )
            raise ValueError(err)

        coords_translated = deepcopy(self.values)

        # Only apply translation to coordinates that do not already have the desired origin
        origin_different_mask = np.where(self.origin != origin_out)[0]
        origin_same_mask = np.where(self.origin == origin_out)[0]
        if len(coords_translated[origin_same_mask]) > 0:
            info = (
                f"Translation will not be applied to the {len(coords_translated[origin_same_mask])} "
                "coordinates that already have the desired origin."
            )
            logger.info(info)

        if len(vectors.shape) == 2:
            coords_translated[origin_different_mask] = (
                coords_translated[origin_different_mask]
                + vectors[origin_different_mask]
            )
        else:
            coords_translated[origin_different_mask] = (
                coords_translated[origin_different_mask] + vectors
            )

        covariances_translated = deepcopy(self.covariances)

        data = {}
        data["x"] = coords_translated[:, 0]
        data["y"] = coords_translated[:, 1]
        data["z"] = coords_translated[:, 2]
        data["vx"] = coords_translated[:, 3]
        data["vy"] = coords_translated[:, 4]
        data["vz"] = coords_translated[:, 5]
        data["times"] = deepcopy(self.times)
        data["covariances"] = covariances_translated
        data["origin"] = deepcopy(origin_out)
        data["frame"] = deepcopy(self.frame)
        data["units"] = deepcopy(self.units)
        data["names"] = deepcopy(self.names)
        return CartesianCoordinates(**data)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        coord_cols: dict = CARTESIAN_COLS,
        origin_col: str = "origin",
        frame_col: str = "frame",
    ) -> "CartesianCoordinates":
        """
        Create a CartesianCoordinates class from a dataframe.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing Cartesian coordinates and optionally their
            times and covariances.
        coord_cols : dict
            Dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = {}
                coord_cols["x"] = Column name of x distance values
                coord_cols["y"] = Column name of y distance values
                coord_cols["z"] = Column name of z distance values
                coord_cols["vx"] = Column name of x velocity values
                coord_cols["vy"] = Column name of y velocity values
                coord_cols["vz"] = Column name of z velocity values
        origin_col : str
            Name of the column containing the origin of each coordinate.
        """
        data = Coordinates._dict_from_df(
            df, coord_cols=coord_cols, origin_col=origin_col, frame_col=frame_col
        )
        return cls(**data)
