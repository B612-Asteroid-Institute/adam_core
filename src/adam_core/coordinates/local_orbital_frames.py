"""Local orbital frames (SANA orbit-relative registry names) for covariances.

RSW (RTN, RIC): x along position, z along the orbital angular momentum h.
TNW: x along velocity, z along h. VNC: x along velocity, y along h. The third
axis completes each right-handed set, so VNC is TNW with rows (T, W, -N) and the
two must never be labelled as each other. ``_INERTIAL`` variants rotate position
and velocity alike, ``_ROTATING`` variants add the two-body frame rate to the
velocity rows. Bare names resolve to ``_INERTIAL``. Only covariances are
expressed here. State vectors stay inertial.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt
import quivr as qv

from ..time import Timestamp
from .cartesian import SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE, CartesianCoordinates
from .covariances import CoordinateCovariances, apply_linear_covariance_transform
from .origin import Origin

if TYPE_CHECKING:
    from ..orbits.orbits import Orbits

_FAMILIES = ("RSW", "TNW", "VNC")
_FRAMES = tuple(f"{f}_{kind}" for f in _FAMILIES for kind in ("ROTATING", "INERTIAL"))
_ALIASES = {"RTN": "RSW", "RIC": "RSW"}
_Mu = Optional[Union[float, npt.ArrayLike]]


def _canonical_frame(frame: str) -> str:
    name = str(frame).strip().upper()
    name = _ALIASES.get(name, name)
    if name in _FAMILIES:  # a bare name means the quasi-inertial variant
        name += "_INERTIAL"
    if name not in _FRAMES:
        raise ValueError(
            f"Unknown local orbital frame {frame!r}, expected one of {_FRAMES}."
        )
    return name


def _rotation_matrices(coords: CartesianCoordinates, frame: str) -> np.ndarray:
    if coords.frame not in ("equatorial", "ecliptic"):
        raise ValueError(
            f"Local orbital frames need an inertial frame, got {coords.frame!r}."
        )
    degenerate = coords.h_mag < SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE
    if degenerate.any():
        raise ValueError(
            f"State {int(np.flatnonzero(degenerate)[0])} has no orbit plane."
        )
    family = _canonical_frame(frame).split("_")[0]
    if family == "RSW":
        return coords.ric3_matrix
    if family == "TNW":
        return np.stack(
            [coords.v_hat, np.cross(coords.h_hat, coords.v_hat), coords.h_hat], axis=1
        )
    return np.stack(
        [coords.v_hat, coords.h_hat, np.cross(coords.v_hat, coords.h_hat)], axis=1
    )


def _unit_vector_rate(
    vectors: np.ndarray, vector_rates: np.ndarray, norms: np.ndarray
) -> np.ndarray:
    """d(u/|u|)/dt for each row, given u and du/dt."""
    unit = vectors / norms[:, None]
    along = np.einsum("ij,ij->i", vector_rates, unit)[:, None]
    return (vector_rates - along * unit) / norms[:, None]


def _angular_velocity(coords: CartesianCoordinates, frame: str, mu: _Mu) -> np.ndarray:
    family = frame.split("_")[0]
    h_hat = coords.h_hat
    if mu is None:
        try:
            mu = coords.origin.mu()
        except ValueError as exc:
            raise ValueError(
                f"No gravitational parameter for {coords.origin.code.unique().to_pylist()}. "
                "Pass mu= in AU^3/day^2 or use the _INERTIAL variant."
            ) from exc
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (len(coords),))
    acceleration = -mu[:, None] * coords.r / coords.r_mag[:, None] ** 3
    r_hat_rate = _unit_vector_rate(coords.r, coords.v, coords.r_mag)
    v_hat_rate = _unit_vector_rate(coords.v, acceleration, coords.v_mag)
    # Under two-body motion only r_hat and v_hat turn, the orbit normal does not.
    x_rate = r_hat_rate if family == "RSW" else v_hat_rate
    y_rate = np.cross(h_hat, x_rate)  # rate of the axis completing (x, h_hat)
    zero_rate = np.zeros_like(h_hat)
    if family == "VNC":  # VNC rows are (x, h_hat, -y)
        axis_rates = np.stack([x_rate, zero_rate, -y_rate], axis=1)
    else:
        axis_rates = np.stack([x_rate, y_rate, zero_rate], axis=1)
    # For an orthonormal triad with de_i = omega x e_i, sum_i e_i x de_i = 2 omega.
    return 0.5 * np.cross(_rotation_matrices(coords, frame), axis_rates, axis=2).sum(
        axis=1
    )


def local_frame_jacobians(
    coords: CartesianCoordinates, frame: str, mu: _Mu = None
) -> npt.NDArray[np.float64]:
    """
    (N, 6, 6) Jacobians from inertial position and velocity to a local orbital
    frame (name or alias), so a covariance there is ``J @ C @ J.T``. The top
    left block is the rotation. ``mu`` (AU^3/day^2, default the origin's) sets
    the ``_ROTATING`` frame rate.
    """
    canonical_frame = _canonical_frame(frame)
    rotation = _rotation_matrices(coords, canonical_frame)
    jacobians = np.zeros((len(coords), 6, 6))
    jacobians[:, :3, :3] = rotation
    jacobians[:, 3:, 3:] = rotation
    if canonical_frame.endswith("_ROTATING"):
        omega = _angular_velocity(coords, canonical_frame, mu)
        jacobians[:, 3:, :3] = np.cross(omega[:, None, :], rotation)
    return jacobians


class LocalFrameCovariances(qv.Table):
    """
    Per epoch covariances in a local orbital frame (AU, AU/day), one row per
    source orbit. Rows without a source covariance stay NaN.
    """

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    time = Timestamp.as_column()
    covariance = CoordinateCovariances.as_column()
    origin = Origin.as_column()
    #: Canonical local frame name, and the inertial frame its axes were built from.
    frame = qv.StringAttribute(default="unspecified")
    inertial_frame = qv.StringAttribute(default="unspecified")

    @classmethod
    def from_orbits(
        cls, orbits: "Orbits", frame: str = "VNC_ROTATING", mu: _Mu = None
    ) -> "LocalFrameCovariances":
        """Rotate the covariances of ``orbits`` into ``frame`` (name or alias)."""
        canonical_frame = _canonical_frame(frame)
        coords = orbits.coordinates
        if coords.covariance.is_all_nan():
            raise ValueError("The orbits carry no covariance.")
        rotated = apply_linear_covariance_transform(
            local_frame_jacobians(coords, canonical_frame, mu),
            coords.covariance.to_matrix(),
        )
        return cls.from_kwargs(
            orbit_id=orbits.orbit_id,
            object_id=orbits.object_id,
            time=coords.time,
            covariance=CoordinateCovariances.from_matrix(rotated),
            origin=coords.origin,
            frame=canonical_frame,
            inertial_frame=coords.frame,
        )
