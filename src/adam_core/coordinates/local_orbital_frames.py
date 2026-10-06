"""Local orbital frames and covariances expressed in them.

Names follow the SANA orbit-relative reference frames registry used by the
CCSDS navigation data messages. RSW has x along the position and z along the
orbital angular momentum (also called RTN and RIC). TNW has x along the
velocity and z along the angular momentum. VNC has x along the velocity and y
along the angular momentum. The third axis completes each right-handed set, so
VNC and TNW share two axes and must never be labelled as each other.

``_INERTIAL`` variants rotate position and velocity with the same 3x3 matrix.
``_ROTATING`` variants also carry the frame angular velocity, from two-body
motion about the origin, into the velocity rows. Bare names resolve to the
``_INERTIAL`` variants, as the CCSDS messages use them. Only covariances are
expressed in these frames. State vectors stay inertial.
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

__all__ = ["LocalFrameCovariances", "local_frame_jacobians"]

_FRAMES = (
    "RSW_ROTATING",
    "RSW_INERTIAL",
    "TNW_ROTATING",
    "TNW_INERTIAL",
    "VNC_ROTATING",
    "VNC_INERTIAL",
)
_ALIASES = {
    "RSW": "RSW_INERTIAL",
    "RTN": "RSW_INERTIAL",
    "RIC": "RSW_INERTIAL",
    "TNW": "TNW_INERTIAL",
    "VNC": "VNC_INERTIAL",
}
_Mu = Optional[Union[float, npt.ArrayLike]]


def _canonical(frame: str) -> str:
    label = str(frame).strip().upper()
    label = _ALIASES.get(label, label)
    if label not in _FRAMES:
        raise ValueError(
            f"Unknown local orbital frame {frame!r}, expected one of "
            f"{list(_FRAMES)} or an alias in {list(_ALIASES)}."
        )
    return label


def _unit_vectors(coords: CartesianCoordinates) -> tuple[np.ndarray, ...]:
    """Validated r_hat, v_hat and the unit orbit normal."""
    if coords.frame not in ("equatorial", "ecliptic"):
        raise ValueError(
            f"Local orbital frames need states in an inertial frame, got {coords.frame!r}."
        )
    h_mag = coords.h_mag
    degenerate = h_mag < SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE
    if degenerate.any():
        raise ValueError(
            f"State at row {int(np.flatnonzero(degenerate)[0])} has no orbit plane."
        )
    return coords.r_hat, coords.v_hat, coords.h / h_mag[:, None]


def _rotation_matrices(coords: CartesianCoordinates, frame: str) -> np.ndarray:
    family = _canonical(frame).split("_")[0]
    r_hat, v_hat, w_hat = _unit_vectors(coords)
    if family == "RSW":
        return coords.ric3_matrix
    if family == "TNW":
        return np.stack([v_hat, np.cross(w_hat, v_hat), w_hat], axis=1)
    return np.stack([v_hat, w_hat, np.cross(v_hat, w_hat)], axis=1)


def _unit_rate(vectors: np.ndarray, rates: np.ndarray, norms: np.ndarray) -> np.ndarray:
    # d(u/|u|)/dt = (du - (du . u_hat) u_hat) / |u|
    unit = vectors / norms[:, None]
    return (rates - np.einsum("ij,ij->i", rates, unit)[:, None] * unit) / norms[:, None]


def _angular_velocity(coords: CartesianCoordinates, frame: str, mu: _Mu) -> np.ndarray:
    family = frame.split("_")[0]
    _, _, w_hat = _unit_vectors(coords)
    if mu is None:
        try:
            mu = coords.origin.mu()
        except ValueError as exc:
            raise ValueError(
                "A _ROTATING frame needs the origin's gravitational parameter, "
                f"unknown for {coords.origin.code.unique().to_pylist()}. Pass mu= "
                "in AU^3/day^2 or use the _INERTIAL variant."
            ) from exc
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (len(coords),))
    acceleration = -mu[:, None] * coords.r / coords.r_mag[:, None] ** 3
    dr_hat = _unit_rate(coords.r, coords.v, coords.r_mag)
    dv_hat = _unit_rate(coords.v, acceleration, coords.v_mag)
    # Under two-body motion the orbit normal is fixed, so only r_hat and v_hat turn.
    fixed = np.zeros_like(w_hat)
    if family == "RSW":
        rates = np.stack([dr_hat, np.cross(w_hat, dr_hat), fixed], axis=1)
    elif family == "TNW":
        rates = np.stack([dv_hat, np.cross(w_hat, dv_hat), fixed], axis=1)
    else:
        rates = np.stack([dv_hat, fixed, np.cross(dv_hat, w_hat)], axis=1)
    # For an orthonormal triad with de_i = omega x e_i, sum_i e_i x de_i = 2 omega.
    return 0.5 * np.cross(_rotation_matrices(coords, frame), rates, axis=2).sum(axis=1)


def local_frame_jacobians(
    coords: CartesianCoordinates, frame: str, mu: _Mu = None
) -> npt.NDArray[np.float64]:
    """
    (N, 6, 6) Jacobians from inertial position and velocity to a local orbital
    frame (name or alias), so a covariance there is ``J @ C @ J.T``. The top
    left block is the 3x3 rotation. ``_ROTATING`` variants take ``mu`` in
    AU^3/day^2 for the frame rate (default: the origin's value).
    """
    canonical = _canonical(frame)
    rotation = _rotation_matrices(coords, canonical)
    jacobians = np.zeros((len(coords), 6, 6))
    jacobians[:, :3, :3] = rotation
    jacobians[:, 3:, 3:] = rotation
    if canonical.endswith("_ROTATING"):
        omega = _angular_velocity(coords, canonical, mu)
        jacobians[:, 3:, :3] = np.cross(omega[:, None, :], rotation)
    return jacobians


class LocalFrameCovariances(qv.Table):
    """
    Per-epoch state covariances in a local orbital frame, AU and AU/day. Rows
    align one to one with the source orbits, which stay inertial. Rows without
    a source covariance stay NaN.
    """

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    time = Timestamp.as_column()
    covariance = CoordinateCovariances.as_column()
    origin = Origin.as_column()
    #: Canonical frame name, for example ``"VNC_ROTATING"``.
    frame = qv.StringAttribute(default="unspecified")
    #: Inertial adam_core frame the axes were built from.
    reference_frame = qv.StringAttribute(default="unspecified")

    @classmethod
    def from_orbits(
        cls, orbits: "Orbits", frame: str = "VNC_ROTATING", mu: _Mu = None
    ) -> "LocalFrameCovariances":
        """Rotate the covariances of ``orbits`` into ``frame`` (name or alias)."""
        canonical = _canonical(frame)
        coords = orbits.coordinates
        if coords.covariance.is_all_nan():
            raise ValueError("The orbits carry no covariance.")
        rotated = apply_linear_covariance_transform(
            local_frame_jacobians(coords, canonical, mu), coords.covariance.to_matrix()
        )
        return cls.from_kwargs(
            orbit_id=orbits.orbit_id,
            object_id=orbits.object_id,
            time=coords.time,
            covariance=CoordinateCovariances.from_matrix(rotated),
            origin=coords.origin,
            frame=canonical,
            reference_frame=coords.frame,
        )
