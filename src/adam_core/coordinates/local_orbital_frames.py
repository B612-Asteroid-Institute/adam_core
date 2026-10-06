"""Local orbital frames and covariances expressed in them.

Frame names follow the SANA orbit-relative reference frames registry used by
the CCSDS navigation data messages.

* ``RSW``: x along the position, z along the orbital angular momentum, y
  completes the right-handed set. Also known as RTN and RIC.
* ``TNW``: x along the velocity, z along the orbital angular momentum, y
  completes the right-handed set.
* ``VNC``: x along the velocity, y along the orbital angular momentum, z
  completes the right-handed set.

VNC and TNW share two axes and differ in the third, so one must never be
labelled as the other.

The ``_INERTIAL`` variants are fixed at the epoch, so a covariance is rotated
with the same 3x3 rotation on position and velocity. The ``_ROTATING``
variants carry the frame angular velocity into the velocity rows, with the
rate taken from two-body motion about the coordinate origin. The bare names
``RSW``, ``RTN``, ``RIC``, ``TNW`` and ``VNC`` resolve to the ``_INERTIAL``
variants, as the CCSDS orbit data messages use them.

Only covariances are expressed in these frames. State vectors stay inertial.
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
from .units import convert_cartesian_covariance_au_to_km

if TYPE_CHECKING:
    from ..orbits.orbits import Orbits

__all__ = [
    "LOCAL_ORBITAL_FRAMES",
    "LocalFrameCovariances",
    "local_frame_angular_velocity",
    "local_frame_jacobians",
    "local_frame_rotation_matrices",
    "resolve_local_orbital_frame",
]

#: Canonical frame names, as registered with SANA.
LOCAL_ORBITAL_FRAMES = (
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
_INERTIAL_FRAMES = ("equatorial", "ecliptic")

_Mu = Optional[Union[float, npt.ArrayLike]]


def resolve_local_orbital_frame(frame: str) -> str:
    """Canonical SANA name for a frame name or alias, case insensitive."""
    label = str(frame).strip().upper()
    label = _ALIASES.get(label, label)
    if label not in LOCAL_ORBITAL_FRAMES:
        raise ValueError(
            f"Unknown local orbital frame: {frame!r}. Expected one of "
            f"{list(LOCAL_ORBITAL_FRAMES)} or an alias in {list(_ALIASES)}."
        )
    return label


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def _unit_rate(vectors: np.ndarray, rates: np.ndarray) -> np.ndarray:
    # d(u/|u|)/dt = (du - (du . u_hat) u_hat) / |u|
    norms = np.linalg.norm(vectors, axis=1)[:, None]
    unit = vectors / norms
    return (rates - np.einsum("ij,ij->i", rates, unit)[:, None] * unit) / norms


def _state_vectors(
    coords: CartesianCoordinates, frame: str
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the inputs and return the frame family with r, v and r x v."""
    family = resolve_local_orbital_frame(frame).split("_")[0]
    if coords.frame not in _INERTIAL_FRAMES:
        raise ValueError(
            "Local orbital frames are built from states in an inertial frame. "
            f"Got frame {coords.frame!r}, expected one of {list(_INERTIAL_FRAMES)}."
        )
    r = np.asarray(coords.r, dtype=np.float64)
    v = np.asarray(coords.v, dtype=np.float64)
    h = np.cross(r, v)
    degenerate = np.linalg.norm(h, axis=1) < SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE
    if degenerate.any():
        raise ValueError(
            f"State at row {int(np.flatnonzero(degenerate)[0])} has no defined "
            "orbit plane, so a local orbital frame cannot be built from it."
        )
    return family, r, v, h


def _axes(family: str, r_hat, v_hat, w_hat) -> np.ndarray:
    if family == "RSW":
        return np.stack([r_hat, np.cross(w_hat, r_hat), w_hat], axis=1)
    if family == "TNW":
        return np.stack([v_hat, np.cross(w_hat, v_hat), w_hat], axis=1)
    return np.stack([v_hat, w_hat, np.cross(v_hat, w_hat)], axis=1)


def _axis_rates(family: str, w_hat, dr_hat, dv_hat) -> np.ndarray:
    # Under two-body motion the orbit normal is fixed, so only the position
    # and velocity directions turn.
    fixed = np.zeros_like(w_hat)
    if family == "RSW":
        return np.stack([dr_hat, np.cross(w_hat, dr_hat), fixed], axis=1)
    if family == "TNW":
        return np.stack([dv_hat, np.cross(w_hat, dv_hat), fixed], axis=1)
    return np.stack([dv_hat, fixed, np.cross(dv_hat, w_hat)], axis=1)


def _resolve_mu(coords: CartesianCoordinates, mu: _Mu) -> np.ndarray:
    if mu is not None:
        return np.broadcast_to(np.asarray(mu, dtype=np.float64), (len(coords),))
    try:
        return np.asarray(coords.origin.mu(), dtype=np.float64)
    except ValueError as exc:
        raise ValueError(
            "The frame rate of a _ROTATING frame needs the gravitational "
            "parameter of the origin, which is not known for "
            f"{coords.origin.code.unique().to_pylist()}. Pass mu= in AU^3/day^2 "
            "or use the _INERTIAL variant."
        ) from exc


def local_frame_rotation_matrices(
    coords: CartesianCoordinates, frame: str
) -> npt.NDArray[np.float64]:
    """
    Rotation matrices from the inertial frame of ``coords`` to a local orbital frame.

    Parameters
    ----------
    coords : `~adam_core.coordinates.cartesian.CartesianCoordinates` (N)
        States in the ``"equatorial"`` or ``"ecliptic"`` frame.
    frame : str
        Frame name or alias, see :func:`resolve_local_orbital_frame`.

    Returns
    -------
    rotation : `~numpy.ndarray` (N, 3, 3)
        Row ``i`` is unit axis ``i`` of the local frame in the inertial frame,
        so ``rotation @ x`` maps an inertial vector into the local frame. The
        rotating and inertial variants share this matrix.
    """
    family, r, v, h = _state_vectors(coords, frame)
    return _axes(family, _unit(r), _unit(v), _unit(h))


def local_frame_angular_velocity(
    coords: CartesianCoordinates, frame: str, mu: _Mu = None
) -> npt.NDArray[np.float64]:
    """
    Angular velocity of a local orbital frame in inertial components, rad/day.

    The rate follows two-body motion about the coordinate origin, with ``mu``
    in AU^3/day^2 taken from the origin unless given. It is zero for the
    ``_INERTIAL`` variants.
    """
    canonical = resolve_local_orbital_frame(frame)
    family, r, v, h = _state_vectors(coords, canonical)
    if canonical.endswith("_INERTIAL"):
        return np.zeros((len(coords), 3))
    acceleration = (
        -_resolve_mu(coords, mu)[:, None] * r / np.linalg.norm(r, axis=1)[:, None] ** 3
    )
    w_hat = _unit(h)
    axes = _axes(family, _unit(r), _unit(v), w_hat)
    rates = _axis_rates(family, w_hat, _unit_rate(r, v), _unit_rate(v, acceleration))
    # For an orthonormal triad with de_i = omega x e_i, sum_i e_i x de_i = 2 omega.
    return 0.5 * np.cross(axes, rates, axis=2).sum(axis=1)


def local_frame_jacobians(
    coords: CartesianCoordinates, frame: str, mu: _Mu = None
) -> npt.NDArray[np.float64]:
    """
    6x6 Jacobians from inertial position and velocity to a local orbital frame.

    A covariance in the local frame is ``J @ covariance @ J.T``. For the
    ``_INERTIAL`` variants ``J`` is block diagonal with the rotation on both
    blocks. For the ``_ROTATING`` variants a velocity relative to the frame is
    ``v - omega x r``, so the velocity rows also depend on position.
    """
    canonical = resolve_local_orbital_frame(frame)
    rotation = local_frame_rotation_matrices(coords, canonical)
    jacobians = np.zeros((len(coords), 6, 6))
    jacobians[:, :3, :3] = rotation
    jacobians[:, 3:, 3:] = rotation
    if canonical.endswith("_ROTATING"):
        omega = local_frame_angular_velocity(coords, canonical, mu)
        jacobians[:, 3:, :3] = np.cross(omega[:, None, :], rotation)
    return jacobians


class LocalFrameCovariances(qv.Table):
    """
    Per-epoch state covariances expressed in a local orbital frame.

    A covariance product, not a state representation. The states it describes
    stay in their inertial frame and are referenced by ``object_id``,
    ``orbit_id`` and ``time``. Rows are aligned one to one with the source
    orbits, in the same order. Rows without a source covariance stay NaN.
    Values are in AU and AU/day, see :meth:`to_matrix_km` for kilometres.
    """

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    time = Timestamp.as_column()
    covariance = CoordinateCovariances.as_column()
    #: Center the orbit and the frame basis are referred to, for example SUN.
    origin = Origin.as_column()
    #: Canonical frame name such as ``"VNC_ROTATING"``, or the label a file
    #: carried when read from an OEM.
    frame = qv.StringAttribute(default="unspecified")
    #: Inertial adam_core frame the basis was built from.
    reference_frame = qv.StringAttribute(default="unspecified")

    @classmethod
    def from_orbits(
        cls, orbits: "Orbits", frame: str = "VNC_ROTATING", mu: _Mu = None
    ) -> "LocalFrameCovariances":
        """
        Rotate the covariances of ``orbits`` into a local orbital frame.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            States with covariances in an inertial frame. Extended 9x9
            covariances are reduced to their 6x6 coordinate block.
        frame : str, optional
            Frame name or alias. Default ``"VNC_ROTATING"``.
        mu : float or `~numpy.ndarray` (N), optional
            Gravitational parameter in AU^3/day^2 for the frame rate of the
            ``_ROTATING`` variants. Defaults to the origin's value.
        """
        canonical = resolve_local_orbital_frame(frame)
        coords = orbits.coordinates
        if coords.covariance.is_all_nan():
            raise ValueError("Orbits carry no covariance to express in a local frame.")
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

    def to_matrix_km(self) -> npt.NDArray[np.float64]:
        """Covariance matrices as an (N, 6, 6) array in km and km/s."""
        return convert_cartesian_covariance_au_to_km(self.covariance.to_matrix())
