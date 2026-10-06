"""Local orbital frames and covariances expressed in them.

A local orbital frame is built from a state vector. Its axes follow the
position, velocity and orbit normal of the body at each epoch. adam_core keeps
state vectors in inertial frames. The only quantity expressed in a local
orbital frame here is the covariance of a state, which is what navigation
consumers ask for when they want uncertainties split along the velocity,
across the orbit plane and in the remaining in-plane direction.

Frame names follow the SANA orbit-relative reference frames registry used by
the CCSDS navigation data messages. Three families are supported, each with
two variants.

* ``RSW``: x along the position vector, z along the orbital angular momentum,
  y completes the right-handed set. Also known as RTN and RIC.
* ``TNW``: x along the velocity vector, z along the orbital angular momentum,
  y completes the right-handed set.
* ``VNC``: x along the velocity vector, y along the orbital angular momentum,
  z completes the right-handed set. VNC and TNW share two axes and differ in
  the third, so one must never be labelled as the other.

The ``_INERTIAL`` variants treat the frame as fixed at the epoch. A covariance
is rotated block by block with the same 3x3 rotation on position and velocity.
The ``_ROTATING`` variants carry the frame angular velocity into the velocity
rows, so velocity uncertainties are relative to the rotating axes. The frame
rate is computed from two-body motion about the coordinate origin. The bare
names ``RSW``, ``RTN``, ``RIC``, ``TNW`` and ``VNC`` are aliases of the
``_INERTIAL`` variants, which is how the CCSDS orbit data messages use them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

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
    "LOCAL_ORBITAL_FRAME_ALIASES",
    "LocalFrameCovariances",
    "LocalOrbitalFrame",
    "local_frame_angular_velocity",
    "local_frame_jacobians",
    "local_frame_rotation_matrices",
    "resolve_local_orbital_frame",
]

LocalOrbitalFrame = Literal[
    "RSW_ROTATING",
    "RSW_INERTIAL",
    "TNW_ROTATING",
    "TNW_INERTIAL",
    "VNC_ROTATING",
    "VNC_INERTIAL",
]

#: Canonical frame names, as registered with SANA.
LOCAL_ORBITAL_FRAMES: tuple[str, ...] = (
    "RSW_ROTATING",
    "RSW_INERTIAL",
    "TNW_ROTATING",
    "TNW_INERTIAL",
    "VNC_ROTATING",
    "VNC_INERTIAL",
)

#: Bare names accepted as input. They resolve to the quasi-inertial variant,
#: matching how the CCSDS orbit data messages use RSW, RTN and TNW.
LOCAL_ORBITAL_FRAME_ALIASES: dict[str, str] = {
    "RSW": "RSW_INERTIAL",
    "RTN": "RSW_INERTIAL",
    "RIC": "RSW_INERTIAL",
    "TNW": "TNW_INERTIAL",
    "VNC": "VNC_INERTIAL",
}

#: Inertial adam_core frames a local orbital frame can be built from.
_INERTIAL_FRAMES = ("equatorial", "ecliptic")

#: Velocities below this magnitude in AU/day cannot define a frame axis.
_VELOCITY_TOLERANCE = 1e-20


def resolve_local_orbital_frame(frame: str) -> str:
    """
    Resolve a frame label or alias to its canonical SANA name.

    Parameters
    ----------
    frame : str
        Canonical name such as ``"VNC_ROTATING"`` or alias such as ``"RTN"``.
        Case insensitive.

    Returns
    -------
    str
        One of :data:`LOCAL_ORBITAL_FRAMES`.

    Raises
    ------
    ValueError
        If the label is not a known local orbital frame.
    """
    label = str(frame).strip().upper()
    if label in LOCAL_ORBITAL_FRAMES:
        return label
    if label in LOCAL_ORBITAL_FRAME_ALIASES:
        return LOCAL_ORBITAL_FRAME_ALIASES[label]
    raise ValueError(
        f"Unknown local orbital frame: {frame!r}. Expected one of "
        f"{list(LOCAL_ORBITAL_FRAMES)} or an alias in "
        f"{list(LOCAL_ORBITAL_FRAME_ALIASES)}."
    )


def _unit(vectors: np.ndarray, norms: np.ndarray) -> np.ndarray:
    return vectors / norms[:, None]


def _unit_derivative(
    vectors: np.ndarray, derivatives: np.ndarray, norms: np.ndarray
) -> np.ndarray:
    # d(u/|u|)/dt with u_hat = u/|u| is (du - (du . u_hat) u_hat) / |u|.
    unit = _unit(vectors, norms)
    radial_rate = np.einsum("ij,ij->i", derivatives, unit)
    return (derivatives - radial_rate[:, None] * unit) / norms[:, None]


def _check_inertial_frame(coords: CartesianCoordinates) -> None:
    if coords.frame not in _INERTIAL_FRAMES:
        raise ValueError(
            "Local orbital frames are built from states in an inertial frame. "
            f"Got frame {coords.frame!r}, expected one of {list(_INERTIAL_FRAMES)}."
        )


def _resolve_mu(
    coords: CartesianCoordinates, mu: Optional[Union[float, npt.ArrayLike]]
) -> np.ndarray:
    n = len(coords)
    if mu is None:
        try:
            return np.asarray(coords.origin.mu(), dtype=np.float64)
        except ValueError as exc:
            raise ValueError(
                "The frame rate of a _ROTATING local orbital frame needs the "
                "gravitational parameter of the origin, which is not known for "
                f"origin {coords.origin.code.unique().to_pylist()}. Pass mu= in "
                "AU^3/day^2 or use the _INERTIAL variant."
            ) from exc
    values = np.asarray(mu, dtype=np.float64)
    if values.ndim == 0:
        return np.full(n, float(values))
    if values.shape != (n,):
        raise ValueError(
            f"mu must be a scalar or have shape ({n},), got {values.shape}."
        )
    return values


def _basis_and_rates(
    coords: CartesianCoordinates,
    frame: str,
    mu: Optional[Union[float, npt.ArrayLike]] = None,
    with_rates: bool = True,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return the local frame basis rows and their time derivatives.

    Returns
    -------
    basis : `~numpy.ndarray` (N, 3, 3)
        Row ``i`` is unit axis ``i`` of the local frame in the inertial frame.
    rates : `~numpy.ndarray` (N, 3, 3) or None
        Time derivative of each row under two-body motion about the origin,
        in 1/day. None when ``with_rates`` is False, in which case ``mu`` is
        never resolved, so origins without a known gravitational parameter
        still get a basis.
    """
    canonical = resolve_local_orbital_frame(frame)
    _check_inertial_frame(coords)
    family = canonical.split("_")[0]

    r = np.asarray(coords.r, dtype=np.float64)
    v = np.asarray(coords.v, dtype=np.float64)
    h = np.cross(r, v)
    r_mag = np.linalg.norm(r, axis=1)
    v_mag = np.linalg.norm(v, axis=1)
    h_mag = np.linalg.norm(h, axis=1)

    finite = np.isfinite(r_mag) & np.isfinite(v_mag) & np.isfinite(h_mag)
    degenerate = finite & (
        (h_mag < SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE) | (v_mag < _VELOCITY_TOLERANCE)
    )
    if degenerate.any():
        row = int(np.flatnonzero(degenerate)[0])
        raise ValueError(
            f"State at row {row} has no defined orbit plane or velocity direction, "
            "so a local orbital frame cannot be built from it."
        )

    r_hat = _unit(r, r_mag)
    v_hat = _unit(v, v_mag)
    w_hat = _unit(h, h_mag)
    if family == "RSW":
        basis = np.stack([r_hat, np.cross(w_hat, r_hat), w_hat], axis=1)
    elif family == "TNW":
        basis = np.stack([v_hat, np.cross(w_hat, v_hat), w_hat], axis=1)
    elif family == "VNC":
        basis = np.stack([v_hat, w_hat, np.cross(v_hat, w_hat)], axis=1)
    else:  # pragma: no cover - guarded by resolve_local_orbital_frame
        raise ValueError(f"Unhandled local orbital frame family: {family}")
    if not with_rates:
        return basis, None

    mu_values = _resolve_mu(coords, mu)
    acceleration = -mu_values[:, None] * r / r_mag[:, None] ** 3
    h_rate = np.cross(r, acceleration)
    r_hat_rate = _unit_derivative(r, v, r_mag)
    v_hat_rate = _unit_derivative(v, acceleration, v_mag)
    w_hat_rate = _unit_derivative(h, h_rate, h_mag)

    if family == "RSW":
        e1, de1 = r_hat, r_hat_rate
        e3, de3 = w_hat, w_hat_rate
        de2 = np.cross(de3, e1) + np.cross(e3, de1)
    elif family == "TNW":
        e1, de1 = v_hat, v_hat_rate
        e3, de3 = w_hat, w_hat_rate
        de2 = np.cross(de3, e1) + np.cross(e3, de1)
    else:
        e1, de1 = v_hat, v_hat_rate
        e2, de2 = w_hat, w_hat_rate
        de3 = np.cross(de1, e2) + np.cross(e1, de2)

    rates = np.stack([de1, de2, de3], axis=1)
    return basis, rates


def local_frame_rotation_matrices(
    coords: CartesianCoordinates, frame: str
) -> npt.NDArray[np.float64]:
    """
    Rotation matrices from the inertial frame of ``coords`` to a local orbital frame.

    Parameters
    ----------
    coords : `~adam_core.coordinates.cartesian.CartesianCoordinates` (N)
        States in an inertial frame (``"equatorial"`` or ``"ecliptic"``).
    frame : str
        Local orbital frame name or alias, see :func:`resolve_local_orbital_frame`.

    Returns
    -------
    rotation : `~numpy.ndarray` (N, 3, 3)
        Row ``i`` is unit axis ``i`` of the local frame expressed in the
        inertial frame. ``rotation @ x`` maps an inertial vector into the
        local frame. The rotating and inertial variants share this matrix.
    """
    basis, _ = _basis_and_rates(coords, frame, with_rates=False)
    return basis


def local_frame_angular_velocity(
    coords: CartesianCoordinates,
    frame: str,
    mu: Optional[Union[float, npt.ArrayLike]] = None,
) -> npt.NDArray[np.float64]:
    """
    Angular velocity of a local orbital frame, in inertial components.

    The rate follows two-body motion about the coordinate origin. For a
    circular orbit it is the mean motion along the orbit normal for every
    family. For the ``_INERTIAL`` variants the result is zero, because those
    frames are treated as fixed at the epoch.

    Parameters
    ----------
    coords : `~adam_core.coordinates.cartesian.CartesianCoordinates` (N)
        States in an inertial frame.
    frame : str
        Local orbital frame name or alias.
    mu : float or `~numpy.ndarray` (N), optional
        Gravitational parameter in AU^3/day^2. Defaults to the origin's value.

    Returns
    -------
    omega : `~numpy.ndarray` (N, 3)
        Angular velocity in rad/day, inertial components.
    """
    canonical = resolve_local_orbital_frame(frame)
    if canonical.endswith("_INERTIAL"):
        _basis_and_rates(coords, canonical, with_rates=False)
        return np.zeros((len(coords), 3))
    basis, rates = _basis_and_rates(coords, canonical, mu)
    # For an orthonormal triad with de_i = omega x e_i, the sum of e_i x de_i
    # over the three axes equals 2 omega.
    return 0.5 * np.cross(basis, rates, axis=2).sum(axis=1)


def local_frame_jacobians(
    coords: CartesianCoordinates,
    frame: str,
    mu: Optional[Union[float, npt.ArrayLike]] = None,
) -> npt.NDArray[np.float64]:
    """
    6x6 Jacobians from inertial position and velocity to a local orbital frame.

    For the ``_INERTIAL`` variants the Jacobian is block diagonal with the same
    3x3 rotation on position and velocity. For the ``_ROTATING`` variants the
    velocity rows also carry the frame rate, because a velocity relative to a
    rotating frame is ``v - omega x r``. The Jacobian is then
    ``[[R, 0], [-R [omega]x, R]]`` where ``[omega]x`` is the cross-product
    matrix of the angular velocity.

    Parameters
    ----------
    coords : `~adam_core.coordinates.cartesian.CartesianCoordinates` (N)
        States in an inertial frame.
    frame : str
        Local orbital frame name or alias.
    mu : float or `~numpy.ndarray` (N), optional
        Gravitational parameter in AU^3/day^2 used for the frame rate.
        Defaults to the origin's value. Ignored for ``_INERTIAL`` variants.

    Returns
    -------
    jacobians : `~numpy.ndarray` (N, 6, 6)
        Jacobians such that a covariance in the local frame is
        ``J @ covariance @ J.T``.
    """
    canonical = resolve_local_orbital_frame(frame)
    rotation = local_frame_rotation_matrices(coords, canonical)
    n = len(coords)
    jacobians = np.zeros((n, 6, 6))
    jacobians[:, :3, :3] = rotation
    jacobians[:, 3:, 3:] = rotation
    if canonical.endswith("_ROTATING"):
        omega = local_frame_angular_velocity(coords, canonical, mu)
        skew = np.zeros((n, 3, 3))
        skew[:, 0, 1] = -omega[:, 2]
        skew[:, 0, 2] = omega[:, 1]
        skew[:, 1, 0] = omega[:, 2]
        skew[:, 1, 2] = -omega[:, 0]
        skew[:, 2, 0] = -omega[:, 1]
        skew[:, 2, 1] = omega[:, 0]
        jacobians[:, 3:, :3] = -np.einsum("nij,njk->nik", rotation, skew)
    return jacobians


class LocalFrameCovariances(qv.Table):
    """
    Per-epoch state covariances expressed in a local orbital frame.

    This is a covariance product, not a state representation. The state vector
    it describes stays in its inertial frame (for example a Sun-centered
    equatorial ``Orbits`` table) and is referenced by ``object_id``,
    ``orbit_id`` and ``time``. Rows are aligned one to one with the source
    orbits in the same order. Rows whose source covariance was missing stay
    NaN.

    Values are in AU and AU/day like every other covariance in adam_core.
    Use :meth:`to_matrix_km` for kilometre units.
    """

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    #: Epoch of the state the covariance belongs to, same scale as the source.
    time = Timestamp.as_column()
    #: 6x6 covariance in the local orbital frame, AU and AU/day.
    covariance = CoordinateCovariances.as_column()
    #: Center the orbit and the frame basis are referred to, for example SUN.
    origin = Origin.as_column()
    #: Local orbital frame, a canonical SANA name such as ``"VNC_ROTATING"``
    #: or, for covariances read from a file, the label the file carried.
    frame = qv.StringAttribute(default="unspecified")
    #: Inertial adam_core frame the basis was built from, ``"equatorial"``
    #: or ``"ecliptic"``.
    reference_frame = qv.StringAttribute(default="unspecified")

    @classmethod
    def from_orbits(
        cls,
        orbits: "Orbits",
        frame: str = "VNC_ROTATING",
        mu: Optional[Union[float, npt.ArrayLike]] = None,
    ) -> "LocalFrameCovariances":
        """
        Rotate the covariances of ``orbits`` into a local orbital frame.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            States with covariances in an inertial frame. Non gravitational
            9x9 covariance blocks are reduced to their 6x6 coordinate block.
        frame : str, optional
            Local orbital frame name or alias. Default ``"VNC_ROTATING"``.
        mu : float or `~numpy.ndarray` (N), optional
            Gravitational parameter in AU^3/day^2 for the frame rate of the
            ``_ROTATING`` variants. Defaults to the origin's value.

        Returns
        -------
        covariances : `LocalFrameCovariances` (N)
            One row per input row, in input order. The ``frame`` attribute is
            the canonical frame name.

        Raises
        ------
        ValueError
            If the orbits are not in an inertial frame, if a state is
            degenerate, or if every covariance is missing.
        """
        canonical = resolve_local_orbital_frame(frame)
        coords = orbits.coordinates
        if coords.covariance.is_all_nan():
            raise ValueError("Orbits carry no covariance to express in a local frame.")
        source = coords.covariance.to_matrix()
        missing = np.isnan(source).all(axis=(1, 2))
        jacobians = local_frame_jacobians(coords, canonical, mu)
        rotated = apply_linear_covariance_transform(
            jacobians, np.nan_to_num(source, nan=0.0)
        )
        rotated[missing] = np.nan
        return cls.from_kwargs(
            orbit_id=orbits.orbit_id,
            object_id=orbits.object_id,
            time=coords.time,
            covariance=CoordinateCovariances.from_matrix(rotated),
            origin=coords.origin,
            frame=canonical,
            reference_frame=coords.frame,
        )

    def to_matrix(self) -> npt.NDArray[np.float64]:
        """Covariance matrices as an (N, 6, 6) array in AU and AU/day."""
        return self.covariance.to_matrix()

    def to_matrix_km(self) -> npt.NDArray[np.float64]:
        """Covariance matrices as an (N, 6, 6) array in km and km/s."""
        return convert_cartesian_covariance_au_to_km(self.to_matrix())
