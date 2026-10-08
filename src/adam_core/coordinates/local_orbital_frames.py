"""Local orbital frames (SANA orbit-relative registry names) for covariances.

RSW (RTN, RIC): x along position, z along the orbital angular momentum h.
TNW: x along velocity, z along h. VNC: x along velocity, y along h. The third
axis completes each right-handed set, so VNC is TNW with rows (T, W, -N) and the
two must never be labelled as each other. ``_INERTIAL`` variants rotate position
and velocity alike, ``_ROTATING`` variants add the two-body frame rate to the
velocity rows. Bare names resolve to ``_INERTIAL``. Only covariances are
expressed here. State vectors stay inertial.

The Jacobian and the product ``J @ C @ J.T`` are evaluated in double-double
arithmetic (pairs of floats carrying about 32 digits), because the rotating
frame velocity rows are small differences of large terms and lose two to three
digits in plain double precision. The results are rounded once, at the end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt
import quivr as qv

from ..time import Timestamp
from .cartesian import SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE, CartesianCoordinates
from .covariances import CoordinateCovariances
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


# Double-double arithmetic on arrays: a value is a (hi, lo) pair of float64 arrays
# with hi + lo exact to about 1e-32 relative. Error free transformations after
# Dekker and Knuth as in the QD library (Hida, Li, Bailey). No fused multiply-add
# is assumed, so every platform gives the same bits.
_SPLITTER = 134217729.0  # 2**27 + 1


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _quick_two_sum(a, b):  # |a| >= |b|
    s = a + b
    return s, b - (s - a)


def _two_prod(a, b):
    p = a * b
    ca, cb = _SPLITTER * a, _SPLITTER * b
    ah = ca - (ca - a)
    bh = cb - (cb - b)
    al, bl = a - ah, b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _dd(a):
    a = np.asarray(a, dtype=np.float64)
    return a, np.zeros_like(a)


def _add(x, y):
    s, e = _two_sum(x[0], y[0])
    return _quick_two_sum(s, e + (x[1] + y[1]))


def _neg(x):
    return -x[0], -x[1]


def _sub(x, y):
    return _add(x, _neg(y))


def _mul(x, y):
    p, e = _two_prod(x[0], y[0])
    return _quick_two_sum(p, e + (x[0] * y[1] + x[1] * y[0]))


def _div(x, y):
    q1 = x[0] / y[0]
    r = _sub(x, _mul(_dd(q1), y))
    q2 = r[0] / y[0]
    r = _sub(r, _mul(_dd(q2), y))
    return _add(_quick_two_sum(q1, q2), _dd(r[0] / y[0]))


def _sqrt(x):
    s = np.sqrt(x[0])
    r = _sub(x, _mul(_dd(s), _dd(s)))
    return _quick_two_sum(s, r[0] / (2.0 * s))


def _dot(x, y):  # (N, 3) pairs -> (N,) pair
    acc = _mul((x[0][:, 0], x[1][:, 0]), (y[0][:, 0], y[1][:, 0]))
    for k in (1, 2):
        acc = _add(acc, _mul((x[0][:, k], x[1][:, k]), (y[0][:, k], y[1][:, k])))
    return acc


def _cross(x, y):  # (N, 3) pairs -> (N, 3) pair
    hi, lo = np.empty_like(x[0]), np.empty_like(x[0])
    for k, (i, j) in enumerate(((1, 2), (2, 0), (0, 1))):
        c = _sub(
            _mul((x[0][:, i], x[1][:, i]), (y[0][:, j], y[1][:, j])),
            _mul((x[0][:, j], x[1][:, j]), (y[0][:, i], y[1][:, i])),
        )
        hi[:, k], lo[:, k] = c
    return hi, lo


def _scale(x, s):  # (N, 3) pair times (N,) pair
    return _mul(x, (s[0][:, None], s[1][:, None]))


def _unit(x):
    norm = _sqrt(_dot(x, x))
    return _div(x, (norm[0][:, None], norm[1][:, None])), norm


def _jacobian_dd(coords: CartesianCoordinates, frame: str, mu: _Mu):
    """(hi, lo) pair of (N, 6, 6) Jacobians from inertial position and velocity
    (AU, AU/day) to the canonical local frame ``frame``."""
    if coords.frame not in ("equatorial", "ecliptic"):
        raise ValueError(
            f"Local orbital frames need an inertial frame, got {coords.frame!r}."
        )
    degenerate = coords.h_mag < SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE
    if degenerate.any():
        raise ValueError(
            f"State {int(np.flatnonzero(degenerate)[0])} has no orbit plane."
        )
    family, kind = frame.split("_")
    r, v = _dd(coords.r), _dd(coords.v)
    r_hat, r_mag = _unit(r)
    v_hat, v_mag = _unit(v)
    h_hat, _ = _unit(_cross(r, v))
    if family == "RSW":
        rows = (r_hat, _cross(h_hat, r_hat), h_hat)
    elif family == "TNW":
        rows = (v_hat, _cross(h_hat, v_hat), h_hat)
    else:
        rows = (v_hat, h_hat, _cross(v_hat, h_hat))

    n = len(coords)
    hi, lo = np.zeros((n, 6, 6)), np.zeros((n, 6, 6))
    for i, row in enumerate(rows):
        hi[:, i, :3], lo[:, i, :3] = row
        hi[:, i + 3, 3:], lo[:, i + 3, 3:] = row
    if kind == "INERTIAL":
        return hi, lo

    if mu is None:
        try:
            mu = coords.origin.mu()
        except ValueError as exc:
            raise ValueError(
                f"No gravitational parameter for {coords.origin.code.unique().to_pylist()}. "
                "Pass mu= in AU^3/day^2 or use the _INERTIAL variant."
            ) from exc
    mu = _dd(np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,)))
    # Under two-body motion only r_hat and v_hat turn, the orbit normal does not.
    # d(u/|u|)/dt = (du/dt - (du/dt . u_hat) u_hat) / |u|.
    if family == "RSW":
        x_hat, x_mag, x_dot = r_hat, r_mag, v
    else:
        r_cubed = _mul(r_mag, _mul(r_mag, r_mag))
        x_hat, x_mag, x_dot = v_hat, v_mag, _neg(_scale(r, _div(mu, r_cubed)))
    x_rate = _div(
        _sub(x_dot, _scale(x_hat, _dot(x_dot, x_hat))),
        (x_mag[0][:, None], x_mag[1][:, None]),
    )
    y_rate = _cross(h_hat, x_rate)  # rate of the axis completing (x, h_hat)
    zero = _dd(np.zeros((n, 3)))
    rates = (x_rate, zero, _neg(y_rate)) if family == "VNC" else (x_rate, y_rate, zero)
    # For an orthonormal triad with de_i = omega x e_i, sum_i e_i x de_i = 2 omega.
    omega = _cross(rows[0], rates[0])
    for row, rate in zip(rows[1:], rates[1:]):
        omega = _add(omega, _cross(row, rate))
    omega = _scale(omega, _dd(np.full(n, 0.5)))
    for i, row in enumerate(rows):
        hi[:, i + 3, :3], lo[:, i + 3, :3] = _cross(omega, row)
    # omega is along the orbit normal, so that row of the rate block is exactly zero.
    normal_row = 1 if family == "VNC" else 2
    hi[:, normal_row + 3, :3] = lo[:, normal_row + 3, :3] = 0.0
    return hi, lo


def _rotate_covariance_dd(jacobian, covariances: np.ndarray) -> np.ndarray:
    """``J @ C @ J.T`` for a double-double ``jacobian`` and float64 ``covariances``
    (N, 6, 6), accumulated in double-double and rounded once. The input is
    symmetrised exactly first (stored covariances carry ulp level asymmetry that
    the cancellation in the velocity rows would amplify) and the result is
    symmetric to the bit."""
    j_hi, j_lo = jacobian
    c = _mul(
        _add(_dd(covariances), _dd(np.swapaxes(covariances, 1, 2))),
        _dd(np.full(covariances.shape, 0.5)),
    )

    def product(a, b):  # (N, 6, 6) pairs, a @ b
        acc = _mul(
            (a[0][:, :, 0, None], a[1][:, :, 0, None]),
            (b[0][:, None, 0, :], b[1][:, None, 0, :]),
        )
        for k in range(1, 6):
            acc = _add(
                acc,
                _mul(
                    (a[0][:, :, k, None], a[1][:, :, k, None]),
                    (b[0][:, None, k, :], b[1][:, None, k, :]),
                ),
            )
        return acc

    jc = product((j_hi, j_lo), c)
    hi, lo = product(jc, (np.swapaxes(j_hi, 1, 2), np.swapaxes(j_lo, 1, 2)))
    rotated = np.triu(hi + lo)
    return rotated + np.swapaxes(np.triu(rotated, 1), 1, 2)


def local_frame_jacobians(
    coords: CartesianCoordinates, frame: str, mu: _Mu = None
) -> npt.NDArray[np.float64]:
    """
    (N, 6, 6) Jacobians from inertial position and velocity to a local orbital
    frame (name or alias), so a covariance there is ``J @ C @ J.T``. The top
    left block is the rotation. ``mu`` (AU^3/day^2, default the origin's) sets
    the ``_ROTATING`` frame rate. Correctly rounded from a double-double evaluation.
    """
    hi, lo = _jacobian_dd(coords, _canonical_frame(frame), mu)
    return hi + lo


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
        rotated = _rotate_covariance_dd(
            _jacobian_dd(coords, canonical_frame, mu), coords.covariance.to_matrix()
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
