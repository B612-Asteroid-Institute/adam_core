"""Local orbital frames (SANA orbit-relative registry names) for covariances.

RSW (RTN, RIC): x along position, z along the orbital angular momentum h.
TNW: x along velocity, z along h. VNC: x along velocity, y along h. Bare names
resolve to ``_INERTIAL``, ``_ROTATING`` adds the two-body frame rate to the
velocity rows. The Jacobians and the exact covariance product live in Rust.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt
import quivr as qv

from adam_core import _rust_native

from ..time import Timestamp
from .cartesian import CartesianCoordinates
from .covariances import CoordinateCovariances
from .origin import Origin

if TYPE_CHECKING:
    from ..orbits.orbits import Orbits

_Mu = Optional[Union[float, npt.ArrayLike]]


def _kernel_inputs(coords: CartesianCoordinates, frame: str, mu: _Mu):
    """Canonical frame name, (N, 6) values and (N,) mu for the Rust kernels."""
    if coords.frame not in ("equatorial", "ecliptic"):
        raise ValueError(
            f"Local orbital frames need an inertial frame, got {coords.frame!r}."
        )
    canonical = _rust_native.local_frame_canonical_name(frame)
    if mu is None:
        mu = coords.origin.mu() if canonical.endswith("_ROTATING") else 0.0
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (len(coords),))
    values = np.ascontiguousarray(coords.values, dtype=np.float64)
    return canonical, values, np.ascontiguousarray(mu)


def local_frame_jacobians(
    coords: CartesianCoordinates, frame: str, mu: _Mu = None
) -> npt.NDArray[np.float64]:
    """
    (N, 6, 6) Jacobians from inertial position and velocity to a local orbital
    frame (name or alias), so a covariance there is ``J @ C @ J.T``. ``mu``
    (AU^3/day^2, default the origin's) sets the ``_ROTATING`` frame rate.
    """
    canonical, values, mu = _kernel_inputs(coords, frame, mu)
    return _rust_native.local_frame_jacobians_numpy(values, mu, canonical)


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
        coords = orbits.coordinates
        if coords.covariance.is_all_nan():
            raise ValueError("The orbits carry no covariance.")
        canonical, values, mu = _kernel_inputs(coords, frame, mu)
        rotated = _rust_native.local_frame_covariances_numpy(
            values,
            np.ascontiguousarray(coords.covariance.to_matrix(), dtype=np.float64),
            mu,
            canonical,
        )
        return cls.from_kwargs(
            orbit_id=orbits.orbit_id,
            object_id=orbits.object_id,
            time=coords.time,
            covariance=CoordinateCovariances.from_matrix(rotated),
            origin=coords.origin,
            frame=canonical,
            inertial_frame=coords.frame,
        )
