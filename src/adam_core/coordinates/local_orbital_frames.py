"""Covariances in the local orbital frames RSW (RTN, RIC), TNW and VNC."""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt
import quivr as qv

from .. import _rust_native
from ..time import Timestamp
from .cartesian import CartesianCoordinates
from .covariances import CoordinateCovariances
from .origin import Origin

_Mu = Optional[Union[float, npt.ArrayLike]]
if TYPE_CHECKING:
    from ..orbits.orbits import Orbits


def _kernel_inputs(coords: CartesianCoordinates, frame: str, mu: _Mu):
    if coords.frame not in ("equatorial", "ecliptic"):
        raise ValueError(
            f"Local orbital frames need an inertial frame, got {coords.frame!r}."
        )
    canonical = _rust_native.local_frame_canonical_name(frame)
    if mu is None:
        mu = coords.origin.mu() if canonical.endswith("_ROTATING") else 0.0
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (len(coords),))
    return np.ascontiguousarray(coords.values), np.ascontiguousarray(mu), canonical


def local_frame_jacobians(
    coords: CartesianCoordinates, frame: str, mu: _Mu = None
) -> npt.NDArray[np.float64]:
    """(N, 6, 6) Jacobians J into ``frame``; mu (AU^3/day^2) defaults to the origin's."""
    return _rust_native.local_frame_jacobians_numpy(*_kernel_inputs(coords, frame, mu))


class LocalFrameCovariances(qv.Table):
    """Covariances in a local orbital frame (AU, AU/day), one row per orbit."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    time = Timestamp.as_column()
    covariance = CoordinateCovariances.as_column()
    origin = Origin.as_column()
    frame = qv.StringAttribute(default="unspecified")
    inertial_frame = qv.StringAttribute(default="unspecified")

    @classmethod
    def from_orbits(
        cls, orbits: "Orbits", frame: str = "VNC_ROTATING", mu: _Mu = None
    ) -> "LocalFrameCovariances":
        """Rotate the covariances of ``orbits`` into ``frame``, e.g. ``TNW_INERTIAL``."""
        coords = orbits.coordinates
        if coords.covariance.is_all_nan():
            raise ValueError("The orbits carry no covariance.")
        values, mu, canonical = _kernel_inputs(coords, frame, mu)
        rotated = _rust_native.local_frame_covariances_numpy(
            values, np.ascontiguousarray(coords.covariance.to_matrix()), mu, canonical
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
