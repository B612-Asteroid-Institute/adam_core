from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

from .._rust.api import (
    add_light_time_numpy as _rust_add_light_time_numpy,
    add_stellar_aberration_numpy as _rust_add_stellar_aberration_numpy,
)
from ..constants import Constants as c
from ._rust_compat import require_rust

MU = c.MU


def _require_rust_light_time(
    orbits: np.ndarray,
    observer_positions: np.ndarray,
    mus: np.ndarray,
    lt_tol: float,
    max_iter: int,
    tol: float,
    max_lt_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    result = _rust_add_light_time_numpy(
        orbits,
        observer_positions,
        mus,
        lt_tol=lt_tol,
        max_iter=max_iter,
        tol=tol,
        max_lt_iter=max_lt_iter,
    )
    if result is None:
        raise RuntimeError(
            "adam_core._rust_native is required for light-time correction"
        )
    return result


def _as_state_rows(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (6,):
            raise ValueError(f"{name} must have shape (6,) or (N, 6)")
        arr = arr.reshape(1, 6)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"{name} must have shape (6,) or (N, 6)")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_position_rows(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(f"{name} must have shape (3,) or (N, 3)")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (N, 3)")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _broadcast_mu(mu: npt.ArrayLike, n: int) -> np.ndarray:
    arr = np.asarray(mu, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"mu must be scalar or have shape ({n},)")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _add_light_time(
    orbit: npt.ArrayLike,
    t0: float,
    observer_position: npt.ArrayLike,
    lt_tol: float = 1e-10,
    mu: float = MU,
    max_iter: int = 1000,
    tol: float = 1e-15,
    max_lt_iter: int = 10,
) -> Tuple[np.ndarray, np.float64]:
    """
    Compatibility wrapper for the historical single-row light-time helper.

    The current implementation delegates to the mandatory Rust backend. `t0`
    is retained for signature compatibility; the correction depends only on
    the state and the solved light-time interval.
    """
    _ = t0
    orbits = _as_state_rows(orbit, name="orbit")
    observer_positions = _as_position_rows(observer_position, name="observer_position")
    mus = _broadcast_mu(mu, 1)
    corrected, light_time = _require_rust_light_time(
        orbits,
        observer_positions,
        mus,
        lt_tol,
        max_iter,
        tol,
        max_lt_iter,
    )
    return corrected[0], np.float64(light_time[0])


def _add_light_time_vmap(
    orbits: npt.ArrayLike,
    t0: npt.ArrayLike,
    observer_positions: npt.ArrayLike,
    lt_tol: float = 1e-10,
    mu: npt.ArrayLike = MU,
    max_iter: int = 1000,
    tol: float = 1e-15,
    max_lt_iter: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compatibility wrapper for the historical vmapped light-time helper.
    """
    return add_light_time(
        orbits,
        t0,
        observer_positions,
        lt_tol=lt_tol,
        mu=mu,
        max_iter=max_iter,
        tol=tol,
        max_lt_iter=max_lt_iter,
    )


def add_light_time(
    orbits: npt.ArrayLike,
    t0: npt.ArrayLike,
    observer_positions: npt.ArrayLike,
    lt_tol: float = 1e-10,
    mu: npt.ArrayLike = MU,
    max_iter: int = 1000,
    tol: float = 1e-15,
    max_lt_iter: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply light-time correction to barycentric Cartesian states.

    This preserves the baseline public module path while routing the actual
    solve through the Rust light-time kernel used by production ephemeris code.
    `t0` is accepted for API compatibility.
    """
    _ = t0
    orbits_arr = _as_state_rows(orbits, name="orbits")
    observer_arr = _as_position_rows(observer_positions, name="observer_positions")
    if observer_arr.shape[0] != orbits_arr.shape[0]:
        raise ValueError("observer_positions must have the same row count as orbits")
    mus = _broadcast_mu(mu, orbits_arr.shape[0])
    return _require_rust_light_time(
        orbits_arr,
        observer_arr,
        mus,
        lt_tol,
        max_iter,
        tol,
        max_lt_iter,
    )


def add_stellar_aberration(
    orbits: npt.ArrayLike,
    observer_states: npt.ArrayLike,
) -> np.ndarray:
    """
    Apply stellar aberration to topocentric position vectors.

    This compatibility API is backed by
    ``adam_core_rs_coords::apply_stellar_aberration_row``. The velocity
    components are not returned, matching the historical public helper.
    """
    orbits_arr = _as_state_rows(orbits, name="orbits")
    observer_arr = _as_state_rows(observer_states, name="observer_states")
    if observer_arr.shape[0] != orbits_arr.shape[0]:
        raise ValueError("observer_states must have the same row count as orbits")
    return require_rust(
        _rust_add_stellar_aberration_numpy(orbits_arr, observer_arr),
        "dynamics.add_stellar_aberration",
    )


__all__ = [
    "_add_light_time",
    "_add_light_time_vmap",
    "add_light_time",
    "add_stellar_aberration",
]
