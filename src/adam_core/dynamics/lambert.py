"""
Lambert's problem — public API.

The full Izzo solver lives in Rust (`adam_core._rust_native.izzo_lambert_numpy`).
This module is a thin Python surface over that kernel.
"""

from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from ..constants import Constants as C

MU = C.MU


def izzo_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float = MU,
    M: int = 0,
    prograde: bool = True,
    low_path: bool = True,
    maxiter: int = 35,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve one Lambert problem through the shared Rust Izzo kernel."""
    from .._rust.api import izzo_lambert_numpy

    values_r1 = np.asarray(r1, dtype=np.float64)
    values_r2 = np.asarray(r2, dtype=np.float64)
    velocities_1, velocities_2 = izzo_lambert_numpy(
        values_r1.reshape(1, 3),
        values_r2.reshape(1, 3),
        np.array([tof], dtype=np.float64),
        float(mu),
        int(M),
        bool(prograde),
        bool(low_path),
        int(maxiter),
        float(atol),
        float(rtol),
    )
    return velocities_1[0], velocities_2[0]


def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: Union[np.ndarray, float],
    mu: float = MU,
    prograde: bool = True,
    max_iter: int = 35,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Lambert's problem for multiple initial and final positions and times of flight.

    Parameters
    ----------
    r1 : array_like (N, 3)
        Initial position vectors in au.
    r2 : array_like (N, 3)
        Final position vectors in au.
    tof : array_like (N) or float
        Times of flight in days.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of au³/day².
    prograde : bool, optional
        If True, assume prograde motion. If False, assume retrograde motion.
    max_iter : int, optional
        Maximum number of iterations for convergence.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    v1 : ndarray (N, 3)
        Initial velocity vectors in au/day with origin at the attractor
    v2 : ndarray (N, 3)
        Final velocity vectors in au/day with origin at the attractor
    """
    from .._rust.api import izzo_lambert_numpy

    r1_np = np.asarray(r1, dtype=np.float64)
    r2_np = np.asarray(r2, dtype=np.float64)
    if r1_np.ndim == 1:
        r1_np = r1_np.reshape(1, -1)
    if r2_np.ndim == 1:
        r2_np = r2_np.reshape(1, -1)
    if isinstance(tof, (int, float)):
        tof_np = np.full(r1_np.shape[0], float(tof), dtype=np.float64)
    else:
        tof_np = np.asarray(tof, dtype=np.float64)

    result = izzo_lambert_numpy(
        r1_np,
        r2_np,
        tof_np,
        float(mu),
        0,
        bool(prograde),
        True,
        int(max_iter),
        float(tol),
        float(tol),
    )
    return result


def calculate_c3(v1: np.ndarray, body_v: np.ndarray) -> npt.NDArray[np.float64]:
    """
    Calculate the C3 of a spacecraft given its velocity relative to a body.

    Parameters
    ----------
    v1 : array_like (N, 3)
        Velocity of the spacecraft in au/d.
    body_v : array_like (N, 3)
        Velocity of the body in au/d.

    Returns
    -------
    c3 : array_like (N)
        C3 of the spacecraft in au²/d².
    """
    from adam_core import _rust_native

    return np.asarray(
        _rust_native.calculate_c3_numpy(
            np.ascontiguousarray(np.asarray(v1, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(body_v, dtype=np.float64)),
        ),
        dtype=np.float64,
    )
