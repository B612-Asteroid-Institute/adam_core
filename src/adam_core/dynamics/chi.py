from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

from .._rust.api import calc_chi_numpy as _rust_calc_chi_numpy
from ..constants import Constants as c
from ._rust_compat import (
    ScalarOrArray,
    as_rows,
    broadcast_to_rows,
    require_rust,
)

MU = c.MU
CHI_TYPES = Tuple[
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
]


@dataclass(frozen=True)
class ChiDiagnostics:
    dt: float
    mu: float
    r_norm: float
    v_norm: float
    alpha: float
    chi: float
    finite: bool


def _is_scalar_like(values: npt.ArrayLike) -> bool:
    return np.asarray(values, dtype=np.float64).ndim == 0


def _restore_chi_rows(rows: np.ndarray, *, scalar: bool) -> CHI_TYPES:
    if scalar:
        return tuple(np.float64(rows[0, i]) for i in range(7))  # type: ignore[return-value]
    return tuple(
        np.ascontiguousarray(rows[:, i], dtype=np.float64) for i in range(7)
    )  # type: ignore[return-value]


def calc_chi(
    r: npt.ArrayLike,
    v: npt.ArrayLike,
    dt: npt.ArrayLike,
    mu: npt.ArrayLike = MU,
    max_iter: int = 100,
    tol: float = 1e-16,
) -> CHI_TYPES:
    """
    Calculate universal anomaly chi and the first six Stumpff coefficients.

    This compatibility API is backed by ``adam_core_rs_coords::calc_chi``.
    """
    r_rows, r_single = as_rows(r, name="r", width=3)
    v_rows, v_single = as_rows(v, name="v", width=3)
    if v_rows.shape[0] != r_rows.shape[0]:
        raise ValueError("v must have the same row count as r")

    n = r_rows.shape[0]
    dts = broadcast_to_rows(dt, n, name="dt")
    mus = broadcast_to_rows(mu, n, name="mu")
    rows = require_rust(
        _rust_calc_chi_numpy(r_rows, v_rows, dts, mus, max_iter=max_iter, tol=tol),
        "dynamics.calc_chi",
    )
    scalar = r_single and v_single and _is_scalar_like(dt) and _is_scalar_like(mu)
    return _restore_chi_rows(rows, scalar=scalar)


def calc_chi_diagnostics(
    r: np.ndarray,
    v: np.ndarray,
    dt: float,
    mu: float = MU,
    max_iter: int = 100,
    tol: float = 1e-16,
) -> ChiDiagnostics:
    """Host-side chi diagnostics helper for fail-fast error reporting."""
    r_arr = np.asarray(r, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    r_norm = float(np.linalg.norm(r_arr))
    v_norm = float(np.linalg.norm(v_arr))
    alpha = float(-(v_norm**2) / mu + 2.0 / r_norm) if r_norm > 0 else np.nan
    chi = float(calc_chi(r_arr, v_arr, dt, mu=mu, max_iter=max_iter, tol=tol)[0])
    finite = bool(
        np.isfinite(r_norm)
        and np.isfinite(v_norm)
        and np.isfinite(alpha)
        and np.isfinite(chi)
    )
    return ChiDiagnostics(
        dt=float(dt),
        mu=float(mu),
        r_norm=r_norm,
        v_norm=v_norm,
        alpha=alpha,
        chi=chi,
        finite=finite,
    )
