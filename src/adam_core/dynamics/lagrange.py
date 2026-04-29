from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

from .._rust.api import (
    apply_lagrange_coefficients_numpy as _rust_apply_lagrange_coefficients_numpy,
    calc_lagrange_coefficients_numpy as _rust_calc_lagrange_coefficients_numpy,
)
from ..constants import Constants as C
from ._rust_compat import (
    ScalarOrArray,
    as_rows,
    broadcast_to_rows,
    require_rust,
)
from .stumpff import STUMPFF_TYPES

MU = C.MU
LAGRANGE_TYPES = Tuple[ScalarOrArray, ScalarOrArray, ScalarOrArray, ScalarOrArray]


def _is_scalar_like(values: npt.ArrayLike) -> bool:
    return np.asarray(values, dtype=np.float64).ndim == 0


def _restore_rows(
    rows: np.ndarray, width: int, *, scalar: bool
) -> tuple[ScalarOrArray, ...]:
    if scalar:
        return tuple(np.float64(rows[0, i]) for i in range(width))
    return tuple(
        np.ascontiguousarray(rows[:, i], dtype=np.float64) for i in range(width)
    )


def calc_lagrange_coefficients(
    r: npt.ArrayLike,
    v: npt.ArrayLike,
    dt: npt.ArrayLike,
    mu: npt.ArrayLike = MU,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> tuple[LAGRANGE_TYPES, STUMPFF_TYPES, ScalarOrArray]:
    """
    Calculate Lagrange coefficients for propagating an initial state by ``dt``.

    This compatibility API is backed by
    ``adam_core_rs_coords::calc_lagrange_coefficients``.
    """
    r_rows, r_single = as_rows(r, name="r", width=3)
    v_rows, v_single = as_rows(v, name="v", width=3)
    if v_rows.shape[0] != r_rows.shape[0]:
        raise ValueError("v must have the same row count as r")

    n = r_rows.shape[0]
    dts = broadcast_to_rows(dt, n, name="dt")
    mus = broadcast_to_rows(mu, n, name="mu")
    coeffs, stumpff, chi = require_rust(
        _rust_calc_lagrange_coefficients_numpy(
            r_rows,
            v_rows,
            dts,
            mus,
            max_iter=max_iter,
            tol=tol,
        ),
        "dynamics.calc_lagrange_coefficients",
    )
    scalar = r_single and v_single and _is_scalar_like(dt) and _is_scalar_like(mu)
    coeffs_out = _restore_rows(coeffs, 4, scalar=scalar)
    stumpff_out = _restore_rows(stumpff, 6, scalar=scalar)
    chi_out: ScalarOrArray
    if scalar:
        chi_out = np.float64(chi[0])
    else:
        chi_out = np.ascontiguousarray(chi, dtype=np.float64)
    return coeffs_out, stumpff_out, chi_out  # type: ignore[return-value]


def apply_lagrange_coefficients(
    r: npt.ArrayLike,
    v: npt.ArrayLike,
    f: npt.ArrayLike,
    g: npt.ArrayLike,
    f_dot: npt.ArrayLike,
    g_dot: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Lagrange coefficients to position and velocity vectors.

    This compatibility API is backed by
    ``adam_core_rs_coords::apply_lagrange_coefficients``.
    """
    r_rows, r_single = as_rows(r, name="r", width=3)
    v_rows, v_single = as_rows(v, name="v", width=3)
    if v_rows.shape[0] != r_rows.shape[0]:
        raise ValueError("v must have the same row count as r")

    n = r_rows.shape[0]
    coeffs = np.ascontiguousarray(
        np.column_stack(
            [
                broadcast_to_rows(f, n, name="f"),
                broadcast_to_rows(g, n, name="g"),
                broadcast_to_rows(f_dot, n, name="f_dot"),
                broadcast_to_rows(g_dot, n, name="g_dot"),
            ]
        ),
        dtype=np.float64,
    )
    r_new, v_new = require_rust(
        _rust_apply_lagrange_coefficients_numpy(r_rows, v_rows, coeffs),
        "dynamics.apply_lagrange_coefficients",
    )
    scalar = (
        r_single
        and v_single
        and _is_scalar_like(f)
        and _is_scalar_like(g)
        and _is_scalar_like(f_dot)
        and _is_scalar_like(g_dot)
    )
    if scalar:
        return np.ascontiguousarray(r_new[0]), np.ascontiguousarray(v_new[0])
    return r_new, v_new
