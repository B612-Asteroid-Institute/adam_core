from __future__ import annotations

import numpy.typing as npt

from .._rust.api import solve_barker_numpy as _rust_solve_barker_numpy
from ._rust_compat import ScalarOrArray, flatten_input, require_rust, restore_shape


def solve_barker(M: npt.ArrayLike) -> ScalarOrArray:
    """
    Solve Barker's equation for true anomaly given parabolic mean anomaly.

    This compatibility API is backed by ``adam_core_rs_coords::solve_barker``.
    """
    m_flat, shape = flatten_input(M)
    out = require_rust(_rust_solve_barker_numpy(m_flat), "dynamics.solve_barker")
    return restore_shape(out, shape)
