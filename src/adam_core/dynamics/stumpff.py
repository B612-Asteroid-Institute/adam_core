from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

from .._rust.api import calc_stumpff_numpy as _rust_calc_stumpff_numpy
from ._rust_compat import ScalarOrArray, flatten_input, require_rust

STUMPFF_TYPES = Tuple[
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
    ScalarOrArray,
]


def calc_stumpff(psi: npt.ArrayLike) -> STUMPFF_TYPES:
    """
    Calculate the first six Stumpff functions for variable ``psi``.

    This compatibility API is backed by ``adam_core_rs_coords::calc_stumpff``.
    """
    psi_flat, shape = flatten_input(psi)
    rows = require_rust(_rust_calc_stumpff_numpy(psi_flat), "dynamics.calc_stumpff")
    if shape == ():
        return tuple(np.float64(rows[0, i]) for i in range(6))  # type: ignore[return-value]
    return tuple(
        np.ascontiguousarray(rows[:, i].reshape(shape), dtype=np.float64)
        for i in range(6)
    )  # type: ignore[return-value]
