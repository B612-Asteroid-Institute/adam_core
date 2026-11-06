"""Typed runtime boundary for SciPy's optional least-squares helper."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


class LeastSquaresResult(Protocol):
    x: npt.NDArray[np.float64]
    fun: npt.NDArray[np.float64]
    success: bool
    cost: float


@runtime_checkable
class LeastSquares(Protocol):
    def __call__(
        self,
        function: Callable[..., object],
        x0: object,
        **kwargs: object,
    ) -> LeastSquaresResult: ...


def scipy_least_squares() -> LeastSquares:
    """Load and validate the optional SciPy least-squares entry point."""
    optimize = importlib.import_module("scipy.optimize")
    function = getattr(optimize, "least_squares", None)
    if not isinstance(function, LeastSquares):
        raise TypeError("scipy.optimize must export callable least_squares")
    return function
