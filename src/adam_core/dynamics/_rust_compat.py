from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T")
ScalarOrArray = np.ndarray | np.float64


def require_rust(result: T | None, api_name: str) -> T:
    if result is None:
        raise RuntimeError(f"adam_core._rust_native is required for {api_name}")
    return result


def as_rows(values: npt.ArrayLike, *, name: str, width: int) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (width,):
            raise ValueError(f"{name} must have shape ({width},) or (N, {width})")
        return np.ascontiguousarray(arr.reshape(1, width)), True
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{name} must have shape ({width},) or (N, {width})")
    return np.ascontiguousarray(arr), False


def broadcast_to_rows(values: npt.ArrayLike, n: int, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.shape != (n,):
        raise ValueError(f"{name} must be scalar or have shape ({n},)")
    return np.ascontiguousarray(flat)


def broadcast_pair(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    left_arr, right_arr = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    shape = left_arr.shape
    return (
        np.ascontiguousarray(left_arr.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(right_arr.reshape(-1), dtype=np.float64),
        shape,
    )


def flatten_input(values: npt.ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(values, dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(-1), dtype=np.float64), arr.shape


def restore_shape(values: npt.ArrayLike, shape: tuple[int, ...]) -> ScalarOrArray:
    arr = np.asarray(values, dtype=np.float64).reshape(shape)
    if shape == ():
        return np.float64(arr[()])
    return np.ascontiguousarray(arr)
