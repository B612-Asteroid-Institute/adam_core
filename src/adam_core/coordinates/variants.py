from __future__ import annotations

from typing import Generic, Literal, Optional, Protocol, TypeVar

import numpy as np
import pyarrow as pa
import quivr as qv

from . import types
from .cartesian import CartesianCoordinates
from .cometary import CometaryCoordinates
from .keplerian import KeplerianCoordinates
from .spherical import SphericalCoordinates

T = TypeVar("T", bound=types.CoordinateType, covariant=True)


class VariantCoordinatesTable(Generic[T], Protocol):
    """
    A protocol for a generic table of variant coordinates.
    """

    @property
    def index(self) -> pa.Int64Array: ...

    @property
    def sample(self) -> T: ...

    @property
    def weight(self) -> pa.lib.DoubleArray: ...

    @property
    def weight_cov(self) -> pa.lib.DoubleArray: ...


def _coordinate_dimensions(coordinates: types.CoordinateType) -> list[str]:
    if isinstance(coordinates, CartesianCoordinates):
        return ["x", "y", "z", "vx", "vy", "vz"]
    if isinstance(coordinates, SphericalCoordinates):
        return ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
    if isinstance(coordinates, KeplerianCoordinates):
        return ["a", "e", "i", "raan", "ap", "M"]
    if isinstance(coordinates, CometaryCoordinates):
        return ["q", "e", "i", "raan", "ap", "tp"]
    raise ValueError(f"Unsupported coordinate type: {type(coordinates)}")


def create_coordinate_variants(
    coordinates: types.CoordinateType,
    method: Literal["auto", "sigma-point", "monte-carlo"] = "auto",
    num_samples: int = 10000,
    alpha: float = 1,
    beta: float = 0,
    kappa: float = 0,
    seed: Optional[int] = None,
) -> VariantCoordinatesTable[types.CoordinateType]:
    """
    Sample and create variants for the given coordinates by sampling the covariance matrices.
    There are three supported methods:

    1. sigma-point: Sample the covariance matrix using sigma points. This is the fastest method,
       but can be inaccurate if the covariance matrix is not well behaved.
    2. monte-carlo: Sample the covariance matrix using a monte carlo method.
       This is the slowest method, but is the most accurate.
    3. auto: Automatically select the best method based on the covariance matrix.
       If the covariance matrix is well behaved then sigma-point sampling will be used.
       If the covariance matrix is not well behaved then monte-carlo sampling will be used.

    When sampling with monte-carlo, 10k samples are drawn. Sigma-point sampling draws 13 samples
    for 6-dimensional coordinates.

    .. warning::

        This function does not yet handle sampling of covariances and coordinates with missing values.

    Parameters
    ----------
    coordinates :
        The coordinates to sample.
    method:
        The method to use for sampling the covariance matrix. If 'auto' is selected then the method
        will be automatically selected based on the covariance matrix. The default is 'auto'.
    num_samples : int, optional
        The number of samples to draw when sampling with monte-carlo.
    alpha:
        Spread of the sigma points between 1e^-2 and 1.
    beta:
        Prior knowledge of the distribution when generating sigma points usually set to 2 for a Gaussian.
    kappa:
        Secondary scaling parameter when generating sigma points usually set to 0.


    Returns
    -------
        The variant coordinates.

    Raises
    ------
    ValueError:
        If the covariance matrices are all undefined.
        If the input coordinates are not supported.
    """

    covariances = coordinates.covariance.to_matrix()
    if np.all(np.isnan(covariances)):
        raise ValueError(
            "Cannot sample coordinate covariances when covariances are all undefined."
        )

    class VariantCoordinates(qv.Table):
        index = qv.Int64Column()
        sample = coordinates.as_column()
        weight = qv.Float64Column()
        weight_cov = qv.Float64Column()

    dimensions = _coordinate_dimensions(coordinates)
    means = coordinates.values

    # One Rust crossing owns per-row NaN validation, sigma-point/Monte-Carlo/
    # auto sampling policy, and index assembly. Monte Carlo draws use the
    # Rust-native RNG (decision 2026-07-03: statistically equivalent to, but
    # not bit-identical with, the legacy scipy sampler); the seed is threaded
    # per-row and into the auto-mode fallback, matching VariantOrbits.create.
    from adam_core import _rust_native

    samples, weights, weights_cov, index = (
        _rust_native.sample_coordinate_variants_numpy(
            np.ascontiguousarray(means, dtype=np.float64),
            np.ascontiguousarray(covariances, dtype=np.float64),
            method,
            int(num_samples),
            seed,
            float(alpha),
            float(beta),
            float(kappa),
        )
    )
    take_index = pa.array(index, type=pa.int64())

    return VariantCoordinates.from_kwargs(
        index=index,
        sample=coordinates.from_kwargs(
            origin=coordinates.origin.take(take_index),
            time=coordinates.time.take(take_index),
            frame=coordinates.frame,
            **{dim: samples[:, dim_index] for dim_index, dim in enumerate(dimensions)},
        ),
        weight=weights,
        weight_cov=weights_cov,
    )
