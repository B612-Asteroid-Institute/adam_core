import numpy as np
import pytest

from ...time import Timestamp
from ..cartesian import CartesianCoordinates
from ..cometary import CometaryCoordinates
from ..covariances import CoordinateCovariances
from ..keplerian import KeplerianCoordinates
from ..origin import Origin, OriginCodes
from ..spherical import SphericalCoordinates
from ..transform import transform_coordinates
from ..transform import _cartesian_to_spherical
from ..covariances import transform_covariances_jacobian


@pytest.mark.parametrize(
    "representation",
    [SphericalCoordinates, KeplerianCoordinates, CometaryCoordinates],
    ids=lambda x: f"to={x.__name__},",
)
@pytest.mark.parametrize(
    "frame", ["equatorial", "ecliptic"], ids=lambda x: f"frame={x},"
)
@pytest.mark.parametrize(
    "origin",
    [OriginCodes.SUN, OriginCodes.SOLAR_SYSTEM_BARYCENTER],
    ids=lambda x: f"origin={x.name},",
)
@pytest.mark.parametrize("size", [1, 50, 100])
@pytest.mark.benchmark(group="coord_transforms")
def test_benchmark_transform_cartesian_coordinates(
    benchmark, representation, frame, origin, size
):
    if origin == OriginCodes.SOLAR_SYSTEM_BARYCENTER:
        pytest.skip("barycenter transform not yet implemented")

    if frame == "ecliptic":
        frame_in = "equatorial"
    else:
        frame_in = "ecliptic"

    from_coords = CartesianCoordinates.from_kwargs(
        x=np.array([1] * size),
        y=np.array([1] * size),
        z=np.array([1] * size),
        vx=np.array([1] * size),
        vy=np.array([1] * size),
        vz=np.array([1] * size),
        time=Timestamp.from_mjd([50000] * size),
        origin=Origin.from_kwargs(code=["SUN"] * size),
        frame=frame_in,
    )
    benchmark(
        transform_coordinates,
        from_coords,
        representation_out=representation,
        frame_out=frame,
        origin_out=origin,
    )


@pytest.mark.benchmark(group="coordinate_covariances")
def test_benchmark_CoordinateCovariances_to_matrix(benchmark):

    covariances_filled = [np.random.random(36) for _ in range(500)]
    covariances_missing = [None for _ in range(500)]
    coordinate_covariances = CoordinateCovariances.from_kwargs(
        values=covariances_filled + covariances_missing
    )
    benchmark(coordinate_covariances.to_matrix)


@pytest.mark.benchmark(group="coordinate_covariances")
def test_benchmark_CoordinateCovariances_from_matrix(benchmark):

    covariances = np.random.random((1000, 6, 6))
    covariances[500:, :, :] = np.nan
    benchmark(CoordinateCovariances.from_matrix, covariances)


@pytest.mark.parametrize(
    "chunk_size",
    [None, 200],
    ids=lambda x: "chunk_size=None" if x is None else f"chunk_size={x}",
)
@pytest.mark.benchmark(group="coordinate_covariances_jacobian")
def test_benchmark_transform_covariances_jacobian_varying_n_cold_cache(benchmark, chunk_size):
    """
    Stress-test for shape-driven recompiles:
    - We call transform_covariances_jacobian() across a variety of N (like THOR per-state batches).
    - We clear JAX in-process caches at the start so each benchmark sample includes compilation.
      This makes the difference between chunked vs unchunked robust and visible.
    """
    rng = np.random.default_rng(0)
    sizes = [50, 200, 868, 2176, 2303]
    n_max = max(sizes)

    coords_all = rng.normal(size=(n_max, 6)).astype(np.float64)

    # Create PSD-ish covariances: A @ A.T (+ small diagonal) per row.
    A = rng.normal(size=(n_max, 6, 6)).astype(np.float64)
    cov_all = A @ np.transpose(A, axes=(0, 2, 1))
    cov_all += np.eye(6, dtype=np.float64)[None, :, :] * 1e-6

    def run():
        import jax

        clear = getattr(jax, "clear_caches", None)
        if callable(clear):
            clear()

        out = None
        for n in sizes:
            out = transform_covariances_jacobian(
                coords_all[:n],
                cov_all[:n],
                _cartesian_to_spherical,
                chunk_size=chunk_size,
            )
        return out

    result = benchmark(run)
    benchmark.extra_info.update({"sizes": sizes, "chunk_size": chunk_size, "result_shape": result.shape})
