import numpy as np
import pytest

from ...time import Timestamp
from ...utils.spice import clear_spkez_cache
from ..cartesian import CartesianCoordinates
from ..cometary import CometaryCoordinates
from ..covariances import CoordinateCovariances
from ..keplerian import KeplerianCoordinates
from ..origin import Origin, OriginCodes
from ..spherical import SphericalCoordinates
from ..transform import clear_translation_cache, transform_coordinates


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
        time=Timestamp.from_mjd([50000] * size, scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"] * size),
        frame=frame_in,
    )

    def clear_result_caches() -> None:
        clear_translation_cache()
        clear_spkez_cache()

    result = benchmark.pedantic(
        transform_coordinates,
        args=(from_coords,),
        kwargs={
            "representation_out": representation,
            "frame_out": frame,
            "origin_out": origin,
        },
        setup=clear_result_caches,
        rounds=7,
        warmup_rounds=1,
        iterations=1,
    )
    assert len(result) == size


@pytest.mark.benchmark(group="coordinate_covariances")
def test_benchmark_coordinate_covariances_to_matrix(benchmark):
    rng = np.random.default_rng(0)
    covariances_filled = [rng.random(36) for _ in range(500)]
    covariances_missing = [None for _ in range(500)]
    coordinate_covariances = CoordinateCovariances.from_kwargs(
        values=covariances_filled + covariances_missing
    )
    benchmark(coordinate_covariances.to_matrix)


@pytest.mark.benchmark(group="coordinate_covariances")
def test_benchmark_coordinate_covariances_from_matrix(benchmark):
    rng = np.random.default_rng(0)
    covariances = rng.random((1000, 6, 6))
    covariances[500:, :, :] = np.nan
    benchmark(CoordinateCovariances.from_matrix, covariances)
