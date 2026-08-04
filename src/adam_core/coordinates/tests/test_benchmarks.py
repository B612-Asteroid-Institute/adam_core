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
