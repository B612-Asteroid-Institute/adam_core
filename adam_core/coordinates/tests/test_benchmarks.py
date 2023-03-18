import pytest
import numpy as np
import astropy.time

from ..cartesian import CartesianCoordinates
from ..spherical import SphericalCoordinates
from ..transform import transform_coordinates


@pytest.mark.parametrize(
    "representation", ["spherical", "keplerian", "cometary"], ids=lambda x: f"to={x},"
)
@pytest.mark.parametrize(
    "frame", ["equatorial", "ecliptic"], ids=lambda x: f"frame={x},"
)
@pytest.mark.parametrize(
    "origin", ["heliocenter", "barycenter"], ids=lambda x: f"origin={x},"
)
@pytest.mark.benchmark(group="coord_transforms")
def test_benchmark_transform_cartesian_coordinates(
    benchmark, representation, frame, origin
):
    if origin == "barycenter":
        pytest.skip("barycenter transform not yet implemented")
    from_coords = CartesianCoordinates(
        x=np.array([1]),
        y=np.array([1]),
        z=np.array([1]),
        vx=np.array([1]),
        vy=np.array([1]),
        vz=np.array([1]),
        times=astropy.time.Time(50000, format="mjd")
    )
    benchmark(
        transform_coordinates,
        coords=from_coords,
        representation_out=representation,
        frame_out=frame,
        origin_out=origin,
    )
