import numpy as np
import pytest
from astropy.time import Time

from ..cartesian import CartesianCoordinates
from ..cometary import CometaryCoordinates
from ..keplerian import KeplerianCoordinates
from ..origin import Origin, OriginCodes
from ..spherical import SphericalCoordinates
from ..times import Times
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
@pytest.mark.benchmark(group="coord_transforms")
def test_benchmark_transform_cartesian_coordinates(
    benchmark, representation, frame, origin
):
    if origin == OriginCodes.SOLAR_SYSTEM_BARYCENTER:
        pytest.skip("barycenter transform not yet implemented")

    from_coords = CartesianCoordinates.from_kwargs(
        x=np.array([1]),
        y=np.array([1]),
        z=np.array([1]),
        vx=np.array([1]),
        vy=np.array([1]),
        vz=np.array([1]),
        time=Times.from_astropy(
            Time([50000], format="mjd"),
        ),
        origin=Origin.from_kwargs(code=["SUN"]),
    )
    benchmark(
        transform_coordinates,
        from_coords,
        representation,
        frame_out=frame,
        origin_out=origin,
    )
