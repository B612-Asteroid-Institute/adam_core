import numpy as np

from ..origin import Origin
from ..spherical import SphericalCoordinates


def test_SphericalCoordinates_to_unit_sphere():
    # Test that the to_unit_sphere method works as expected. This method
    # should return a copy of the coordinates with rho set to 1.0 and
    # vrho set to 0.0. If only_missing is True, then only coordinates
    # that have NaN values for rho will be set to 1.0 and coordinates
    # that have NaN values for vrho will be set to 0.0.
    coords = SphericalCoordinates.from_kwargs(
        rho=[1.0, np.nan, 3.0],
        lon=[30.0, 45.0, 60.0],
        lat=[-60, 0.0, 60.0],
        vrho=[np.nan, 0.002, -0.01],
        vlon=[0.005, 0.006, 0.007],
        vlat=[-0.0005, 0.0006, -0.007],
        origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        frame="ecliptic",
    )

    unit_sphere = coords.to_unit_sphere(only_missing=True)
    np.testing.assert_allclose(unit_sphere.rho.to_numpy(), [1.0, 1.0, 3.0])
    np.testing.assert_allclose(unit_sphere.vrho.to_numpy(), [0.0, 0.002, -0.01])
    np.testing.assert_allclose(unit_sphere.lon.to_numpy(), [30.0, 45.0, 60.0])
    np.testing.assert_allclose(unit_sphere.lat.to_numpy(), [-60.0, 0.0, 60.0])
    np.testing.assert_allclose(unit_sphere.vlon.to_numpy(), [0.005, 0.006, 0.007])
    np.testing.assert_allclose(unit_sphere.vlat.to_numpy(), [-0.0005, 0.0006, -0.007])

    unit_sphere = coords.to_unit_sphere(only_missing=False)
    np.testing.assert_allclose(unit_sphere.rho.to_numpy(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(unit_sphere.vrho.to_numpy(), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(unit_sphere.lon.to_numpy(), [30.0, 45.0, 60.0])
    np.testing.assert_allclose(unit_sphere.lat.to_numpy(), [-60.0, 0.0, 60.0])
    np.testing.assert_allclose(unit_sphere.vlon.to_numpy(), [0.005, 0.006, 0.007])
    np.testing.assert_allclose(unit_sphere.vlat.to_numpy(), [-0.0005, 0.0006, -0.007])
