import numpy as np
import pytest

from ..origin import Origin, OriginCodes, OriginGravitationalParameters


def test_origin_eq__():
    origin = Origin.from_kwargs(code=["SUN", "EARTH", "SUN"])

    # Test equality with string
    np.testing.assert_equal(origin == "SUN", np.array([True, False, True]))
    np.testing.assert_equal(origin != "SUN", np.array([False, True, False]))

    # Test equality with numpy array
    np.testing.assert_equal(
        origin == np.array(["SUN", "EARTH", "SUN"]), np.array([True, True, True])
    )
    np.testing.assert_equal(
        origin == np.array(["SUN", "EARTH", "EARTH"]), np.array([True, True, False])
    )
    np.testing.assert_equal(
        origin != np.array(["SUN", "EARTH", "EARTH"]), np.array([False, False, True])
    )
    np.testing.assert_equal(
        origin == np.array(["SUN", "SUN", "SUN"]), np.array([True, False, True])
    )
    np.testing.assert_equal(
        origin != np.array(["SUN", "SUN", "SUN"]), np.array([False, True, False])
    )

    # Test equality with Origin
    np.testing.assert_equal(
        origin == Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        np.array([True, False, True]),
    )
    np.testing.assert_equal(
        origin != Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        np.array([False, True, False]),
    )
    np.testing.assert_equal(
        origin == Origin.from_kwargs(code=["SUN", "EARTH", "SUN"]),
        np.array([True, True, True]),
    )
    np.testing.assert_equal(
        origin != Origin.from_kwargs(code=["SUN", "EARTH", "SUN"]),
        np.array([False, False, False]),
    )

    # Test equality with OriginCodes
    np.testing.assert_equal(origin == OriginCodes.SUN, np.array([True, False, True]))
    np.testing.assert_equal(origin != OriginCodes.SUN, np.array([False, True, False]))
    np.testing.assert_equal(origin == OriginCodes.EARTH, np.array([False, True, False]))
    np.testing.assert_equal(origin != OriginCodes.EARTH, np.array([True, False, True]))


def test_origin__eq__raises():
    # Test that an error is raised when an unsupported type is passed
    origin = Origin.from_kwargs(code=["SUN", "EARTH", "MARS"])
    with pytest.raises(TypeError):
        origin == 1


def test_origin_mu():
    # Test that the mu function returns the correct values
    origin = Origin.from_kwargs(
        code=["SUN", "MARS_BARYCENTER", "JUPITER_BARYCENTER", "SUN"]
    )

    expected = np.array(
        [
            OriginGravitationalParameters.SUN,
            OriginGravitationalParameters.MARS_BARYCENTER,
            OriginGravitationalParameters.JUPITER_BARYCENTER,
            OriginGravitationalParameters.SUN,
        ]
    )
    np.testing.assert_equal(origin.mu(), expected)


def test_origin_mu_raises():
    # Test that the mu function raises a ValueError when an unsupported code is passed
    origin = Origin.from_kwargs(code=["VESTA"])
    with pytest.raises(ValueError):
        origin.mu()
