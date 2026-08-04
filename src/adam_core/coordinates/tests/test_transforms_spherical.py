import numpy as np
import numpy.testing as npt

from ..transform import (
    _cartesian_to_spherical,
    _spherical_to_cartesian,
    cartesian_to_spherical,
    spherical_to_cartesian,
)

X_AXIS_CARTESIAN = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Y_AXIS_CARTESIAN = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
Z_AXIS_CARTESIAN = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
NEGATIVE_X_AXIS_CARTESIAN = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
NEGATIVE_Y_AXIS_CARTESIAN = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
NEGATIVE_Z_AXIS_CARTESIAN = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
CARTESIAN_AXES = [
    X_AXIS_CARTESIAN,
    Y_AXIS_CARTESIAN,
    Z_AXIS_CARTESIAN,
    NEGATIVE_X_AXIS_CARTESIAN,
    NEGATIVE_Y_AXIS_CARTESIAN,
    NEGATIVE_Z_AXIS_CARTESIAN,
]
CARTESIAN_AXES = np.vstack(CARTESIAN_AXES)

X_AXIS_SPHERICAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Y_AXIS_SPHERICAL = np.array([1.0, 90.0, 0.0, 0.0, 0.0, 0.0])
Z_AXIS_SPHERICAL = np.array([1.0, 0.0, 90.0, 0.0, 0.0, 0.0])
NEGATIVE_X_AXIS_SPHERICAL = np.array([1.0, 180.0, 0.0, 0.0, 0.0, 0.0])
NEGATIVE_Y_AXIS_SPHERICAL = np.array([1.0, 270.0, 0.0, 0.0, 0.0, 0.0])
NEGATIVE_Z_AXIS_SPHERICAL = np.array([1.0, 0.0, -90.0, 0.0, 0.0, 0.0])
SPHERICAL_AXES = [
    X_AXIS_SPHERICAL,
    Y_AXIS_SPHERICAL,
    Z_AXIS_SPHERICAL,
    NEGATIVE_X_AXIS_SPHERICAL,
    NEGATIVE_Y_AXIS_SPHERICAL,
    NEGATIVE_Z_AXIS_SPHERICAL,
]
SPHERICAL_AXES = np.vstack(SPHERICAL_AXES)


RELATIVE_TOLERANCE = 0.0
ABSOLUTE_TOLERANCE = 1e-15


def test__cartesian_to_spherical():
    # Test _cartesian_to_spherical correctly converts cartesian axes to their equivalent in spherical axes
    # Note velocity conversions are not explicitly tested here
    for cartesian_axis, spherical_axis in zip(CARTESIAN_AXES, SPHERICAL_AXES):
        npt.assert_allclose(
            _cartesian_to_spherical(cartesian_axis),
            spherical_axis,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )


def test_cartesian_to_spherical():
    # Test cartesian_to_spherical (vmapped conversion) correctly converts cartesian axes
    # to their equivalent in spherical axes
    # Note velocity conversions are not explicitly tested here
    npt.assert_allclose(
        cartesian_to_spherical(CARTESIAN_AXES),
        SPHERICAL_AXES,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


def test__spherical_to_cartesian():
    # Test _spherical_to_cartesian correctly converts spherical axes to their equivalent in cartesian axes
    # Note velocity conversions are not explicitly tested here
    for cartesian_axis, spherical_axis in zip(CARTESIAN_AXES, SPHERICAL_AXES):
        npt.assert_allclose(
            _spherical_to_cartesian(spherical_axis),
            cartesian_axis,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )


def test_spherical_to_cartesian():
    # Test spherical_to_cartesian (vmapped conversion) correctly converts spherical axes
    # to their equivalent in cartesian axes
    # Note velocity conversions are not explicitly tested here
    npt.assert_allclose(
        spherical_to_cartesian(SPHERICAL_AXES),
        CARTESIAN_AXES,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
