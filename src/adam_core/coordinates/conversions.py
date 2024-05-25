from copy import deepcopy
from typing import List, Union

import astropy.units
import numpy as np

from .types import Coordinates

# This module is not maintained.
__doctest_skip__ = ["*"]


def _convert_coordinates_units(
    coords: Union[np.ndarray, np.ma.masked_array], units: List, desired_units: List
) -> Union[np.ndarray, np.ma.masked_array]:
    """
    Convert coordinate units to desired units.

    Parameters
    ----------
    coords : `~numpy.ndarray` or `~numpy.ma.masked_array` (N, D)
        Coordinates that need to be converted.
    units : List (D)
        Current units for each coordinate dimension.
    desired_units : List (D)
        Desired units for each coordinate dimension.

    Returns
    -------
    coords_converted : `~numpy.ndarray` or `~numpy.ma.masked_array` (N, D)
        Coordinates converted to the desired coordinate units.

    Raises
    ------
    ValueError : If units or desired_units do not have length D.
    """
    N, D = coords.shape
    coords_converted = coords.copy()

    if (len(units) != D) or (len(desired_units) != D):
        err = f"Length of units or desired_units does not match the number of coordinate dimensions (D={D})"
        raise ValueError(err)

    for i in range(D):
        coords_converted[:, i] = coords[:, i] * units[i].to(desired_units[i])

    return coords_converted


def _convert_covariances_units(
    covariances: Union[np.ndarray, np.ma.masked_array],
    units: np.ndarray,
    desired_units: np.ndarray,
) -> Union[np.ndarray, np.ma.masked_array]:
    """
    Convert covariance units to desired units.

    Parameters
    ----------
    covariances : `~numpy.ndarray` or `~numpy.ma.masked_array` (N, D, D)
        Covariances that need to be converted.
    units : `~numpy.ndarray` (D)
        Current units for each coordinate dimension. Note, these are not the units for
        the elements of the covariance matrices but the actual units of the mean values/states to
        which the covariance matrices belong.
    desired_units : `~numpy.ndarray` (D)
        Desired units for each coordinate dimension.

    Returns
    -------
    covariances_converted : `~numpy.ndarray` or `~numpy.ma.masked_array` (N, D, D)
        Covariances converted to the desired coordinate units.

    Raises
    ------
    ValueError : If units or desired_units do not have length D.
    """
    N, D, D = covariances.shape

    if (len(units) != D) or (len(desired_units) != D):
        err = f"Length of units or desired_units does not match the number of coordinate dimensions (D={D})"
        raise ValueError(err)

    coords_units_2d = units.copy().reshape(1, -1)
    covariance_units = np.dot(coords_units_2d.T, coords_units_2d)

    desired_units_2d = desired_units.copy().reshape(1, -1)
    desired_units = np.dot(desired_units_2d.T, desired_units_2d)

    conversion_matrix = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        for j in range(D):
            conversion_matrix[i, j] = covariance_units[i, j].to(desired_units[i, j])

    covariances_converted = conversion_matrix * covariances

    return covariances_converted


def convert_coordinates(
    coords: Coordinates,
    desired_units: Union[List[astropy.units.Unit], dict[str, astropy.units.Unit]],
):
    """
    Convert the units in a Coordinates object to desired units.

    Parameters
    ----------
    coords :
        Coordinates that need to be converted.
    desired_units : list, `~numpy.ndarray`, dict
        Desired units for each coordinate dimension expressed as an array-like
        or a dictionary keyed on coordinate dimensions and with values that represent
        the desired units. If a dictionary is passed, then only the coordinate dimensions
        that need to be converted need to be defined.

    Returns
    -------
    coords_converted : `~thor.coordinates.Coordinates` (N, D)
        Coordinates converted to the desired coordinate units.

    Raises
    ------
    ValueError : If units or desired_units do not have length D.
    ValueError : If desired_units is not a list, `~numpy.ndarray`, dict

    Examples
    --------
    >>> import astropy.units
    >>> from adam_core.coordinates import CartesianCoordinates, convert_coordinates
    >>> # By default, x, y, and z are in AU
    >>> coords = CartesianCoordinates.from_kwargs(
    ...    x=[1, 1, 1],
    ...    y=[4, 5, 6],
    ...    z=[7, 8, 9],
    ...    vx=[10, 11, 12],
    ...    vy=[13, 14, 15],
    ...    vz=[16, 17, 18],
    ... )
    >>> desired_units = {
    ...     "x" : astropy.units.km
    ... }
    >>> coordinates_converted = convert_coordinates(coords, desired_units)
    >>> print(coordinates_converted.x.to_pylist())
    [149597870.0, 149597870.0, 149597870.0]
    >>> print(coordinates_converted.y.to_pylist())
    [4.0, 5.0, 6.0]

    """
    N, D = coords.values.shape

    desired_units_ = {}
    if isinstance(desired_units, dict):
        for k, v in coords.units.items():
            if k not in desired_units:
                desired_units_[k] = v
            else:
                desired_units_[k] = desired_units[k]

    elif isinstance(desired_units, (list, np.ndarray)):
        if len(desired_units) != D:
            err = (
                "Length of units or desired_units does not match the"
                f" number of coordinate dimensions (D={D})"
            )
            raise ValueError(err)

        for i, (k, v) in enumerate(coords.units.items()):
            desired_units_[k] = desired_units[i]

    else:
        err = "desired_units should be one of {list, '~numpy.ndarray`, dict}"
        raise ValueError(err)

    # Make a copy of the input coordinates
    coords_converted = deepcopy(coords)
    desired_units_array = np.array(list(desired_units_.values()))
    current_units = np.array(list(coords.units.values()))

    # Modify the underlying coordinates array
    values_converted = _convert_coordinates_units(
        coords.values,
        current_units,
        desired_units_array,
    )
    coords_converted._values = values_converted

    # Modify the underlying covariance matrices
    if coords.covariances is not None:
        covariances_converted = _convert_covariances_units(
            coords.covariances,
            current_units,
            desired_units_array,
        )
        coords_converted._covariances = covariances_converted

    # Update the units dictionary to reflect the new units
    coords_converted._units = desired_units_

    return coords_converted
