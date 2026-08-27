"""
Unit conversion utilities for coordinate systems.

This module provides functions to convert between different units commonly
used in astrodynamics, particularly between AU/day and km/s systems.
"""

from typing import Union

import numpy as np

from ..constants import KM_P_AU, S_P_DAY  # noqa: F401  (re-exported constants)


def _rust_scaled(values: Union[float, np.ndarray], kernel_name: str):
    """Route a float-or-ndarray unit conversion through one Rust crossing."""
    from adam_core import _rust_native

    array = np.asarray(values, dtype=np.float64)
    flat = np.ascontiguousarray(array.reshape(-1))
    out = np.asarray(
        getattr(_rust_native, kernel_name)(flat), dtype=np.float64
    ).reshape(array.shape)
    if isinstance(values, np.ndarray):
        return out
    return float(out)


__all__ = [
    "au_to_km",
    "km_to_au",
    "au_per_day_to_km_per_s",
    "km_per_s_to_au_per_day",
    "convert_cartesian_covariance_au_to_km",
    "convert_cartesian_covariance_km_to_au",
    "convert_cartesian_values_au_to_km",
    "convert_cartesian_values_km_to_au",
]


def au_to_km(values_au: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert position values from AU to km.

    Parameters
    ----------
    values_au : float or np.ndarray
        Position values in AU

    Returns
    -------
    float or np.ndarray
        Position values in km
    """
    return _rust_scaled(values_au, "au_to_km_numpy")


def km_to_au(values_km: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert position values from km to AU.

    Parameters
    ----------
    values_km : float or np.ndarray
        Position values in km

    Returns
    -------
    float or np.ndarray
        Position values in AU
    """
    return _rust_scaled(values_km, "km_to_au_numpy")


def au_per_day_to_km_per_s(
    values_au_day: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert velocity values from AU/day to km/s.

    Parameters
    ----------
    values_au_day : float or np.ndarray
        Velocity values in AU/day

    Returns
    -------
    float or np.ndarray
        Velocity values in km/s
    """
    return _rust_scaled(values_au_day, "au_per_day_to_km_per_s_numpy")


def km_per_s_to_au_per_day(
    values_km_s: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert velocity values from km/s to AU/day.

    Parameters
    ----------
    values_km_s : float or np.ndarray
        Velocity values in km/s

    Returns
    -------
    float or np.ndarray
        Velocity values in AU/day
    """
    return _rust_scaled(values_km_s, "km_per_s_to_au_per_day_numpy")


def convert_cartesian_values_au_to_km(values_au: np.ndarray) -> np.ndarray:
    """
    Convert CartesianCoordinates values from AU/AU-day to km/km-s units.

    Parameters
    ----------
    values_au : np.ndarray (N, 6)
        Coordinate values in AU and AU/day units:
        [x, y, z, vx, vy, vz] where positions are in AU and velocities in AU/day

    Returns
    -------
    np.ndarray (N, 6)
        Coordinate values in km and km/s units:
        [x, y, z, vx, vy, vz] where positions are in km and velocities in km/s
    """
    from adam_core import _rust_native

    return np.asarray(
        _rust_native.convert_cartesian_values_au_to_km_numpy(
            np.ascontiguousarray(values_au, dtype=np.float64)
        ),
        dtype=np.float64,
    )


def convert_cartesian_values_km_to_au(values_km: np.ndarray) -> np.ndarray:
    """
    Convert CartesianCoordinates values from km/km-s to AU/AU-day units.

    Parameters
    ----------
    values_km : np.ndarray (N, 6)
        Coordinate values in km and km/s units:
        [x, y, z, vx, vy, vz] where positions are in km and velocities in km/s

    Returns
    -------
    np.ndarray (N, 6)
        Coordinate values in AU and AU/day units:
        [x, y, z, vx, vy, vz] where positions are in AU and velocities in AU/day
    """
    from adam_core import _rust_native

    return np.asarray(
        _rust_native.convert_cartesian_values_km_to_au_numpy(
            np.ascontiguousarray(values_km, dtype=np.float64)
        ),
        dtype=np.float64,
    )


def convert_cartesian_covariance_au_to_km(covariance_au: np.ndarray) -> np.ndarray:
    """
    Convert CartesianCoordinates covariance matrix from AU units to km units.

    Parameters
    ----------
    covariance_au : np.ndarray (N, 6, 6)
        Covariance matrices in AU and AU/day units

    Returns
    -------
    np.ndarray (N, 6, 6)
        Covariance matrices in km and km/s units

    Notes
    -----
    The covariance matrix elements are converted as follows:
    - Position-position terms (AU²) → km²
    - Position-velocity terms (AU·AU/day) → km·km/s
    - Velocity-velocity terms ((AU/day)²) → (km/s)²
    """
    from adam_core import _rust_native

    return np.asarray(
        _rust_native.convert_cartesian_covariance_au_to_km_numpy(
            np.ascontiguousarray(covariance_au, dtype=np.float64)
        ),
        dtype=np.float64,
    )


def convert_cartesian_covariance_km_to_au(covariance_km: np.ndarray) -> np.ndarray:
    """
    Convert CartesianCoordinates covariance matrix from km units to AU units.

    Parameters
    ----------
    covariance_km : np.ndarray (N, 6, 6)
        Covariance matrices in km and km/s units

    Returns
    -------
    np.ndarray (N, 6, 6)
        Covariance matrices in AU and AU/day units

    Notes
    -----
    The covariance matrix elements are converted as follows:
    - Position-position terms (km²) → AU²
    - Position-velocity terms (km·km/s) → AU·AU/day
    - Velocity-velocity terms ((km/s)²) → (AU/day)²
    """
    from adam_core import _rust_native

    return np.asarray(
        _rust_native.convert_cartesian_covariance_km_to_au_numpy(
            np.ascontiguousarray(covariance_km, dtype=np.float64)
        ),
        dtype=np.float64,
    )
