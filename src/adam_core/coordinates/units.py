"""
Unit conversion utilities for coordinate systems.

This module provides functions to convert between different units commonly
used in astrodynamics, particularly between AU/day and km/s systems.
"""

from typing import Union

import numpy as np

from ..constants import KM_P_AU, S_P_DAY

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
    return values_au * KM_P_AU


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
    return values_km / KM_P_AU


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
    return values_au_day * KM_P_AU / S_P_DAY


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
    return values_km_s / KM_P_AU * S_P_DAY


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
    values_km = values_au.copy()
    values_km[:, :3] = au_to_km(values_au[:, :3])  # positions
    values_km[:, 3:] = au_per_day_to_km_per_s(values_au[:, 3:])  # velocities
    return values_km


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
    values_au = values_km.copy()
    values_au[:, :3] = km_to_au(values_km[:, :3])  # positions
    values_au[:, 3:] = km_per_s_to_au_per_day(values_km[:, 3:])  # velocities
    return values_au


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
    # Unit conversion vector: [km, km, km, km/s, km/s, km/s]
    unit_conversion = np.array(
        [
            KM_P_AU,
            KM_P_AU,
            KM_P_AU,
            KM_P_AU / S_P_DAY,
            KM_P_AU / S_P_DAY,
            KM_P_AU / S_P_DAY,
        ]
    )

    # Create conversion matrix by outer product
    conversion_matrix = np.outer(unit_conversion, unit_conversion)

    return covariance_au * conversion_matrix


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
    # Unit conversion vector: [AU, AU, AU, AU/day, AU/day, AU/day]
    unit_conversion = np.array(
        [
            1 / KM_P_AU,
            1 / KM_P_AU,
            1 / KM_P_AU,
            S_P_DAY / KM_P_AU,
            S_P_DAY / KM_P_AU,
            S_P_DAY / KM_P_AU,
        ]
    )

    # Create conversion matrix by outer product
    conversion_matrix = np.outer(unit_conversion, unit_conversion)

    return covariance_km * conversion_matrix
