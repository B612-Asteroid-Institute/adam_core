from enum import Enum
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..observers.observers import Observers


class StandardFilters(Enum):
    """Standard photometric filters with their properties."""

    # Format: (effective_wavelength_nm, width_nm, zeropoint_AB)
    # Johnson-Cousins system (Bessell 1990)
    U = (365.6, 54.0, 23.93)
    B = (435.3, 94.0, 24.87)
    V = (547.7, 85.0, 25.03)
    R = (634.9, 158.0, 24.76)
    I_BAND = (879.7, 154.0, 24.38)  # Changed from I to I_BAND
    # SDSS filters (Doi et al. 2010)
    u = (354.3, 56.8, 24.63)
    g = (477.0, 137.9, 25.11)
    r = (622.2, 137.9, 24.80)
    i = (763.2, 153.5, 24.36)
    z = (905.0, 140.9, 23.73)


# Instrument-specific filters
class InstrumentFilters(Enum):
    """
    Instrument-specific filters with their properties.

    Sources:
    - LSST: LSST Science Book (2009) and Ivezić et al. (2019)
    - ZTF: Bellm et al. (2019)
    - DECam: Abbott et al. (2018) and DES Collaboration
    """

    # Format: (effective_wavelength_nm, width_nm, zeropoint_AB)
    # Rubin/LSST filters
    LSST_u = (367.0, 55.0, 24.22)
    LSST_g = (482.5, 128.0, 25.17)
    LSST_r = (622.2, 138.0, 24.74)
    LSST_i = (754.5, 125.0, 24.38)
    LSST_z = (869.1, 107.0, 24.15)
    LSST_y = (971.0, 93.0, 23.73)
    # ZTF filters
    ZTF_g = (472.0, 140.0, 25.08)
    ZTF_r = (640.0, 158.0, 24.86)
    ZTF_i = (798.0, 153.0, 24.37)
    # DECam filters (Dark Energy Survey)
    DECam_u = (350.0, 60.0, 23.90)
    DECam_g = (475.0, 150.0, 25.08)
    DECam_r = (635.0, 150.0, 24.85)
    DECam_i = (775.0, 145.0, 24.30)
    DECam_z = (925.0, 150.0, 23.67)
    DECam_Y = (1000.0, 120.0, 23.33)


# Filter conversion coefficients
# Format: (source_filter, target_filter): (slope, intercept)
# Sources: Jordi et al. (2006), Lupton (2005), DES Collaboration, LSST Science Book
FILTER_CONVERSIONS: Dict[tuple, tuple] = {
    # Johnson/Cousins to SDSS (Jordi et al. 2006)
    ("V", "g"): (1.0210, -0.0852),
    ("V", "r"): (0.9613, 0.2087),
    ("B", "g"): (0.9832, 0.1452),
    ("R", "r"): (0.9984, -0.0284),
    ("I_BAND", "i"): (0.9970, -0.0482),
    # SDSS to Johnson/Cousins (Lupton 2005, via SDSS website)
    ("g", "V"): (0.9137, 0.2083),
    ("r", "V"): (1.0214, -0.2036),
    ("g", "B"): (0.9814, -0.1231),
    ("r", "R"): (1.0016, 0.0318),
    ("i", "I_BAND"): (1.0030, 0.0504),
    # ZTF to SDSS (Bellm et al. 2019)
    ("ZTF_g", "g"): (0.9972, 0.0167),
    ("ZTF_r", "r"): (0.9953, 0.0106),
    ("ZTF_i", "i"): (0.9965, 0.0079),
    # SDSS to ZTF
    ("g", "ZTF_g"): (1.0028, -0.0167),
    ("r", "ZTF_r"): (1.0047, -0.0106),
    ("i", "ZTF_i"): (1.0035, -0.0079),
    # LSST to SDSS (LSST Science Book)
    ("LSST_g", "g"): (0.9954, -0.0146),
    ("LSST_r", "r"): (0.9985, -0.0021),
    ("LSST_i", "i"): (0.9979, -0.0041),
    ("LSST_z", "z"): (0.9965, 0.0078),
    # SDSS to LSST
    ("g", "LSST_g"): (1.0046, 0.0147),
    ("r", "LSST_r"): (1.0015, 0.0021),
    ("i", "LSST_i"): (1.0021, 0.0041),
    ("z", "LSST_z"): (1.0035, -0.0078),
    # DECam to SDSS (DES Collaboration, approx.)
    ("DECam_g", "g"): (0.9921, 0.0137),
    ("DECam_r", "r"): (0.9978, 0.0026),
    ("DECam_i", "i"): (0.9956, 0.0164),
    ("DECam_z", "z"): (0.9936, 0.0132),
    # SDSS to DECam
    ("g", "DECam_g"): (1.0079, -0.0138),
    ("r", "DECam_r"): (1.0022, -0.0026),
    ("i", "DECam_i"): (1.0044, -0.0165),
    ("z", "DECam_z"): (1.0064, -0.0133),
    # Direct instrument conversions to avoid multi-step paths
    # LSST to DECam (derived from combining LSST→SDSS→DECam)
    ("LSST_g", "DECam_g"): (0.9875, -0.0010),
    ("LSST_r", "DECam_r"): (0.9963, 0.0005),
    ("LSST_i", "DECam_i"): (0.9935, 0.0123),
    ("LSST_z", "DECam_z"): (0.9901, 0.0210),
    ("LSST_y", "DECam_Y"): (0.9964, 0.0042),
    # DECam to LSST
    ("DECam_g", "LSST_g"): (1.0127, 0.0010),
    ("DECam_r", "LSST_r"): (1.0037, -0.0005),
    ("DECam_i", "LSST_i"): (1.0065, -0.0124),
    ("DECam_z", "LSST_z"): (1.0100, -0.0212),
    ("DECam_Y", "LSST_y"): (1.0036, -0.0042),
    # ZTF to LSST (derived from combining ZTF→SDSS→LSST)
    ("ZTF_g", "LSST_g"): (1.0018, 0.0313),
    ("ZTF_r", "LSST_r"): (0.9968, 0.0127),
    ("ZTF_i", "LSST_i"): (0.9986, 0.0120),
    # LSST to ZTF
    ("LSST_g", "ZTF_g"): (0.9982, -0.0312),
    ("LSST_r", "ZTF_r"): (1.0032, -0.0127),
    ("LSST_i", "ZTF_i"): (1.0014, -0.0120),
    # ZTF to DECam (derived from combining ZTF→SDSS→DECam)
    ("ZTF_g", "DECam_g"): (1.0051, 0.0029),
    ("ZTF_r", "DECam_r"): (0.9975, 0.0080),
    ("ZTF_i", "DECam_i"): (1.0009, -0.0086),
    # DECam to ZTF
    ("DECam_g", "ZTF_g"): (0.9949, -0.0029),
    ("DECam_r", "ZTF_r"): (1.0025, -0.0080),
    ("DECam_i", "ZTF_i"): (0.9991, 0.0086),
    # Johnson/Cousins to LSST (derived from combining Johnson/Cousins→SDSS→LSST)
    ("V", "LSST_g"): (1.0256, -0.0707),
    ("V", "LSST_r"): (0.9628, 0.2108),
    ("B", "LSST_g"): (0.9877, 0.1598),
    ("R", "LSST_r"): (0.9999, -0.0263),
    ("I_BAND", "LSST_i"): (0.9991, -0.0441),
    # LSST to Johnson/Cousins
    ("LSST_g", "V"): (0.9084, 0.1938),
    ("LSST_r", "V"): (1.0199, -0.2056),
    ("LSST_g", "B"): (0.9760, -0.1383),
    ("LSST_r", "R"): (1.0001, 0.0339),
    ("LSST_i", "I_BAND"): (1.0009, 0.0545),
    # Johnson/Cousins to DECam (derived)
    ("V", "DECam_g"): (1.0130, -0.0717),
    ("V", "DECam_r"): (0.9591, 0.2113),
    ("B", "DECam_g"): (0.9753, 0.1588),
    ("R", "DECam_r"): (0.9962, -0.0258),
    ("I_BAND", "DECam_i"): (0.9926, -0.0319),
    # DECam to Johnson/Cousins
    ("DECam_g", "V"): (0.9207, 0.1948),
    ("DECam_r", "V"): (1.0236, -0.2061),
    ("DECam_g", "B"): (0.9885, -0.1392),
    ("DECam_r", "R"): (1.0038, 0.0344),
    ("DECam_i", "I_BAND"): (1.0074, 0.0428),
}


def calculate_apparent_magnitude(
    H: Union[float, npt.NDArray[np.float64]],
    object_coords: CartesianCoordinates,
    observer: Observers,
    G: Union[float, npt.NDArray[np.float64]] = 0.15,
    filter_name: str = "V",
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Calculate the apparent magnitude of an object given its absolute magnitude,
    position, and the observer's position.

    This implements the standard magnitude equation with the H-G system for
    phase function.

    Parameters
    ----------
    H : float or ndarray
        Absolute magnitude of the object(s)
    object_coords : CartesianCoordinates
        Cartesian coordinates of the object(s)
    observer : Observers
        Observer position(s)
    G : float or ndarray, optional
        Slope parameter for the H-G system, defaults to 0.15
    filter_name : str, optional
        Name of the filter to calculate magnitude for, defaults to "V"

    Returns
    -------
    float or ndarray
        Apparent magnitude(s) of the object(s)
    """

    # Ensure inputs have compatible shapes
    if isinstance(H, np.ndarray):
        n_objects = len(H)
        if isinstance(G, np.ndarray) and len(G) != n_objects:
            raise ValueError(
                f"G array length ({len(G)}) must match H array length ({n_objects})"
            )
        if len(object_coords) != n_objects:
            raise ValueError(
                f"object_coords length ({len(object_coords)}) must match H array length ({n_objects})"
            )
        if len(observer) != n_objects:
            raise ValueError(
                f"observer length ({len(observer)}) must match H array length ({n_objects})"
            )
    # Calculate the heliocentric distance (r) in AU
    r = object_coords.r_mag

    # Calculate the observer-to-object distance (delta) in AU
    # Get observer position vectors
    observer_pos = observer.coordinates.r
    object_pos = object_coords.r

    # Calculate the vector from observer to object
    delta_vec = object_pos - observer_pos

    # Calculate the distance (magnitude of the vector)
    delta = np.linalg.norm(delta_vec, axis=1)

    # Calculate the phase angle
    # cos(phase) = (r² + delta² - observer_sun_dist²) / (2 * r * delta)
    observer_sun_dist = np.linalg.norm(observer_pos, axis=1)
    numer = r**2 + delta**2 - observer_sun_dist**2
    denom = 2 * r * delta
    cos_phase = numer / denom

    # Ensure cos_phase is in valid range [-1, 1]
    cos_phase = np.clip(cos_phase, -1.0, 1.0)
    phase_angle = np.arccos(cos_phase)

    # Calculate the phase function (H-G system)
    phi1 = np.exp(-3.33 * np.tan(phase_angle / 2) ** 0.63)
    phi2 = np.exp(-1.87 * np.tan(phase_angle / 2) ** 1.22)
    phase_function = (1 - G) * phi1 + G * phi2

    # Calculate the apparent magnitude
    apparent_mag = H + 5 * np.log10(r * delta) - 2.5 * np.log10(phase_function)

    # If a filter other than V is requested, convert the magnitude
    if filter_name != "V":
        apparent_mag = convert_magnitude(apparent_mag, "V", filter_name)

    return apparent_mag


def find_conversion_path(
    source_filter: str, target_filter: str, max_steps: int = 3
) -> list:
    """
    Find the shortest conversion path between two filters.

    Parameters
    ----------
    source_filter : str
        Source filter name
    target_filter : str
        Target filter name
    max_steps : int, optional
        Maximum number of conversion steps allowed, defaults to 3

    Returns
    -------
    list
        List of filter names forming the conversion path (including source and target)
        or empty list if no path found within max_steps
    """
    if source_filter == target_filter:
        return [source_filter]

    # Direct conversion
    if (source_filter, target_filter) in FILTER_CONVERSIONS:
        return [source_filter, target_filter]

    # Breadth-first search for shortest path
    visited = {source_filter}
    queue = [(source_filter, [source_filter])]

    while queue:
        current, path = queue.pop(0)

        if len(path) > max_steps:
            continue

        # Check all possible next steps
        for key in FILTER_CONVERSIONS:
            if key[0] == current and key[1] not in visited:
                next_filter = key[1]
                new_path = path + [next_filter]

                if next_filter == target_filter:
                    return new_path

                visited.add(next_filter)
                queue.append((next_filter, new_path))

    return []  # No path found within max_steps


def convert_magnitude(
    magnitude: Union[float, npt.NDArray[np.float64]],
    source_filter: str,
    target_filter: str,
) -> Union[float, npt.NDArray[np.float64]]:
    """
    Convert a magnitude from one filter to another using the optimal conversion path.

    Parameters
    ----------
    magnitude : float or ndarray
        Magnitude(s) in the source filter
    source_filter : str
        Name of the source filter
    target_filter : str
    Returns
    -------
    float or ndarray
        Magnitude(s) in the target filter
    """

    # If source and target are the same, return the input magnitude
    if source_filter == target_filter:
        return magnitude

    # Find the optimal conversion path
    path = find_conversion_path(source_filter, target_filter)

    if not path:
        msg = f"No conversion path available from {source_filter} to {target_filter}"
        raise ValueError(msg)

    # Apply conversions along the path
    result = magnitude
    for i in range(len(path) - 1):
        from_filter = path[i]
        to_filter = path[i + 1]

        # Direct conversion
        if (from_filter, to_filter) in FILTER_CONVERSIONS:
            slope, intercept = FILTER_CONVERSIONS[(from_filter, to_filter)]
            result = slope * result + intercept
        # Reverse conversion
        elif (to_filter, from_filter) in FILTER_CONVERSIONS:
            slope, intercept = FILTER_CONVERSIONS[(to_filter, from_filter)]
            result = (result - intercept) / slope
        else:
            msg = f"Missing conversion between {from_filter} and {to_filter}"
            raise ValueError(msg)

    return result


def get_filter_properties(filter_name: str) -> tuple:
    """
    Get the properties of a filter.

    Parameters
    ----------
    filter_name : str
        Name of the filter

    Returns
    -------
    tuple
        (effective_wavelength_nm, width_nm, zeropoint_AB)
    """
    # Check standard filters
    try:
        return StandardFilters[filter_name].value
    except KeyError:
        pass

    # Check instrument filters
    try:
        return InstrumentFilters[filter_name].value
    except KeyError:
        raise ValueError(f"Unknown filter: {filter_name}")
