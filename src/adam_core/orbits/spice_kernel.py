from typing import Dict, Optional

import numpy as np
import quivr as qv
import spiceypy as sp
from naif_de440 import de440
from naif_earth_itrf93 import earth_itrf93
from naif_eop_high_prec import eop_high_prec
from naif_eop_historical import eop_historical
from naif_eop_predict import eop_predict
from naif_leapseconds import leapseconds

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import OriginCodes
from ..coordinates.transform import transform_coordinates
from ..orbits import Orbits
from ..propagator import Propagator
from ..time import Timestamp

DEFAULT_KERNELS = [
    leapseconds,
    de440,
    eop_predict,
    eop_historical,
    eop_high_prec,
    earth_itrf93,
]


J2000_TDB_JD = 2451545.0


def fit_chebyshev(
    coordinates: CartesianCoordinates,
    window_start: float,
    window_end: float,
    degree: int,
) -> tuple[np.ndarray, float, float]:
    """
    Fit Chebyshev polynomials to position and velocity data.

    Parameters
    ----------
    coordinates : CartesianCoordinates
        Coordinates to fit
    window_start : float
        Start time in ET seconds
    window_end : float
        End time in ET seconds
    degree : int
        Degree of Chebyshev polynomials

    Returns
    -------
    tuple
        (coefficients, mid_time, half_interval)
        coefficients has shape (6, degree+1)
    """
    # Get states for this window and convert to km and km/s
    et_times = coordinates.time.et().to_numpy()
    mask = (et_times >= window_start) & (et_times <= window_end)
    states = coordinates.values[mask].copy()
    states[:, :3] *= 149597870.7  # au to km
    states[:, 3:] *= 149597870.7 / 86400.0  # au/day to km/s
    times = coordinates.time.et().to_numpy()[mask]

    # Scale time to [-1, 1] interval
    mid_time = (window_end + window_start) / 2
    half_interval = (window_end - window_start) / 2
    scaled_times = (times - mid_time) / half_interval

    # Compute Chebyshev polynomials
    T = np.zeros((len(times), degree + 1))
    T[:, 0] = 1
    T[:, 1] = scaled_times
    for i in range(2, degree + 1):
        T[:, i] = 2 * scaled_times * T[:, i - 1] - T[:, i - 2]

    # Fit polynomials to each component
    coefficients = np.zeros((6, degree + 1))
    for i in range(6):
        coefficients[i] = np.linalg.lstsq(T, states[:, i], rcond=None)[0]

    return coefficients, mid_time, half_interval


def orbits_to_spk(
    orbits: Orbits,
    output_file: str,
    start_time: Timestamp,
    end_time: Timestamp,
    propagator: Optional[Propagator] = None,
    max_processes: Optional[int] = None,
    step_days: float = 1.0 / 12, # Every 2 hours
    target_id_start: int = 1000000,
    cheby_degree: int = 15,
    window_days: float = 32.0,
    comment: str = "SPK file generated from adam_core Orbits",
) -> Dict[str, int]:
    """
    Convert Orbits object to a SPICE SPK file using Chebyshev polynomials (Type 3).
    """
    # Generate propagation times
    num_steps = (
        int((end_time.mjd().to_numpy() - start_time.mjd().to_numpy()) / step_days) + 1
    )

    times = qv.concatenate(
        [start_time.add_fractional_days(i * step_days) for i in range(num_steps)]
    )
    # Propagate orbits if propagator provided
    if propagator is not None:
        orbits = propagator.propagate_orbits(
            orbits, times, max_processes=max_processes
        )

    print(orbits.coordinates.origin.as_OriginCodes().value)
    print(orbits.coordinates.frame)
    
    # Transform everything to a Sun origin and
    # ecliptic frame
    # Verify all orbits have the same origin
    sun_coordinates = transform_coordinates(
        orbits.coordinates,
        frame_out="equatorial",
        origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    )

    orbits = orbits.set_column("coordinates", sun_coordinates)

    print(orbits.coordinates.origin.as_OriginCodes().value)
    print(orbits.coordinates.frame)

    # Create the SPK file
    handle = sp.spkopn(output_file, comment, 0)

    # Add each orbit as a separate segment
    window_seconds = window_days * 86400.0

    target_id_mappings = {}

    for i, (orbit_id, orbit) in enumerate(orbits.group_by_orbit_id()):
        # ensure orbit is sorted by time
        orbit = orbit.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

        target_id = target_id_start + i

        target_id_mappings[orbit_id] = target_id

        # Get time range for this orbit
        orbit_start = orbit.coordinates.time.min().et()[0].as_py()
        orbit_end = orbit.coordinates.time.max().et()[0].as_py()

        num_windows = int(np.ceil((orbit_end - orbit_start) / window_seconds))


        # Initialize arrays for SPKW02
        cheby_coeffs = []
        window_starts = []

        for w in range(num_windows):
            start_time = orbit_start + w * window_seconds
            end_time = min(start_time + window_seconds, orbit_end)

            # Fit Chebyshev polynomials
            coeffs, mid_time, half_interval = fit_chebyshev(
                orbit.coordinates, start_time, end_time, cheby_degree
            )

            cheby_coeffs.append(coeffs.flatten())
            window_starts.append(start_time)

        # Convert to numpy arrays
        cheby_coeffs = np.array(cheby_coeffs)
        window_starts = np.array(window_starts)
        # Write Type 3 SPK segment
        spk_frame = "J2000" if orbits.coordinates.frame == "equatorial" else "ECLIPJ2000"
        sp.spkw03(
            handle,
            target_id,
            orbits.coordinates.origin.as_OriginCodes().value,
            spk_frame,
            orbit_start,
            orbit_end,
            str(orbit_id),
            window_seconds,
            len(cheby_coeffs),
            cheby_degree,
            cheby_coeffs.flatten(),
            window_starts[0],
        )

    # Close the SPK file
    sp.spkcls(handle)

    return target_id_mappings
