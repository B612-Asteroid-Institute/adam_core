import logging
from typing import Dict, Optional

import numpy as np
import quivr as qv
import spiceypy as sp
from astropy import units as u
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
from ..utils.spice import setup_SPICE

DEFAULT_KERNELS = [
    leapseconds,
    de440,
    eop_predict,
    eop_historical,
    eop_high_prec,
    earth_itrf93,
]

# Add after DEFAULT_KERNELS
logger = logging.getLogger(__name__)

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
    step_days: float = 1.0 / 4,  # Every 6 hours
    target_id_start: int = 1000000,
    window_days: float = 32.0,
    comment: str = "SPK file generated from adam_core Orbits",
    kernel_type: str = "w03",
) -> Dict[str, int]:
    """
    Convert Orbits object to a SPICE SPK file using Chebyshev polynomials (Type 3).
    """
    logger.info(f"Creating SPK file: {output_file}")
    logger.info(
        f"Time range: {start_time.to_astropy().isot} to {end_time.to_astropy().isot}"
    )
    logger.info(f"Kernel type: {kernel_type}")

    # ensure SPICE is ready to go
    setup_SPICE()

    # default to a chebyshev degree of 15
    cheby_degree = 15

    # Generate propagation times
    num_steps = (
        int((end_time.mjd().to_numpy() - start_time.mjd().to_numpy()) / step_days) + 1
    )
    logger.debug(f"Generated {num_steps} time steps with step size {step_days} days")

    times = qv.concatenate(
        [start_time.add_fractional_days(i * step_days) for i in range(num_steps)]
    )
    # Propagate orbits if propagator provided
    if propagator is not None:
        logger.debug("Propagating orbits...")
        orbits = propagator.propagate_orbits(
            orbits, times, max_processes=max_processes, chunk_size=1
        )
        logger.debug("Orbit propagation complete")

    # Transform everything to a Sun origin and
    # ecliptic frame
    # Verify all orbits have the same origin
    ssb_coordinates = transform_coordinates(
        orbits.coordinates,
        frame_out="equatorial",
        origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    )

    orbits = orbits.set_column("coordinates", ssb_coordinates)

    # Create the SPK file
    handle = sp.spkopn(output_file, comment, 0)
    target_id_mappings = {}

    for i, (orbit_id, orbit) in enumerate(orbits.group_by_orbit_id()):
        logger.debug(
            f"Processing orbit {orbit_id} ({i} / ({len(orbits.orbit_id.unique())})"
        )
        # ensure orbit is sorted by time
        orbit = orbit.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

        target_id = target_id_start + i

        target_id_mappings[orbit_id] = target_id
        logger.debug(f"Orbit {orbit_id} -> Target ID: {target_id}")

        # Get time range for this orbit
        orbit_start = orbit.coordinates.time.min().et()[0].as_py()
        orbit_end = orbit.coordinates.time.max().et()[0].as_py()

        if kernel_type == "w03":
            write_spkw03_segment(
                orbit,
                handle,
                target_id,
                orbit_start,
                orbit_end,
                window_days * 86400.0,
                cheby_degree,
            )
        elif kernel_type == "w09":
            write_spkw09_segment(
                orbit,
                handle,
                target_id,
                orbit_start,
                orbit_end,
            )
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    # Close the SPK file
    sp.spkcls(handle)
    logger.info(f"Successfully created SPK file: {output_file}")

    return target_id_mappings


def write_spkw03_segment(
    propagated_orbit: Orbits,
    handle: int,
    target_id: int,
    start_time: float,
    end_time: float,
    window_seconds: float = 86400.0,
    cheby_degree: int = 15,
) -> None:
    logger.debug(f"Writing SPK type 03 segment for target ID {target_id}")
    # assert orbit id is unique
    assert propagated_orbit.orbit_id.unique()

    spk_frame = (
        "J2000" if propagated_orbit.coordinates.frame == "equatorial" else "ECLIPJ2000"
    )

    segment_id = (
        str(propagated_orbit.orbit_id[0].as_py())
        + "_"
        + str(start_time)
        + "_"
        + str(end_time)
    )

    # SPICE has a limit of 40 characters for the segment id
    segment_id = segment_id[:40]

    num_windows = int(np.ceil((end_time - start_time) / window_seconds))
    logger.debug(f"Fitting {num_windows} Chebyshev windows")

    # Initialize arrays for SPKW03
    cheby_coeffs = []
    window_starts = []

    for w in range(num_windows):

        start_time_window = start_time + w * window_seconds
        end_time_window = min(start_time_window + window_seconds, end_time)

        coeffs, mid_time, half_interval = fit_chebyshev(
            propagated_orbit.coordinates,
            start_time_window,
            end_time_window,
            cheby_degree,
        )

        cheby_coeffs.append(coeffs.flatten())
        window_starts.append(start_time)

    # Convert to numpy arrays
    cheby_coeffs = np.array(cheby_coeffs)
    window_starts = np.array(window_starts)

    # Write the SPKW03 segment
    sp.spkw03(
        handle,
        target_id,
        propagated_orbit.coordinates.origin.as_OriginCodes().value,
        spk_frame,
        start_time,
        end_time,
        segment_id,
        window_seconds,
        len(cheby_coeffs),
        cheby_degree,
        cheby_coeffs.flatten(),
        window_starts[0],
    )


def write_spkw09_segment(
    propagated_orbit: Orbits,
    handle: int,
    target_id: int,
    start_time: float,
    end_time: float,
) -> None:
    logger.debug(f"Writing SPK type 09 segment for target ID {target_id}")
    # assert orbit id is unique
    assert propagated_orbit.orbit_id.unique()

    spk_frame = (
        "J2000" if propagated_orbit.coordinates.frame == "equatorial" else "ECLIPJ2000"
    )

    segment_id = (
        str(propagated_orbit.orbit_id[0].as_py())
        + "_"
        + str(start_time)
        + "_"
        + str(end_time)
    )

    # SPICE has a limit of 40 characters for the segment id
    segment_id = segment_id[:40]

    epochs_tdb = (
        propagated_orbit.coordinates.time.rescale("tdb")
        .jd()
        .to_numpy(zero_copy_only=False)
    )
    epochs_et = np.array([sp.str2et(f"JD {i:.15f} TDB".format(i)) for i in epochs_tdb])

    states = propagated_orbit.coordinates.values
    states[:, 0:6] *= u.au.to(u.km)
    states[:, 3:6] /= (u.d).to(u.s)

    sp.spkw09(
        handle,
        target_id,
        propagated_orbit.coordinates.origin.as_OriginCodes().value,
        spk_frame,
        epochs_et[0],
        epochs_et[-1],
        segment_id,
        15,
        len(epochs_et),
        np.ascontiguousarray(states),
        epochs_et,
    )
