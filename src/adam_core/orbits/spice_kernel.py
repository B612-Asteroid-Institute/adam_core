import logging
from typing import Dict, Optional

import numpy as np
import quivr as qv
from astropy import units as u
from naif_de440 import de440
from naif_earth_itrf93 import earth_itrf93
from naif_eop_high_prec import eop_high_prec
from naif_eop_historical import eop_historical
from naif_eop_predict import eop_predict
from naif_leapseconds import leapseconds

from .._rust import naif_spk_writer
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

logger = logging.getLogger(__name__)

J2000_TDB_JD = 2451545.0

# NAIF frame IDs as stored in SPK segment summaries. CSPICE's
# `sp.spkw03`/`sp.spkw09` accept a frame name and translate internally;
# our writer serializes the DAF directly so we do the translation here.
_NAIF_FRAME_IDS = {
    "J2000": 1,
    "ECLIPJ2000": 17,
}


def fit_chebyshev(
    coordinates: CartesianCoordinates,
    window_start: float,
    window_end: float,
    degree: int,
    mid_time: Optional[float] = None,
    half_interval: Optional[float] = None,
) -> tuple[np.ndarray, float, float]:
    """Fit Chebyshev polynomials to position and velocity over a window.

    Parameters
    ----------
    coordinates : CartesianCoordinates
        Coordinates to fit (AU and AU/day).
    window_start, window_end : float
        Window bounds (ET seconds) used to select samples from `coordinates`.
    degree : int
        Degree of the Chebyshev fit.
    mid_time, half_interval : float, optional
        If provided, the fit uses these values to scale time to [-1, 1]
        instead of deriving them from `window_start`/`window_end`. Callers
        that need bit-parity with CSPICE's spkw03 pass the same MID /
        RADIUS values the reader will derive from INIT + INTLEN so the
        polynomial is evaluated at the same scaled time on readback.

    Returns
    -------
    coefficients : (6, degree+1) ndarray, mid_time : float, half_interval : float
    """
    et_times = coordinates.time.et().to_numpy()
    mask = (et_times >= window_start) & (et_times <= window_end)
    states = coordinates.values[mask].copy()
    states[:, :3] *= 149597870.7  # au -> km
    states[:, 3:] *= 149597870.7 / 86400.0  # au/day -> km/s
    times = coordinates.time.et().to_numpy()[mask]

    if mid_time is None:
        mid_time = (window_end + window_start) / 2
    if half_interval is None:
        half_interval = (window_end - window_start) / 2
    scaled_times = (times - mid_time) / half_interval

    T = np.zeros((len(times), degree + 1))
    T[:, 0] = 1
    T[:, 1] = scaled_times
    for i in range(2, degree + 1):
        T[:, i] = 2 * scaled_times * T[:, i - 1] - T[:, i - 2]

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
    """Convert Orbits to a SPICE SPK file using pure-Rust DAF serialization.

    Writes Type 3 (Chebyshev pos+vel) or Type 9 (Lagrange discrete state)
    segments through adam-core's native Rust SPK writer. Output bytes are
    assembled in memory and committed with an atomic rename so partial
    files never survive a crash. No CSPICE linkage is required.

    The ``comment`` argument is accepted for API compatibility with the
    legacy CSPICE-backed writer but is not currently serialized into the
    DAF comment area (the reader path doesn't need it).
    """
    logger.info(f"Creating SPK file: {output_file}")
    logger.info(
        f"Time range: {start_time.to_astropy().isot} to {end_time.to_astropy().isot}"
    )
    logger.info(f"Kernel type: {kernel_type}")

    writer = naif_spk_writer("adam-core")
    if writer is None:
        raise RuntimeError(
            "adam-core native SPK writer unavailable; SPK writing requires "
            "the compiled adam_core._rust_native extension."
        )

    cheby_degree = 15

    start_mjd = start_time.mjd().to_numpy(zero_copy_only=False).item()
    end_mjd = end_time.mjd().to_numpy(zero_copy_only=False).item()
    num_steps = int((end_mjd - start_mjd) / step_days) + 1
    logger.debug(f"Generated {num_steps} time steps with step size {step_days} days")

    times = qv.concatenate(
        [start_time.add_fractional_days(i * step_days) for i in range(num_steps)]
    )
    if propagator is not None:
        logger.debug("Propagating orbits...")
        orbits = propagator.propagate_orbits(
            orbits, times, max_processes=max_processes, chunk_size=1
        )
        logger.debug("Orbit propagation complete")

    ssb_coordinates = transform_coordinates(
        orbits.coordinates,
        frame_out="equatorial",
        origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    )
    orbits = orbits.set_column("coordinates", ssb_coordinates)

    target_id_mappings: Dict[str, int] = {}

    for i, (orbit_id, orbit) in enumerate(orbits.group_by_orbit_id()):
        logger.debug(
            f"Processing orbit {orbit_id} ({i} / ({len(orbits.orbit_id.unique())})"
        )
        orbit = orbit.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

        target_id = target_id_start + i
        target_id_mappings[orbit_id] = target_id
        logger.debug(f"Orbit {orbit_id} -> Target ID: {target_id}")

        orbit_start = orbit.coordinates.time.min().et()[0].as_py()
        orbit_end = orbit.coordinates.time.max().et()[0].as_py()

        if kernel_type == "w03":
            _add_type3_segment(
                writer,
                orbit,
                target_id,
                orbit_start,
                orbit_end,
                window_days * 86400.0,
                cheby_degree,
            )
        elif kernel_type == "w09":
            _add_type9_segment(writer, orbit, target_id)
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    writer.write(output_file)
    logger.info(f"Successfully created SPK file: {output_file}")

    return target_id_mappings


def _frame_id_for(frame: str) -> int:
    spice_name = "J2000" if frame == "equatorial" else "ECLIPJ2000"
    return _NAIF_FRAME_IDS[spice_name]


def _segment_id_for(orbit: Orbits, start_time: float, end_time: float) -> str:
    segment_id = (
        str(orbit.orbit_id[0].as_py()) + "_" + str(start_time) + "_" + str(end_time)
    )
    # SPICE has a 40-byte limit on segment IDs; preserve legacy truncation.
    return segment_id[:40]


def _add_type3_segment(
    writer,
    propagated_orbit: Orbits,
    target_id: int,
    start_time: float,
    end_time: float,
    window_seconds: float,
    cheby_degree: int,
) -> None:
    logger.debug(f"Writing SPK type 03 segment for target ID {target_id}")
    assert propagated_orbit.orbit_id.unique()

    frame_id = _frame_id_for(propagated_orbit.coordinates.frame)
    segment_id = _segment_id_for(propagated_orbit, start_time, end_time)

    num_windows = int(np.ceil((end_time - start_time) / window_seconds))
    logger.debug(f"Fitting {num_windows} Chebyshev windows")

    n_coef = cheby_degree + 1
    row_len = 2 + 6 * n_coef
    records_coeffs = np.empty((num_windows, row_len), dtype=np.float64)

    for w in range(num_windows):
        start_time_window = start_time + w * window_seconds
        end_time_window = min(start_time_window + window_seconds, end_time)

        # CSPICE's spkw03 stores MID / RADIUS per record derived from
        # INIT + INTLEN, not the clipped fit window. Use the same
        # convention for both the fit's time scaling and the on-disk
        # record so readback evaluates the polynomial at the same
        # scaled time the fitter saw.
        record_mid = start_time + (w + 0.5) * window_seconds
        record_half = window_seconds / 2.0
        coeffs, _mid, _half = fit_chebyshev(
            propagated_orbit.coordinates,
            start_time_window,
            end_time_window,
            cheby_degree,
            mid_time=record_mid,
            half_interval=record_half,
        )

        records_coeffs[w, 0] = record_mid
        records_coeffs[w, 1] = record_half
        # `coeffs` has shape (6, degree+1). Emit x, y, z, vx, vy, vz blocks
        # in that order to match the on-disk SPK Type 3 record layout.
        for comp in range(6):
            col_start = 2 + comp * n_coef
            records_coeffs[w, col_start : col_start + n_coef] = coeffs[comp]

    center_code = propagated_orbit.coordinates.origin.as_OriginCodes().value

    writer.add_type3(
        int(target_id),
        int(center_code),
        int(frame_id),
        float(start_time),
        float(end_time),
        segment_id,
        float(start_time),
        float(window_seconds),
        records_coeffs,
    )


def _add_type9_segment(
    writer,
    propagated_orbit: Orbits,
    target_id: int,
) -> None:
    logger.debug(f"Writing SPK type 09 segment for target ID {target_id}")
    assert propagated_orbit.orbit_id.unique()

    frame_id = _frame_id_for(propagated_orbit.coordinates.frame)

    epochs_tdb = (
        propagated_orbit.coordinates.time.rescale("tdb")
        .jd()
        .to_numpy(zero_copy_only=False)
    )
    # ET == seconds since J2000 TDB; pure-arithmetic equivalent of
    # sp.str2et(f"JD {jd_tdb} TDB") and avoids a per-epoch CSPICE round-trip.
    epochs_et = (epochs_tdb - J2000_TDB_JD) * 86400.0

    states = propagated_orbit.coordinates.values.copy()
    states[:, 0:6] *= u.au.to(u.km)
    states[:, 3:6] /= (u.d).to(u.s)

    segment_id = _segment_id_for(
        propagated_orbit, float(epochs_et[0]), float(epochs_et[-1])
    )
    center_code = propagated_orbit.coordinates.origin.as_OriginCodes().value

    writer.add_type9(
        int(target_id),
        int(center_code),
        int(frame_id),
        float(epochs_et[0]),
        float(epochs_et[-1]),
        segment_id,
        15,
        np.ascontiguousarray(states, dtype=np.float64),
        np.ascontiguousarray(epochs_et, dtype=np.float64),
    )


def write_spkw03_segment(
    propagated_orbit: Orbits,
    handle,
    target_id: int,
    start_time: float,
    end_time: float,
    window_seconds: float = 86400.0,
    cheby_degree: int = 15,
) -> None:
    """Backward-compat shim for callers that passed a writer-like handle.

    The `handle` parameter is accepted for API compatibility with the
    legacy CSPICE-backed helper and must be a native `NaifSpkWriter` instance.
    """
    _add_type3_segment(
        handle,
        propagated_orbit,
        target_id,
        start_time,
        end_time,
        window_seconds,
        cheby_degree,
    )


def write_spkw09_segment(
    propagated_orbit: Orbits,
    handle,
    target_id: int,
    start_time: float,
    end_time: float,
) -> None:
    """Backward-compat shim for callers that passed a writer-like handle."""
    _add_type9_segment(handle, propagated_orbit, target_id)
