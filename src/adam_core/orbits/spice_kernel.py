import logging
from typing import Dict, Optional

import numpy as np
import quivr as qv
from naif_de440 import de440
from naif_earth_itrf93 import earth_itrf93
from naif_eop_high_prec import eop_high_prec
from naif_eop_historical import eop_historical
from naif_eop_predict import eop_predict
from naif_leapseconds import leapseconds

from .._rust import naif_spk_writer as naif_spk_writer  # noqa: F401
from ..coordinates.cartesian import CartesianCoordinates
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
    from adam_core import _rust_native as _rn

    from .arrow_bridge import coordinates_to_ipc

    return _rn.spk_fit_chebyshev(
        coordinates_to_ipc(coordinates, "cartesian"),
        window_start,
        window_end,
        degree,
        mid_time,
        half_interval,
    )


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

    if propagator is not None:
        start_mjd = start_time.mjd().to_numpy(zero_copy_only=False).item()
        end_mjd = end_time.mjd().to_numpy(zero_copy_only=False).item()
        num_steps = int((end_mjd - start_mjd) / step_days) + 1
        logger.debug(
            f"Generated {num_steps} time steps with step size {step_days} days"
        )
        times = qv.concatenate(
            [start_time.add_fractional_days(i * step_days) for i in range(num_steps)]
        )
        logger.debug("Propagating orbits...")
        orbits = propagator.propagate_orbits(
            orbits, times, max_processes=max_processes, chunk_size=1
        )
        logger.debug("Orbit propagation complete")

    from adam_core import _rust_native as _rn

    from .arrow_bridge import orbits_to_ipc

    mappings = _rn.spk_write_orbits_product(
        orbits_to_ipc(orbits),
        output_file,
        target_id_start,
        window_days,
        kernel_type,
    )
    logger.info(f"Successfully created SPK file: {output_file}")
    return dict(mappings)


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
    from .arrow_bridge import orbits_to_ipc

    handle.add_type3_orbits(
        orbits_to_ipc(propagated_orbit),
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
    from .arrow_bridge import orbits_to_ipc

    handle.add_type9_orbits(orbits_to_ipc(propagated_orbit), target_id)
