"""Arrow-native two-body ephemeris generation."""

from typing import Optional

from .._rust.api import generate_ephemeris_arrow
from .._rust.arrow import ensure_spice_backend, table_from_record_batch
from ..observers.observers import Observers
from ..orbits.arrow_bridge import observers_to_record_batch, orbits_to_record_batch
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from .exceptions import DynamicsNumericalError


def _numerical_error(message: str) -> DynamicsNumericalError:
    reason = message.partition("reason=")[2].partition(";")[0] or "ephemeris_failure"
    context: dict[str, object] = {}
    for item in message.split("; ")[1:]:
        key, separator, value = item.partition("=")
        if not separator:
            continue
        try:
            context[key] = int(value)
        except ValueError:
            context[key] = value
    return DynamicsNumericalError(stage="ephemeris", reason=reason, context=context)


def generate_ephemeris_2body(
    propagated_orbits: Orbits,
    observers: Observers,
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
    predict_magnitudes: bool = True,
    *,
    predict_phase_angle: bool = False,
    max_processes: Optional[int] = 1,
    chunk_size: int = 100,
) -> Ephemeris:
    """Generate paired topocentric ephemerides entirely in Rust.

    ``propagated_orbits`` and ``observers`` are handed to Rust as two nested
    RecordBatches. Rust owns frame/origin normalization, pairwise propagation,
    light-time and stellar-aberration correction, covariance transport,
    photometry, diagnostics, and construction of the finished Ephemeris batch.
    Python directly wraps that returned RecordBatch without rebuilding columns.

    ``max_processes`` remains an accepted, ignored compatibility option; no
    Python process fan-out occurs and local parallelism uses Rust's warmed
    global Rayon pool.
    """
    if len(propagated_orbits) != len(observers):
        raise AssertionError(
            "Orbits and observers must be paired and orbits must be propagated "
            "to observer times."
        )
    del max_processes
    ensure_spice_backend()
    try:
        result = generate_ephemeris_arrow(
            orbits_to_record_batch(propagated_orbits),
            observers_to_record_batch(observers),
            lt_tol=lt_tol,
            max_iter=max_iter,
            tol=tol,
            stellar_aberration=stellar_aberration,
            predict_magnitudes=predict_magnitudes,
            predict_phase_angle=predict_phase_angle,
            chunk_size=chunk_size,
            thread_limit=None,
        )
    except RuntimeError as error:
        message = str(error)
        if message.startswith("ephemeris row failure:"):
            raise _numerical_error(message) from error
        raise
    return table_from_record_batch(Ephemeris, result)
