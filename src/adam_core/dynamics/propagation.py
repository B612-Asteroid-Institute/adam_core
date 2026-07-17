from typing import Optional

import pyarrow as pa

from .._rust import propagate_orbits_arrow
from ..orbits.arrow_bridge import orbits_from_record_batch, orbits_to_record_batch
from ..orbits.orbits import Orbits
from ..time import Timestamp
from .exceptions import DynamicsNumericalError

_PROPAGATION_FAILURE_PREFIX = "propagation row failure:"


def _target_times_record_batch(times: Timestamp) -> pa.RecordBatch:
    """Expose target epochs as one metadata-stamped Arrow RecordBatch."""
    table = times.table.combine_chunks()
    metadata = dict(table.schema.metadata or {})
    metadata[b"adam_core_time_scale"] = times.scale.encode()
    table = table.replace_schema_metadata(metadata)
    arrays = [column.combine_chunks() for column in table.columns]
    return pa.RecordBatch.from_arrays(arrays, schema=table.schema)


def _raise_propagation_row_failure(error: RuntimeError) -> None:
    message = str(error)
    if not message.startswith(_PROPAGATION_FAILURE_PREFIX):
        raise error
    fields: dict[str, str] = {}
    for item in message.removeprefix(_PROPAGATION_FAILURE_PREFIX).strip().split(";"):
        key, separator, value = item.strip().partition("=")
        if separator:
            fields[key] = value
    reason = fields.pop("reason", "non_finite_output_state")
    context: dict[str, object] = {"rust_error": message}
    for key, value in fields.items():
        try:
            context[key] = int(value)
        except ValueError:
            context[key] = value
    raise DynamicsNumericalError(
        stage="propagation",
        reason=reason,
        context=context,
    ) from error


def _propagate_2body_serial(
    orbits: Orbits,
    times: Timestamp,
    *,
    max_iter: int,
    tol: float,
    chunk_size: int | None = None,
    thread_limit: int | None = None,
) -> Orbits:
    """One-crossing Arrow-native 2-body propagation implementation."""
    try:
        output = propagate_orbits_arrow(
            orbits_to_record_batch(orbits),
            _target_times_record_batch(times),
            max_iter=max_iter,
            tol=tol,
            chunk_size=chunk_size,
            thread_limit=thread_limit,
        )
    except RuntimeError as error:
        _raise_propagation_row_failure(error)
        raise AssertionError("unreachable") from error
    return orbits_from_record_batch(output)


def propagate_2body(
    orbits: Orbits,
    times: Timestamp,
    max_iter: int = 1000,
    tol: float = 1e-14,
    *,
    max_processes: Optional[int] = 1,
    chunk_size: int = 100,
) -> Orbits:
    """Propagate Cartesian orbits with two-body universal-anomaly dynamics.

    The complete orbit×epoch cross product runs in Rust behind one
    PyArrow→Rust→PyArrow crossing. Rust owns time-scale conversion, origin-mu
    lookup, Rayon chunking, covariance transport, physical-parameter
    repetition, stable row ordering, and output RecordBatch assembly. Python
    only materializes the two input RecordBatches and directly wraps the output.

    Parameters
    ----------
    orbits : `~adam_core.orbits.orbits.Orbits` (N)
        Cartesian orbits in au and au/day.
    times : `~adam_core.time.Timestamp` (M)
        Epochs to which every orbit is propagated. Output is orbit-major with
        target epoch order preserved within each orbit.
    max_iter : int, optional
        Maximum universal-anomaly solver iterations.
    tol : float, optional
        Universal-anomaly convergence tolerance.
    max_processes : int, optional
        Retained compatibility control for callers that previously selected an
        outer process count. No Python/Ray compute fan-out occurs; local
        parallelism remains owned by Rust's warmed global Rayon pool.
    chunk_size : int, optional
        Number of input orbits per Rust propagation chunk.

    Returns
    -------
    `~adam_core.orbits.orbits.Orbits` (N*M)
        Fully assembled propagated orbit table.
    """
    # The retired Python process-count option is accepted but ignored; warmed
    # local parallelism belongs to Rust's process-global Rayon pool.
    del max_processes
    return _propagate_2body_serial(
        orbits,
        times,
        max_iter=max_iter,
        tol=tol,
        chunk_size=chunk_size,
        thread_limit=None,
    )
