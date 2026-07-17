"""Shared Arrow C Data Interface crossing helpers (canonical in-process bridge).

The canonical Python<->Rust crossing for composed / typed-table surfaces
(bead personal-cmy.36): a quivr ``Table`` is handed to Rust as a pyarrow
``RecordBatch`` and the Rust-produced ``RecordBatch`` is wrapped straight back
into a quivr ``Table``. Rust owns the setup, composition, and table assembly;
the Python veneer stays thin. This replaces the numpy-flat marshaling in
``adam_core._rust.api`` for composed surfaces (there is no per-column
``.to_numpy()`` split and no ``from_kwargs`` rebuild on the way back).

This module intentionally holds only the pieces that are shared across every
surface and exercised today:

* :func:`stamp_adam_core_metadata` / :func:`to_quivr_metadata` -- the canonical
  translation between quivr attribute metadata (``coordinates.frame`` /
  ``coordinates.time.scale``) and the ``adam_core_*`` schema keys the Rust codec
  reads. ``adam_core.orbits.arrow_bridge`` imports these so the mapping lives in
  one place.
* :func:`table_from_record_batch` -- generic output wrapper used by every
  surface to turn a Rust ``RecordBatch`` into its quivr ``Table``.
* :func:`ensure_spice_backend` -- idempotent SPICE + MPC-obscodes setup that
  returns the process-global Rust backend, so veneers stop orchestrating
  ``setup_SPICE`` / ``get_backend`` / obscodes loading inline per call.

Input-side builders are surface-specific (each surface packs different columns)
and live with their surface; they are added as surfaces convert onto this
bridge rather than being guessed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type, TypeVar

import pyarrow as pa

if TYPE_CHECKING:
    import quivr as qv

    _TableT = TypeVar("_TableT", bound=qv.Table)

_ADAM_CORE_SCHEMA_VERSION = b"1"


def stamp_adam_core_metadata(
    table: pa.Table,
    representation: str,
    frame: str,
    scale: str,
    schema_name: str,
) -> pa.Table:
    """Stamp the canonical ``adam_core_*`` schema metadata the Rust codec reads."""
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"adam_core_schema": schema_name.encode(),
            b"adam_core_schema_version": _ADAM_CORE_SCHEMA_VERSION,
            b"adam_core_representation": representation.encode(),
            b"adam_core_frame": frame.encode(),
            b"adam_core_time_scale": scale.encode(),
        }
    )
    return table.replace_schema_metadata(metadata)


def to_quivr_metadata(table: pa.Table) -> pa.Table:
    """Translate Rust ``adam_core_*`` metadata back into quivr attribute keys."""
    metadata = dict(table.schema.metadata or {})
    frame = metadata.get(b"adam_core_frame", b"unspecified").decode()
    scale = metadata.get(b"adam_core_time_scale", b"utc").decode()
    prefix = b"coordinates." if "coordinates" in table.column_names else b""
    quivr_metadata = {
        key: value
        for key, value in metadata.items()
        if b"." in key and not key.startswith(b"adam_core")
    }
    quivr_metadata.update(
        {
            prefix + b"frame": frame.encode(),
            prefix + b"time.scale": scale.encode(),
        }
    )
    if "aberrated_coordinates" in table.column_names:
        aberrated_frame = metadata.get(b"adam_core_aberrated_frame", b"ecliptic")
        aberrated_scale = metadata.get(
            b"adam_core_aberrated_time_scale", scale.encode()
        )
        quivr_metadata.update(
            {
                b"aberrated_coordinates.frame": aberrated_frame,
                b"aberrated_coordinates.time.scale": aberrated_scale,
            }
        )
    return table.replace_schema_metadata(quivr_metadata)


def table_from_record_batch(
    cls: "Type[_TableT]", record_batch: pa.RecordBatch
) -> "_TableT":
    """Wrap a Rust-produced nested ``RecordBatch`` as its quivr ``Table``.

    The single output half of the canonical crossing: translate the Rust
    ``adam_core_*`` schema metadata to quivr attribute keys and reconstruct the
    table with ``from_pyarrow`` (no numpy, no ``from_kwargs``).
    """
    table = pa.Table.from_batches([record_batch])
    return cls.from_pyarrow(to_quivr_metadata(table))


def ensure_spice_backend():
    """Idempotent SPICE + MPC-obscodes setup; return the process-global backend.

    Centralizes the per-call ``setup_SPICE`` / ``get_backend`` / obscodes wiring
    so surface veneers do not orchestrate it inline. Both setup calls are
    idempotent (kernels are furnsh-idempotent by path; obscodes load only when
    the backend reports zero loaded sites). Moving this initialization fully
    Rust-side (lazy on first use) is tracked under personal-cmy.36.1; the kernel
    data provenance that blocks it is personal-3uy.
    """
    from ..utils.spice import get_backend, setup_mpc_obscodes, setup_SPICE

    setup_SPICE()
    setup_mpc_obscodes()
    return get_backend()
