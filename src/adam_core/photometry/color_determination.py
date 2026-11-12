"""Canonical-filter-generic asteroid color estimation.

The public facade in this module intentionally owns no scientific computation.
It marshals native ADAM tables and configuration through one private IPC adapter
that delegates to the public Rust API, then validates the exact Arrow result.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pyarrow as pa
import quivr as qv

from ..observations import SourceCatalog
from ..observations.arrow_bridge import observations_to_ipc
from ..orbits import Orbits
from ..orbits.arrow_bridge import orbits_to_ipc
from ..utils.spice import DEFAULT_KERNELS
from .bandpasses import BandpassComposition
from .bandpasses.api import _composition_args


class ColorFit(qv.Table):
    """One per-object color-fit outcome and aligned canonical-filter products.

    Rust returns this exact schema. For ``status == "ok"``, every list is
    non-null. ``filter_ids`` contains final active canonical filters in
    deterministic lexical order; absolute magnitudes, their sigmas, per-filter
    used counts, and the row-major magnitude covariance align with that order.
    The four pair lists enumerate every ``lhs < rhs`` pair and define
    ``color = H_lhs - H_rhs``. A one-filter result has present empty pair lists.

    List children are never Arrow null. Floating-point non-identifiability is
    represented by NaN. Forced ``phase_param_out_of_bounds`` retains only
    ``filter_ids`` and ``num_used_by_filter`` among the lists; every other
    non-ok status has all list fields null. Scalar fit diagnostics remain when
    a fit ran. ``no_orbit`` and ``multiple_orbits`` retain ``num_obs`` but have
    null classification counts; failures after row classification may retain
    counts, where ``num_used`` means rows admitted to the attempted fit. For
    every classified object the aggregate partition is exact::

        num_obs = num_invalid + num_unsupported_filter + num_clipped + num_used

    See ``migration/public_surface/photometry_constants.md`` for the complete
    Rust-native contract.
    """

    object_id = qv.LargeStringColumn()
    status = qv.LargeStringColumn()
    error = qv.LargeStringColumn(nullable=True)
    filter_ids = qv.LargeListColumn(
        pa.field("item", pa.large_string(), nullable=False), nullable=True
    )
    absolute_magnitudes = qv.LargeListColumn(
        pa.field("item", pa.float64(), nullable=False), nullable=True
    )
    absolute_magnitude_sigmas = qv.LargeListColumn(
        pa.field("item", pa.float64(), nullable=False), nullable=True
    )
    num_used_by_filter = qv.LargeListColumn(
        pa.field("item", pa.int64(), nullable=False), nullable=True
    )
    magnitude_covariance = qv.LargeListColumn(
        pa.field("item", pa.float64(), nullable=False), nullable=True
    )
    color_lhs_filter_ids = qv.LargeListColumn(
        pa.field("item", pa.large_string(), nullable=False), nullable=True
    )
    color_rhs_filter_ids = qv.LargeListColumn(
        pa.field("item", pa.large_string(), nullable=False), nullable=True
    )
    colors = qv.LargeListColumn(
        pa.field("item", pa.float64(), nullable=False), nullable=True
    )
    color_sigmas = qv.LargeListColumn(
        pa.field("item", pa.float64(), nullable=False), nullable=True
    )
    phase_param = qv.Float64Column(nullable=True)
    phase_param_sigma = qv.Float64Column(nullable=True)
    chi2 = qv.Float64Column(nullable=True)
    reduced_chi2 = qv.Float64Column(nullable=True)
    dof = qv.Int64Column(nullable=True)
    rank = qv.Int64Column(nullable=True)
    num_params = qv.Int64Column(nullable=True)
    converged = qv.BooleanColumn(nullable=True)
    num_obs = qv.Int64Column(nullable=True)
    num_invalid = qv.Int64Column(nullable=True)
    num_unsupported_filter = qv.Int64Column(nullable=True)
    num_clipped = qv.Int64Column(nullable=True)
    num_used = qv.Int64Column(nullable=True)


def _color_fit_from_native_batch(record_batch: pa.RecordBatch) -> ColorFit:
    """Validate the raw native schema exactly before adding quivr metadata."""
    if not record_batch.schema.equals(ColorFit.schema, check_metadata=True):
        raise ValueError(
            "Rust color-fit RecordBatch schema does not exactly match ColorFit.schema: "
            f"expected {ColorFit.schema}, got {record_batch.schema}"
        )
    from .._rust.arrow import table_from_record_batch

    return table_from_record_batch(ColorFit, record_batch)


@lru_cache(maxsize=1)
def _mpc_obscodes_json() -> str:
    """Read immutable installed observatory data for native setup."""
    from mpc_obscodes import mpc_obscodes

    return Path(mpc_obscodes).read_text(encoding="utf-8")


def _native_record_batch(raw: bytes) -> pa.RecordBatch:
    """Decode the native one-batch IPC result without deriving any columns."""
    with pa.ipc.open_stream(pa.py_buffer(raw)) as reader:
        try:
            batch = reader.read_next_batch()
        except StopIteration as error:
            raise ValueError(
                "Rust color-fit IPC stream contains no record batch"
            ) from error
        try:
            reader.read_next_batch()
        except StopIteration:
            return batch
    raise ValueError("Rust color-fit IPC stream contains more than one record batch")


def estimate_colors(
    observations: SourceCatalog,
    orbits: Orbits,
    phi_type: Literal["HG12star", "HG", "c1c2"],
    force_g_bounds: bool = True,
    color_term_composition: BandpassComposition | None = None,
    strict_band_mapping: bool = False,
    filter_projection: Mapping[str, str] | None = None,
) -> ColorFit:
    """Estimate canonical-filter absolute magnitudes and colors per object.

    This is a thin facade over the public high-level
    ``adam_core::estimate_colors`` Rust API through one private IPC adapter.
    Rust owns object grouping and isolation, orbit matching and propagation,
    observer geometry, validity classification, canonical filter resolution,
    optional projection, every phase-law fit, clipping, identifiability,
    covariance, diagnostics,
    statuses/counts, and final Arrow assembly.

    Parameters
    ----------
    observations
        Native :class:`~adam_core.observations.SourceCatalog`. Rows with null
        ``object_id`` are skipped; every attributable row is retained for
        diagnostic accounting.
    orbits
        Native :class:`~adam_core.orbits.Orbits`. A nonempty table must use
        Cartesian ecliptic coordinates; frame/representation/epoch errors are
        call-level validation errors. Heliocentric origin is checked per matched
        orbit. Objects with zero or multiple matching orbits receive isolated
        status rows; an empty table assigns ``no_orbit`` to every object.
        Active non-gravitational solutions are rejected per object.
    phi_type
        ``"HG12star"``, ``"HG"``, or ``"c1c2"``.
    force_g_bounds
        If true, withhold scientific list products when a fitted slope is
        outside its model validity bounds: ``G12*`` in ``[0, 1]`` under
        ``"HG12star"`` and ``G`` in ``[-0.255, 1]`` under ``"HG"``. It has no
        effect with ``"c1c2"``. Fit diagnostics and filter/count lists remain
        available.
    color_term_composition
        Reflectance template ID or ``(weight_C, weight_S)`` mix required by
        ``filter_projection``. It is invalid without an explicit projection.
    strict_band_mapping
        Disable only generic SDSS/PS1 fallback mapping. Canonical IDs and exact
        observatory mappings remain accepted; unresolved rows are counted as
        unsupported instead of aborting the batch.
    filter_projection
        Explicit canonical source-to-target mapping. Only listed sources are
        eligible, including identity edges; Rust converts using
        ``color_term_composition`` before pooling and emits target IDs. It is
        invalid without a composition. ``None`` (default) is native mode: every
        resolved canonical filter is fitted as a distinct intercept.

    Returns
    -------
    ColorFit
        One row per first-appearing non-null observation object ID. Status is
        one of ``"ok"``, ``"no_orbit"``, ``"multiple_orbits"``,
        ``"no_usable_photometry"``, ``"too_many_outliers"``,
        ``"phase_param_out_of_bounds"``, or ``"failed"``. ``no_orbit``,
        ``multiple_orbits``, and failures before row classification retain only
        ``error`` and ``num_obs`` among nullable scalar diagnostics/counts.
        Failures after classification retain an exact count partition, where
        ``num_used`` means rows admitted to the attempted fit.

    Raises
    ------
    ValueError
        If whole-call configuration, projection/composition pairing, the
        nonempty orbit table's frame/representation/epochs, native setup, IPC,
        or returned schema is invalid. A matching orbit with a non-heliocentric
        origin is instead isolated as an object-level ``failed`` row.
    """
    from adam_core import _rust_native

    template_id: str | None = None
    mix: tuple[float, float] | None = None
    if color_term_composition is not None:
        template_id, mix = _composition_args(color_term_composition)

    raw = _rust_native.estimate_colors_ipc(
        observations_to_ipc(observations),
        orbits_to_ipc(orbits),
        list(map(str, DEFAULT_KERNELS)),
        _mpc_obscodes_json(),
        phi_type,
        force_g_bounds=force_g_bounds,
        strict_band_mapping=strict_band_mapping,
        projection=None if filter_projection is None else dict(filter_projection),
        template_id=template_id,
        mix=mix,
    )
    return _color_fit_from_native_batch(_native_record_batch(raw))
