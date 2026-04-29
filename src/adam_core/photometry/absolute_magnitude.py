from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..observations.detections import PointSourceDetections
from ..observations.exposures import Exposures
from ..orbits.physical_parameters import PhysicalParameters
from .bandpasses.api import map_to_canonical_filter_bands
from .magnitude import predict_magnitudes
from .magnitude_common import BandpassComposition


class GroupedPhysicalParameters(qv.Table):
    object_id = qv.LargeStringColumn()
    physical_parameters = PhysicalParameters.as_column()
    n_fit_detections = qv.Int64Column()


def _as_float64_nan(a: pa.Array | pa.ChunkedArray) -> npt.NDArray[np.float64]:
    # Arrow nulls map to NaN in the output.
    return np.asarray(a.to_numpy(zero_copy_only=False), dtype=np.float64)


def _mad_sigma(x: npt.NDArray[np.float64]) -> float:
    # Robust scale estimate: 1.4826 * MAD for Gaussian-consistent sigma.
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return 1.4826 * mad


def _fit_absolute_magnitude_rows(
    *,
    h_rows: npt.NDArray[np.float64],
    sigma_rows: npt.NDArray[np.float64],
) -> tuple[float, float | None, float | None, float | None, int]:
    """
    Fit H statistics for one grouped set of rows. Dispatches to the rust
    kernel `fit_absolute_magnitude_rows_numpy` and translates NaN sentinels
    in the result back to None.

    Returns
    -------
    (H_hat, H_sigma, sigma_eff, chi2_red, n_used)
    """
    from .._rust import fit_absolute_magnitude_rows_numpy as _rust_fit

    n_used = int(h_rows.size)
    if n_used <= 0:
        raise ValueError("h_rows must be non-empty")

    out = _rust_fit(
        np.ascontiguousarray(h_rows, dtype=np.float64),
        np.ascontiguousarray(sigma_rows, dtype=np.float64),
    )
    H_hat, H_sigma, sigma_eff, chi2_red, n_used_out = out

    if not np.isfinite(H_hat):
        # Match legacy ValueError on degenerate input (all-NaN rows or
        # weighted-mean wsum non-finite); rust returns NaN as sentinel.
        raise ValueError("invalid weights derived from mag_sigma")

    return (
        float(H_hat),
        None if not np.isfinite(H_sigma) else float(H_sigma),
        None if not np.isfinite(sigma_eff) else float(sigma_eff),
        None if not np.isfinite(chi2_red) else float(chi2_red),
        int(n_used_out),
    )


def estimate_absolute_magnitude_v_from_detections(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
    *,
    composition: BandpassComposition,
    G: float = 0.15,
    strict_band_mapping: bool = False,
    reference_filter: str = "V",
) -> PhysicalParameters:
    """
    Estimate V-band absolute magnitude H from observed apparent magnitudes.

    Assumptions
    -----------
    - Orbit has already been fit; `object_coords` are the heliocentric object coordinates at the
      observation times (aligned 1:1 with `detections`).
    - We estimate H only; `G` and `composition` are treated as fixed inputs.

    Parameters
    ----------
    detections
        Point-source detections. Uses `mag` and (optionally) `mag_sigma`.
    exposures
        Exposures referenced by `detections.exposure_id`. Uses `observatory_code` and `filter`.
    object_coords
        Object coordinates aligned with detections (same length and ordering).
    composition
        Required. Bandpass template ID (e.g. 'NEO') or (weight_C, weight_S) mix.
    G
        Fixed H-G slope parameter.
    strict_band_mapping
        If True, disallow SDSS/PS1 fallback filters when mapping reported bands.
    reference_filter
        Must be 'V' for this function.
    """
    if reference_filter != "V":
        raise ValueError(
            "reference_filter must be 'V' for estimate_absolute_magnitude_v_from_detections"
        )

    n_det = len(detections)
    if n_det == 0:
        raise ValueError("detections must be non-empty")
    if len(object_coords) != n_det:
        raise ValueError(
            f"object_coords length ({len(object_coords)}) must match detections length ({n_det})"
        )

    # ---------------------------------------------------------------------
    # Align exposures to detections order (vectorized join by exposure_id).
    # ---------------------------------------------------------------------
    if pc.any(pc.is_null(detections.exposure_id)).as_py():
        raise ValueError("detections.exposure_id must be non-null to link to exposures")

    idx = pc.fill_null(pc.index_in(detections.exposure_id, value_set=exposures.id), -1)
    idx_np = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(idx_np < 0):
        missing = np.unique(
            np.asarray(
                detections.exposure_id.to_numpy(zero_copy_only=False), dtype=object
            )[idx_np < 0].astype(str)
        ).tolist()
        raise ValueError(f"detections reference unknown exposure_id(s): {missing}")

    exposures_aligned = exposures.take(pa.array(idx_np, type=pa.int32()))

    # Resolve (observatory_code, band/filter) -> canonical vendored filter_id.
    canonical = map_to_canonical_filter_bands(
        exposures_aligned.observatory_code,
        exposures_aligned.filter,
        allow_fallback_filters=not strict_band_mapping,
    )
    exposures_canon = exposures_aligned.set_column(
        "filter", pa.array(canonical.tolist(), type=pa.large_string())
    )

    # ---------------------------------------------------------------------
    # Forward model once at H=0 to get per-row offset, then solve H.
    # ---------------------------------------------------------------------
    m0 = predict_magnitudes(
        H=0.0,
        object_coords=object_coords,
        exposures=exposures_canon,
        G=G,
        reference_filter="V",
        composition=composition,
    )

    mag = _as_float64_nan(detections.mag)
    valid = np.isfinite(mag) & np.isfinite(m0)
    n_used = int(np.count_nonzero(valid))
    if n_used == 0:
        raise ValueError(
            "no valid rows: need finite detections.mag and forward-model magnitudes"
        )

    H_i = mag[valid] - np.asarray(m0, dtype=np.float64)[valid]
    mag_sigma = _as_float64_nan(detections.mag_sigma)
    sigma_used = mag_sigma[valid]
    H_hat, H_sigma, sigma_eff, chi2_red, _ = _fit_absolute_magnitude_rows(
        h_rows=np.asarray(H_i, dtype=np.float64),
        sigma_rows=np.asarray(sigma_used, dtype=np.float64),
    )

    # Represent this as a 1-row PhysicalParameters table so callers can directly attach
    # to `Orbits.physical_parameters`.
    return PhysicalParameters.from_kwargs(
        H_v=[H_hat],
        H_v_sigma=[H_sigma],
        G=[float(G)],
        G_sigma=[None],
        sigma_eff=[sigma_eff],
        chi2_red=[chi2_red],
    )


def estimate_absolute_magnitude_v_from_detections_grouped(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
    object_ids: pa.Array | pa.ChunkedArray | Sequence[str | None],
    *,
    composition: BandpassComposition,
    G: float = 0.15,
    strict_band_mapping: bool = False,
    reference_filter: str = "V",
) -> GroupedPhysicalParameters:
    """
    Vectorized grouped H-fit for many objects in one pass.

    Parameters
    ----------
    detections
        Point-source detections for all groups.
    exposures
        Exposure table referenced by `detections.exposure_id`.
    object_coords
        Object coordinates aligned 1:1 with detections.
    object_ids
        Group label for each detection row (same length as detections).

    Returns
    -------
    GroupedPhysicalParameters
        One row per object_id with nested physical parameters and fit row counts.
    """
    if reference_filter != "V":
        raise ValueError(
            "reference_filter must be 'V' for estimate_absolute_magnitude_v_from_detections_grouped"
        )
    n_det = len(detections)
    if n_det == 0:
        return GroupedPhysicalParameters.from_kwargs(
            object_id=[],
            physical_parameters=PhysicalParameters.empty(),
            n_fit_detections=[],
        )
    if len(object_coords) != n_det:
        raise ValueError(
            f"object_coords length ({len(object_coords)}) must match detections length ({n_det})"
        )

    ids_arr = (
        pa.array(object_ids, type=pa.large_string())
        if not isinstance(object_ids, (pa.Array, pa.ChunkedArray))
        else pc.cast(object_ids, pa.large_string())
    )
    ids_np = np.asarray(ids_arr.to_numpy(zero_copy_only=False), dtype=object)
    if ids_np.size != n_det:
        raise ValueError(
            f"object_ids length ({ids_np.size}) must match detections length ({n_det})"
        )

    if pc.any(pc.is_null(detections.exposure_id)).as_py():
        raise ValueError("detections.exposure_id must be non-null to link to exposures")
    idx = pc.fill_null(pc.index_in(detections.exposure_id, value_set=exposures.id), -1)
    idx_np = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(idx_np < 0):
        missing = np.unique(
            np.asarray(
                detections.exposure_id.to_numpy(zero_copy_only=False), dtype=object
            )[idx_np < 0].astype(str)
        ).tolist()
        raise ValueError(f"detections reference unknown exposure_id(s): {missing}")
    exposures_aligned = exposures.take(pa.array(idx_np, type=pa.int32()))
    canonical = map_to_canonical_filter_bands(
        exposures_aligned.observatory_code,
        exposures_aligned.filter,
        allow_fallback_filters=not strict_band_mapping,
    )
    exposures_canon = exposures_aligned.set_column(
        "filter", pa.array(canonical.tolist(), type=pa.large_string())
    )

    m0 = np.asarray(
        predict_magnitudes(
            H=0.0,
            object_coords=object_coords,
            exposures=exposures_canon,
            G=G,
            reference_filter="V",
            composition=composition,
        ),
        dtype=np.float64,
    )
    mag = _as_float64_nan(detections.mag)
    mag_sigma = _as_float64_nan(detections.mag_sigma)
    valid = np.isfinite(mag) & np.isfinite(m0)
    valid = valid & np.asarray([x is not None for x in ids_np], dtype=bool)
    if not np.any(valid):
        return GroupedPhysicalParameters.from_kwargs(
            object_id=[],
            physical_parameters=PhysicalParameters.empty(),
            n_fit_detections=[],
        )

    ids_v = ids_np[valid].astype(str)
    h_v = (mag - m0)[valid]
    sig_v = mag_sigma[valid]

    order = np.argsort(ids_v, kind="mergesort")
    ids_v = ids_v[order]
    h_v = h_v[order]
    sig_v = sig_v[order]

    # Compute group offsets in one O(N) pass over the sorted ids_v.
    # ids_v is sorted; group breaks where consecutive ids differ.
    n = int(ids_v.size)
    if n == 0:
        return GroupedPhysicalParameters.from_kwargs(
            object_id=[],
            physical_parameters=PhysicalParameters.empty(),
            n_fit_detections=[],
        )
    breaks = np.concatenate([
        [0],
        np.flatnonzero(ids_v[1:] != ids_v[:-1]) + 1,
        [n],
    ]).astype(np.int64)

    from .._rust import fit_absolute_magnitude_grouped_numpy as _rust_fit_grouped

    fit_out = _rust_fit_grouped(
        np.ascontiguousarray(h_v, dtype=np.float64),
        np.ascontiguousarray(sig_v, dtype=np.float64),
        breaks,
    )
    H_hat_arr, H_sig_arr, sig_eff_arr, chi2_arr, n_used_arr = fit_out

    # Drop groups where the fit produced NaN H_hat (matches legacy's
    # try/except continue: degenerate inputs are silently skipped).
    keep = np.isfinite(H_hat_arr)
    out_id = [str(ids_v[breaks[i]]) for i in range(len(breaks) - 1) if keep[i]]
    out_H = H_hat_arr[keep].tolist()
    out_H_sigma = [
        None if not np.isfinite(v) else float(v) for v in H_sig_arr[keep]
    ]
    out_sigma_eff = [
        None if not np.isfinite(v) else float(v) for v in sig_eff_arr[keep]
    ]
    out_chi2_red = [
        None if not np.isfinite(v) else float(v) for v in chi2_arr[keep]
    ]
    out_n = [int(v) for v in n_used_arr[keep]]

    physical_parameters = PhysicalParameters.from_kwargs(
        H_v=out_H,
        H_v_sigma=out_H_sigma,
        G=[float(G)] * len(out_id),
        G_sigma=[None] * len(out_id),
        sigma_eff=out_sigma_eff,
        chi2_red=out_chi2_red,
    )
    return GroupedPhysicalParameters.from_kwargs(
        object_id=out_id,
        physical_parameters=physical_parameters,
        n_fit_detections=out_n,
    )
