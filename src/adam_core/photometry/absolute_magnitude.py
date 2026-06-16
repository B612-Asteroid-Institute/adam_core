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
    Fit H statistics for one grouped set of rows.

    Returns
    -------
    (H_hat, H_sigma, sigma_eff, chi2_red, n_used)
    """
    n_used = int(h_rows.size)
    if n_used <= 0:
        raise ValueError("h_rows must be non-empty")

    have_all_sigma = bool(np.all(np.isfinite(sigma_rows)))
    if have_all_sigma:
        w = 1.0 / np.square(sigma_rows)
        wsum = float(np.sum(w))
        if not np.isfinite(wsum) or wsum <= 0.0:
            raise ValueError("invalid weights derived from mag_sigma")
        H_hat = float(np.sum(w * h_rows) / wsum)
    else:
        H_hat = float(np.mean(h_rows))

    resid = h_rows - H_hat
    sigma_eff = None
    if n_used >= 2:
        s = _mad_sigma(resid)
        if np.isfinite(s):
            sigma_eff = float(s)

    chi2_red = None
    H_sigma = None
    if have_all_sigma and n_used >= 2:
        w = 1.0 / np.square(sigma_rows)
        chi2 = float(np.sum(w * np.square(resid)))
        chi2_red = chi2 / float(n_used - 1)
        H_sigma = float(np.sqrt(1.0 / np.sum(w)))
        if np.isfinite(chi2_red) and chi2_red > 1.0:
            H_sigma = float(H_sigma * np.sqrt(chi2_red))
    elif sigma_eff is not None and n_used >= 2:
        H_sigma = float(sigma_eff / np.sqrt(float(n_used)))

    return H_hat, H_sigma, sigma_eff, chi2_red, n_used


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

    out_id: list[str] = []
    out_H: list[float] = []
    out_H_sigma: list[float | None] = []
    out_sigma_eff: list[float | None] = []
    out_chi2_red: list[float | None] = []
    out_n: list[int] = []

    i0 = 0
    n = int(ids_v.size)
    while i0 < n:
        oid = str(ids_v[i0])
        i1 = i0 + 1
        while i1 < n and str(ids_v[i1]) == oid:
            i1 += 1
        try:
            H_hat, H_sig, sigma_eff, chi2_red, n_used = _fit_absolute_magnitude_rows(
                h_rows=np.asarray(h_v[i0:i1], dtype=np.float64),
                sigma_rows=np.asarray(sig_v[i0:i1], dtype=np.float64),
            )
        except Exception:
            i0 = i1
            continue
        out_id.append(oid)
        out_H.append(float(H_hat))
        out_H_sigma.append(None if H_sig is None else float(H_sig))
        out_sigma_eff.append(None if sigma_eff is None else float(sigma_eff))
        out_chi2_red.append(None if chi2_red is None else float(chi2_red))
        out_n.append(int(n_used))
        i0 = i1

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
