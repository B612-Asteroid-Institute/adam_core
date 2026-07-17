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
from .bandpasses.api import _composition_args, _data_dir_str
from .magnitude_common import BandpassComposition

_DEFAULT_EXPOSURES_OBSERVERS = Exposures.observers


class GroupedPhysicalParameters(qv.Table):
    object_id = qv.LargeStringColumn()
    physical_parameters = PhysicalParameters.as_column()
    n_fit_detections = qv.Int64Column()


def _as_float64_nan(a: pa.Array | pa.ChunkedArray) -> npt.NDArray[np.float64]:
    # Arrow nulls map to NaN in the output.
    return np.asarray(a.to_numpy(zero_copy_only=False), dtype=np.float64)


def _run_absolute_magnitude_fit(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
    *,
    composition: BandpassComposition,
    G: float,
    strict_band_mapping: bool,
    object_ids: list[str | None] | None,
):
    from adam_core import _rust_native

    template_id, mix = _composition_args(composition)
    common = (
        _data_dir_str(),
        np.ascontiguousarray(_as_float64_nan(detections.mag)),
        np.ascontiguousarray(_as_float64_nan(detections.mag_sigma)),
        np.ascontiguousarray(np.asarray(object_coords.r, dtype=np.float64)),
    )
    observer_method = exposures.observers
    if getattr(observer_method, "__func__", None) is _DEFAULT_EXPOSURES_OBSERVERS:
        from .._rust.arrow import ensure_spice_backend

        ensure_spice_backend()
        return _rust_native.fit_absolute_magnitude_complete_numpy(
            *common,
            detections.exposure_id.to_pylist(),
            [str(value) for value in exposures.id.to_pylist()],
            [str(value) for value in exposures.observatory_code.to_pylist()],
            [str(value) for value in exposures.filter.to_pylist()],
            exposures.start_time.days.to_pylist(),
            exposures.start_time.nanos.to_pylist(),
            exposures.duration.to_pylist(),
            exposures.start_time.scale,
            float(G),
            bool(strict_band_mapping),
            template_id,
            mix,
            object_ids,
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
    observers = exposures_aligned.observers()
    return _rust_native.fit_absolute_magnitude_facade_numpy(
        *common,
        np.ascontiguousarray(np.asarray(observers.coordinates.r, dtype=np.float64)),
        [str(value) for value in exposures_aligned.observatory_code.to_pylist()],
        [str(value) for value in exposures_aligned.filter.to_pylist()],
        float(G),
        bool(strict_band_mapping),
        template_id,
        mix,
        object_ids,
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

    fit_result = _run_absolute_magnitude_fit(
        detections,
        exposures,
        object_coords,
        composition=composition,
        G=G,
        strict_band_mapping=strict_band_mapping,
        object_ids=None,
    )
    if isinstance(fit_result, pa.RecordBatch):
        from .._rust.arrow import table_from_record_batch

        return table_from_record_batch(PhysicalParameters, fit_result)
    _, h_values, h_sigma_values, sigma_eff_values, chi2_values, _ = fit_result
    H_hat = h_values[0]
    H_sigma = h_sigma_values[0]
    sigma_eff = sigma_eff_values[0]
    chi2_red = chi2_values[0]

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

    fit_result = _run_absolute_magnitude_fit(
        detections,
        exposures,
        object_coords,
        composition=composition,
        G=G,
        strict_band_mapping=strict_band_mapping,
        object_ids=ids_arr.to_pylist(),
    )
    if isinstance(fit_result, pa.RecordBatch):
        from .._rust.arrow import table_from_record_batch

        return table_from_record_batch(GroupedPhysicalParameters, fit_result)
    out_id, out_H, out_H_sigma, out_sigma_eff, out_chi2_red, out_n = fit_result

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
