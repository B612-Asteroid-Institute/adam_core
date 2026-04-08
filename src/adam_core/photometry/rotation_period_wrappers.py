from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import OriginCodes
from ..coordinates.transform import transform_coordinates
from ..observations.detections import PointSourceDetections
from ..observations.exposures import Exposures
from ..observers.utils import calculate_observing_night
from ..photometry.magnitude import calculate_phase_angle
from ..photometry.rotation_period_types import (
    GroupedRotationPeriodResults,
    RotationPeriodObservations,
    RotationPeriodResult,
)

__all__ = [
    "build_rotation_period_observations_from_detections",
    "estimate_rotation_period_from_detections",
    "estimate_rotation_period_from_detections_grouped",
]


def _as_float64_nan(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)


def _align_exposures_to_detections(
    detections: PointSourceDetections, exposures: Exposures
) -> Exposures:
    if len(detections) == 0:
        raise ValueError("detections must be non-empty")
    if pc.any(pc.is_null(detections.exposure_id)).as_py():
        raise ValueError("detections.exposure_id must be non-null to align exposures")

    idx = pc.fill_null(pc.index_in(detections.exposure_id, value_set=exposures.id), -1)
    idx_np = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(idx_np < 0):
        missing = np.unique(
            np.asarray(
                detections.exposure_id.to_numpy(zero_copy_only=False), dtype=object
            )[idx_np < 0].astype(str)
        ).tolist()
        raise ValueError(f"detections reference unknown exposure_id(s): {missing}")

    return exposures.take(pa.array(idx_np, type=pa.int32()))


def _extract_result_row(result: RotationPeriodResult) -> dict[str, object]:
    if len(result) != 1:
        raise ValueError("rotation-period kernel must return exactly one row")

    return {
        "period_days": result.period_days[0].as_py(),
        "period_hours": result.period_hours[0].as_py(),
        "frequency_cycles_per_day": result.frequency_cycles_per_day[0].as_py(),
        "fourier_order": result.fourier_order[0].as_py(),
        "phase_c1": result.phase_c1[0].as_py(),
        "phase_c2": result.phase_c2[0].as_py(),
        "residual_sigma_mag": result.residual_sigma_mag[0].as_py(),
        "n_observations": result.n_observations[0].as_py(),
        "n_fit_observations": result.n_fit_observations[0].as_py(),
        "n_clipped": result.n_clipped[0].as_py(),
        "n_filters": result.n_filters[0].as_py(),
        "n_sessions": result.n_sessions[0].as_py(),
        "used_session_offsets": result.used_session_offsets[0].as_py(),
        "is_period_doubled": result.is_period_doubled[0].as_py(),
        "is_ambiguous": result.is_ambiguous[0].as_py(),
        "confidence_label": result.confidence_label[0].as_py(),
        "ambiguity_reason": result.ambiguity_reason[0].as_py(),
        "n_harmonic_near_ties": result.n_harmonic_near_ties[0].as_py(),
        "used_grid_fallback": result.used_grid_fallback[0].as_py(),
        "harmonic_sigma_tolerance_mag": result.harmonic_sigma_tolerance_mag[0].as_py(),
        "lsm_period_days": result.lsm_period_days[0].as_py(),
        "lsm_period_hours": result.lsm_period_hours[0].as_py(),
        "lsm_frequency_cycles_per_day": result.lsm_frequency_cycles_per_day[0].as_py(),
        "lsm_harmonic_agreement": result.lsm_harmonic_agreement[0].as_py(),
    }


def build_rotation_period_observations_from_detections(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
) -> RotationPeriodObservations:
    n_det = len(detections)
    if len(object_coords) != n_det:
        raise ValueError(
            f"object_coords length ({len(object_coords)}) must match detections length ({n_det})"
        )
    if not np.all(object_coords.origin == OriginCodes.SUN):
        raise ValueError(
            "object_coords must be heliocentric (origin=SUN). "
            "Use transform_coordinates(..., origin_out=OriginCodes.SUN) first."
        )

    exposures_aligned = _align_exposures_to_detections(detections, exposures)
    observers_helio = exposures_aligned.observers(frame="ecliptic", origin=OriginCodes.SUN)
    object_coords_helio = transform_coordinates(
        object_coords,
        CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    if len(observers_helio) != n_det:
        raise ValueError(
            "internal error: aligned observers length does not match detections length"
        )

    time = object_coords_helio.time.rescale("tdb")
    mag = _as_float64_nan(detections.mag)
    mag_sigma = _as_float64_nan(detections.mag_sigma)
    r_au = np.linalg.norm(np.asarray(object_coords_helio.r, dtype=np.float64), axis=1)
    delta_vec = np.asarray(object_coords_helio.r, dtype=np.float64) - np.asarray(
        observers_helio.coordinates.r, dtype=np.float64
    )
    delta_au = np.linalg.norm(delta_vec, axis=1)
    phase_angle_deg = np.asarray(
        calculate_phase_angle(object_coords_helio, observers_helio), dtype=np.float64
    )

    if np.any(~np.isfinite(mag)):
        raise ValueError("detections.mag must be finite for rotation-period analysis")
    if np.any(~np.isfinite(r_au)) or np.any(r_au <= 0.0):
        raise ValueError("invalid heliocentric distance(s) for rotation-period analysis")
    if np.any(~np.isfinite(delta_au)) or np.any(delta_au <= 0.0):
        raise ValueError("invalid observer distance(s) for rotation-period analysis")
    if np.any(~np.isfinite(phase_angle_deg)):
        raise ValueError("invalid phase angle(s) for rotation-period analysis")

    observing_night = calculate_observing_night(
        exposures_aligned.observatory_code,
        exposures_aligned.start_time.rescale("utc"),
    )
    observatory_code = np.asarray(
        exposures_aligned.observatory_code.to_numpy(zero_copy_only=False),
        dtype=object,
    )
    observing_night_np = np.asarray(observing_night.to_numpy(zero_copy_only=False), dtype=np.int64)
    session_id = pa.array(
        [f"{str(code)}:{int(night)}" for code, night in zip(observatory_code.tolist(), observing_night_np.tolist())],
        type=pa.large_string(),
    )

    return RotationPeriodObservations.from_kwargs(
        time=time,
        mag=pa.array(mag, type=pa.float64()),
        mag_sigma=pa.array(
            mag_sigma,
            mask=~np.isfinite(mag_sigma),
            type=pa.float64(),
        ),
        filter=exposures_aligned.filter,
        session_id=session_id,
        r_au=pa.array(r_au, type=pa.float64()),
        delta_au=pa.array(delta_au, type=pa.float64()),
        phase_angle_deg=pa.array(phase_angle_deg, type=pa.float64()),
    )


def estimate_rotation_period_from_detections(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
    *,
    max_processes: int | None = None,
    parallel_chunk_size: int | None = None,
    **search_kwargs,
) -> RotationPeriodResult:
    observations = build_rotation_period_observations_from_detections(
        detections, exposures, object_coords
    )

    # Lazy import keeps this wrapper file usable before the kernel bead lands.
    from .rotation_period_fourier import (
        estimate_rotation_period as _estimate_rotation_period,
    )

    return _estimate_rotation_period(
        observations,
        max_processes=max_processes,
        parallel_chunk_size=parallel_chunk_size,
        **search_kwargs,
    )


def estimate_rotation_period_from_detections_grouped(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
    object_ids: pa.Array | pa.ChunkedArray | Sequence[str | None],
    *,
    max_processes: int | None = None,
    parallel_chunk_size: int | None = None,
    **search_kwargs,
) -> GroupedRotationPeriodResults:
    n_det = len(detections)
    if n_det == 0:
        return GroupedRotationPeriodResults.from_kwargs(
            object_id=[],
            result=RotationPeriodResult.empty(),
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

    valid_mask = np.asarray([x is not None for x in ids_np], dtype=bool)
    if not np.any(valid_mask):
        return GroupedRotationPeriodResults.from_kwargs(
            object_id=[],
            result=RotationPeriodResult.empty(),
        )

    valid_ids = ids_np[valid_mask].astype(str)
    valid_indices = np.flatnonzero(valid_mask)
    order = np.argsort(valid_ids, kind="mergesort")
    ids_sorted = valid_ids[order]
    indices_sorted = valid_indices[order]

    out_object_id: list[str] = []
    out_period_days: list[float] = []
    out_period_hours: list[float] = []
    out_frequency_cycles_per_day: list[float] = []
    out_fourier_order: list[int] = []
    out_phase_c1: list[float] = []
    out_phase_c2: list[float] = []
    out_residual_sigma_mag: list[float] = []
    out_n_observations: list[int] = []
    out_n_fit_observations: list[int] = []
    out_n_clipped: list[int] = []
    out_n_filters: list[int] = []
    out_n_sessions: list[int] = []
    out_used_session_offsets: list[bool] = []
    out_is_period_doubled: list[bool] = []
    out_is_ambiguous: list[bool] = []
    out_confidence_label: list[str] = []
    out_ambiguity_reason: list[str | None] = []
    out_n_harmonic_near_ties: list[int] = []
    out_used_grid_fallback: list[bool] = []
    out_harmonic_sigma_tolerance_mag: list[float] = []
    out_lsm_period_days: list[float | None] = []
    out_lsm_period_hours: list[float | None] = []
    out_lsm_frequency_cycles_per_day: list[float | None] = []
    out_lsm_harmonic_agreement: list[bool | None] = []

    i0 = 0
    n = int(ids_sorted.size)
    while i0 < n:
        oid = str(ids_sorted[i0])
        i1 = i0 + 1
        while i1 < n and str(ids_sorted[i1]) == oid:
            i1 += 1

        idx = pa.array(indices_sorted[i0:i1], type=pa.int32())
        try:
            detections_i = detections.take(idx)
            object_coords_i = object_coords.take(idx)
            observations_i = build_rotation_period_observations_from_detections(
                detections_i,
                exposures,
                object_coords_i,
            )

            from .rotation_period_fourier import (
                estimate_rotation_period as _estimate_rotation_period,
            )

            result_i = _estimate_rotation_period(
                observations_i,
                max_processes=max_processes,
                parallel_chunk_size=parallel_chunk_size,
                **search_kwargs,
            )
            row = _extract_result_row(result_i)
        except Exception:
            i0 = i1
            continue

        out_object_id.append(oid)
        out_period_days.append(row["period_days"])
        out_period_hours.append(row["period_hours"])
        out_frequency_cycles_per_day.append(row["frequency_cycles_per_day"])
        out_fourier_order.append(row["fourier_order"])
        out_phase_c1.append(row["phase_c1"])
        out_phase_c2.append(row["phase_c2"])
        out_residual_sigma_mag.append(row["residual_sigma_mag"])
        out_n_observations.append(int(row["n_observations"]))
        out_n_fit_observations.append(int(row["n_fit_observations"]))
        out_n_clipped.append(int(row["n_clipped"]))
        out_n_filters.append(int(row["n_filters"]))
        out_n_sessions.append(int(row["n_sessions"]))
        out_used_session_offsets.append(bool(row["used_session_offsets"]))
        out_is_period_doubled.append(bool(row["is_period_doubled"]))
        out_is_ambiguous.append(bool(row["is_ambiguous"]))
        out_confidence_label.append(str(row["confidence_label"]))
        out_ambiguity_reason.append(
            None if row["ambiguity_reason"] is None else str(row["ambiguity_reason"])
        )
        out_n_harmonic_near_ties.append(int(row["n_harmonic_near_ties"]))
        out_used_grid_fallback.append(bool(row["used_grid_fallback"]))
        out_harmonic_sigma_tolerance_mag.append(float(row["harmonic_sigma_tolerance_mag"]))
        out_lsm_period_days.append(
            None if row["lsm_period_days"] is None else float(row["lsm_period_days"])
        )
        out_lsm_period_hours.append(
            None if row["lsm_period_hours"] is None else float(row["lsm_period_hours"])
        )
        out_lsm_frequency_cycles_per_day.append(
            None
            if row["lsm_frequency_cycles_per_day"] is None
            else float(row["lsm_frequency_cycles_per_day"])
        )
        out_lsm_harmonic_agreement.append(
            None
            if row["lsm_harmonic_agreement"] is None
            else bool(row["lsm_harmonic_agreement"])
        )

        i0 = i1

    result = RotationPeriodResult.from_kwargs(
        period_days=out_period_days,
        period_hours=out_period_hours,
        frequency_cycles_per_day=out_frequency_cycles_per_day,
        fourier_order=out_fourier_order,
        phase_c1=out_phase_c1,
        phase_c2=out_phase_c2,
        residual_sigma_mag=out_residual_sigma_mag,
        n_observations=out_n_observations,
        n_fit_observations=out_n_fit_observations,
        n_clipped=out_n_clipped,
        n_filters=out_n_filters,
        n_sessions=out_n_sessions,
        used_session_offsets=out_used_session_offsets,
        is_period_doubled=out_is_period_doubled,
        is_ambiguous=out_is_ambiguous,
        confidence_label=out_confidence_label,
        ambiguity_reason=out_ambiguity_reason,
        n_harmonic_near_ties=out_n_harmonic_near_ties,
        used_grid_fallback=out_used_grid_fallback,
        harmonic_sigma_tolerance_mag=out_harmonic_sigma_tolerance_mag,
        lsm_period_days=out_lsm_period_days,
        lsm_period_hours=out_lsm_period_hours,
        lsm_frequency_cycles_per_day=out_lsm_frequency_cycles_per_day,
        lsm_harmonic_agreement=out_lsm_harmonic_agreement,
    )
    return GroupedRotationPeriodResults.from_kwargs(object_id=out_object_id, result=result)
