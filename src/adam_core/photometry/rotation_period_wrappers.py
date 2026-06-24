from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

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


def _align_observers_to_exposures(
    observers,
    exposures_full: Exposures,
    exposures_aligned: Exposures,
):
    if len(observers) == len(exposures_aligned):
        return observers
    if len(observers) != len(exposures_full):
        raise ValueError(
            "internal error: aligned observers length does not match detections length"
        )
    idx = pc.fill_null(
        pc.index_in(exposures_aligned.id, value_set=exposures_full.id), -1
    )
    idx_np = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(idx_np < 0):
        raise ValueError(
            "internal error: exposure alignment failed for observer mapping"
        )
    return observers.take(pa.array(idx_np, type=pa.int32()))


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
    observers_helio = exposures_aligned.observers(
        frame="ecliptic", origin=OriginCodes.SUN
    )
    observers_helio = _align_observers_to_exposures(
        observers_helio,
        exposures,
        exposures_aligned,
    )
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
        raise ValueError(
            "invalid heliocentric distance(s) for rotation-period analysis"
        )
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
    observing_night_np = np.asarray(
        observing_night.to_numpy(zero_copy_only=False), dtype=np.int64
    )
    session_id = pa.array(
        [
            f"{str(code)}:{int(night)}"
            for code, night in zip(
                observatory_code.tolist(), observing_night_np.tolist()
            )
        ],
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
        predicted_mag_v=pa.nulls(n_det, type=pa.float64()),
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
        **search_kwargs,
    )


def estimate_rotation_period_from_detections_grouped(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
    object_ids: pa.Array | pa.ChunkedArray | Sequence[str | None],
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
    out_results: list[RotationPeriodResult] = []

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
                **search_kwargs,
            )
            if len(result_i) != 1:
                raise ValueError("rotation-period kernel must return exactly one row")
        except ValueError as exc:
            # Expected data failure (insufficient/degenerate input or a kernel
            # ValueError): emit a one-row insufficient_data result for this object so it
            # is NEVER silently dropped -- the grouped output keeps one row per id.
            result_i = RotationPeriodResult.single_insufficient(
                reasons=[f"solve_error: {exc}"],
                confidence_flags=["solve_error"],
                n_observations=int(i1 - i0),
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - attach object-id context, then re-raise
            raise RuntimeError(
                f"rotation-period solve failed for object_id {oid!r}"
            ) from exc

        out_object_id.append(oid)
        out_results.append(result_i)

        i0 = i1

    result = (
        RotationPeriodResult.empty() if not out_results else qv.concatenate(out_results)
    )
    return GroupedRotationPeriodResults.from_kwargs(
        object_id=out_object_id, result=result
    )
