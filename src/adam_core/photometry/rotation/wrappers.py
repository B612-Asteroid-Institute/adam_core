from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import OriginCodes
from ...coordinates.transform import transform_coordinates
from ...observations.detections import PointSourceDetections
from ...observations.exposures import Exposures
from ...observers.observers import Observers
from ...observers.utils import calculate_observing_night
from ...time import Timestamp
from ..magnitude import calculate_phase_angle
from .core import (
    GroupedRotationPeriodResults,
    RotationPeriodObservations,
    RotationPeriodResult,
)

_DEFAULT_EXPOSURES_OBSERVERS = Exposures.observers


__all__ = [
    "build_rotation_period_observations_from_detections",
    "estimate_rotation_period_best_apparition",
    "estimate_rotation_period_from_detections",
    "estimate_rotation_period_from_detections_grouped",
]


def _as_float64_nan(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)


def _native_detection_inputs(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
) -> tuple[Any, ...]:
    return (
        detections.exposure_id.to_pylist(),
        np.ascontiguousarray(_as_float64_nan(detections.mag)),
        np.ascontiguousarray(_as_float64_nan(detections.mag_sigma)),
        [str(value) for value in exposures.id.to_pylist()],
        [str(value) for value in exposures.observatory_code.to_pylist()],
        [
            None if value is None else str(value)
            for value in exposures.filter.to_pylist()
        ],
        exposures.start_time.days.to_pylist(),
        exposures.start_time.nanos.to_pylist(),
        _as_float64_nan(exposures.duration).tolist(),
        exposures.start_time.scale,
        np.ascontiguousarray(
            np.column_stack(
                [
                    _as_float64_nan(object_coords.x),
                    _as_float64_nan(object_coords.y),
                    _as_float64_nan(object_coords.z),
                ]
            ),
            dtype=np.float64,
        ),
        object_coords.time.days.to_pylist(),
        object_coords.time.nanos.to_pylist(),
        object_coords.time.scale,
        object_coords.frame,
    )


def _native_rotation_path_available(object_coords: CartesianCoordinates) -> bool:
    return (
        Exposures.observers is _DEFAULT_EXPOSURES_OBSERVERS
        and object_coords.frame in {"ecliptic", "equatorial"}
    )


def _validate_detection_coordinates(
    detections: PointSourceDetections,
    object_coords: CartesianCoordinates,
) -> None:
    if len(object_coords) != len(detections):
        raise ValueError(
            f"object_coords length ({len(object_coords)}) must match detections length ({len(detections)})"
        )
    if not np.all(object_coords.origin == OriginCodes.SUN):
        raise ValueError(
            "object_coords must be heliocentric (origin=SUN). "
            "Use transform_coordinates(..., origin_out=OriginCodes.SUN) first."
        )


def _align_exposures_to_detections(
    detections: PointSourceDetections, exposures: Exposures
) -> Exposures:
    if len(detections) == 0:
        raise ValueError("detections must be non-empty")
    # pyarrow.compute ships incomplete type stubs; any/is_null/index_in/fill_null
    # are valid at runtime (exercised by the test suite) but absent from the stubs.
    if pc.any(pc.is_null(detections.exposure_id)).as_py():  # type: ignore[attr-defined]
        raise ValueError("detections.exposure_id must be non-null to align exposures")

    idx = pc.fill_null(  # type: ignore[no-untyped-call]
        pc.index_in(detections.exposure_id, value_set=exposures.id),  # type: ignore[attr-defined]
        -1,
    )
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
    observers: Observers,
    exposures_full: Exposures,
    exposures_aligned: Exposures,
) -> Observers:
    if len(observers) == len(exposures_aligned):
        return observers
    if len(observers) != len(exposures_full):
        raise RuntimeError(
            "internal error: aligned observers length does not match detections length"
        )
    idx = pc.fill_null(  # type: ignore[no-untyped-call]
        pc.index_in(  # type: ignore[attr-defined]
            exposures_aligned.id, value_set=exposures_full.id
        ),
        -1,
    )
    idx_np = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(idx_np < 0):
        raise RuntimeError(
            "internal error: exposure alignment failed for observer mapping"
        )
    return observers.take(pa.array(idx_np, type=pa.int32()))


def build_rotation_period_observations_from_detections(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
) -> RotationPeriodObservations:
    n_det = len(detections)
    _validate_detection_coordinates(detections, object_coords)

    if _native_rotation_path_available(object_coords):
        from adam_core import _rust_native as _rn
        from adam_core._rust.arrow import ensure_spice_backend

        ensure_spice_backend()
        native = _rn.rotation_period_observations_from_detections(
            *_native_detection_inputs(detections, exposures, object_coords)
        )
        return RotationPeriodObservations.from_kwargs(
            time=Timestamp.from_kwargs(
                days=native["time_days"],
                nanos=native["time_nanos"],
                scale="tdb",
            ),
            mag=native["magnitude"],
            mag_sigma=native["magnitude_sigma"],
            filter=native["filter"],
            session_id=native["session_id"],
            r_au=native["r_au"],
            delta_au=native["delta_au"],
            phase_angle_deg=native["phase_angle_deg"],
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
        raise RuntimeError(
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
    **search_kwargs: Any,
) -> RotationPeriodResult:
    from . import estimator as estimator_module

    if (
        estimator_module.estimate_rotation_period
        is estimator_module._ESTIMATE_ROTATION_PERIOD_CANONICAL
        and _native_rotation_path_available(object_coords)
    ):
        from adam_core._rust.arrow import ensure_spice_backend

        _validate_detection_coordinates(detections, object_coords)
        ensure_spice_backend()
        result = (
            estimator_module._estimate_rotation_period_from_detection_inputs_native(
                _native_detection_inputs(detections, exposures, object_coords),
                None,
                **search_kwargs,
            )
        )
        assert isinstance(result, RotationPeriodResult)
        return result

    observations = build_rotation_period_observations_from_detections(
        detections, exposures, object_coords
    )
    return estimator_module.estimate_rotation_period(observations, **search_kwargs)


def estimate_rotation_period_from_detections_grouped(
    detections: PointSourceDetections,
    exposures: Exposures,
    object_coords: CartesianCoordinates,
    object_ids: pa.Array | pa.ChunkedArray | Sequence[str | None],
    **search_kwargs: Any,
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
        else pc.cast(object_ids, pa.large_string())  # type: ignore[no-untyped-call]
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

    from . import estimator as estimator_module

    if (
        estimator_module.estimate_rotation_period
        is estimator_module._ESTIMATE_ROTATION_PERIOD_CANONICAL
    ):
        object_ids_native = [None if value is None else str(value) for value in ids_np]
        if _native_rotation_path_available(object_coords):
            from adam_core._rust.arrow import ensure_spice_backend

            _validate_detection_coordinates(detections, object_coords)
            ensure_spice_backend()
            grouped = (
                estimator_module._estimate_rotation_period_from_detection_inputs_native(
                    _native_detection_inputs(detections, exposures, object_coords),
                    object_ids_native,
                    **search_kwargs,
                )
            )
            assert isinstance(grouped, GroupedRotationPeriodResults)
            return grouped
        observations = build_rotation_period_observations_from_detections(
            detections, exposures, object_coords
        )
        return estimator_module._estimate_rotation_period_grouped_native(
            observations,
            object_ids_native,
            **search_kwargs,
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

            from .estimator import estimate_rotation_period as _estimate_rotation_period

            result_i = _estimate_rotation_period(
                observations_i,
                **search_kwargs,
            )
            if len(result_i) != 1:
                # Internal kernel-contract violation, NOT an expected data failure -- a
                # RuntimeError so it bypasses the `except ValueError` insufficient path
                # below and surfaces (with object-id context) via the Exception handler.
                raise RuntimeError("rotation-period kernel must return exactly one row")
        except ValueError as exc:
            # Expected data failure: the observation builder or the solver's input
            # validation raised a ValueError for insufficient/degenerate input. Emit a
            # one-row insufficient_data result for this object so it is NEVER silently
            # dropped -- the grouped output keeps one row per id. (Internal invariant
            # failures raise RuntimeError above and are NOT caught here.)
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


def _apparition_index_groups(
    mjd: np.ndarray,
    gap_days: float,
) -> list[np.ndarray]:
    """Group observation indices into apparitions.

    An apparition is a maximal run of (time-sorted) observations in which no
    consecutive pair is separated by more than ``gap_days``. Returns one int64
    index array per apparition, in chronological order; indices refer to the
    ORIGINAL (unsorted) observation order.
    """
    order = np.argsort(np.asarray(mjd, dtype=np.float64), kind="stable")
    sorted_t = np.asarray(mjd, dtype=np.float64)[order]
    breaks = np.nonzero(np.diff(sorted_t) > float(gap_days))[0]
    return [np.asarray(group, dtype=np.int64) for group in np.split(order, breaks + 1)]


_VERDICT_RANK = {"single_period": 2, "period_family": 1, "insufficient_data": 0}


def estimate_rotation_period_best_apparition(
    observations: RotationPeriodObservations,
    *,
    apparition_gap_days: float = 120.0,
    **solver_kwargs: Any,
) -> RotationPeriodResult:
    """Solve each apparition separately and keep the highest-confidence result.

    Ground-based lightcurves of the same asteroid from different apparitions
    differ in viewing aspect (and therefore amplitude), photometric noise, and
    nightly cadence, so the diurnal-alias structure of each apparition differs
    too. An apparition that happens to sample the rotation cleanly can yield a
    confident, correct period where the densest apparition -- or all
    apparitions pooled -- locks onto a sampling alias or hedges. This helper
    partitions the observations into apparitions (separated by more than
    ``apparition_gap_days``), runs :func:`estimate_rotation_period` on each
    independently, and returns the result of the most confident apparition.

    The selection rule uses no knowledge of any reference answer: rank the
    verdicts ``single_period`` > ``period_family`` > ``insufficient_data``,
    tie-break on higher ``amplitude_snr``, then on more observations, then on
    the earlier apparition. Measured on the 118-object LCDB standard-candle
    calibration set, this policy raised confident (``single_period``) claims
    from 35 to 43 while the strict precision of those claims improved (0.800 ->
    0.837) and the wrong-family count was unchanged -- selection shopping did
    not introduce false confidence on that set, but the guarantee is
    empirical, not structural.

    The chosen row is returned with a ``apparition_selected_<k>_of_<n>``
    confidence flag appended (1-based, chronological). An apparition whose
    solve fails with an expected ``ValueError`` participates as an
    ``insufficient_data`` candidate flagged ``solve_error``; an unexpected
    error is re-raised with the apparition attached. Apparitions solve
    serially; for large batches, parallelize per apparition yourself.
    """
    # Lazy import keeps this wrapper module importable without the solver kernel.
    from . import estimator as estimator_module

    _estimate_rotation_period = estimator_module.estimate_rotation_period
    if (
        _estimate_rotation_period
        is estimator_module._ESTIMATE_ROTATION_PERIOD_CANONICAL
    ):
        return estimator_module._estimate_rotation_period_best_apparition_native(
            observations,
            float(apparition_gap_days),
            **solver_kwargs,
        )

    n_obs = len(observations)
    mjd = np.asarray(
        observations.time.rescale("tdb").mjd().to_numpy(False), dtype=np.float64
    )
    groups = _apparition_index_groups(mjd, apparition_gap_days)
    if n_obs == 0 or len(groups) <= 1:
        # Single apparition (or empty, where the solver raises its canonical
        # error): delegate wholesale so behavior matches the direct call.
        result = _estimate_rotation_period(observations, **solver_kwargs)
        return _with_apparition_flag(result, selected=1, total=1)

    candidates: list[tuple[int, RotationPeriodResult]] = []
    for k, indices in enumerate(groups):
        subset = observations.take(pa.array(indices, type=pa.int64()))
        try:
            result_k = _estimate_rotation_period(subset, **solver_kwargs)
        except ValueError as exc:
            # Expected per-apparition data failure: keep it as a candidate so
            # a fully-unsolvable object still returns one insufficient row.
            result_k = RotationPeriodResult.single_insufficient(
                reasons=[f"solve_error: {exc}"],
                confidence_flags=["solve_error"],
                n_observations=int(len(indices)),
            )
        except Exception as exc:  # noqa: BLE001 - attach apparition context
            raise RuntimeError(
                f"rotation-period solve failed for apparition {k + 1} of "
                f"{len(groups)}"
            ) from exc
        candidates.append((k, result_k))

    def _score(
        item: tuple[int, RotationPeriodResult],
    ) -> tuple[float, float, float, float]:
        k, result_k = item
        verdict = str(result_k.period_verdict[0].as_py())
        snr = result_k.amplitude_snr[0].as_py()
        return (
            float(_VERDICT_RANK.get(verdict, 0)),
            float("-inf") if snr is None else float(snr),
            float(result_k.n_observations[0].as_py() or 0),
            -float(k),
        )

    best_k, best = max(candidates, key=_score)
    return _with_apparition_flag(best, selected=best_k + 1, total=len(groups))


def _with_apparition_flag(
    result: RotationPeriodResult, *, selected: int, total: int
) -> RotationPeriodResult:
    """Append an ``apparition_selected_<k>_of_<n>`` confidence flag to a row."""
    flags = list(result.confidence_flags[0].as_py() or [])
    flags.append(f"apparition_selected_{selected}_of_{total}")
    column_index = result.table.schema.get_field_index("confidence_flags")
    table = result.table.set_column(
        column_index,
        result.table.schema.field("confidence_flags"),
        pa.array([flags], type=pa.large_list(pa.large_string())),
    )
    return RotationPeriodResult.from_pyarrow(table)
