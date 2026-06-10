from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import quivr as qv

from ..time import Timestamp

if TYPE_CHECKING:
    from ..coordinates.cartesian import CartesianCoordinates
    from ..observations.detections import PointSourceDetections
    from ..observations.exposures import Exposures


class RotationPeriodObservations(qv.Table):
    time = Timestamp.as_column()
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column(nullable=True)
    predicted_mag_v = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn(nullable=True)
    session_id = qv.LargeStringColumn(nullable=True)
    r_au = qv.Float64Column()
    delta_au = qv.Float64Column()
    phase_angle_deg = qv.Float64Column()

    @classmethod
    def from_point_source_observations(
        cls,
        detections: PointSourceDetections,
        exposures: Exposures,
        object_coords: CartesianCoordinates,
    ) -> RotationPeriodObservations:
        """Build observations from adam_core point-source detections + exposures.

        Links this table to the core adam_core observation primitives: one row per
        ``PointSourceDetections`` entry, with ``filter`` and the per-exposure
        observing geometry (heliocentric distance ``r_au``, observer distance
        ``delta_au``, and solar ``phase_angle_deg``) derived from the aligned
        ``Exposures`` and the object's heliocentric ``CartesianCoordinates``.

        ``object_coords`` must be heliocentric (origin=SUN) and the same length and
        order as ``detections``; ``detections.exposure_id`` is used to align each
        detection to its exposure. ``mag`` / ``r_au`` / ``delta_au`` /
        ``phase_angle_deg`` must be finite (and the distances positive) or a
        ``ValueError`` is raised.
        """
        # Lazy import avoids a module-load cycle (rotation_period_wrappers imports
        # this module); the geometry pipeline lives in that adapter module.
        from .rotation_period_wrappers import (
            build_rotation_period_observations_from_detections,
        )

        return build_rotation_period_observations_from_detections(
            detections, exposures, object_coords
        )


class RotationPeriodResult(qv.Table):
    period_days = qv.Float64Column()
    period_hours = qv.Float64Column()
    frequency_cycles_per_day = qv.Float64Column()
    primary_method = qv.LargeStringColumn()
    profile = qv.LargeStringColumn()
    period_verdict = qv.LargeStringColumn()
    reliability_code = qv.LargeStringColumn()
    confidence_flags = qv.LargeListColumn(pa.large_string(), nullable=True)
    insufficiency_reasons = qv.LargeListColumn(pa.large_string(), nullable=True)
    is_valid = qv.BooleanColumn()
    is_reliable = qv.BooleanColumn()
    period_lower_days = qv.Float64Column(nullable=True)
    period_upper_days = qv.Float64Column(nullable=True)
    relative_period_uncertainty = qv.Float64Column(nullable=True)
    alternate_period_days = qv.LargeListColumn(pa.float64(), nullable=True)
    fourier_period_days = qv.Float64Column(nullable=True)
    fourier_order = qv.Int64Column(nullable=True)
    fourier_sigma_threshold = qv.Float64Column(nullable=True)
    fourier_phase_c1 = qv.Float64Column(nullable=True)
    fourier_phase_c2 = qv.Float64Column(nullable=True)
    residual_sigma_mag = qv.Float64Column(nullable=True)
    fourier_is_valid = qv.BooleanColumn(nullable=True)
    fourier_is_reliable = qv.BooleanColumn(nullable=True)
    fourier_alternate_period_days = qv.LargeListColumn(pa.float64(), nullable=True)
    lsm_period_days = qv.Float64Column(nullable=True)
    lsm_power = qv.Float64Column(nullable=True)
    lsm_power_gap = qv.Float64Column(nullable=True)
    lsm_candidate_period_days = qv.LargeListColumn(pa.float64(), nullable=True)
    lsm_candidate_powers = qv.LargeListColumn(pa.float64(), nullable=True)
    lsm_is_reliable = qv.BooleanColumn(nullable=True)
    lsm_false_alarm_probability = qv.Float64Column(nullable=True)
    phase_coverage_fraction = qv.Float64Column(nullable=True)
    n_rotations_spanned = qv.Float64Column(nullable=True)
    amplitude_snr = qv.Float64Column(nullable=True)
    n_significant_aliases = qv.Int64Column(nullable=True)
    n_observations = qv.Int64Column()
    n_fit_observations = qv.Int64Column()
    n_clipped = qv.Int64Column()
    n_filters = qv.Int64Column()
    n_sessions = qv.Int64Column()
    used_session_offsets = qv.BooleanColumn()
    is_period_doubled = qv.BooleanColumn()


class GroupedRotationPeriodResults(qv.Table):
    object_id = qv.LargeStringColumn()
    result = RotationPeriodResult.as_column()
