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
    profile = qv.LargeStringColumn()
    period_verdict = qv.LargeStringColumn()
    # LCDB-U-style reliability code as a STRING ("3"/"2"/"1", highest first). Kept a
    # string (not int) to mirror LCDB U codes and stay forward-compatible with
    # qualified codes (e.g. "1+"); do NOT sort or compare it numerically.
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

    @classmethod
    def single_insufficient(
        cls,
        *,
        reasons: list[str],
        confidence_flags: list[str] | None = None,
        n_observations: int = 0,
        n_filters: int = 0,
        n_sessions: int = 0,
        profile: str = "default",
    ) -> "RotationPeriodResult":
        """One-row ``insufficient_data`` result: NaN period, every nullable diagnostic
        ``None``.

        The canonical builder for the insufficient verdict. Used both by the solver's
        early-exit path and by the detection wrappers when an object cannot be solved,
        so a grouped solve returns one row per object id rather than silently dropping
        failures. ``period_verdict``/``reliability_code`` are the contract constants
        (``"insufficient_data"`` / ``"1"``).
        """
        return cls.from_kwargs(
            period_days=[float("nan")],
            period_hours=[float("nan")],
            frequency_cycles_per_day=[float("nan")],
            profile=[profile],
            period_verdict=["insufficient_data"],
            reliability_code=["1"],
            confidence_flags=[list(confidence_flags or [])],
            insufficiency_reasons=[list(reasons)],
            is_valid=[False],
            is_reliable=[False],
            period_lower_days=[None],
            period_upper_days=[None],
            relative_period_uncertainty=[None],
            alternate_period_days=[[]],
            fourier_period_days=[None],
            fourier_order=[None],
            fourier_sigma_threshold=[None],
            fourier_phase_c1=[None],
            fourier_phase_c2=[None],
            residual_sigma_mag=[None],
            fourier_is_valid=[None],
            fourier_is_reliable=[None],
            fourier_alternate_period_days=[[]],
            phase_coverage_fraction=[None],
            n_rotations_spanned=[None],
            amplitude_snr=[None],
            n_significant_aliases=[None],
            n_observations=[int(n_observations)],
            n_fit_observations=[0],
            n_clipped=[0],
            n_filters=[int(n_filters)],
            n_sessions=[int(n_sessions)],
            used_session_offsets=[False],
            is_period_doubled=[False],
        )


class GroupedRotationPeriodResults(qv.Table):
    object_id = qv.LargeStringColumn()
    result = RotationPeriodResult.as_column()
