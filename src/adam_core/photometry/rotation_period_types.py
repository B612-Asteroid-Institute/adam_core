import quivr as qv

from ..time import Timestamp


class RotationPeriodObservations(qv.Table):
    time = Timestamp.as_column()
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn(nullable=True)
    session_id = qv.LargeStringColumn(nullable=True)
    r_au = qv.Float64Column()
    delta_au = qv.Float64Column()
    phase_angle_deg = qv.Float64Column()


class RotationPeriodResult(qv.Table):
    period_days = qv.Float64Column()
    period_hours = qv.Float64Column()
    frequency_cycles_per_day = qv.Float64Column()
    fourier_order = qv.Int64Column()
    phase_c1 = qv.Float64Column()
    phase_c2 = qv.Float64Column()
    residual_sigma_mag = qv.Float64Column()
    n_observations = qv.Int64Column()
    n_fit_observations = qv.Int64Column()
    n_clipped = qv.Int64Column()
    n_filters = qv.Int64Column()
    n_sessions = qv.Int64Column()
    used_session_offsets = qv.BooleanColumn()
    is_period_doubled = qv.BooleanColumn()
    is_ambiguous = qv.BooleanColumn()
    confidence_label = qv.LargeStringColumn()
    ambiguity_reason = qv.LargeStringColumn(nullable=True)
    n_harmonic_near_ties = qv.Int64Column()
    used_grid_fallback = qv.BooleanColumn()
    harmonic_sigma_tolerance_mag = qv.Float64Column()
    lsm_period_days = qv.Float64Column(nullable=True)
    lsm_period_hours = qv.Float64Column(nullable=True)
    lsm_frequency_cycles_per_day = qv.Float64Column(nullable=True)
    lsm_harmonic_agreement = qv.BooleanColumn(nullable=True)


class GroupedRotationPeriodResults(qv.Table):
    object_id = qv.LargeStringColumn()
    result = RotationPeriodResult.as_column()
