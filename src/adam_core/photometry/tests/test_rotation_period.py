from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

import adam_core.photometry.rotation_period_fourier as rotation_period_fourier
from ...time import Timestamp
from ..rotation_period_fourier import (
    MAX_PLAUSIBLE_SINGLE_PERIOD_HOURS,
    _FourierCluster,
    _FourierSolution,
    _LSMCandidate,
    _LSMMethodResult,
    _LSMSolution,
    _MethodFamily,
    _best_harmonic_factor,
    _build_fourier_result,
    _build_frequency_grid,
    _is_simple_harmonic_factor,
    _paper_profile,
    _primary_from_method,
    estimate_rotation_period,
)
from ..rotation_period_fourier_core import (
    _FitResult,
    _FitWithPeriod,
    _build_fixed_design,
    _directional_f_test_confidence,
    _fit_frequency,
    _select_order,
    _sigma_threshold_from_confidence,
)
from ..rotation_period_types import RotationPeriodObservations, RotationPeriodResult


def _scalar(value) -> object:
    if hasattr(value, "as_py"):
        return value.as_py()
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr.reshape(-1)[0].item()


def _make_rotation_observations(
    *,
    n: int = 60,
    span_days: float = 0.08,
    period_days: float = 0.02,
    amplitude: float = 0.35,
    single_peaked: bool = False,
    noise_sigma: float = 0.01,
    include_predicted: bool = False,
    filter_names: tuple[str, ...] = ("LSST_g", "LSST_r"),
) -> RotationPeriodObservations:
    mjd = np.linspace(60000.0, 60000.0 + span_days, n, dtype=np.float64)
    time = Timestamp.from_mjd(mjd, scale="tdb")
    t_rel = mjd - mjd.min()
    filters = np.asarray(
        [filter_names[idx % len(filter_names)] for idx in range(n)],
        dtype=object,
    )
    sessions = np.asarray(
        [f"X05:{int(60000 + (idx // max(1, n // 6)))}" for idx in range(n)],
        dtype=object,
    )
    r_au = np.full(n, 2.0, dtype=np.float64)
    delta_au = np.full(n, 1.5, dtype=np.float64)
    phase_angle = 12.0 + 4.0 * np.sin(2.0 * np.pi * t_rel / max(span_days, 1.0e-6))

    baseline = 15.0 + 5.0 * np.log10(r_au * delta_au) + 0.015 * phase_angle + 0.0015 * np.square(phase_angle)
    phase = 2.0 * np.pi * t_rel / period_days
    if single_peaked:
        rotation = amplitude * np.cos(phase)
    else:
        rotation = 0.10 * np.cos(phase) + amplitude * np.cos(2.0 * phase) + 0.07 * np.sin(2.0 * phase)
    rotation = rotation + np.asarray([0.03 if name == "LSST_g" else 0.0 for name in filters], dtype=np.float64)
    rng = np.random.default_rng(20260414)
    mag = baseline + rotation + rng.normal(0.0, noise_sigma, size=n)
    predicted = baseline if include_predicted else None

    return RotationPeriodObservations.from_kwargs(
        time=time,
        mag=pa.array(mag, type=pa.float64()),
        mag_sigma=pa.array(np.full(n, 0.03, dtype=np.float64), type=pa.float64()),
        predicted_mag_v=None if predicted is None else pa.array(predicted, type=pa.float64()),
        filter=pa.array(filters.tolist(), type=pa.large_string()),
        session_id=pa.array(sessions.tolist(), type=pa.large_string()),
        r_au=pa.array(r_au, type=pa.float64()),
        delta_au=pa.array(delta_au, type=pa.float64()),
        phase_angle_deg=pa.array(phase_angle, type=pa.float64()),
    )


def _fit_template(
    *,
    frequency: float,
    fourier_order: int,
    rss: float,
    df: int,
    sigma: float | None = None,
) -> _FitResult:
    coeffs = np.zeros(3 + 2 * fourier_order, dtype=np.float64)
    coeffs[-2] = 0.2
    coeffs[-1] = 0.1
    return _FitResult(
        frequency=float(frequency),
        fourier_order=int(fourier_order),
        coeffs=coeffs,
        residual_sigma=float(np.sqrt(rss / df) if sigma is None else sigma),
        rss=float(rss),
        df=int(df),
        n_par=3 + 2 * fourier_order,
        mask=np.ones(df + 3 + 2 * fourier_order, dtype=bool),
        n_fit=int(df + 3 + 2 * fourier_order),
        n_clipped=0,
        n_filters=1,
        phase_c1_idx=1,
        phase_c2_idx=2,
        sum_weights=float(df + 3 + 2 * fourier_order),
        n_local_maxima=2,
    )


def test_directional_f_test_confidence_and_sigma_threshold():
    small = _fit_template(frequency=20.0, fourier_order=2, rss=30.0, df=20)
    large = _fit_template(frequency=20.0, fourier_order=3, rss=12.0, df=18)
    confidence = _directional_f_test_confidence(small, large)
    assert 0.90 < confidence < 1.0

    sigma_threshold = _sigma_threshold_from_confidence(0.05, 18, 0.95)
    assert sigma_threshold > 0.05


def test_profile_specific_order_selection_thresholds():
    small = _fit_template(frequency=20.0, fourier_order=2, rss=30.0, df=20)
    better = _fit_template(frequency=20.0, fourier_order=3, rss=14.0, df=18)
    confidence = _directional_f_test_confidence(small, better)
    assert 0.90 < confidence < 0.95

    candidates = {2: small, 3: better}
    assert _select_order(candidates, 0.90).fourier_order == 3
    assert _select_order(candidates, 0.95).fourier_order == 2


def test_fourier_single_peak_is_doubled():
    observations = _make_rotation_observations(single_peaked=True, period_days=0.02)
    result = estimate_rotation_period(
        observations,
        method_mode="fourier",
        paper_profile="greenstreet_2026",
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert isinstance(result, RotationPeriodResult)
    assert float(_scalar(result.period_days[0])) == pytest.approx(0.04, rel=0.10)
    assert all(
        float(period) > 0.03 for period in result.fourier_alternate_period_days[0].as_py()
    )


def test_lsm_single_peak_candidates_are_rejected():
    observations = _make_rotation_observations(single_peaked=True, period_days=0.02)
    result = estimate_rotation_period(observations, method_mode="lsm")

    assert isinstance(result, RotationPeriodResult)
    assert _scalar(result.primary_method[0]) == "lsm"
    assert float(_scalar(result.period_days[0])) == pytest.approx(0.04, rel=0.10)
    assert all(
        abs(float(period) - 0.02) > 0.002 for period in result.lsm_candidate_period_days[0].as_py()
    )
    assert bool(_scalar(result.lsm_is_reliable[0])) is False


def test_fourier_accepted_solution_clustering_produces_interval_and_alternates():
    observations = _make_rotation_observations()
    time = np.asarray(observations.time.rescale("tdb").mjd().to_numpy(False), dtype=np.float64)
    mag = np.asarray(observations.mag.to_numpy(zero_copy_only=False), dtype=np.float64)
    phase_angle = np.asarray(observations.phase_angle_deg.to_numpy(zero_copy_only=False), dtype=np.float64)
    filter_labels = np.asarray(observations.filter.to_numpy(zero_copy_only=False), dtype=object)
    weights = np.full(len(observations), 1.0 / 0.03**2, dtype=np.float64)
    time_rel = time - time.min()
    y = mag - 5.0 * np.log10(
        np.asarray(observations.r_au.to_numpy(zero_copy_only=False), dtype=np.float64)
        * np.asarray(observations.delta_au.to_numpy(zero_copy_only=False), dtype=np.float64)
    )
    design_info = _build_fixed_design(filter_labels, None, phase_angle)
    fit_a = _fit_frequency(time_rel, y, design_info, 50.0, 2, clip_sigma=3.0, weights=weights)
    fit_b = _fit_frequency(time_rel, y, design_info, 33.3333333333, 2, clip_sigma=3.0, weights=weights)
    assert fit_a is not None
    assert fit_b is not None

    frequencies = np.linspace(30.0, 55.0, 8, dtype=np.float64)
    scores = np.full(frequencies.shape, 0.30, dtype=np.float64)
    fits_by_index = {
        1: replace(fit_a, residual_sigma=0.05, frequency=float(frequencies[1]), n_local_maxima=2),
        2: replace(fit_a, residual_sigma=0.051, frequency=float(frequencies[2]), n_local_maxima=2),
        5: replace(fit_b, residual_sigma=0.053, frequency=float(frequencies[5]), n_local_maxima=2),
        6: replace(fit_b, residual_sigma=0.054, frequency=float(frequencies[6]), n_local_maxima=2),
    }
    scores[1] = 0.05
    scores[2] = 0.051
    scores[5] = 0.053
    scores[6] = 0.054
    result = _build_fourier_result(
        chosen_fit=fits_by_index[1],
        order_grid_results={2: (scores, fits_by_index)},
        frequencies=frequencies,
        profile=_paper_profile("greenstreet_2026"),
    )

    assert result.period_lower_days <= result.best_period.period_days <= result.period_upper_days
    assert len(result.alternate_period_days) == 1


def test_lsm_with_predicted_magnitude_matches_offline_approximation():
    observations_offline = _make_rotation_observations(include_predicted=False)
    observations_pred = _make_rotation_observations(include_predicted=True)

    result_offline = estimate_rotation_period(observations_offline, method_mode="lsm")
    result_pred = estimate_rotation_period(observations_pred, method_mode="lsm")

    assert float(_scalar(result_pred.period_days[0])) == pytest.approx(
        float(_scalar(result_offline.period_days[0])),
        rel=0.10,
    )
    assert _scalar(result_pred.period_lower_days[0]) is None
    assert _scalar(result_pred.period_upper_days[0]) is None
    assert _scalar(result_pred.relative_period_uncertainty[0]) is None


def test_harmonic_factor_classifies_simple_vs_alias_ratios():
    simple_factor, simple_mismatch = _best_harmonic_factor(
        4.0,
        2.0,
        harmonic_period_factors=(0.5, 2.0 / 3.0, 1.0, 1.5, 2.0),
    )
    assert simple_mismatch == pytest.approx(0.0)
    assert simple_factor == pytest.approx(0.5)
    assert _is_simple_harmonic_factor(simple_factor) is True

    alias_factor, alias_mismatch = _best_harmonic_factor(
        6.0,
        4.0,
        harmonic_period_factors=(0.5, 2.0 / 3.0, 1.0, 1.5, 2.0),
    )
    assert alias_mismatch == pytest.approx(0.0)
    assert alias_factor == pytest.approx(2.0 / 3.0)
    assert _is_simple_harmonic_factor(alias_factor) is False


def test_hybrid_consensus_class():
    observations = _make_rotation_observations()
    result = estimate_rotation_period(
        observations,
        method_mode="hybrid",
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert _scalar(result.primary_method[0]) in {"fourier", "lsm"}
    assert _scalar(result.method_agreement_class[0]) in {"consensus", "weak_consensus"}


def test_hybrid_method_dominant_class():
    observations = _make_rotation_observations(single_peaked=True)
    result = estimate_rotation_period(
        observations,
        method_mode="hybrid",
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert _scalar(result.primary_method[0]) == "fourier"
    assert _scalar(result.method_agreement_class[0]) in {"consensus", "weak_consensus"}
    # Per D1 the verdict is one of the three categorical states and the
    # reliability_code is the matching ordinal; is_reliable is derived from it.
    verdict = _scalar(result.period_verdict[0])
    assert verdict in {"single_period", "period_family", "insufficient_data"}
    assert _scalar(result.reliability_code[0]) in {"1", "2", "3"}
    assert bool(_scalar(result.is_reliable[0])) is (verdict == "single_period")
    assert bool(_scalar(result.is_valid[0])) is (
        verdict in {"single_period", "period_family"}
    )


def test_hybrid_unreliable_lsm_branch_does_not_displace_fourier(monkeypatch):
    observations = _make_rotation_observations()

    def fake_run_lsm(**kwargs):  # noqa: ARG001
        return _LSMMethodResult(
            best_candidate=_LSMCandidate(
                frequency=20.0,
                period_days=0.05,
                power=0.8,
                coeffs=np.zeros(4, dtype=np.float64),
                n_maxima=2,
                n_minima=2,
                amplitude_mag=0.2,
            ),
            power_gap=0.05,
            candidate_period_days=[0.05],
            candidate_powers=[0.8],
            is_reliable=False,
            amplitude_mag=0.2,
        )

    monkeypatch.setattr(rotation_period_fourier, "_run_lsm", fake_run_lsm)
    result = estimate_rotation_period(
        observations,
        method_mode="hybrid",
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert _scalar(result.primary_method[0]) == "fourier"
    assert _scalar(result.method_agreement_class[0]) == "method_dominant"


def test_hybrid_can_return_winning_family_representative_not_method_primary():
    primary_fit = _fit_template(frequency=10.0, fourier_order=2, rss=1.0, df=40)
    alternate_fit = _fit_template(frequency=5.0, fourier_order=2, rss=1.1, df=40)
    primary_period = _FitWithPeriod(
        fit=primary_fit,
        period_days=0.10,
        period_hours=2.4,
        is_period_doubled=False,
    )
    alternate_period = _FitWithPeriod(
        fit=alternate_fit,
        period_days=0.20,
        period_hours=4.8,
        is_period_doubled=False,
    )
    primary_cluster = _FourierCluster(
        indices=np.asarray([0], dtype=np.int64),
        best=primary_period,
        period_lower_days=0.099,
        period_upper_days=0.101,
        sigma_best=1.0,
        raw_weight=0.2,
    )
    alternate_cluster = _FourierCluster(
        indices=np.asarray([1], dtype=np.int64),
        best=alternate_period,
        period_lower_days=0.199,
        period_upper_days=0.201,
        sigma_best=1.1,
        raw_weight=0.8,
    )
    fourier_solution = _FourierSolution(
        chosen=primary_period,
        primary_cluster=primary_cluster,
        sigma_threshold=1.2,
        clusters=[primary_cluster, alternate_cluster],
        period_lower_days=0.099,
        period_upper_days=0.101,
        relative_period_uncertainty=0.01,
        alternate_period_days=[0.20],
        is_valid=True,
        is_reliable=True,
        amplitude_mag=0.3,
        used_session_offsets=False,
        fit_summary=primary_fit,
        sigma_curve=np.asarray([1.0, 1.1], dtype=np.float64),
    )
    lsm_solution = _LSMSolution(
        period_days=0.20,
        power=0.9,
        power_gap=0.2,
        candidate_period_days=[0.20],
        candidate_powers=[0.9],
        is_reliable=True,
        amplitude_mag=0.3,
        n_fit_observations=60,
        n_clipped=0,
        false_alarm_probability=1.0e-4,
    )
    family_primary = _MethodFamily(
        representative_period_days=0.10,
        fourier_cluster=primary_cluster,
        contains_fourier_primary=True,
        family_weight_fourier=0.1,
        combined_weight=0.05,
    )
    family_alternate = _MethodFamily(
        representative_period_days=0.20,
        fourier_cluster=alternate_cluster,
        lsm_candidate=_LSMCandidate(period_days=0.20, power=0.9, coeffs=np.zeros(4, dtype=np.float64)),
        contains_lsm_primary=True,
        family_weight_fourier=1.0,
        family_weight_lsm=0.8,
        combined_weight=0.9,
    )

    # Times that fold to ~full phase coverage over many rotations at P=0.20 d.
    t_rel = np.linspace(0.0, 2.0, 60, dtype=np.float64)
    primary = _primary_from_method(
        method_mode="hybrid",
        fourier_solution=fourier_solution,
        lsm_solution=lsm_solution,
        families=[family_primary, family_alternate],
        harmonic_period_factors=(0.5, 1.0, 2.0),
        filter_labels=np.asarray(["g"] * 30 + ["r"] * 30, dtype=object),
        t_rel=t_rel,
        span_days=2.0,
        min_rotations_in_span=2.0,
        residual_sigma_mag=0.02,
    )

    assert primary["primary_method"] == "fourier"
    assert primary["period_days"] == pytest.approx(0.20)
    assert primary["winner_contains_fourier_primary"] is False
    assert primary["winner_contains_lsm_primary"] is True
    assert primary["method_agreement_class"] == "consensus"
    # Two credible families survive (n_significant_aliases == 2), so even though
    # Fourier and LSM agree on the winning family the alias gate demotes the
    # verdict to period_family / reliability_code "2" (D1 "believe the family").
    assert primary["period_verdict"] == "period_family"
    assert primary["reliability_code"] == "2"
    assert primary["is_reliable"] is False
    assert primary["is_valid"] is True
    assert "dual_method_agree" in primary["confidence_flags"]
    assert "conflicting_aliases" in primary["insufficiency_reasons"]


def test_early_exit_on_insufficient_matches_full_path_verdict():
    # Deliberately no-signal data: flat, noise-only reduced magnitudes over a
    # multi-night span. The robust scatter is below the photometric noise, so
    # both the cheap pre-check and the full solve flag amplitude_below_noise.
    n = 40
    mjd = np.linspace(60000.0, 60003.0, n, dtype=np.float64)
    time = Timestamp.from_mjd(mjd, scale="tdb")
    filters = np.asarray(["LSST_r"] * n, dtype=object)
    rng = np.random.default_rng(2026)
    mag = 18.0 + rng.normal(0.0, 0.005, size=n)
    observations = RotationPeriodObservations.from_kwargs(
        time=time,
        mag=pa.array(mag, type=pa.float64()),
        mag_sigma=pa.array(np.full(n, 0.05, dtype=np.float64), type=pa.float64()),
        filter=pa.array(filters.tolist(), type=pa.large_string()),
        r_au=pa.array(np.full(n, 2.0, dtype=np.float64), type=pa.float64()),
        delta_au=pa.array(np.full(n, 1.5, dtype=np.float64), type=pa.float64()),
        phase_angle_deg=pa.array(np.full(n, 12.0, dtype=np.float64), type=pa.float64()),
    )

    early = estimate_rotation_period(
        observations, method_mode="fourier", early_exit_on_insufficient=True
    )
    assert isinstance(early, RotationPeriodResult)
    assert len(early) == 1
    assert _scalar(early.period_verdict[0]) == "insufficient_data"
    assert np.isnan(float(_scalar(early.period_days[0])))
    assert bool(_scalar(early.is_valid[0])) is False
    assert "amplitude_below_noise" in early.insufficiency_reasons[0].as_py()

    # Same input without the fast path must reach the SAME verdict.
    full = estimate_rotation_period(
        observations, method_mode="fourier", early_exit_on_insufficient=False
    )
    assert _scalar(full.period_verdict[0]) == "insufficient_data"


@pytest.mark.parametrize("search_fidelity", ["validated_staged", "exact_grid"])
def test_fourier_search_fidelities_match(search_fidelity: str):
    observations = _make_rotation_observations()
    result = estimate_rotation_period(
        observations,
        method_mode="fourier",
        search_fidelity=search_fidelity,
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert float(_scalar(result.period_days[0])) == pytest.approx(0.02, rel=0.10)
    assert int(_scalar(result.fourier_order[0])) == 2


def test_staged_and_exact_fourier_match():
    observations = _make_rotation_observations()
    staged = estimate_rotation_period(
        observations,
        method_mode="fourier",
        search_fidelity="validated_staged",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )
    exact = estimate_rotation_period(
        observations,
        method_mode="fourier",
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert float(_scalar(staged.period_days[0])) == pytest.approx(float(_scalar(exact.period_days[0])), rel=0.02)
    assert int(_scalar(staged.fourier_order[0])) == int(_scalar(exact.fourier_order[0]))
    assert float(_scalar(staged.period_lower_days[0])) == pytest.approx(float(_scalar(exact.period_lower_days[0])), rel=0.05)
    assert float(_scalar(staged.period_upper_days[0])) == pytest.approx(float(_scalar(exact.period_upper_days[0])), rel=0.05)


def test_numpy_and_jax_exact_paths_match():
    observations = _make_rotation_observations()
    numpy_result = estimate_rotation_period(
        observations,
        method_mode="fourier",
        search_fidelity="exact_grid",
        exact_evaluation_backend="numpy",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )
    jax_result = estimate_rotation_period(
        observations,
        method_mode="fourier",
        search_fidelity="exact_grid",
        exact_evaluation_backend="jax",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert float(_scalar(numpy_result.period_days[0])) == pytest.approx(float(_scalar(jax_result.period_days[0])), rel=1.0e-8)
    assert int(_scalar(numpy_result.fourier_order[0])) == int(_scalar(jax_result.fourier_order[0]))


def test_max_search_period_cap_and_long_period_guardrail(monkeypatch):
    # rp-e4a.22 step 2: the cap raises f_min so the longest searched period is
    # bounded; default (None) leaves the floor at min_rotations_in_span / span.
    # A long span pushes the natural f_min below the cap floor so the cap bites.
    span_days = 100.0
    uncapped = _build_frequency_grid(
        span_days=span_days,
        min_rotations_in_span=2.0,
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )
    capped = _build_frequency_grid(
        span_days=span_days,
        min_rotations_in_span=2.0,
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
        max_search_period_hours=72.0,
    )
    assert float(uncapped[0]) == pytest.approx(2.0 / span_days)  # 0.02 cyc/day
    assert float(capped[0]) == pytest.approx(24.0 / 72.0)  # raised floor 0.333
    assert float(capped[0]) > float(uncapped[0])

    # rp-e4a.22 step 1: a single_period verdict at a period beyond
    # MAX_PLAUSIBLE_SINGLE_PERIOD_HOURS is downgraded to period_family while the
    # reported period value is preserved.
    observations = _make_rotation_observations()
    long_period_days = (MAX_PLAUSIBLE_SINGLE_PERIOD_HOURS + 24.0) / 24.0
    real_primary = rotation_period_fourier._primary_from_method

    def fake_primary(**kwargs):
        primary = real_primary(**kwargs)
        primary["period_days"] = long_period_days
        primary["period_verdict"] = "single_period"
        primary["reliability_code"] = "3"
        primary["insufficiency_reasons"] = []
        primary["is_valid"] = True
        primary["is_reliable"] = True
        return primary

    monkeypatch.setattr(rotation_period_fourier, "_primary_from_method", fake_primary)
    result = estimate_rotation_period(
        observations,
        method_mode="fourier",
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert _scalar(result.period_verdict[0]) == "period_family"
    assert _scalar(result.reliability_code[0]) == "2"
    assert bool(_scalar(result.is_reliable[0])) is False
    assert "period_implausibly_long" in list(result.insufficiency_reasons[0].as_py())
    # The recovered period value itself is still reported, only downgraded.
    assert float(_scalar(result.period_hours[0])) == pytest.approx(long_period_days * 24.0)
