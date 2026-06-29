from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import adam_core.photometry.rotation_period_fourier as rotation_period_fourier

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observers.observers import Observers
from ...time import Timestamp
from ..magnitude import calculate_apparent_magnitude_v_and_phase_angle
from ..rotation_period_fourier import (
    MAX_PLAUSIBLE_SINGLE_PERIOD_HOURS,
    _build_frequency_grid,
    _grid_was_capped,
    _observation_count_sufficient,
    estimate_rotation_period,
)
from ..rotation_period_fourier_core import (
    _f_test_confidence,
    _FitResult,
    _hg_phase_reduced,
    _select_order,
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

    baseline = (
        15.0
        + 5.0 * np.log10(r_au * delta_au)
        + 0.015 * phase_angle
        + 0.0015 * np.square(phase_angle)
    )
    phase = 2.0 * np.pi * t_rel / period_days
    if single_peaked:
        rotation = amplitude * np.cos(phase)
    else:
        rotation = (
            0.10 * np.cos(phase)
            + amplitude * np.cos(2.0 * phase)
            + 0.07 * np.sin(2.0 * phase)
        )
    rotation = rotation + np.asarray(
        [0.03 if name == "LSST_g" else 0.0 for name in filters], dtype=np.float64
    )
    rng = np.random.default_rng(20260414)
    mag = baseline + rotation + rng.normal(0.0, noise_sigma, size=n)

    return RotationPeriodObservations.from_kwargs(
        time=time,
        mag=pa.array(mag, type=pa.float64()),
        mag_sigma=pa.array(np.full(n, 0.03, dtype=np.float64), type=pa.float64()),
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
        n_fit=int(df + 3 + 2 * fourier_order),
        n_clipped=0,
        phase_c1_idx=1,
        phase_c2_idx=2,
    )


def test_profile_specific_order_selection_thresholds():
    small = _fit_template(frequency=20.0, fourier_order=2, rss=30.0, df=20)
    better = _fit_template(frequency=20.0, fourier_order=3, rss=14.0, df=18)
    confidence = _f_test_confidence(small, better)
    assert 0.90 < confidence < 0.95

    candidates = {2: small, 3: better}
    assert _select_order(candidates, 0.90).fourier_order == 3
    assert _select_order(candidates, 0.95).fourier_order == 2


def test_fourier_single_peak_is_doubled():
    observations = _make_rotation_observations(single_peaked=True, period_days=0.02)
    result = estimate_rotation_period(
        observations,
        profile="default",
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert isinstance(result, RotationPeriodResult)
    assert float(_scalar(result.period_days[0])) == pytest.approx(0.04, rel=0.10)
    assert all(
        float(period) > 0.03
        for period in result.fourier_alternate_period_days[0].as_py()
    )


def test_observation_count_sufficient_gates_on_dominant_band():
    # The signal-gate observation count fires on the MOST-POPULATED band (the
    # jointly-fit shape is carried by the richest band + per-band offsets), not on
    # the top two bands each clearing the floor.
    def labels(counts: dict[str, int]) -> np.ndarray:
        out: list[str] = []
        for band, n in counts.items():
            out.extend([band] * n)
        return np.asarray(out, dtype=object)

    # Single-band semantics are preserved exactly: >= 30 passes, < 30 fails.
    assert _observation_count_sufficient(labels({"R": 30})) is True
    assert _observation_count_sufficient(labels({"R": 29})) is False
    # The fix: a single-band-dominant lightcurve (Beate-like {C: 457, R: 27})
    # passes even though the secondary band has < 30 -- the legacy top-two rule
    # wrongly rejected this as too_few_observations.
    assert _observation_count_sufficient(labels({"C": 457, "R": 27})) is True
    # Genuinely sparse data stays insufficient: no band reaches the floor.
    assert _observation_count_sufficient(labels({"R": 22})) is False
    # Balanced-but-thin multi-band (no single band >= 30) is deliberately kept
    # out -- the regime most exposed to harmonic-alias confusion.
    assert _observation_count_sufficient(labels({"g": 12, "r": 12, "i": 12})) is False


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

    early = estimate_rotation_period(observations, early_exit_on_insufficient=True)
    assert isinstance(early, RotationPeriodResult)
    assert len(early) == 1
    assert _scalar(early.period_verdict[0]) == "insufficient_data"
    assert np.isnan(float(_scalar(early.period_days[0])))
    assert bool(_scalar(early.is_valid[0])) is False
    assert "amplitude_below_noise" in early.insufficiency_reasons[0].as_py()

    # Same input without the fast path must reach the SAME verdict.
    full = estimate_rotation_period(observations, early_exit_on_insufficient=False)
    assert _scalar(full.period_verdict[0]) == "insufficient_data"


@pytest.mark.parametrize("search_fidelity", ["validated_staged", "exact_grid"])
def test_fourier_search_fidelities_match(search_fidelity: str):
    observations = _make_rotation_observations()
    result = estimate_rotation_period(
        observations,
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
        search_fidelity="validated_staged",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )
    exact = estimate_rotation_period(
        observations,
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert float(_scalar(staged.period_days[0])) == pytest.approx(
        float(_scalar(exact.period_days[0])), rel=0.02
    )
    assert int(_scalar(staged.fourier_order[0])) == int(_scalar(exact.fourier_order[0]))
    assert float(_scalar(staged.period_lower_days[0])) == pytest.approx(
        float(_scalar(exact.period_lower_days[0])), rel=0.05
    )
    assert float(_scalar(staged.period_upper_days[0])) == pytest.approx(
        float(_scalar(exact.period_upper_days[0])), rel=0.05
    )


@pytest.mark.parametrize(
    "period_days, span_days, noise_sigma",
    [
        (0.020, 0.08, 0.010),
        (0.035, 0.10, 0.020),
        (0.014, 0.09, 0.015),
    ],
)
def test_staged_matches_exact_on_large_grid(period_days, span_days, noise_sigma):
    # rp-e4a.12 / review #8: with a >2048-point grid the validated_staged COARSE path
    # (coarse sample -> local-minima refine) actually runs, and the global-best guard
    # must keep it from missing the global-best grid region. Staged must match the
    # exact-grid period across several periods/spans/noise levels, and the result must
    # advertise that the staged heuristic was used.
    obs = _make_rotation_observations(
        n=80, span_days=span_days, period_days=period_days, noise_sigma=noise_sigma
    )
    kw = dict(
        exact_evaluation_backend="jax",
        max_frequency_cycles_per_day=700.0,
        frequency_grid_scale=40.0,
    )
    staged = estimate_rotation_period(obs, search_fidelity="validated_staged", **kw)
    exact = estimate_rotation_period(obs, search_fidelity="exact_grid", **kw)

    assert (
        _build_frequency_grid(
            span_days=span_days,
            min_rotations_in_span=2.0,
            max_frequency_cycles_per_day=700.0,
            frequency_grid_scale=40.0,
        ).size
        > 2048
    )  # the staged coarse path is genuinely exercised
    assert "staged_search_used" in list(staged.confidence_flags[0].as_py() or [])
    assert float(_scalar(staged.period_days[0])) == pytest.approx(
        float(_scalar(exact.period_days[0])), rel=0.05
    )


def test_grid_cap_diagnostic():
    # rp-e4a.12 / review #9: the hard frequency-grid cap is detectable as a diagnostic.
    assert (
        _grid_was_capped(
            span_days=4000.0,
            min_rotations_in_span=2.0,
            max_frequency_cycles_per_day=1000.0,
            frequency_grid_scale=30.0,
        )
        is True
    )
    assert (
        _grid_was_capped(
            span_days=0.1,
            min_rotations_in_span=2.0,
            max_frequency_cycles_per_day=120.0,
            frequency_grid_scale=40.0,
        )
        is False
    )


def test_numpy_and_jax_exact_paths_match():
    observations = _make_rotation_observations()
    numpy_result = estimate_rotation_period(
        observations,
        search_fidelity="exact_grid",
        exact_evaluation_backend="numpy",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )
    jax_result = estimate_rotation_period(
        observations,
        search_fidelity="exact_grid",
        exact_evaluation_backend="jax",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert float(_scalar(numpy_result.period_days[0])) == pytest.approx(
        float(_scalar(jax_result.period_days[0])), rel=1.0e-8
    )
    assert int(_scalar(numpy_result.fourier_order[0])) == int(
        _scalar(jax_result.fourier_order[0])
    )


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
        primary.period_days = long_period_days
        primary.period_verdict = "single_period"
        primary.reliability_code = "3"
        primary.insufficiency_reasons = []
        primary.is_valid = True
        primary.is_reliable = True
        return primary

    monkeypatch.setattr(rotation_period_fourier, "_primary_from_method", fake_primary)
    result = estimate_rotation_period(
        observations,
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert _scalar(result.period_verdict[0]) == "period_family"
    assert _scalar(result.reliability_code[0]) == "2"
    assert bool(_scalar(result.is_reliable[0])) is False
    assert "period_implausibly_long" in list(result.insufficiency_reasons[0].as_py())
    # The recovered period value itself is still reported, only downgraded.
    assert float(_scalar(result.period_hours[0])) == pytest.approx(
        long_period_days * 24.0
    )


def test_subharmonic_alias_below_grid_is_capped_to_family():
    # True period 20 h but the 30 h span (min_rotations_in_span=2) puts f_true
    # below the grid floor, so the solver locks onto the in-grid 2x alias (~10 h).
    # The lightcurve carries an odd 1st harmonic that genuinely distinguishes P
    # from P/2, so the sub-harmonic guardrail must refuse a confident single_period
    # and cap at period_family rather than emit the 0.5x alias (the cardinal D1
    # failure).
    rng = np.random.default_rng(7)
    p_true_d = 20.0 / 24.0
    n = 180
    t = np.sort(rng.uniform(0.0, 30.0 / 24.0, size=n)) + 60500.0
    omega = 2.0 * np.pi / p_true_d
    dt = t - t[0]
    rot = (
        0.20 * np.cos(2.0 * omega * dt)
        + 0.10 * np.cos(4.0 * omega * dt + 0.4)
        + 0.05 * np.cos(omega * dt + 1.1)
    )
    mag = 18.0 + rot + rng.normal(0.0, 0.01, size=n)
    obs = RotationPeriodObservations.from_kwargs(
        time=Timestamp.from_mjd(pa.array(t, type=pa.float64()), scale="tdb"),
        mag=pa.array(mag, type=pa.float64()),
        mag_sigma=pa.array(np.full(n, 0.01), type=pa.float64()),
        filter=pa.array(["r"] * n, type=pa.large_string()),
        session_id=pa.nulls(n, type=pa.large_string()),
        r_au=pa.array(np.full(n, 1.0), type=pa.float64()),
        delta_au=pa.array(np.full(n, 1.0), type=pa.float64()),
        phase_angle_deg=pa.array(np.linspace(5.0, 5.4, n), type=pa.float64()),
    )
    result = estimate_rotation_period(
        obs, search_fidelity="exact_grid", max_frequency_cycles_per_day=50.0
    )
    # The recovered value is the in-grid 0.5x alias (~10 h) -- kept, but the
    # verdict is hedged, not a confident single_period.
    assert float(_scalar(result.period_hours[0])) == pytest.approx(10.0, rel=0.1)
    assert _scalar(result.period_verdict[0]) == "period_family"
    assert "subharmonic_unresolved" in list(result.insufficiency_reasons[0].as_py())


def test_hg_phase_reduced_matches_magnitude_module() -> None:
    """The rotation prior's IAU H-G phase term must track adam_core's canonical
    (JAX) magnitude implementation -- the two re-implement the same law with the
    same constants (-3.33/0.63/-1.87/1.22), so they must not silently drift.
    (PR#200 review #11.)

    Strategy: drive the public apparent-magnitude path over a phase-angle sweep,
    back out its phase term (mag_v - H - 5*log10(r*delta)), and compare it to
    ``_hg_phase_reduced`` evaluated at the SAME returned phase angle. Feeding the
    identical alpha through both paths isolates the phase *function* (exp/pow/log
    + the H-G constants) from the shared geometry, so the only thing under test
    is that the two formulas agree.
    """
    alpha_target_deg = np.linspace(1.0, 120.0, 24)
    n = alpha_target_deg.size
    r_helio = 2.0
    delta = 1.0
    h_mag = 15.0

    # Object on +x at r_helio; its direction to the Sun is (-1, 0, 0). Put the
    # observer one delta away along a direction rotated by alpha from that, so the
    # Sun-object-observer angle (the phase angle) sweeps alpha_target_deg.
    alpha_rad = np.radians(alpha_target_deg)
    obs_x = r_helio - delta * np.cos(alpha_rad)
    obs_y = -delta * np.sin(alpha_rad)
    zeros = np.zeros(n)
    time = Timestamp.from_mjd([60000.0] * n, scale="tdb")
    obj = CartesianCoordinates.from_kwargs(
        x=np.full(n, r_helio),
        y=zeros,
        z=zeros,
        vx=zeros,
        vy=zeros,
        vz=zeros,
        time=time,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * n),
    )
    observer = Observers.from_kwargs(
        code=["500"] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=obs_x,
            y=obs_y,
            z=zeros,
            vx=zeros,
            vy=zeros,
            vz=zeros,
            time=time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * n),
        ),
    )

    distance_modulus = 5.0 * np.log10(r_helio * delta)
    for g_value in (0.15, 0.25):
        mags_v, alpha_deg = calculate_apparent_magnitude_v_and_phase_angle(
            h_mag, obj, observer, G=g_value
        )
        # Sanity: the constructed geometry really does span a wide phase range.
        assert float(alpha_deg.min()) < 5.0
        assert float(alpha_deg.max()) > 110.0
        # Phase term as the magnitude module computes it, with H and the distance
        # modulus 5*log10(r*delta) removed -- this is exactly -2.5*log10(phi).
        phase_term_ref = mags_v - h_mag - distance_modulus
        phase_term_rotation = _hg_phase_reduced(alpha_deg, g_value)
        np.testing.assert_allclose(
            phase_term_rotation, phase_term_ref, rtol=0.0, atol=1e-9
        )
