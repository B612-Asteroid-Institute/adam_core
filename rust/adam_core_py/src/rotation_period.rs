use adam_core_rs_coords::rotation_period::{
    alias_bucket, estimate_rotation_period, estimate_rotation_period_best_apparition,
    estimate_rotation_period_grouped, harmonic_adjusted_error_pct, near_day_alias,
    relative_error_pct, within_tolerance, RotationPeriodConfig, RotationPeriodEstimate,
    RotationPeriodInput,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::hint::black_box;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn input_from_python(
    time_days: PyReadonlyArray1<'_, i64>,
    time_nanos: PyReadonlyArray1<'_, i64>,
    time_scale: &str,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    filter: Vec<String>,
    session_id: Vec<Option<String>>,
    r_au: PyReadonlyArray1<'_, f64>,
    delta_au: PyReadonlyArray1<'_, f64>,
    phase_angle_deg: PyReadonlyArray1<'_, f64>,
) -> PyResult<RotationPeriodInput> {
    let days = time_days
        .as_slice()
        .map_err(|_| PyValueError::new_err("time days must be contiguous"))?;
    let nanos = time_nanos
        .as_slice()
        .map_err(|_| PyValueError::new_err("time nanos must be contiguous"))?;
    Ok(RotationPeriodInput {
        time: crate::timestamp_ops::time_array(days, nanos, time_scale)?,
        magnitude: magnitude
            .as_slice()
            .map_err(|_| PyValueError::new_err("magnitude must be contiguous"))?
            .to_vec(),
        magnitude_sigma: magnitude_sigma
            .as_slice()
            .map_err(|_| PyValueError::new_err("magnitude_sigma must be contiguous"))?
            .to_vec(),
        filter,
        session_id,
        r_au: r_au
            .as_slice()
            .map_err(|_| PyValueError::new_err("r_au must be contiguous"))?
            .to_vec(),
        delta_au: delta_au
            .as_slice()
            .map_err(|_| PyValueError::new_err("delta_au must be contiguous"))?
            .to_vec(),
        phase_angle_deg: phase_angle_deg
            .as_slice()
            .map_err(|_| PyValueError::new_err("phase_angle_deg must be contiguous"))?
            .to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn config_from_python(
    search_fidelity: String,
    fourier_orders: Option<Vec<usize>>,
    clip_sigma: f64,
    min_rotations_in_span: f64,
    max_frequency_cycles_per_day: f64,
    frequency_grid_scale: f64,
    max_search_period_hours: Option<f64>,
    early_exit_on_insufficient: bool,
    session_mode: String,
    auto_session_min_observations_per_group: usize,
    auto_session_bic_improvement: f64,
) -> RotationPeriodConfig {
    RotationPeriodConfig {
        search_fidelity,
        fourier_orders,
        clip_sigma,
        min_rotations_in_span,
        max_frequency_cycles_per_day,
        frequency_grid_scale,
        max_search_period_hours,
        early_exit_on_insufficient,
        session_mode,
        auto_session_min_observations_per_group,
        auto_session_bic_improvement,
    }
}

pub(crate) fn estimate_to_dict<'py>(
    py: Python<'py>,
    estimate: RotationPeriodEstimate,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("period_days", estimate.period_days)?;
    output.set_item("period_hours", estimate.period_days * 24.0)?;
    output.set_item(
        "frequency_cycles_per_day",
        if estimate.period_days.is_finite() && estimate.period_days > 0.0 {
            1.0 / estimate.period_days
        } else {
            f64::NAN
        },
    )?;
    output.set_item("period_verdict", estimate.period_verdict)?;
    output.set_item("reliability_code", estimate.reliability_code)?;
    output.set_item("confidence_flags", estimate.confidence_flags)?;
    output.set_item("insufficiency_reasons", estimate.insufficiency_reasons)?;
    output.set_item("is_valid", estimate.is_valid)?;
    output.set_item("is_reliable", estimate.is_reliable)?;
    output.set_item("period_lower_days", estimate.period_lower_days)?;
    output.set_item("period_upper_days", estimate.period_upper_days)?;
    output.set_item(
        "relative_period_uncertainty",
        estimate.relative_period_uncertainty,
    )?;
    output.set_item("alternate_period_days", estimate.alternate_period_days)?;
    output.set_item("fourier_period_days", estimate.fourier_period_days)?;
    output.set_item("fourier_order", estimate.fourier_order)?;
    output.set_item("fourier_sigma_threshold", estimate.fourier_sigma_threshold)?;
    output.set_item("fourier_phase_c1", estimate.fourier_phase_c1)?;
    output.set_item("fourier_phase_c2", estimate.fourier_phase_c2)?;
    output.set_item("residual_sigma_mag", estimate.residual_sigma_mag)?;
    output.set_item("fourier_is_valid", estimate.fourier_is_valid)?;
    output.set_item("fourier_is_reliable", estimate.fourier_is_reliable)?;
    output.set_item(
        "fourier_alternate_period_days",
        estimate.fourier_alternate_period_days,
    )?;
    output.set_item("phase_coverage_fraction", estimate.phase_coverage_fraction)?;
    output.set_item("n_rotations_spanned", estimate.n_rotations_spanned)?;
    output.set_item("amplitude_snr", estimate.amplitude_snr)?;
    output.set_item("n_significant_aliases", estimate.n_significant_aliases)?;
    output.set_item("n_observations", estimate.n_observations)?;
    output.set_item("n_fit_observations", estimate.n_fit_observations)?;
    output.set_item("n_clipped", estimate.n_clipped)?;
    output.set_item("n_filters", estimate.n_filters)?;
    output.set_item("n_sessions", estimate.n_sessions)?;
    output.set_item("used_session_offsets", estimate.used_session_offsets)?;
    output.set_item("is_period_doubled", estimate.is_period_doubled)?;
    Ok(output)
}

#[pyfunction]
#[pyo3(signature = (
    time_days, time_nanos, time_scale, magnitude, magnitude_sigma, filter, session_id, r_au, delta_au,
    phase_angle_deg, search_fidelity, fourier_orders, clip_sigma,
    min_rotations_in_span, max_frequency_cycles_per_day, frequency_grid_scale,
    max_search_period_hours, early_exit_on_insufficient, session_mode,
    auto_session_min_observations_per_group, auto_session_bic_improvement
))]
#[allow(clippy::too_many_arguments)]
fn rotation_period_estimate<'py>(
    py: Python<'py>,
    time_days: PyReadonlyArray1<'_, i64>,
    time_nanos: PyReadonlyArray1<'_, i64>,
    time_scale: &str,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    filter: Vec<String>,
    session_id: Vec<Option<String>>,
    r_au: PyReadonlyArray1<'_, f64>,
    delta_au: PyReadonlyArray1<'_, f64>,
    phase_angle_deg: PyReadonlyArray1<'_, f64>,
    search_fidelity: String,
    fourier_orders: Option<Vec<usize>>,
    clip_sigma: f64,
    min_rotations_in_span: f64,
    max_frequency_cycles_per_day: f64,
    frequency_grid_scale: f64,
    max_search_period_hours: Option<f64>,
    early_exit_on_insufficient: bool,
    session_mode: String,
    auto_session_min_observations_per_group: usize,
    auto_session_bic_improvement: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let input = input_from_python(
        time_days,
        time_nanos,
        time_scale,
        magnitude,
        magnitude_sigma,
        filter,
        session_id,
        r_au,
        delta_au,
        phase_angle_deg,
    )?;
    let config = config_from_python(
        search_fidelity,
        fourier_orders,
        clip_sigma,
        min_rotations_in_span,
        max_frequency_cycles_per_day,
        frequency_grid_scale,
        max_search_period_hours,
        early_exit_on_insufficient,
        session_mode,
        auto_session_min_observations_per_group,
        auto_session_bic_improvement,
    );
    let estimate = py
        .allow_threads(|| estimate_rotation_period(&input, &config))
        .map_err(PyValueError::new_err)?;
    estimate_to_dict(py, estimate)
}

#[pyfunction]
#[pyo3(signature = (
    time_days, time_nanos, time_scale, magnitude, magnitude_sigma, filter, session_id, r_au,
    delta_au, phase_angle_deg, object_ids, search_fidelity, fourier_orders,
    clip_sigma, min_rotations_in_span, max_frequency_cycles_per_day, frequency_grid_scale,
    max_search_period_hours, early_exit_on_insufficient, session_mode,
    auto_session_min_observations_per_group, auto_session_bic_improvement
))]
#[allow(clippy::too_many_arguments)]
fn rotation_period_estimate_grouped<'py>(
    py: Python<'py>,
    time_days: PyReadonlyArray1<'_, i64>,
    time_nanos: PyReadonlyArray1<'_, i64>,
    time_scale: &str,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    filter: Vec<String>,
    session_id: Vec<Option<String>>,
    r_au: PyReadonlyArray1<'_, f64>,
    delta_au: PyReadonlyArray1<'_, f64>,
    phase_angle_deg: PyReadonlyArray1<'_, f64>,
    object_ids: Vec<Option<String>>,
    search_fidelity: String,
    fourier_orders: Option<Vec<usize>>,
    clip_sigma: f64,
    min_rotations_in_span: f64,
    max_frequency_cycles_per_day: f64,
    frequency_grid_scale: f64,
    max_search_period_hours: Option<f64>,
    early_exit_on_insufficient: bool,
    session_mode: String,
    auto_session_min_observations_per_group: usize,
    auto_session_bic_improvement: f64,
) -> PyResult<Bound<'py, PyList>> {
    let input = input_from_python(
        time_days,
        time_nanos,
        time_scale,
        magnitude,
        magnitude_sigma,
        filter,
        session_id,
        r_au,
        delta_au,
        phase_angle_deg,
    )?;
    let config = config_from_python(
        search_fidelity,
        fourier_orders,
        clip_sigma,
        min_rotations_in_span,
        max_frequency_cycles_per_day,
        frequency_grid_scale,
        max_search_period_hours,
        early_exit_on_insufficient,
        session_mode,
        auto_session_min_observations_per_group,
        auto_session_bic_improvement,
    );
    let estimates = py
        .allow_threads(|| estimate_rotation_period_grouped(&input, &object_ids, &config))
        .map_err(PyValueError::new_err)?;
    let output = PyList::empty(py);
    for (object_id, estimate) in estimates {
        output.append((object_id, estimate_to_dict(py, estimate)?))?;
    }
    Ok(output)
}

#[pyfunction]
#[pyo3(signature = (
    time_days, time_nanos, time_scale, magnitude, magnitude_sigma, filter, session_id, r_au,
    delta_au, phase_angle_deg, apparition_gap_days, search_fidelity, fourier_orders,
    clip_sigma, min_rotations_in_span, max_frequency_cycles_per_day, frequency_grid_scale,
    max_search_period_hours, early_exit_on_insufficient, session_mode,
    auto_session_min_observations_per_group, auto_session_bic_improvement
))]
#[allow(clippy::too_many_arguments)]
fn rotation_period_estimate_best_apparition<'py>(
    py: Python<'py>,
    time_days: PyReadonlyArray1<'_, i64>,
    time_nanos: PyReadonlyArray1<'_, i64>,
    time_scale: &str,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    filter: Vec<String>,
    session_id: Vec<Option<String>>,
    r_au: PyReadonlyArray1<'_, f64>,
    delta_au: PyReadonlyArray1<'_, f64>,
    phase_angle_deg: PyReadonlyArray1<'_, f64>,
    apparition_gap_days: f64,
    search_fidelity: String,
    fourier_orders: Option<Vec<usize>>,
    clip_sigma: f64,
    min_rotations_in_span: f64,
    max_frequency_cycles_per_day: f64,
    frequency_grid_scale: f64,
    max_search_period_hours: Option<f64>,
    early_exit_on_insufficient: bool,
    session_mode: String,
    auto_session_min_observations_per_group: usize,
    auto_session_bic_improvement: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let input = input_from_python(
        time_days,
        time_nanos,
        time_scale,
        magnitude,
        magnitude_sigma,
        filter,
        session_id,
        r_au,
        delta_au,
        phase_angle_deg,
    )?;
    let config = config_from_python(
        search_fidelity,
        fourier_orders,
        clip_sigma,
        min_rotations_in_span,
        max_frequency_cycles_per_day,
        frequency_grid_scale,
        max_search_period_hours,
        early_exit_on_insufficient,
        session_mode,
        auto_session_min_observations_per_group,
        auto_session_bic_improvement,
    );
    let (estimate, _, _) = py
        .allow_threads(|| {
            estimate_rotation_period_best_apparition(&input, &config, apparition_gap_days)
        })
        .map_err(PyValueError::new_err)?;
    estimate_to_dict(py, estimate)
}

#[pyfunction]
#[pyo3(signature = (
    time_days, time_nanos, time_scale, magnitude, magnitude_sigma, filter, session_id, r_au, delta_au,
    phase_angle_deg, reps, trials, warmup_reps, search_fidelity,
    fourier_orders, clip_sigma, min_rotations_in_span,
    max_frequency_cycles_per_day, frequency_grid_scale, max_search_period_hours,
    early_exit_on_insufficient, session_mode,
    auto_session_min_observations_per_group, auto_session_bic_improvement
))]
#[allow(clippy::too_many_arguments)]
fn benchmark_rotation_period_native(
    py: Python<'_>,
    time_days: PyReadonlyArray1<'_, i64>,
    time_nanos: PyReadonlyArray1<'_, i64>,
    time_scale: &str,
    magnitude: PyReadonlyArray1<'_, f64>,
    magnitude_sigma: PyReadonlyArray1<'_, f64>,
    filter: Vec<String>,
    session_id: Vec<Option<String>>,
    r_au: PyReadonlyArray1<'_, f64>,
    delta_au: PyReadonlyArray1<'_, f64>,
    phase_angle_deg: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    search_fidelity: String,
    fourier_orders: Option<Vec<usize>>,
    clip_sigma: f64,
    min_rotations_in_span: f64,
    max_frequency_cycles_per_day: f64,
    frequency_grid_scale: f64,
    max_search_period_hours: Option<f64>,
    early_exit_on_insufficient: bool,
    session_mode: String,
    auto_session_min_observations_per_group: usize,
    auto_session_bic_improvement: f64,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let input = input_from_python(
        time_days,
        time_nanos,
        time_scale,
        magnitude,
        magnitude_sigma,
        filter,
        session_id,
        r_au,
        delta_au,
        phase_angle_deg,
    )?;
    let config = config_from_python(
        search_fidelity,
        fourier_orders,
        clip_sigma,
        min_rotations_in_span,
        max_frequency_cycles_per_day,
        frequency_grid_scale,
        max_search_period_hours,
        early_exit_on_insufficient,
        session_mode,
        auto_session_min_observations_per_group,
        auto_session_bic_improvement,
    );
    py.allow_threads(|| {
        estimate_rotation_period(&input, &config).map_err(PyValueError::new_err)?;
        let mut trial_samples = Vec::with_capacity(trials);
        for _ in 0..trials {
            for _ in 0..warmup_reps {
                black_box(
                    estimate_rotation_period(&input, &config).map_err(PyValueError::new_err)?,
                );
            }
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                let started = Instant::now();
                let result =
                    estimate_rotation_period(&input, &config).map_err(PyValueError::new_err)?;
                black_box(result);
                samples.push(started.elapsed().as_secs_f64());
            }
            trial_samples.push(samples);
        }
        Ok(trial_samples)
    })
}

#[pyfunction]
fn rotation_period_hours_numpy<'py>(
    py: Python<'py>,
    period_days: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = period_days
        .as_slice()
        .map_err(|_| PyValueError::new_err("period_days must be contiguous"))?
        .iter()
        .map(|value| value * 24.0)
        .collect::<Vec<_>>();
    Ok(values.into_pyarray(py))
}

#[pyfunction]
fn rotation_relative_error_pct(recovered: f64, truth: f64) -> f64 {
    relative_error_pct(recovered, truth)
}

#[pyfunction]
fn rotation_harmonic_adjusted_error_pct(recovered: f64, truth: f64) -> (f64, f64) {
    harmonic_adjusted_error_pct(recovered, truth)
}

#[pyfunction]
fn rotation_alias_bucket(factor: f64) -> String {
    alias_bucket(factor).to_string()
}

#[pyfunction]
fn rotation_within_tolerance(recovered: f64, truth: f64, tolerance: f64) -> bool {
    within_tolerance(recovered, truth, tolerance)
}

#[pyfunction]
#[pyo3(signature = (recovered, truth, tolerance=0.02))]
fn rotation_near_day_alias(recovered: f64, truth: f64, tolerance: f64) -> bool {
    near_day_alias(recovered, truth, tolerance)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rotation_period_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_period_estimate_grouped, m)?)?;
    m.add_function(wrap_pyfunction!(
        rotation_period_estimate_best_apparition,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(benchmark_rotation_period_native, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_period_hours_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_relative_error_pct, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_harmonic_adjusted_error_pct, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_alias_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_within_tolerance, m)?)?;
    m.add_function(wrap_pyfunction!(rotation_near_day_alias, m)?)?;
    Ok(())
}
