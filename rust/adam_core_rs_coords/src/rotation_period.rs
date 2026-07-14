//! Rust-native asteroid rotation-period estimation.
//!
//! This module owns the frequency search, clipped Fourier regression, alias
//! clustering, confidence classification, and diagnostic assembly used by the
//! Python ``photometry.rotation`` veneer.  It intentionally has no Python or
//! Arrow dependency so non-Python consumers can run the same estimator.

use crate::{TimeArray, TimeScale};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::f64::consts::TAU;

const LIGHT_SPEED_AU_PER_DAY: f64 = 173.144_632_674_240_3;
const MAX_FREQUENCIES: usize = 200_000;
const MIN_OBSERVATIONS: usize = 8;
const PRE_SOLVE_OBSERVATION_MARGIN: usize = 2;
const PRE_SOLVE_AMPLITUDE_SNR_FLOOR: f64 = 1.5;
const PRE_SOLVE_SCATTER_FLOOR: f64 = 0.02;
const OBSERVATION_COUNT_MIN: usize = 30;
const AMPLITUDE_SNR_MIN: f64 = 3.0;
const PHASE_COVERAGE_MIN_FAMILY: f64 = 0.5;
const PHASE_COVERAGE_MIN_SINGLE: f64 = 0.7;
const PHASE_BINS: usize = 20;
const MAX_PLAUSIBLE_SINGLE_PERIOD_HOURS: f64 = 240.0;
const PRIOR_SIGMA_FLOOR: f64 = 1.0e-6;
const MAX_CLIP_ITERATIONS: usize = 8;
const DEFAULT_PROFILE_MAX_FOURIER_ORDER: usize = 6;

#[derive(Clone, Debug)]
pub struct RotationPeriodInput {
    pub time: TimeArray,
    pub magnitude: Vec<f64>,
    pub magnitude_sigma: Vec<f64>,
    pub filter: Vec<String>,
    pub session_id: Vec<Option<String>>,
    pub r_au: Vec<f64>,
    pub delta_au: Vec<f64>,
    pub phase_angle_deg: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct RotationPeriodConfig {
    pub search_fidelity: String,
    pub fourier_orders: Option<Vec<usize>>,
    pub clip_sigma: f64,
    pub min_rotations_in_span: f64,
    pub max_frequency_cycles_per_day: f64,
    pub frequency_grid_scale: f64,
    pub max_search_period_hours: Option<f64>,
    pub early_exit_on_insufficient: bool,
    pub session_mode: String,
    pub auto_session_min_observations_per_group: usize,
    pub auto_session_bic_improvement: f64,
}

impl Default for RotationPeriodConfig {
    fn default() -> Self {
        Self {
            search_fidelity: "validated_staged".to_string(),
            fourier_orders: None,
            clip_sigma: 3.0,
            min_rotations_in_span: 2.0,
            max_frequency_cycles_per_day: 1000.0,
            frequency_grid_scale: 30.0,
            max_search_period_hours: None,
            early_exit_on_insufficient: true,
            session_mode: "auto".to_string(),
            auto_session_min_observations_per_group: 6,
            auto_session_bic_improvement: 10.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RotationPeriodEstimate {
    pub period_days: f64,
    pub period_verdict: String,
    pub reliability_code: String,
    pub confidence_flags: Vec<String>,
    pub insufficiency_reasons: Vec<String>,
    pub is_valid: bool,
    pub is_reliable: bool,
    pub period_lower_days: Option<f64>,
    pub period_upper_days: Option<f64>,
    pub relative_period_uncertainty: Option<f64>,
    pub alternate_period_days: Vec<f64>,
    pub fourier_period_days: Option<f64>,
    pub fourier_order: Option<i64>,
    pub fourier_sigma_threshold: Option<f64>,
    pub fourier_phase_c1: Option<f64>,
    pub fourier_phase_c2: Option<f64>,
    pub residual_sigma_mag: Option<f64>,
    pub fourier_is_valid: Option<bool>,
    pub fourier_is_reliable: Option<bool>,
    pub fourier_alternate_period_days: Vec<f64>,
    pub phase_coverage_fraction: Option<f64>,
    pub n_rotations_spanned: Option<f64>,
    pub amplitude_snr: Option<f64>,
    pub n_significant_aliases: Option<i64>,
    pub n_observations: i64,
    pub n_fit_observations: i64,
    pub n_clipped: i64,
    pub n_filters: i64,
    pub n_sessions: i64,
    pub used_session_offsets: bool,
    pub is_period_doubled: bool,
}

impl RotationPeriodEstimate {
    fn insufficient(
        reasons: Vec<String>,
        n_observations: usize,
        n_filters: usize,
        n_sessions: usize,
    ) -> Self {
        Self {
            period_days: f64::NAN,
            period_verdict: "insufficient_data".to_string(),
            reliability_code: "1".to_string(),
            confidence_flags: Vec::new(),
            insufficiency_reasons: reasons,
            is_valid: false,
            is_reliable: false,
            period_lower_days: None,
            period_upper_days: None,
            relative_period_uncertainty: None,
            alternate_period_days: Vec::new(),
            fourier_period_days: None,
            fourier_order: None,
            fourier_sigma_threshold: None,
            fourier_phase_c1: None,
            fourier_phase_c2: None,
            residual_sigma_mag: None,
            fourier_is_valid: None,
            fourier_is_reliable: None,
            fourier_alternate_period_days: Vec::new(),
            phase_coverage_fraction: None,
            n_rotations_spanned: None,
            amplitude_snr: None,
            n_significant_aliases: None,
            n_observations: n_observations as i64,
            n_fit_observations: 0,
            n_clipped: 0,
            n_filters: n_filters as i64,
            n_sessions: n_sessions as i64,
            used_session_offsets: false,
            is_period_doubled: false,
        }
    }
}

#[derive(Clone)]
struct Design {
    rows: Vec<Vec<f64>>,
    phase_c1: usize,
    phase_c2: usize,
}

#[derive(Clone, Debug)]
struct Fit {
    frequency: f64,
    order: usize,
    coeffs: Vec<f64>,
    sigma: f64,
    rss: f64,
    df: i64,
    n_fit: usize,
    n_clipped: usize,
    phase_c1: usize,
    phase_c2: usize,
}

impl Fit {
    fn n_parameters(&self) -> usize {
        self.coeffs.len()
    }
}

#[derive(Clone)]
struct PeriodFit {
    fit: Fit,
    period_days: f64,
    doubled: bool,
}

#[derive(Clone)]
struct Cluster {
    indices: Vec<usize>,
    best: PeriodFit,
    lower: f64,
    upper: f64,
}

struct Solution {
    chosen: PeriodFit,
    clusters: Vec<Cluster>,
    lower: f64,
    upper: f64,
    relative_uncertainty: f64,
    alternates: Vec<f64>,
    valid: bool,
    reliable: bool,
    amplitude: f64,
    fit: Fit,
    sigma_threshold: f64,
}

#[derive(Clone, Copy)]
struct SessionSummary {
    count: usize,
    minimum_group_count: usize,
    median_span: f64,
}

pub fn estimate_rotation_period_grouped(
    input: &RotationPeriodInput,
    object_ids: &[Option<String>],
    config: &RotationPeriodConfig,
) -> Result<Vec<(String, RotationPeriodEstimate)>, String> {
    if object_ids.len() != input.time.len() {
        return Err(format!(
            "object_ids length ({}) must match detections length ({})",
            object_ids.len(),
            input.time.len()
        ));
    }
    let mut rows: Vec<(String, usize)> = object_ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| id.clone().map(|id| (id, index)))
        .collect();
    rows.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    let mut output = Vec::new();
    let mut start = 0;
    while start < rows.len() {
        let id = rows[start].0.clone();
        let mut end = start + 1;
        while end < rows.len() && rows[end].0 == id {
            end += 1;
        }
        let indices: Vec<usize> = rows[start..end].iter().map(|row| row.1).collect();
        let subset = subset_input(input, &indices)?;
        let estimate = estimate_rotation_period(&subset, config).unwrap_or_else(|error| {
            let mut estimate = RotationPeriodEstimate::insufficient(
                vec![format!("solve_error: {error}")],
                indices.len(),
                ordered_unique(&subset.filter).len(),
                0,
            );
            estimate.confidence_flags.push("solve_error".to_string());
            estimate
        });
        output.push((id, estimate));
        start = end;
    }
    Ok(output)
}

pub fn estimate_rotation_period_best_apparition(
    input: &RotationPeriodInput,
    config: &RotationPeriodConfig,
    gap_days: f64,
) -> Result<(RotationPeriodEstimate, usize, usize), String> {
    if input.time.is_empty() {
        return estimate_rotation_period(input, config).map(|estimate| (estimate, 1, 1));
    }
    let time = input
        .time
        .rescale(TimeScale::Tdb)
        .map_err(|error| error.to_string())?
        .mjd_values();
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&left, &right| total_cmp(time[left], time[right]).then(left.cmp(&right)));
    let mut groups = vec![Vec::new()];
    for (position, &index) in order.iter().enumerate() {
        if position > 0 && time[index] - time[order[position - 1]] > gap_days {
            groups.push(Vec::new());
        }
        groups.last_mut().unwrap().push(index);
    }
    if groups.len() == 1 {
        let mut estimate = estimate_rotation_period(input, config)?;
        estimate
            .confidence_flags
            .push("apparition_selected_1_of_1".to_string());
        return Ok((estimate, 1, 1));
    }
    let total = groups.len();
    let mut candidates = Vec::with_capacity(total);
    for (group_index, indices) in groups.iter().enumerate() {
        let subset = subset_input(input, indices)?;
        let estimate = estimate_rotation_period(&subset, config).unwrap_or_else(|error| {
            let mut estimate = RotationPeriodEstimate::insufficient(
                vec![format!("solve_error: {error}")],
                indices.len(),
                ordered_unique(&subset.filter).len(),
                0,
            );
            estimate.confidence_flags.push("solve_error".to_string());
            estimate
        });
        candidates.push((group_index, estimate));
    }
    let (selected_index, mut selected) = candidates
        .into_iter()
        .max_by(compare_apparition_candidates)
        .unwrap();
    selected.confidence_flags.push(format!(
        "apparition_selected_{}_of_{total}",
        selected_index + 1
    ));
    Ok((selected, selected_index + 1, total))
}

fn compare_apparition_candidates(
    left: &(usize, RotationPeriodEstimate),
    right: &(usize, RotationPeriodEstimate),
) -> Ordering {
    fn rank(verdict: &str) -> i32 {
        match verdict {
            "single_period" => 2,
            "period_family" => 1,
            _ => 0,
        }
    }
    rank(&left.1.period_verdict)
        .cmp(&rank(&right.1.period_verdict))
        .then_with(|| {
            total_cmp(
                left.1.amplitude_snr.unwrap_or(f64::NEG_INFINITY),
                right.1.amplitude_snr.unwrap_or(f64::NEG_INFINITY),
            )
        })
        .then(left.1.n_observations.cmp(&right.1.n_observations))
        .then_with(|| right.0.cmp(&left.0))
}

fn subset_input(
    input: &RotationPeriodInput,
    indices: &[usize],
) -> Result<RotationPeriodInput, String> {
    fn take<T: Clone>(values: &[T], indices: &[usize]) -> Vec<T> {
        indices.iter().map(|&index| values[index].clone()).collect()
    }
    let epochs = take(&input.time.epochs, indices);
    Ok(RotationPeriodInput {
        time: TimeArray::new(input.time.scale, epochs).map_err(|error| error.to_string())?,
        magnitude: take(&input.magnitude, indices),
        magnitude_sigma: take(&input.magnitude_sigma, indices),
        filter: take(&input.filter, indices),
        session_id: take(&input.session_id, indices),
        r_au: take(&input.r_au, indices),
        delta_au: take(&input.delta_au, indices),
        phase_angle_deg: take(&input.phase_angle_deg, indices),
    })
}

pub fn estimate_rotation_period(
    input: &RotationPeriodInput,
    config: &RotationPeriodConfig,
) -> Result<RotationPeriodEstimate, String> {
    validate_config(config)?;
    validate_input(input)?;
    let n = input.time.len();
    let time_mjd_tdb = input
        .time
        .rescale(TimeScale::Tdb)
        .map_err(|error| error.to_string())?
        .mjd_values();
    let filters = input.filter.clone();
    let n_filters = ordered_unique(&filters).len();
    let corrected_time: Vec<f64> = time_mjd_tdb
        .iter()
        .zip(&input.delta_au)
        .map(|(&time, &delta)| time - delta / LIGHT_SPEED_AU_PER_DAY)
        .collect();
    let time_min = corrected_time.iter().copied().fold(f64::INFINITY, f64::min);
    let relative_time: Vec<f64> = corrected_time.iter().map(|time| time - time_min).collect();
    let span = relative_time
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let reduced_magnitude: Vec<f64> = input
        .magnitude
        .iter()
        .zip(&input.r_au)
        .zip(&input.delta_au)
        .map(|((&magnitude, &r), &delta)| magnitude - 5.0 * (r * delta).log10())
        .collect();
    let sessions = normalized_sessions(&input.session_id);
    let session_summary = summarize_sessions(&corrected_time, &filters, sessions.as_deref());
    let orders = normalized_orders(config)?;

    if config.early_exit_on_insufficient {
        let reasons = pre_solve_reasons(
            n,
            n_filters,
            span,
            &reduced_magnitude,
            &input.magnitude_sigma,
            orders[0],
            config,
        );
        if !reasons.is_empty() {
            return Ok(RotationPeriodEstimate::insufficient(
                reasons,
                n,
                n_filters,
                session_summary.count,
            ));
        }
    }
    if !span.is_finite() || span <= 0.0 {
        return Err("observation time span must be positive".to_string());
    }

    let frequencies = frequency_grid(span, config)?;
    let weights = weights_from_sigma(&input.magnitude_sigma);
    let base_design = build_design(&filters, None, &input.phase_angle_deg);
    let baseline = derive_solution(
        &relative_time,
        &reduced_magnitude,
        &base_design,
        &frequencies,
        weights.as_deref(),
        &orders,
        config,
    )?;
    let mut chosen = baseline;
    let mut chosen_design = base_design;
    let mut used_session_offsets = false;

    if let Some(session_labels) = sessions.as_deref() {
        if config.session_mode == "use" || config.session_mode == "auto" {
            let session_design =
                build_design(&filters, Some(session_labels), &input.phase_angle_deg);
            let session_solution = derive_solution(
                &relative_time,
                &reduced_magnitude,
                &session_design,
                &frequencies,
                weights.as_deref(),
                &orders,
                config,
            )?;
            let select_session = config.session_mode == "use"
                || (session_summary.count >= 2
                    && session_summary.minimum_group_count
                        >= config.auto_session_min_observations_per_group
                    && session_summary.median_span > 0.0
                    && bic(&session_solution.fit) + config.auto_session_bic_improvement
                        < bic(&chosen.fit));
            if select_session {
                chosen = session_solution;
                chosen_design = session_design;
                used_session_offsets = true;
            }
        }
    }

    let period = chosen.chosen.period_days;
    let amplitude_snr = finite_ratio(chosen.amplitude, chosen.fit.sigma);
    let phase_coverage = phase_coverage_fraction(&relative_time, period);
    let rotations_spanned = if period.is_finite() && period > 0.0 {
        Some(span / period)
    } else {
        None
    };
    let aliases = chosen.clusters.len();
    let enough_observations = most_populated_filter_count(&filters) >= OBSERVATION_COUNT_MIN;
    let (mut verdict, mut reliability, flags, mut reasons) = classify_confidence(
        amplitude_snr,
        phase_coverage,
        rotations_spanned,
        config.min_rotations_in_span,
        aliases,
        chosen.chosen.doubled,
        enough_observations,
        chosen.reliable,
        chosen.valid,
    );

    if verdict == "single_period" && period * 24.0 > MAX_PLAUSIBLE_SINGLE_PERIOD_HOURS {
        verdict = "period_family".to_string();
        reliability = "2".to_string();
        reasons.push("period_implausibly_long".to_string());
    }
    if verdict == "single_period"
        && period.is_finite()
        && period > 0.0
        && 0.5 / period < frequencies[0]
    {
        if let Some(half_fit) = fit_frequency(
            &relative_time,
            &reduced_magnitude,
            &chosen_design,
            0.5 / period,
            DEFAULT_PROFILE_MAX_FOURIER_ORDER.max(*orders.last().unwrap()),
            config.clip_sigma,
            weights.as_deref(),
            true,
            true,
        ) {
            if half_fit.sigma < chosen.fit.sigma {
                verdict = "period_family".to_string();
                reliability = "2".to_string();
                reasons.push("subharmonic_unresolved".to_string());
            }
        }
    }

    let mut confidence_flags = flags;
    if config.search_fidelity == "validated_staged" && frequencies.len() > 2048 {
        confidence_flags.push("staged_search_used".to_string());
    }
    if frequency_grid_was_capped(span, config) {
        confidence_flags.push("grid_capped".to_string());
    }
    let is_valid = verdict != "insufficient_data";
    let is_reliable = verdict == "single_period";

    Ok(RotationPeriodEstimate {
        period_days: period,
        period_verdict: verdict,
        reliability_code: reliability,
        confidence_flags,
        insufficiency_reasons: reasons,
        is_valid,
        is_reliable,
        period_lower_days: Some(chosen.lower),
        period_upper_days: Some(chosen.upper),
        relative_period_uncertainty: Some(chosen.relative_uncertainty),
        alternate_period_days: chosen.alternates.clone(),
        fourier_period_days: Some(period),
        fourier_order: Some(chosen.fit.order as i64),
        fourier_sigma_threshold: Some(chosen.sigma_threshold),
        fourier_phase_c1: Some(chosen.fit.coeffs[chosen.fit.phase_c1]),
        fourier_phase_c2: Some(chosen.fit.coeffs[chosen.fit.phase_c2]),
        residual_sigma_mag: Some(chosen.fit.sigma),
        fourier_is_valid: Some(chosen.valid),
        fourier_is_reliable: Some(chosen.reliable),
        fourier_alternate_period_days: chosen.alternates,
        phase_coverage_fraction: phase_coverage,
        n_rotations_spanned: rotations_spanned,
        amplitude_snr,
        n_significant_aliases: Some(aliases as i64),
        n_observations: n as i64,
        n_fit_observations: chosen.fit.n_fit as i64,
        n_clipped: chosen.fit.n_clipped as i64,
        n_filters: n_filters as i64,
        n_sessions: session_summary.count as i64,
        used_session_offsets,
        is_period_doubled: chosen.chosen.doubled,
    })
}

fn validate_config(config: &RotationPeriodConfig) -> Result<(), String> {
    if config.clip_sigma <= 0.0 {
        return Err("clip_sigma must be positive".to_string());
    }
    if config.min_rotations_in_span <= 0.0 {
        return Err("min_rotations_in_span must be positive".to_string());
    }
    if config.max_frequency_cycles_per_day <= 0.0 {
        return Err("max_frequency_cycles_per_day must be positive".to_string());
    }
    if config.frequency_grid_scale <= 0.0 {
        return Err("frequency_grid_scale must be positive".to_string());
    }
    if config
        .max_search_period_hours
        .is_some_and(|value| value <= 0.0)
    {
        return Err("max_search_period_hours must be positive when set".to_string());
    }
    if config.search_fidelity != "validated_staged" && config.search_fidelity != "exact_grid" {
        return Err(
            "search_fidelity must be one of {'validated_staged', 'exact_grid'}".to_string(),
        );
    }
    if config.session_mode != "ignore"
        && config.session_mode != "use"
        && config.session_mode != "auto"
    {
        return Err("session_mode must be one of {'ignore', 'use', 'auto'}".to_string());
    }
    if config.auto_session_min_observations_per_group == 0 {
        return Err("auto_session_min_observations_per_group must be positive".to_string());
    }
    if config.auto_session_bic_improvement < 0.0 {
        return Err("auto_session_bic_improvement must be non-negative".to_string());
    }
    Ok(())
}

fn validate_input(input: &RotationPeriodInput) -> Result<(), String> {
    let n = input.time.len();
    if n == 0 {
        return Err("observations must be non-empty".to_string());
    }
    let lengths = [
        input.magnitude.len(),
        input.magnitude_sigma.len(),
        input.filter.len(),
        input.session_id.len(),
        input.r_au.len(),
        input.delta_au.len(),
        input.phase_angle_deg.len(),
    ];
    if lengths.iter().any(|&length| length != n) {
        return Err("all observation columns must be 1D and aligned".to_string());
    }
    if input
        .time
        .mjd_values()
        .iter()
        .chain(&input.magnitude)
        .chain(&input.r_au)
        .chain(&input.delta_au)
        .chain(&input.phase_angle_deg)
        .any(|value| !value.is_finite())
    {
        return Err(
            "observations must contain finite time, mag, r_au, delta_au, and phase_angle_deg"
                .to_string(),
        );
    }
    if input.r_au.iter().any(|&value| value <= 0.0)
        || input.delta_au.iter().any(|&value| value <= 0.0)
    {
        return Err("r_au and delta_au must be positive".to_string());
    }
    Ok(())
}

fn normalized_orders(config: &RotationPeriodConfig) -> Result<Vec<usize>, String> {
    let mut orders = config
        .fourier_orders
        .clone()
        .unwrap_or_else(|| vec![2, 3, 4, 5, 6]);
    orders.sort_unstable();
    orders.dedup();
    if orders.is_empty() {
        return Err("fourier order set must be non-empty".to_string());
    }
    if orders[0] == 0 {
        return Err("fourier orders must be positive".to_string());
    }
    Ok(orders)
}

fn ordered_unique(values: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut output = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            output.push(value.clone());
        }
    }
    output
}

fn normalized_sessions(values: &[Option<String>]) -> Option<Vec<String>> {
    if !values.iter().any(Option::is_some) {
        return None;
    }
    Some(
        values
            .iter()
            .map(|value| {
                value
                    .clone()
                    .unwrap_or_else(|| "__missing_session__".to_string())
            })
            .collect(),
    )
}

fn build_design(filters: &[String], sessions: Option<&[String]>, phase: &[f64]) -> Design {
    let unique_filters = ordered_unique(filters);
    let n = filters.len();
    let mut columns = vec![vec![1.0; n]];
    for label in unique_filters.iter().skip(1) {
        columns.push(
            filters
                .iter()
                .map(|value| if value == label { 1.0 } else { 0.0 })
                .collect(),
        );
    }
    if let Some(sessions) = sessions {
        for filter in &unique_filters {
            let labels: Vec<String> = filters
                .iter()
                .zip(sessions)
                .filter_map(|(candidate, session)| (candidate == filter).then_some(session.clone()))
                .collect();
            for session in ordered_unique(&labels).iter().skip(1) {
                columns.push(
                    filters
                        .iter()
                        .zip(sessions)
                        .map(|(candidate_filter, candidate_session)| {
                            if candidate_filter == filter && candidate_session == session {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect(),
                );
            }
        }
    }
    columns.push(phase.to_vec());
    columns.push(phase.iter().map(|value| value * value).collect());
    let rows = (0..n)
        .map(|row| columns.iter().map(|column| column[row]).collect())
        .collect::<Vec<Vec<f64>>>();
    let width = columns.len();
    Design {
        rows,
        phase_c1: width - 2,
        phase_c2: width - 1,
    }
}

fn summarize_sessions(
    time: &[f64],
    filters: &[String],
    sessions: Option<&[String]>,
) -> SessionSummary {
    let Some(sessions) = sessions else {
        return SessionSummary {
            count: 0,
            minimum_group_count: 0,
            median_span: 0.0,
        };
    };
    let unique_sessions = ordered_unique(sessions);
    let unique_filters = ordered_unique(filters);
    let mut group_counts = Vec::new();
    let mut spans = Vec::new();
    for session in &unique_sessions {
        let session_times: Vec<f64> = sessions
            .iter()
            .zip(time)
            .filter_map(|(candidate, &value)| (candidate == session).then_some(value))
            .collect();
        if !session_times.is_empty() {
            let minimum = session_times.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = session_times
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            spans.push(maximum - minimum);
        }
        for filter in &unique_filters {
            let count = sessions
                .iter()
                .zip(filters)
                .filter(|(candidate_session, candidate_filter)| {
                    *candidate_session == session && *candidate_filter == filter
                })
                .count();
            if count > 0 {
                group_counts.push(count);
            }
        }
    }
    SessionSummary {
        count: unique_sessions.len(),
        minimum_group_count: group_counts.into_iter().min().unwrap_or(0),
        median_span: median(&spans).unwrap_or(0.0),
    }
}

fn pre_solve_reasons(
    n: usize,
    n_filters: usize,
    span: f64,
    reduced_magnitude: &[f64],
    magnitude_sigma: &[f64],
    minimum_order: usize,
    config: &RotationPeriodConfig,
) -> Vec<String> {
    let mut reasons = Vec::new();
    let baseline_parameters = n_filters + 2 * minimum_order + 2;
    if n < MIN_OBSERVATIONS || n < baseline_parameters + PRE_SOLVE_OBSERVATION_MARGIN {
        reasons.push("too_few_observations".to_string());
    }
    let minimum_frequency = if span.is_finite() && span > 0.0 {
        config.min_rotations_in_span / span
    } else {
        f64::INFINITY
    };
    if !span.is_finite() || span <= 0.0 || minimum_frequency >= config.max_frequency_cycles_per_day
    {
        reasons.push("insufficient_time_span".to_string());
    }
    if reduced_magnitude.iter().all(|value| value.is_finite()) {
        let center = median(reduced_magnitude).unwrap_or(f64::NAN);
        let deviations: Vec<f64> = reduced_magnitude
            .iter()
            .map(|value| (value - center).abs())
            .collect();
        let scatter = 1.4826 * median(&deviations).unwrap_or(f64::NAN);
        let finite_sigma: Vec<f64> = magnitude_sigma
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        let noise = median(&finite_sigma).unwrap_or(PRE_SOLVE_SCATTER_FLOOR);
        if scatter < PRE_SOLVE_AMPLITUDE_SNR_FLOOR * noise {
            reasons.push("amplitude_below_noise".to_string());
        }
    }
    reasons
}

fn weights_from_sigma(sigma: &[f64]) -> Option<Vec<f64>> {
    if sigma.is_empty()
        || sigma
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        None
    } else {
        Some(sigma.iter().map(|value| 1.0 / (value * value)).collect())
    }
}

fn frequency_bounds(span: f64, config: &RotationPeriodConfig) -> Result<(f64, f64), String> {
    let mut minimum = config.min_rotations_in_span / span;
    if let Some(maximum_period) = config.max_search_period_hours {
        minimum = minimum.max(24.0 / maximum_period);
    }
    if !minimum.is_finite() || minimum <= 0.0 {
        return Err("derived minimum frequency is invalid".to_string());
    }
    if config.max_frequency_cycles_per_day <= minimum {
        return Err(
            "max_frequency_cycles_per_day must exceed the minimum searchable frequency".to_string(),
        );
    }
    Ok((minimum, config.max_frequency_cycles_per_day))
}

fn requested_frequency_count(span: f64, config: &RotationPeriodConfig) -> Result<usize, String> {
    let (minimum, maximum) = frequency_bounds(span, config)?;
    Ok(((config.frequency_grid_scale * span * (maximum - minimum)).ceil() as usize + 1).max(2))
}

fn frequency_grid(span: f64, config: &RotationPeriodConfig) -> Result<Vec<f64>, String> {
    let (minimum, maximum) = frequency_bounds(span, config)?;
    let count = requested_frequency_count(span, config)?.min(MAX_FREQUENCIES);
    let step = (maximum - minimum) / (count - 1) as f64;
    Ok((0..count)
        .map(|index| minimum + index as f64 * step)
        .collect())
}

fn frequency_grid_was_capped(span: f64, config: &RotationPeriodConfig) -> bool {
    requested_frequency_count(span, config).is_ok_and(|count| count > MAX_FREQUENCIES)
}

fn derive_solution(
    time: &[f64],
    target: &[f64],
    design: &Design,
    frequencies: &[f64],
    weights: Option<&[f64]>,
    orders: &[usize],
    config: &RotationPeriodConfig,
) -> Result<Solution, String> {
    let mut candidates = Vec::new();
    for &order in orders {
        if let Some(fit) =
            search_best_fit(time, target, design, frequencies, weights, order, config)
        {
            candidates.push(fit);
        }
    }
    if candidates.is_empty() {
        return Err("no valid rotation-period fit could be found".to_string());
    }
    candidates.sort_by_key(|fit| fit.order);
    let selected_order = select_order(&candidates, 0.90).order;
    let fits: Vec<Option<Fit>> = frequencies
        .par_iter()
        .map(|&frequency| {
            fit_frequency(
                time,
                target,
                design,
                frequency,
                selected_order,
                config.clip_sigma,
                weights,
                true,
                false,
            )
        })
        .collect();
    let best_index = fits
        .iter()
        .enumerate()
        .filter_map(|(index, fit)| fit.as_ref().map(|fit| (index, fit.sigma)))
        .min_by(|left, right| total_cmp(left.1, right.1))
        .map(|item| item.0)
        .ok_or_else(|| "failed to evaluate Fourier sigma curve".to_string())?;
    let best = fits[best_index].clone().unwrap();
    let chosen = fit_with_period(best.clone());
    let sigma_threshold = sigma_threshold(&best, 0.95);
    let mut accepted: Vec<usize> = fits
        .iter()
        .enumerate()
        .filter_map(|(index, fit)| {
            fit.as_ref()
                .filter(|fit| fit.sigma.is_finite() && fit.sigma <= sigma_threshold)
                .map(|_| index)
        })
        .collect();
    if accepted.is_empty() {
        accepted.push(best_index);
    }
    let mut clusters = cluster_fits(&fits, &accepted, frequencies);
    if clusters.is_empty() {
        clusters.push(Cluster {
            indices: vec![best_index],
            best: chosen.clone(),
            lower: chosen.period_days,
            upper: chosen.period_days,
        });
    }
    clusters.sort_by(|left, right| total_cmp(left.best.fit.sigma, right.best.fit.sigma));
    let primary_index = clusters
        .iter()
        .position(|cluster| cluster.indices.contains(&best_index))
        .unwrap_or(0);
    let primary = clusters[primary_index].clone();
    let uncertainty = ((chosen.period_days - primary.lower)
        .abs()
        .max((primary.upper - chosen.period_days).abs()))
        / chosen.period_days;
    let alternates = clusters
        .iter()
        .enumerate()
        .filter_map(|(index, cluster)| (index != primary_index).then_some(cluster.best.period_days))
        .collect::<Vec<_>>();
    let maximum_deviation = clusters
        .iter()
        .map(|cluster| {
            (cluster.lower - chosen.period_days)
                .abs()
                .max((cluster.upper - chosen.period_days).abs())
        })
        .fold(0.0, f64::max);
    let reliable = maximum_deviation <= (2.0 * chosen.period_days).max(7.0 / 24.0);
    let amplitude = periodic_amplitude(&best, 4096);
    Ok(Solution {
        chosen,
        clusters,
        lower: primary.lower,
        upper: primary.upper,
        relative_uncertainty: uncertainty,
        alternates,
        valid: true,
        reliable,
        amplitude,
        fit: best,
        sigma_threshold,
    })
}

fn search_best_fit(
    time: &[f64],
    target: &[f64],
    design: &Design,
    frequencies: &[f64],
    weights: Option<&[f64]>,
    order: usize,
    config: &RotationPeriodConfig,
) -> Option<Fit> {
    if config.search_fidelity == "exact_grid" || frequencies.len() <= 2048 {
        return frequencies
            .par_iter()
            .filter_map(|&frequency| {
                fit_frequency(
                    time,
                    target,
                    design,
                    frequency,
                    order,
                    config.clip_sigma,
                    weights,
                    true,
                    false,
                )
            })
            .min_by(|left, right| total_cmp(left.sigma, right.sigma));
    }
    let stride = ((frequencies.len() as f64 / 1024.0).ceil() as usize).max(1);
    let mut coarse_indices: Vec<usize> = (0..frequencies.len()).step_by(stride).collect();
    if coarse_indices.last().copied() != Some(frequencies.len() - 1) {
        coarse_indices.push(frequencies.len() - 1);
    }
    let coarse: Vec<(usize, f64)> = coarse_indices
        .par_iter()
        .map(|&index| {
            let score = fit_frequency(
                time,
                target,
                design,
                frequencies[index],
                order,
                config.clip_sigma,
                weights,
                false,
                false,
            )
            .map_or(f64::NAN, |fit| fit.sigma);
            (index, score)
        })
        .collect();
    let mut local_minima = Vec::new();
    for position in 0..coarse.len() {
        let score = coarse[position].1;
        if !score.is_finite() {
            continue;
        }
        let left = position
            .checked_sub(1)
            .map_or(f64::INFINITY, |index| coarse[index].1);
        let right = coarse
            .get(position + 1)
            .map_or(f64::INFINITY, |item| item.1);
        if score <= left && score <= right {
            local_minima.push(coarse[position]);
        }
    }
    if local_minima.is_empty() {
        local_minima.extend(coarse.iter().copied().filter(|item| item.1.is_finite()));
    }
    local_minima.sort_by(|left, right| total_cmp(left.1, right.1));
    let mut centers = Vec::new();
    for (index, _) in local_minima {
        if centers
            .iter()
            .all(|&existing: &usize| existing.abs_diff(index) >= stride)
        {
            centers.push(index);
        }
        if centers.len() == 12 {
            break;
        }
    }
    if let Some(global_best) = coarse
        .iter()
        .filter(|item| item.1.is_finite())
        .min_by(|left, right| total_cmp(left.1, right.1))
    {
        centers.push(global_best.0);
    }
    let radius = (4 * stride).max(8);
    let mut exact_indices = BTreeSet::new();
    for center in centers {
        let lower = center.saturating_sub(radius);
        let upper = (center + radius).min(frequencies.len() - 1);
        exact_indices.extend(lower..=upper);
    }
    exact_indices
        .into_iter()
        .collect::<Vec<_>>()
        .par_iter()
        .filter_map(|&index| {
            fit_frequency(
                time,
                target,
                design,
                frequencies[index],
                order,
                config.clip_sigma,
                weights,
                true,
                false,
            )
        })
        .min_by(|left, right| total_cmp(left.sigma, right.sigma))
}

#[allow(clippy::too_many_arguments)]
fn fit_frequency(
    time: &[f64],
    target: &[f64],
    design: &Design,
    frequency: f64,
    order: usize,
    clip_sigma: f64,
    weights: Option<&[f64]>,
    clipped: bool,
    robust_solve: bool,
) -> Option<Fit> {
    let n = target.len();
    let n_parameters = design.rows[0].len() + 2 * order;
    if n <= n_parameters {
        return None;
    }
    let mut mask = vec![true; n];
    let iterations = if clipped { MAX_CLIP_ITERATIONS } else { 1 };
    for _ in 0..iterations {
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(index, &keep)| keep.then_some(index))
            .collect();
        if indices.len() <= n_parameters {
            return None;
        }
        let (coeffs, residuals, sigma, rss, df) = solve_fit(
            time,
            target,
            design,
            frequency,
            order,
            weights,
            &indices,
            robust_solve,
        )?;
        if !clipped {
            return Some(Fit {
                frequency,
                order,
                coeffs,
                sigma,
                rss,
                df,
                n_fit: indices.len(),
                n_clipped: n - indices.len(),
                phase_c1: design.phase_c1,
                phase_c2: design.phase_c2,
            });
        }
        let keep: Vec<bool> = residuals
            .iter()
            .map(|residual| residual.abs() <= clip_sigma * sigma)
            .collect();
        if keep.iter().all(|&value| value) {
            return Some(Fit {
                frequency,
                order,
                coeffs,
                sigma,
                rss,
                df,
                n_fit: indices.len(),
                n_clipped: n - indices.len(),
                phase_c1: design.phase_c1,
                phase_c2: design.phase_c2,
            });
        }
        for (&index, &keep) in indices.iter().zip(&keep) {
            if !keep {
                mask[index] = false;
            }
        }
    }
    let indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(index, &keep)| keep.then_some(index))
        .collect();
    if indices.len() <= n_parameters {
        return None;
    }
    let (coeffs, _, sigma, rss, df) = solve_fit(
        time,
        target,
        design,
        frequency,
        order,
        weights,
        &indices,
        robust_solve,
    )?;
    Some(Fit {
        frequency,
        order,
        coeffs,
        sigma,
        rss,
        df,
        n_fit: indices.len(),
        n_clipped: n - indices.len(),
        phase_c1: design.phase_c1,
        phase_c2: design.phase_c2,
    })
}

type FitSolve = (Vec<f64>, Vec<f64>, f64, f64, i64);

#[allow(clippy::too_many_arguments)]
fn solve_fit(
    time: &[f64],
    target: &[f64],
    design: &Design,
    frequency: f64,
    order: usize,
    weights: Option<&[f64]>,
    indices: &[usize],
    robust_solve: bool,
) -> Option<FitSolve> {
    let width = design.rows[0].len() + 2 * order;
    let mut normal = vec![0.0; width * width];
    let mut rhs = vec![0.0; width];
    let mut row = vec![0.0; width];
    for &index in indices {
        row[..design.rows[index].len()].copy_from_slice(&design.rows[index]);
        let omega = TAU * frequency * time[index];
        let offset = design.rows[index].len();
        for harmonic in 1..=order {
            let angle = harmonic as f64 * omega;
            let (sin, cos) = angle.sin_cos();
            row[offset + 2 * (harmonic - 1)] = cos;
            row[offset + 2 * (harmonic - 1) + 1] = sin;
        }
        let weight = weights.map_or(1.0, |weights| weights[index]);
        add_normal_row(&mut normal, &mut rhs, &row, target[index], weight);
    }
    let (prior_target, prior_weights) = phase_prior(design, width);
    let mut prior = vec![0.0; width];
    prior[design.phase_c1] = 1.0;
    add_normal_row(
        &mut normal,
        &mut rhs,
        &prior,
        prior_target[0],
        prior_weights[0],
    );
    prior.fill(0.0);
    prior[design.phase_c2] = 1.0;
    add_normal_row(
        &mut normal,
        &mut rhs,
        &prior,
        prior_target[1],
        prior_weights[1],
    );
    let coeffs = if robust_solve {
        solve_design_svd(
            time,
            target,
            design,
            frequency,
            order,
            weights,
            indices,
            &prior_target,
            &prior_weights,
        )?
    } else {
        solve_linear_system(normal, rhs, width)?
    };
    let mut residuals = Vec::with_capacity(indices.len());
    let mut rss = 0.0;
    let mut weight_sum = 0.0;
    for &index in indices {
        row[..design.rows[index].len()].copy_from_slice(&design.rows[index]);
        let omega = TAU * frequency * time[index];
        let offset = design.rows[index].len();
        for harmonic in 1..=order {
            let angle = harmonic as f64 * omega;
            let (sin, cos) = angle.sin_cos();
            row[offset + 2 * (harmonic - 1)] = cos;
            row[offset + 2 * (harmonic - 1) + 1] = sin;
        }
        let predicted = row
            .iter()
            .zip(&coeffs)
            .map(|(x, beta)| x * beta)
            .sum::<f64>();
        let residual = target[index] - predicted;
        let weight = weights.map_or(1.0, |weights| weights[index]);
        residuals.push(residual);
        rss += weight * residual * residual;
        weight_sum += weight;
    }
    let df = indices.len() as i64 - width as i64;
    if df <= 0 || weight_sum <= 0.0 {
        return None;
    }
    let sigma_squared = if weights.is_some() {
        target.len() as f64 / weight_sum * rss / df as f64
    } else {
        rss / df as f64
    };
    let sigma = sigma_squared.max(0.0).sqrt();
    sigma
        .is_finite()
        .then_some((coeffs, residuals, sigma, rss, df))
}

fn add_normal_row(normal: &mut [f64], rhs: &mut [f64], row: &[f64], target: f64, weight: f64) {
    let width = row.len();
    for i in 0..width {
        rhs[i] += weight * row[i] * target;
        for j in 0..=i {
            let value = weight * row[i] * row[j];
            normal[i * width + j] += value;
            if i != j {
                normal[j * width + i] += value;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_design_svd(
    time: &[f64],
    target: &[f64],
    design: &Design,
    frequency: f64,
    order: usize,
    weights: Option<&[f64]>,
    indices: &[usize],
    prior_target: &[f64; 2],
    prior_weights: &[f64; 2],
) -> Option<Vec<f64>> {
    let fixed_width = design.rows[0].len();
    let width = fixed_width + 2 * order;
    let rows = indices.len() + 2;
    let mut matrix = vec![0.0; rows * width];
    let mut response = vec![0.0; rows];
    for (output_row, &input_row) in indices.iter().enumerate() {
        let scale = weights.map_or(1.0, |weights| weights[input_row]).sqrt();
        let destination = &mut matrix[output_row * width..(output_row + 1) * width];
        for (value, &fixed) in destination[..fixed_width]
            .iter_mut()
            .zip(&design.rows[input_row])
        {
            *value = fixed * scale;
        }
        let omega = TAU * frequency * time[input_row];
        for harmonic in 1..=order {
            let angle = harmonic as f64 * omega;
            let (sin, cos) = angle.sin_cos();
            let offset = fixed_width + 2 * (harmonic - 1);
            destination[offset] = cos * scale;
            destination[offset + 1] = sin * scale;
        }
        response[output_row] = target[input_row] * scale;
    }
    for prior_index in 0..2 {
        let output_row = indices.len() + prior_index;
        let scale = prior_weights[prior_index].sqrt();
        let coefficient = if prior_index == 0 {
            design.phase_c1
        } else {
            design.phase_c2
        };
        matrix[output_row * width + coefficient] = scale;
        response[output_row] = prior_target[prior_index] * scale;
    }
    let matrix = DMatrix::from_row_slice(rows, width, &matrix);
    let response = DVector::from_column_slice(&response);
    let svd = matrix.svd(true, true);
    // Match NumPy/LAPACK's rcond=None cutoff: machine epsilon times the
    // larger matrix dimension times the leading singular value.
    let largest_singular = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = f64::EPSILON * rows.max(width) as f64 * largest_singular;
    let solution = svd.solve(&response, cutoff).ok()?;
    let output = solution.as_slice().to_vec();
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(output)
}

fn solve_linear_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for pivot in 0..n {
        let best = (pivot..n).max_by(|&left, &right| {
            total_cmp(
                matrix[left * n + pivot].abs(),
                matrix[right * n + pivot].abs(),
            )
        })?;
        if matrix[best * n + pivot].abs() <= 1.0e-14 {
            return None;
        }
        if best != pivot {
            for column in 0..n {
                matrix.swap(pivot * n + column, best * n + column);
            }
            rhs.swap(pivot, best);
        }
        let diagonal = matrix[pivot * n + pivot];
        for row in (pivot + 1)..n {
            let factor = matrix[row * n + pivot] / diagonal;
            if factor == 0.0 {
                continue;
            }
            matrix[row * n + pivot] = 0.0;
            for column in (pivot + 1)..n {
                matrix[row * n + column] -= factor * matrix[pivot * n + column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut result = vec![0.0; n];
    for row in (0..n).rev() {
        let remainder = ((row + 1)..n)
            .map(|column| matrix[row * n + column] * result[column])
            .sum::<f64>();
        result[row] = (rhs[row] - remainder) / matrix[row * n + row];
    }
    result
        .iter()
        .all(|value| value.is_finite())
        .then_some(result)
}

fn phase_prior(design: &Design, width: usize) -> ([f64; 2], [f64; 2]) {
    let minimum_phase = design
        .rows
        .iter()
        .map(|row| row[design.phase_c1])
        .fold(f64::INFINITY, f64::min);
    let center = minimum_phase.clamp(1.0, 120.0);
    let start = (center - 0.5).max(1.0);
    let end = (center + 0.5).min(120.0);
    let mut x = Vec::with_capacity(25);
    let mut low = Vec::with_capacity(25);
    let mut high = Vec::with_capacity(25);
    for index in 0..25 {
        let alpha = start + index as f64 * (end - start) / 24.0;
        x.push(alpha);
        low.push(hg_phase_correction(alpha, -0.25));
        high.push(hg_phase_correction(alpha, 0.95));
    }
    let low_coeffs = quadratic_fit(&x, &low);
    let high_coeffs = quadratic_fit(&x, &high);
    let c1_min = low_coeffs[1].min(high_coeffs[1]);
    let c1_max = low_coeffs[1].max(high_coeffs[1]);
    let c2_min = low_coeffs[2].min(high_coeffs[2]);
    let c2_max = low_coeffs[2].max(high_coeffs[2]);
    let _ = width;
    let sigma1 = (0.5 * (c1_max - c1_min).abs()).max(PRIOR_SIGMA_FLOOR);
    let sigma2 = (0.5 * (c2_max - c2_min).abs()).max(PRIOR_SIGMA_FLOOR);
    (
        [0.5 * (c1_min + c1_max), 0.5 * (c2_min + c2_max)],
        [1.0 / (sigma1 * sigma1), 1.0 / (sigma2 * sigma2)],
    )
}

fn hg_phase_correction(alpha_degrees: f64, slope: f64) -> f64 {
    let tangent = (0.5 * alpha_degrees.to_radians()).tan();
    let phi1 = (-3.33 * tangent.powf(0.63)).exp();
    let phi2 = (-1.87 * tangent.powf(1.22)).exp();
    let phase = ((1.0 - slope) * phi1 + slope * phi2).max(1.0e-12);
    -2.5 * phase.log10()
}

fn quadratic_fit(x: &[f64], y: &[f64]) -> [f64; 3] {
    let mut normal = vec![0.0; 9];
    let mut rhs = vec![0.0; 3];
    for (&x, &y) in x.iter().zip(y) {
        let row = [1.0, x, x * x];
        add_normal_row(&mut normal, &mut rhs, &row, y, 1.0);
    }
    let coeffs = solve_linear_system(normal, rhs, 3).unwrap_or_else(|| vec![0.0; 3]);
    [coeffs[0], coeffs[1], coeffs[2]]
}

fn select_order(fits: &[Fit], required_confidence: f64) -> &Fit {
    for (index, candidate) in fits.iter().enumerate() {
        let significantly_worse = fits[(index + 1)..]
            .iter()
            .any(|higher| f_test_confidence(candidate, higher) > required_confidence);
        if !significantly_worse {
            return candidate;
        }
    }
    fits.last().unwrap()
}

fn f_test_confidence(smaller: &Fit, larger: &Fit) -> f64 {
    if larger.n_parameters() <= smaller.n_parameters()
        || smaller.df <= 0
        || larger.df <= 0
        || !smaller.sigma.is_finite()
        || !larger.sigma.is_finite()
    {
        return 0.0;
    }
    let ratio = smaller.sigma * smaller.sigma / (larger.sigma * larger.sigma);
    if !ratio.is_finite() || ratio <= 1.0 {
        return 0.0;
    }
    FisherSnedecor::new(smaller.df as f64, larger.df as f64)
        .map_or(0.0, |distribution| distribution.cdf(ratio))
}

fn sigma_threshold(fit: &Fit, confidence: f64) -> f64 {
    if fit.df <= 0 {
        return f64::INFINITY;
    }
    FisherSnedecor::new(fit.df as f64, fit.df as f64)
        .map(|distribution| fit.sigma * distribution.inverse_cdf(confidence).sqrt())
        .unwrap_or(f64::INFINITY)
}

fn bic(fit: &Fit) -> f64 {
    if fit.n_fit <= fit.n_parameters() || fit.rss <= 0.0 {
        f64::INFINITY
    } else {
        let n = fit.n_fit as f64;
        n * (fit.rss / n).ln() + fit.n_parameters() as f64 * n.ln()
    }
}

fn fit_with_period(fit: Fit) -> PeriodFit {
    let doubled = local_maxima_count(&fit, 2048) == 1;
    let period_days = if doubled { 2.0 } else { 1.0 } / fit.frequency;
    PeriodFit {
        fit,
        period_days,
        doubled,
    }
}

fn periodic_values(fit: &Fit, points: usize) -> Vec<f64> {
    let start = fit.coeffs.len() - 2 * fit.order;
    (0..points)
        .map(|index| {
            let phase = index as f64 / points as f64;
            (1..=fit.order)
                .map(|harmonic| {
                    let angle = TAU * harmonic as f64 * phase;
                    let (sin, cos) = angle.sin_cos();
                    let offset = start + 2 * (harmonic - 1);
                    fit.coeffs[offset] * cos + fit.coeffs[offset + 1] * sin
                })
                .sum()
        })
        .collect()
}

fn local_maxima_count(fit: &Fit, points: usize) -> usize {
    let values = periodic_values(fit, points);
    (0..points)
        .filter(|&index| {
            values[index] > values[(index + points - 1) % points]
                && values[index] >= values[(index + 1) % points]
        })
        .count()
}

fn periodic_amplitude(fit: &Fit, points: usize) -> f64 {
    let values = periodic_values(fit, points);
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum - minimum
}

fn cluster_fits(fits: &[Option<Fit>], accepted: &[usize], frequencies: &[f64]) -> Vec<Cluster> {
    let mut items = Vec::new();
    for &index in accepted {
        let Some(fit) = fits[index].clone() else {
            continue;
        };
        let period_fit = fit_with_period(fit);
        let center = frequencies[index];
        let left = if index > 0 {
            frequencies[index - 1]
        } else {
            center
        };
        let right = frequencies.get(index + 1).copied().unwrap_or(center);
        let half_step = 0.5 * (center - left).abs().max((right - center).abs());
        let factor = if period_fit.doubled { 2.0 } else { 1.0 };
        let lower = factor / (center + half_step);
        let upper = factor / (center - half_step).max(f64::EPSILON);
        items.push((index, period_fit, lower.min(upper), lower.max(upper)));
    }
    items.sort_by(|left, right| {
        total_cmp(left.2, right.2)
            .then_with(|| total_cmp(left.3, right.3))
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut groups: Vec<Vec<(usize, PeriodFit, f64, f64)>> = Vec::new();
    for item in items {
        if let Some(previous) = groups.last_mut() {
            let previous_upper = previous
                .iter()
                .map(|entry| entry.3)
                .fold(f64::NEG_INFINITY, f64::max);
            if item.2 <= previous_upper + 1.0e-12 {
                previous.push(item);
                continue;
            }
        }
        groups.push(vec![item]);
    }
    groups
        .into_iter()
        .map(|group| {
            let best = group
                .iter()
                .map(|item| item.1.clone())
                .min_by(|left, right| total_cmp(left.fit.sigma, right.fit.sigma))
                .unwrap();
            Cluster {
                indices: group.iter().map(|item| item.0).collect(),
                best,
                lower: group
                    .iter()
                    .map(|item| item.2)
                    .fold(f64::INFINITY, f64::min),
                upper: group
                    .iter()
                    .map(|item| item.3)
                    .fold(f64::NEG_INFINITY, f64::max),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn classify_confidence(
    amplitude_snr: Option<f64>,
    phase_coverage: Option<f64>,
    rotations_spanned: Option<f64>,
    minimum_rotations: f64,
    aliases: usize,
    doubled: bool,
    enough_observations: bool,
    reliable: bool,
    valid: bool,
) -> (String, String, Vec<String>, Vec<String>) {
    let mut flags = Vec::new();
    let mut reasons = Vec::new();
    if below(amplitude_snr, AMPLITUDE_SNR_MIN) {
        reasons.push("amplitude_below_noise".to_string());
    }
    if below(phase_coverage, PHASE_COVERAGE_MIN_FAMILY) {
        reasons.push("phase_coverage_low".to_string());
    }
    if below(rotations_spanned, minimum_rotations) {
        reasons.push("spans_too_few_rotations".to_string());
    }
    if !enough_observations {
        reasons.push("too_few_observations".to_string());
    }
    if !reasons.is_empty() {
        return (
            "insufficient_data".to_string(),
            "1".to_string(),
            flags,
            reasons,
        );
    }
    let good_coverage = phase_coverage.is_some_and(|value| value >= PHASE_COVERAGE_MIN_SINGLE);
    if good_coverage {
        flags.push("good_phase_coverage".to_string());
    }
    if rotations_spanned.is_some_and(|value| value >= 2.0 * minimum_rotations) {
        flags.push("multi_night".to_string());
    }
    if !doubled {
        flags.push("two_max_two_min".to_string());
    }
    let alias_ambiguous = aliases >= 2;
    if alias_ambiguous {
        reasons.push("conflicting_aliases".to_string());
    }
    if doubled {
        reasons.push("single_max_alias".to_string());
    }
    if !good_coverage && !alias_ambiguous && !doubled {
        reasons.push("phase_coverage_low".to_string());
    }
    let eligible_single = !alias_ambiguous && !doubled && good_coverage;
    let verdict = if eligible_single && reliable {
        "single_period"
    } else {
        if !valid {
            reasons.push("no_precision".to_string());
        }
        "period_family"
    };
    let reliability = if verdict == "single_period" { "3" } else { "2" };
    (verdict.to_string(), reliability.to_string(), flags, reasons)
}

fn below(value: Option<f64>, threshold: f64) -> bool {
    value.is_none_or(|value| !value.is_finite() || value < threshold)
}

fn finite_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (numerator.is_finite() && denominator.is_finite() && denominator > 0.0)
        .then_some(numerator / denominator)
}

fn phase_coverage_fraction(time: &[f64], period: f64) -> Option<f64> {
    if !period.is_finite() || period <= 0.0 || time.is_empty() {
        return None;
    }
    let mut occupied = [false; PHASE_BINS];
    for &time in time {
        let phase = (time / period).rem_euclid(1.0);
        let index = ((phase * PHASE_BINS as f64).floor() as usize).min(PHASE_BINS - 1);
        occupied[index] = true;
    }
    Some(occupied.iter().filter(|&&value| value).count() as f64 / PHASE_BINS as f64)
}

fn most_populated_filter_count(filters: &[String]) -> usize {
    let mut counts = HashMap::new();
    for filter in filters {
        *counts.entry(filter).or_insert(0_usize) += 1;
    }
    counts.values().copied().max().unwrap_or(0)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| total_cmp(*left, *right));
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Some(0.5 * (sorted[middle - 1] + sorted[middle]))
    } else {
        Some(sorted[middle])
    }
}

fn total_cmp(left: f64, right: f64) -> Ordering {
    left.total_cmp(&right)
}

pub const HARMONIC_FACTORS: [f64; 11] = [
    0.25,
    1.0 / 3.0,
    0.5,
    2.0 / 3.0,
    0.75,
    1.0,
    4.0 / 3.0,
    1.5,
    2.0,
    3.0,
    4.0,
];

pub fn relative_error_pct(recovered: f64, truth: f64) -> f64 {
    100.0 * (recovered - truth).abs() / truth
}

pub fn harmonic_adjusted_error_pct(recovered: f64, truth: f64) -> (f64, f64) {
    HARMONIC_FACTORS
        .iter()
        .map(|&factor| (relative_error_pct(recovered * factor, truth), factor))
        .min_by(|left, right| total_cmp(left.0, right.0))
        .unwrap()
}

pub fn alias_bucket(factor: f64) -> &'static str {
    const LABELS: [(f64, &str); 11] = [
        (1.0, "1x"),
        (0.25, "1/4x"),
        (1.0 / 3.0, "1/3x"),
        (0.5, "1/2x"),
        (2.0 / 3.0, "2/3x"),
        (0.75, "3/4x"),
        (4.0 / 3.0, "4/3x"),
        (1.5, "3/2x"),
        (2.0, "2x"),
        (3.0, "3x"),
        (4.0, "4x"),
    ];
    LABELS
        .iter()
        .find_map(|&(value, label)| ((factor - value).abs() <= 0.05 * value).then_some(label))
        .unwrap_or("other")
}

pub fn within_tolerance(recovered: f64, truth: f64, tolerance_fraction: f64) -> bool {
    relative_error_pct(recovered, truth) <= tolerance_fraction * 100.0
}

pub fn near_day_alias(recovered_hours: f64, truth_hours: f64, tolerance_fraction: f64) -> bool {
    if recovered_hours <= 0.0 || truth_hours <= 0.0 {
        return false;
    }
    let recovered_frequency = 24.0 / recovered_hours;
    let truth_frequency = 24.0 / truth_hours;
    for offset in [1.0, 2.0] {
        for aliased in [truth_frequency + offset, truth_frequency - offset] {
            if aliased > 0.0
                && (recovered_frequency - aliased).abs() / truth_frequency <= tolerance_fraction
            {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_helpers_match_contract() {
        assert_eq!(relative_error_pct(11.0, 10.0), 10.0);
        assert_eq!(alias_bucket(0.5), "1/2x");
        assert!(within_tolerance(10.1, 10.0, 0.02));
        assert!(near_day_alias(8.0, 12.0, 0.01));
    }

    #[test]
    fn synthetic_period_is_recovered() {
        let period = 0.04;
        let time: Vec<f64> = (0..96)
            .map(|index| 60_000.0 + index as f64 / 48.0)
            .collect();
        let magnitude: Vec<f64> = time
            .iter()
            .map(|time| 18.0 + 0.2 * (TAU * (*time - 60_000.0) / period).cos())
            .collect();
        let input = RotationPeriodInput {
            time: TimeArray::from_mjd(TimeScale::Tdb, &time).unwrap(),
            magnitude,
            magnitude_sigma: vec![0.02; 96],
            filter: vec!["r".to_string(); 96],
            session_id: vec![None; 96],
            r_au: vec![2.0; 96],
            delta_au: vec![1.0; 96],
            phase_angle_deg: vec![20.0; 96],
        };
        let config = RotationPeriodConfig {
            search_fidelity: "exact_grid".to_string(),
            fourier_orders: Some(vec![2]),
            max_frequency_cycles_per_day: 80.0,
            frequency_grid_scale: 10.0,
            ..RotationPeriodConfig::default()
        };
        let result = estimate_rotation_period(&input, &config).unwrap();
        // A single-max sinusoid is interpreted as half a double-peaked asteroid
        // lightcurve, so the reported physical period is doubled.
        assert!((result.period_days - 2.0 * period).abs() / (2.0 * period) < 0.1);
    }
}
