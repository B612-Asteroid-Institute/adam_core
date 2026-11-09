//! Canonical-filter-generic asteroid color estimation.
//!
//! This module owns the numerical and Arrow contracts for the fused color
//! estimator.  The PyO3 boundary supplies typed source/orbit geometry, but all
//! photometric validity classification, canonical-band resolution, optional
//! explicit projection, fitting, clipping, identifiability, covariance, status
//! selection, deterministic ordering, and result assembly happen here.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, LargeListArray, LargeStringArray, RecordBatch,
};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema};
use faer::Mat;
use nalgebra::{DMatrix, DVector};

use crate::bandpasses::BandpassData;
use crate::types::{SchemaError, SchemaResult};

const FITABILITY_TOL: f64 = 1.0e-8;
const MIN_RETAINED_FRACTION: f64 = 0.5;
const MIN_CLIP_REDUCED_CHI2: f64 = 1.0e-12;
const HG_PHASE_FLOOR: f64 = 1.0e-12;
const HG12_PHASE_FLOOR: f64 = 1.0e-10;

/// Supported phase laws for color estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPhaseModel {
    Hg12Star,
    Hg,
    C1C2,
}

impl ColorPhaseModel {
    pub fn parse(value: &str) -> SchemaResult<Self> {
        match value {
            "HG12star" => Ok(Self::Hg12Star),
            "HG" => Ok(Self::Hg),
            "c1c2" => Ok(Self::C1C2),
            _ => Err(invalid(format!(
                "Unsupported phi_type {value:?}; expected one of ('HG12star', 'HG', 'c1c2')"
            ))),
        }
    }

    fn phase_parameter_count(self) -> usize {
        match self {
            Self::C1C2 => 2,
            Self::Hg12Star | Self::Hg => 1,
        }
    }

    fn nominal_bounds(self) -> Option<(f64, f64, &'static str)> {
        match self {
            Self::Hg12Star => Some((0.0, 1.0, "G12*")),
            Self::Hg => Some((-0.255, 1.0, "G")),
            Self::C1C2 => None,
        }
    }
}

/// Reflectance composition required by an explicit filter projection.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorComposition {
    Template(String),
    Mix(f64, f64),
}

impl ColorComposition {
    fn args(&self) -> (Option<&str>, Option<(f64, f64)>) {
        match self {
            Self::Template(value) => (Some(value.as_str()), None),
            Self::Mix(c, s) => (None, Some((*c, *s))),
        }
    }
}

/// Explicit canonical source-filter to canonical target-filter projection.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorProjection {
    pub source_to_target: BTreeMap<String, String>,
    pub composition: ColorComposition,
}

/// Whole-call estimator options.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorEstimatorConfig {
    pub phase_model: ColorPhaseModel,
    pub force_phase_bounds: bool,
    pub strict_band_mapping: bool,
    pub projection: Option<ColorProjection>,
}

/// One source row after orbit propagation and observer-state preparation.
///
/// Positions are heliocentric ecliptic AU.  Missing positions mean geometry
/// preparation failed for the object; callers should emit a terminal `failed`
/// row rather than pass such a row to [`estimate_color_object`].
#[derive(Debug, Clone, PartialEq)]
pub struct ColorObservation {
    pub magnitude: Option<f64>,
    pub magnitude_sigma: Option<f64>,
    pub observatory_code: String,
    pub reported_filter: Option<String>,
    pub object_position: [f64; 3],
    pub observer_position: [f64; 3],
}

/// One final `ColorFit` row.  Option on a list is the top-level Arrow null;
/// floating non-identifiability inside a present list is represented by NaN.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorFitRow {
    pub object_id: String,
    pub status: String,
    pub error: Option<String>,
    pub filter_ids: Option<Vec<String>>,
    pub absolute_magnitudes: Option<Vec<f64>>,
    pub absolute_magnitude_sigmas: Option<Vec<f64>>,
    pub num_used_by_filter: Option<Vec<i64>>,
    pub magnitude_covariance: Option<Vec<f64>>,
    pub color_lhs_filter_ids: Option<Vec<String>>,
    pub color_rhs_filter_ids: Option<Vec<String>>,
    pub colors: Option<Vec<f64>>,
    pub color_sigmas: Option<Vec<f64>>,
    pub phase_param: Option<f64>,
    pub phase_param_sigma: Option<f64>,
    pub chi2: Option<f64>,
    pub reduced_chi2: Option<f64>,
    pub dof: Option<i64>,
    pub rank: Option<i64>,
    pub num_params: Option<i64>,
    pub converged: Option<bool>,
    pub num_obs: Option<i64>,
    pub num_invalid: Option<i64>,
    pub num_unsupported_filter: Option<i64>,
    pub num_clipped: Option<i64>,
    pub num_used: Option<i64>,
}

impl ColorFitRow {
    fn base(object_id: impl Into<String>, num_obs: usize) -> Self {
        Self {
            object_id: object_id.into(),
            status: "ok".to_string(),
            error: None,
            filter_ids: None,
            absolute_magnitudes: None,
            absolute_magnitude_sigmas: None,
            num_used_by_filter: None,
            magnitude_covariance: None,
            color_lhs_filter_ids: None,
            color_rhs_filter_ids: None,
            colors: None,
            color_sigmas: None,
            phase_param: None,
            phase_param_sigma: None,
            chi2: None,
            reduced_chi2: None,
            dof: None,
            rank: None,
            num_params: None,
            converged: None,
            num_obs: Some(num_obs as i64),
            num_invalid: None,
            num_unsupported_filter: None,
            num_clipped: None,
            num_used: None,
        }
    }

    /// Terminal row used when orbit matching or geometry preparation fails.
    pub fn terminal(
        object_id: impl Into<String>,
        num_obs: usize,
        status: &str,
        error: impl Into<String>,
    ) -> Self {
        let mut row = Self::base(object_id, num_obs);
        row.status = status.to_string();
        row.error = Some(error.into());
        row
    }

    fn classify_counts(&mut self, invalid: usize, unsupported: usize, clipped: usize, used: usize) {
        self.num_invalid = Some(invalid as i64);
        self.num_unsupported_filter = Some(unsupported as i64);
        self.num_clipped = Some(clipped as i64);
        self.num_used = Some(used as i64);
        debug_assert_eq!(
            self.num_obs,
            Some((invalid + unsupported + clipped + used) as i64)
        );
    }
}

fn invalid(message: String) -> SchemaError {
    SchemaError::InvalidRecordBatch(message)
}

/// Validate whole-call projection arguments and all synthetic conversions once.
/// Invalid configuration fails the call rather than becoming one failure per object.
pub fn validate_color_config(
    data: &BandpassData,
    config: &ColorEstimatorConfig,
) -> SchemaResult<()> {
    if let Some(projection) = &config.projection {
        let (template, mix) = projection.composition.args();
        for (source, target) in &projection.source_to_target {
            if source.is_empty() || target.is_empty() {
                return Err(invalid(
                    "projection source and target filter IDs must be non-empty".to_string(),
                ));
            }
            if !data.has_filter(source) {
                return Err(invalid(format!(
                    "Unknown canonical projection source filter_id: {source}"
                )));
            }
            if !data.has_filter(target) {
                return Err(invalid(format!(
                    "Unknown canonical projection target filter_id: {target}"
                )));
            }
            // Identities are deliberately validated too: composition is required
            // for projection mode even when every currently-used edge is identity.
            data.delta_mag(template, mix, source, target)?;
        }
        // Validate the composition even for an empty explicit mapping.
        data.delta_table(template, mix)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct EligibleRow {
    reduced_magnitude: f64,
    sigma: f64,
    alpha_deg: f64,
    filter_id: String,
}

/// Fit one independently-isolated object from raw source photometry and prepared
/// heliocentric geometry.
pub fn estimate_color_object(
    data: &BandpassData,
    object_id: &str,
    observations: &[ColorObservation],
    config: &ColorEstimatorConfig,
) -> ColorFitRow {
    let mut output = ColorFitRow::base(object_id, observations.len());

    let mut canonical = vec![None; observations.len()];
    let reported_rows: Vec<usize> = observations
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row.reported_filter.as_ref().map(|_| index))
        .collect();
    if !reported_rows.is_empty() {
        let codes = reported_rows
            .iter()
            .map(|&index| observations[index].observatory_code.clone())
            .collect::<Vec<_>>();
        let bands = reported_rows
            .iter()
            .map(|&index| observations[index].reported_filter.clone().unwrap())
            .collect::<Vec<_>>();
        match data.map_to_canonical_filter_bands_optional(
            &codes,
            &bands,
            !config.strict_band_mapping,
            true,
        ) {
            Ok(mapped) => {
                for (&index, value) in reported_rows.iter().zip(mapped) {
                    canonical[index] = value;
                }
            }
            Err(error) => {
                return ColorFitRow::terminal(
                    object_id,
                    observations.len(),
                    "failed",
                    format!("filter resolution failed: {error}"),
                );
            }
        }
    }

    let mut invalid_count = 0usize;
    let mut unsupported_count = 0usize;
    let mut classified = Vec::new();
    let mut conversion_cache: BTreeMap<(String, String), f64> = BTreeMap::new();

    for (index, row) in observations.iter().enumerate() {
        // Invalid photometry takes precedence over filter support.
        let valid_photometry = matches!(row.magnitude, Some(value) if value.is_finite())
            && matches!(row.magnitude_sigma, Some(value) if value.is_finite() && value > 0.0);
        if !valid_photometry {
            invalid_count += 1;
            continue;
        }
        let Some(source_filter) = canonical[index].as_ref() else {
            unsupported_count += 1;
            continue;
        };
        let (target_filter, delta_mag) = match &config.projection {
            None => (source_filter.clone(), 0.0),
            Some(projection) => {
                let Some(target) = projection.source_to_target.get(source_filter) else {
                    unsupported_count += 1;
                    continue;
                };
                let key = (source_filter.clone(), target.clone());
                let delta = if let Some(value) = conversion_cache.get(&key) {
                    *value
                } else {
                    let (template, mix) = projection.composition.args();
                    match data.delta_mag(template, mix, source_filter, target) {
                        Ok(value) => {
                            conversion_cache.insert(key, value);
                            value
                        }
                        Err(error) => {
                            return ColorFitRow::terminal(
                                object_id,
                                observations.len(),
                                "failed",
                                format!("bandpass projection failed: {error}"),
                            );
                        }
                    }
                };
                (target.clone(), delta)
            }
        };

        classified.push((index, target_filter, delta_mag));
    }

    let classified_count = classified.len();
    let classified_failure = |message: &str| {
        let mut failed = ColorFitRow::terminal(object_id, observations.len(), "failed", message);
        // `num_used` means rows admitted to the attempted scientific fit. Even
        // when geometry prevents a result, this preserves the exhaustive row
        // partition without relabeling valid supported photometry as clipped.
        failed.classify_counts(invalid_count, unsupported_count, 0, classified_count);
        failed
    };
    let mut eligible = Vec::with_capacity(classified.len());
    for (index, target_filter, delta_mag) in classified {
        let row = &observations[index];
        let Some((r, delta, alpha_deg)) =
            observing_geometry(row.object_position, row.observer_position)
        else {
            return classified_failure(
                "invalid photometry geometry: non-finite or non-positive distance",
            );
        };
        let reduced = row.magnitude.expect("photometry classified as valid")
            - 5.0 * (r * delta).log10()
            + delta_mag;
        if !reduced.is_finite() || !alpha_deg.is_finite() {
            return classified_failure("invalid reduced magnitude or phase angle");
        }
        eligible.push(EligibleRow {
            reduced_magnitude: reduced,
            sigma: row
                .magnitude_sigma
                .expect("photometric uncertainty classified as valid"),
            alpha_deg,
            filter_id: target_filter,
        });
    }

    if eligible.is_empty() {
        output.status = "no_usable_photometry".to_string();
        output.error = Some(format!(
            "None of the {} observation(s) has valid, supported photometry",
            observations.len()
        ));
        output.classify_counts(invalid_count, unsupported_count, 0, 0);
        return output;
    }

    let fit = match fit_eligible_rows(&eligible, config.phase_model) {
        Ok(fit) => fit,
        Err(FitFailure::TooManyOutliers { included }) => {
            let used = included.iter().filter(|&&value| value).count();
            let clipped = eligible.len() - used;
            output.status = "too_many_outliers".to_string();
            output.error = Some(format!(
                "Outlier rejection removed {clipped} of the {} eligible observation(s); fit is unreliable",
                eligible.len()
            ));
            output.classify_counts(invalid_count, unsupported_count, clipped, used);
            return output;
        }
        Err(FitFailure::Other(error)) => {
            output.status = "failed".to_string();
            output.error = Some(error);
            // Classification completed. With no statistical result, preserve a
            // complete partition by treating every eligible row as used.
            output.classify_counts(invalid_count, unsupported_count, 0, eligible.len());
            return output;
        }
    };

    let used = fit.included.iter().filter(|&&value| value).count();
    let clipped = eligible.len() - used;
    output.classify_counts(invalid_count, unsupported_count, clipped, used);

    let mut active_filters: Vec<String> = fit.active_filters.clone();
    active_filters.sort(); // Rust String ordering is byte-wise Unicode scalar/UTF-8 lexical order.
    let parameter_for_filter: BTreeMap<&str, usize> = fit
        .active_filters
        .iter()
        .enumerate()
        .map(|(index, filter)| (filter.as_str(), fit.n_phase + index))
        .collect();
    let sorted_parameter_indices = active_filters
        .iter()
        .map(|filter| parameter_for_filter[filter.as_str()])
        .collect::<Vec<_>>();

    let counts = active_filters
        .iter()
        .map(|filter| {
            eligible
                .iter()
                .zip(&fit.included)
                .filter(|(row, included)| **included && row.filter_id == *filter)
                .count() as i64
        })
        .collect::<Vec<_>>();
    debug_assert_eq!(counts.iter().sum::<i64>(), used as i64);

    let mut magnitudes = Vec::with_capacity(active_filters.len());
    let mut magnitude_sigmas = Vec::with_capacity(active_filters.len());
    let fitable_h = sorted_parameter_indices
        .iter()
        .map(|&index| fit.analysis.is_fitable_unit(index))
        .collect::<Vec<_>>();
    for (&index, &fitable) in sorted_parameter_indices.iter().zip(&fitable_h) {
        magnitudes.push(if fitable { fit.values[index] } else { f64::NAN });
        magnitude_sigmas.push(parameter_sigma(&fit, index, fitable));
    }

    let n_filters = active_filters.len();
    let mut exposed_covariance = vec![f64::NAN; n_filters * n_filters];
    if fit.dof > 0 {
        for i in 0..n_filters {
            for j in 0..n_filters {
                if fitable_h[i] && fitable_h[j] {
                    exposed_covariance[i * n_filters + j] =
                        fit.covariance[(sorted_parameter_indices[i], sorted_parameter_indices[j])];
                }
            }
        }
    }

    let mut lhs = Vec::new();
    let mut rhs = Vec::new();
    let mut colors = Vec::new();
    let mut color_sigmas = Vec::new();
    for i in 0..n_filters {
        for j in (i + 1)..n_filters {
            lhs.push(active_filters[i].clone());
            rhs.push(active_filters[j].clone());
            let pi = sorted_parameter_indices[i];
            let pj = sorted_parameter_indices[j];
            let contrast_fitable = fit.analysis.is_fitable_contrast(pi, pj);
            if contrast_fitable {
                colors.push(fit.values[pi] - fit.values[pj]);
                if fit.dof > 0 {
                    let variance = fit.covariance[(pi, pi)] + fit.covariance[(pj, pj)]
                        - 2.0 * fit.covariance[(pi, pj)];
                    color_sigmas.push(positive_sqrt_or_nan(variance));
                } else {
                    color_sigmas.push(f64::NAN);
                }
            } else {
                colors.push(f64::NAN);
                color_sigmas.push(f64::NAN);
            }
        }
    }

    let (phase_param, phase_param_sigma) = match config.phase_model {
        ColorPhaseModel::C1C2 => (f64::NAN, f64::NAN),
        _ => {
            let fitable = fit.analysis.is_fitable_unit(0);
            (
                if fitable { fit.values[0] } else { f64::NAN },
                parameter_sigma(&fit, 0, fitable),
            )
        }
    };

    output.phase_param = Some(phase_param);
    output.phase_param_sigma = Some(phase_param_sigma);
    output.chi2 = Some(fit.chi2);
    output.reduced_chi2 = Some(fit.reduced_chi2);
    output.dof = Some(fit.dof as i64);
    output.rank = Some(fit.analysis.rank as i64);
    output.num_params = Some(fit.values.len() as i64);
    output.converged = Some(fit.converged);
    output.filter_ids = Some(active_filters);
    output.num_used_by_filter = Some(counts);

    if let Some((low, high, label)) = config.phase_model.nominal_bounds() {
        if phase_param.is_finite() && !(low..=high).contains(&phase_param) {
            let message = format!(
                "Fitted {label} = {phase_param:.4} for {object_id} is outside the validity bounds [{low}, {high}]"
            );
            if config.force_phase_bounds {
                output.status = "phase_param_out_of_bounds".to_string();
                output.error = Some(message);
                // filter_ids and counts are intentionally retained; every
                // scientific product and pair-list is a top-level null.
                return output;
            }
        }
    }

    output.absolute_magnitudes = Some(magnitudes);
    output.absolute_magnitude_sigmas = Some(magnitude_sigmas);
    output.magnitude_covariance = Some(exposed_covariance);
    output.color_lhs_filter_ids = Some(lhs);
    output.color_rhs_filter_ids = Some(rhs);
    output.colors = Some(colors);
    output.color_sigmas = Some(color_sigmas);
    output
}

fn observing_geometry(object: [f64; 3], observer: [f64; 3]) -> Option<(f64, f64, f64)> {
    let r2 = object.iter().map(|value| value * value).sum::<f64>();
    let delta_vector = [
        object[0] - observer[0],
        object[1] - observer[1],
        object[2] - observer[2],
    ];
    let delta2 = delta_vector.iter().map(|value| value * value).sum::<f64>();
    let observer2 = observer.iter().map(|value| value * value).sum::<f64>();
    let r = r2.sqrt();
    let delta = delta2.sqrt();
    if !r.is_finite() || !delta.is_finite() || r <= 0.0 || delta <= 0.0 {
        return None;
    }
    let cosine = ((r2 + delta2 - observer2) / (2.0 * r * delta)).clamp(-1.0, 1.0);
    let alpha = 2.0
        * (0.0_f64.max(1.0 - cosine))
            .sqrt()
            .atan2((0.0_f64.max(1.0 + cosine)).sqrt());
    Some((r, delta, alpha.to_degrees()))
}

#[derive(Debug)]
enum FitFailure {
    TooManyOutliers { included: Vec<bool> },
    Other(String),
}

#[derive(Debug)]
struct MatrixAnalysis {
    rank: usize,
    v_t: DMatrix<f64>,
}

impl MatrixAnalysis {
    fn is_fitable_vector(&self, vector: &DVector<f64>) -> bool {
        if self.rank == self.v_t.nrows() {
            return true;
        }
        let mut leak2 = 0.0;
        for null_row in self.rank..self.v_t.nrows() {
            let projection = (0..self.v_t.ncols())
                .map(|column| self.v_t[(null_row, column)] * vector[column])
                .sum::<f64>();
            leak2 += projection * projection;
        }
        leak2.sqrt() <= FITABILITY_TOL * vector.norm()
    }

    fn is_fitable_unit(&self, index: usize) -> bool {
        let mut vector = DVector::zeros(self.v_t.ncols());
        vector[index] = 1.0;
        self.is_fitable_vector(&vector)
    }

    fn is_fitable_contrast(&self, left: usize, right: usize) -> bool {
        let mut vector = DVector::zeros(self.v_t.ncols());
        vector[left] = 1.0;
        vector[right] = -1.0;
        self.is_fitable_vector(&vector)
    }
}

#[derive(Debug)]
struct ColorFitNumerics {
    active_filters: Vec<String>,
    n_phase: usize,
    values: DVector<f64>,
    covariance: DMatrix<f64>,
    analysis: MatrixAnalysis,
    included: Vec<bool>,
    chi2: f64,
    reduced_chi2: f64,
    dof: isize,
    converged: bool,
}

struct OnceFit {
    active_filters: Vec<String>,
    n_phase: usize,
    values: DVector<f64>,
    weighted_jacobian: DMatrix<f64>,
    model: Vec<f64>,
    converged: bool,
}

fn fit_eligible_rows(
    rows: &[EligibleRow],
    phase_model: ColorPhaseModel,
) -> Result<ColorFitNumerics, FitFailure> {
    let mut included = vec![true; rows.len()];
    let mut slope_guess = 0.15;
    loop {
        let once = fit_once(rows, &included, phase_model, slope_guess)?;
        if phase_model != ColorPhaseModel::C1C2 {
            slope_guess = once.values[0];
        }
        let decomposition = robust_svd(&once.weighted_jacobian)?;
        let analysis = analyze_matrix(
            &decomposition,
            once.weighted_jacobian.nrows(),
            once.weighted_jacobian.ncols(),
        );
        let chi2 = rows
            .iter()
            .zip(&included)
            .zip(&once.model)
            .filter(|((_, included), _)| **included)
            .map(|((row, _), model)| ((model - row.reduced_magnitude) / row.sigma).powi(2))
            .sum::<f64>();
        let used = included.iter().filter(|&&value| value).count();
        let dof = used as isize - analysis.rank as isize;
        let scatter = if dof > 0 {
            let value = chi2 / dof as f64;
            if value >= MIN_CLIP_REDUCED_CHI2 {
                value
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };
        let mut removed = false;
        for (index, row) in rows.iter().enumerate() {
            if included[index]
                && ((once.model[index] - row.reduced_magnitude) / row.sigma).powi(2) > 9.0 * scatter
            {
                included[index] = false;
                removed = true;
            }
        }
        if removed {
            continue;
        }
        let retained = included.iter().filter(|&&value| value).count();
        if (retained as f64) < MIN_RETAINED_FRACTION * rows.len() as f64 {
            return Err(FitFailure::TooManyOutliers { included });
        }

        let reduced_chi2 = if dof > 0 { chi2 / dof as f64 } else { f64::NAN };
        let covariance = covariance_from_jtj_pseudoinverse(&decomposition, reduced_chi2, dof);
        return Ok(ColorFitNumerics {
            active_filters: once.active_filters,
            n_phase: once.n_phase,
            values: once.values,
            covariance,
            analysis,
            included,
            chi2,
            reduced_chi2,
            dof,
            converged: once.converged,
        });
    }
}

fn active_filters(rows: &[EligibleRow], included: &[bool]) -> Vec<String> {
    let mut filters = Vec::new();
    let mut seen = BTreeSet::new();
    // Parameter order need not be public ordering, but deterministic lexical
    // order simplifies diagnostics and makes repeated clipping stable.
    for (row, &keep) in rows.iter().zip(included) {
        if keep {
            seen.insert(row.filter_id.clone());
        }
    }
    filters.extend(seen);
    filters
}

fn fit_once(
    rows: &[EligibleRow],
    included: &[bool],
    phase_model: ColorPhaseModel,
    slope_guess: f64,
) -> Result<OnceFit, FitFailure> {
    let filters = active_filters(rows, included);
    if filters.is_empty() {
        return Err(FitFailure::TooManyOutliers {
            included: included.to_vec(),
        });
    }
    let n_phase = phase_model.phase_parameter_count();
    let width = n_phase + filters.len();
    let included_indices = included
        .iter()
        .enumerate()
        .filter_map(|(index, &keep)| keep.then_some(index))
        .collect::<Vec<_>>();
    let filter_index: BTreeMap<&str, usize> = filters
        .iter()
        .enumerate()
        .map(|(index, filter)| (filter.as_str(), index))
        .collect();

    if phase_model == ColorPhaseModel::C1C2 {
        let mut matrix = DMatrix::zeros(included_indices.len(), width);
        let mut response = DVector::zeros(included_indices.len());
        for (output, &index) in included_indices.iter().enumerate() {
            let row = &rows[index];
            let alpha = row.alpha_deg.to_radians();
            let weight = 1.0 / row.sigma;
            matrix[(output, 0)] = alpha * weight;
            matrix[(output, 1)] = alpha * alpha * weight;
            matrix[(output, n_phase + filter_index[row.filter_id.as_str()])] = weight;
            response[output] = row.reduced_magnitude * weight;
        }
        let decomposition = robust_svd(&matrix)?;
        let values =
            minimum_norm_solution(&decomposition, &response, matrix.nrows(), matrix.ncols());
        let model = rows
            .iter()
            .map(|row| {
                let Some(&band_index) = filter_index.get(row.filter_id.as_str()) else {
                    return f64::NAN;
                };
                let alpha = row.alpha_deg.to_radians();
                values[0] * alpha + values[1] * alpha * alpha + values[n_phase + band_index]
            })
            .collect();
        return Ok(OnceFit {
            active_filters: filters,
            n_phase,
            values,
            weighted_jacobian: matrix,
            model,
            converged: true,
        });
    }

    let mut values = DVector::zeros(width);
    values[0] = slope_guess;
    for (filter, &band_index) in &filter_index {
        let matching = included_indices
            .iter()
            .filter(|&&index| rows[index].filter_id == *filter)
            .map(|&index| rows[index].reduced_magnitude);
        let (sum, count) = matching.fold((0.0, 0usize), |(sum, count), value| {
            (sum + value, count + 1)
        });
        values[n_phase + band_index] = sum / count as f64;
    }

    // Unbounded trust-region reflective least squares, matching SciPy's default
    // relative termination policy and evaluation budget. The exact SVD step is
    // stable for rank-deficient layouts and avoids normal-equation squaring.
    const FTOL: f64 = 1.0e-8;
    const XTOL: f64 = 1.0e-8;
    const GTOL: f64 = 1.0e-8;
    let max_nfev = 100 * width;
    let mut nfev = 1usize;
    let mut radius = 0.0_f64;
    let mut alpha = 0.0_f64;
    let mut converged = false;
    let mut current = nonlinear_residual_jacobian(
        rows,
        &included_indices,
        &filters,
        &filter_index,
        phase_model,
        &values,
    )?;
    'iterations: while nfev < max_nfev {
        let (residual, jacobian) = (&current.0, &current.1);
        let gradient = jacobian.transpose() * residual;
        if gradient.amax() <= GTOL {
            converged = true;
            break;
        }
        if radius == 0.0 {
            radius = values.norm().max(1.0);
        }
        let decomposition = robust_svd(jacobian)?;
        let mut eigenvalues = vec![0.0; width];
        for (index, &singular) in decomposition.singular_values.iter().enumerate() {
            eigenvalues[index] = singular * singular;
        }
        let q = (0..width)
            .map(|component| {
                (0..width)
                    .map(|column| decomposition.v_t[(component, column)] * gradient[column])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let state_norm = values.norm();
        let mut accepted = false;
        for _ in 0..crate::trust_region::MAX_STEP_ATTEMPTS {
            if nfev >= max_nfev {
                break 'iterations;
            }
            let (step_eigen, alpha_used) = crate::trust_region::trust_region_step(
                &eigenvalues,
                &q,
                radius,
                alpha,
                crate::trust_region::RankDeficientStepPolicy::MinimumNormInterior,
            );
            alpha = alpha_used;
            let step_scaled = DVector::from_iterator(
                width,
                (0..width).map(|column| {
                    (0..width)
                        .map(|component| {
                            decomposition.v_t[(component, column)] * step_eigen[component]
                        })
                        .sum::<f64>()
                }),
            );
            let step_scaled_norm = step_scaled.norm();
            let step = step_scaled;
            let step_norm = step.norm();
            if !step.iter().all(|value| value.is_finite()) || step_scaled_norm == 0.0 {
                break 'iterations;
            }
            let candidate_values = &values + &step;
            let candidate = nonlinear_residual_jacobian(
                rows,
                &included_indices,
                &filters,
                &filter_index,
                phase_model,
                &candidate_values,
            )?;
            nfev += 1;
            let old_cost = residual.norm_squared();
            let new_cost = candidate.0.norm_squared();
            if !new_cost.is_finite() {
                radius = 0.25 * step_scaled_norm;
                continue;
            }
            let actual_reduction = 0.5 * (old_cost - new_cost);
            let predicted_reduction = -(0..width)
                .map(|index| {
                    q[index] * step_eigen[index]
                        + 0.5 * eigenvalues[index] * step_eigen[index].powi(2)
                })
                .sum::<f64>();
            let ratio = if predicted_reduction > 0.0 {
                actual_reduction / predicted_reduction
            } else {
                0.0
            };
            let new_radius = if ratio < 0.25 {
                0.25 * step_scaled_norm
            } else if ratio > 0.75 && step_scaled_norm > 0.95 * radius {
                2.0 * radius
            } else {
                radius
            };
            if new_radius > 0.0 {
                alpha *= radius / new_radius;
            }
            radius = new_radius;
            let xtol_met = step_norm < XTOL * (XTOL + state_norm);
            if actual_reduction > 0.0 {
                let ftol_met = (old_cost - new_cost) < FTOL * old_cost && ratio > 0.25;
                values = candidate_values;
                current = candidate;
                accepted = true;
                if ftol_met || xtol_met {
                    converged = true;
                }
                break;
            }
            if xtol_met {
                converged = true;
                break 'iterations;
            }
        }
        if converged || !accepted {
            break;
        }
    }

    // Recompute at the accepted solution, and construct an all-row model used
    // by clipping while keeping the Jacobian restricted to included rows.
    let (_, weighted_jacobian) = nonlinear_residual_jacobian(
        rows,
        &included_indices,
        &filters,
        &filter_index,
        phase_model,
        &values,
    )?;
    let model = rows
        .iter()
        .map(|row| {
            let Some(&band_index) = filter_index.get(row.filter_id.as_str()) else {
                return Ok(f64::NAN);
            };
            phase_correction(phase_model, row.alpha_deg, values[0])
                .map(|(correction, _)| correction + values[n_phase + band_index])
        })
        .collect::<Result<Vec<_>, FitFailure>>()?;
    Ok(OnceFit {
        active_filters: filters,
        n_phase,
        values,
        weighted_jacobian,
        model,
        converged,
    })
}

fn nonlinear_residual_jacobian(
    rows: &[EligibleRow],
    included_indices: &[usize],
    filters: &[String],
    filter_index: &BTreeMap<&str, usize>,
    phase_model: ColorPhaseModel,
    values: &DVector<f64>,
) -> Result<(DVector<f64>, DMatrix<f64>), FitFailure> {
    let width = 1 + filters.len();
    let mut residual = DVector::zeros(included_indices.len());
    let mut jacobian = DMatrix::zeros(included_indices.len(), width);
    for (output, &index) in included_indices.iter().enumerate() {
        let row = &rows[index];
        let (correction, derivative) = phase_correction(phase_model, row.alpha_deg, values[0])?;
        let h_index = 1 + filter_index[row.filter_id.as_str()];
        let weight = 1.0 / row.sigma;
        residual[output] = (correction + values[h_index] - row.reduced_magnitude) * weight;
        jacobian[(output, 0)] = derivative * weight;
        jacobian[(output, h_index)] = weight;
    }
    Ok((residual, jacobian))
}

fn phase_correction(
    model: ColorPhaseModel,
    alpha_deg: f64,
    slope: f64,
) -> Result<(f64, f64), FitFailure> {
    match model {
        ColorPhaseModel::Hg => Ok(hg_correction(alpha_deg, slope)),
        ColorPhaseModel::Hg12Star => hg12star_correction(alpha_deg, slope),
        ColorPhaseModel::C1C2 => unreachable!("c1c2 is linear"),
    }
}

fn hg_correction(alpha_deg: f64, slope: f64) -> (f64, f64) {
    let tan_half = (0.5 * alpha_deg.to_radians()).tan();
    let phi1 = (-3.33 * tan_half.powf(0.63)).exp();
    let phi2 = (-1.87 * tan_half.powf(1.22)).exp();
    let phase = (1.0 - slope) * phi1 + slope * phi2;
    let clipped = phase.max(HG_PHASE_FLOOR);
    let correction = -2.5 * clipped.log10();
    let derivative = if phase > HG_PHASE_FLOOR {
        -2.5 / std::f64::consts::LN_10 * (phi2 - phi1) / clipped
    } else {
        0.0
    };
    (correction, derivative)
}

const XI12_X: [f64; 6] = [7.5, 30.0, 60.0, 90.0, 120.0, 150.0];
const XI1_Y: [f64; 6] = [
    7.5e-1,
    3.348_601_6e-1,
    1.341_056_0e-1,
    5.110_475_6e-2,
    2.146_568_7e-2,
    3.639_698_9e-3,
];
const XI1_D: [f64; 6] = [
    -1.909_859_3,
    -5.546_343_2e-1,
    -2.440_459_9e-1,
    -9.498_043_8e-2,
    -2.141_142_4e-2,
    -9.132_861_2e-2,
];
const XI2_Y: [f64; 6] = [
    9.25e-1,
    6.288_416_9e-1,
    3.175_549_5e-1,
    1.271_636_7e-1,
    2.237_390_3e-2,
    1.650_568_9e-4,
];
const XI2_D: [f64; 6] = [
    -5.729_578_0e-1,
    -7.670_536_7e-1,
    -4.566_578_9e-1,
    -2.807_180_9e-1,
    -1.117_325_7e-1,
    -8.657_313_8e-8,
];
const XI3_X: [f64; 9] = [0.0, 0.3, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0];
const XI3_Y: [f64; 9] = [
    1.0,
    8.338_118_5e-1,
    5.773_542_4e-1,
    4.214_477_2e-1,
    2.317_423_0e-1,
    1.034_817_8e-1,
    6.173_347_3e-2,
    1.610_700_6e-2,
    0.0,
];
const XI3_D: [f64; 9] = [
    -1.063_009_7e-1,
    -4.118_043_9e1,
    -1.036_691_5e1,
    -7.578_461_5,
    -3.696_095_0,
    -7.860_565_2e-1,
    -4.652_701_2e-1,
    -2.045_954_5e-1,
    0.0,
];

fn hermite_spline<const N: usize>(
    x_deg: f64,
    xs_deg: &[f64; N],
    ys: &[f64; N],
    ds: &[f64; N],
) -> f64 {
    let x = x_deg.to_radians();
    let xs = xs_deg.map(f64::to_radians);
    let right = xs.partition_point(|value| *value <= x);
    let index = right.saturating_sub(1).min(N - 2);
    let dx = xs[index + 1] - xs[index];
    let dy = ys[index + 1] - ys[index];
    let t = (x - xs[index]) / dx;
    let a = ds[index] * dx - dy;
    let b = -ds[index + 1] * dx + dy;
    (1.0 - t) * ys[index] + t * ys[index + 1] + t * (1.0 - t) * ((1.0 - t) * a + t * b)
}

fn hg12star_basis(alpha_deg: f64) -> (f64, f64, f64) {
    let phi1 = if alpha_deg <= 7.5 {
        1.0 - (6.0 / std::f64::consts::PI) * alpha_deg.to_radians()
    } else {
        hermite_spline(alpha_deg, &XI12_X, &XI1_Y, &XI1_D)
    };
    let phi2 = if alpha_deg <= 7.5 {
        1.0 - (9.0 / (5.0 * std::f64::consts::PI)) * alpha_deg.to_radians()
    } else {
        hermite_spline(alpha_deg, &XI12_X, &XI2_Y, &XI2_D)
    };
    let phi3 = if alpha_deg <= 30.0 {
        hermite_spline(alpha_deg, &XI3_X, &XI3_Y, &XI3_D)
    } else {
        0.0
    };
    (phi1, phi2, phi3)
}

fn hg12star_correction(alpha_deg: f64, slope: f64) -> Result<(f64, f64), FitFailure> {
    if !(0.0..=150.0).contains(&alpha_deg) {
        return Err(FitFailure::Other(format!(
            "HG12* phase angle {alpha_deg:.2} deg is outside [0, 150] deg"
        )));
    }
    const G1_SCALE: f64 = 0.842_936_49;
    const G2_SCALE: f64 = 0.535_133_50;
    let (phi1, phi2, phi3) = hg12star_basis(alpha_deg);
    let g1 = G1_SCALE * slope;
    let g2 = G2_SCALE * (1.0 - slope);
    let combined = g1 * phi1 + g2 * phi2 + (1.0 - g1 - g2) * phi3;
    let clipped = combined.max(HG12_PHASE_FLOOR);
    let correction = -2.5 * clipped.log10();
    let d_combined = G1_SCALE * phi1 - G2_SCALE * phi2 + (G2_SCALE - G1_SCALE) * phi3;
    let derivative = if combined > HG12_PHASE_FLOOR {
        -2.5 / std::f64::consts::LN_10 * d_combined / clipped
    } else {
        0.0
    };
    Ok((correction, derivative))
}

#[derive(Debug)]
struct RobustSvd {
    singular_values: Vec<f64>,
    u: DMatrix<f64>,
    v_t: DMatrix<f64>,
}

fn robust_svd(matrix: &DMatrix<f64>) -> Result<RobustSvd, FitFailure> {
    let mut input = Mat::<f64>::zeros(matrix.nrows(), matrix.ncols());
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            input[(row, column)] = matrix[(row, column)];
        }
    }
    // The common overdetermined case needs thin U but already receives a
    // complete square V. Only underdetermined fits request the full factors so
    // their explicit right-null basis is retained without an O(rows^2) U for
    // ordinary large objects.
    let decomposition = if matrix.nrows() >= matrix.ncols() {
        input.thin_svd()
    } else {
        input.svd()
    }
    .map_err(|error| FitFailure::Other(format!("color-fit SVD failed: {error:?}")))?;
    let singular_values = (0..matrix.nrows().min(matrix.ncols()))
        .map(|index| decomposition.S()[index])
        .collect::<Vec<_>>();
    let mut u = DMatrix::zeros(matrix.nrows(), singular_values.len());
    for row in 0..matrix.nrows() {
        for column in 0..singular_values.len() {
            u[(row, column)] = decomposition.U()[(row, column)];
        }
    }
    let mut v_t = DMatrix::zeros(matrix.ncols(), matrix.ncols());
    for row in 0..matrix.ncols() {
        for column in 0..matrix.ncols() {
            v_t[(row, column)] = decomposition.V()[(column, row)];
        }
    }
    Ok(RobustSvd {
        singular_values,
        u,
        v_t,
    })
}

fn svd_cutoff(singular: &[f64], rows: usize, columns: usize) -> f64 {
    singular.first().copied().unwrap_or(0.0) * rows.max(columns) as f64 * f64::EPSILON
}

fn minimum_norm_solution(
    decomposition: &RobustSvd,
    response: &DVector<f64>,
    rows: usize,
    columns: usize,
) -> DVector<f64> {
    let cutoff = svd_cutoff(&decomposition.singular_values, rows, columns);
    let mut values = DVector::zeros(columns);
    for component in 0..decomposition.singular_values.len() {
        let singular = decomposition.singular_values[component];
        if singular <= cutoff {
            continue;
        }
        let coefficient = (0..rows)
            .map(|row| decomposition.u[(row, component)] * response[row])
            .sum::<f64>()
            / singular;
        for column in 0..columns {
            values[column] += decomposition.v_t[(component, column)] * coefficient;
        }
    }
    values
}

fn analyze_matrix(decomposition: &RobustSvd, rows: usize, columns: usize) -> MatrixAnalysis {
    let cutoff = svd_cutoff(&decomposition.singular_values, rows, columns);
    let rank = decomposition
        .singular_values
        .iter()
        .filter(|&&value| value > cutoff)
        .count();
    MatrixAnalysis {
        rank,
        v_t: decomposition.v_t.clone(),
    }
}

fn covariance_from_jtj_pseudoinverse(
    decomposition: &RobustSvd,
    reduced_chi2: f64,
    dof: isize,
) -> DMatrix<f64> {
    let width = decomposition.v_t.ncols();
    if dof <= 0 {
        return DMatrix::from_element(width, width, f64::NAN);
    }
    let largest_squared = decomposition
        .singular_values
        .first()
        .map_or(0.0, |value| value * value);
    // NumPy's pinv(J.T @ J) default rcond is 1e-15. Apply that cutoff to
    // the eigenvalues s^2 of J.T @ J, independently of the matrix-rank
    // threshold used for diagnostics and null-space fitability.
    let cutoff = 1.0e-15 * largest_squared;
    let mut covariance = DMatrix::zeros(width, width);
    for component in 0..decomposition.singular_values.len() {
        let eigenvalue = decomposition.singular_values[component].powi(2);
        if eigenvalue <= cutoff {
            continue;
        }
        let scale = reduced_chi2 / eigenvalue;
        for row in 0..width {
            for column in 0..width {
                covariance[(row, column)] += scale
                    * decomposition.v_t[(component, row)]
                    * decomposition.v_t[(component, column)];
            }
        }
    }
    covariance
}

fn positive_sqrt_or_nan(value: f64) -> f64 {
    if value > 0.0 {
        value.sqrt()
    } else {
        f64::NAN
    }
}

fn parameter_sigma(fit: &ColorFitNumerics, index: usize, fitable: bool) -> f64 {
    if !fitable || fit.dof <= 0 {
        return f64::NAN;
    }
    positive_sqrt_or_nan(fit.covariance[(index, index)])
}

/// Exact final Rust-native `ColorFit` schema (without temporary fixed-band fields).
pub fn color_fit_schema() -> Arc<Schema> {
    let list = |name: &str, item_type: DataType| {
        Field::new(
            name,
            DataType::LargeList(Arc::new(Field::new("item", item_type, false))),
            true,
        )
    };
    Arc::new(Schema::new(vec![
        Field::new("object_id", DataType::LargeUtf8, false),
        Field::new("status", DataType::LargeUtf8, false),
        Field::new("error", DataType::LargeUtf8, true),
        list("filter_ids", DataType::LargeUtf8),
        list("absolute_magnitudes", DataType::Float64),
        list("absolute_magnitude_sigmas", DataType::Float64),
        list("num_used_by_filter", DataType::Int64),
        list("magnitude_covariance", DataType::Float64),
        list("color_lhs_filter_ids", DataType::LargeUtf8),
        list("color_rhs_filter_ids", DataType::LargeUtf8),
        list("colors", DataType::Float64),
        list("color_sigmas", DataType::Float64),
        Field::new("phase_param", DataType::Float64, true),
        Field::new("phase_param_sigma", DataType::Float64, true),
        Field::new("chi2", DataType::Float64, true),
        Field::new("reduced_chi2", DataType::Float64, true),
        Field::new("dof", DataType::Int64, true),
        Field::new("rank", DataType::Int64, true),
        Field::new("num_params", DataType::Int64, true),
        Field::new("converged", DataType::Boolean, true),
        Field::new("num_obs", DataType::Int64, true),
        Field::new("num_invalid", DataType::Int64, true),
        Field::new("num_unsupported_filter", DataType::Int64, true),
        Field::new("num_clipped", DataType::Int64, true),
        Field::new("num_used", DataType::Int64, true),
    ]))
}

fn string_lists(values: impl Iterator<Item = Option<Vec<String>>>, field: Arc<Field>) -> ArrayRef {
    let rows = values.collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut valid = Vec::with_capacity(rows.len());
    let mut flattened = Vec::new();
    offsets.push(0_i64);
    for row in rows {
        valid.push(row.is_some());
        if let Some(items) = row {
            flattened.extend(items);
        }
        offsets.push(flattened.len() as i64);
    }
    Arc::new(LargeListArray::new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(LargeStringArray::from_iter_values(flattened)) as ArrayRef,
        Some(NullBuffer::from_iter(valid)),
    ))
}

fn f64_lists(values: impl Iterator<Item = Option<Vec<f64>>>, field: Arc<Field>) -> ArrayRef {
    let rows = values.collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut valid = Vec::with_capacity(rows.len());
    let mut flattened = Vec::new();
    offsets.push(0_i64);
    for row in rows {
        valid.push(row.is_some());
        if let Some(items) = row {
            flattened.extend(items);
        }
        offsets.push(flattened.len() as i64);
    }
    Arc::new(LargeListArray::new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Float64Array::from(flattened)) as ArrayRef,
        Some(NullBuffer::from_iter(valid)),
    ))
}

fn i64_lists(values: impl Iterator<Item = Option<Vec<i64>>>, field: Arc<Field>) -> ArrayRef {
    let rows = values.collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut valid = Vec::with_capacity(rows.len());
    let mut flattened = Vec::new();
    offsets.push(0_i64);
    for row in rows {
        valid.push(row.is_some());
        if let Some(items) = row {
            flattened.extend(items);
        }
        offsets.push(flattened.len() as i64);
    }
    Arc::new(LargeListArray::new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Int64Array::from(flattened)) as ArrayRef,
        Some(NullBuffer::from_iter(valid)),
    ))
}

/// Assemble rows into the exact Arrow schema consumed by the thin Python facade.
pub fn color_fit_record_batch(rows: &[ColorFitRow]) -> SchemaResult<RecordBatch> {
    let schema = color_fit_schema();
    let list_field = |index: usize| match schema.field(index).data_type() {
        DataType::LargeList(field) => field.clone(),
        _ => unreachable!("known list field"),
    };
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from_iter_values(
            rows.iter().map(|row| row.object_id.as_str()),
        )),
        Arc::new(LargeStringArray::from_iter_values(
            rows.iter().map(|row| row.status.as_str()),
        )),
        Arc::new(LargeStringArray::from_iter(
            rows.iter().map(|row| row.error.as_deref()),
        )),
        string_lists(rows.iter().map(|row| row.filter_ids.clone()), list_field(3)),
        f64_lists(
            rows.iter().map(|row| row.absolute_magnitudes.clone()),
            list_field(4),
        ),
        f64_lists(
            rows.iter().map(|row| row.absolute_magnitude_sigmas.clone()),
            list_field(5),
        ),
        i64_lists(
            rows.iter().map(|row| row.num_used_by_filter.clone()),
            list_field(6),
        ),
        f64_lists(
            rows.iter().map(|row| row.magnitude_covariance.clone()),
            list_field(7),
        ),
        string_lists(
            rows.iter().map(|row| row.color_lhs_filter_ids.clone()),
            list_field(8),
        ),
        string_lists(
            rows.iter().map(|row| row.color_rhs_filter_ids.clone()),
            list_field(9),
        ),
        f64_lists(rows.iter().map(|row| row.colors.clone()), list_field(10)),
        f64_lists(
            rows.iter().map(|row| row.color_sigmas.clone()),
            list_field(11),
        ),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.phase_param).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.phase_param_sigma)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.chi2).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.reduced_chi2).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.dof).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.rank).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.num_params).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.converged).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.num_obs).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.num_invalid).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.num_unsupported_filter)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.num_clipped).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.num_used).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, arrays).map_err(|error| invalid(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use std::path::PathBuf;

    fn data() -> BandpassData {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/adam_core/photometry/bandpasses/data");
        BandpassData::load(&path).unwrap()
    }

    fn geometry(alpha_deg: f64) -> ([f64; 3], [f64; 3]) {
        let alpha = alpha_deg.to_radians();
        // Object at 2 AU and observer chosen so the object-centric Sun/observer
        // angle is alpha, while retaining finite positive distances.
        ([2.0, 0.0, 0.0], [2.0 - alpha.cos(), -alpha.sin(), 0.0])
    }

    fn observation(filter: &str, alpha: f64, reduced_mag: f64, sigma: f64) -> ColorObservation {
        let (object, observer) = geometry(alpha);
        let r = 2.0_f64;
        let delta = 1.0_f64;
        ColorObservation {
            magnitude: Some(reduced_mag + 5.0 * (r * delta).log10()),
            magnitude_sigma: Some(sigma),
            observatory_code: "X05".to_string(),
            reported_filter: Some(filter.to_string()),
            object_position: object,
            observer_position: observer,
        }
    }

    fn config(model: ColorPhaseModel) -> ColorEstimatorConfig {
        ColorEstimatorConfig {
            phase_model: model,
            force_phase_bounds: false,
            strict_band_mapping: false,
            projection: None,
        }
    }

    #[test]
    fn hg12star_knots_and_analytic_derivative_match_reference_values() {
        let (correction, derivative) = hg12star_correction(30.0, 0.15).unwrap();
        assert!((correction - 1.209_068_194_161_344_6).abs() < 1.0e-14);
        assert!((derivative - 0.179_365_364_389_740_57).abs() < 1.0e-14);
        let opposition = hg12star_correction(0.0, 0.15).unwrap();
        assert!(opposition.0.abs() < 1.0e-14);
        assert!(opposition.1.abs() < 1.0e-14);
        assert!(hg12star_correction(150.01, 0.15).is_err());
    }

    #[test]
    fn phase_models_recover_native_filter_intercepts_and_sorted_pairs() {
        let data = data();
        for model in [
            ColorPhaseModel::Hg,
            ColorPhaseModel::Hg12Star,
            ColorPhaseModel::C1C2,
        ] {
            let mut rows = Vec::new();
            for (filter, h) in [("ATLAS_o", 18.4), ("SDSS_z", 18.1), ("PS1_y", 18.0)] {
                for alpha in [5.0, 12.0, 25.0, 40.0] {
                    let correction = match model {
                        ColorPhaseModel::Hg => hg_correction(alpha, 0.24).0,
                        ColorPhaseModel::Hg12Star => hg12star_correction(alpha, 0.24).unwrap().0,
                        ColorPhaseModel::C1C2 => {
                            0.4 * alpha.to_radians() - 0.2 * alpha.to_radians().powi(2)
                        }
                    };
                    rows.push(observation(filter, alpha, h + correction, 0.03));
                }
            }
            let fit = estimate_color_object(&data, "x", &rows, &config(model));
            assert_eq!(fit.status, "ok", "{model:?}: {:?}", fit.error);
            assert_eq!(
                fit.filter_ids.as_ref().unwrap(),
                &["ATLAS_o", "PS1_y", "SDSS_z"]
            );
            assert_eq!(fit.num_used_by_filter.as_ref().unwrap(), &[4, 4, 4]);
            assert_eq!(fit.colors.as_ref().unwrap().len(), 3);
            assert_eq!(fit.num_obs, Some(12));
            assert_eq!(fit.num_used, Some(12));
            let magnitudes = fit.absolute_magnitudes.as_ref().unwrap();
            for (value, expected) in magnitudes.iter().zip([18.4, 18.0, 18.1]) {
                assert!(
                    (value - expected).abs() < 1.0e-6,
                    "{model:?}: {value} != {expected}"
                );
            }
        }
    }

    #[test]
    fn invalid_precedes_unsupported_and_strict_fallback_is_counted() {
        let data = data();
        let mut rows = vec![
            observation("g", 10.0, 18.0, 0.1),
            observation("not-a-filter", 15.0, 18.0, 0.1),
            observation("not-a-filter", 20.0, 18.0, 0.1),
        ];
        rows[2].magnitude = None;
        for row in &mut rows {
            row.observatory_code = "500".to_string();
        }
        let mut cfg = config(ColorPhaseModel::C1C2);
        cfg.strict_band_mapping = true;
        let fit = estimate_color_object(&data, "x", &rows, &cfg);
        assert_eq!(fit.status, "no_usable_photometry");
        assert_eq!(fit.num_invalid, Some(1));
        assert_eq!(fit.num_unsupported_filter, Some(2));
        assert_eq!(fit.num_clipped, Some(0));
        assert_eq!(fit.num_used, Some(0));
        assert_eq!(fit.num_obs, Some(3));
    }

    #[test]
    fn classified_geometry_failure_retains_exact_count_partition() {
        let data = data();
        let mut rows = vec![
            observation("SDSS_g", 10.0, 18.0, 0.1),
            observation("unknown", 10.0, 18.0, 0.1),
            observation("unknown", 10.0, 18.0, 0.1),
        ];
        rows[0].object_position[0] = f64::NAN;
        rows[2].magnitude = None;
        let mut cfg = config(ColorPhaseModel::C1C2);
        cfg.strict_band_mapping = true;
        let fit = estimate_color_object(&data, "x", &rows, &cfg);
        assert_eq!(fit.status, "failed");
        assert_eq!(fit.num_invalid, Some(1));
        assert_eq!(fit.num_unsupported_filter, Some(1));
        assert_eq!(fit.num_clipped, Some(0));
        assert_eq!(fit.num_used, Some(1));
        assert_eq!(fit.num_obs, Some(3));
    }

    #[test]
    fn projection_requires_explicit_identity_and_outputs_targets() {
        let data = data();
        let rows = vec![
            observation("SDSS_g", 5.0, 18.0, 0.05),
            observation("SDSS_g", 15.0, 18.1, 0.05),
            observation("LSST_r", 25.0, 17.7, 0.05),
            observation("LSST_r", 35.0, 17.8, 0.05),
        ];
        let mut mapping = BTreeMap::new();
        mapping.insert("SDSS_g".to_string(), "LSST_g".to_string());
        let mut cfg = config(ColorPhaseModel::C1C2);
        cfg.projection = Some(ColorProjection {
            source_to_target: mapping,
            composition: ColorComposition::Template("C".to_string()),
        });
        validate_color_config(&data, &cfg).unwrap();
        let fit = estimate_color_object(&data, "x", &rows, &cfg);
        assert_eq!(fit.filter_ids.as_ref().unwrap(), &["LSST_g"]);
        assert_eq!(fit.num_unsupported_filter, Some(2));
        assert_eq!(fit.num_used, Some(2));
    }

    #[test]
    fn rank_deficiency_masks_h_but_preserves_fitable_contrast() {
        let data = data();
        let rows = vec![
            observation("SDSS_g", 20.0, 18.1, 0.05),
            observation("SDSS_g", 20.0, 18.3, 0.05),
            observation("SDSS_r", 20.0, 17.5, 0.05),
            observation("SDSS_r", 20.0, 17.7, 0.05),
        ];
        let fit = estimate_color_object(&data, "x", &rows, &config(ColorPhaseModel::Hg));
        assert_eq!(fit.status, "ok");
        assert!(fit
            .absolute_magnitudes
            .as_ref()
            .unwrap()
            .iter()
            .all(|x| x.is_nan()));
        assert!((fit.colors.as_ref().unwrap()[0] - 0.6).abs() < 1.0e-10);
        assert!(fit.color_sigmas.as_ref().unwrap()[0].is_finite());
        let cov = fit.magnitude_covariance.as_ref().unwrap();
        assert!(cov.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn c1c2_rank_deficiency_preserves_exact_weighted_mean_contrast() {
        let data = data();
        let sigma = 0.05;
        let g = [18.00, 18.01, 17.99, 18.02, 17.98, 18.00];
        let r = [17.60, 17.59, 17.61, 17.58, 17.62, 17.60];
        let rows = g
            .into_iter()
            .map(|magnitude| observation("SDSS_g", 20.0, magnitude, sigma))
            .chain(
                r.into_iter()
                    .map(|magnitude| observation("SDSS_r", 20.0, magnitude, sigma)),
            )
            .collect::<Vec<_>>();
        let fit = estimate_color_object(&data, "x", &rows, &config(ColorPhaseModel::C1C2));
        assert_eq!(fit.status, "ok", "{fit:?}");
        assert_eq!(fit.rank, Some(2));
        assert!(fit
            .absolute_magnitudes
            .as_ref()
            .unwrap()
            .iter()
            .all(|value| value.is_nan()));
        let mean_g = g.iter().sum::<f64>() / g.len() as f64;
        let mean_r = r.iter().sum::<f64>() / r.len() as f64;
        let expected_color = mean_g - mean_r;
        let expected_chi2 = g
            .iter()
            .map(|value| ((value - mean_g) / sigma).powi(2))
            .chain(r.iter().map(|value| ((value - mean_r) / sigma).powi(2)))
            .sum::<f64>();
        assert!((fit.colors.as_ref().unwrap()[0] - expected_color).abs() < 1.0e-12);
        assert!((fit.chi2.unwrap() - expected_chi2).abs() < 1.0e-10);
        let expected_variance = expected_chi2 / 10.0
            * (sigma * sigma / g.len() as f64 + sigma * sigma / r.len() as f64);
        assert!(
            (fit.color_sigmas.as_ref().unwrap()[0].powi(2) - expected_variance).abs() < 1.0e-12
        );
    }

    #[test]
    fn c1c2_per_filter_single_angles_marks_the_color_non_fitable() {
        let data = data();
        let rows = [18.0, 18.1, 17.9]
            .map(|magnitude| observation("SDSS_g", 10.0, magnitude, 0.05))
            .into_iter()
            .chain([17.5, 17.6, 17.4].map(|magnitude| observation("SDSS_r", 30.0, magnitude, 0.05)))
            .collect::<Vec<_>>();
        let fit = estimate_color_object(&data, "x", &rows, &config(ColorPhaseModel::C1C2));
        assert_eq!(fit.status, "ok", "{fit:?}");
        assert!(fit.colors.as_ref().unwrap()[0].is_nan());
        assert!(fit.color_sigmas.as_ref().unwrap()[0].is_nan());
    }

    #[test]
    fn c1c2_mixed_phase_coverage_identifies_sparse_filter_contrast() {
        let data = data();
        let correction = |alpha: f64| {
            let radians = alpha.to_radians();
            0.4 * radians - 0.2 * radians * radians
        };
        let mut rows = [5.0, 20.0, 45.0]
            .map(|alpha| observation("SDSS_g", alpha, 18.2 + correction(alpha), 0.05))
            .to_vec();
        rows.push(observation("SDSS_r", 30.0, 17.7 + correction(30.0), 0.05));
        let fit = estimate_color_object(&data, "x", &rows, &config(ColorPhaseModel::C1C2));
        assert_eq!(fit.status, "ok", "{fit:?}");
        assert_eq!(fit.rank, Some(4));
        assert!((fit.colors.as_ref().unwrap()[0] - 0.5).abs() < 1.0e-10);
        assert!(fit.color_sigmas.as_ref().unwrap()[0].is_nan());
    }

    #[test]
    fn one_filter_has_present_empty_pair_lists() {
        let data = data();
        let rows = [5.0, 15.0, 25.0, 35.0]
            .map(|alpha| observation("ATLAS_c", alpha, 18.0 + 0.01 * alpha, 0.05));
        let fit = estimate_color_object(&data, "x", &rows, &config(ColorPhaseModel::C1C2));
        assert_eq!(fit.status, "ok");
        assert_eq!(fit.color_lhs_filter_ids, Some(Vec::new()));
        assert_eq!(fit.color_rhs_filter_ids, Some(Vec::new()));
        assert_eq!(fit.colors, Some(Vec::new()));
        assert_eq!(fit.color_sigmas, Some(Vec::new()));
    }

    #[test]
    fn arrow_schema_and_nonnullable_child_data_are_exact() {
        let data = data();
        let rows = [5.0, 15.0, 25.0, 35.0]
            .map(|alpha| observation("ATLAS_c", alpha, 18.0 + 0.01 * alpha, 0.05));
        let ok = estimate_color_object(&data, "x", &rows, &config(ColorPhaseModel::C1C2));
        let failed = ColorFitRow::terminal("y", 2, "no_orbit", "missing");
        let batch = color_fit_record_batch(&[ok, failed]).unwrap();
        assert_eq!(batch.schema(), color_fit_schema());
        assert_eq!(batch.num_columns(), 25);
        let batch_schema = batch.schema();
        for index in 3..=11 {
            let list = batch
                .column(index)
                .as_any()
                .downcast_ref::<LargeListArray>()
                .unwrap();
            let DataType::LargeList(item) = batch_schema.field(index).data_type() else {
                panic!("expected large_list")
            };
            assert!(!item.is_nullable());
            assert_eq!(list.values().null_count(), 0);
            assert!(list.is_valid(0));
            assert!(list.is_null(1));
        }
        // Schema-drift guard covers field order, list type and both levels of nullability.
        let expected_names = [
            "object_id",
            "status",
            "error",
            "filter_ids",
            "absolute_magnitudes",
            "absolute_magnitude_sigmas",
            "num_used_by_filter",
            "magnitude_covariance",
            "color_lhs_filter_ids",
            "color_rhs_filter_ids",
            "colors",
            "color_sigmas",
            "phase_param",
            "phase_param_sigma",
            "chi2",
            "reduced_chi2",
            "dof",
            "rank",
            "num_params",
            "converged",
            "num_obs",
            "num_invalid",
            "num_unsupported_filter",
            "num_clipped",
            "num_used",
        ];
        assert_eq!(
            batch
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            expected_names
        );
    }

    #[test]
    fn too_many_outliers_retains_exact_classified_count_partition() {
        let data = data();
        let mut rows = Vec::new();
        for _ in 0..8 {
            rows.push(observation("SDSS_g", 20.0, 18.0, 1.0));
        }
        for exponent in 0..22 {
            rows.push(observation(
                "SDSS_g",
                20.0,
                18.0 + 10.0_f64.powi(exponent),
                1.0,
            ));
        }
        let fit = estimate_color_object(&data, "x", &rows, &config(ColorPhaseModel::C1C2));
        assert_eq!(fit.status, "too_many_outliers", "{fit:?}");
        assert!(fit.filter_ids.is_none());
        assert!(fit.absolute_magnitudes.is_none());
        assert_eq!(
            fit.num_obs.unwrap(),
            fit.num_invalid.unwrap()
                + fit.num_unsupported_filter.unwrap()
                + fit.num_clipped.unwrap()
                + fit.num_used.unwrap()
        );
        assert_eq!(fit.num_used, Some(10));
        assert_eq!(fit.num_clipped, Some(20));
    }

    #[test]
    fn nonlinear_narrow_phase_range_uses_scipy_relative_convergence_policy() {
        let data = data();
        for model in [ColorPhaseModel::Hg, ColorPhaseModel::Hg12Star] {
            let mut rows = Vec::new();
            for (filter, h, offset) in [("SDSS_g", 18.2, 0.0), ("SDSS_r", 17.7, 0.003)] {
                for index in 0..8 {
                    let alpha = 10.0 + 0.001 * index as f64;
                    let correction = match model {
                        ColorPhaseModel::Hg => hg_correction(alpha, 0.25).0,
                        ColorPhaseModel::Hg12Star => hg12star_correction(alpha, 0.25).unwrap().0,
                        ColorPhaseModel::C1C2 => unreachable!(),
                    };
                    rows.push(observation(filter, alpha, h + correction + offset, 0.05));
                }
            }
            let fit = estimate_color_object(&data, "x", &rows, &config(model));
            assert_eq!(fit.status, "ok", "{model:?}: {fit:?}");
            assert_eq!(fit.converged, Some(true), "{model:?}: {fit:?}");
            assert!((fit.colors.as_ref().unwrap()[0] - 0.497).abs() < 1.0e-5);
        }
    }

    #[test]
    fn c1c2_near_rank_cutoff_matches_numpy_matrix_rank_semantics() {
        let make_rows = |spacing_deg: f64| {
            (0..8)
                .map(|index| {
                    let alpha_deg = 20.0 + spacing_deg * index as f64;
                    let alpha = alpha_deg.to_radians();
                    let filter_id = if index < 4 { "SDSS_g" } else { "SDSS_r" };
                    let intercept = if index < 4 { 18.0 } else { 17.6 };
                    EligibleRow {
                        reduced_magnitude: intercept + 0.4 * alpha - 0.2 * alpha * alpha
                            + 0.001 * (index as f64 - 3.5),
                        sigma: 0.05,
                        alpha_deg,
                        filter_id: filter_id.to_string(),
                    }
                })
                .collect::<Vec<_>>()
        };

        // Frozen NumPy 2.4 matrix_rank cases around
        // s_max * max(rows, columns) * eps: the third singular direction is
        // retained at 1e-13 degrees and discarded at 1e-14 degrees.
        let above = fit_eligible_rows(&make_rows(1.0e-13), ColorPhaseModel::C1C2).unwrap();
        let below = fit_eligible_rows(&make_rows(1.0e-14), ColorPhaseModel::C1C2).unwrap();
        assert_eq!(above.analysis.rank, 3);
        assert_eq!(below.analysis.rank, 2);
    }

    #[test]
    fn hg_runaway_matches_scipy_evaluation_budget_and_nonconvergence() {
        let rows = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0].map(|alpha_deg| EligibleRow {
            reduced_magnitude: 18.0 - 0.1 * alpha_deg,
            sigma: 0.05,
            alpha_deg,
            filter_id: "SDSS_g".to_string(),
        });
        let fit = fit_eligible_rows(&rows, ColorPhaseModel::Hg).unwrap();

        // scipy.optimize.least_squares (SciPy 1.17.1) exhausts its default
        // 100 * n evaluation budget for this runaway solution. Its frozen
        // reference is G=598754.2221, H=28.65634, chi2=23457.55049.
        assert!(!fit.converged);
        assert_eq!(fit.included.iter().filter(|&&keep| keep).count(), 8);
        assert_eq!(fit.analysis.rank, 2);
        assert!((fit.values[0] - 598_754.222_1).abs() < 1.0);
        assert!((fit.values[1] - 28.656_340).abs() < 1.0e-4);
        assert!((fit.chi2 - 23_457.550_49).abs() < 0.1);
    }

    #[test]
    fn forced_bounds_retain_only_filter_and_count_lists() {
        let data = data();
        let mut rows = Vec::new();
        for alpha in [5.0, 10.0, 20.0, 30.0, 40.0] {
            rows.push(observation(
                "SDSS_g",
                alpha,
                18.0 + hg_correction(alpha, 1.2).0,
                0.02,
            ));
        }
        let mut cfg = config(ColorPhaseModel::Hg);
        cfg.force_phase_bounds = true;
        let fit = estimate_color_object(&data, "x", &rows, &cfg);
        assert_eq!(fit.status, "phase_param_out_of_bounds");
        assert!(fit.filter_ids.is_some());
        assert!(fit.num_used_by_filter.is_some());
        assert!(fit.absolute_magnitudes.is_none());
        assert!(fit.colors.is_none());
        assert!(fit.phase_param.unwrap() > 1.0);
    }
}
