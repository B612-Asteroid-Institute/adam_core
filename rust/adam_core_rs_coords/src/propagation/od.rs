//! Backend-generic orbit-determination drivers (beads personal-cmy.7 and
//! personal-dqk).
//!
//! All drivers live in the permissive core, generic over the [`Propagator`]
//! trait: candidate states are predicted through the shared backend-generic
//! barycentric ephemeris workflow, so each iteration costs one base
//! prediction plus one batched multi-state Jacobian prediction. GPL backends
//! (the adam-assist equivalent) only plug in their `Propagator`
//! implementation; no OD logic lives behind the GPL boundary.
//!
//! Three public work units mirror adam-core's public OD surfaces:
//!
//! * [`fit_orbit_least_squares_barycentric`] — the Gauss-Newton fitter behind
//!   `ASSISTPropagator.fit_least_squares`.
//! * [`fit_orbit_least_squares_evaluated_barycentric`] — the fitter fused with
//!   the final `evaluate_orbits`-style residual/statistics pass, so
//!   `adam_core.orbit_determination.fit_least_squares` is one crossing.
//! * [`od_fit_barycentric`] — the full legacy `od()` differential-correction
//!   loop: delta bounding, finite/central perturbation batching, weighted
//!   normal equations, condition/covariance sanity rejections, acceptance
//!   bookkeeping, and chi2-ranked outlier retries.
//! * [`vallado_least_squares_barycentric`] — the public `LeastSquares`
//!   (Vallado RMS) algorithm with central/forward differences, perturbation
//!   backoff, rejected-update semantics, and the debug iteration trace.
//!
//! Linear algebra intentionally reuses the crate's pivoted Gauss-Jordan
//! solve/inverse rather than LAPACK, so results agree with the legacy
//! scipy/numpy paths to solver round-off, not bit-for-bit. Outlier ties in
//! `od_fit_barycentric` are broken by observation index (stable sort);
//! numpy's introsort tie order is unspecified.

use super::ephemeris::{generate_ephemeris_barycentric, EphemerisOptions};
use super::{PropagationError, PropagationResultValue, Propagator};
use crate::orbit_least_squares::{
    fit_orbit_least_squares_with_predictor, inverse_6x6, solve_6x6, LeastSquaresConfig,
    LeastSquaresFit,
};
use crate::translation::OriginTranslationProvider;
use crate::types::time::TimeScaleProvider;
use crate::types::Frame;
use crate::{
    bound_longitude_value, chi2_survival, compute_residuals_chi2_flat, CoordinateBatch,
    CoordinateValues, CovarianceBatch, ObjectId, ObserverBatch, OrbitBatch, OrbitId, OriginArray,
    TimeArray, Validity,
};

/// Legacy light-time failure phrase. adam-core's public `LeastSquares` class
/// treats provider errors containing this text as a rejected trial step; the
/// core drivers preserve that contract for per-row ephemeris failures.
pub const INVALID_LIGHT_TIME_MESSAGE: &str = "Light travel time is NaN or too large";

/// Differentially correct a single orbit against astrometric observations
/// using Gauss-Newton least squares, with predictions generated through the
/// backend-generic barycentric ephemeris workflow.
///
/// `orbit` must contain exactly one row (Cartesian, with an epoch);
/// `observed` are spherical coordinates with per-row covariance aligned
/// row-wise with `observers`. The fitted state and `inv(JᵀJ)` covariance are
/// expressed in the input orbit's frame/origin.
#[allow(clippy::too_many_arguments)]
pub fn fit_orbit_least_squares_barycentric<P, T>(
    propagator: &P,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
    config: &LeastSquaresConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<LeastSquaresFit>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    if orbit.len() != 1 {
        return Err(PropagationError::InvalidRequest(
            "least-squares OD corrects exactly one orbit".to_string(),
        ));
    }
    let n = observers.len();
    if observed.len() != n {
        return Err(PropagationError::InvalidRequest(format!(
            "observed rows ({}) must match observer rows ({n})",
            observed.len()
        )));
    }
    let observed_flat = spherical_flat(observed, "observed")?;
    let observed_cov = observed_covariance_flat(observed)?;
    let geometry = OrbitGeometry::from_orbit(orbit)?;

    let predict = |states: &[[f64; 6]]| -> Result<Vec<f64>, String> {
        match predict_spherical(
            propagator,
            states,
            &geometry,
            observers,
            options,
            provider,
            translation_provider,
        ) {
            Ok(Ok(values)) => Ok(values),
            Ok(Err(message)) => Err(message),
            Err(err) => Err(err.to_string()),
        }
    };

    fit_orbit_least_squares_with_predictor(
        geometry.state,
        &observed_flat,
        &observed_cov,
        n,
        config,
        predict,
    )
    .map_err(PropagationError::Backend)
}

/// Final `evaluate_orbits`-style statistics for one fitted state against the
/// full observation set, computed in the same crossing as the fit.
#[derive(Debug, Clone)]
pub struct FitEvaluation {
    /// `(N, 6)` row-major residual values (post longitude-wrap and cos-lat).
    pub residuals: Vec<f64>,
    /// `(N,)` per-observation chi².
    pub chi2: Vec<f64>,
    /// `(N,)` per-observation degrees of freedom.
    pub dof: Vec<i64>,
    /// `(N,)` chi-squared survival probability.
    pub probability: Vec<f64>,
    /// Orbit chi²: sum over included observations.
    pub orbit_chi2: f64,
    /// Orbit reduced chi²: `orbit_chi2 / (sum(dof) - parameters)`.
    pub reduced_chi2: f64,
    /// Arc length (days) across included observations.
    pub arc_length: f64,
    /// Number of included observations.
    pub num_obs: usize,
    /// `(N,)` outlier flags (`true` = ignored).
    pub outlier: Vec<bool>,
}

/// Fused fit + evaluation product for the public `fit_least_squares` veneer.
#[derive(Debug, Clone)]
pub struct EvaluatedLeastSquaresFit {
    pub fit: LeastSquaresFit,
    pub evaluation: FitEvaluation,
}

/// Evaluate one Cartesian state against the full observation set: one
/// ephemeris prediction plus the shared residual/chi²/probability kernel and
/// the `evaluate_orbits` ignore-mask statistics.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_orbit_barycentric<P, T>(
    propagator: &P,
    state: [f64; 6],
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    ignore: &[bool],
    parameters: i64,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<FitEvaluation>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    let n = observed.len();
    if ignore.len() != n || observers.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed, observers, and ignore must have equal length".to_string(),
        ));
    }
    let observed_flat = spherical_flat(observed, "observed")?;
    let observed_cov = observed_covariance_flat(observed)?;
    let times_mjd = observed_times_mjd(observed)?;
    let geometry = OrbitGeometry::from_orbit(orbit)?;
    let predicted = predict_spherical(
        propagator,
        &[state],
        &geometry,
        observers,
        options,
        provider,
        translation_provider,
    )?
    .map_err(PropagationError::Backend)?;
    let residuals = full_residuals(&observed_flat, &observed_cov, &predicted, n)?;
    let included: Vec<bool> = ignore.iter().map(|&flag| !flag).collect();
    let num_included = included.iter().filter(|&&keep| keep).count();
    if num_included == 0 {
        return Err(PropagationError::InvalidRequest(
            "zero-size array to reduction operation maximum which has no identity".to_string(),
        ));
    }
    let (arc_length, _, _) = masked_arc_length(&times_mjd, &included);
    let mut chi2_total = 0.0;
    let mut dof_total = 0_i64;
    for ((&keep, &chi2), &dof) in included
        .iter()
        .zip(residuals.chi2.iter())
        .zip(residuals.dof.iter())
    {
        if keep {
            chi2_total += chi2;
            dof_total += dof;
        }
    }
    let probability = survival_probabilities(&residuals.chi2, &residuals.dof);
    Ok(FitEvaluation {
        residuals: residuals.residuals,
        chi2: residuals.chi2,
        dof: residuals.dof,
        probability,
        orbit_chi2: chi2_total,
        reduced_chi2: chi2_total / (dof_total - parameters) as f64,
        arc_length,
        num_obs: num_included,
        outlier: ignore.to_vec(),
    })
}

/// One-crossing `fit_least_squares`: fit on the non-ignored subset, then the
/// final evaluation over the full observation set with the ignore mask.
#[allow(clippy::too_many_arguments)]
pub fn fit_orbit_least_squares_evaluated_barycentric<P, T>(
    propagator: &P,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    ignore: &[bool],
    options: &EphemerisOptions,
    config: &LeastSquaresConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<EvaluatedLeastSquaresFit>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    let n = observed.len();
    if ignore.len() != n {
        return Err(PropagationError::InvalidRequest(
            "ignore mask must match observation length".to_string(),
        ));
    }
    let keep: Vec<bool> = ignore.iter().map(|&flag| !flag).collect();
    let observed_fit = filter_coordinate_batch(observed, &keep)?;
    let observers_fit = filter_observer_batch(observers, &keep)?;
    let fit = fit_orbit_least_squares_barycentric(
        propagator,
        orbit,
        &observed_fit,
        &observers_fit,
        options,
        config,
        provider,
        translation_provider,
    )?;
    let evaluation = evaluate_orbit_barycentric(
        propagator,
        fit.state,
        orbit,
        observed,
        observers,
        ignore,
        6,
        options,
        provider,
        translation_provider,
    )?;
    Ok(EvaluatedLeastSquaresFit { fit, evaluation })
}

// ---------------------------------------------------------------------------
// Legacy `od()` differential-correction loop
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdMethod {
    Central,
    Finite,
}

#[derive(Debug, Clone, Copy)]
pub struct OdConfig {
    pub rchi2_threshold: f64,
    pub min_obs: usize,
    pub min_arc_length: f64,
    pub contamination_percentage: f64,
    pub delta: f64,
    pub max_iter: usize,
    pub method: OdMethod,
}

/// Full `od()` product for one orbit. When `found` is false the public
/// veneer emits empty `FittedOrbits`/`FittedOrbitMembers` tables.
#[derive(Debug, Clone)]
pub struct OdOutput {
    pub found: bool,
    pub state: [f64; 6],
    pub covariance: [f64; 36],
    pub arc_length: f64,
    pub num_obs: usize,
    pub chi2_total: f64,
    pub reduced_chi2: f64,
    pub iterations: usize,
    pub improved: bool,
    /// `(N, 6)` residual values for every observation (last accepted orbit).
    pub residuals: Vec<f64>,
    pub residual_chi2: Vec<f64>,
    pub residual_dof: Vec<i64>,
    pub residual_probability: Vec<f64>,
    /// `(N,)` outlier flags (`true` = removed as outlier).
    pub outlier: Vec<bool>,
}

impl OdOutput {
    fn not_found(n: usize, iterations: usize) -> Self {
        Self {
            found: false,
            state: [f64::NAN; 6],
            covariance: [f64::NAN; 36],
            arc_length: f64::NAN,
            num_obs: 0,
            chi2_total: f64::NAN,
            reduced_chi2: f64::NAN,
            iterations,
            improved: false,
            residuals: Vec::new(),
            residual_chi2: Vec::new(),
            residual_dof: vec![0; 0],
            residual_probability: Vec::new(),
            outlier: vec![false; n],
        }
    }
}

const OD_DELTA_INCREASE_FACTOR: f64 = 5.0;
const OD_DELTA_DECREASE_FACTOR: f64 = 100.0;

/// The legacy `adam_core.orbit_determination.od` differential-correction
/// loop for one orbit, executed entirely in Rust over the backend-generic
/// ephemeris workflow. Semantics mirror the pinned legacy implementation,
/// including its intentional quirks: the weighted normal-equation right-hand
/// side always uses the *initial* residual columns, and after outlier removal
/// the weight/RHS rows are the first `num_obs` unmasked rows while the
/// partials use mask-filtered rows.
#[allow(clippy::too_many_arguments)]
pub fn od_fit_barycentric<P, T>(
    propagator: &P,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &OdConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<OdOutput>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    let n_all = observed.len();
    if observers.len() != n_all {
        return Err(PropagationError::InvalidRequest(
            "observed and observers must have equal length".to_string(),
        ));
    }
    let geometry = OrbitGeometry::from_orbit(orbit)?;
    let observed_flat = spherical_flat(observed, "observed")?;
    let observed_cov = observed_covariance_flat(observed)?;
    let times_mjd = observed_times_mjd(observed)?;
    // sigmas[:, 1:3]: sqrt of covariance diagonal entries 1 and 2.
    let sigma_lon_lat: Vec<[f64; 2]> = (0..n_all)
        .map(|row| {
            [
                observed_cov[row * 36 + 7].sqrt(),
                observed_cov[row * 36 + 14].sqrt(),
            ]
        })
        .collect();

    let central = config.method == OdMethod::Central;
    let num_params = 6_usize;

    let mut converged = false;
    let mut improved = false;
    let mut solution_found = false;
    let mut processable = true;
    let mut first_solution = true;
    let mut iterations = 0_usize;

    if n_all < config.min_obs {
        processable = false;
    }
    if !processable {
        return Ok(OdOutput::not_found(n_all, iterations));
    }

    let max_outliers = ((n_all as f64) * (config.contamination_percentage / 100.0))
        .min((n_all - config.min_obs) as f64) as usize;
    let mut outliers_tried = 0_usize;

    // Initial residuals for the input orbit.
    let initial_predicted = predict_spherical(
        propagator,
        &[geometry.state],
        &geometry,
        observers,
        options,
        provider,
        translation_provider,
    )?
    .map_err(PropagationError::Backend)?;
    let initial_residuals =
        full_residuals(&observed_flat, &observed_cov, &initial_predicted, n_all)?;
    // Legacy quirk: the normal-equation RHS uses these initial residual
    // columns for the entire run.
    let residuals0_cols: Vec<[f64; 2]> = (0..n_all)
        .map(|row| {
            [
                initial_residuals.residuals[row * 6 + 1],
                initial_residuals.residuals[row * 6 + 2],
            ]
        })
        .collect();
    let chi2_initial = initial_residuals.chi2.clone();
    let chi2_total_initial: f64 = chi2_initial.iter().sum();
    let rchi2_initial = chi2_total_initial / (2 * n_all - 6) as f64;

    let mut current_state = geometry.state;
    let mut current_cov = [f64::NAN; 36];
    let mut current_residuals = initial_residuals.clone();
    let mut num_obs = n_all;
    let mut chi2_total_prev = chi2_total_initial;
    let mut rchi2_prev = rchi2_initial;
    let mut ids_mask = vec![true; n_all];
    let mut delta_prev = config.delta;
    let mut max_iter_i = config.max_iter;
    let max_iter_outliers = config.max_iter * (max_outliers + 1);

    while !converged && processable {
        iterations += 1;
        if iterations == max_iter_outliers + 1 {
            break;
        }
        if iterations == max_iter_i + 1 && (solution_found || max_outliers == outliers_tried) {
            break;
        }

        if delta_prev < 1e-14 {
            delta_prev *= OD_DELTA_INCREASE_FACTOR;
        } else if delta_prev > 1e-2 {
            delta_prev /= OD_DELTA_DECREASE_FACTOR;
        }

        // Nominal ephemeris (legacy computes it every iteration; the finite
        // branch consumes it).
        let nominal = predict_spherical(
            propagator,
            &[current_state],
            &geometry,
            observers,
            options,
            provider,
            translation_provider,
        )?
        .map_err(PropagationError::Backend)?;

        // Batched perturbed candidates: plus block, then minus block.
        let mut deltas_diag = [0.0_f64; 6];
        for (index, slot) in deltas_diag.iter_mut().enumerate() {
            *slot = current_state[index] * delta_prev;
        }
        let num_pert = if central { 12 } else { 6 };
        let mut pert_states = Vec::with_capacity(num_pert);
        for index in 0..num_params {
            let mut state = current_state;
            state[index] += deltas_diag[index];
            pert_states.push(state);
        }
        if central {
            for index in 0..num_params {
                let mut state = current_state;
                state[index] -= deltas_diag[index];
                pert_states.push(state);
            }
        }
        let perturbed = predict_spherical(
            propagator,
            &pert_states,
            &geometry,
            observers,
            options,
            provider,
            translation_provider,
        )?
        .map_err(PropagationError::Backend)?;

        // Partial derivatives: (num_obs, 2, 6) with mask-filtered rows.
        let masked_rows: Vec<usize> = (0..n_all).filter(|&row| ids_mask[row]).collect();
        let mut partials = vec![[[0.0_f64; 6]; 2]; num_obs];
        for index in 0..num_params {
            let plus = &perturbed[index * n_all * 6..(index + 1) * n_all * 6];
            let minus_storage;
            let minus: &[f64] = if central {
                minus_storage = &perturbed
                    [(num_params + index) * n_all * 6..(num_params + index + 1) * n_all * 6];
                minus_storage
            } else {
                &nominal
            };
            let delta_denom = if central {
                deltas_diag[index] * 2.0
            } else {
                deltas_diag[index]
            };
            let columns = residual_lon_lat_columns(plus, minus, n_all);
            for (slot, &row) in masked_rows.iter().enumerate().take(num_obs) {
                partials[slot][0][index] = columns[row * 2] / delta_denom;
                partials[slot][1][index] = columns[row * 2 + 1] / delta_denom;
            }
        }

        // Weighted normal equations. Legacy quirk: weights and RHS index the
        // first `num_obs` unmasked rows.
        let mut atwa = [[0.0_f64; 6]; 6];
        let mut atwb = [0.0_f64; 6];
        for row in 0..num_obs {
            let w_lon = 1.0 / (sigma_lon_lat[row][0] * sigma_lon_lat[row][0]);
            let w_lat = 1.0 / (sigma_lon_lat[row][1] * sigma_lon_lat[row][1]);
            for k in 0..6 {
                for l in 0..6 {
                    atwa[k][l] += partials[row][0][k] * w_lon * partials[row][0][l]
                        + partials[row][1][k] * w_lat * partials[row][1][l];
                }
                atwb[k] += partials[row][0][k] * w_lon * residuals0_cols[row][0]
                    + partials[row][1][k] * w_lat * residuals0_cols[row][1];
            }
        }

        let atwa_condition = symmetric_condition_number(&atwa);
        let atwb_condition = vector_condition_number(&atwb);
        if atwa_condition > 1e15 || atwb_condition > 1e15 {
            delta_prev /= OD_DELTA_DECREASE_FACTOR;
            continue;
        }
        let atwa_has_nan = atwa.iter().flatten().any(|value| value.is_nan());
        if atwa_has_nan || atwb.iter().any(|value| value.is_nan()) {
            delta_prev *= OD_DELTA_INCREASE_FACTOR;
            continue;
        }

        let Some(delta_state) = solve_6x6(&atwa, &atwb) else {
            delta_prev *= OD_DELTA_INCREASE_FACTOR;
            continue;
        };
        let Some(covariance_matrix) = inverse_6x6(&atwa) else {
            delta_prev *= OD_DELTA_INCREASE_FACTOR;
            continue;
        };
        let variances: Vec<f64> = (0..6)
            .map(|index| covariance_matrix[index * 6 + index])
            .collect();
        if variances
            .iter()
            .any(|&value| value <= 0.0 || value.is_nan())
        {
            delta_prev /= OD_DELTA_DECREASE_FACTOR;
            continue;
        }
        let r_sigma = (variances[0] + variances[1] + variances[2]).sqrt();
        let r_mag = (current_state[0] * current_state[0]
            + current_state[1] * current_state[1]
            + current_state[2] * current_state[2])
            .sqrt();
        if r_sigma / r_mag > 1.0 {
            delta_prev /= OD_DELTA_DECREASE_FACTOR;
            continue;
        }
        if covariance_matrix.iter().any(|value| value.is_nan()) {
            delta_prev *= OD_DELTA_INCREASE_FACTOR;
            continue;
        }

        let delta_position = (delta_state[0] * delta_state[0]
            + delta_state[1] * delta_state[1]
            + delta_state[2] * delta_state[2])
            .sqrt();
        if delta_position < 1e-16 {
            delta_prev *= OD_DELTA_DECREASE_FACTOR;
            continue;
        }
        if delta_position > 100.0 {
            delta_prev /= OD_DELTA_DECREASE_FACTOR;
            continue;
        }

        let mut state_iter = current_state;
        for index in 0..6 {
            state_iter[index] += delta_state[index];
        }
        let v_mag = (state_iter[3] * state_iter[3]
            + state_iter[4] * state_iter[4]
            + state_iter[5] * state_iter[5])
            .sqrt();
        if v_mag > 1.0 {
            delta_prev *= OD_DELTA_INCREASE_FACTOR;
            continue;
        }

        let predicted_iter = predict_spherical(
            propagator,
            &[state_iter],
            &geometry,
            observers,
            options,
            provider,
            translation_provider,
        )?
        .map_err(PropagationError::Backend)?;
        let residuals_iter = full_residuals(&observed_flat, &observed_cov, &predicted_iter, n_all)?;
        let chi2_total_iter: f64 = (0..n_all)
            .filter(|&row| ids_mask[row])
            .map(|row| residuals_iter.chi2[row])
            .sum();
        let rchi2_iter = chi2_total_iter / (2 * num_obs - num_params) as f64;
        let included = ids_mask.clone();
        let (arc_length, _, _) = masked_arc_length(&times_mjd, &included);

        if (rchi2_iter < rchi2_prev || first_solution) && arc_length >= config.min_arc_length {
            first_solution = false;
            current_state = state_iter;
            current_cov = covariance_matrix;
            current_residuals = residuals_iter;
            chi2_total_prev = chi2_total_iter;
            rchi2_prev = rchi2_iter;
            if rchi2_prev <= rchi2_initial {
                improved = true;
            }
            if rchi2_prev <= config.rchi2_threshold {
                solution_found = true;
                converged = true;
            }
        } else if max_outliers > 0
            && outliers_tried <= max_outliers
            && iterations > max_iter_i
            && !solution_found
        {
            // Reset to the input orbit and retry with the highest-chi2
            // observations removed (ranked on the initial chi2 array, per
            // the legacy reset-then-argsort order).
            current_state = geometry.state;
            current_cov = [f64::NAN; 36];
            current_residuals = initial_residuals.clone();
            chi2_total_prev = chi2_total_initial;
            rchi2_prev = rchi2_initial;
            delta_prev = config.delta;

            let mut order: Vec<usize> = (0..n_all).collect();
            order.sort_by(|&a, &b| chi2_initial[a].total_cmp(&chi2_initial[b]));
            let removed = outliers_tried + 1;
            let outlier_indices: Vec<usize> = order[n_all - removed..].to_vec();
            num_obs = n_all - outlier_indices.len();
            for (row, slot) in ids_mask.iter_mut().enumerate() {
                *slot = !outlier_indices.contains(&row);
            }
            let (arc_length_outlier, _, _) = masked_arc_length(&times_mjd, &ids_mask);
            outliers_tried += 1;
            if arc_length_outlier >= config.min_arc_length {
                max_iter_i = config.max_iter * (outliers_tried + 1);
            }
        }
    }

    if !solution_found || !processable || first_solution {
        return Ok(OdOutput::not_found(n_all, iterations));
    }

    let (arc_length_final, _, _) = masked_arc_length(&times_mjd, &ids_mask);
    let probability = survival_probabilities(&current_residuals.chi2, &current_residuals.dof);
    Ok(OdOutput {
        found: true,
        state: current_state,
        covariance: current_cov,
        arc_length: arc_length_final,
        num_obs,
        chi2_total: chi2_total_prev,
        reduced_chi2: rchi2_prev,
        iterations,
        improved,
        residuals: current_residuals.residuals,
        residual_chi2: current_residuals.chi2,
        residual_dof: current_residuals.dof,
        residual_probability: probability,
        outlier: (0..n_all).map(|row| !ids_mask[row]).collect(),
    })
}

// ---------------------------------------------------------------------------
// Public `LeastSquares` (Vallado RMS) algorithm
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct ValladoConfig {
    pub use_central_difference: bool,
    pub perturbation_initial_fraction: f64,
    pub perturbation_multiplier: f64,
    pub rms_epsilon: f64,
    pub max_iterations: usize,
}

/// One debug-trace record, mirroring the legacy `debug_info["iterations"]`
/// dict entries. The first record carries only `rchi2`, `rms`, and
/// `perturbation`.
#[derive(Debug, Clone)]
pub struct ValladoIteration {
    pub rchi2: f64,
    pub rms: f64,
    pub delta_rms: Option<f64>,
    pub converged: Option<bool>,
    pub perturbation: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValladoStatus {
    /// `orbit_prev` is an accepted update; `state`/`covariance` are valid.
    Updated,
    /// `orbit_prev` is still the caller's initial orbit.
    Initial,
    /// Legacy returns `None`: no improvement was found.
    NotImproved,
}

#[derive(Debug, Clone)]
pub struct ValladoResult {
    pub status: ValladoStatus,
    pub state: [f64; 6],
    pub covariance: [f64; 36],
    pub num_observations: usize,
    pub iterations: Vec<ValladoIteration>,
    pub corrections: Vec<[f64; 6]>,
    pub exit_message: Option<String>,
}

/// The public `LeastSquares.least_squares` Vallado RMS algorithm, executed
/// entirely in Rust over the backend-generic ephemeris workflow. Per-row
/// ephemeris failures are treated as the legacy invalid-light-time condition:
/// partials failures back off the perturbation fraction and rejected updates
/// record an error iteration, exactly like the legacy `ValueError` handling.
#[allow(clippy::too_many_arguments)]
pub fn vallado_least_squares_barycentric<P, T>(
    propagator: &P,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &ValladoConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<ValladoResult>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    let n = observed.len();
    if observers.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed and observers must have equal length".to_string(),
        ));
    }
    let geometry = OrbitGeometry::from_orbit(orbit)?;
    let observed_flat = spherical_flat(observed, "observed")?;
    let observed_cov = observed_covariance_flat(observed)?;
    // Weights: 1 / sigma^2 over lon/lat only.
    let mut weights = vec![[0.0_f64; 2]; n];
    for (row, slot) in weights.iter_mut().enumerate() {
        let sigma_lon = observed_cov[row * 36 + 7].sqrt();
        let sigma_lat = observed_cov[row * 36 + 14].sqrt();
        slot[0] = 1.0 / (sigma_lon * sigma_lon);
        slot[1] = 1.0 / (sigma_lat * sigma_lat);
        if slot[0].is_nan() || slot[1].is_nan() {
            return Err(PropagationError::InvalidRequest(
                "Weights have NaNs, check sigmas of observations".to_string(),
            ));
        }
    }

    let mut iteration_records = Vec::new();
    let mut corrections_trace: Vec<[f64; 6]> = Vec::new();
    let mut exit_message: Option<String> = None;

    let mut perturbation_fraction = config.perturbation_initial_fraction;
    let mut status = ValladoStatus::Initial;
    let mut current_state = geometry.state;
    let mut current_cov = [f64::NAN; 36];

    // Nominal ephemeris + residual columns for the initial orbit.
    let mut nominal = match predict_spherical(
        propagator,
        &[current_state],
        &geometry,
        observers,
        options,
        provider,
        translation_provider,
    )? {
        Ok(values) => values,
        Err(message) => return Err(PropagationError::Backend(message)),
    };
    let mut residual_cols = residual_lon_lat_columns(&observed_flat, &nominal, n);
    let mut rms_initial = weighted_rms(&residual_cols, &weights, n);
    iteration_records.push(ValladoIteration {
        rchi2: reduced_chi2_full(&observed_flat, &observed_cov, &nominal, n, 6)?,
        rms: rms_initial,
        delta_rms: None,
        converged: None,
        perturbation: perturbation_fraction,
        error: None,
    });

    let mut converged = false;
    let mut iteration = 0_usize;
    while !converged && iteration < config.max_iterations {
        if rms_initial < 1e-20 {
            exit_message = Some(format!(
                "RMS is zero ({rms_initial:?}) after {iteration} iterations"
            ));
            converged = true;
            break;
        }
        iteration += 1;

        // Perturbed candidate batching (6 or 12 states in one crossing).
        let mut d_per_param = [0.0_f64; 6];
        for (index, slot) in d_per_param.iter_mut().enumerate() {
            let mut di = current_state[index] * perturbation_fraction;
            if di.abs() < 1e-20 {
                di = if index < 3 { 1.0 } else { 0.01 } * perturbation_fraction;
            }
            *slot = di;
        }
        let num_pert = if config.use_central_difference { 12 } else { 6 };
        let mut pert_states = Vec::with_capacity(num_pert);
        for index in 0..6 {
            let mut state = current_state;
            state[index] += d_per_param[index];
            pert_states.push(state);
        }
        if config.use_central_difference {
            for index in 0..6 {
                let mut state = current_state;
                state[index] -= d_per_param[index];
                pert_states.push(state);
            }
        }
        let perturbed = match predict_spherical(
            propagator,
            &pert_states,
            &geometry,
            observers,
            options,
            provider,
            translation_provider,
        )? {
            Ok(values) => values,
            Err(_light_time) => {
                if perturbation_fraction > 1e-12 {
                    perturbation_fraction *= config.perturbation_multiplier;
                    continue;
                }
                exit_message = Some(
                    "Partials produced invalid light-time and perturbation is already tiny. Stopping now"
                        .to_string(),
                );
                if iteration > 1 {
                    return Ok(finish_vallado(
                        status,
                        current_state,
                        current_cov,
                        n,
                        iteration_records,
                        corrections_trace,
                        exit_message,
                    ));
                }
                return Ok(finish_vallado(
                    ValladoStatus::NotImproved,
                    current_state,
                    current_cov,
                    n,
                    iteration_records,
                    corrections_trace,
                    exit_message,
                ));
            }
        };

        // Jacobian (N, 2, 6).
        let mut partials = vec![[[0.0_f64; 6]; 2]; n];
        for index in 0..6 {
            let plus = &perturbed[index * n * 6..(index + 1) * n * 6];
            let columns;
            let denom;
            if config.use_central_difference {
                let minus = &perturbed[(6 + index) * n * 6..(6 + index + 1) * n * 6];
                columns = residual_lon_lat_columns(plus, minus, n);
                denom = 2.0 * d_per_param[index];
            } else {
                columns = residual_lon_lat_columns(plus, &nominal, n);
                denom = d_per_param[index];
            }
            for row in 0..n {
                partials[row][0][index] = columns[row * 2] / denom;
                partials[row][1][index] = columns[row * 2 + 1] / denom;
            }
        }

        // Weighted normal equations with the CURRENT residual columns.
        let mut atwa = [[0.0_f64; 6]; 6];
        let mut atwb = [0.0_f64; 6];
        for row in 0..n {
            for k in 0..6 {
                for l in 0..6 {
                    atwa[k][l] += partials[row][0][k] * weights[row][0] * partials[row][0][l]
                        + partials[row][1][k] * weights[row][1] * partials[row][1][l];
                }
                atwb[k] += partials[row][0][k] * weights[row][0] * residual_cols[row * 2]
                    + partials[row][1][k] * weights[row][1] * residual_cols[row * 2 + 1];
            }
        }
        let Some(atwa_inverse) = inverse_6x6(&atwa) else {
            return Err(PropagationError::Backend(
                "normal-equation matrix is singular".to_string(),
            ));
        };
        let mut corrections = [0.0_f64; 6];
        for k in 0..6 {
            for l in 0..6 {
                corrections[k] += atwa_inverse[k * 6 + l] * atwb[l];
            }
        }

        let mut updated_state = current_state;
        for index in 0..6 {
            updated_state[index] += corrections[index];
        }

        // `(nominal ephemeris, residual columns, rchi2)` for an evaluable
        // update; `None` marks the legacy rejected-update (invalid
        // light-time) trial step.
        let mut update: Option<(Vec<f64>, Vec<f64>, f64)> = None;
        let mut rms_updated = f64::INFINITY;
        match predict_spherical(
            propagator,
            &[updated_state],
            &geometry,
            observers,
            options,
            provider,
            translation_provider,
        )? {
            Ok(values) => {
                let columns = residual_lon_lat_columns(&observed_flat, &values, n);
                rms_updated = weighted_rms(&columns, &weights, n);
                let delta_rms = (rms_initial - rms_updated) / rms_initial;
                converged = delta_rms.abs() < config.rms_epsilon;
                let rchi2 = reduced_chi2_full(&observed_flat, &observed_cov, &values, n, 6)?;
                update = Some((values, columns, rchi2));
                iteration_records.push(ValladoIteration {
                    rchi2,
                    rms: rms_updated,
                    delta_rms: Some(delta_rms),
                    converged: Some(converged),
                    perturbation: perturbation_fraction,
                    error: None,
                });
            }
            Err(message) => {
                converged = false;
                iteration_records.push(ValladoIteration {
                    rchi2: f64::INFINITY,
                    rms: f64::INFINITY,
                    delta_rms: Some(f64::NEG_INFINITY),
                    converged: Some(false),
                    perturbation: perturbation_fraction,
                    error: Some(message),
                });
            }
        }
        corrections_trace.push(corrections);

        let mut accepted = false;
        if rms_updated < rms_initial {
            if let Some((values, columns, _)) = update {
                current_state = updated_state;
                current_cov = atwa_inverse;
                status = ValladoStatus::Updated;
                nominal = values;
                residual_cols = columns;
                rms_initial = rms_updated;
                accepted = true;
            }
        }
        if !accepted && !converged {
            if perturbation_fraction > 1e-12 {
                perturbation_fraction *= config.perturbation_multiplier;
                continue;
            }
            exit_message =
                Some("RMS is worse and perturbation is already tiny. Stopping now".to_string());
            if iteration > 1 {
                return Ok(finish_vallado(
                    status,
                    current_state,
                    current_cov,
                    n,
                    iteration_records,
                    corrections_trace,
                    exit_message,
                ));
            }
            return Ok(finish_vallado(
                ValladoStatus::NotImproved,
                current_state,
                current_cov,
                n,
                iteration_records,
                corrections_trace,
                exit_message,
            ));
        }
    }

    if converged {
        return Ok(finish_vallado(
            status,
            current_state,
            current_cov,
            n,
            iteration_records,
            corrections_trace,
            exit_message,
        ));
    }
    if iteration >= config.max_iterations {
        exit_message = Some(format!("Reached max iteration of {iteration}"));
        return Ok(finish_vallado(
            status,
            current_state,
            current_cov,
            n,
            iteration_records,
            corrections_trace,
            exit_message,
        ));
    }
    Ok(finish_vallado(
        ValladoStatus::NotImproved,
        current_state,
        current_cov,
        n,
        iteration_records,
        corrections_trace,
        exit_message,
    ))
}

#[allow(clippy::too_many_arguments)]
fn finish_vallado(
    status: ValladoStatus,
    state: [f64; 6],
    covariance: [f64; 36],
    num_observations: usize,
    iterations: Vec<ValladoIteration>,
    corrections: Vec<[f64; 6]>,
    exit_message: Option<String>,
) -> ValladoResult {
    ValladoResult {
        status,
        state,
        covariance,
        num_observations,
        iterations,
        corrections,
        exit_message,
    }
}

// ---------------------------------------------------------------------------
// Shared internals
// ---------------------------------------------------------------------------

/// Epoch/frame/origin context extracted once from the single input orbit.
struct OrbitGeometry {
    state: [f64; 6],
    epoch: crate::Epoch,
    scale: crate::TimeScale,
    frame: Frame,
    origin: crate::OriginId,
}

impl OrbitGeometry {
    fn from_orbit(orbit: &OrbitBatch) -> PropagationResultValue<Self> {
        if orbit.len() != 1 {
            return Err(PropagationError::InvalidRequest(
                "orbit determination corrects exactly one orbit".to_string(),
            ));
        }
        let state = orbit.coordinates.values.cartesian().ok_or_else(|| {
            PropagationError::InvalidRequest(
                "orbit determination requires Cartesian orbit coordinates".to_string(),
            )
        })?[0];
        let times = orbit.coordinates.times.as_ref().ok_or_else(|| {
            PropagationError::InvalidRequest(
                "orbit determination requires an orbit epoch".to_string(),
            )
        })?;
        Ok(Self {
            state,
            epoch: times.epochs[0],
            scale: times.scale,
            frame: orbit.coordinates.frame,
            origin: orbit.coordinates.origins.origins[0].clone(),
        })
    }
}

fn spherical_flat(batch: &CoordinateBatch, label: &str) -> PropagationResultValue<Vec<f64>> {
    let values = batch.values.spherical().ok_or_else(|| {
        PropagationError::InvalidRequest(format!(
            "{label} coordinates must be spherical for orbit determination"
        ))
    })?;
    Ok(values.iter().flat_map(|row| row.iter().copied()).collect())
}

fn observed_covariance_flat(observed: &CoordinateBatch) -> PropagationResultValue<Vec<f64>> {
    Ok(observed
        .covariance
        .as_ref()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "orbit determination requires observed covariance".to_string(),
            )
        })?
        .values_row_major
        .clone())
}

fn observed_times_mjd(observed: &CoordinateBatch) -> PropagationResultValue<Vec<f64>> {
    Ok(observed
        .times
        .as_ref()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "orbit determination requires observation times".to_string(),
            )
        })?
        .mjd_values())
}

/// Predict spherical topocentric coordinates for `states`, returned as
/// candidate-major row blocks in INPUT candidate order. Backends may emit
/// rows in their own public order (the ASSIST backend sorts by orbit id, so
/// `od-candidate-10` would precede `od-candidate-2`); rows are scattered back
/// through the diagnostics' input orbit/observer indices. Outer `Err` is a
/// request/setup failure; inner `Err` is a per-row numerical failure carrying
/// the legacy light-time message.
fn predict_spherical<P, T>(
    propagator: &P,
    states: &[[f64; 6]],
    geometry: &OrbitGeometry,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<Result<Vec<f64>, String>>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    let candidates = candidate_batch(states, geometry)?;
    let result = generate_ephemeris_barycentric(
        propagator,
        &candidates,
        observers,
        options,
        provider,
        translation_provider,
    )?;
    for row in 0..result.ephemeris.coordinates.len() {
        if !result.ephemeris.validity.is_valid(row) {
            return Ok(Err(format!(
                "{INVALID_LIGHT_TIME_MESSAGE} (ephemeris row {row} failed)"
            )));
        }
    }
    let values = result
        .ephemeris
        .coordinates
        .values
        .spherical()
        .ok_or_else(|| {
            PropagationError::Backend("ephemeris output was not spherical".to_string())
        })?;
    let n_obs = observers.len();
    let expected_rows = states.len() * n_obs;
    if values.len() != expected_rows || result.diagnostics.rows.len() != expected_rows {
        return Err(PropagationError::Backend(format!(
            "ephemeris returned {} rows for {} candidates x {} observers",
            values.len(),
            states.len(),
            n_obs
        )));
    }
    let mut out = vec![f64::NAN; expected_rows * 6];
    let mut filled = vec![false; expected_rows];
    for (row, diagnostic) in result.diagnostics.rows.iter().enumerate() {
        if diagnostic.input_orbit_index >= states.len() || diagnostic.observer_index >= n_obs {
            return Err(PropagationError::Backend(
                "ephemeris diagnostics do not index the candidate grid".to_string(),
            ));
        }
        let slot = diagnostic.input_orbit_index * n_obs + diagnostic.observer_index;
        if filled[slot] {
            return Err(PropagationError::Backend(
                "ephemeris diagnostics repeated a candidate grid slot".to_string(),
            ));
        }
        out[slot * 6..slot * 6 + 6].copy_from_slice(&values[row]);
        filled[slot] = true;
    }
    Ok(Ok(out))
}

fn candidate_batch(
    states: &[[f64; 6]],
    geometry: &OrbitGeometry,
) -> PropagationResultValue<OrbitBatch> {
    let coordinates = CoordinateBatch::cartesian(
        states.to_vec(),
        geometry.frame,
        OriginArray::repeat(geometry.origin.clone(), states.len()),
        Some(TimeArray::new(
            geometry.scale,
            vec![geometry.epoch; states.len()],
        )?),
        None,
    )?;
    OrbitBatch::new(
        // Zero-padded ids: backends honoring the public sorted-output
        // contract (the ASSIST propagator sorts blocks lexicographically by
        // orbit id) must return candidate blocks in input order, otherwise
        // Jacobian plus/minus blocks would be paired against the wrong
        // candidates once more than ten states are batched.
        (0..states.len())
            .map(|index| OrbitId(format!("od-candidate-{index:04}")))
            .collect(),
        vec![None::<ObjectId>; states.len()],
        coordinates,
    )
    .map_err(Into::into)
}

#[derive(Debug, Clone)]
struct FullResiduals {
    residuals: Vec<f64>,
    chi2: Vec<f64>,
    dof: Vec<i64>,
}

/// `Residuals.calculate` semantics for the full observation set: observed
/// covariance plus zeroed predicted covariance, spherical wrap and cos-lat.
fn full_residuals(
    observed_flat: &[f64],
    observed_cov: &[f64],
    predicted_flat: &[f64],
    n: usize,
) -> PropagationResultValue<FullResiduals> {
    let predicted_cov = vec![0.0_f64; n * 36];
    let output = compute_residuals_chi2_flat(
        observed_flat,
        predicted_flat,
        observed_cov,
        &predicted_cov,
        n,
        6,
        true,
    )
    .map_err(|err| PropagationError::Backend(format!("residual computation failed: {err:?}")))?;
    Ok(FullResiduals {
        residuals: output.residuals,
        chi2: output.chi2,
        dof: output.dof,
    })
}

fn survival_probabilities(chi2: &[f64], dof: &[i64]) -> Vec<f64> {
    chi2.iter()
        .zip(dof.iter())
        .map(|(&chi2, &dof)| {
            if chi2.is_nan() {
                f64::NAN
            } else {
                chi2_survival(chi2, dof as f64)
            }
        })
        .collect()
}

/// Reduced chi² of a full predicted set: `sum(chi2) / (sum(dof) - parameters)`.
fn reduced_chi2_full(
    observed_flat: &[f64],
    observed_cov: &[f64],
    predicted_flat: &[f64],
    n: usize,
    parameters: i64,
) -> PropagationResultValue<f64> {
    let residuals = full_residuals(observed_flat, observed_cov, predicted_flat, n)?;
    let chi2_total: f64 = residuals.chi2.iter().sum();
    let dof_total: i64 = residuals.dof.iter().sum();
    Ok(chi2_total / (dof_total - parameters) as f64)
}

/// RA/Dec residual columns with the legacy longitude wrap and cos(lat)
/// convention (`_spherical_residual_columns_from_values` /
/// `compute_residuals_ndarray(...)[:, 1:3]`). Returns `(n, 2)` row-major.
fn residual_lon_lat_columns(observed_flat: &[f64], predicted_flat: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * 2];
    for row in 0..n {
        let obs_lon = observed_flat[row * 6 + 1];
        let obs_lat = observed_flat[row * 6 + 2];
        let wrapped = bound_longitude_value(obs_lon, obs_lon - predicted_flat[row * 6 + 1]);
        out[row * 2] = wrapped * obs_lat.to_radians().cos();
        out[row * 2 + 1] = obs_lat - predicted_flat[row * 6 + 2];
    }
    out
}

/// Vallado RMS: `sqrt(sum(residuals^2 * weights) / (N * 2))`.
fn weighted_rms(residual_cols: &[f64], weights: &[[f64; 2]], n: usize) -> f64 {
    let mut total = 0.0_f64;
    for row in 0..n {
        total += residual_cols[row * 2] * residual_cols[row * 2] * weights[row][0];
        total += residual_cols[row * 2 + 1] * residual_cols[row * 2 + 1] * weights[row][1];
    }
    (total / (n as f64 * 2.0)).sqrt()
}

/// `(max - min, max, min)` of `values[mask]`.
fn masked_arc_length(values: &[f64], mask: &[bool]) -> (f64, f64, f64) {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for (value, &keep) in values.iter().zip(mask.iter()) {
        if keep {
            minimum = minimum.min(*value);
            maximum = maximum.max(*value);
        }
    }
    (maximum - minimum, maximum, minimum)
}

/// 2-norm condition number of a symmetric 6x6 matrix via cyclic Jacobi
/// eigenvalues (singular values of a symmetric matrix are |eigenvalues|).
/// Non-finite input yields NaN, mirroring `np.linalg.cond` failing over to
/// the legacy NaN branch.
#[allow(clippy::needless_range_loop)]
fn symmetric_condition_number(a: &[[f64; 6]; 6]) -> f64 {
    if a.iter().flatten().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    let mut m = *a;
    for _sweep in 0..64 {
        let mut off_diagonal = 0.0_f64;
        for i in 0..6 {
            for j in (i + 1)..6 {
                off_diagonal += m[i][j] * m[i][j];
            }
        }
        if off_diagonal.sqrt() < 1e-300 {
            break;
        }
        for p in 0..6 {
            for q in (p + 1)..6 {
                if m[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..6 {
                    let mkp = m[k][p];
                    let mkq = m[k][q];
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for k in 0..6 {
                    let mpk = m[p][k];
                    let mqk = m[q][k];
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
            }
        }
    }
    let mut magnitudes: Vec<f64> = (0..6).map(|index| m[index][index].abs()).collect();
    magnitudes.sort_by(f64::total_cmp);
    magnitudes[5] / magnitudes[0]
}

/// `np.linalg.cond` of an `(6, 1)` matrix: the single singular value is the
/// vector norm, so the ratio is 1 unless the vector is zero (NaN).
fn vector_condition_number(b: &[f64; 6]) -> f64 {
    if b.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    let norm = b.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm == 0.0 {
        f64::NAN
    } else {
        1.0
    }
}

fn filter_coordinate_batch(
    batch: &CoordinateBatch,
    keep: &[bool],
) -> PropagationResultValue<CoordinateBatch> {
    let rows: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter_map(|(index, &flag)| flag.then_some(index))
        .collect();
    let raw = batch.values.raw_values();
    let filtered_rows: Vec<[f64; 6]> = rows.iter().map(|&index| raw[index]).collect();
    let values = match batch.values {
        CoordinateValues::Cartesian(_) => CoordinateValues::Cartesian(filtered_rows),
        CoordinateValues::Spherical(_) => CoordinateValues::Spherical(filtered_rows),
        CoordinateValues::Keplerian(_) => CoordinateValues::Keplerian(filtered_rows),
        CoordinateValues::Cometary(_) => CoordinateValues::Cometary(filtered_rows),
        CoordinateValues::Geodetic(_) => CoordinateValues::Geodetic(filtered_rows),
    };
    let origins = OriginArray::new(
        rows.iter()
            .map(|&index| batch.origins.origins[index].clone())
            .collect(),
    );
    let times = match &batch.times {
        Some(times) => Some(TimeArray::new(
            times.scale,
            rows.iter().map(|&index| times.epochs[index]).collect(),
        )?),
        None => None,
    };
    let covariance = match &batch.covariance {
        Some(covariance) => {
            let stride = covariance.dimension * covariance.dimension;
            let mut values = Vec::with_capacity(rows.len() * stride);
            for &index in &rows {
                values.extend_from_slice(
                    &covariance.values_row_major[index * stride..(index + 1) * stride],
                );
            }
            let mut filtered = CovarianceBatch::new(
                rows.len(),
                covariance.dimension,
                values,
                covariance.units.clone(),
            )?;
            if let Some(validity) = &covariance.row_validity {
                let bools: Vec<bool> = rows.iter().map(|&index| validity.is_valid(index)).collect();
                filtered = filtered.with_row_validity(Validity::from_bools(&bools))?;
            }
            Some(filtered)
        }
        None => None,
    };
    CoordinateBatch::new(values, batch.frame, origins, times, covariance).map_err(Into::into)
}

fn filter_observer_batch(
    observers: &ObserverBatch,
    keep: &[bool],
) -> PropagationResultValue<ObserverBatch> {
    let codes = observers
        .code
        .iter()
        .zip(keep.iter())
        .filter(|(_, &flag)| flag)
        .map(|(code, _)| code.clone())
        .collect();
    let coordinates = filter_coordinate_batch(&observers.coordinates, keep)?;
    ObserverBatch::new(codes, coordinates).map_err(Into::into)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::propagation::{
        CovariancePropagation, EphemerisOptions, EpochPolicy, PropagationOptions, TwoBodyPropagator,
    };
    use crate::types::{Frame, SchemaError, SchemaResult};
    use crate::{CovarianceUnits, Epoch, ObservatoryCode, OriginId, TimeScale};

    struct NoopProvider;

    impl TimeScaleProvider for NoopProvider {
        fn rescale(&self, _times: &TimeArray, _new_scale: TimeScale) -> SchemaResult<TimeArray> {
            Err(SchemaError::InvalidRecordBatch(
                "test provider should not be called".to_string(),
            ))
        }
    }

    struct ZeroTranslationProvider;

    impl OriginTranslationProvider for ZeroTranslationProvider {
        fn origin_translation_vectors(
            &self,
            origins: &OriginArray,
            _target_origin: &OriginId,
            _frame: Frame,
            _times: &TimeArray,
        ) -> SchemaResult<Vec<[f64; 6]>> {
            Ok(vec![[0.0; 6]; origins.len()])
        }
    }

    const TRUTH_STATE: [f64; 6] = [1.2, 0.1, 0.05, -0.002, 0.016, 0.001];
    const NUM_OBS: usize = 8;

    fn ephemeris_options() -> EphemerisOptions {
        EphemerisOptions {
            propagation: PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
            output_time_scale: TimeScale::Tdb,
            ..EphemerisOptions::default()
        }
    }

    fn truth_orbit(state: [f64; 6]) -> OrbitBatch {
        let coordinates = CoordinateBatch::cartesian(
            vec![state],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap()),
            None,
        )
        .unwrap();
        OrbitBatch::new(
            vec![OrbitId("fit-orbit".to_string())],
            vec![Some(ObjectId("fit-orbit".to_string()))],
            coordinates,
        )
        .unwrap()
    }

    fn observers() -> ObserverBatch {
        let times = TimeArray::new(
            TimeScale::Tdb,
            (0..NUM_OBS)
                .map(|row| Epoch::new(60_002 + 3 * row as i64, 0))
                .collect(),
        )
        .unwrap();
        let states: Vec<[f64; 6]> = (0..NUM_OBS)
            .map(|row| {
                let theta = 0.35 + 0.05 * row as f64;
                [
                    theta.cos(),
                    theta.sin(),
                    0.0,
                    -0.0172 * theta.sin(),
                    0.0172 * theta.cos(),
                    0.0,
                ]
            })
            .collect();
        ObserverBatch::new(
            (0..NUM_OBS)
                .map(|row| ObservatoryCode(format!("T{row:02}")))
                .collect(),
            CoordinateBatch::cartesian(
                states,
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::Named("SUN".to_string()), NUM_OBS),
                Some(times),
                None,
            )
            .unwrap(),
        )
        .unwrap()
    }

    /// Noise-free spherical observations synthesized from the truth orbit.
    fn synthetic_observations() -> CoordinateBatch {
        let orbit = truth_orbit(TRUTH_STATE);
        let observers = observers();
        let result = generate_ephemeris_barycentric(
            &TwoBodyPropagator::default(),
            &orbit,
            &observers,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let predicted = result.ephemeris.coordinates;
        let sigma = 1.0 / 3600.0;
        let mut covariance = vec![0.0_f64; NUM_OBS * 36];
        for row in 0..NUM_OBS {
            covariance[row * 36] = 1.0;
            covariance[row * 36 + 7] = sigma * sigma;
            covariance[row * 36 + 14] = sigma * sigma;
            covariance[row * 36 + 21] = 1.0;
            covariance[row * 36 + 28] = 1.0;
            covariance[row * 36 + 35] = 1.0;
        }
        CoordinateBatch::new(
            predicted.values.clone(),
            predicted.frame,
            predicted.origins.clone(),
            predicted.times.clone(),
            Some(
                CovarianceBatch::new(
                    NUM_OBS,
                    6,
                    covariance,
                    CovarianceUnits::Coordinate(crate::CoordinateRepresentation::Spherical),
                )
                .unwrap(),
            ),
        )
        .unwrap()
    }

    fn perturbed_start() -> OrbitBatch {
        let mut state = TRUTH_STATE;
        state[0] += 1e-4;
        state[1] -= 1e-4;
        state[2] += 5e-5;
        state[3] += 1e-6;
        state[4] -= 1e-6;
        state[5] += 1e-6;
        truth_orbit(state)
    }

    #[test]
    fn fit_evaluated_recovers_truth_and_reports_statistics() {
        let observed = synthetic_observations();
        let observers = observers();
        let ignore = vec![false; NUM_OBS];
        let output = fit_orbit_least_squares_evaluated_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers,
            &ignore,
            &ephemeris_options(),
            &LeastSquaresConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(output.fit.converged);
        // Gauss-Newton finite-difference floor for this synthetic geometry.
        for index in 0..6 {
            assert!((output.fit.state[index] - TRUTH_STATE[index]).abs() < 5e-5);
        }
        assert!(output.evaluation.orbit_chi2 < 1e-2);
        assert_eq!(output.evaluation.num_obs, NUM_OBS);
        assert_eq!(output.evaluation.outlier, vec![false; NUM_OBS]);
        assert!((output.evaluation.arc_length - 21.0).abs() < 1e-9);
        let manual_chi2: f64 = output.evaluation.chi2.iter().sum();
        assert!((output.evaluation.orbit_chi2 - manual_chi2).abs() < 1e-12);
        assert!(output.evaluation.reduced_chi2 < 1.0);
        assert_eq!(output.evaluation.dof, vec![6; NUM_OBS]);
    }

    #[test]
    fn fit_evaluated_honors_ignore_mask() {
        let observed = synthetic_observations();
        let observers = observers();
        let mut ignore = vec![false; NUM_OBS];
        ignore[1] = true;
        ignore[6] = true;
        let output = fit_orbit_least_squares_evaluated_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers,
            &ignore,
            &ephemeris_options(),
            &LeastSquaresConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(output.evaluation.num_obs, NUM_OBS - 2);
        assert_eq!(output.evaluation.outlier, ignore);
        assert_eq!(output.evaluation.chi2.len(), NUM_OBS);
        let included_chi2: f64 = output
            .evaluation
            .chi2
            .iter()
            .zip(ignore.iter())
            .filter_map(|(&chi2, &flag)| (!flag).then_some(chi2))
            .sum();
        assert!((output.evaluation.orbit_chi2 - included_chi2).abs() < 1e-12);
    }

    #[test]
    fn od_fit_finds_solution_and_improves() {
        let observed = synthetic_observations();
        let observers = observers();
        let config = OdConfig {
            rchi2_threshold: 10.0,
            min_obs: 5,
            min_arc_length: 1.0,
            contamination_percentage: 0.0,
            delta: 1e-6,
            max_iter: 20,
            method: OdMethod::Central,
        };
        let output = od_fit_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers,
            &config,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(output.found);
        assert!(output.improved);
        assert!(output.reduced_chi2 <= config.rchi2_threshold);
        assert_eq!(output.num_obs, NUM_OBS);
        assert_eq!(output.outlier, vec![false; NUM_OBS]);
        assert!((output.arc_length - 21.0).abs() < 1e-9);
        assert!(output.iterations >= 1);
        assert_eq!(output.residual_chi2.len(), NUM_OBS);
        assert_eq!(output.residual_probability.len(), NUM_OBS);
        for index in 0..3 {
            assert!((output.state[index] - TRUTH_STATE[index]).abs() < 1e-4);
        }
        assert!(output.covariance.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn od_fit_below_min_obs_is_not_found() {
        let observed = synthetic_observations();
        let observers = observers();
        let config = OdConfig {
            rchi2_threshold: 10.0,
            min_obs: NUM_OBS + 1,
            min_arc_length: 1.0,
            contamination_percentage: 0.0,
            delta: 1e-6,
            max_iter: 20,
            method: OdMethod::Central,
        };
        let output = od_fit_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers,
            &config,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(!output.found);
        assert_eq!(output.iterations, 0);
        assert!(output.residual_chi2.is_empty());
    }

    #[test]
    fn vallado_zero_rms_returns_initial_orbit() {
        let observed = synthetic_observations();
        let observers = observers();
        let config = ValladoConfig {
            use_central_difference: true,
            perturbation_initial_fraction: 1e-6,
            perturbation_multiplier: 0.5,
            rms_epsilon: 1e-3,
            max_iterations: 20,
        };
        let output = vallado_least_squares_barycentric(
            &TwoBodyPropagator::default(),
            &truth_orbit(TRUTH_STATE),
            &observed,
            &observers,
            &config,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(output.status, ValladoStatus::Initial);
        assert_eq!(output.iterations.len(), 1);
        assert!(output
            .exit_message
            .as_deref()
            .unwrap()
            .starts_with("RMS is zero"));
        assert_eq!(output.num_observations, NUM_OBS);
    }

    #[test]
    fn vallado_improves_perturbed_orbit() {
        let observed = synthetic_observations();
        let observers = observers();
        let config = ValladoConfig {
            use_central_difference: false,
            perturbation_initial_fraction: 1e-6,
            perturbation_multiplier: 0.5,
            rms_epsilon: 1e-3,
            max_iterations: 20,
        };
        let output = vallado_least_squares_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers,
            &config,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(output.status, ValladoStatus::Updated);
        assert!(output.iterations.len() > 1);
        let first_rms = output.iterations[0].rms;
        let last_rms = output.iterations.last().unwrap().rms;
        assert!(last_rms < first_rms);
        assert_eq!(output.corrections.len(), output.iterations.len() - 1);
        for index in 0..3 {
            assert!((output.state[index] - TRUTH_STATE[index]).abs() < 1e-5);
        }
        assert!(output.covariance.iter().all(|value| value.is_finite()));
    }
}
