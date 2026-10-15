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
//!
//! # IOD policy
//!
//! The default IOD stage ([`IodConfig::default`], `run_iod` and
//! `NativeOrbitFitter` in Python) follows the practice of OrbFit, Find_Orb
//! and JPL: a Gauss preliminary orbit is a seed for differential
//! correction, not the answer (Milani & Gronchi 2010, ch. 7–9). Every
//! number below is a `Default` on its config type, which documents its
//! source; the Python signature defaults are read from these (`IOD_DEFAULTS`).
//!
//! * **Ordering.** Observations are searched in a canonical order (epoch,
//!   then station code, then input row), whatever order the caller supplies;
//!   every output array keeps the caller's order. Identical inputs give
//!   identical results.
//! * **Inputs.** Gauss takes equatorial directions and heliocentric
//!   ecliptic observer positions; ecliptic observations and equatorial
//!   observers are rotated, observers not centred on the Sun are refused.
//!   An arc shorter than `min_arc_length` (1 day) is not searched, and a
//!   whole-arc fit whose rejection leaves less is not accepted.
//! * **Duplicate epochs.** A triplet never holds two observations at the
//!   same epoch (Gauss needs three distinct times).
//! * **Nights.** A station's observations form one observing night until a
//!   gap of more than 8 hours or 16 hours of continuous coverage
//!   ([`observation_nights`]); the UTC calendar day would split nights that
//!   straddle 0h UTC.
//! * **Triplets** ([`SpanSearchConfig`]). Gauss truncates the f and g
//!   series, so a preliminary orbit degrades with the fraction of the orbit
//!   its triplet spans (Milani & Gronchi 2010, sec. 9.1; full-arc reduced
//!   chi2 ~1e4 at a 3-day span against ~1e9 at 67 days on 2012 TF79). The
//!   search visits windows (densest 10 d, densest 30 d, last 30 d, first
//!   30 d, whole arc) and in each walks (first, last) pairs by distance of
//!   their span from 6 days, one triplet per pair and per pair of nights,
//!   lazily, up to 32 triplets.
//! * **Velocity** ([`VelocityMethod`]). `Auto` by default:
//!   Herrick-Gibbs when consecutive position vectors are within 5 degrees
//!   (Vallado 2013, sec. 7.5.2), Gibbs otherwise.
//! * **Multiple roots.** Every physical root of a triplet's degree-8
//!   polynomial is a candidate (up to three genuine solutions: Charlier;
//!   Gronchi 2009): real to a relative 1e-6 (so a near-double root is kept),
//!   de-duplicated, with all three topocentric ranges at least 1e-4 au (the
//!   spurious zero-range root and negative ranges are dropped). When one
//!   converges, its sibling roots are corrected too.
//! * **Acceptance** ([`IodAcceptance::Corrected`], [`CorrectionConfig`]).
//!   Candidates are corrected in increasing prefit reduced chi2, each raw
//!   Gauss orbit scored on the observations of its first stage
//!   ([`PrefitScope::Local`]: how good it is where it was built, not how far
//!   it extrapolates over a long arc), at most 8, until 2 reach the whole
//!   arc. Each is corrected in stages from its
//!   triplet's span (at least 6 observations), growing the span fourfold or
//!   jumping an apparition gap to the nearest night, each stage within
//!   `rchi2_threshold` (200, the legacy IOD gate and the ladder's first).
//!   Stages before the whole arc reject nothing
//!   and down-weight with Huber's loss (gated on the robust cost); the
//!   whole-arc stage is linear, with CMC2003 rejection (Carpino, Milani &
//!   Chesley 2003) when above the threshold, rejecting at most
//!   `contamination_percentage` of the arc (20 %; OrbFit's 50 % bound is for
//!   refining a known orbit, and let a wrong root fit by discarding half of
//!   2002 RV112's arc).
//! * **Ranking.** Lowest whole-arc score: every observation's chi2 clipped
//!   at CMC2003's rejection threshold (8), over the whole arc's degrees of
//!   freedom, so candidates that kept different observations are compared on
//!   the same data. Values within a relative 1e-6 tie and go to the lowest
//!   candidate index. A converged solution outside the accepted one's
//!   3-sigma region yet within the same score margin (6-dof 99.73 %
//!   quantile, 20.06) is reported in `IodResult::ambiguous_with`.
//! * **Continuation and failure.** A candidate whose prediction fails is
//!   counted and skipped; caller errors propagate. When nothing is accepted
//!   the result says why (`not_found_reason`), naming the last stall.
//! * **Evidence.** On the first solutions of 126 walk-forward objects
//!   (ASSIST, MPC observations, 2026-10-14) this policy seeded 116 of 122
//!   against 16 for the legacy `combinations` + prefit gate and 114 for the
//!   study's seeding ladder, with the ladder's final reduced chi2 (median
//!   ratio 1.000), median 1.9 s per object; single-object reruns since
//!   recover 433 Eros (Huber stages) and 3200 Phaethon (central-difference
//!   continuation, 1,160 s). The legacy policy stays selectable
//!   (`observation_selection_method="combinations"`, `acceptance="prefit"`).

use super::ephemeris::{generate_ephemeris_barycentric, EphemerisOptions};
use super::{PropagationError, PropagationResultValue, Propagator};
use crate::differential_correction::LossType;
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
use adam_core_rs_orbit_determination::{gauss_iod_candidates, VELOCITY_METHOD_AUTO};

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
    let predictor = PropagatorPredictor {
        propagator,
        options,
        provider,
        translation_provider,
    };
    evaluate_orbit(
        &predictor, state, orbit, observed, observers, ignore, parameters,
    )
}

/// [`evaluate_orbit_barycentric`] over any [`SphericalPredictor`].
pub fn evaluate_orbit(
    predictor: &dyn SphericalPredictor,
    state: [f64; 6],
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    ignore: &[bool],
    parameters: i64,
) -> PropagationResultValue<FitEvaluation> {
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
    let predicted = predictor
        .predict_spherical(&[state], &geometry, observers)?
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
// Legacy Gauss initial-orbit-determination acceptance loop
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationSelectionMethod {
    Combinations,
    FirstMiddleLast,
    Thirds,
    /// Span-bounded triplets swept across search windows (see
    /// [`SpanSearchConfig`]).
    Spans,
}

impl ObservationSelectionMethod {
    /// The Python spelling of ``observation_selection_method``.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "combinations" => Ok(Self::Combinations),
            "first+middle+last" => Ok(Self::FirstMiddleLast),
            "thirds" => Ok(Self::Thirds),
            "spans" => Ok(Self::Spans),
            other => Err(format!(
                "observation_selection_method must be one of 'combinations', \
                 'first+middle+last', 'thirds', 'spans'; got {other:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Combinations => "combinations",
            Self::FirstMiddleLast => "first+middle+last",
            Self::Thirds => "thirds",
            Self::Spans => "spans",
        }
    }
}

/// Heliocentric gravitational parameter (au³/day²) of [`IodConfig::default`]:
/// adam-core's `Constants.MU`.
pub const IOD_DEFAULT_MU: f64 = 0.000_295_912_208_284_119_56;
/// Speed of light (au/day) of [`IodConfig::default`]: adam-core's
/// `Constants.C`.
pub const IOD_DEFAULT_SPEED_OF_LIGHT: f64 = 173.144_632_674_240_34;

/// Settings of the span-bounded triplet search
/// ([`ObservationSelectionMethod::Spans`]).
///
/// Gauss's method truncates the f and g series, so a preliminary orbit
/// degrades as the triplet spans a larger fraction of the orbit; on NEO
/// arcs its two-body fit to the whole arc worsened from reduced chi2 ~1e4
/// at a 3-day triplet span to ~1e9 at 67 days (2012 TF79, sweep of
/// 2026-07-28), while longest-arc-first is the THOR tracklet-linking
/// heuristic (Moeyens et al. 2021) the legacy `combinations` order ports.
/// The search therefore visits windows of the arc in turn (the densest
/// `dense_window_days[0]` days, the densest `dense_window_days[1]` days,
/// the last and the first `edge_window_days`, then the whole arc: the
/// seeding windows of the 2026-10-12 walk-forward), and within each window
/// walks (first, last) pairs by distance of their span from
/// `target_span_days`, one triplet per pair with the middle observation
/// nearest the pair's midpoint. Validated on the walk-forward first
/// solutions (see the module's IOD policy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpanSearchConfig {
    /// Preferred triplet span (days): the middle of the 3 / 6 / 12-day
    /// spans of the 2026-07-28 sweep.
    pub target_span_days: f64,
    /// Pairs spanning less than this are never used (0 = any two distinct
    /// epochs).
    pub min_span_days: f64,
    /// Pairs spanning more than this are never used (infinite = no bound).
    pub max_span_days: f64,
    /// Lengths of the two densest-window searches, in this order.
    pub dense_window_days: [f64; 2],
    /// Length of the last- and first-window searches.
    pub edge_window_days: f64,
    /// Triplet budget when `IodConfig::max_triplets` is None: the order of
    /// the study ladder's candidate count (two triplets x two selections x
    /// up to four windows, times the roots).
    pub max_triplets: usize,
}

impl Default for SpanSearchConfig {
    fn default() -> Self {
        Self {
            target_span_days: 6.0,
            min_span_days: 0.0,
            max_span_days: f64::INFINITY,
            dense_window_days: [10.0, 30.0],
            edge_window_days: 30.0,
            max_triplets: 32,
        }
    }
}

/// How an IOD candidate is accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IodAcceptance {
    /// The legacy gate: the raw Gauss orbit's two-body fit to every
    /// observation (after the outlier trial) is within `rchi2_threshold`;
    /// the first such candidate wins.
    Prefit,
    /// The Gauss orbit only seeds differential corrections (Milani &
    /// Gronchi 2010; the practice of OrbFit, Find_Orb and Aegis):
    /// each candidate is corrected on the observations spanned by its
    /// triplet, then on a geometrically growing span up to the whole arc
    /// (see [`CorrectionConfig`]), every stage within `rchi2_threshold`;
    /// the converged candidate with the lowest final reduced chi2 wins.
    Corrected,
}

impl IodAcceptance {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefit => "prefit",
            Self::Corrected => "corrected",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "prefit" => Ok(Self::Prefit),
            "corrected" => Ok(Self::Corrected),
            other => Err(format!(
                "acceptance must be 'prefit' or 'corrected'; got {other:?}"
            )),
        }
    }
}

/// Which observations score a Gauss candidate's preliminary orbit, the
/// order in which corrected acceptance corrects candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefitScope {
    /// The observations of the candidate's first correction stage (its
    /// triplet's span, grown to `min_stage_obs`): how good the orbit is
    /// where it was built. A Gauss orbit from three observations days apart
    /// extrapolates poorly, so its chi2 over a long arc mostly measures the
    /// extrapolation (2002 RV112's three roots: 1e4, 4e5 and 3e9), and
    /// scoring it there costs a full-arc propagation per candidate.
    Local,
    /// Every observation (the legacy score).
    WholeArc,
}

impl PrefitScope {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "local" => Ok(Self::Local),
            "whole_arc" => Ok(Self::WholeArc),
            other => Err(format!(
                "prefit_scope must be one of 'local', 'whole_arc'; got {other:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::WholeArc => "whole_arc",
        }
    }
}

/// Settings of [`IodAcceptance::Corrected`]: the staged correction of
/// each candidate, ported from the seeding ladder of the 2026-10-12
/// walk-forward (`od_walkforward.py` `_arc_seed` / `_arc_expand`), which
/// seeded 98 of the 100 first solutions the legacy full-arc gate missed.
/// No outlier rejection runs during these stages: an intermediate orbit
/// fitted to part of the arc must not reject observations it cannot yet
/// predict well; rejection belongs to the refinement of the accepted
/// orbit. The partial stages down-weight instead (`stage_loss`), as
/// OrbFit, JPL and Find_Orb weight or reject inside every correction.
/// Validated on the walk-forward first solutions (see the module's IOD
/// policy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrectionConfig {
    /// Candidates corrected at most, in increasing prefit reduced chi2. 8
    /// (an engineering bound, validated on the walk-forward first
    /// solutions): the order of the study ladder's seed attempts (four
    /// windows x two selections) and a quarter of the 32-triplet budget,
    /// which bounds the cost of an arc that cannot be seeded.
    pub max_corrections: usize,
    /// The search stops once this many candidates converged on the whole
    /// arc (2: more than one guards against a wrong-basin first success,
    /// whose sibling roots are corrected too; the ladder stopped at one).
    pub max_converged: usize,
    /// Each stage spans `growth` times the previous span (4: the
    /// walk-forward `_arc_expand` factor that seeded 98 of the 100 first
    /// solutions the legacy gate missed).
    pub growth: f64,
    /// The first stage holds at least this many observations (grown from
    /// the triplet's span until it does): 6, the IOD `min_obs` and the
    /// ladder's window minimum (12 coordinates for 6 parameters).
    pub min_stage_obs: usize,
    /// Stages per candidate at most before it is abandoned (40: the
    /// walk-forward `_arc_expand` bound; fourfold growth covers a 60-year
    /// arc from one night in about 8 stages, each gap jump adds one).
    pub max_stages: usize,
    /// Loss of the stages before the whole arc. `Huber` (the fit's
    /// `f_scale`) bounds the pull of observations a partial orbit predicts
    /// badly (433 Eros: 1960s plates stalled a 7,140-observation stage at
    /// reduced chi2 2,516) and gates each stage on its robust cost;
    /// `Linear` fits and gates on plain chi2. The whole-arc stage is always
    /// linear, followed by CMC2003 rejection when above the threshold.
    pub stage_loss: LossType,
    /// Observations that score each candidate's preliminary orbit, which
    /// orders the corrections (see [`PrefitScope`]).
    pub prefit_scope: PrefitScope,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            max_corrections: 8,
            max_converged: 2,
            growth: 4.0,
            min_stage_obs: 6,
            max_stages: 40,
            stage_loss: LossType::Huber,
            prefit_scope: PrefitScope::Local,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IodConfig {
    pub min_obs: usize,
    pub min_arc_length: f64,
    pub contamination_percentage: f64,
    pub rchi2_threshold: f64,
    pub observation_selection_method: ObservationSelectionMethod,
    pub light_time: bool,
    pub mu: f64,
    pub speed_of_light: f64,
    /// Upper bound on the number of candidate triplets searched. `None`
    /// keeps the legacy budget `3 * (max_outliers + 1)`, which couples the
    /// search to `contamination_percentage`; `Some(k)` searches at most `k`
    /// triplets whatever the contamination percentage (which then only
    /// sets the outlier allowance).
    pub max_triplets: Option<usize>,
    /// Velocity method of the Gauss candidates; `Auto` picks Gibbs or
    /// Herrick-Gibbs per root by the separation of its position vectors.
    pub velocity_method: VelocityMethod,
    /// Settings of the `Spans` selection method (ignored by the others).
    pub spans: SpanSearchConfig,
    /// Prefit (legacy) or corrected acceptance. `Corrected` needs the time
    /// and translation providers of the refinement and runs through
    /// `run_iod_with` (`od_refine.rs`); [`run_iod`] alone is the prefit
    /// search.
    pub acceptance: IodAcceptance,
    pub correction: CorrectionConfig,
}

impl Default for IodConfig {
    /// `NativeOrbitFitter`'s IOD settings: at least six observations over a
    /// day, 20 % contamination, reduced chi2 200, span-bounded triplets with
    /// light-time correction, corrected acceptance.
    fn default() -> Self {
        Self {
            min_obs: 6,
            min_arc_length: 1.0,
            contamination_percentage: 20.0,
            rchi2_threshold: 200.0,
            observation_selection_method: ObservationSelectionMethod::Spans,
            light_time: true,
            mu: IOD_DEFAULT_MU,
            speed_of_light: IOD_DEFAULT_SPEED_OF_LIGHT,
            max_triplets: None,
            // Herrick-Gibbs where the positions are within 5 degrees
            // (Vallado 2013, sec. 7.5.2), Gibbs otherwise; Herrick-Gibbs beat
            // Gibbs at every span of the 2026-07-28 sweep, often 100x in chi2.
            velocity_method: VelocityMethod::Auto,
            spans: SpanSearchConfig::default(),
            acceptance: IodAcceptance::Corrected,
            correction: CorrectionConfig::default(),
        }
    }
}

/// Product of the complete preliminary-IOD decision loop for one linkage.
/// `state` is heliocentric/ecliptic at `epoch` (exact days and nanoseconds
/// in `epoch_scale`); residual/member arrays retain the caller's
/// observation order. The search counters describe the
/// run whether or not an orbit was accepted; `not_found_reason` says why
/// none was.
#[derive(Debug, Clone)]
pub struct IodOutput {
    pub found: bool,
    pub state: [f64; 6],
    /// Epoch of `state`: the Gauss epoch of the accepted candidate, exact to
    /// the nanosecond in `epoch_scale`. `None` unless `found`.
    pub epoch: Option<crate::Epoch>,
    /// Time scale of `epoch`: that of the observation times.
    pub epoch_scale: crate::TimeScale,
    pub arc_length: f64,
    pub num_obs: usize,
    pub chi2_total: f64,
    pub reduced_chi2: f64,
    pub residuals: Vec<f64>,
    pub residual_chi2: Vec<f64>,
    pub residual_dof: Vec<i64>,
    pub residual_probability: Vec<f64>,
    pub solution: Vec<bool>,
    pub outlier: Vec<bool>,
    /// Triplets for which Gauss candidates were generated.
    pub triplets_tried: usize,
    /// Candidates scored against the observations (successful predictions).
    pub candidates_evaluated: usize,
    /// Candidates skipped because their prediction failed (see [`iod_fit`]).
    pub candidates_failed: usize,
    /// Why no orbit was accepted; `None` when `found`.
    pub not_found_reason: Option<String>,
}

impl IodOutput {
    fn not_found(n: usize, epoch_scale: crate::TimeScale) -> Self {
        Self {
            found: false,
            state: [f64::NAN; 6],
            epoch: None,
            epoch_scale,
            arc_length: f64::NAN,
            num_obs: 0,
            chi2_total: f64::NAN,
            reduced_chi2: f64::NAN,
            residuals: Vec::new(),
            residual_chi2: Vec::new(),
            residual_dof: Vec::new(),
            residual_probability: Vec::new(),
            solution: vec![false; n],
            outlier: vec![false; n],
            triplets_tried: 0,
            candidates_evaluated: 0,
            candidates_failed: 0,
            not_found_reason: None,
        }
    }

    fn not_found_because(n: usize, epoch_scale: crate::TimeScale, reason: String) -> Self {
        Self {
            not_found_reason: Some(reason),
            ..Self::not_found(n, epoch_scale)
        }
    }
}

/// Velocity method of a Gauss preliminary orbit. `Auto` is a setting only:
/// it resolves per root to Herrick-Gibbs when consecutive position vectors
/// are within `AUTO_HERRICK_GIBBS_MAX_SEPARATION_DEG` (5 degrees, Vallado
/// 2013) and to Gibbs otherwise; candidates record the resolved method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityMethod {
    Gauss,
    Gibbs,
    HerrickGibbs,
    Auto,
}

impl VelocityMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gauss => "gauss",
            Self::Gibbs => "gibbs",
            Self::HerrickGibbs => "herrick+gibbs",
            Self::Auto => "auto",
        }
    }

    /// The Python spelling of ``velocity_method``.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "gauss" => Ok(Self::Gauss),
            "gibbs" => Ok(Self::Gibbs),
            "herrick+gibbs" => Ok(Self::HerrickGibbs),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "velocity_method must be one of 'gauss', 'gibbs', 'herrick+gibbs', 'auto'; \
                 got {other:?}"
            )),
        }
    }

    fn code(self) -> i32 {
        match self {
            Self::Gauss => 0,
            Self::Gibbs => 1,
            Self::HerrickGibbs => 2,
            Self::Auto => VELOCITY_METHOD_AUTO,
        }
    }

    fn from_code(code: i32) -> Self {
        match code {
            0 => Self::Gauss,
            2 => Self::HerrickGibbs,
            _ => Self::Gibbs,
        }
    }
}

/// One Gauss candidate that was scored against the observations. Every row
/// index is in the caller's observation order.
#[derive(Debug, Clone)]
pub struct IodCandidate {
    /// Heliocentric ecliptic Cartesian state at `epoch`.
    pub state: [f64; 6],
    /// Gauss epoch of the candidate, exact to the nanosecond in `epoch_scale`.
    pub epoch: crate::Epoch,
    pub epoch_scale: crate::TimeScale,
    /// Rows of the earliest and latest observation of the search window
    /// (the whole arc under the current policy).
    pub window: (usize, usize),
    /// Rows of the three observations the candidate was built from.
    pub triplet: [usize; 3],
    /// Which of the triplet's `root_count` physical Gauss roots this is.
    pub root_index: usize,
    pub root_count: usize,
    pub velocity_method: VelocityMethod,
    /// Score of the preliminary orbit (through the IOD predictor) against
    /// every observation, or, under corrected acceptance with
    /// `PrefitScope::Local`, against the observations of its first
    /// correction stage (`num_obs` and `arc_length` then describe those).
    pub prefit_chi2: f64,
    pub prefit_reduced_chi2: f64,
    /// Rows the outlier trial removed to reach acceptance (empty otherwise).
    pub removed: Vec<usize>,
    /// Statistics after the outlier trial (the prefit ones unless rows were
    /// removed).
    pub chi2_total: f64,
    pub reduced_chi2: f64,
    pub num_obs: usize,
    pub arc_length: f64,
    /// Reduced chi2 of the corrected orbit on the observations it kept
    /// (corrected acceptance); `None` when the candidate was not corrected or
    /// did not reach the whole arc.
    pub postfit_reduced_chi2: Option<f64>,
    /// The corrected orbit's whole-arc score, which ranks candidates: every
    /// observation's chi2 clipped at CMC2003's rejection threshold (so a
    /// rejected observation counts as at least that), summed over the whole
    /// arc and divided by its degrees of freedom. Candidates that kept
    /// different observations are compared on the same data.
    pub whole_arc_score: Option<f64>,
    /// Whether the staged correction reached the whole arc within the
    /// threshold at every stage.
    pub converged: bool,
    /// The corrected state at `epoch` (corrected acceptance, converged).
    pub corrected_state: Option<[f64; 6]>,
    /// Row-major 6x6 covariance of `corrected_state` from the whole-arc fit
    /// (NaN when singular); used to tell distinct solutions apart.
    pub corrected_covariance: Option<[f64; 36]>,
    /// Correction stages run (0 when not corrected).
    pub stages: usize,
    /// Why the correction stopped short of the whole arc, if it did.
    pub correction_note: Option<String>,
    pub accepted: bool,
    /// The per-observation evaluation, kept for the accepted candidate (and
    /// every converged one once a6w.4 lands); `None` for the rest so dense
    /// arcs do not retain one residual table per candidate.
    pub evaluation: Option<FitEvaluation>,
}

/// Product of [`run_iod`]: every scored candidate, which one was accepted,
/// the search counters and the reason when none was.
#[derive(Debug, Clone)]
pub struct IodResult {
    pub candidates: Vec<IodCandidate>,
    /// Index into `candidates` of the accepted orbit; `None` when none was.
    pub accepted: Option<usize>,
    /// Time scale of every candidate epoch (that of the observation times).
    pub epoch_scale: crate::TimeScale,
    /// Triplets for which Gauss candidates were generated.
    pub triplets_tried: usize,
    /// Candidates scored against the observations (successful predictions).
    pub candidates_evaluated: usize,
    /// Candidates skipped because their prediction failed.
    pub candidates_failed: usize,
    /// Bounded corrections run on candidates (0 until bead a6w.4).
    pub corrections_run: usize,
    /// Why no orbit was accepted; `None` when one was.
    pub not_found_reason: Option<String>,
    /// Indices of the converged candidates that are a distinct solution
    /// about as good as the accepted one (corrected acceptance; see
    /// `run_iod_corrected`). Non-empty means the IOD is ambiguous.
    pub ambiguous_with: Vec<usize>,
}

impl IodResult {
    pub fn found(&self) -> bool {
        self.accepted.is_some()
    }

    pub fn accepted_candidate(&self) -> Option<&IodCandidate> {
        self.accepted.map(|index| &self.candidates[index])
    }

    /// The typed hand-off to refinement: the accepted orbit (its corrected
    /// state under corrected acceptance, else the Gauss state) as a
    /// heliocentric ecliptic `OrbitBatch` at its exact epoch (days and
    /// nanoseconds in `epoch_scale`); `None` when nothing was accepted.
    pub fn seed_orbit(&self) -> PropagationResultValue<Option<OrbitBatch>> {
        let Some(candidate) = self.accepted_candidate() else {
            return Ok(None);
        };
        let coordinates = CoordinateBatch::cartesian(
            vec![candidate.corrected_state.unwrap_or(candidate.state)],
            Frame::Ecliptic,
            OriginArray::repeat(crate::OriginId::from_code("SUN"), 1),
            Some(TimeArray::new(
                candidate.epoch_scale,
                vec![candidate.epoch],
            )?),
            None,
        )?;
        Ok(Some(OrbitBatch::new(
            vec![OrbitId("iod-orbit".to_string())],
            vec![None],
            coordinates,
        )?))
    }

    /// Whether another converged solution is distinct from the accepted
    /// one yet fits about as well.
    pub fn ambiguous(&self) -> bool {
        !self.ambiguous_with.is_empty()
    }

    pub(super) fn not_found(epoch_scale: crate::TimeScale, reason: String) -> Self {
        Self {
            candidates: Vec::new(),
            accepted: None,
            epoch_scale,
            triplets_tried: 0,
            candidates_evaluated: 0,
            candidates_failed: 0,
            corrections_run: 0,
            not_found_reason: Some(reason),
            ambiguous_with: Vec::new(),
        }
    }

    fn with_search_counters(mut self, counters: &SearchCounters) -> Self {
        self.triplets_tried = counters.triplets_tried;
        self.candidates_evaluated = counters.candidates_evaluated;
        self.candidates_failed = counters.candidates_failed;
        self
    }

    /// The legacy one-orbit view ([`IodOutput`]) of this result over `n`
    /// observations: the accepted candidate's orbit, statistics and member
    /// arrays, or a not-found output carrying the counters and the reason.
    pub fn to_output(&self, n: usize) -> IodOutput {
        let mut output = match self.accepted_candidate() {
            Some(candidate) => {
                let evaluation = candidate
                    .evaluation
                    .clone()
                    .unwrap_or_else(|| FitEvaluation {
                        residuals: Vec::new(),
                        chi2: Vec::new(),
                        dof: Vec::new(),
                        probability: Vec::new(),
                        orbit_chi2: candidate.prefit_chi2,
                        reduced_chi2: candidate.prefit_reduced_chi2,
                        arc_length: candidate.arc_length,
                        num_obs: candidate.num_obs,
                        outlier: Vec::new(),
                    });
                let corrected = candidate.corrected_state.is_some();
                IodOutput {
                    found: true,
                    state: candidate.corrected_state.unwrap_or(candidate.state),
                    epoch: Some(candidate.epoch),
                    epoch_scale: candidate.epoch_scale,
                    arc_length: if corrected {
                        evaluation.arc_length
                    } else {
                        candidate.arc_length
                    },
                    num_obs: if corrected {
                        evaluation.num_obs
                    } else {
                        candidate.num_obs
                    },
                    chi2_total: if corrected {
                        evaluation.orbit_chi2
                    } else {
                        candidate.chi2_total
                    },
                    reduced_chi2: if corrected {
                        evaluation.reduced_chi2
                    } else {
                        candidate.reduced_chi2
                    },
                    residuals: evaluation.residuals,
                    residual_chi2: evaluation.chi2,
                    residual_dof: evaluation.dof,
                    residual_probability: evaluation.probability,
                    solution: (0..n).map(|row| candidate.triplet.contains(&row)).collect(),
                    outlier: (0..n).map(|row| candidate.removed.contains(&row)).collect(),
                    triplets_tried: 0,
                    candidates_evaluated: 0,
                    candidates_failed: 0,
                    not_found_reason: None,
                }
            }
            None => IodOutput::not_found_because(
                n,
                self.epoch_scale,
                self.not_found_reason
                    .clone()
                    .unwrap_or_else(|| "no orbit accepted".to_string()),
            ),
        };
        output.triplets_tried = self.triplets_tried;
        output.candidates_evaluated = self.candidates_evaluated;
        output.candidates_failed = self.candidates_failed;
        output
    }
}

/// Running totals of one IOD search.
#[derive(Debug, Default)]
struct SearchCounters {
    triplets_tried: usize,
    candidates_evaluated: usize,
    candidates_failed: usize,
    /// Message of the last candidate prediction failure.
    last_failure: Option<String>,
    /// Lowest reduced chi2 among the scored candidates (before outlier
    /// trials), for the not-found reason.
    best_reduced_chi2: f64,
}

/// Complete legacy `iod()` work unit: triplet selection, fused Gauss
/// candidate generation, backend ephemeris/residual scoring, outlier trials,
/// and first-acceptable-candidate selection. Observations are searched in a
/// canonical order (epoch, then station code, then input row) whatever
/// order the caller supplies them in; all output member arrays keep the
/// caller's order.
#[allow(clippy::too_many_arguments)]
pub fn iod_fit_barycentric<P, T>(
    propagator: &P,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &IodConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<IodOutput>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    let predictor = PropagatorPredictor {
        propagator,
        options,
        provider,
        translation_provider,
    };
    iod_fit(&predictor, observed, observers, config)
}

/// [`iod_fit_barycentric`] over any [`SphericalPredictor`].
///
/// The search is deterministic in the caller's row order: observations are
/// sorted by epoch (integer days and nanoseconds), then station code, then
/// input row before triplets are selected, and every member array of the
/// output is mapped back to the input order. A zero contamination
/// percentage only disables outlier removal; a rejected candidate is
/// followed by the next candidate and the next triplet. A candidate whose
/// prediction fails (`PropagationError::Backend`, `BackendProtocol`,
/// `ThreadPool`, including a per-row numerical failure) is counted in
/// `candidates_failed` and skipped; `InvalidRequest` and `Schema` errors
/// are caller errors and propagate. When every candidate failed the result
/// is not found with the last failure message as `not_found_reason`.
///
/// The product is the typed candidate set [`IodResult`]: every scored
/// candidate with its triplet, root, prefit statistics and (for the
/// accepted one) the full evaluation, plus the search counters. The
/// policy is unchanged: the first candidate within `rchi2_threshold`, with
/// the outlier trial, is accepted. `iod_predictor` is the propagator that
/// scores candidates; it is explicit so that IOD and refinement can run on
/// different backends.
pub fn run_iod(
    iod_predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &IodConfig,
) -> PropagationResultValue<IodResult> {
    let n = observed.len();
    if observers.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed and observers must have equal length".to_string(),
        ));
    }
    let epoch_scale = observation_time_scale(observed)?;
    if n == 0 || n < config.min_obs {
        return Ok(IodResult::not_found(
            epoch_scale,
            format!("{n} observations, fewer than min_obs = {}", config.min_obs),
        ));
    }
    let order = iod_observation_order(observed, observers)?;
    let sorted_observed = take_coordinate_batch(observed, &order)?;
    let sorted_observers = take_observer_batch(observers, &order)?;
    let result = run_iod_sorted(
        iod_predictor,
        &sorted_observed,
        &sorted_observers,
        config,
        false,
    )?;
    Ok(restore_input_order(result, &order))
}

/// Every candidate of the search, scored against all observations, with
/// no acceptance and no outlier trial (the candidate pool of corrected
/// acceptance). Rows are in the caller's order; `accepted` is None and
/// `not_found_reason` describes the pool.
pub(super) fn iod_candidate_pool(
    iod_predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &IodConfig,
) -> PropagationResultValue<IodResult> {
    let n = observed.len();
    if observers.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed and observers must have equal length".to_string(),
        ));
    }
    let epoch_scale = observation_time_scale(observed)?;
    if n == 0 || n < config.min_obs {
        return Ok(IodResult::not_found(
            epoch_scale,
            format!("{n} observations, fewer than min_obs = {}", config.min_obs),
        ));
    }
    // A short arc does not determine an orbit (a night cannot fix six
    // parameters); the prefit gate checks this per candidate, the corrected
    // acceptance here and again after rejection.
    let times = observed_times_mjd(observed)?;
    let arc = times.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - times.iter().copied().fold(f64::INFINITY, f64::min);
    if arc < config.min_arc_length {
        return Ok(IodResult::not_found(
            epoch_scale,
            format!(
                "arc of {arc:.4} days, shorter than min_arc_length = {}",
                config.min_arc_length
            ),
        ));
    }
    let order = iod_observation_order(observed, observers)?;
    let sorted_observed = take_coordinate_batch(observed, &order)?;
    let sorted_observers = take_observer_batch(observers, &order)?;
    let result = run_iod_sorted(
        iod_predictor,
        &sorted_observed,
        &sorted_observers,
        config,
        true,
    )?;
    Ok(restore_input_order(result, &order))
}

/// The legacy one-orbit view of [`run_iod`] (`IodOutput`): kept for the
/// linkage batch driver and for callers of the pre-candidate-set contract;
/// new code should use [`run_iod`]. Deprecated in favour of `run_iod`.
pub fn iod_fit(
    predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &IodConfig,
) -> PropagationResultValue<IodOutput> {
    run_iod(predictor, observed, observers, config).map(|result| result.to_output(observed.len()))
}

/// Time scale of the observation epochs (the scale of every IOD epoch).
fn observation_time_scale(observed: &CoordinateBatch) -> PropagationResultValue<crate::TimeScale> {
    Ok(observed
        .times
        .as_ref()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "orbit determination requires observation times".to_string(),
            )
        })?
        .scale)
}

/// Permutation putting the observations in the canonical IOD search order:
/// epoch (days, then nanoseconds), station code, input row.
fn iod_observation_order(
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
) -> PropagationResultValue<Vec<usize>> {
    let times = observed.times.as_ref().ok_or_else(|| {
        PropagationError::InvalidRequest(
            "orbit determination requires observation times".to_string(),
        )
    })?;
    let mut order: Vec<usize> = (0..observed.len()).collect();
    order.sort_by(|&a, &b| {
        let (ea, eb) = (times.epochs[a], times.epochs[b]);
        (ea.days, ea.nanos)
            .cmp(&(eb.days, eb.nanos))
            .then_with(|| observers.code[a].0.cmp(&observers.code[b].0))
            .then(a.cmp(&b))
    });
    Ok(order)
}

/// Map a result computed on the sorted view (`order[k]` = input row of
/// sorted row `k`) back to the input order: every candidate's rows and the
/// member arrays of its evaluation.
fn restore_input_order(mut result: IodResult, order: &[usize]) -> IodResult {
    fn unpermute<T: Clone>(sorted: &[T], order: &[usize], width: usize) -> Vec<T> {
        if sorted.is_empty() {
            return Vec::new();
        }
        let mut out = sorted.to_vec();
        for (k, &row) in order.iter().enumerate() {
            out[row * width..(row + 1) * width]
                .clone_from_slice(&sorted[k * width..(k + 1) * width]);
        }
        out
    }
    for candidate in result.candidates.iter_mut() {
        candidate.window = (order[candidate.window.0], order[candidate.window.1]);
        candidate.triplet = std::array::from_fn(|slot| order[candidate.triplet[slot]]);
        for row in candidate.removed.iter_mut() {
            *row = order[*row];
        }
        if let Some(evaluation) = candidate.evaluation.as_mut() {
            evaluation.residuals = unpermute(&evaluation.residuals, order, 6);
            evaluation.chi2 = unpermute(&evaluation.chi2, order, 1);
            evaluation.dof = unpermute(&evaluation.dof, order, 1);
            evaluation.probability = unpermute(&evaluation.probability, order, 1);
            evaluation.outlier = unpermute(&evaluation.outlier, order, 1);
        }
    }
    result
}

/// J2000 obliquity of the ecliptic (IAU 1976, 84381.448 arcsec), the
/// rotation the Gauss kernel applies to its equatorial directions.
const GAUSS_OBLIQUITY_RAD: f64 = 84_381.448 / 3600.0 * std::f64::consts::PI / 180.0;

/// The geometry the Gauss kernel assumes: equatorial (RA, Dec) directions in
/// degrees and heliocentric ecliptic observer positions (au). Ecliptic
/// observations and equatorial observers are rotated; other frames, and
/// observers not centred on the Sun, are refused (the kernel would silently
/// use them as heliocentric: a barycentric observer is about 0.005 au off).
/// Equatorial (RA, Dec) directions (degrees) and heliocentric ecliptic
/// observer positions (au), row for row.
type GaussGeometry = (Vec<[f64; 2]>, Vec<[f64; 3]>);

fn gauss_geometry(
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
) -> PropagationResultValue<GaussGeometry> {
    let (sin_e, cos_e) = GAUSS_OBLIQUITY_RAD.sin_cos();
    let values = observed.values.spherical().ok_or_else(|| {
        PropagationError::InvalidRequest("IOD requires spherical observed coordinates".to_string())
    })?;
    let directions: Vec<[f64; 2]> = match observed.frame {
        Frame::Equatorial => values.iter().map(|row| [row[1], row[2]]).collect(),
        Frame::Ecliptic => values
            .iter()
            .map(|row| {
                let (lon, lat) = (row[1].to_radians(), row[2].to_radians());
                let ec = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
                let eq = [
                    ec[0],
                    cos_e * ec[1] - sin_e * ec[2],
                    sin_e * ec[1] + cos_e * ec[2],
                ];
                [
                    eq[1].atan2(eq[0]).to_degrees().rem_euclid(360.0),
                    eq[2].clamp(-1.0, 1.0).asin().to_degrees(),
                ]
            })
            .collect(),
        other => {
            return Err(PropagationError::InvalidRequest(format!(
                "IOD requires equatorial or ecliptic observed coordinates; got {}",
                other.as_str()
            )))
        }
    };
    let coordinates = &observers.coordinates;
    if let Some(origin) = coordinates
        .origins
        .origins
        .iter()
        .find(|origin| !super::ephemeris::is_sun_origin(origin))
    {
        return Err(PropagationError::InvalidRequest(format!(
            "IOD requires heliocentric observer positions (origin SUN); got {}",
            origin.code()
        )));
    }
    let positions = coordinates.values.cartesian().ok_or_else(|| {
        PropagationError::InvalidRequest("IOD requires Cartesian observer coordinates".to_string())
    })?;
    let observer_positions: Vec<[f64; 3]> = match coordinates.frame {
        Frame::Ecliptic => positions
            .iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect(),
        Frame::Equatorial => positions
            .iter()
            .map(|row| {
                [
                    row[0],
                    cos_e * row[1] + sin_e * row[2],
                    -sin_e * row[1] + cos_e * row[2],
                ]
            })
            .collect(),
        other => {
            return Err(PropagationError::InvalidRequest(format!(
                "IOD requires ecliptic or equatorial observer coordinates; got {}",
                other.as_str()
            )))
        }
    };
    Ok((directions, observer_positions))
}

/// [`run_iod`] on observations already in the canonical search order.
fn run_iod_sorted(
    predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &IodConfig,
    collect_all: bool,
) -> PropagationResultValue<IodResult> {
    let n = observed.len();
    let (gauss_directions, gauss_observers) = gauss_geometry(observed, observers)?;
    let times = observed_times_mjd(observed)?;
    let epoch_scale = observation_time_scale(observed)?;

    let nights = observation_nights(&times, Some(&observers.code));
    let max_outliers =
        (((n as f64) * (config.contamination_percentage / 100.0)) as usize).min(n - config.min_obs);
    // The legacy `iod()` loop tries `3 * (max_outliers + 1)` triplets
    // (`obs_ids[: (3 * (max_outliers + 1))]` on the (k, 3) selection):
    // three candidate triplets per allowed outlier. `max_triplets`
    // decouples the search budget from the outlier allowance.
    let budget = match config.max_triplets {
        Some(0) => {
            return Err(PropagationError::InvalidRequest(
                "max_triplets must be at least 1".to_string(),
            ))
        }
        Some(budget) => budget,
        None if config.observation_selection_method == ObservationSelectionMethod::Spans => {
            config.spans.max_triplets.max(1)
        }
        None => 3 * (max_outliers + 1),
    };
    let triplets: Vec<([usize; 3], (usize, usize))> =
        if config.observation_selection_method == ObservationSelectionMethod::Spans {
            span_triplets(&times, &nights, &config.spans, budget)
        } else {
            select_observation_triplets_bounded(&times, config.observation_selection_method, budget)
                .into_iter()
                .map(|triplet| (triplet, (0, n - 1)))
                .collect()
        };
    if triplets.is_empty() {
        return Ok(IodResult::not_found(
            epoch_scale,
            "no triplet of observations with three distinct epochs".to_string(),
        ));
    }

    let mut counters = SearchCounters {
        best_reduced_chi2: f64::INFINITY,
        ..SearchCounters::default()
    };
    // Corrected acceptance with a local prefit scores each candidate on its
    // first correction stage only (the same rows `run_iod_corrected` starts
    // from), predicting just those observations.
    let local_prefit = collect_all
        && config.acceptance == IodAcceptance::Corrected
        && config.correction.prefit_scope == PrefitScope::Local;
    let mut candidates: Vec<IodCandidate> = Vec::new();
    for (triplet, window) in triplets {
        counters.triplets_tried += 1;
        let ra = std::array::from_fn(|slot| gauss_directions[triplet[slot]][0]);
        let dec = std::array::from_fn(|slot| gauss_directions[triplet[slot]][1]);
        let triplet_times = std::array::from_fn(|slot| times[triplet[slot]]);
        let triplet_observers = std::array::from_fn(|slot| gauss_observers[triplet[slot]]);
        let gauss = gauss_iod_candidates(
            ra,
            dec,
            triplet_times,
            triplet_observers,
            config.velocity_method.code(),
            config.light_time,
            config.mu,
            config.speed_of_light,
        );
        let candidate_count = gauss.len();
        let scope: Option<(Vec<usize>, CoordinateBatch, ObserverBatch)> = if local_prefit {
            let mask =
                super::od_refine::first_stage_mask(&times, &nights, triplet, &config.correction);
            let rows: Vec<usize> = (0..n).filter(|&row| mask[row]).collect();
            let scoped_observed = take_coordinate_batch(observed, &rows)?;
            let scoped_observers = take_observer_batch(observers, &rows)?;
            Some((rows, scoped_observed, scoped_observers))
        } else {
            None
        };
        let (score_observed, score_observers) = match &scope {
            Some((_, scoped_observed, scoped_observers)) => (scoped_observed, scoped_observers),
            None => (observed, observers),
        };
        let score_n = score_observed.len();
        for (candidate, root) in gauss.iter().enumerate() {
            let state = root.state;
            if !state.iter().all(|value| value.is_finite()) {
                continue;
            }
            let velocity_method = VelocityMethod::from_code(root.velocity_method);
            let epoch = mjd_epoch(root.epoch);
            let candidate_coordinates = CoordinateBatch::cartesian(
                vec![state],
                Frame::Ecliptic,
                OriginArray::repeat(crate::OriginId::from_code("SUN"), 1),
                Some(TimeArray::new(epoch_scale, vec![epoch])?),
                None,
            )?;
            let orbit = OrbitBatch::new(
                vec![OrbitId("iod-candidate-0000".to_string())],
                vec![None],
                candidate_coordinates,
            )?;
            let evaluation = match evaluate_orbit(
                predictor,
                state,
                &orbit,
                score_observed,
                score_observers,
                &vec![false; score_n],
                6,
            ) {
                Ok(evaluation) => evaluation,
                // A failed prediction is local to this candidate.
                Err(
                    err @ (PropagationError::Backend(_)
                    | PropagationError::BackendProtocol(_)
                    | PropagationError::ThreadPool(_)),
                ) => {
                    counters.candidates_failed += 1;
                    counters.last_failure = Some(err.to_string());
                    continue;
                }
                Err(err) => return Err(err),
            };
            counters.candidates_evaluated += 1;
            let mut record = IodCandidate {
                state,
                epoch,
                epoch_scale,
                window,
                triplet,
                root_index: candidate,
                root_count: candidate_count,
                velocity_method,
                prefit_chi2: evaluation.orbit_chi2,
                prefit_reduced_chi2: evaluation.reduced_chi2,
                removed: Vec::new(),
                chi2_total: evaluation.orbit_chi2,
                reduced_chi2: evaluation.reduced_chi2,
                num_obs: evaluation.num_obs,
                arc_length: evaluation.arc_length,
                postfit_reduced_chi2: None,
                whole_arc_score: None,
                converged: false,
                corrected_state: None,
                corrected_covariance: None,
                stages: 0,
                correction_note: None,
                accepted: false,
                evaluation: None,
            };

            if collect_all {
                counters.best_reduced_chi2 =
                    counters.best_reduced_chi2.min(evaluation.reduced_chi2);
                candidates.push(record);
                continue;
            }
            if evaluation.reduced_chi2 <= config.rchi2_threshold {
                record.accepted = true;
                record.evaluation = Some(evaluation);
                candidates.push(record);
                return Ok(IodResult {
                    accepted: Some(candidates.len() - 1),
                    candidates,
                    epoch_scale,
                    triplets_tried: 0,
                    candidates_evaluated: 0,
                    candidates_failed: 0,
                    corrections_run: 0,
                    ambiguous_with: Vec::new(),
                    not_found_reason: None,
                }
                .with_search_counters(&counters));
            }
            counters.best_reduced_chi2 = counters.best_reduced_chi2.min(evaluation.reduced_chi2);
            if max_outliers == 0 {
                // Zero contamination only disables outlier removal: the
                // next candidate, then the next triplet, is tried.
                candidates.push(record);
                continue;
            }

            let triplet_mask: Vec<bool> = (0..n).map(|row| triplet.contains(&row)).collect();
            let mut removable: Vec<usize> = (0..n).filter(|&row| !triplet_mask[row]).collect();
            removable.sort_by(|&left, &right| {
                evaluation.chi2[left]
                    .total_cmp(&evaluation.chi2[right])
                    .then(left.cmp(&right))
            });
            for count in 1..=max_outliers.min(removable.len()) {
                let removed = removable[removable.len() - count..].to_vec();
                let removed_chi2: f64 = removed.iter().map(|&row| evaluation.chi2[row]).sum();
                let num_obs = n - removed.len();
                let chi2 = evaluation.orbit_chi2 - removed_chi2;
                let reduced = chi2 / (2 * num_obs - 6) as f64;
                let keep: Vec<bool> = (0..n).map(|row| !removed.contains(&row)).collect();
                let (arc_length, _, _) = masked_arc_length(&times, &keep);
                if reduced <= config.rchi2_threshold && arc_length >= config.min_arc_length {
                    record.removed = removed;
                    record.chi2_total = chi2;
                    record.reduced_chi2 = reduced;
                    record.num_obs = num_obs;
                    record.arc_length = arc_length;
                    record.accepted = true;
                    record.evaluation = Some(evaluation);
                    candidates.push(record);
                    return Ok(IodResult {
                        accepted: Some(candidates.len() - 1),
                        candidates,
                        epoch_scale,
                        triplets_tried: 0,
                        candidates_evaluated: 0,
                        candidates_failed: 0,
                        corrections_run: 0,
                        ambiguous_with: Vec::new(),
                        not_found_reason: None,
                    }
                    .with_search_counters(&counters));
                }
            }
            candidates.push(record);
        }
    }
    let reason = if counters.candidates_evaluated == 0 && counters.candidates_failed > 0 {
        format!(
            "every candidate prediction failed ({} candidates from {} triplets); last failure: {}",
            counters.candidates_failed,
            counters.triplets_tried,
            counters.last_failure.as_deref().unwrap_or("unknown")
        )
    } else if counters.candidates_evaluated == 0 {
        format!(
            "no finite Gauss candidate from {} triplets",
            counters.triplets_tried
        )
    } else {
        format!(
            "{} candidates from {} triplets scored, none within rchi2_threshold = {} \
             (best reduced chi2 {:.4}, {} allowed outliers, {} candidate predictions failed)",
            counters.candidates_evaluated,
            counters.triplets_tried,
            config.rchi2_threshold,
            counters.best_reduced_chi2,
            max_outliers,
            counters.candidates_failed
        )
    };
    Ok(IodResult {
        candidates,
        accepted: None,
        epoch_scale,
        triplets_tried: 0,
        candidates_evaluated: 0,
        candidates_failed: 0,
        corrections_run: 0,
        ambiguous_with: Vec::new(),
        not_found_reason: Some(reason),
    }
    .with_search_counters(&counters))
}

/// Batch the complete IOD work unit across linkage row selections without
/// returning control to Python between linkages. `chunk_size` is a scheduling
/// control owned by Rust; output order always matches `linkage_rows`.
#[allow(clippy::too_many_arguments)]
pub fn iod_fit_linkages_barycentric<P, T>(
    propagator: &P,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    linkage_rows: &[Vec<usize>],
    chunk_size: usize,
    config: &IodConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<Vec<IodOutput>>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    let chunk_size = chunk_size.max(1);
    let mut outputs = Vec::with_capacity(linkage_rows.len());
    for chunk in linkage_rows.chunks(chunk_size) {
        for rows in chunk {
            for &row in rows {
                if row >= observed.len() {
                    return Err(PropagationError::InvalidRequest(format!(
                        "linkage observation row {row} is out of bounds for {} observations",
                        observed.len()
                    )));
                }
            }
            let linkage_observed = take_coordinate_batch(observed, rows)?;
            let linkage_observers = take_observer_batch(observers, rows)?;
            outputs.push(iod_fit_barycentric(
                propagator,
                &linkage_observed,
                &linkage_observers,
                config,
                options,
                provider,
                translation_provider,
            )?);
        }
    }
    Ok(outputs)
}

pub(super) fn mjd_epoch(mjd: f64) -> crate::Epoch {
    let days = mjd.floor();
    crate::Epoch::new(
        days as i64,
        ((mjd - days) * 86_400_000_000_000.0).round() as i64,
    )
}

/// Every candidate triplet of `method` (test oracle for the bounded walk).
#[cfg(test)]
fn select_observation_triplets(
    times: &[f64],
    method: ObservationSelectionMethod,
) -> Vec<[usize; 3]> {
    select_observation_triplets_bounded(times, method, usize::MAX)
}

/// The first `budget` candidate triplets of `method` in the legacy priority
/// order, restricted to three distinct epochs. `Combinations` is
/// enumerated lazily (see [`combination_triplets`]), never as the full
/// `C(n, 3)` list.
fn select_observation_triplets_bounded(
    times: &[f64],
    method: ObservationSelectionMethod,
    budget: usize,
) -> Vec<[usize; 3]> {
    let mut triplets: Vec<[usize; 3]> = match method {
        ObservationSelectionMethod::FirstMiddleLast => {
            percentile_triplet(times, [0.0, 50.0, 100.0])
                .into_iter()
                .collect()
        }
        ObservationSelectionMethod::Thirds => {
            percentile_triplet(times, [100.0 / 6.0, 50.0, 500.0 / 6.0])
                .into_iter()
                .collect()
        }
        ObservationSelectionMethod::Combinations => combination_triplets(times, budget),
        ObservationSelectionMethod::Spans => span_triplets(
            times,
            &observation_nights(times, None),
            &SpanSearchConfig::default(),
            budget,
        )
        .into_iter()
        .map(|(triplet, _)| triplet)
        .collect(),
    };
    triplets.retain(|row| {
        times[row[0]] != times[row[1]]
            && times[row[0]] != times[row[2]]
            && times[row[1]] != times[row[2]]
    });
    triplets.truncate(budget);
    triplets
}

/// The first `budget` combination triplets in the legacy priority order
/// (arc length descending, then the middle observation's offset from the
/// arc midpoint ascending, then index order), skipping triplets with
/// coincident epochs, without materializing all `C(n, 3)` combinations.
///
/// Observations are walked in time order, so a (first, last) pair's arc
/// never grows when the first moves later or the last earlier. A frontier
/// heap seeded with the full-arc pair therefore yields pairs in
/// non-increasing arc order; only the pairs sharing the current arc have
/// their middle observations enumerated and sorted, and the walk stops as
/// soon as the budget is met. Each pair is visited once.
fn combination_triplets(times: &[f64], budget: usize) -> Vec<[usize; 3]> {
    use std::cmp::Ordering;
    use std::collections::{BinaryHeap, HashSet};

    #[derive(PartialEq)]
    struct Pair {
        arc: f64,
        first: usize,
        last: usize,
    }
    impl Eq for Pair {}
    impl Ord for Pair {
        fn cmp(&self, other: &Self) -> Ordering {
            self.arc
                .total_cmp(&other.arc)
                .then_with(|| other.first.cmp(&self.first))
                .then_with(|| other.last.cmp(&self.last))
        }
    }
    impl PartialOrd for Pair {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = times.len();
    let mut out = Vec::new();
    if n < 3 || budget == 0 {
        return out;
    }
    // Positions in time order (stable, so ties keep index order): the
    // driver already sorts its observations, and the legacy index order
    // of a sorted input is the time order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
    let t = |position: usize| times[order[position]];

    let mut heap = BinaryHeap::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut visit = |heap: &mut BinaryHeap<Pair>, first: usize, last: usize| {
        if first < last && seen.insert((first, last)) {
            heap.push(Pair {
                arc: t(last) - t(first),
                first,
                last,
            });
        }
    };
    visit(&mut heap, 0, n - 1);

    while let Some(top) = heap.pop() {
        let arc = top.arc;
        let mut group = vec![(top.first, top.last)];
        visit(&mut heap, top.first + 1, top.last);
        visit(&mut heap, top.first, top.last - 1);
        while heap.peek().is_some_and(|next| next.arc == arc) {
            let next = heap.pop().expect("peeked");
            group.push((next.first, next.last));
            visit(&mut heap, next.first + 1, next.last);
            visit(&mut heap, next.first, next.last - 1);
        }
        if arc == 0.0 {
            // Every remaining pair has coincident epochs.
            break;
        }
        let mut candidates: Vec<(f64, usize, usize, usize)> = Vec::new();
        for (first, last) in group {
            let midpoint = (t(last) + t(first)) / 2.0;
            for middle in first + 1..last {
                if t(middle) == t(first) || t(middle) == t(last) {
                    continue;
                }
                candidates.push(((midpoint - t(middle)).abs(), first, middle, last));
            }
        }
        candidates.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        for (_, first, middle, last) in candidates {
            out.push([order[first], order[middle], order[last]]);
            if out.len() == budget {
                return out;
            }
        }
    }
    out
}

/// The first `budget` triplets of the span-bounded search
/// ([`SpanSearchConfig`]), each with the rows of its search window.
///
/// Windows, in order: the densest `dense_window_days[0]` and
/// `dense_window_days[1]` days (the earliest on ties), the last and the
/// first `edge_window_days`, then the whole arc; repeated windows are
/// skipped. Within a window every first observation contributes a lazy
/// stream of last observations in increasing |span - target| (two pointers
/// from the target outwards, bounded by the span limits); the streams are
/// merged by a heap keyed (|span - target|, first, last), so pairs are
/// visited in that total order without enumerating them all. Each pair gives
/// one triplet, its middle the observation nearest the pair's midpoint
/// (earliest on ties) with an epoch distinct from both ends; pairs with no
/// such middle are skipped, as are pairs whose ends fall on a pair of nights
/// (integer days) already used, in this or an earlier window: observations
/// of the same two nights give near-identical Gauss geometry, so a dense
/// night would otherwise spend the whole budget on one configuration. The
/// walk stops at `budget`.
/// Gap that separates two observing nights of one station (days): 8 hours.
/// A ground station's daytime gap is at least 10 hours (a night lasts at
/// most about 14), and gaps inside a night rarely exceed 8, so a station's
/// observations separated by more than this belong to different nights.
/// Unlike the UTC calendar day (`floor(MJD)`), this does not split the
/// nights of stations east of the Americas, which straddle 0h UTC.
pub const OBSERVING_NIGHT_GAP_DAYS: f64 = 8.0 / 24.0;

/// Longest observing night (days): 16 hours, above the longest dark period
/// at survey latitudes (about 14). A station observing continuously (a
/// space telescope, or round-the-clock cadence) starts a new night after it,
/// so its arc is not chained into one "night".
pub const OBSERVING_NIGHT_MAX_DAYS: f64 = 16.0 / 24.0;

/// Observing-night id of every observation: each station's observations, in
/// time order, start a new night after a gap above
/// [`OBSERVING_NIGHT_GAP_DAYS`] or once the night has lasted
/// [`OBSERVING_NIGHT_MAX_DAYS`]; ids are numbered by first observation time
/// (then row). Without `stations` every row is one station.
pub(super) fn observation_nights(
    times: &[f64],
    stations: Option<&[crate::ObservatoryCode]>,
) -> Vec<usize> {
    let n = times.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]).then(a.cmp(&b)));
    let station = |row: usize| stations.map(|codes| codes[row].0.as_str());
    // Per station: (start of its current night, last time, night id).
    let mut last_seen: std::collections::HashMap<Option<&str>, (f64, f64, usize)> =
        std::collections::HashMap::new();
    let mut nights = vec![0_usize; n];
    let mut next_id = 0_usize;
    for &row in &order {
        let key = station(row);
        let t = times[row];
        let (start, id) = match last_seen.get(&key) {
            Some(&(start, last, id))
                if t - last <= OBSERVING_NIGHT_GAP_DAYS
                    && t - start <= OBSERVING_NIGHT_MAX_DAYS =>
            {
                (start, id)
            }
            _ => {
                next_id += 1;
                (t, next_id - 1)
            }
        };
        nights[row] = id;
        last_seen.insert(key, (start, t, id));
    }
    nights
}

fn span_triplets(
    times: &[f64],
    nights: &[usize],
    config: &SpanSearchConfig,
    budget: usize,
) -> Vec<([usize; 3], (usize, usize))> {
    use std::cmp::{Ordering, Reverse};
    use std::collections::{BinaryHeap, HashSet};

    let n = times.len();
    let mut out = Vec::new();
    if n < 3 || budget == 0 {
        return out;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]).then(a.cmp(&b)));
    let t: Vec<f64> = order.iter().map(|&row| times[row]).collect();

    // Windows as inclusive position ranges over the time-sorted view.
    let densest = |length: f64| -> (usize, usize) {
        let mut best = (0, 0);
        let mut best_count = 0;
        let mut hi = 0;
        for lo in 0..n {
            if hi < lo {
                hi = lo;
            }
            while hi + 1 < n && t[hi + 1] - t[lo] <= length {
                hi += 1;
            }
            if hi - lo + 1 > best_count {
                best_count = hi - lo + 1;
                best = (lo, hi);
            }
        }
        best
    };
    let last_from = t.partition_point(|&value| value < t[n - 1] - config.edge_window_days);
    let first_to = t.partition_point(|&value| value <= t[0] + config.edge_window_days) - 1;
    let mut windows: Vec<(usize, usize)> = Vec::new();
    for window in [
        densest(config.dense_window_days[0]),
        densest(config.dense_window_days[1]),
        (last_from, n - 1),
        (0, first_to),
        (0, n - 1),
    ] {
        if !windows.contains(&window) {
            windows.push(window);
        }
    }

    #[derive(PartialEq)]
    struct Key(f64, usize, usize);
    impl Eq for Key {}
    impl Ord for Key {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0
                .total_cmp(&other.0)
                .then(self.1.cmp(&other.1))
                .then(self.2.cmp(&other.2))
        }
    }
    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let target = config.target_span_days;
    let span_ok = |first: usize, last: usize| {
        let span = t[last] - t[first];
        span > 0.0 && span >= config.min_span_days && span <= config.max_span_days
    };
    let deviation = |first: usize, last: usize| ((t[last] - t[first]) - target).abs();

    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for (lo, hi) in windows {
        if hi < lo + 2 {
            continue;
        }
        // Per first position: the lower side as (current, end of its run of
        // equal epochs), walked upwards within each run so that ties come
        // out in position order, and the next higher last; `None` once a
        // side is exhausted.
        type Down = Option<(usize, usize)>;
        let run_start = |first: usize, k: usize| {
            first + 1 + t[first + 1..=k].partition_point(|&value| value < t[k])
        };
        let down_run = |first: usize, k: usize| -> Down {
            if k <= first || !span_ok(first, k) {
                return None;
            }
            Some((run_start(first, k), k))
        };
        let mut cursors: Vec<(Down, Option<usize>)> = vec![(None, None); hi + 1];
        let mut heap: BinaryHeap<Reverse<Key>> = BinaryHeap::new();
        let next_of = |first: usize, cursor: &(Down, Option<usize>)| -> Option<usize> {
            match (cursor.0.map(|(current, _)| current), cursor.1) {
                (Some(down), Some(up)) => {
                    if deviation(first, down) <= deviation(first, up) {
                        Some(down)
                    } else {
                        Some(up)
                    }
                }
                (Some(down), None) => Some(down),
                (None, Some(up)) => Some(up),
                (None, None) => None,
            }
        };
        for first in lo..hi {
            // Split point: the first last with span >= target.
            let split =
                first + 1 + t[first + 1..=hi].partition_point(|&value| value - t[first] < target);
            let down = down_run(first, split - 1);
            let up = (split <= hi && span_ok(first, split)).then_some(split);
            cursors[first] = (down, up);
            if let Some(last) = next_of(first, &cursors[first]) {
                heap.push(Reverse(Key(deviation(first, last), first, last)));
            }
        }
        while let Some(Reverse(Key(_, first, last))) = heap.pop() {
            // Advance this first's stream past `last`.
            let (down, up) = cursors[first];
            if let Some((current, end)) = down.filter(|&(current, _)| current == last) {
                cursors[first].0 = if current < end {
                    Some((current + 1, end))
                } else {
                    down_run(first, run_start(first, end) - 1)
                };
            } else if up == Some(last) {
                cursors[first].1 = (last < hi && span_ok(first, last + 1)).then(|| last + 1);
            }
            if let Some(next) = next_of(first, &cursors[first]) {
                heap.push(Reverse(Key(deviation(first, next), first, next)));
            }

            let midpoint = (t[first] + t[last]) / 2.0;
            let middle = (first + 1..last)
                .filter(|&m| t[m] != t[first] && t[m] != t[last])
                .min_by(|&a, &b| {
                    (t[a] - midpoint)
                        .abs()
                        .total_cmp(&(t[b] - midpoint).abs())
                        .then(a.cmp(&b))
                });
            let Some(middle) = middle else { continue };
            let triplet = [order[first], order[middle], order[last]];
            if seen.insert((nights[order[first]], nights[order[last]])) {
                out.push((triplet, (order[lo], order[hi])));
                if out.len() == budget {
                    return out;
                }
            }
        }
    }
    out
}

fn percentile_triplet(times: &[f64], percentiles: [f64; 3]) -> Option<[usize; 3]> {
    if times.is_empty() {
        return None;
    }
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut indices = Vec::with_capacity(3);
    for percentile in percentiles {
        let rank = percentile / 100.0 * (times.len() as f64 - 1.0);
        let lower = rank.floor();
        let fractional = rank - lower;
        let nearest = if fractional < 0.5 {
            lower
        } else if fractional > 0.5 {
            lower + 1.0
        } else if (lower as i64) % 2 == 0 {
            lower
        } else {
            lower + 1.0
        } as usize;
        let value = sorted[nearest.min(sorted.len() - 1)];
        let index = times.iter().position(|item| *item == value)?;
        if !indices.contains(&index) {
            indices.push(index);
        }
    }
    (indices.len() == 3).then(|| [indices[0], indices[1], indices[2]])
}

// ---------------------------------------------------------------------------
// Shared internals
// ---------------------------------------------------------------------------

/// Epoch/frame/origin context extracted once from the single input orbit.
/// Epoch, frame and origin of the single orbit being corrected: the
/// geometry every candidate state shares.
#[derive(Debug, Clone)]
pub struct OrbitGeometry {
    pub state: [f64; 6],
    pub epoch: crate::Epoch,
    pub scale: crate::TimeScale,
    pub frame: Frame,
    pub origin: crate::OriginId,
}

impl OrbitGeometry {
    pub fn from_orbit(orbit: &OrbitBatch) -> PropagationResultValue<Self> {
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

pub(super) fn spherical_flat(
    batch: &CoordinateBatch,
    label: &str,
) -> PropagationResultValue<Vec<f64>> {
    let values = batch.values.spherical().ok_or_else(|| {
        PropagationError::InvalidRequest(format!(
            "{label} coordinates must be spherical for orbit determination"
        ))
    })?;
    Ok(values.iter().flat_map(|row| row.iter().copied()).collect())
}

/// The message of every "no observed covariance" refusal: names the
/// remedy for observations without reported uncertainties.
pub(super) const MISSING_COVARIANCE_REMEDY: &str = "orbit determination requires an \
observed covariance with finite, positive lon/lat variances; observations without \
reported uncertainties need a SigmaFillModel among the run_od observation models to \
supply them";

pub(super) fn observed_covariance_flat(
    observed: &CoordinateBatch,
) -> PropagationResultValue<Vec<f64>> {
    Ok(observed
        .covariance
        .as_ref()
        .ok_or_else(|| PropagationError::InvalidRequest(MISSING_COVARIANCE_REMEDY.to_string()))?
        .values_row_major
        .clone())
}

pub(super) fn observed_times_mjd(observed: &CoordinateBatch) -> PropagationResultValue<Vec<f64>> {
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
pub(super) fn predict_spherical<P, T>(
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
    let candidates = candidate_orbit_batch(states, geometry)?;
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

/// Candidate states as a one-epoch `OrbitBatch` sharing `geometry`, with
/// zero-padded ids so backends sorting by id keep input order.
pub fn candidate_orbit_batch(
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

/// Source of predicted topocentric spherical coordinates for candidate
/// states: the one operation every orbit-determination driver needs from a
/// backend. [`PropagatorPredictor`] implements it for any Rust
/// [`Propagator`] through the barycentric ephemeris workflow; a Python
/// propagator is wrapped by a callback implementation in the binding layer.
pub trait SphericalPredictor {
    /// Predict `(M * N, 6)` row-major spherical coordinates (candidate-major,
    /// observer order within each candidate) for `states` at the epoch,
    /// frame and origin of `geometry`. Outer `Err` is a request failure;
    /// inner `Err` a per-row numerical failure (a rejected trial step).
    fn predict_spherical(
        &self,
        states: &[[f64; 6]],
        geometry: &OrbitGeometry,
        observers: &ObserverBatch,
    ) -> PropagationResultValue<Result<Vec<f64>, String>>;

    /// The propagation backend behind the predictions (for provenance and
    /// derivative compatibility). The default names the adapter type.
    fn backend(&self) -> super::BackendIdentity {
        super::BackendIdentity {
            name: std::any::type_name::<Self>().to_string(),
            version: "unknown".to_string(),
            two_body_dynamics: false,
        }
    }
}

/// [`SphericalPredictor`] over a Rust [`Propagator`]: one
/// `generate_ephemeris_barycentric` crossing per prediction.
pub struct PropagatorPredictor<'a, P, T> {
    pub propagator: &'a P,
    pub options: &'a EphemerisOptions,
    pub provider: &'a dyn TimeScaleProvider,
    pub translation_provider: &'a T,
}

impl<P, T> SphericalPredictor for PropagatorPredictor<'_, P, T>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    fn backend(&self) -> super::BackendIdentity {
        self.propagator.backend()
    }

    fn predict_spherical(
        &self,
        states: &[[f64; 6]],
        geometry: &OrbitGeometry,
        observers: &ObserverBatch,
    ) -> PropagationResultValue<Result<Vec<f64>, String>> {
        predict_spherical(
            self.propagator,
            states,
            geometry,
            observers,
            self.options,
            self.provider,
            self.translation_provider,
        )
    }
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
pub(super) fn residual_lon_lat_columns(
    observed_flat: &[f64],
    predicted_flat: &[f64],
    n: usize,
) -> Vec<f64> {
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
pub(super) fn masked_arc_length(values: &[f64], mask: &[bool]) -> (f64, f64, f64) {
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

pub(super) fn filter_coordinate_batch(
    batch: &CoordinateBatch,
    keep: &[bool],
) -> PropagationResultValue<CoordinateBatch> {
    let rows: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter_map(|(index, &flag)| flag.then_some(index))
        .collect();
    take_coordinate_batch(batch, &rows)
}

pub(super) fn take_coordinate_batch(
    batch: &CoordinateBatch,
    rows: &[usize],
) -> PropagationResultValue<CoordinateBatch> {
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
            for &index in rows {
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

pub(super) fn filter_observer_batch(
    observers: &ObserverBatch,
    keep: &[bool],
) -> PropagationResultValue<ObserverBatch> {
    let rows: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter_map(|(index, &flag)| flag.then_some(index))
        .collect();
    take_observer_batch(observers, &rows)
}

pub(super) fn take_observer_batch(
    observers: &ObserverBatch,
    rows: &[usize],
) -> PropagationResultValue<ObserverBatch> {
    let codes = rows
        .iter()
        .map(|&row| observers.code[row].clone())
        .collect();
    let coordinates = take_coordinate_batch(&observers.coordinates, rows)?;
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
    use adam_core_rs_orbit_determination::gauss_iod_fused;

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

    /// Gauss on the longest-arc triplets of this synthetic geometry seeds
    /// coarsely (best reduced chi2 ~3e3 at 1" sigmas); the threshold gates
    /// those candidates in while a half-degree displaced prediction (reduced
    /// chi2 ~1e6) stays out.
    fn iod_config(contamination_percentage: f64) -> IodConfig {
        IodConfig {
            min_obs: 6,
            min_arc_length: 1.0,
            contamination_percentage,
            rchi2_threshold: 1e4,
            observation_selection_method: ObservationSelectionMethod::Combinations,
            light_time: true,
            mu: crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap(),
            speed_of_light: 173.144_632_674_240_3,
            max_triplets: None,
            velocity_method: VelocityMethod::Gibbs,
            spans: SpanSearchConfig::default(),
            acceptance: IodAcceptance::Prefit,
            correction: CorrectionConfig::default(),
        }
    }

    fn two_body_predictor<'a>(
        propagator: &'a TwoBodyPropagator,
        options: &'a EphemerisOptions,
    ) -> PropagatorPredictor<'a, TwoBodyPropagator, ZeroTranslationProvider> {
        PropagatorPredictor {
            propagator,
            options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        }
    }

    /// Fault injection around a predictor: `fail_calls` prediction calls
    /// (counting from the first) return `failure`; `garbage_calls` return
    /// the inner prediction displaced by half a degree in longitude (a
    /// finite but hopeless candidate).
    struct FaultyPredictor<'a> {
        inner: &'a dyn SphericalPredictor,
        calls: std::cell::Cell<usize>,
        fail_calls: usize,
        failure: Option<PropagationError>,
        garbage_calls: usize,
    }

    impl SphericalPredictor for FaultyPredictor<'_> {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            if call < self.fail_calls {
                return match &self.failure {
                    Some(err) => Err(err.clone()),
                    None => Ok(Err("light-time iteration failed".to_string())),
                };
            }
            let mut predicted = self.inner.predict_spherical(states, geometry, observers)?;
            if call < self.fail_calls + self.garbage_calls {
                if let Ok(values) = predicted.as_mut() {
                    for row in values.chunks_exact_mut(6) {
                        row[1] += 0.5;
                    }
                }
            }
            Ok(predicted)
        }
    }

    /// The accepted state sits at the Gauss epoch: compare with the truth
    /// propagated there. Gauss on this geometry is coarse (see
    /// `iod_config`), so the bound is loose.
    fn assert_near_truth(output: &IodOutput) {
        let mu = crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap();
        let truth_at_epoch = crate::propagate_2body_row::<f64>(
            TRUTH_STATE,
            output.epoch.expect("accepted epoch").mjd() - 60_000.0,
            mu,
            1000,
            1e-14,
        );
        for index in 0..3 {
            let error = (output.state[index] - truth_at_epoch[index]).abs();
            assert!(error < 0.05, "axis {index}: {error}");
        }
    }

    fn assert_same_orbit(a: &IodOutput, b: &IodOutput) {
        assert!(a.found && b.found);
        assert_eq!(a.state, b.state);
        assert_eq!(a.epoch, b.epoch);
        assert_eq!(a.epoch_scale, b.epoch_scale);
        assert_eq!(a.num_obs, b.num_obs);
        assert_eq!(a.chi2_total, b.chi2_total);
        assert_eq!(a.arc_length, b.arc_length);
    }

    #[test]
    fn iod_observation_order_sorts_by_epoch_then_code_then_row() {
        let times = TimeArray::new(
            TimeScale::Tdb,
            vec![
                Epoch::new(60_002, 5),
                Epoch::new(60_001, 0),
                Epoch::new(60_001, 0),
                Epoch::new(60_001, 0),
                Epoch::new(60_002, 0),
            ],
        )
        .unwrap();
        let observed = CoordinateBatch::cartesian(
            vec![[0.0; 6]; 5],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 5),
            Some(times.clone()),
            None,
        )
        .unwrap();
        let observers = ObserverBatch::new(
            ["X05", "I41", "F51", "F51", "X05"]
                .iter()
                .map(|code| ObservatoryCode(code.to_string()))
                .collect(),
            CoordinateBatch::cartesian(
                vec![[0.0; 6]; 5],
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::Named("SUN".to_string()), 5),
                Some(times),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            iod_observation_order(&observed, &observers).unwrap(),
            vec![2, 3, 1, 4, 0]
        );
    }

    #[test]
    fn iod_fit_accepts_the_noise_free_arc_in_input_order() {
        let output = iod_fit_barycentric(
            &TwoBodyPropagator::default(),
            &synthetic_observations(),
            &observers(),
            &iod_config(0.0),
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(output.found, "{:?}", output.not_found_reason);
        assert_eq!(output.num_obs, NUM_OBS);
        assert_eq!(output.triplets_tried, 1);
        assert_eq!(output.candidates_failed, 0);
        assert!(output.candidates_evaluated >= 1);
        assert!(output.not_found_reason.is_none());
        assert_eq!(output.solution.iter().filter(|&&s| s).count(), 3);
        assert_eq!(output.outlier, vec![false; NUM_OBS]);
        assert_near_truth(&output);
    }

    #[test]
    fn iod_fit_is_invariant_to_input_order() {
        let observed = synthetic_observations();
        let observers = observers();
        let permutation = [5_usize, 2, 7, 0, 3, 6, 1, 4];
        let shuffled_observed = take_coordinate_batch(&observed, &permutation).unwrap();
        let shuffled_observers = take_observer_batch(&observers, &permutation).unwrap();
        for contamination in [0.0, 20.0] {
            let config = iod_config(contamination);
            let run = |observed: &CoordinateBatch, observers: &ObserverBatch| {
                iod_fit_barycentric(
                    &TwoBodyPropagator::default(),
                    observed,
                    observers,
                    &config,
                    &ephemeris_options(),
                    &NoopProvider,
                    &ZeroTranslationProvider,
                )
                .unwrap()
            };
            let reference = run(&observed, &observers);
            let shuffled = run(&shuffled_observed, &shuffled_observers);
            assert_same_orbit(&reference, &shuffled);
            assert_eq!(reference.triplets_tried, shuffled.triplets_tried);
            assert_eq!(
                reference.candidates_evaluated,
                shuffled.candidates_evaluated
            );
            for (k, &row) in permutation.iter().enumerate() {
                assert_eq!(shuffled.solution[k], reference.solution[row]);
                assert_eq!(shuffled.outlier[k], reference.outlier[row]);
                assert_eq!(shuffled.residual_chi2[k], reference.residual_chi2[row]);
                assert_eq!(shuffled.residual_dof[k], reference.residual_dof[row]);
                assert_eq!(
                    shuffled.residuals[k * 6..(k + 1) * 6],
                    reference.residuals[row * 6..(row + 1) * 6]
                );
            }
        }
    }

    #[test]
    fn iod_fit_zero_contamination_continues_past_a_rejected_triplet() {
        let observed = synthetic_observations();
        let observers = observers();
        let config = iod_config(0.0);
        // Every finite candidate of the first triplet is made hopeless.
        let times = observed_times_mjd(&observed).unwrap();
        let first = select_observation_triplets(&times, config.observation_selection_method)[0];
        let observed_values = observed.values.spherical().unwrap();
        let observer_positions = observers.coordinates.values.cartesian().unwrap();
        let (_, states_flat) = gauss_iod_fused(
            std::array::from_fn(|slot| observed_values[first[slot]][1]),
            std::array::from_fn(|slot| observed_values[first[slot]][2]),
            std::array::from_fn(|slot| times[first[slot]]),
            std::array::from_fn(|slot| {
                let row = observer_positions[first[slot]];
                [row[0], row[1], row[2]]
            }),
            1,
            config.light_time,
            config.mu,
            config.speed_of_light,
        );
        let finite_candidates = states_flat
            .chunks_exact(6)
            .filter(|state| state.iter().all(|v| v.is_finite()))
            .count();
        assert!(finite_candidates >= 1);

        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = two_body_predictor(&propagator, &options);
        let predictor = FaultyPredictor {
            inner: &inner,
            calls: std::cell::Cell::new(0),
            fail_calls: 0,
            failure: None,
            garbage_calls: finite_candidates,
        };
        let output = iod_fit(&predictor, &observed, &observers, &config).unwrap();
        assert!(output.found, "{:?}", output.not_found_reason);
        assert!(output.triplets_tried >= 2);
        assert!(output.candidates_evaluated > finite_candidates);
        assert_eq!(output.candidates_failed, 0);
        assert_eq!(output.outlier, vec![false; NUM_OBS]);
        assert_near_truth(&output);
    }

    #[test]
    fn iod_fit_skips_a_candidate_whose_prediction_fails() {
        let observed = synthetic_observations();
        let observers = observers();
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = two_body_predictor(&propagator, &options);
        for failure in [
            Some(PropagationError::Backend(
                "backend refused the candidate".to_string(),
            )),
            Some(PropagationError::BackendProtocol("short batch".to_string())),
            Some(PropagationError::ThreadPool("no threads".to_string())),
            None, // in-band per-row failure
        ] {
            let predictor = FaultyPredictor {
                inner: &inner,
                calls: std::cell::Cell::new(0),
                fail_calls: 1,
                failure,
                garbage_calls: 0,
            };
            let output = iod_fit(&predictor, &observed, &observers, &iod_config(20.0)).unwrap();
            assert!(output.found, "{:?}", output.not_found_reason);
            assert_eq!(output.candidates_failed, 1);
            assert!(output.candidates_evaluated >= 1);
            assert_eq!(output.outlier, vec![false; NUM_OBS]);
        }
    }

    /// The materialized legacy selection: every combination, sorted by
    /// (arc descending, midpoint offset ascending) with a stable sort, then
    /// filtered to distinct epochs.
    fn legacy_combination_triplets(times: &[f64]) -> Vec<[usize; 3]> {
        let n = times.len();
        let mut rows = Vec::new();
        for first in 0..n {
            for second in first + 1..n {
                for third in second + 1..n {
                    rows.push([first, second, third]);
                }
            }
        }
        rows.sort_by(|a, b| {
            let arc_a = times[a[2]] - times[a[0]];
            let arc_b = times[b[2]] - times[b[0]];
            let mid_a = ((times[a[2]] + times[a[0]]) / 2.0 - times[a[1]]).abs();
            let mid_b = ((times[b[2]] + times[b[0]]) / 2.0 - times[b[1]]).abs();
            arc_b.total_cmp(&arc_a).then(mid_a.total_cmp(&mid_b))
        });
        rows.retain(|row| {
            times[row[0]] != times[row[1]]
                && times[row[0]] != times[row[2]]
                && times[row[1]] != times[row[2]]
        });
        rows
    }

    #[test]
    fn combination_triplets_match_the_materialized_legacy_order() {
        let cases: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 4.0, 8.0],
            // Regular cadence: many pairs share an arc, many middles tie.
            (0..9).map(|k| k as f64).collect(),
            // Coincident epochs inside and at the ends of the arc.
            vec![0.0, 0.5, 0.5, 2.0, 3.5, 3.5, 7.0, 7.0],
            vec![0.0, 0.3, 1.1, 1.7, 4.0, 4.2, 5.9, 8.8, 9.0, 13.5],
            vec![3.0, 3.0, 3.0],
        ];
        for times in cases {
            let expected = legacy_combination_triplets(&times);
            assert_eq!(
                select_observation_triplets(&times, ObservationSelectionMethod::Combinations),
                expected,
                "{times:?}"
            );
            for budget in 0..=expected.len() + 1 {
                assert_eq!(
                    select_observation_triplets_bounded(
                        &times,
                        ObservationSelectionMethod::Combinations,
                        budget
                    ),
                    expected[..budget.min(expected.len())].to_vec(),
                    "{times:?} budget {budget}"
                );
            }
        }
    }

    #[test]
    fn combination_triplets_do_not_materialize_all_combinations() {
        // 3000 observations: C(3000, 3) is 4.5e9 triplets, far beyond what
        // the legacy materialization could hold; the bounded walk only
        // touches the full-arc pair's middles.
        let n = 3000;
        let times: Vec<f64> = (0..n).map(|k| 60_000.0 + 0.25 * k as f64).collect();
        let triplets = select_observation_triplets_bounded(
            &times,
            ObservationSelectionMethod::Combinations,
            24,
        );
        assert_eq!(triplets.len(), 24);
        // Longest arc first; of its middles the two closest to the midpoint
        // tie and index order breaks the tie.
        assert_eq!(triplets[0], [0, 1499, n - 1]);
        assert_eq!(triplets[1], [0, 1500, n - 1]);
        for window in triplets.windows(2) {
            let arc = |t: &[usize; 3]| times[t[2]] - times[t[0]];
            assert!(arc(&window[0]) >= arc(&window[1]));
        }
        let distinct: std::collections::HashSet<[usize; 3]> = triplets.iter().copied().collect();
        assert_eq!(distinct.len(), 24);
    }

    /// Materialized oracle of `span_triplets`: every window, every pair in
    /// bounds sorted by (|span - target|, first, last), one middle per pair,
    /// triplets deduplicated across windows in window order.
    fn materialized_span_triplets(
        times: &[f64],
        config: &SpanSearchConfig,
    ) -> Vec<([usize; 3], (usize, usize))> {
        let nights = observation_nights(times, None);
        let n = times.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| times[a].total_cmp(&times[b]).then(a.cmp(&b)));
        let t: Vec<f64> = order.iter().map(|&row| times[row]).collect();
        let densest = |length: f64| {
            let mut best = (0, 0);
            let mut best_count = 0;
            for lo in 0..n {
                let hi = (lo..n).filter(|&k| t[k] - t[lo] <= length).max().unwrap();
                if hi - lo + 1 > best_count {
                    best_count = hi - lo + 1;
                    best = (lo, hi);
                }
            }
            best
        };
        let last_from = (0..n)
            .find(|&k| t[k] >= t[n - 1] - config.edge_window_days)
            .unwrap();
        let first_to = (0..n)
            .filter(|&k| t[k] <= t[0] + config.edge_window_days)
            .max()
            .unwrap();
        let mut windows = Vec::new();
        for w in [
            densest(config.dense_window_days[0]),
            densest(config.dense_window_days[1]),
            (last_from, n - 1),
            (0, first_to),
            (0, n - 1),
        ] {
            if !windows.contains(&w) {
                windows.push(w);
            }
        }
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (lo, hi) in windows {
            let mut pairs = Vec::new();
            for first in lo..=hi {
                for last in first + 1..=hi {
                    let span = t[last] - t[first];
                    if span > 0.0 && span >= config.min_span_days && span <= config.max_span_days {
                        pairs.push(((span - config.target_span_days).abs(), first, last));
                    }
                }
            }
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            for (_, first, last) in pairs {
                let midpoint = (t[first] + t[last]) / 2.0;
                let middle = (first + 1..last)
                    .filter(|&m| t[m] != t[first] && t[m] != t[last])
                    .min_by(|&a, &b| {
                        (t[a] - midpoint)
                            .abs()
                            .total_cmp(&(t[b] - midpoint).abs())
                            .then(a.cmp(&b))
                    });
                if let Some(middle) = middle {
                    let triplet = [order[first], order[middle], order[last]];
                    if seen.insert((nights[order[first]], nights[order[last]])) {
                        out.push((triplet, (order[lo], order[hi])));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn span_triplets_match_the_materialized_order() {
        let mut cases: Vec<(Vec<f64>, SpanSearchConfig)> = Vec::new();
        let base = SpanSearchConfig::default();
        // Nightly cadence over 80 days with a 40-day gap; tied spans.
        let nightly: Vec<f64> = (0..20)
            .map(|k| k as f64)
            .chain((60..80).map(|k| k as f64))
            .collect();
        cases.push((nightly.clone(), base));
        // Several observations per night, coincident epochs, unsorted input.
        let mut dense: Vec<f64> = (0..15)
            .flat_map(|night| [night as f64, night as f64 + 0.01, night as f64 + 0.01])
            .collect();
        dense.reverse();
        cases.push((dense, base));
        // Span bounds that exclude most pairs.
        cases.push((
            nightly.clone(),
            SpanSearchConfig {
                min_span_days: 4.0,
                max_span_days: 9.0,
                ..base
            },
        ));
        cases.push((
            vec![
                0.0, 0.3, 1.1, 1.7, 4.0, 4.2, 5.9, 8.8, 9.0, 13.5, 47.0, 51.0,
            ],
            base,
        ));
        cases.push((vec![3.0, 3.0, 3.0, 3.0], base));
        for (times, config) in cases {
            let expected = materialized_span_triplets(&times, &config);
            for budget in [0, 1, 2, 5, 17, expected.len(), expected.len() + 3] {
                assert_eq!(
                    span_triplets(&times, &observation_nights(&times, None), &config, budget),
                    expected[..budget.min(expected.len())].to_vec(),
                    "{times:?} budget {budget}"
                );
            }
            for (triplet, _) in &expected {
                let epochs: std::collections::HashSet<u64> =
                    triplet.iter().map(|&row| times[row].to_bits()).collect();
                assert_eq!(epochs.len(), 3, "coincident epochs in {triplet:?}");
            }
        }
    }

    #[test]
    fn predictors_report_the_propagator_behind_them() {
        let propagator = TwoBodyPropagator::default();
        let identity = propagator.backend();
        assert_eq!(identity.name, "adam_core.two_body");
        assert!(identity.two_body_dynamics);
        assert_eq!(identity.version, env!("CARGO_PKG_VERSION"));
        let options = ephemeris_options();
        let predictor = two_body_predictor(&propagator, &options);
        assert_eq!(predictor.backend(), identity);
    }

    #[test]
    fn observing_nights_follow_stations_not_utc_days() {
        // Station A observes across 0h UTC (a European night), then the next
        // night; station B observes during A's first night; a space
        // telescope observes every 6 hours for two days.
        let codes: Vec<crate::ObservatoryCode> = [
            "A", "A", "A", "A", "B", "S", "S", "S", "S", "S", "S", "S", "S",
        ]
        .iter()
        .map(|c| crate::ObservatoryCode(c.to_string()))
        .collect();
        let times = [
            100.80, 100.95, 101.10, 101.80, 100.90, 100.0, 100.25, 100.5, 100.75, 101.0, 101.25,
            101.5, 101.75,
        ];
        let nights = observation_nights(&times, Some(&codes));
        assert_eq!(nights[0], nights[1]);
        assert_eq!(
            nights[1], nights[2],
            "a night straddling 0h UTC is one night"
        );
        assert_ne!(nights[2], nights[3], "the next night is another");
        assert_ne!(nights[4], nights[0], "another station's night is its own");
        // Continuous coverage is cut into nights of at most 16 hours.
        let space: Vec<usize> = nights[5..].to_vec();
        assert_eq!(space[0], space[2]);
        assert_ne!(space[0], space[3]);
        assert!(space.windows(2).all(|w| w[0] <= w[1]));
        // Without stations, every row is one station.
        let single = observation_nights(&times[..4], None);
        assert_eq!(single, vec![single[0], single[0], single[0], single[3]]);
        assert_ne!(single[0], single[3]);
    }

    #[test]
    fn gauss_geometry_rotates_frames_and_refuses_other_origins() {
        let observed = synthetic_observations();
        let observers = observers();
        assert_eq!(observed.frame, Frame::Equatorial);
        let (directions, positions) = gauss_geometry(&observed, &observers).unwrap();
        let (sin_e, cos_e) = GAUSS_OBLIQUITY_RAD.sin_cos();

        // The same observations in the ecliptic frame, the observers in the
        // equatorial frame: the Gauss geometry is unchanged.
        let values = observed.values.spherical().unwrap();
        let ecliptic_values: Vec<[f64; 6]> = values
            .iter()
            .map(|row| {
                let (ra, dec) = (row[1].to_radians(), row[2].to_radians());
                let eq = [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()];
                let ec = [
                    eq[0],
                    cos_e * eq[1] + sin_e * eq[2],
                    -sin_e * eq[1] + cos_e * eq[2],
                ];
                [
                    row[0],
                    ec[1].atan2(ec[0]).to_degrees().rem_euclid(360.0),
                    ec[2].asin().to_degrees(),
                    row[3],
                    row[4],
                    row[5],
                ]
            })
            .collect();
        let ecliptic = CoordinateBatch::new(
            CoordinateValues::Spherical(ecliptic_values),
            Frame::Ecliptic,
            observed.origins.clone(),
            observed.times.clone(),
            None,
        )
        .unwrap();
        let cartesian = observers.coordinates.values.cartesian().unwrap();
        let equatorial_observers = ObserverBatch::new(
            observers.code.clone(),
            CoordinateBatch::cartesian(
                cartesian
                    .iter()
                    .map(|row| {
                        [
                            row[0],
                            cos_e * row[1] - sin_e * row[2],
                            sin_e * row[1] + cos_e * row[2],
                            row[3],
                            row[4],
                            row[5],
                        ]
                    })
                    .collect(),
                Frame::Equatorial,
                observers.coordinates.origins.clone(),
                observers.coordinates.times.clone(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let (rotated_directions, rotated_positions) =
            gauss_geometry(&ecliptic, &equatorial_observers).unwrap();
        for row in 0..NUM_OBS {
            for axis in 0..2 {
                assert!((rotated_directions[row][axis] - directions[row][axis]).abs() < 1e-10);
            }
            for axis in 0..3 {
                assert!((rotated_positions[row][axis] - positions[row][axis]).abs() < 1e-14);
            }
        }

        // Observers not centred on the Sun are refused, not used as if.
        let geocentric = ObserverBatch::new(
            observers.code.clone(),
            CoordinateBatch::cartesian(
                cartesian.to_vec(),
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::Named("EARTH".to_string()), NUM_OBS),
                observers.coordinates.times.clone(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let err = gauss_geometry(&observed, &geocentric).unwrap_err();
        assert!(err.to_string().contains("heliocentric"), "{err}");
    }

    #[test]
    fn span_triplets_prefer_the_target_span_in_the_densest_window() {
        // 3000 observations every six hours: the first triplet spans the
        // target exactly, its middle at the pair's midpoint, inside the
        // densest 10-day window (the first one, on ties).
        let n = 3000;
        let times: Vec<f64> = (0..n).map(|k| 60_000.0 + 0.25 * k as f64).collect();
        let config = SpanSearchConfig::default();
        let nights = observation_nights(&times, None);
        let triplets = span_triplets(&times, &nights, &config, config.max_triplets);
        assert_eq!(triplets.len(), config.max_triplets);
        let (first, window) = triplets[0];
        assert_eq!(times[first[2]] - times[first[0]], config.target_span_days);
        assert_eq!(
            times[first[1]] - times[first[0]],
            config.target_span_days / 2.0
        );
        assert_eq!(window, (0, 40));
        let mut night_pairs = std::collections::HashSet::new();
        for (triplet, (lo, hi)) in &triplets {
            assert!(*lo <= triplet[0] && triplet[2] <= *hi);
            // One triplet per pair of observing nights.
            assert!(night_pairs.insert((nights[triplet[0]], nights[triplet[2]])));
        }
    }

    #[test]
    fn spans_search_accepts_the_arc_and_records_windows_and_velocity() {
        let config = IodConfig {
            observation_selection_method: ObservationSelectionMethod::Spans,
            velocity_method: VelocityMethod::Auto,
            ..iod_config(0.0)
        };
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let predictor = two_body_predictor(&propagator, &options);
        let observed = synthetic_observations();
        let observers = observers();
        let result = run_iod(&predictor, &observed, &observers, &config).unwrap();
        assert!(result.found(), "{:?}", result.not_found_reason);
        let accepted = result.accepted_candidate().unwrap();
        assert!(matches!(
            accepted.velocity_method,
            VelocityMethod::Gibbs | VelocityMethod::HerrickGibbs
        ));
        let mjd = observed_times_mjd(&observed).unwrap();
        for candidate in &result.candidates {
            assert_ne!(candidate.velocity_method, VelocityMethod::Auto);
            let (lo, hi) = candidate.window;
            assert!(candidate
                .triplet
                .iter()
                .all(|&row| mjd[row] >= mjd[lo] && mjd[row] <= mjd[hi]));
        }
        assert_near_truth(&result.to_output(observed.len()));
        // Budget: the spans default when max_triplets is None.
        let none = run_iod(
            &predictor,
            &observed,
            &observers,
            &IodConfig {
                rchi2_threshold: 1e-30,
                ..config
            },
        )
        .unwrap();
        assert!(!none.found());
        assert!(none.triplets_tried <= config.spans.max_triplets);
    }

    #[test]
    fn velocity_method_setting_reaches_every_candidate() {
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let predictor = two_body_predictor(&propagator, &options);
        for method in [VelocityMethod::Gibbs, VelocityMethod::HerrickGibbs] {
            let config = IodConfig {
                velocity_method: method,
                rchi2_threshold: 1e-30,
                max_triplets: Some(4),
                ..iod_config(0.0)
            };
            let result =
                run_iod(&predictor, &synthetic_observations(), &observers(), &config).unwrap();
            assert!(!result.candidates.is_empty());
            assert!(result
                .candidates
                .iter()
                .all(|c| c.velocity_method == method));
        }
        assert_eq!(VelocityMethod::parse("auto").unwrap(), VelocityMethod::Auto);
        assert!(VelocityMethod::parse("laplace").is_err());
        assert_eq!(
            ObservationSelectionMethod::parse("spans").unwrap(),
            ObservationSelectionMethod::Spans
        );
    }

    #[test]
    fn iod_fit_search_budget_is_independent_of_contamination() {
        let observed = synthetic_observations();
        let observers = observers();
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = two_body_predictor(&propagator, &options);

        // One triplet, 20 % contamination: the outlier allowance is intact
        // (max_outliers = 1) and the search stops after one triplet.
        let config = IodConfig {
            max_triplets: Some(1),
            ..iod_config(20.0)
        };
        let output = iod_fit(&inner, &observed, &observers, &config).unwrap();
        assert!(output.found, "{:?}", output.not_found_reason);
        assert_eq!(output.triplets_tried, 1);

        // Both roots of the first triplet made hopeless: a budget of one
        // triplet is exhausted, a budget of three finds the next one.
        let hopeless = |budget: Option<usize>| {
            let predictor = FaultyPredictor {
                inner: &inner,
                calls: std::cell::Cell::new(0),
                fail_calls: 0,
                failure: None,
                garbage_calls: 2,
            };
            let config = IodConfig {
                max_triplets: budget,
                ..iod_config(20.0)
            };
            iod_fit(&predictor, &observed, &observers, &config).unwrap()
        };
        let exhausted = hopeless(Some(1));
        assert!(!exhausted.found);
        assert_eq!(exhausted.triplets_tried, 1);
        assert!(exhausted
            .not_found_reason
            .as_deref()
            .unwrap()
            .contains("none within rchi2_threshold"));
        let found = hopeless(Some(3));
        assert!(found.found, "{:?}", found.not_found_reason);
        assert_eq!(found.triplets_tried, 2);
        assert_near_truth(&found);

        // Zero contamination with its own budget still searches past the
        // legacy three triplets.
        let config = IodConfig {
            rchi2_threshold: 1e-30,
            max_triplets: Some(5),
            ..iod_config(0.0)
        };
        let output = iod_fit(&inner, &observed, &observers, &config).unwrap();
        assert!(!output.found);
        assert_eq!(output.triplets_tried, 5);

        let config = IodConfig {
            max_triplets: Some(0),
            ..iod_config(0.0)
        };
        assert!(matches!(
            iod_fit(&inner, &observed, &observers, &config),
            Err(PropagationError::InvalidRequest(_))
        ));
    }

    #[test]
    fn iod_fit_never_pairs_duplicate_epochs_in_a_triplet() {
        // The last observation duplicated: same epoch, same station, same
        // observer state, an extra row.
        let observed = synthetic_observations();
        let observers = observers();
        let rows: Vec<usize> = (0..NUM_OBS).chain([NUM_OBS - 1]).collect();
        let observed = take_coordinate_batch(&observed, &rows).unwrap();
        let observers = take_observer_batch(&observers, &rows).unwrap();
        let n = rows.len();
        let twins = [NUM_OBS - 1, NUM_OBS];
        let config = iod_config(0.0);
        let run = |observed: &CoordinateBatch, observers: &ObserverBatch| {
            iod_fit_barycentric(
                &TwoBodyPropagator::default(),
                observed,
                observers,
                &config,
                &ephemeris_options(),
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
        };
        let output = run(&observed, &observers);
        assert!(output.found, "{:?}", output.not_found_reason);
        assert_eq!(output.num_obs, n);
        assert_eq!(output.solution.len(), n);
        assert_eq!(output.solution.iter().filter(|&&s| s).count(), 3);
        // The duplicated epoch is used at most once per triplet.
        assert!(!(output.solution[twins[0]] && output.solution[twins[1]]));
        assert_eq!(output.outlier, vec![false; n]);
        assert_near_truth(&output);

        // Indistinguishable rows swapped in the input: the same orbit and
        // statistics, the duplicate epoch still used at most once.
        let permutation: Vec<usize> = [4, NUM_OBS, 1, 6, 0, NUM_OBS - 1, 3, 2, 5].to_vec();
        let swapped = run(
            &take_coordinate_batch(&observed, &permutation).unwrap(),
            &take_observer_batch(&observers, &permutation).unwrap(),
        );
        assert_same_orbit(&output, &swapped);
        let twin_rows: Vec<usize> = permutation
            .iter()
            .enumerate()
            .filter(|(_, &row)| twins.contains(&row))
            .map(|(k, _)| k)
            .collect();
        assert_eq!(twin_rows.len(), 2);
        assert!(!(swapped.solution[twin_rows[0]] && swapped.solution[twin_rows[1]]));
        assert_eq!(
            swapped.solution.iter().filter(|&&s| s).count(),
            output.solution.iter().filter(|&&s| s).count()
        );
    }

    #[test]
    fn iod_fit_scores_every_root_of_a_triplet() {
        // Every triplet of this geometry has two finite Gauss roots (the
        // true one near 1.23 au and a spurious one near 1.11 au).
        let observed = synthetic_observations();
        let observers = observers();
        let times = observed_times_mjd(&observed).unwrap();
        let config = iod_config(0.0);
        let first = select_observation_triplets(&times, config.observation_selection_method)[0];
        let observed_values = observed.values.spherical().unwrap();
        let observer_positions = observers.coordinates.values.cartesian().unwrap();
        let (_, states_flat) = gauss_iod_fused(
            std::array::from_fn(|slot| observed_values[first[slot]][1]),
            std::array::from_fn(|slot| observed_values[first[slot]][2]),
            std::array::from_fn(|slot| times[first[slot]]),
            std::array::from_fn(|slot| {
                let row = observer_positions[first[slot]];
                [row[0], row[1], row[2]]
            }),
            1,
            config.light_time,
            config.mu,
            config.speed_of_light,
        );
        let roots: Vec<[f64; 6]> = states_flat
            .chunks_exact(6)
            .filter(|state| state.iter().all(|v| v.is_finite()))
            .map(|state| std::array::from_fn(|axis| state[axis]))
            .collect();
        assert!(roots.len() >= 2, "fixture no longer yields multiple roots");

        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = two_body_predictor(&propagator, &options);

        // Control: the first root is acceptable and is taken without
        // scoring the second (first-acceptable candidate, as documented).
        let control = iod_fit(&inner, &observed, &observers, &config).unwrap();
        assert!(control.found, "{:?}", control.not_found_reason);
        assert_eq!(control.triplets_tried, 1);
        assert_eq!(control.candidates_evaluated, 1);
        assert_eq!(control.state, roots[0]);

        // The first root's prediction made hopeless: the second root of the
        // same triplet is scored before any other triplet is tried.
        let predictor = FaultyPredictor {
            inner: &inner,
            calls: std::cell::Cell::new(0),
            fail_calls: 0,
            failure: None,
            garbage_calls: 1,
        };
        let output = iod_fit(&predictor, &observed, &observers, &config).unwrap();
        assert!(output.found, "{:?}", output.not_found_reason);
        assert!(output.candidates_evaluated >= 2);
        assert_ne!(output.state, roots[0]);
        if output.triplets_tried == 1 {
            assert_eq!(output.state, roots[1]);
        }
    }

    #[test]
    fn iod_fit_reports_total_prediction_failure_as_not_found() {
        let observed = synthetic_observations();
        let observers = observers();
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = two_body_predictor(&propagator, &options);
        let predictor = FaultyPredictor {
            inner: &inner,
            calls: std::cell::Cell::new(0),
            fail_calls: usize::MAX,
            failure: Some(PropagationError::Backend(
                "the Python propagator raised an exception: boom".to_string(),
            )),
            garbage_calls: 0,
        };
        let output = iod_fit(&predictor, &observed, &observers, &iod_config(20.0)).unwrap();
        assert!(!output.found);
        assert_eq!(output.candidates_evaluated, 0);
        assert!(output.candidates_failed >= 1);
        assert_eq!(output.triplets_tried, 3 * (1 + 1));
        assert_eq!(output.solution, vec![false; NUM_OBS]);
        assert_eq!(output.outlier, vec![false; NUM_OBS]);
        let reason = output.not_found_reason.expect("reason");
        assert!(
            reason.contains("every candidate prediction failed"),
            "{reason}"
        );
        assert!(reason.contains("boom"), "{reason}");
    }

    #[test]
    fn iod_fit_propagates_caller_errors_and_explains_rejections() {
        let observed = synthetic_observations();
        let observers = observers();
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = two_body_predictor(&propagator, &options);
        let predictor = FaultyPredictor {
            inner: &inner,
            calls: std::cell::Cell::new(0),
            fail_calls: 1,
            failure: Some(PropagationError::InvalidRequest("bad request".to_string())),
            garbage_calls: 0,
        };
        assert!(matches!(
            iod_fit(&predictor, &observed, &observers, &iod_config(20.0)),
            Err(PropagationError::InvalidRequest(_))
        ));

        // Every candidate scored and rejected: the reason says so.
        let config = IodConfig {
            rchi2_threshold: 1e-30,
            ..iod_config(0.0)
        };
        let output = iod_fit(&inner, &observed, &observers, &config).unwrap();
        assert!(!output.found);
        assert_eq!(output.triplets_tried, 3);
        assert!(output.candidates_evaluated >= 3);
        let reason = output.not_found_reason.expect("reason");
        assert!(reason.contains("none within rchi2_threshold"), "{reason}");

        let output = iod_fit(
            &inner,
            &observed,
            &observers,
            &IodConfig {
                min_obs: NUM_OBS + 1,
                ..iod_config(0.0)
            },
        )
        .unwrap();
        assert!(output
            .not_found_reason
            .unwrap()
            .contains("fewer than min_obs"));
    }

    #[test]
    fn iod_triplet_selection_preserves_legacy_priority() {
        let times = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(
            select_observation_triplets(&times, ObservationSelectionMethod::FirstMiddleLast),
            vec![[0, 2, 3]]
        );
        assert!(select_observation_triplets(&times, ObservationSelectionMethod::Thirds).is_empty());
        let combinations =
            select_observation_triplets(&times, ObservationSelectionMethod::Combinations);
        assert_eq!(combinations[0], [0, 2, 3]);
        assert_eq!(combinations.len(), 4);
    }

    #[test]
    fn iod_triplet_selection_rejects_coincident_epochs() {
        let times = [1.0, 1.0, 2.0];
        assert!(
            select_observation_triplets(&times, ObservationSelectionMethod::Combinations)
                .is_empty()
        );
        assert!(
            select_observation_triplets(&times, ObservationSelectionMethod::FirstMiddleLast)
                .is_empty()
        );
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
