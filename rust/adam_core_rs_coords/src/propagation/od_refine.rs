//! Backend-generic whitened-residual differential correction (the Rust
//! orchestration behind `adam_core.orbit_determination.fit_least_squares`).
//!
//! [`fit_orbit_whitened_barycentric`] runs the complete core-owned fit in
//! one call, generic over the [`Propagator`] trait like the drivers in
//! `od.rs`: residuals are the whitened (cos-lat corrected, inverse-Cholesky
//! scaled) angular residuals predicted through the backend-generic
//! barycentric ephemeris workflow; the Jacobian is either the exact 2-body
//! forward-mode autodiff Jacobian (`whitened_2body_jacobian`, the analytic
//! default), a central difference through the propagator, or scipy's
//! 2-point forward difference; the loss is linear or Huber (iteratively
//! reweighted, scipy's `least_squares` scaling convention); the covariance is
//! `inv(JᵀJ)` of the robust-cost Jacobian, validated by the weak-direction
//! probe with the central-difference fallback; and the final
//! `evaluate_orbits`-style pass over the full observation set is part of the
//! same adam-core workflow.
//!
//! The optimizer is the trust-region Gauss-Newton method of scipy's
//! `least_squares(method="trf", x_scale="jac")` without bounds, the solver
//! the Python baseline used: an exact trust-region step on the scaled
//! normal equations, the radius driven by the actual-to-predicted reduction
//! ratio, scipy's `ftol` / `xtol` / `gtol` stopping rules on the scaled
//! gradient and the unscaled step, and a non-finite guard that never
//! reports convergence. The contract is spelled out on
//! [`WhitenedFitConfig`]. It converges to the same minimum as scipy to
//! solver tolerance, not to bit-identical iterates; the covariance, the
//! residuals and every reported statistic are functions of the solution
//! only.

use super::ephemeris::EphemerisOptions;
use super::od::{
    evaluate_orbit, filter_coordinate_batch, filter_observer_batch, observed_covariance_flat,
    residual_lon_lat_columns, spherical_flat, FitEvaluation, OrbitGeometry, PropagatorPredictor,
    SphericalPredictor,
};
use super::{PropagationError, PropagationResultValue, Propagator};
use crate::differential_correction::{
    observation_whitening_matrices, robust_cost, robust_jacobian_scale, robust_weights,
    whiten_residual_pairs, whitened_2body_jacobian, LossType, TwoBodyJacobianTerms,
    TwoBodyModelConfig, HUBER_F_SCALE_DEFAULT,
};
use crate::observation_uncertainty::{ObservationUncertaintyModel, OrbitDeterminationAstrometry};
use crate::orbit_least_squares::inverse_6x6;
use crate::translation::{
    deduplicated_origin_translation_vectors, normalize_coordinates_to, OriginTranslationProvider,
};
use crate::types::time::TimeScaleProvider;
use crate::types::{origin_mu_au3_day2, Frame, OriginArray, OriginId, TimeScale};
use crate::{CoordinateBatch, ObserverBatch, OrbitBatch};

/// Window of acceptable measured delta-cost at a 1-sigma displacement along
/// the covariance's weakest direction (`_DELTA_CHI2_WINDOW`).
pub const DELTA_CHI2_WINDOW: (f64, f64) = (0.1, 10.0);

/// How the residual Jacobian is obtained for the solver and the covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianMethod {
    /// Derivatives that follow the selected dynamics (the default): the
    /// exact two-body Jacobian when the backend declares two-body dynamics,
    /// the backend's own state-transition matrices when it supplies them
    /// (`Propagator`), central differences through the full predictor
    /// otherwise. Resolved once per fit.
    Auto,
    /// Exact 2-body state-transition Jacobian by forward-mode autodiff (exact
    /// only for two-body dynamics).
    Analytic,
    /// The propagation backend's state-transition matrices (e.g. ASSIST's
    /// variational equations) chained through the exact light-time and
    /// projection derivatives: exact for the backend's dynamics.
    Propagator,
    /// Central differences through the full residual pipeline.
    Central,
    /// scipy's forward-difference `"2-point"` Jacobian.
    TwoPoint,
}

impl JacobianMethod {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "analytic" => Ok(Self::Analytic),
            "propagator" => Ok(Self::Propagator),
            "central" => Ok(Self::Central),
            "2-point" => Ok(Self::TwoPoint),
            _ => Err(format!(
                "jacobian must be one of 'auto', 'analytic', 'propagator', 'central', \
                 '2-point'; got {value:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Analytic => "analytic",
            Self::Propagator => "propagator",
            Self::Central => "central",
            Self::TwoPoint => "2-point",
        }
    }

    /// The concrete method for `predictor` (see [`JacobianMethod::Auto`]).
    pub fn resolve(self, predictor: &dyn SphericalPredictor) -> Self {
        match self {
            Self::Auto if predictor.backend().two_body_dynamics => Self::Analytic,
            Self::Auto if predictor.supports_stm() => Self::Propagator,
            Self::Auto => Self::Central,
            other => other,
        }
    }
}

/// Settings of one whitened differential correction. Defaults mirror
/// `fit_least_squares` (`xtol = ftol = gtol = 1e-12`, linear loss,
/// `f_scale = 1.345`, analytic Jacobian, covariance validation on).
///
/// # The refinement contract
///
/// One meaning, shared by `fit_least_squares`, the worst-residual loop
/// (`iterative_fit`) and the CMC2003 loop (`cmc2003_fit`):
///
/// * **Objective.** The residual vector is the whitened (cos-latitude
///   corrected, inverse-Cholesky scaled) `(lon, lat)` residual pairs of the
///   included observations, `observed - predicted`. The minimized cost is
///   `sum(r^2)` (chi2) for the linear loss and `f_scale^2 * sum(rho(r^2 /
///   f_scale^2))` for Huber, twice scipy's `cost`. Reported `chi2` /
///   `reduced_chi2` are always the plain chi2 of the included observations.
/// * **Derivatives.** `jacobian` selects the exact two-body autodiff
///   Jacobian (`Analytic`), central differences through the predictor
///   (`Central`, 12 evaluations per iteration) or forward differences
///   (`TwoPoint`, 6). With any predictor that is not two-body the analytic
///   Jacobian is inexact and the solver converges to that Jacobian's fixed
///   point, not to the cost minimum; `Central` is exact for any predictor.
///   When the analytic Jacobian turns non-finite at an iterate whose
///   residuals are finite (two-body STM on very long arcs: 3200 Phaethon,
///   15,000 days), the fit continues from that iterate with `Central`
///   (with a warning and only the remaining shared `max_nfev` /
///   `max_iterations` budget), and the covariance uses `Central` too.
/// * **Scaling.** Marquardt / scipy `x_scale="jac"`: each parameter is
///   scaled by the (non-decreasing) norm of its Jacobian column, with 1 for a
///   zero column; the trust region lives in scaled coordinates with initial
///   radius `||scale * x0||` (1 when zero).
/// * **Step and acceptance.** Each iteration solves the exact trust-region
///   subproblem of the scaled Gauss-Newton model (Moré–Sorensen Newton
///   iteration on the eigen-decomposition of the scaled normal matrix). A
///   trial is accepted when the cost decreases. The radius becomes
///   `0.25 * ||step||` when the actual-to-predicted reduction ratio is below
///   0.25, doubles when the ratio exceeds 0.75 and the step hit the boundary,
///   and is otherwise kept. A trial whose prediction fails or is non-finite
///   is rejected with the same shrink. Rejected trials never move the
///   iterate.
/// * **Termination (`status_code`, `converged = status_code > 0`).**
///   `1` scaled-gradient infinity norm `max_k |g_k| / scale_k <= gtol`,
///   tested before each step; `2` an accepted step reduced the cost by less
///   than `ftol * cost` with ratio above 0.25; `3` an accepted step, or a
///   rejected step after the region shrank, was below `xtol * (xtol +
///   ||x||)`; `4` both 2 and 3; `0` an iteration or evaluation limit was
///   reached first; `-1` no step of the attempt sequence decreased the cost
///   while the region was still above the `xtol` resolution (stalled);
///   `-2` the Jacobian, the gradient or the normal matrix was non-finite at
///   the current iterate (for `Analytic`, under the central-difference
///   continuation as well, or with non-finite residuals). A non-finite quantity never yields a positive
///   status.
/// * **Limits.** `max_iterations` is one shared optimizer-iteration budget and
///   `max_nfev` is one shared residual-vector evaluation budget for the whole
///   solve, including analytic-to-central fallback and the approximate then
///   exact phases of a propagator Jacobian. The latter counts the initial
///   residual, every finite-difference column, and every accepted or rejected
///   trial. No phase receives a fresh nominal budget. `iterations` retains its
///   legacy meaning (residual evaluations); `optimizer_iterations` reports
///   actual Jacobian/step iterations. Reaching either limit returns the current
///   iterate with status 0 unless a stopping rule was already met.
/// * **Covariance.** `inv(J^T J)` of the robust-cost Jacobian at the
///   solution, NaN when singular (with a warning), validated by the
///   weak-direction probe when `validate_covariance` (central-difference
///   fallback for the analytic Jacobian). A singular or ill-conditioned fit
///   therefore keeps its state and statistics and loses only the covariance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhitenedFitConfig {
    pub loss: LossType,
    pub f_scale: f64,
    pub jacobian: JacobianMethod,
    pub validate_covariance: bool,
    pub xtol: f64,
    pub ftol: f64,
    pub gtol: f64,
    /// Shared maximum optimizer iterations across every solver phase.
    pub max_iterations: usize,
    /// Shared solver residual-vector evaluation budget. Includes each phase's
    /// initial state, rejected trials and every finite-difference column (not
    /// just batch crossings). Final covariance/probe/reporting work is separate.
    pub max_nfev: Option<usize>,
    /// Universal-Kepler / light-time settings of the analytic 2-body model.
    pub two_body: TwoBodyModelConfig,
}

impl Default for WhitenedFitConfig {
    fn default() -> Self {
        Self {
            loss: LossType::Linear,
            f_scale: HUBER_F_SCALE_DEFAULT,
            jacobian: JacobianMethod::Auto,
            validate_covariance: true,
            xtol: 1e-12,
            ftol: 1e-12,
            gtol: 1e-12,
            max_iterations: 100,
            max_nfev: None,
            two_body: TwoBodyModelConfig::default(),
        }
    }
}

impl WhitenedFitConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.f_scale.is_finite() || self.f_scale <= 0.0 {
            return Err(format!(
                "f_scale must be a positive finite number; got {:?}",
                self.f_scale
            ));
        }
        for (name, value) in [
            ("xtol", self.xtol),
            ("ftol", self.ftol),
            ("gtol", self.gtol),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be a finite non-negative number"));
            }
        }
        if self.max_iterations == 0 {
            return Err("max_iterations must be positive".to_string());
        }
        if self.max_nfev == Some(0) {
            return Err("max_nfev must be positive or None".to_string());
        }
        if self.two_body.propagation_max_iter == 0
            || self.two_body.ephemeris_max_iter == 0
            || self.two_body.max_lt_iter == 0
        {
            return Err("two-body iteration budgets must be positive".to_string());
        }
        for (name, value) in [
            ("propagation_tol", self.two_body.propagation_tol),
            ("lt_tol", self.two_body.lt_tol),
            ("ephemeris_tol", self.two_body.ephemeris_tol),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("two_body.{name} must be positive and finite"));
            }
        }
        Ok(())
    }
}

/// Product of one whitened differential correction: the solution, its
/// (validated) covariance, solver diagnostics, per-observation weights and
/// the fused evaluation over the full observation set.
#[derive(Debug, Clone)]
pub struct WhitenedFitOutput {
    /// Fitted Cartesian state at the orbit epoch, in the input orbit's
    /// frame/origin.
    pub state: [f64; 6],
    /// Row-major 6x6 parameter covariance of the robust cost at the solution
    /// (NaN when it could not be computed).
    pub covariance: [f64; 36],
    /// Residual-vector evaluations spent by the solver, including numerical
    /// Jacobians. Retains the legacy field name; excludes final diagnostics.
    /// `max_nfev` bounds this count; see [`WhitenedFitConfig`].
    pub iterations: usize,
    /// Actual optimizer iterations (Jacobian/step iterations), shared across
    /// approximate, exact, and fallback phases.
    pub optimizer_iterations: usize,
    /// Whether a convergence stopping rule was met within both solver limits
    /// (`status_code > 0`; see the contract on [`WhitenedFitConfig`]).
    pub converged: bool,
    /// scipy-style status: 0 = iteration/evaluation limit, 1 = gtol, 2 = ftol, 3 = xtol,
    /// 4 = both ftol and xtol, -1 = step rejected (stalled).
    pub status_code: i64,
    /// Minimized objective at the solution (chi2 for the linear loss, the
    /// robust cost otherwise).
    pub cost: f64,
    /// Plain (unweighted) chi2 of the included observations at the solution.
    pub fit_chi2: f64,
    /// `(N,)` effective weight of every observation in the solution: 0 for
    /// ignored rows, otherwise the smaller of the two per-component IRLS
    /// weights (all ones for the linear loss).
    pub weights: Vec<f64>,
    /// `(2M,)` whitened residual components of the included observations at
    /// the solution.
    pub residuals_whitened: Vec<f64>,
    /// `evaluate_orbits`-style statistics over the full observation set.
    pub evaluation: FitEvaluation,
    /// Diagnostics the Python veneer re-emits as `RuntimeWarning`s, in order.
    pub warnings: Vec<String>,
}

/// The included subset of a whitened least-squares problem together with
/// everything needed to evaluate its residual vector and Jacobians.
pub(super) struct WhitenedProblem<'a> {
    predictor: &'a dyn SphericalPredictor,
    geometry: OrbitGeometry,
    epoch_mjd_tdb: f64,
    observed_flat: Vec<f64>,
    observers: ObserverBatch,
    whiteners: Vec<[f64; 4]>,
    analytic_terms: Option<TwoBodyJacobianTerms>,
    /// `config.jacobian` resolved for the predictor (never `Auto`).
    pub(super) jacobian_method: JacobianMethod,
    two_body: TwoBodyModelConfig,
    m: usize,
}

impl<'a> WhitenedProblem<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new<T: OriginTranslationProvider>(
        predictor: &'a dyn SphericalPredictor,
        orbit: &OrbitBatch,
        observed: &CoordinateBatch,
        observers: &ObserverBatch,
        config: &WhitenedFitConfig,
        provider: &dyn TimeScaleProvider,
        translation_provider: &T,
    ) -> PropagationResultValue<Self> {
        let m = observed.len();
        super::od::validate_od_observations(observed, observers, true)?;
        let distinct_epochs = super::od::distinct_observation_epochs(observed)?;
        if distinct_epochs < 3 {
            return Err(PropagationError::InvalidRequest(format!(
                "rank-deficient fit: {distinct_epochs} distinct observation epochs; at least 3 are required"
            )));
        }
        if m == 0 {
            return Err(PropagationError::InvalidRequest(
                "least-squares fitting requires at least one included observation".to_string(),
            ));
        }
        let geometry = OrbitGeometry::from_orbit(orbit)?;
        let epoch_mjd_tdb = crate::TimeArray::new(geometry.scale, vec![geometry.epoch])?
            .rescale_with_provider(TimeScale::Tdb, provider)?
            .mjd_values()[0];
        let observed_flat = spherical_flat(observed, "observed")?;
        for row in 0..m {
            let block = &observed_flat[row * 6..(row + 1) * 6];
            if block[0].is_finite() || block[3..6].iter().any(|v| v.is_finite()) {
                return Err(PropagationError::InvalidRequest(
                    "fit_least_squares only supports angular (lon, lat) observations; \
                     found finite values in rho or velocity dimensions."
                        .to_string(),
                ));
            }
        }
        let observed_cov = observed_covariance_flat(observed)?;
        let lat_deg: Vec<f64> = (0..m).map(|row| observed_flat[row * 6 + 2]).collect();
        let whiteners = observation_whitening_matrices(&observed_cov, &lat_deg)
            .map_err(|err| PropagationError::InvalidRequest(err.to_string()))?;

        let jacobian_method = config.jacobian.resolve(predictor);
        let analytic_terms = if matches!(
            jacobian_method,
            JacobianMethod::Analytic | JacobianMethod::Propagator
        ) {
            Some(analytic_jacobian_terms(
                &geometry,
                observers,
                &lat_deg,
                &whiteners,
                provider,
                translation_provider,
            )?)
        } else {
            None
        };

        Ok(Self {
            predictor,
            geometry,
            epoch_mjd_tdb,
            observed_flat,
            observers: observers.clone(),
            whiteners,
            analytic_terms,
            jacobian_method,
            two_body: config.two_body,
            m,
        })
    }

    /// Whitened residual vectors (`2M` each) of `states`, predicted in one
    /// ephemeris crossing. Outer `Err` is a request failure; inner `Err` a
    /// per-row numerical failure of the backend.
    pub(super) fn residuals(
        &self,
        states: &[[f64; 6]],
    ) -> PropagationResultValue<Result<Vec<Vec<f64>>, String>> {
        let predicted =
            match self
                .predictor
                .predict_spherical(states, &self.geometry, &self.observers)?
            {
                Ok(values) => values,
                Err(message) => return Ok(Err(message)),
            };
        let stride = self.m * 6;
        let mut out = Vec::with_capacity(states.len());
        for candidate in 0..states.len() {
            let block = &predicted[candidate * stride..(candidate + 1) * stride];
            let pairs = residual_lon_lat_columns(&self.observed_flat, block, self.m);
            out.push(whiten_residual_pairs(&self.whiteners, &pairs));
        }
        Ok(Ok(out))
    }

    fn residuals_one(&self, state: &[f64; 6]) -> PropagationResultValue<Vec<f64>> {
        self.residuals(&[*state])?
            .map(|mut batch| batch.swap_remove(0))
            .map_err(PropagationError::RowFailure)
    }

    /// `(2M, 6)` row-major Jacobian of the whitened residual vector.
    fn jacobian(
        &self,
        method: JacobianMethod,
        state: &[f64; 6],
        base: &[f64],
    ) -> PropagationResultValue<Vec<f64>> {
        match method {
            JacobianMethod::Auto => self.jacobian(self.jacobian_method, state, base),
            JacobianMethod::Analytic => {
                let terms = self.analytic_terms.as_ref().ok_or_else(|| {
                    PropagationError::InvalidRequest(
                        "analytic Jacobian terms were not prepared".to_string(),
                    )
                })?;
                whitened_2body_jacobian(*state, self.epoch_mjd_tdb, terms, &self.two_body)
                    .map_err(PropagationError::RowFailure)
            }
            JacobianMethod::Propagator => {
                let terms = self.analytic_terms.as_ref().ok_or_else(|| {
                    PropagationError::InvalidRequest(
                        "propagator Jacobian terms were not prepared".to_string(),
                    )
                })?;
                if self.geometry.frame != crate::types::Frame::Ecliptic
                    || !super::ephemeris::is_sun_origin(&self.geometry.origin)
                {
                    return Err(PropagationError::InvalidRequest(
                        "jacobian='propagator' needs a heliocentric ecliptic orbit".to_string(),
                    ));
                }
                let epoch = super::od::mjd_epoch(self.epoch_mjd_tdb);
                let times: Vec<crate::Epoch> = terms
                    .times_mjd_tdb
                    .iter()
                    .map(|&t| super::od::mjd_epoch(t))
                    .collect();
                let rows = self.predictor.propagate_stm(
                    *state,
                    self.geometry.active_non_gravitational_parameters(),
                    epoch,
                    &times,
                )?;
                if rows.len() != times.len() || rows.iter().any(|row| !row.valid) {
                    // A failed row: non-finite, so the solver's non-finite
                    // handling (central-difference continuation) applies.
                    return Ok(vec![f64::NAN; 2 * self.m * 6]);
                }
                let states: Vec<[f64; 6]> = rows.iter().map(|row| row.state).collect();
                let stms: Vec<[f64; 36]> = rows.iter().map(|row| row.stm).collect();
                crate::differential_correction::whitened_stm_jacobian(
                    &states,
                    &stms,
                    terms,
                    &self.two_body,
                )
                .map_err(PropagationError::RowFailure)
            }
            JacobianMethod::Central => {
                // h_k = 1e-6 * max(|x_k|, 1e-3): `_central_difference_jacobian`.
                let mut states = Vec::with_capacity(12);
                let mut steps = [0.0_f64; 6];
                for k in 0..6 {
                    let step = 1e-6 * state[k].abs().max(1e-3);
                    steps[k] = step;
                    let mut plus = *state;
                    let mut minus = *state;
                    plus[k] += step;
                    minus[k] -= step;
                    states.push(plus);
                    states.push(minus);
                }
                let batch = self
                    .residuals(&states)?
                    .map_err(PropagationError::RowFailure)?;
                let rows = 2 * self.m;
                let mut jac = vec![0.0_f64; rows * 6];
                for k in 0..6 {
                    let plus = &batch[2 * k];
                    let minus = &batch[2 * k + 1];
                    for i in 0..rows {
                        jac[i * 6 + k] = (plus[i] - minus[i]) / (2.0 * steps[k]);
                    }
                }
                Ok(jac)
            }
            JacobianMethod::TwoPoint => {
                // scipy `2-point`: h = sqrt(eps) * sign(x) * max(1, |x|).
                let mut states = Vec::with_capacity(6);
                let mut steps = [0.0_f64; 6];
                for k in 0..6 {
                    let sign = if state[k] >= 0.0 { 1.0 } else { -1.0 };
                    let h = 1.490_116_119_384_765_6e-8 * sign * state[k].abs().max(1.0);
                    let mut plus = *state;
                    plus[k] += h;
                    steps[k] = plus[k] - state[k];
                    states.push(plus);
                }
                let batch = self
                    .residuals(&states)?
                    .map_err(PropagationError::RowFailure)?;
                let rows = 2 * self.m;
                let mut jac = vec![0.0_f64; rows * 6];
                for k in 0..6 {
                    let plus = &batch[k];
                    for i in 0..rows {
                        jac[i * 6 + k] = (plus[i] - base[i]) / steps[k];
                    }
                }
                Ok(jac)
            }
        }
    }
}

/// Observation-dependent constants of the analytic 2-body Jacobian
/// (`_analytic_jacobian_terms`): observation times in TDB, barycentric
/// ecliptic observer and Sun states, cos(latitude) and the whiteners.
fn analytic_jacobian_terms<T: OriginTranslationProvider>(
    geometry: &OrbitGeometry,
    observers: &ObserverBatch,
    lat_deg: &[f64],
    whiteners: &[[f64; 4]],
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<TwoBodyJacobianTerms> {
    let sun = OriginId::from_code("SUN");
    if geometry.origin != sun || geometry.frame != Frame::Ecliptic {
        return Err(PropagationError::InvalidRequest(
            "the analytic Jacobian requires a heliocentric ecliptic orbit".to_string(),
        ));
    }
    let times =
        observers.coordinates.times.as_ref().ok_or_else(|| {
            PropagationError::InvalidRequest("observers require epochs".to_string())
        })?;
    let times_mjd_tdb = times
        .rescale_with_provider(TimeScale::Tdb, provider)?
        .mjd_values();
    let barycentric = normalize_coordinates_to(
        &observers.coordinates,
        &OriginId::SolarSystemBarycenter,
        Frame::Ecliptic,
        translation_provider,
    )?;
    let observer_states = barycentric
        .values
        .cartesian()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "observer coordinates must be Cartesian for orbit determination".to_string(),
            )
        })?
        .to_vec();
    let sun_states = deduplicated_origin_translation_vectors(
        translation_provider,
        &OriginArray::repeat(sun.clone(), observers.len()),
        &OriginId::SolarSystemBarycenter,
        Frame::Ecliptic,
        times,
    )?;
    let mu_helio = origin_mu_au3_day2(&sun).map_err(|err| {
        PropagationError::InvalidRequest(format!("unknown gravitational parameter: {err}"))
    })?;
    let mu_bary = origin_mu_au3_day2(&OriginId::SolarSystemBarycenter).map_err(|err| {
        PropagationError::InvalidRequest(format!("unknown gravitational parameter: {err}"))
    })?;
    Ok(TwoBodyJacobianTerms {
        times_mjd_tdb,
        observer_states,
        sun_states,
        cos_lats: lat_deg.iter().map(|lat| lat.to_radians().cos()).collect(),
        whiteners: whiteners.to_vec(),
        mu_helio,
        mu_bary,
    })
}

// ---------------------------------------------------------------------------
// Dense 6-parameter linear algebra
// ---------------------------------------------------------------------------

/// `JᵀJ` and `Jᵀf` of a `(rows, 6)` row-major Jacobian.
#[allow(clippy::needless_range_loop)]
fn normal_equations(jac: &[f64], f: &[f64]) -> ([[f64; 6]; 6], [f64; 6]) {
    let rows = f.len();
    let mut a = [[0.0_f64; 6]; 6];
    let mut g = [0.0_f64; 6];
    for i in 0..rows {
        let row = &jac[i * 6..(i + 1) * 6];
        for k in 0..6 {
            g[k] += row[k] * f[i];
            for l in k..6 {
                a[k][l] += row[k] * row[l];
            }
        }
    }
    for k in 0..6 {
        for l in 0..k {
            a[k][l] = a[l][k];
        }
    }
    (a, g)
}

/// Cyclic Jacobi eigen-decomposition of a symmetric 6x6 matrix: eigenvalues
/// and the matching eigenvectors as columns of `v` (`v[i][k]` is component
/// `i` of eigenvector `k`).
#[allow(clippy::needless_range_loop)]
pub fn symmetric_eigen_6x6(a: &[[f64; 6]; 6]) -> ([f64; 6], [[f64; 6]; 6]) {
    let mut m = *a;
    let mut v = [[0.0_f64; 6]; 6];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
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
                for k in 0..6 {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut eigenvalues = [0.0_f64; 6];
    for k in 0..6 {
        eigenvalues[k] = m[k][k];
    }
    (eigenvalues, v)
}

fn covariance_matrix(covariance: &[f64; 36]) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for (i, row) in out.iter_mut().enumerate() {
        row.copy_from_slice(&covariance[i * 6..(i + 1) * 6]);
    }
    out
}

/// Iteratively-reweighted least squares: rows of the residual Jacobian and
/// the residuals are multiplied by `sqrt(rho'(r))`, so the Gauss-Newton step
/// solves `Jᵀ W J δ = -Jᵀ W r` with `W = diag(rho')` (all ones for the linear
/// loss). The stationary point `Jᵀ psi(r) = 0` is the M-estimate scipy's
/// solver converges to, and the system stays well conditioned from any start
/// (scipy's curvature scaling floors tail rows at machine epsilon, which
/// leaves nothing to damp against when every residual starts in the tail).
fn irls_problem(
    jac: &[f64],
    residuals: &[f64],
    loss: LossType,
    f_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let weights = robust_weights(residuals, loss, f_scale);
    let rows = residuals.len();
    let mut jac_weighted = jac.to_vec();
    let mut f_weighted = Vec::with_capacity(rows);
    for i in 0..rows {
        let root = weights[i].sqrt();
        for k in 0..6 {
            jac_weighted[i * 6 + k] *= root;
        }
        f_weighted.push(residuals[i] * root);
    }
    (jac_weighted, f_weighted)
}

/// scipy's robust-loss scaling: rows of the residual Jacobian are multiplied
/// by `sqrt(rho' + 2 rho'' r²)` and the residuals by `rho'` over that factor,
/// so `JᵀJ` approximates the Hessian of the robust cost.
fn scaled_problem(
    jac: &[f64],
    residuals: &[f64],
    loss: LossType,
    f_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let scale = robust_jacobian_scale(residuals, loss, f_scale);
    let weights = robust_weights(residuals, loss, f_scale);
    let rows = residuals.len();
    let mut jac_scaled = jac.to_vec();
    let mut f_scaled = Vec::with_capacity(rows);
    for i in 0..rows {
        for k in 0..6 {
            jac_scaled[i * 6 + k] *= scale[i];
        }
        f_scaled.push(residuals[i] * weights[i] / scale[i]);
    }
    (jac_scaled, f_scaled)
}

/// `inv(JᵀJ)` of the robust-cost Jacobian at `residuals`.
fn covariance_from_jacobian(
    jac: &[f64],
    residuals: &[f64],
    loss: LossType,
    f_scale: f64,
) -> Option<[f64; 36]> {
    let (jac_scaled, _) = scaled_problem(jac, residuals, loss, f_scale);
    let (a, _) = normal_equations(&jac_scaled, residuals);
    inverse_6x6(&a)
        .filter(|cov| cov.iter().all(|v| v.is_finite()))
        .map(canonicalize_covariance_symmetry)
}

fn canonicalize_covariance_symmetry(mut covariance: [f64; 36]) -> [f64; 36] {
    // Numerical elimination can round the two triangles differently on weak
    // short arcs. A covariance is symmetric by construction; canonicalize
    // that identity before structural validation.
    for row in 0..6 {
        for col in row + 1..6 {
            let value = 0.5 * (covariance[row * 6 + col] + covariance[col * 6 + row]);
            covariance[row * 6 + col] = value;
            covariance[col * 6 + row] = value;
        }
    }
    covariance
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

struct Solution {
    state: [f64; 6],
    residuals: Vec<f64>,
    cost: f64,
    /// Every residual-vector evaluation, Jacobian columns included.
    nfev: usize,
    /// Jacobian/step iterations actually entered.
    optimizer_iterations: usize,
    status: i64,
}

/// Solver exit codes; see the contract on [`WhitenedFitConfig`].
const STATUS_LIMIT: i64 = 0;
const STATUS_GTOL: i64 = 1;
const STATUS_FTOL: i64 = 2;
const STATUS_XTOL: i64 = 3;
const STATUS_FTOL_XTOL: i64 = 4;
const STATUS_STALLED: i64 = -1;
const STATUS_NON_FINITE: i64 = -2;

/// Trial steps attempted within one iteration before the solver reports a
/// stall. With the radius shrinking fourfold per rejection this spans far
/// more than the dynamic range between the initial radius and the `xtol`
/// resolution.
const MAX_STEP_ATTEMPTS: usize = 64;

fn norm6(v: &[f64; 6]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Exact trust-region step of the scaled Gauss-Newton model (scipy's
/// `solve_lsq_trust_region`, on the eigen-decomposition of the scaled normal
/// matrix instead of the SVD of the scaled Jacobian): in the eigenbasis
/// `p(alpha)_i = -q_i / (lambda_i + alpha)`, with `alpha >= 0` the smallest
/// value giving `||p|| <= radius`, found by Moré–Sorensen's safeguarded
/// Newton iteration on `||p(alpha)|| - radius`. Returns the step in the
/// eigenbasis and the `alpha` used (0 for the Gauss-Newton step).
fn trust_region_step(
    eigenvalues: &[f64; 6],
    q: &[f64; 6],
    radius: f64,
    initial_alpha: f64,
) -> ([f64; 6], f64) {
    let radius = radius.max(f64::MIN_POSITIVE);
    let q_norm = norm6(q);
    if q_norm == 0.0 {
        return ([0.0; 6], 0.0);
    }
    let lambda_max = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let lambda_min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let full_rank = lambda_max > 0.0 && lambda_min > f64::EPSILON * f64::EPSILON * lambda_max;
    let p_of =
        |alpha: f64| -> [f64; 6] { std::array::from_fn(|i| -q[i] / (eigenvalues[i] + alpha)) };
    if full_rank {
        let p = p_of(0.0);
        if norm6(&p) <= radius {
            return (p, 0.0);
        }
    }
    let phi_and_derivative = |alpha: f64| -> (f64, f64) {
        let p_norm = norm6(&p_of(alpha));
        let phi = p_norm - radius;
        let phi_prime = -(0..6)
            .map(|i| q[i] * q[i] / (eigenvalues[i] + alpha).powi(3))
            .sum::<f64>()
            / p_norm;
        (phi, phi_prime)
    };
    let mut alpha_upper = q_norm / radius;
    let mut alpha_lower = if full_rank {
        let (phi, phi_prime) = phi_and_derivative(0.0);
        -phi / phi_prime
    } else {
        0.0
    };
    let fallback = |lower: f64, upper: f64| (0.001 * upper).max((lower * upper).sqrt());
    let mut alpha = initial_alpha;
    if alpha.is_nan() || alpha <= 0.0 || alpha < alpha_lower || alpha > alpha_upper {
        alpha = fallback(alpha_lower, alpha_upper);
    }
    for _ in 0..10 {
        if alpha < alpha_lower || alpha > alpha_upper {
            alpha = fallback(alpha_lower, alpha_upper);
        }
        let (phi, phi_prime) = phi_and_derivative(alpha);
        if phi < 0.0 {
            alpha_upper = alpha;
        }
        let ratio = phi / phi_prime;
        alpha_lower = alpha_lower.max(alpha - ratio);
        alpha -= (phi + radius) * ratio / radius;
        if phi.abs() < 0.01 * radius {
            break;
        }
    }
    let mut p = p_of(alpha);
    let p_norm = norm6(&p);
    if p_norm > 0.0 {
        // The iteration stops within 1 % of the radius; land exactly on it.
        let factor = radius / p_norm;
        for value in p.iter_mut() {
            *value *= factor;
        }
    }
    (p, alpha)
}

fn remaining_fit_config(
    config: &WhitenedFitConfig,
    used_iterations: usize,
    used_nfev: usize,
    jacobian: JacobianMethod,
) -> Option<WhitenedFitConfig> {
    let max_iterations = config.max_iterations.saturating_sub(used_iterations);
    let max_nfev = config.max_nfev.map(|limit| limit.saturating_sub(used_nfev));
    if max_iterations == 0 || max_nfev == Some(0) {
        None
    } else {
        Some(WhitenedFitConfig {
            jacobian,
            max_iterations,
            max_nfev,
            ..*config
        })
    }
}

/// Trust-region Gauss-Newton on the robust cost: scipy's `least_squares`
/// `method="trf"` without bounds and with `x_scale="jac"`, the solver the
/// Python baseline used. See the contract on [`WhitenedFitConfig`].
/// [`solve`], continued with central differences from the iterate where an
/// analytic Jacobian turned non-finite (residuals still finite); returns the
/// Jacobian method in effect at the solution.
fn solve_with_fallback(
    problem: &WhitenedProblem<'_>,
    initial_state: [f64; 6],
    config: &WhitenedFitConfig,
    warnings: &mut Vec<String>,
) -> PropagationResultValue<(Solution, JacobianMethod)> {
    // The backend's state-transition matrices are exact but cost a
    // variational integration each: the solver steps with the cheap two-body
    // Jacobian, then converges with the exact one. A fit with the two-body
    // Jacobian alone stops at that Jacobian's fixed point, not at the minimum
    // of the backend's dynamics; the exact iterations move it there, so the
    // solution and its covariance are exact at a few STM evaluations.
    if config.jacobian == JacobianMethod::Propagator && problem.analytic_terms.is_some() {
        let mut approximate = solve(
            problem,
            initial_state,
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Analytic,
                ..*config
            },
        )?;
        let start = if approximate.state.iter().all(|v| v.is_finite()) {
            approximate.state
        } else {
            initial_state
        };
        let Some(exact_config) = remaining_fit_config(
            config,
            approximate.optimizer_iterations,
            approximate.nfev,
            JacobianMethod::Propagator,
        ) else {
            // The requested backend Jacobian never ran, so convergence of
            // the cheap seed phase is not convergence of the requested solve.
            approximate.status = STATUS_LIMIT;
            return Ok((approximate, JacobianMethod::Propagator));
        };
        let (exact, method) =
            solve_with_nonfinite_fallback(problem, start, &exact_config, warnings)?;
        return Ok((
            Solution {
                nfev: approximate.nfev + exact.nfev,
                optimizer_iterations: approximate.optimizer_iterations + exact.optimizer_iterations,
                ..exact
            },
            method,
        ));
    }
    solve_with_nonfinite_fallback(problem, initial_state, config, warnings)
}

/// [`solve`], continued with central differences when an analytic or
/// backend Jacobian turns non-finite at an iterate with finite residuals.
fn solve_with_nonfinite_fallback(
    problem: &WhitenedProblem<'_>,
    initial_state: [f64; 6],
    config: &WhitenedFitConfig,
    warnings: &mut Vec<String>,
) -> PropagationResultValue<(Solution, JacobianMethod)> {
    let solution = solve(problem, initial_state, config)?;
    if solution.status != STATUS_NON_FINITE
        || !matches!(
            config.jacobian,
            JacobianMethod::Analytic | JacobianMethod::Propagator
        )
        || !solution.residuals.iter().all(|r| r.is_finite())
    {
        return Ok((solution, config.jacobian));
    }
    warnings.push(format!(
        "The {} Jacobian was non-finite at the current iterate; the fit continued \
             with central differences through the propagator.",
        if config.jacobian == JacobianMethod::Analytic {
            "analytic two-body"
        } else {
            "propagator's"
        }
    ));
    let Some(central) = remaining_fit_config(
        config,
        solution.optimizer_iterations,
        solution.nfev,
        JacobianMethod::Central,
    ) else {
        return Ok((solution, config.jacobian));
    };
    let retry = solve(problem, solution.state, &central)?;
    Ok((
        Solution {
            nfev: solution.nfev + retry.nfev,
            optimizer_iterations: solution.optimizer_iterations + retry.optimizer_iterations,
            ..retry
        },
        JacobianMethod::Central,
    ))
}

fn solve(
    problem: &WhitenedProblem<'_>,
    initial_state: [f64; 6],
    config: &WhitenedFitConfig,
) -> PropagationResultValue<Solution> {
    let loss = config.loss;
    let f_scale = config.f_scale;
    let mut state = initial_state;
    // A non-finite initial residual (e.g. a NaN observation) is not an
    // error here: the gradient is then non-finite and the fit ends with
    // status -2 at the start state, whose residuals let the rejection loops
    // flag and drop the offending observation.
    let mut residuals = problem.residuals_one(&state)?;
    let mut cost = robust_cost(&residuals, loss, f_scale);
    // One honest budget: `nfev` counts every residual evaluation (the
    // legacy `iterations` column), including finite-difference columns.
    let mut nfev = 1_usize;
    let max_nfev = config.max_nfev.unwrap_or(usize::MAX);
    let mut optimizer_iterations = 0_usize;
    let jacobian_nfev = match config.jacobian {
        JacobianMethod::Analytic | JacobianMethod::Propagator | JacobianMethod::Auto => 0,
        JacobianMethod::Central => 12,
        JacobianMethod::TwoPoint => 6,
    };
    let mut status = STATUS_LIMIT;
    // scipy `x_scale="jac"`: column norms, non-decreasing across iterations.
    let mut scale = [0.0_f64; 6];
    let mut radius = 0.0_f64;
    let mut alpha = 0.0_f64;

    'iterations: for iteration in 0..config.max_iterations {
        if nfev >= max_nfev || jacobian_nfev > max_nfev.saturating_sub(nfev) {
            break;
        }
        optimizer_iterations += 1;
        nfev += jacobian_nfev;
        let jac = problem.jacobian(config.jacobian, &state, &residuals)?;
        let (jac_weighted, f_weighted) = irls_problem(&jac, &residuals, loss, f_scale);
        let (a, g) = normal_equations(&jac_weighted, &f_weighted);
        if !a.iter().flatten().all(|v| v.is_finite()) || !g.iter().all(|v| v.is_finite()) {
            // A non-finite derivative is never convergence, whatever the
            // (NaN-ignoring) gradient norm would have said.
            status = STATUS_NON_FINITE;
            break;
        }
        for k in 0..6 {
            scale[k] = scale[k].max(a[k][k].sqrt());
            if scale[k] == 0.0 {
                scale[k] = 1.0;
            }
        }
        // scipy's gtol: infinity norm of the gradient scaled by `1 / scale`,
        // so a weakly constrained parameter is judged in its own units.
        let g_scaled_norm = (0..6)
            .map(|k| (g[k] / scale[k]).abs())
            .fold(0.0_f64, f64::max);
        if g_scaled_norm <= config.gtol {
            status = STATUS_GTOL;
            break;
        }
        if iteration == 0 {
            radius = (0..6)
                .map(|k| (state[k] * scale[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            if radius == 0.0 {
                radius = 1.0;
            }
        }
        // The Gauss-Newton model in scaled coordinates, in the eigenbasis of
        // its normal matrix (`vectors[j][i]` is component `j` of vector `i`).
        let mut a_scaled = [[0.0_f64; 6]; 6];
        let mut g_scaled = [0.0_f64; 6];
        for j in 0..6 {
            g_scaled[j] = g[j] / scale[j];
            for k in 0..6 {
                a_scaled[j][k] = a[j][k] / (scale[j] * scale[k]);
            }
        }
        let (eigenvalues, vectors) = symmetric_eigen_6x6(&a_scaled);
        let eigenvalues: [f64; 6] = std::array::from_fn(|i| eigenvalues[i].max(0.0));
        let q: [f64; 6] =
            std::array::from_fn(|i| (0..6).map(|j| vectors[j][i] * g_scaled[j]).sum());
        let state_norm = norm6(&state);

        let mut accepted = false;
        for _attempt in 0..MAX_STEP_ATTEMPTS {
            if nfev >= max_nfev {
                // Budget exhaustion is neither a stall nor convergence.
                break 'iterations;
            }
            let (p_eigen, alpha_used) = trust_region_step(&eigenvalues, &q, radius, alpha);
            alpha = alpha_used;
            let p_scaled: [f64; 6] =
                std::array::from_fn(|j| (0..6).map(|i| vectors[j][i] * p_eigen[i]).sum());
            let step_scaled_norm = norm6(&p_scaled);
            let step: [f64; 6] = std::array::from_fn(|k| p_scaled[k] / scale[k]);
            let step_norm = norm6(&step);
            // Predicted reduction of the half-cost quadratic model.
            let predicted = -(0..6)
                .map(|i| q[i] * p_eigen[i] + 0.5 * eigenvalues[i] * p_eigen[i] * p_eigen[i])
                .sum::<f64>();
            let mut trial = state;
            for k in 0..6 {
                trial[k] += step[k];
            }
            nfev += 1;
            let trial_residuals = match problem.residuals(&[trial])? {
                Ok(mut batch) => batch.swap_remove(0),
                Err(_) => {
                    // A failed prediction (e.g. light time) rejects the step
                    // and shrinks the region like a non-finite trial.
                    radius = 0.25 * step_scaled_norm;
                    continue;
                }
            };
            let trial_cost = robust_cost(&trial_residuals, loss, f_scale);
            if !trial_cost.is_finite() || !trial_residuals.iter().all(|r| r.is_finite()) {
                radius = 0.25 * step_scaled_norm;
                continue;
            }
            let actual = 0.5 * (cost - trial_cost);
            let ratio = if predicted > 0.0 {
                actual / predicted
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
            let xtol_met = step_norm < config.xtol * (config.xtol + state_norm);
            if actual > 0.0 {
                let ftol_met = (cost - trial_cost) < config.ftol * cost && ratio > 0.25;
                state = trial;
                residuals = trial_residuals;
                cost = trial_cost;
                accepted = true;
                status = match (ftol_met, xtol_met) {
                    (true, true) => STATUS_FTOL_XTOL,
                    (true, false) => STATUS_FTOL,
                    (false, true) => STATUS_XTOL,
                    (false, false) => STATUS_LIMIT,
                };
                break;
            }
            if xtol_met {
                // The region shrank below the resolution without any
                // decrease: the iterate is a numerical minimum.
                status = STATUS_XTOL;
                break 'iterations;
            }
        }
        if !accepted {
            status = STATUS_STALLED;
            break;
        }
        if status != STATUS_LIMIT {
            break;
        }
    }

    Ok(Solution {
        state,
        residuals,
        cost,
        nfev,
        optimizer_iterations,
        status,
    })
}

// ---------------------------------------------------------------------------
// Covariance validation
// ---------------------------------------------------------------------------

/// Measured change of the fit cost at a 1-sigma displacement along the
/// covariance's weakest-constrained direction (`_weak_direction_delta_chi2`).
fn weak_direction_delta_cost(
    problem: &WhitenedProblem<'_>,
    state: &[f64; 6],
    covariance: &[f64; 36],
    cost_solution: f64,
    config: &WhitenedFitConfig,
) -> PropagationResultValue<Result<f64, String>> {
    let (eigenvalues, vectors) = symmetric_eigen_6x6(&covariance_matrix(covariance));
    let mut weakest = 0;
    for k in 1..6 {
        if eigenvalues[k] > eigenvalues[weakest] {
            weakest = k;
        }
    }
    if eigenvalues[weakest].is_nan() || eigenvalues[weakest] <= 0.0 {
        return Ok(Err("the covariance has no positive eigenvalue".to_string()));
    }
    let sigma = eigenvalues[weakest].sqrt();
    let mut plus = *state;
    let mut minus = *state;
    for i in 0..6 {
        plus[i] += sigma * vectors[i][weakest];
        minus[i] -= sigma * vectors[i][weakest];
    }
    let batch = match problem.residuals(&[plus, minus])? {
        Ok(batch) => batch,
        Err(message) => return Ok(Err(message)),
    };
    if batch.iter().flatten().any(|v| !v.is_finite()) {
        return Ok(Err(
            "residuals at the probe displacement are not finite".to_string()
        ));
    }
    let cost_plus = robust_cost(&batch[0], config.loss, config.f_scale);
    let cost_minus = robust_cost(&batch[1], config.loss, config.f_scale);
    Ok(Ok(0.5 * (cost_plus + cost_minus) - cost_solution))
}

/// `_validated_covariance`: the weak-direction consistency check with the
/// central-difference fallback for the analytic Jacobian; every diagnostic
/// is appended to `warnings` with the Python message text.
///
/// Only the in-band probe outcomes (no positive eigenvalue, a rejected
/// trial prediction, non-finite probe residuals) are warnings that leave
/// the covariance unvalidated. A backend error (`PropagationError`) raised
/// while predicting the probe displacement propagates like one raised by
/// any other prediction. Only an in-band row failure is a rejected trial.
const COVARIANCE_SYMMETRY_RELATIVE_TOLERANCE: f64 = 1.0e-12;
const COVARIANCE_MIN_RELATIVE_EIGENVALUE: f64 = 1.0e-12;
/// A numerically converged state without a finite symmetric positive-definite
/// covariance is not a successful scientific fit. This remains candidate-local
/// in corrected IOD; CMC applies its stricter conditioning/probe policy later.
const STATUS_INVALID_COVARIANCE: i64 = -3;

/// Finite/symmetry gate shared by general fits and CMC calibration.
fn symmetric_covariance(covariance: &[f64; 36]) -> Result<[[f64; 6]; 6], String> {
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err("the fitted state covariance is non-finite".to_string());
    }
    let scale = covariance
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err("the fitted state covariance is singular".to_string());
    }
    for row in 0..6 {
        for col in row + 1..6 {
            if (covariance[row * 6 + col] - covariance[col * 6 + row]).abs()
                > COVARIANCE_SYMMETRY_RELATIVE_TOLERANCE * scale
            {
                return Err("the fitted state covariance is not symmetric".to_string());
            }
        }
    }
    let mut symmetric = covariance_matrix(covariance);
    for row in 0..6 {
        for col in row + 1..6 {
            let value = 0.5 * (covariance[row * 6 + col] + covariance[col * 6 + row]);
            symmetric[row][col] = value;
            symmetric[col][row] = value;
        }
    }
    Ok(symmetric)
}

/// Scale-invariant structural validity for every successful fit. Position and
/// velocity carry unlike units, so definiteness is tested on the congruent
/// unit-diagonal matrix rather than on raw eigenvalues. This retains finite,
/// full-rank SPD covariances even when their raw condition number is extreme.
fn covariance_correlation_spectrum(covariance: &[f64; 36]) -> Result<(f64, f64), String> {
    let symmetric = symmetric_covariance(covariance)?;
    let mut scales = [0.0_f64; 6];
    for axis in 0..6 {
        let diagonal = symmetric[axis][axis];
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err("the fitted state covariance is not positive-definite".to_string());
        }
        scales[axis] = diagonal.sqrt();
    }
    let scaled: [[f64; 6]; 6] = std::array::from_fn(|row| {
        std::array::from_fn(|col| symmetric[row][col] / (scales[row] * scales[col]))
    });
    let (eigenvalues, _) = symmetric_eigen_6x6(&scaled);
    let lambda_max = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let lambda_min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    if !lambda_max.is_finite() || !lambda_min.is_finite() || lambda_min <= 0.0 {
        return Err("the fitted state covariance is not positive-definite".to_string());
    }
    Ok((lambda_min, lambda_max))
}

fn covariance_validity(covariance: &[f64; 36]) -> Result<(), String> {
    let (lambda_min, lambda_max) = covariance_correlation_spectrum(covariance)?;
    // A six-dimensional correlation matrix has O(1) spectrum. Eigenvalues
    // below n*epsilon relative to its largest mode are numerical null
    // directions, not evidence of a resolved full-rank covariance. This is a
    // scale-free rank test, deliberately far below CMC2003's 1e-12 quality
    // threshold.
    let rank_tolerance = 6.0 * f64::EPSILON * lambda_max;
    if lambda_min <= rank_tolerance {
        return Err(format!(
            "the fitted state covariance is numerically rank-deficient (correlation minimum/maximum eigenvalue ratio = {:.3e})",
            lambda_min / lambda_max
        ));
    }
    Ok(())
}

/// Raw spectrum used only for CMC2003's stricter conditioning threshold.
fn covariance_spectrum(covariance: &[f64; 36]) -> Result<(f64, f64), String> {
    let symmetric = symmetric_covariance(covariance)?;
    let (eigenvalues, _) = symmetric_eigen_6x6(&symmetric);
    let lambda_max = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let lambda_min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    if !(lambda_max.is_finite() && lambda_min.is_finite() && lambda_max > 0.0) {
        return Err("the fitted state covariance has no positive finite eigenvalue".to_string());
    }
    if lambda_min <= 0.0 {
        return Err("the fitted state covariance is not positive-definite".to_string());
    }
    Ok((lambda_min, lambda_max))
}

/// Structural/rank gate required before a covariance may calibrate CMC2003.
fn covariance_quality(covariance: &[f64; 36]) -> Result<(), String> {
    let (lambda_min, lambda_max) = covariance_spectrum(covariance)?;
    if lambda_min <= COVARIANCE_MIN_RELATIVE_EIGENVALUE * lambda_max {
        return Err(format!(
            "the fitted state covariance is rank-deficient or ill-conditioned (minimum/maximum eigenvalue ratio = {:.3e})",
            lambda_min / lambda_max
        ));
    }
    Ok(())
}

fn validated_covariance(
    problem: &WhitenedProblem<'_>,
    covariance: [f64; 36],
    jacobian_method: JacobianMethod,
    state: &[f64; 6],
    residuals_solution: &[f64],
    cost_solution: f64,
    config: &WhitenedFitConfig,
    warnings: &mut Vec<String>,
) -> PropagationResultValue<[f64; 36]> {
    let (lower, upper) = DELTA_CHI2_WINDOW;
    let delta = match weak_direction_delta_cost(problem, state, &covariance, cost_solution, config)?
    {
        Ok(delta) => delta,
        Err(message) => {
            warnings.push(format!(
                "The fit covariance could not be validated: evaluating residuals at a \
                 1-sigma displacement along its weakest direction failed ({message}). The \
                 covariance may be unreliable in weakly constrained directions."
            ));
            return Ok(covariance);
        }
    };
    if (lower..=upper).contains(&delta) {
        return Ok(covariance);
    }
    match jacobian_method {
        JacobianMethod::Analytic => {
            warnings.push(format!(
                "The analytic (2-body) fit covariance failed the weak-direction consistency \
                 check (measured delta-chi2 = {delta:.3} at a 1-sigma displacement along the \
                 weakest axis; expected ~1). This typically indicates dynamics inside the arc \
                 that the 2-body state transition matrix does not model (e.g. a planetary \
                 encounter). Falling back to a central-difference Jacobian computed through \
                 the full residual pipeline."
            ));
            let jac = problem.jacobian(JacobianMethod::Central, state, residuals_solution)?;
            match covariance_from_jacobian(&jac, residuals_solution, config.loss, config.f_scale) {
                Some(fallback) => validated_covariance(
                    problem,
                    fallback,
                    JacobianMethod::Central,
                    state,
                    residuals_solution,
                    cost_solution,
                    config,
                    warnings,
                ),
                None => {
                    warnings.push(
                        "The central-difference fallback covariance could not be computed. \
                         The solution covariance may be unreliable."
                            .to_string(),
                    );
                    Ok(covariance)
                }
            }
        }
        JacobianMethod::TwoPoint => {
            warnings.push(format!(
                "The fit covariance failed the weak-direction consistency check (measured \
                 delta-chi2 = {delta:.3} at a 1-sigma displacement along the weakest axis; \
                 expected ~1). The covariance was computed from the solver's \
                 forward-difference Jacobian, whose implied steps can sit below the numerical \
                 noise floor of the residual pipeline; in weakly constrained (line-of-sight) \
                 directions the resulting confidence is fabricated by finite-difference noise. \
                 Use jacobian='analytic' or jacobian='central' instead."
            ));
            Ok(covariance)
        }
        JacobianMethod::Central | JacobianMethod::Propagator | JacobianMethod::Auto => {
            warnings.push(format!(
                "The fit covariance failed the weak-direction consistency check (measured \
                 delta-chi2 = {delta:.3} at a 1-sigma displacement along the weakest axis; \
                 expected ~1). The covariance may be unreliable in weakly constrained \
                 directions (the chi2 surface disagrees with the local quadratic model, e.g. \
                 because the solution has not fully converged along a flat valley)."
            ));
            Ok(covariance)
        }
    }
}

// ---------------------------------------------------------------------------
// Public work unit
// ---------------------------------------------------------------------------

/// One-crossing `fit_least_squares`: whitened, optionally robust differential
/// correction of `orbit` (one row, Cartesian, with epoch) on the non-ignored
/// subset of `observed` / `observers`, validated covariance, and the final
/// evaluation over the full observation set.
#[allow(clippy::too_many_arguments)]
pub fn fit_orbit_whitened_barycentric<P, T>(
    propagator: &P,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    ignore: &[bool],
    config: &WhitenedFitConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<WhitenedFitOutput>
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
    fit_orbit_whitened_with(
        &predictor,
        orbit,
        observed,
        observers,
        ignore,
        config,
        provider,
        translation_provider,
    )
}

/// [`fit_orbit_whitened_barycentric`] over any [`SphericalPredictor`].
#[allow(clippy::too_many_arguments)]
pub fn fit_orbit_whitened_with<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    ignore: &[bool],
    config: &WhitenedFitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<WhitenedFitOutput> {
    fit_orbit_whitened_impl(
        predictor,
        orbit,
        observed,
        observers,
        ignore,
        config,
        provider,
        translation_provider,
        None,
    )
    .map(|(fit, _)| fit)
}

/// With a full observation problem, finalize covariance and rejection from
/// one central-difference Jacobian at the fitted state. Only selected rows
/// enter the normal matrix, but excluded rows remain available for recovery.
/// Without it, preserve the standalone fitter's Jacobian/validation policy.
#[allow(clippy::too_many_arguments)]
fn fit_orbit_whitened_impl<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    ignore: &[bool],
    config: &WhitenedFitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
    covariance_problem: Option<&WhitenedProblem<'_>>,
) -> PropagationResultValue<(WhitenedFitOutput, Option<Vec<f64>>)> {
    config
        .validate()
        .map_err(PropagationError::InvalidRequest)?;
    let n = observed.len();
    if observers.len() != n || ignore.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed, observers, and ignore must have equal length".to_string(),
        ));
    }
    let keep: Vec<bool> = ignore.iter().map(|&flag| !flag).collect();
    let observed_fit = filter_coordinate_batch(observed, &keep)?;
    let observers_fit = filter_observer_batch(observers, &keep)?;
    let problem = WhitenedProblem::new(
        predictor,
        orbit,
        &observed_fit,
        &observers_fit,
        config,
        provider,
        translation_provider,
    )?;
    // The fit runs with the method resolved for this predictor.
    let resolved = WhitenedFitConfig {
        jacobian: problem.jacobian_method,
        ..*config
    };
    let config = &resolved;

    let mut warnings = Vec::new();
    let (solution, jacobian_method) =
        solve_with_fallback(&problem, problem.geometry.state, config, &mut warnings)?;

    let rejection_jacobian = covariance_problem
        .map(|full| full.jacobian(JacobianMethod::Central, &solution.state, &[]))
        .transpose()?;
    let (jac, covariance_method) = match &rejection_jacobian {
        Some(full_jac) => {
            let selected_jac = full_jac
                .chunks_exact(12)
                .zip(&keep)
                .filter(|(_, selected)| **selected)
                .flat_map(|(rows, _)| rows.iter().copied())
                .collect::<Vec<_>>();
            (selected_jac, JacobianMethod::Central)
        }
        None => (
            problem.jacobian(jacobian_method, &solution.state, &solution.residuals)?,
            jacobian_method,
        ),
    };
    let mut covariance =
        match covariance_from_jacobian(&jac, &solution.residuals, config.loss, config.f_scale) {
            Some(covariance) => covariance,
            None => {
                warnings.push(
                    "The covariance matrix could not be computed. The solution may be unreliable."
                        .to_string(),
                );
                [f64::NAN; 36]
            }
        };
    if config.validate_covariance && covariance.iter().all(|v| v.is_finite()) {
        covariance = validated_covariance(
            &problem,
            covariance,
            covariance_method,
            &solution.state,
            &solution.residuals,
            solution.cost,
            config,
            &mut warnings,
        )?;
    }
    // Outside CMC, a finite symmetric positive-definite covariance is enough:
    // mixed position/velocity units make a universal condition-number cutoff
    // scientifically inappropriate for otherwise valid short arcs. Structural
    // failures make this fit unsuccessful (and therefore candidate-local), not
    // an InvalidRequest that aborts a wider IOD search. CMC uses
    // `covariance_quality` plus the weak-direction probe below.
    let covariance_failure = covariance_problem
        .is_none()
        .then(|| covariance_validity(&covariance).err())
        .flatten();
    if let Some(reason) = &covariance_failure {
        warnings.push(format!(
            "The fit did not produce a usable state covariance: {reason}. The state and statistics are retained, but the fit is marked unsuccessful."
        ));
    }
    let status_code = if covariance_failure.is_some() && solution.status > 0 {
        STATUS_INVALID_COVARIANCE
    } else {
        solution.status
    };

    let evaluation = evaluate_orbit(
        predictor,
        solution.state,
        orbit,
        observed,
        observers,
        ignore,
        6,
    )?;

    let component_weights = robust_weights(&solution.residuals, config.loss, config.f_scale);
    let mut weights = vec![0.0_f64; n];
    let mut included = 0_usize;
    for (row, &flag) in keep.iter().enumerate() {
        if flag {
            weights[row] = component_weights[2 * included].min(component_weights[2 * included + 1]);
            included += 1;
        }
    }
    let fit_chi2 = solution.residuals.iter().map(|r| r * r).sum();

    Ok((
        WhitenedFitOutput {
            state: solution.state,
            covariance,
            iterations: solution.nfev,
            optimizer_iterations: solution.optimizer_iterations,
            converged: status_code > 0,
            status_code,
            cost: solution.cost,
            fit_chi2,
            weights,
            residuals_whitened: solution.residuals,
            evaluation,
            warnings,
        },
        rejection_jacobian,
    ))
}

/// Product of [`validate_fit_covariance_with`].
#[derive(Debug, Clone)]
pub struct CovarianceValidation {
    /// The covariance after validation (the central-difference fallback for
    /// the analytic Jacobian when the probe fails; otherwise the input).
    pub covariance: [f64; 36],
    /// Measured delta-cost at a 1-sigma displacement along the weakest
    /// direction of the INPUT covariance (NaN when the probe failed).
    pub delta_cost: f64,
    /// Diagnostics in the Python `RuntimeWarning` wording, in order.
    pub warnings: Vec<String>,
}

/// The weak-direction consistency check of `fit_least_squares` on an
/// externally supplied covariance of `orbit` (its state is the solution):
/// the probe measurement, and the covariance `fit_least_squares` would have
/// reported for `jacobian_method` after validation.
#[allow(clippy::too_many_arguments)]
pub fn validate_fit_covariance_with<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    ignore: &[bool],
    covariance: [f64; 36],
    jacobian_method: JacobianMethod,
    config: &WhitenedFitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<CovarianceValidation> {
    config
        .validate()
        .map_err(PropagationError::InvalidRequest)?;
    let n = observed.len();
    if observers.len() != n || ignore.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed, observers, and ignore must have equal length".to_string(),
        ));
    }
    let keep: Vec<bool> = ignore.iter().map(|&flag| !flag).collect();
    let observed_fit = filter_coordinate_batch(observed, &keep)?;
    let observers_fit = filter_observer_batch(observers, &keep)?;
    let probe_config = WhitenedFitConfig {
        jacobian: jacobian_method,
        ..*config
    };
    let problem = WhitenedProblem::new(
        predictor,
        orbit,
        &observed_fit,
        &observers_fit,
        &probe_config,
        provider,
        translation_provider,
    )?;
    let state = problem.geometry.state;
    let residuals = problem.residuals_one(&state)?;
    let cost = robust_cost(&residuals, config.loss, config.f_scale);
    // In-band probe failures are reported as NaN (and as a warning by the
    // validation below); a backend error propagates.
    let delta_cost = weak_direction_delta_cost(&problem, &state, &covariance, cost, &probe_config)?
        .unwrap_or(f64::NAN);
    let mut warnings = Vec::new();
    let validated = validated_covariance(
        &problem,
        covariance,
        jacobian_method,
        &state,
        &residuals,
        cost,
        &probe_config,
        &mut warnings,
    )?;
    Ok(CovarianceValidation {
        covariance: validated,
        delta_cost,
        warnings,
    })
}

// ---------------------------------------------------------------------------
// Outlier-rejection loops
// ---------------------------------------------------------------------------

/// A copy of `orbit` (one row) with its Cartesian state replaced and no
/// covariance: the warm-start seed of the next refit pass. Physical and
/// non-gravitational parameter tables are preserved unchanged.
fn orbit_with_state(orbit: &OrbitBatch, state: [f64; 6]) -> PropagationResultValue<OrbitBatch> {
    let coordinates = CoordinateBatch::new(
        crate::CoordinateValues::Cartesian(vec![state]),
        orbit.coordinates.frame,
        orbit.coordinates.origins.clone(),
        orbit.coordinates.times.clone(),
        None,
    )?;
    let mut updated =
        OrbitBatch::new(orbit.orbit_id.clone(), orbit.object_id.clone(), coordinates)?;
    updated.physical_parameters = orbit.physical_parameters.clone();
    updated.non_gravitational_parameters = orbit.non_gravitational_parameters.clone();
    Ok(updated)
}

/// `calculate_max_outliers`: `floor(num_obs * contamination / 100)` capped at
/// `num_obs - min_obs` (never negative).
pub fn max_outliers(num_obs: usize, min_obs: usize, contamination_percentage: f64) -> usize {
    let allowed = num_obs as f64 * (contamination_percentage / 100.0);
    let cap = num_obs as f64 - min_obs as f64;
    allowed.min(cap).max(0.0) as usize
}

/// Worst-observation policy of `remove_lowest_probability_observation` over
/// the rows in `candidates`: the lowest residual probability, ties broken by
/// the largest NaN-ignoring squared-residual norm. A NaN probability makes
/// the minimum undefined (the legacy error).
fn lowest_probability_row(
    candidates: &[usize],
    probability: &[f64],
    residuals: &[f64],
) -> Result<usize, String> {
    if candidates.is_empty() || candidates.iter().any(|&row| probability[row].is_nan()) {
        return Err("Could not identify a lowest-probability observation.".to_string());
    }
    let minimum = candidates
        .iter()
        .map(|&row| probability[row])
        .fold(f64::INFINITY, f64::min);
    let mut best = None;
    let mut best_norm = f64::NEG_INFINITY;
    for &row in candidates {
        if probability[row] != minimum {
            continue;
        }
        let norm: f64 = residuals[row * 6..(row + 1) * 6]
            .iter()
            .filter(|value| !value.is_nan())
            .map(|value| value * value)
            .sum();
        if best.is_none() || norm > best_norm {
            best = Some(row);
            best_norm = norm;
        }
    }
    best.ok_or_else(|| "Could not identify a lowest-probability observation.".to_string())
}

/// Settings of the worst-residual rejection loop (`iterative_fit`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterativeFitConfig {
    pub rchi2_threshold: f64,
    pub min_obs: usize,
    pub min_arc_length: f64,
    pub contamination_percentage: f64,
    pub fit: WhitenedFitConfig,
}

impl Default for IterativeFitConfig {
    fn default() -> Self {
        Self {
            rchi2_threshold: 10.0,
            min_obs: 6,
            min_arc_length: 1.0,
            contamination_percentage: 20.0,
            fit: WhitenedFitConfig::default(),
        }
    }
}

impl IterativeFitConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.fit.validate()?;
        if self.min_obs < 3 {
            return Err("min_obs must be at least 3".to_string());
        }
        if !self.rchi2_threshold.is_finite() || self.rchi2_threshold <= 0.0 {
            return Err("rchi2_threshold must be positive and finite".to_string());
        }
        if !self.min_arc_length.is_finite() || self.min_arc_length < 0.0 {
            return Err("min_arc_length must be finite and non-negative".to_string());
        }
        if !self.contamination_percentage.is_finite()
            || !(0.0..=100.0).contains(&self.contamination_percentage)
        {
            return Err(
                "contamination_percentage must be finite and between 0 and 100".to_string(),
            );
        }
        Ok(())
    }
}

/// Product of the worst-residual loop: the selected pass and the number of
/// passes run. `fit.evaluation.outlier` is the final ignore mask.
#[derive(Debug, Clone)]
pub struct IterativeFitOutput {
    pub fit: WhitenedFitOutput,
    pub passes: usize,
}

/// One-crossing `iterative_fit`: refit while the reduced chi2 exceeds
/// `rchi2_threshold`, removing the lowest-probability solution observation
/// each pass, bounded by `contamination_percentage`, `min_obs` and
/// `min_arc_length`. Returns the converged pass with the lowest reduced chi2
/// or, if no pass converged, the lowest-reduced-chi2 pass overall; the
/// diagnostics of every pass are accumulated on the returned fit.
#[allow(clippy::too_many_arguments)]
pub fn iterative_fit_barycentric<P, T>(
    propagator: &P,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &IterativeFitConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<IterativeFitOutput>
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
    iterative_fit_with(
        &predictor,
        orbit,
        observed,
        observers,
        config,
        provider,
        translation_provider,
    )
}

/// [`iterative_fit_barycentric`] over any [`SphericalPredictor`].
#[allow(clippy::too_many_arguments)]
pub fn iterative_fit_with<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &IterativeFitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<IterativeFitOutput> {
    config
        .validate()
        .map_err(PropagationError::InvalidRequest)?;
    let n = observed.len();
    if observers.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed and observers must have equal length".to_string(),
        ));
    }
    let times_mjd = predictor.dynamical_times_mjd(observed)?;
    let limit = max_outliers(n, config.min_obs, config.contamination_percentage);
    let mut ignore = vec![false; n];
    let mut warnings: Vec<String> = Vec::new();
    let mut best: Option<(WhitenedFitOutput, f64)> = None;
    let mut fallback: Option<(WhitenedFitOutput, f64)> = None;
    let mut passes = 0_usize;

    for _ in 0..=limit {
        passes += 1;
        let mut output = fit_orbit_whitened_with(
            predictor,
            orbit,
            observed,
            observers,
            &ignore,
            &config.fit,
            provider,
            translation_provider,
        )?;
        warnings.append(&mut output.warnings);
        let rchi2 = output.evaluation.reduced_chi2;
        let key = if rchi2.is_nan() { f64::INFINITY } else { rchi2 };
        let converged = output.converged;
        if fallback.as_ref().is_none_or(|(_, current)| key < *current) {
            fallback = Some((output.clone(), key));
        }
        if converged && best.as_ref().is_none_or(|(_, current)| key < *current) {
            best = Some((output.clone(), key));
        }
        if !rchi2.is_nan() && rchi2 <= config.rchi2_threshold {
            break;
        }
        let n_ignored = ignore.iter().filter(|&&flag| flag).count();
        if n_ignored >= limit {
            break;
        }
        let solution: Vec<usize> = (0..n).filter(|&row| !ignore[row]).collect();
        if solution.is_empty() {
            break;
        }
        let worst = lowest_probability_row(
            &solution,
            &output.evaluation.probability,
            &output.evaluation.residuals,
        )
        .map_err(PropagationError::InvalidRequest)?;
        let remaining: Vec<bool> = (0..n).map(|row| !ignore[row] && row != worst).collect();
        let remaining_count = remaining.iter().filter(|&&flag| flag).count();
        let (arc_length, _, _) = super::od::masked_arc_length(&times_mjd, &remaining);
        if remaining_count < config.min_obs || arc_length < config.min_arc_length {
            break;
        }
        ignore[worst] = true;
    }

    let (mut fit, _) = match best.or(fallback) {
        Some(selected) => selected,
        None => {
            return Err(PropagationError::InvalidRequest(
                "iterative_fit ran no pass".to_string(),
            ))
        }
    };
    fit.warnings = warnings;
    Ok(IterativeFitOutput { fit, passes })
}

/// Settings of the Carpino-Milani-Chesley (2003) rejection loop
/// (`cmc2003_fit_detailed`); constants default to OrbFit's `reject.def`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cmc2003FitConfig {
    pub chi2_reject: f64,
    pub chi2_recover: f64,
    pub chi2_frac: f64,
    pub max_iterations: usize,
    pub max_rejected_fraction: f64,
    pub apparition_gap_days: f64,
    pub psd_floor_frac: f64,
    /// Fewest observations rejection may leave selected. The default is
    /// CMC2003's own floor (`CMC2003_MIN_OBS + 1`); corrected IOD raises it
    /// to its `min_obs`. Never below that default.
    pub min_kept: usize,
    pub fit: WhitenedFitConfig,
}

impl Default for Cmc2003FitConfig {
    fn default() -> Self {
        Self {
            chi2_reject: crate::cmc2003::CMC2003_CHI2_REJECT,
            chi2_recover: crate::cmc2003::CMC2003_CHI2_RECOVER,
            chi2_frac: crate::cmc2003::CMC2003_CHI2_FRAC,
            max_iterations: crate::cmc2003::CMC2003_MAX_ITERATIONS,
            max_rejected_fraction: crate::cmc2003::CMC2003_MAX_REJECTED_FRACTION,
            apparition_gap_days: crate::cmc2003::CMC2003_APPARITION_GAP_DAYS,
            psd_floor_frac: crate::cmc2003::CMC2003_PSD_FLOOR_FRAC,
            min_kept: crate::cmc2003::CMC2003_MIN_OBS + 1,
            fit: WhitenedFitConfig::default(),
        }
    }
}

impl Cmc2003FitConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.fit.validate()?;
        for (name, value) in [
            ("chi2_reject", self.chi2_reject),
            ("chi2_recover", self.chi2_recover),
            ("chi2_frac", self.chi2_frac),
            ("max_rejected_fraction", self.max_rejected_fraction),
            ("apparition_gap_days", self.apparition_gap_days),
            ("psd_floor_frac", self.psd_floor_frac),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("cmc2003.{name} must be finite and non-negative"));
            }
        }
        if self.chi2_reject <= self.chi2_recover {
            return Err("cmc2003.chi2_reject must exceed chi2_recover".to_string());
        }
        if self.chi2_frac > 1.0 || self.max_rejected_fraction > 1.0 {
            return Err("cmc2003 fractions must not exceed 1".to_string());
        }
        if self.max_iterations == 0 {
            return Err("cmc2003.max_iterations must be positive".to_string());
        }
        if self.min_kept < 3 {
            return Err("cmc2003.min_kept must be at least 3".to_string());
        }
        Ok(())
    }
}

/// Product of the CMC2003 loop: the final fit (members flag rejected
/// observations as outliers) and the run diagnostics of `CMC2003Fit`.
#[derive(Debug, Clone)]
pub struct Cmc2003FitOutput {
    pub fit: WhitenedFitOutput,
    pub n_iterations: usize,
    pub n_rejected: usize,
    pub n_recovered: usize,
    /// Sorted legacy flag names.
    pub flags: Vec<String>,
}

/// One-crossing `cmc2003_fit_detailed`: repeated whitened fits with the
/// CMC2003 reject / re-include decision against the expected post-fit
/// residual covariance. Covariance and scoring share central differences
/// through the full predictor at the fitted state, independent of the solver
/// Jacobian. Each pass warm-starts from the previous solution.
#[allow(clippy::too_many_arguments)]
pub fn cmc2003_fit_barycentric<P, T>(
    propagator: &P,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &Cmc2003FitConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<Cmc2003FitOutput>
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
    cmc2003_fit_with(
        &predictor,
        orbit,
        observed,
        observers,
        config,
        provider,
        translation_provider,
    )
}

fn cmc_covariance_error(
    reason: impl AsRef<str>,
    output: &WhitenedFitOutput,
    selected: &[bool],
    pass: usize,
) -> PropagationError {
    PropagationError::CmcCovarianceUnavailable(format!(
        "{}; retained_fit={{pass:{pass}, selected:{}, status:{}, converged:{}, cost:{:.17e}, state:{:?}, covariance:{:?}}}",
        reason.as_ref(),
        selected.iter().filter(|&&keep| keep).count(),
        output.status_code,
        output.converged,
        output.cost,
        output.state,
        output.covariance,
    ))
}

/// [`cmc2003_fit_barycentric`] over any [`SphericalPredictor`].
#[allow(clippy::too_many_arguments)]
pub fn cmc2003_fit_with<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &Cmc2003FitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<Cmc2003FitOutput> {
    use crate::cmc2003::{
        cmc2003_apparitions, cmc2003_expected_residual_chi2, cmc2003_select, Cmc2003Flag,
        Cmc2003SelectionOptions, CMC2003_MIN_OBS,
    };
    use std::collections::BTreeSet;

    config
        .validate()
        .map_err(PropagationError::InvalidRequest)?;
    let n = observed.len();
    if n == 0 {
        return Err(PropagationError::InvalidRequest(
            "cmc2003_fit requires at least one observation".to_string(),
        ));
    }
    if observers.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed and observers must have equal length".to_string(),
        ));
    }
    // Prepare all observations once, including rows later excluded from the
    // fit. Numerical finalization does not need any two-body model terms.
    let full_problem = WhitenedProblem::new(
        predictor,
        orbit,
        observed,
        observers,
        &WhitenedFitConfig {
            jacobian: JacobianMethod::Central,
            ..config.fit
        },
        provider,
        translation_provider,
    )?;
    let mjd_utc = observed
        .times
        .as_ref()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "orbit determination requires observation times".to_string(),
            )
        })?
        .rescale_with_provider(TimeScale::Utc, provider)?
        .mjd_values();
    let apparitions = cmc2003_apparitions(&mjd_utc, config.apparition_gap_days);

    // CMC validates the selected-fit covariance explicitly below. Keep the
    // standalone warning/fallback probe off here so one authoritative probe
    // determines whether rejection statistics are calibrated.
    let cmc_fit_config = WhitenedFitConfig {
        validate_covariance: false,
        ..config.fit
    };
    let run_fit = |seed: &OrbitBatch, selected: &[bool]| {
        let ignore: Vec<bool> = selected.iter().map(|&keep| !keep).collect();
        fit_orbit_whitened_impl(
            predictor,
            seed,
            observed,
            observers,
            &ignore,
            &cmc_fit_config,
            provider,
            translation_provider,
            Some(&full_problem),
        )
    };
    let validate_cmc_fit = |seed: &OrbitBatch,
                            selected: &[bool],
                            output: &WhitenedFitOutput,
                            pass: usize|
     -> PropagationResultValue<()> {
        covariance_quality(&output.covariance)
            .map_err(|reason| cmc_covariance_error(reason, output, selected, pass))?;
        let observed_selected = filter_coordinate_batch(observed, selected)?;
        let observers_selected = filter_observer_batch(observers, selected)?;
        let covariance_problem = WhitenedProblem::new(
            predictor,
            seed,
            &observed_selected,
            &observers_selected,
            &cmc_fit_config,
            provider,
            translation_provider,
        )?;
        let probe = weak_direction_delta_cost(
            &covariance_problem,
            &output.state,
            &output.covariance,
            output.cost,
            &cmc_fit_config,
        )?;
        let (lower, upper) = DELTA_CHI2_WINDOW;
        match probe {
            Ok(delta) if (lower..=upper).contains(&delta) => Ok(()),
            Ok(delta) => Err(cmc_covariance_error(
                format!(
                    "weak-direction covariance consistency check failed (delta-chi2 = {delta:.3})"
                ),
                output,
                selected,
                pass,
            )),
            Err(reason) => Err(cmc_covariance_error(
                format!("weak-direction covariance probe failed: {reason}"),
                output,
                selected,
                pass,
            )),
        }
    };

    let mut selected = vec![true; n];
    let mut flags: BTreeSet<Cmc2003Flag> = BTreeSet::new();
    let mut n_recovered_total = 0_usize;
    let mut stuck_passes = 0_usize;
    let mut previous_modifications: Option<usize> = None;
    let mut seed = orbit.clone();
    let mut n_iterations = 0_usize;
    let mut converged = false;
    let mut warnings: Vec<String> = Vec::new();
    let mut current: Option<WhitenedFitOutput> = None;

    for iteration in 0..config.max_iterations {
        n_iterations = iteration + 1;
        let (mut output, rejection_jacobian) = run_fit(&seed, &selected)?;
        warnings.append(&mut output.warnings);

        validate_cmc_fit(&seed, &selected, &output, n_iterations)?;

        seed = orbit_with_state(orbit, output.state)?;

        if n <= CMC2003_MIN_OBS {
            flags.insert(Cmc2003Flag::TooFewObservations);
            converged = true;
            current = Some(output);
            break;
        }

        let residual_pairs: Vec<f64> = (0..n)
            .flat_map(|row| {
                [
                    output.evaluation.residuals[row * 6 + 1],
                    output.evaluation.residuals[row * 6 + 2],
                ]
            })
            .collect();
        let residuals = whiten_residual_pairs(&full_problem.whiteners, &residual_pairs);
        let jacobian = rejection_jacobian.expect("full covariance problem supplies the Jacobian");
        let covariance = if output.covariance.iter().all(|v| v.is_finite()) {
            Some(&output.covariance[..])
        } else {
            None
        };
        let (chi2, pass_flags) = cmc2003_expected_residual_chi2(
            &residuals,
            &jacobian,
            covariance,
            &selected,
            config.psd_floor_frac,
        )
        .map_err(PropagationError::InvalidRequest)?;
        flags.extend(pass_flags);

        let n_selected = selected.iter().filter(|&&keep| keep).count();
        let one_at_a_time = n_selected <= 6 * CMC2003_MIN_OBS || stuck_passes >= 4;
        let selection = cmc2003_select(
            &chi2,
            &selected,
            &apparitions,
            &Cmc2003SelectionOptions {
                chi2_reject: config.chi2_reject,
                chi2_recover: config.chi2_recover,
                chi2_frac: config.chi2_frac,
                max_rejected_fraction: config.max_rejected_fraction,
                one_at_a_time,
                // The selection keeps at least `min_obs + 1`.
                min_obs: config.min_kept.saturating_sub(1).max(CMC2003_MIN_OBS),
            },
        )
        .map_err(PropagationError::InvalidRequest)?;
        flags.extend(selection.flags.iter().copied());
        let n_modifications = selection.n_rejected + selection.n_recovered;
        n_recovered_total += selection.n_recovered;
        stuck_passes = if previous_modifications == Some(n_modifications) && n_modifications > 0 {
            stuck_passes + 1
        } else {
            0
        };
        previous_modifications = Some(n_modifications);
        current = Some(output);
        if n_modifications == 0 {
            converged = true;
            break;
        }
        selected = selection.selected;
    }

    let mut fit = if converged {
        current.ok_or_else(|| {
            PropagationError::InvalidRequest("cmc2003_fit ran no pass".to_string())
        })?
    } else {
        // The selection changed after the last fit: refit so that the
        // returned orbit and members describe the final selection.
        flags.insert(Cmc2003Flag::MaxIterations);
        let (mut output, _) = run_fit(&seed, &selected)?;
        warnings.append(&mut output.warnings);
        validate_cmc_fit(&seed, &selected, &output, n_iterations + 1)?;
        output
    };
    fit.warnings = warnings;
    Ok(Cmc2003FitOutput {
        fit,
        n_iterations,
        n_rejected: selected.iter().filter(|&&keep| !keep).count(),
        n_recovered: n_recovered_total,
        flags: flags.iter().map(|flag| flag.as_str().to_string()).collect(),
    })
}

// ---------------------------------------------------------------------------
// Full orbit determination and run_od
// ---------------------------------------------------------------------------

/// Which outlier treatment refines the IOD orbit (`NativeOrbitFitter`'s
/// ``outlier_rejection``).
#[derive(Debug, Clone, Copy)]
pub enum RefinementConfig {
    WorstResidual(IterativeFitConfig),
    Cmc2003(Cmc2003FitConfig),
}

impl RefinementConfig {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::WorstResidual(config) => config.validate(),
            Self::Cmc2003(config) => config.validate(),
        }
    }
}

impl Default for RefinementConfig {
    /// `NativeOrbitFitter`'s default: CMC2003 rejection with OrbFit's
    /// constants.
    fn default() -> Self {
        Self::Cmc2003(Cmc2003FitConfig::default())
    }
}

/// Settings of the complete native orbit determination: Gauss IOD followed
/// by differential correction with outlier treatment.
#[derive(Debug, Clone, Copy, Default)]
pub struct FullOdConfig {
    pub iod: super::od::IodConfig,
    pub refinement: RefinementConfig,
}

impl FullOdConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.iod.validate()?;
        self.refinement.validate()?;
        Ok(())
    }
}

/// The refined orbit of a full OD run with the loop diagnostics
/// (`passes` for the worst-residual loop; `n_iterations`, `n_rejected`,
/// `n_recovered` and `flags` for CMC2003, zero / empty otherwise).
#[derive(Debug, Clone)]
pub struct RefinedOutput {
    pub fit: WhitenedFitOutput,
    pub passes: usize,
    pub n_rejected: usize,
    pub n_recovered: usize,
    pub flags: Vec<String>,
}

/// Product of `full_od_barycentric`: the IOD decision (`iod.found` false
/// when no preliminary orbit was accepted, in which case `refined` is None)
/// and the refined orbit. Every member array keeps the input observation
/// order.
#[derive(Debug, Clone)]
pub struct FullOdOutput {
    /// The IOD stage: every scored candidate and the accepted one, whose
    /// exact epoch is the epoch of `refined` too.
    pub iod: super::od::IodResult,
    pub refined: Option<RefinedOutput>,
}

/// The IOD stage under its configured acceptance: [`super::od::run_iod`]
/// for `Prefit`, the staged correction of [`run_iod_corrected`] for
/// `Corrected` (whose stage fits use `fit` without covariance validation,
/// with `correction.stage_loss` before the whole arc and the linear loss
/// on it).
#[allow(clippy::too_many_arguments)]
pub fn run_iod_with<T: OriginTranslationProvider>(
    iod_predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &super::od::IodConfig,
    fit: &WhitenedFitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<super::od::IodResult> {
    config
        .validate()
        .map_err(PropagationError::InvalidRequest)?;
    fit.validate().map_err(PropagationError::InvalidRequest)?;
    match config.acceptance {
        super::od::IodAcceptance::Prefit => {
            super::od::run_iod(iod_predictor, observed, observers, config)
        }
        super::od::IodAcceptance::Corrected => run_iod_corrected(
            iod_predictor,
            observed,
            observers,
            config,
            fit,
            provider,
            translation_provider,
        ),
    }
}

/// Outcome of one stage fit: the fit when it is usable as the next seed.
fn stage_fit<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    seed: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    mask: &[bool],
    config: &WhitenedFitConfig,
    rchi2_threshold: f64,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<Result<WhitenedFitOutput, (String, bool)>> {
    let ignore: Vec<bool> = mask.iter().map(|&keep| !keep).collect();
    // A non-finite analytic Jacobian is handled inside the fit (central
    // differences from that iterate).
    let output = match fit_orbit_whitened_with(
        predictor,
        seed,
        observed,
        observers,
        &ignore,
        config,
        provider,
        translation_provider,
    ) {
        Ok(output) => output,
        // Only a failed row returned in-band by an otherwise successful
        // propagation request is local to this candidate.
        Err(err @ PropagationError::RowFailure(_)) => {
            return Ok(Err((format!("prediction failed: {err}"), true)))
        }
        Err(err) => return Err(err),
    };
    // The stage is judged on the objective it minimized: the plain reduced
    // chi2 for the linear loss, scaled by robust cost / chi2 for Huber (so
    // observations the partial orbit down-weights do not fail the stage).
    let rchi2 = if config.loss == LossType::Linear || output.fit_chi2 <= 0.0 {
        output.evaluation.reduced_chi2
    } else {
        output.evaluation.reduced_chi2 * output.cost / output.fit_chi2
    };
    // Any finite fit within the threshold is a usable seed for the next
    // stage, converged or not (the walk-forward accepted budget-exhausted
    // fits at a good state); a non-finite derivative (status -2) is not.
    // Corrected IOD is itself an intermediate state-estimation stage. Even its
    // whole-observation-arc correction hands only a state/objective to the
    // final refinement. Short-arc geometry can make this stage covariance
    // numerically rank deficient although final refinement is full rank.
    // `orbit_with_state` drops covariance before every wider/refinement seed;
    // reportable covariance is filtered separately below.
    let usable = output.status_code != STATUS_NON_FINITE;
    if !usable || !rchi2.is_finite() || rchi2 > rchi2_threshold {
        return Ok(Err((
            format!(
                "stage fit on {} observations: reduced chi2 {:.4}, status {}{}",
                mask.iter().filter(|&&keep| keep).count(),
                rchi2,
                output.status_code,
                output
                    .warnings
                    .last()
                    .map(|warning| format!(" ({warning})"))
                    .unwrap_or_default()
            ),
            false,
        )));
    }
    Ok(Ok(output))
}

/// The next stage's observation mask: every observation within the
/// current span grown `growth` times about its centre; when that adds
/// nothing (an apparition gap) the observing night (`nights`, see
/// `observation_nights`) of the observation nearest the current span,
/// earliest on ties, joins instead, with everything observed during it.
pub(super) fn grow_stage(times: &[f64], nights: &[usize], mask: &[bool], growth: f64) -> Vec<bool> {
    let included = times
        .iter()
        .zip(mask)
        .filter(|(_, &keep)| keep)
        .map(|(&t, _)| t);
    let (t0, t1) = included.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| {
        (lo.min(t), hi.max(t))
    });
    let span = (t1 - t0).max(1.0);
    let pad = (growth - 1.0) / 2.0 * span;
    let grown: Vec<bool> = times
        .iter()
        .zip(mask)
        .map(|(&t, &keep)| keep || (t >= t0 - pad && t <= t1 + pad))
        .collect();
    let count = |m: &[bool]| m.iter().filter(|&&keep| keep).count();
    if count(&grown) > count(mask) {
        return grown;
    }
    let nearest = (0..times.len()).filter(|&row| !mask[row]).min_by(|&a, &b| {
        let distance = |row: usize| (times[row] - t0).abs().min((times[row] - t1).abs());
        distance(a).total_cmp(&distance(b)).then(a.cmp(&b))
    });
    match nearest {
        Some(row) => {
            let (start, end) = (0..times.len())
                .filter(|&other| nights[other] == nights[row])
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), other| {
                    (lo.min(times[other]), hi.max(times[other]))
                });
            times
                .iter()
                .zip(mask)
                .map(|(&t, &keep)| keep || (t >= start && t <= end))
                .collect()
        }
        None => mask.to_vec(),
    }
}

/// Relative reduced-chi2 difference below which two converged IOD
/// candidates tie (far above solver precision, `ftol = 1e-12`, far below
/// any difference between distinct minima that matters).
const RCHI2_TIE_RTOL: f64 = 1e-6;

/// The 6-parameter, 3-sigma (99.73 %) chi2 quantile: the boundary of the
/// joint confidence region of an orbit (Milani & Gronchi 2010, sec. 5.6).
const AMBIGUITY_DELTA_CHI2: f64 = 20.062;

/// Degrees of freedom of the whole arc: every observation's, minus the six
/// orbit parameters.
fn whole_arc_dof(evaluation: &super::od::FitEvaluation) -> usize {
    (evaluation
        .dof
        .iter()
        .map(|&d| d.max(0) as usize)
        .sum::<usize>())
    .saturating_sub(6)
}

/// [`super::od::IodCandidate::whole_arc_score`]: the whole arc's chi2 with
/// every observation clipped at `CMC2003_CHI2_REJECT` (the least a rejected
/// observation can contribute), over the whole arc's degrees of freedom.
fn whole_arc_score(evaluation: &super::od::FitEvaluation) -> f64 {
    let clipped: f64 = evaluation
        .chi2
        .iter()
        .map(|&chi2| {
            if chi2.is_finite() {
                chi2.min(crate::cmc2003::CMC2003_CHI2_REJECT)
            } else {
                crate::cmc2003::CMC2003_CHI2_REJECT
            }
        })
        .sum();
    match whole_arc_dof(evaluation) {
        0 => f64::INFINITY,
        dof => clipped / dof as f64,
    }
}

/// Whether `other` is a second solution as acceptable as `best`: outside
/// `best`'s 3-sigma confidence region (Mahalanobis distance squared under
/// `best`'s covariance, `other` moved to `best`'s epoch by two-body
/// propagation, above `AMBIGUITY_DELTA_CHI2`) yet within the same
/// chi2 margin of it (whole-arc scores times the whole arc's degrees of
/// freedom). This is
/// the multiple-solution case of Gronchi (2009) / Charlier: separate local
/// minima the observations cannot choose between. Without a finite,
/// invertible covariance the solutions are not compared.
fn is_ambiguous_with(best: &super::od::IodCandidate, other: &super::od::IodCandidate) -> bool {
    let (Some(state), Some(other_state), Some(covariance), Some(best_score), Some(other_score)) = (
        best.corrected_state,
        other.corrected_state,
        best.ambiguity_covariance,
        best.whole_arc_score,
        other.whole_arc_score,
    ) else {
        return false;
    };
    let dof = best.evaluation.as_ref().map_or(0, whole_arc_dof);
    if dof == 0 || (other_score - best_score) * dof as f64 >= AMBIGUITY_DELTA_CHI2 {
        return false;
    }
    let Ok(mu) = crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")) else {
        return false;
    };
    let moved = crate::propagate_2body_row::<f64>(
        other_state,
        best.epoch.mjd() - other.epoch.mjd(),
        mu,
        1000,
        1e-14,
    );
    let matrix = nalgebra::Matrix6::from_row_slice(&covariance);
    let Some(inverse) = matrix.try_inverse() else {
        return false;
    };
    let delta = nalgebra::Vector6::from_fn(|axis, _| moved[axis] - state[axis]);
    let distance2 = (delta.transpose() * inverse * delta)[(0, 0)];
    distance2.is_finite() && distance2 > AMBIGUITY_DELTA_CHI2
}

/// The first correction stage of a candidate: the observations its triplet
/// spans, grown with [`grow_stage`] until there are `min_stage_obs` of them
/// (or the arc's all).
pub(super) fn first_stage_mask(
    times: &[f64],
    nights: &[usize],
    triplet: [usize; 3],
    correction: &super::od::CorrectionConfig,
) -> Vec<bool> {
    let (t_lo, t_hi) = triplet
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &row| {
            (lo.min(times[row]), hi.max(times[row]))
        });
    let mut mask: Vec<bool> = times.iter().map(|&t| t >= t_lo && t <= t_hi).collect();
    let count = |m: &[bool]| m.iter().filter(|&&keep| keep).count();
    while count(&mask) < correction.min_stage_obs.min(times.len()) {
        let grown = grow_stage(times, nights, &mask, correction.growth);
        if count(&grown) == count(&mask) {
            break;
        }
        mask = grown;
    }
    mask
}

/// Corrected acceptance ([`super::od::IodAcceptance::Corrected`]): score
/// every candidate of the search, then, in increasing prefit reduced chi2
/// (index order on ties), correct up to `correction.max_corrections` of
/// them in stages: the observations spanned by the triplet (grown until it
/// holds `min_stage_obs`), then spans grown `growth` times (with the gap
/// rule of `grow_stage`) up to the whole arc, each stage seeded by the
/// previous fit and kept only within `rchi2_threshold`. No stage before the
/// whole arc rejects observations: an orbit fitted to part of the arc cannot
/// yet tell an outlier from an observation it does not predict well. Those
/// stages use `correction.stage_loss` (Huber by default: down-weighted,
/// gated on the robust cost); the whole-arc stage is linear. On the
/// whole arc, a fit above the threshold gets the CMC2003 rejection loop
/// (Carpino, Milani & Chesley 2003, OrbFit's constants, but rejecting at
/// most `contamination_percentage` of the arc) and passes if that brings it
/// within the threshold. The search stops once `max_converged`
/// candidates reached the whole arc; every other Gauss root of a converged
/// candidate's triplet (Charlier: up to three per triplet) is corrected too,
/// including roots for which `Auto` resolved a different velocity method,
/// before the search moves on or stops, outside the `max_corrections`
/// budget. The accepted candidate is the converged one with the lowest
/// whole-arc score (chi2 over every observation, each clipped at the CMC2003
/// rejection threshold), the lowest index among those within a relative
/// `RCHI2_TIE_RTOL` of it, never the root order. Its orbit is that whole-arc fit at the Gauss epoch, rejected rows
/// reported as outliers. Converged candidates that are distinct solutions
/// fitting about as well (`is_ambiguous_with`) are listed in
/// `ambiguous_with`.
///
/// Arcs Gauss cannot handle get systematic ranging (`config.ranging`,
/// `super::od_ranging`): an arc shorter than `min_arc_length` (or with too
/// few observations) is not searched and returns the ranging sample in
/// `IodResult::ranging`; when no Gauss candidate converges, the best
/// `ranging.max_corrections` ranging orbits are corrected exactly like
/// Gauss candidates (`CandidateSource::Ranging`).
#[allow(clippy::too_many_arguments)]
pub fn run_iod_corrected<T: OriginTranslationProvider>(
    iod_predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &super::od::IodConfig,
    fit: &WhitenedFitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<super::od::IodResult> {
    if observers.len() != observed.len() {
        return Err(PropagationError::InvalidRequest(
            "observed and observers must have equal length".to_string(),
        ));
    }
    // Search, corrections and ranging all run on the canonical (time-sorted)
    // view, so the result is independent of the caller's row order down to
    // the corrected states; rows are mapped back at the end.
    let order = super::od::iod_observation_order(observed, observers)?;
    let observed = &super::od::take_coordinate_batch(observed, &order)?;
    let observers = &super::od::take_observer_batch(observers, &order)?;
    let mut result = super::od::iod_candidate_pool(iod_predictor, observed, observers, config)?;
    if !result.candidates.is_empty() {
        let pool: Vec<usize> = (0..result.candidates.len()).collect();
        correct_candidates(
            iod_predictor,
            observed,
            observers,
            config,
            fit,
            provider,
            translation_provider,
            &mut result,
            &pool,
        )?;
    }
    if result.accepted.is_none() && config.ranging.enabled {
        ranging_fallback(
            iod_predictor,
            observed,
            observers,
            config,
            fit,
            provider,
            translation_provider,
            &mut result,
        )?;
    }
    Ok(super::od::restore_input_order(result, &order))
}

/// Systematic ranging when Gauss found nothing: attach the sample, and on an
/// arc long enough to accept an orbit correct its best orbits as candidates.
#[allow(clippy::too_many_arguments)]
fn ranging_fallback<T: OriginTranslationProvider>(
    iod_predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &super::od::IodConfig,
    fit: &WhitenedFitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
    result: &mut super::od::IodResult,
) -> PropagationResultValue<()> {
    let times = iod_predictor.dynamical_times_mjd(observed)?;
    let span = times.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - times.iter().copied().fold(f64::INFINITY, f64::min);
    if span.is_nan() || span > config.ranging.max_arc_days {
        return Ok(());
    }
    let ranging = match super::od_ranging::systematic_ranging_with(
        iod_predictor,
        observed,
        observers,
        &config.ranging,
        config.mu,
        provider,
        translation_provider,
    ) {
        Ok(ranging) => ranging,
        // Too little data for an attributable (one epoch): nothing to add.
        Err(PropagationError::InvalidRequest(_)) => return Ok(()),
        Err(err) => return Err(err),
    };
    let n = observed.len();
    let (first, last) = (0..n).fold((0, 0), |(lo, hi), row| {
        (
            if times[row] < times[lo] { row } else { lo },
            if times[row] > times[hi] { row } else { hi },
        )
    });
    let arc = times[last] - times[first];
    let long_enough = n >= config.min_obs && arc >= config.min_arc_length;
    if long_enough && !ranging.samples.is_empty() {
        let attributable = &ranging.attributable;
        let ranked: Vec<usize> = ranging
            .ranked()
            .into_iter()
            .take(config.ranging.max_corrections.max(1))
            .collect();
        let count = ranked.len();
        let dof = (2 * n).saturating_sub(6).max(1) as f64;
        let start = result.candidates.len();
        for (rank, &index) in ranked.iter().enumerate() {
            let sample = &ranging.samples[index];
            result.candidates.push(super::od::IodCandidate {
                source: super::od::CandidateSource::Ranging,
                state: sample.state,
                epoch: attributable.epoch,
                epoch_scale: attributable.epoch_scale,
                window: (first, last),
                triplet: [attributable.reference_row; 3],
                root_index: rank,
                root_count: count,
                velocity_method: config.velocity_method,
                prefit_chi2: sample.chi2,
                prefit_reduced_chi2: sample.chi2 / dof,
                removed: Vec::new(),
                chi2_total: sample.chi2,
                reduced_chi2: sample.chi2 / dof,
                num_obs: n,
                arc_length: arc,
                postfit_reduced_chi2: None,
                whole_arc_score: None,
                converged: false,
                corrected_state: None,
                corrected_covariance: None,
                ambiguity_covariance: None,
                stages: 0,
                correction_note: None,
                accepted: false,
                evaluation: None,
            });
        }
        let pool: Vec<usize> = (start..result.candidates.len()).collect();
        correct_candidates(
            iod_predictor,
            observed,
            observers,
            config,
            fit,
            provider,
            translation_provider,
            result,
            &pool,
        )?;
    } else if let Some(reason) = result.not_found_reason.as_mut() {
        reason.push_str(&format!(
            "; ranging sample attached ({} admissible orbits)",
            ranging.samples.len()
        ));
    }
    result.ranging = Some(ranging);
    Ok(())
}

/// Covariance from an intermediate IOD correction is reportable only when it
/// passes the same general-fit structural/rank gate as a final fit. The state
/// may still seed final refinement when this returns `None`.
fn reportable_iod_covariance(covariance: &[f64; 36]) -> Option<[f64; 36]> {
    covariance_validity(covariance)
        .is_ok()
        .then_some(*covariance)
}

/// Less restrictive internal gate used only for truthful multiple-solution
/// comparison. A covariance that fails solely at the reportable numerical
/// rank floor remains useful here; non-finite, asymmetric, singular, or
/// non-positive-definite matrices do not.
fn ambiguity_iod_covariance(covariance: &[f64; 36]) -> Option<[f64; 36]> {
    covariance_correlation_spectrum(covariance)
        .is_ok()
        .then_some(*covariance)
}

/// Record the only whole-arc rejection failures that are local to one
/// corrected-IOD candidate. Provider, protocol, settings, schema, and global
/// failures remain fatal and are returned unchanged.
fn record_candidate_local_rejection_failure(
    error: PropagationError,
    stages: usize,
    triplet: [usize; 3],
    root_index: usize,
    result: &mut super::od::IodResult,
) -> PropagationResultValue<String> {
    match error {
        err @ (PropagationError::RowFailure(_) | PropagationError::CmcCovarianceUnavailable(_)) => {
            let message = format!("whole-arc rejection failed at stage {stages}: {err}");
            result.candidates_failed += 1;
            result
                .candidate_failures
                .push(super::od::IodCandidateFailure {
                    triplet,
                    root_index,
                    message: message.clone(),
                });
            Ok(message)
        }
        error => Err(error),
    }
}

/// Correct the candidates in `pool` (see [`run_iod_corrected`]) and select
/// the accepted one among every converged candidate of `result`.
#[allow(clippy::too_many_arguments)]
fn correct_candidates<T: OriginTranslationProvider>(
    iod_predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &super::od::IodConfig,
    fit: &WhitenedFitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
    result: &mut super::od::IodResult,
    pool: &[usize],
) -> PropagationResultValue<()> {
    let n = observed.len();
    let times = iod_predictor.civil_times_mjd(observed)?;
    let nights = super::od::observation_nights(&times, Some(&observers.code));
    let stage_config = WhitenedFitConfig {
        loss: LossType::Linear,
        validate_covariance: false,
        ..*fit
    };
    let correction = config.correction;
    let partial_config = WhitenedFitConfig {
        loss: correction.stage_loss,
        ..stage_config
    };
    let mut order: Vec<usize> = pool.to_vec();
    order.sort_by(|&a, &b| {
        result.candidates[a]
            .prefit_reduced_chi2
            .total_cmp(&result.candidates[b].prefit_reduced_chi2)
            .then(a.cmp(&b))
    });
    let mut converged = 0_usize;
    let mut last_note: Option<String> = None;
    let mut primary = order.into_iter().take(correction.max_corrections.max(1));
    // Sibling roots of a converged candidate, corrected before the search
    // moves on or stops (lowest root index first).
    let mut siblings: Vec<usize> = Vec::new();
    loop {
        let index = match siblings.pop() {
            Some(index) => index,
            None => {
                if converged >= correction.max_converged.max(1) {
                    break;
                }
                match primary.next() {
                    Some(index) => index,
                    None => break,
                }
            }
        };
        if result.candidates[index].stages > 0 {
            continue;
        }
        result.corrections_run += 1;
        let candidate = &result.candidates[index];
        let seed_coordinates = CoordinateBatch::cartesian(
            vec![candidate.state],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::from_code("SUN"), 1),
            Some(crate::TimeArray::new(
                candidate.epoch_scale,
                vec![candidate.epoch],
            )?),
            None,
        )?;
        let mut seed = OrbitBatch::new(
            vec![crate::OrbitId("iod-candidate".to_string())],
            vec![None],
            seed_coordinates,
        )?;
        let mut mask = first_stage_mask(&times, &nights, candidate.triplet, &correction);
        let mut stages = 0_usize;
        let mut note: Option<String> = None;
        let mut last_fit: Option<WhitenedFitOutput> = None;
        loop {
            if stages >= correction.max_stages {
                note = Some(format!(
                    "stopped after {stages} stages short of the whole arc"
                ));
                break;
            }
            stages += 1;
            let whole_arc = mask.iter().all(|&keep| keep);
            // The whole-arc stage is judged after rejection (below).
            let threshold = if whole_arc {
                f64::INFINITY
            } else {
                config.rchi2_threshold
            };
            let output = match stage_fit(
                iod_predictor,
                &seed,
                observed,
                observers,
                &mask,
                if whole_arc {
                    &stage_config
                } else {
                    &partial_config
                },
                threshold,
                provider,
                translation_provider,
            )? {
                Ok(output) => output,
                Err((message, prediction_failed)) => {
                    let message = format!("stalled at stage {stages}: {message}");
                    if prediction_failed {
                        result.candidates_failed += 1;
                        result
                            .candidate_failures
                            .push(super::od::IodCandidateFailure {
                                triplet: candidate.triplet,
                                root_index: candidate.root_index,
                                message: message.clone(),
                            });
                    }
                    note = Some(message);
                    break;
                }
            };
            seed = orbit_with_state(&seed, output.state)?;
            if !whole_arc {
                mask = grow_stage(&times, &nights, &mask, correction.growth);
                continue;
            }
            let output = if output.evaluation.reduced_chi2 <= config.rchi2_threshold {
                output
            } else {
                match cmc2003_fit_with(
                    iod_predictor,
                    &seed,
                    observed,
                    observers,
                    &Cmc2003FitConfig {
                        fit: stage_config,
                        // The IOD's own outlier allowance, not OrbFit's 50 %
                        // refinement bound: a wrong orbit must not fit by
                        // discarding the half of the arc it cannot explain
                        // (2002 RV112: 22 of 44 rejected).
                        max_rejected_fraction: (config.contamination_percentage / 100.0)
                            .clamp(0.0, 1.0),
                        // Rejection never leaves fewer than the IOD's min_obs.
                        min_kept: config.min_obs,
                        ..Cmc2003FitConfig::default()
                    },
                    provider,
                    translation_provider,
                ) {
                    Ok(rejected) => rejected.fit,
                    Err(error) => {
                        let message = record_candidate_local_rejection_failure(
                            error,
                            stages,
                            candidate.triplet,
                            candidate.root_index,
                            result,
                        )?;
                        note = Some(message);
                        break;
                    }
                }
            };
            let rchi2 = output.evaluation.reduced_chi2;
            if output.evaluation.num_obs < config.min_obs {
                note = Some(format!(
                    "stalled at stage {stages}: whole arc after rejection keeps {} \
                     observations, fewer than min_obs = {}",
                    output.evaluation.num_obs, config.min_obs
                ));
            } else if output.evaluation.arc_length < config.min_arc_length {
                note = Some(format!(
                    "stalled at stage {stages}: whole arc after rejection spans {:.4} days, \
                     shorter than min_arc_length = {}",
                    output.evaluation.arc_length, config.min_arc_length
                ));
            } else if rchi2.is_finite() && rchi2 <= config.rchi2_threshold {
                last_fit = Some(output);
            } else {
                note = Some(format!(
                    "stalled at stage {stages}: whole arc after rejection: reduced chi2 \
                     {rchi2:.4} ({} observations kept)",
                    output.evaluation.num_obs
                ));
            }
            break;
        }
        let record = &mut result.candidates[index];
        record.stages = stages;
        match (last_fit, note) {
            (Some(output), None) => {
                record.converged = true;
                record.postfit_reduced_chi2 = Some(output.evaluation.reduced_chi2);
                record.whole_arc_score = Some(whole_arc_score(&output.evaluation));
                record.corrected_state = Some(output.state);
                record.corrected_covariance = reportable_iod_covariance(&output.covariance);
                record.ambiguity_covariance = ambiguity_iod_covariance(&output.covariance);
                record.removed = (0..n)
                    .filter(|&row| output.evaluation.outlier[row])
                    .collect();
                record.evaluation = Some(output.evaluation);
                converged += 1;
                let (window, triplet) = (record.window, record.triplet);
                let mut roots: Vec<usize> = (0..result.candidates.len())
                    .filter(|&other| {
                        let c = &result.candidates[other];
                        other != index
                            && c.stages == 0
                            && c.window == window
                            && c.triplet == triplet
                            && !siblings.contains(&other)
                    })
                    .collect();
                roots.sort_by_key(|&other| std::cmp::Reverse(result.candidates[other].root_index));
                siblings.extend(roots);
            }
            (_, note) => {
                last_note = note.clone();
                record.correction_note = note;
            }
        }
    }
    // Roots of one triplet often converge to the same minimum, their
    // scores equal to solver precision: values within
    // `RCHI2_TIE_RTOL` of the lowest tie, and the lowest candidate index
    // (canonical observation order) wins, so the choice does not depend on
    // rounding. A distinct solution that close is reported in
    // `ambiguous_with`.
    // Ranked on the whole arc (`whole_arc_score`), never on the
    // observations each candidate kept: 1.92 on 22 of 44 observations must
    // not beat 2.15 on all 44.
    let postfit =
        |candidate: &super::od::IodCandidate| candidate.whole_arc_score.unwrap_or(f64::INFINITY);
    let lowest = result
        .candidates
        .iter()
        .filter(|candidate| candidate.converged)
        .map(postfit)
        .fold(f64::INFINITY, f64::min);
    let best = result.candidates.iter().position(|candidate| {
        candidate.converged && postfit(candidate) <= lowest + RCHI2_TIE_RTOL * lowest.abs()
    });
    match best {
        Some(index) => {
            result.candidates[index].accepted = true;
            result.accepted = Some(index);
            result.not_found_reason = None;
            let accepted = &result.candidates[index];
            result.ambiguous_with = (0..result.candidates.len())
                .filter(|&other| {
                    other != index
                        && result.candidates[other].converged
                        && is_ambiguous_with(accepted, &result.candidates[other])
                })
                .collect();
        }
        None => {
            result.not_found_reason = Some(format!(
                "{} of {} candidates corrected, all roots failed correction: none reached \
                 the whole arc within rchi2_threshold = {}{}",
                result.corrections_run,
                result.candidates.len(),
                config.rchi2_threshold,
                last_note
                    .map(|note| format!(" (last: {note})"))
                    .unwrap_or_default()
            ));
        }
    }
    Ok(())
}

/// The IOD stage over `iod_propagator` (the canonical `run_iod` stage of
/// the review's `run_iod` / `refine` / `run_od` contract): the candidate
/// search with `config.acceptance` (corrected acceptance runs its stages
/// with `fit`). Its [`super::od::IodResult::seed_orbit`] is the typed
/// hand-off to [`refine_barycentric`].
#[allow(clippy::too_many_arguments)]
pub fn run_iod_barycentric<I, T>(
    iod_propagator: &I,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &super::od::IodConfig,
    fit: &WhitenedFitConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<super::od::IodResult>
where
    I: Propagator,
    T: OriginTranslationProvider,
{
    let predictor = PropagatorPredictor {
        propagator: iod_propagator,
        options,
        provider,
        translation_provider,
    };
    run_iod_with(
        &predictor,
        observed,
        observers,
        config,
        fit,
        provider,
        translation_provider,
    )
}

/// [`run_iod_with`] over many linkage row selections in one workflow.
#[allow(clippy::too_many_arguments)]
pub fn run_iod_linkages_with<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    linkage_rows: &[Vec<usize>],
    config: &super::od::IodConfig,
    fit: &WhitenedFitConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<Vec<super::od::IodResult>> {
    let mut outputs = Vec::with_capacity(linkage_rows.len());
    for rows in linkage_rows {
        if let Some(&row) = rows.iter().find(|&&row| row >= observed.len()) {
            return Err(PropagationError::InvalidRequest(format!(
                "linkage observation row {row} is out of bounds for {} observations",
                observed.len()
            )));
        }
        outputs.push(run_iod_with(
            predictor,
            &super::od::take_coordinate_batch(observed, rows)?,
            &super::od::take_observer_batch(observers, rows)?,
            config,
            fit,
            provider,
            translation_provider,
        )?);
    }
    Ok(outputs)
}

/// [`run_iod_barycentric`] over many linkages in one call: each entry of
/// `linkage_rows` selects that linkage's observation rows, and the outputs
/// are in `linkage_rows` order. A linkage whose search finds nothing yields
/// an [`super::od::IodResult`] without an accepted candidate. Fatal backend,
/// protocol, schema, request and configuration errors abort the entire batch;
/// an out-of-bounds linkage row is likewise invalid input for the batch.
#[allow(clippy::too_many_arguments)]
pub fn run_iod_linkages_barycentric<I, T>(
    iod_propagator: &I,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    linkage_rows: &[Vec<usize>],
    config: &super::od::IodConfig,
    fit: &WhitenedFitConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<Vec<super::od::IodResult>>
where
    I: Propagator,
    T: OriginTranslationProvider,
{
    let predictor = PropagatorPredictor {
        propagator: iod_propagator,
        options,
        provider,
        translation_provider,
    };
    run_iod_linkages_with(
        &predictor,
        observed,
        observers,
        linkage_rows,
        config,
        fit,
        provider,
        translation_provider,
    )
}

/// The refinement stage over `propagator`: differential correction of
/// `seed` (e.g. [`super::od::IodResult::seed_orbit`]) against every
/// observation with the configured outlier treatment.
#[allow(clippy::too_many_arguments)]
pub fn refine_barycentric<R, T>(
    propagator: &R,
    seed: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &RefinementConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<RefinedOutput>
where
    R: Propagator,
    T: OriginTranslationProvider,
{
    let predictor = PropagatorPredictor {
        propagator,
        options,
        provider,
        translation_provider,
    };
    refine_with(
        &predictor,
        seed,
        observed,
        observers,
        config,
        provider,
        translation_provider,
    )
}

/// [`refine_barycentric`] over a [`SphericalPredictor`] adapter.
#[allow(clippy::too_many_arguments)]
pub fn refine_with<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    seed: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &RefinementConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<RefinedOutput> {
    config
        .validate()
        .map_err(PropagationError::InvalidRequest)?;
    Ok(match config {
        RefinementConfig::WorstResidual(loop_config) => {
            let output = iterative_fit_with(
                predictor,
                seed,
                observed,
                observers,
                loop_config,
                provider,
                translation_provider,
            )?;
            RefinedOutput {
                fit: output.fit,
                passes: output.passes,
                n_rejected: 0,
                n_recovered: 0,
                flags: Vec::new(),
            }
        }
        RefinementConfig::Cmc2003(loop_config) => {
            let output = cmc2003_fit_with(
                predictor,
                seed,
                observed,
                observers,
                loop_config,
                provider,
                translation_provider,
            )?;
            RefinedOutput {
                fit: output.fit,
                passes: output.n_iterations,
                n_rejected: output.n_rejected,
                n_recovered: output.n_recovered,
                flags: output.flags,
            }
        }
    })
}

/// One-crossing `OrbitFitter.full_od` for the native fitter: the complete
/// Gauss IOD decision loop (`run_iod_barycentric` on `iod_propagator`)
/// followed by the worst-residual or CMC2003 refinement of the accepted
/// orbit against all observations on `propagator`. The two propagators are
/// explicit; passing the same one twice is the single-backend case.
#[allow(clippy::too_many_arguments)]
pub fn full_od_barycentric<I, R, T>(
    iod_propagator: &I,
    propagator: &R,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &FullOdConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<FullOdOutput>
where
    I: Propagator,
    R: Propagator,
    T: OriginTranslationProvider,
{
    let iod_predictor = PropagatorPredictor {
        propagator: iod_propagator,
        options,
        provider,
        translation_provider,
    };
    let predictor = PropagatorPredictor {
        propagator,
        options,
        provider,
        translation_provider,
    };
    full_od_with(
        &iod_predictor,
        &predictor,
        observed,
        observers,
        config,
        provider,
        translation_provider,
    )
}

/// [`full_od_barycentric`] over any pair of [`SphericalPredictor`]s: IOD
/// scores candidates on `iod_predictor`, refinement runs on `predictor`.
#[allow(clippy::too_many_arguments)]
pub fn full_od_with<T: OriginTranslationProvider>(
    iod_predictor: &dyn SphericalPredictor,
    predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &FullOdConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<FullOdOutput> {
    config
        .validate()
        .map_err(PropagationError::InvalidRequest)?;
    let refinement_fit = match &config.refinement {
        RefinementConfig::WorstResidual(loop_config) => loop_config.fit,
        RefinementConfig::Cmc2003(loop_config) => loop_config.fit,
    };
    let iod = run_iod_with(
        iod_predictor,
        observed,
        observers,
        &config.iod,
        &refinement_fit,
        provider,
        translation_provider,
    )?;
    // The composition of the two stages: the IOD's typed seed (exact
    // epoch) refined against every observation.
    let Some(seed) = iod.seed_orbit()? else {
        return Ok(FullOdOutput { iod, refined: None });
    };
    let refined = refine_with(
        predictor,
        &seed,
        observed,
        observers,
        &config.refinement,
        provider,
        translation_provider,
    )?;
    Ok(FullOdOutput {
        iod,
        refined: Some(refined),
    })
}

/// `ObservationAstrometry`: position and 1-sigma uncertainty (degrees) of
/// every observation, the provenance columns of fitted orbit members.
#[derive(Debug, Clone, PartialEq)]
pub struct AstrometrySnapshot {
    pub lon: Vec<f64>,
    pub lat: Vec<f64>,
    pub sigma_lon: Vec<f64>,
    pub sigma_lat: Vec<f64>,
    pub cov_lonlat: Vec<f64>,
}

impl AstrometrySnapshot {
    fn from_columns(lon: &[f64], lat: &[f64], covariance: &[f64]) -> Self {
        let sqrt_or_nan = |value: f64| if value >= 0.0 { value.sqrt() } else { f64::NAN };
        let n = lon.len();
        Self {
            lon: lon.to_vec(),
            lat: lat.to_vec(),
            sigma_lon: (0..n)
                .map(|row| sqrt_or_nan(covariance[row * 36 + 7]))
                .collect(),
            sigma_lat: (0..n)
                .map(|row| sqrt_or_nan(covariance[row * 36 + 14]))
                .collect(),
            cov_lonlat: (0..n).map(|row| covariance[row * 36 + 8]).collect(),
        }
    }
}

/// Product of `run_od_barycentric`: the astrometry as supplied and as used by
/// the fit, and the full OD run on the used observations.
#[derive(Debug, Clone)]
pub struct RunOdOutput {
    pub original: AstrometrySnapshot,
    pub used: AstrometrySnapshot,
    /// Whether any model changed a value.
    pub models_changed: bool,
    pub full_od: FullOdOutput,
}

/// One-crossing `run_od` for the native fitter: apply the observation models
/// to the ORIGINAL observations in order, run the full orbit determination
/// on the transformed ("used") observations, and return both astrometry
/// snapshots for the members' provenance columns. `obs_code`, `band` and
/// `astcat` are the per-observation station, photometric band and star
/// catalog the models consult.
///
/// The models run before a covariance is required: observations without
/// one (`None`) start as NaN-filled covariance rows, so that a
/// `SigmaFillModel` among `models` can supply the missing sigmas exactly as
/// it does on the composed Python route. After the models every observation
/// must have finite, positive lon/lat variances, else `InvalidRequest`
/// names that model as the remedy.
#[allow(clippy::too_many_arguments)]
pub fn run_od_barycentric<I, R, T>(
    iod_propagator: &I,
    propagator: &R,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    obs_code: &[String],
    band: &[Option<String>],
    astcat: &[Option<String>],
    models: Vec<Box<dyn ObservationUncertaintyModel>>,
    config: &FullOdConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<RunOdOutput>
where
    I: Propagator,
    R: Propagator,
    T: OriginTranslationProvider,
{
    let iod_predictor = PropagatorPredictor {
        propagator: iod_propagator,
        options,
        provider,
        translation_provider,
    };
    let predictor = PropagatorPredictor {
        propagator,
        options,
        provider,
        translation_provider,
    };
    run_od_with(
        &iod_predictor,
        &predictor,
        observed,
        observers,
        obs_code,
        band,
        astcat,
        models,
        config,
        provider,
        translation_provider,
    )
}

/// [`run_od_barycentric`] over any pair of [`SphericalPredictor`]s (IOD,
/// refinement).
#[allow(clippy::too_many_arguments)]
pub fn run_od_with<T: OriginTranslationProvider>(
    iod_predictor: &dyn SphericalPredictor,
    predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    obs_code: &[String],
    band: &[Option<String>],
    astcat: &[Option<String>],
    models: Vec<Box<dyn ObservationUncertaintyModel>>,
    config: &FullOdConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<RunOdOutput> {
    config
        .validate()
        .map_err(PropagationError::InvalidRequest)?;
    let n = observed.len();
    if observers.len() != n || obs_code.len() != n || band.len() != n || astcat.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed, observers, obs_code, band and astcat must have equal length".to_string(),
        ));
    }
    let observed_flat = spherical_flat(observed, "observed")?;
    // No covariance yet: the models get NaN rows to fill (see above).
    let covariance = match &observed.covariance {
        Some(source) => source.values_row_major.clone(),
        None => vec![f64::NAN; n * 36],
    };
    let lon: Vec<f64> = (0..n).map(|row| observed_flat[row * 6 + 1]).collect();
    let lat: Vec<f64> = (0..n).map(|row| observed_flat[row * 6 + 2]).collect();
    let original = AstrometrySnapshot::from_columns(&lon, &lat, &covariance);

    let times = observed.times.as_ref().ok_or_else(|| {
        PropagationError::InvalidRequest(
            "orbit determination requires observation times".to_string(),
        )
    })?;
    let mjd_utc = times
        .rescale_with_provider(TimeScale::Utc, provider)?
        .mjd_values();
    let jd_tdb: Vec<f64> = times
        .rescale_with_provider(TimeScale::Tdb, provider)?
        .mjd_values()
        .into_iter()
        .map(|mjd| mjd + 2_400_000.5)
        .collect();
    let mut astrometry = OrbitDeterminationAstrometry {
        lon,
        lat,
        covariance,
        obs_code: obs_code.to_vec(),
        band: band.to_vec(),
        astcat: astcat.to_vec(),
        mjd_utc,
        jd_tdb,
    };
    let models_changed = if models.is_empty() {
        false
    } else {
        crate::observation_uncertainty::CompositeModel::new(models)
            .map_err(PropagationError::InvalidRequest)?
            .apply(&mut astrometry)
            .map_err(PropagationError::InvalidRequest)?
    };
    let used_snapshot =
        AstrometrySnapshot::from_columns(&astrometry.lon, &astrometry.lat, &astrometry.covariance);

    let missing: Vec<usize> = (0..n)
        .filter(|&row| {
            let var_lon = astrometry.covariance[row * 36 + 7];
            let var_lat = astrometry.covariance[row * 36 + 14];
            !(var_lon.is_finite() && var_lon > 0.0 && var_lat.is_finite() && var_lat > 0.0)
        })
        .collect();
    if !missing.is_empty() {
        return Err(PropagationError::InvalidRequest(format!(
            "{} of {n} observations (first: row {}) have no finite, positive lon/lat \
             variance after the observation models; {}",
            missing.len(),
            missing[0],
            super::od::MISSING_COVARIANCE_REMEDY
        )));
    }

    let used = if models_changed || observed.covariance.is_none() {
        let mut values = observed.values.raw_values().to_vec();
        for (row, value) in values.iter_mut().enumerate() {
            value[1] = astrometry.lon[row];
            value[2] = astrometry.lat[row];
        }
        let (units, row_validity) = match &observed.covariance {
            Some(source) => (source.units.clone(), source.row_validity.clone()),
            None => (
                crate::CovarianceUnits::Coordinate(crate::CoordinateRepresentation::Spherical),
                None,
            ),
        };
        let mut used_covariance =
            crate::CovarianceBatch::new(n, 6, astrometry.covariance.clone(), units)?;
        if let Some(validity) = row_validity {
            used_covariance = used_covariance.with_row_validity(validity)?;
        }
        CoordinateBatch::new(
            crate::CoordinateValues::Spherical(values),
            observed.frame,
            observed.origins.clone(),
            observed.times.clone(),
            Some(used_covariance),
        )?
    } else {
        observed.clone()
    };

    let full_od = full_od_with(
        iod_predictor,
        predictor,
        &used,
        observers,
        config,
        provider,
        translation_provider,
    )?;
    Ok(RunOdOutput {
        original,
        used: used_snapshot,
        models_changed,
        full_od,
    })
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::propagation::{
        generate_ephemeris_barycentric, CovariancePropagation, EpochPolicy, PropagationOptions,
        TwoBodyPropagator,
    };
    use crate::types::{SchemaError, SchemaResult};
    use crate::{
        CovarianceBatch, CovarianceUnits, Epoch, ObjectId, ObservatoryCode, OrbitId, TimeArray,
    };

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
    const NUM_OBS: usize = 12;
    const SIGMA_DEG: f64 = 0.5 / 3600.0;

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

    fn orbit(state: [f64; 6]) -> OrbitBatch {
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
        observers_spaced(4)
    }

    /// Observers every `spacing_days` days: 4 gives the 44-day arc of the
    /// fit and loop tests; 1 gives an 11-day arc on which Gauss IOD seeds
    /// the refinement accurately (the full-OD tests).
    fn observers_spaced(spacing_days: i64) -> ObserverBatch {
        let times = TimeArray::new(
            TimeScale::Tdb,
            (0..NUM_OBS)
                .map(|row| Epoch::new(60_002 + spacing_days * row as i64, 0))
                .collect(),
        )
        .unwrap();
        let states: Vec<[f64; 6]> = (0..NUM_OBS)
            .map(|row| {
                let theta = 0.35 + 0.01725 * (spacing_days * row as i64) as f64;
                [
                    theta.cos(),
                    theta.sin(),
                    0.0,
                    // Consistent with the positions (0.01725 rad/day).
                    -0.01725 * theta.sin(),
                    0.01725 * theta.cos(),
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

    /// Spherical observations synthesized from the truth orbit with a
    /// deterministic pseudo-random sub-sigma scatter, plus one optional
    /// gross outlier.
    fn synthetic_observations(outlier_row: Option<usize>) -> CoordinateBatch {
        synthetic_observations_spaced(4, outlier_row)
    }

    fn synthetic_observations_spaced(
        spacing_days: i64,
        outlier_row: Option<usize>,
    ) -> CoordinateBatch {
        synthetic_observations_with_outliers(spacing_days, outlier_row.as_slice())
    }

    /// [`synthetic_observations_spaced`] with a 12-sigma outlier on every
    /// row of `outlier_rows`.
    fn synthetic_observations_with_outliers(
        spacing_days: i64,
        outlier_rows: &[usize],
    ) -> CoordinateBatch {
        let result = generate_ephemeris_barycentric(
            &TwoBodyPropagator::default(),
            &orbit(TRUTH_STATE),
            &observers_spaced(spacing_days),
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let predicted = result.ephemeris.coordinates;
        let mut values = predicted.values.raw_values().to_vec();
        let mut seed = 12345_u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.6 * SIGMA_DEG
        };
        for (row, value) in values.iter_mut().enumerate() {
            let cos_lat = value[2].to_radians().cos();
            value[1] += noise() / cos_lat;
            value[2] += noise();
            if outlier_rows.contains(&row) {
                value[2] += 12.0 * SIGMA_DEG;
            }
            value[0] = f64::NAN;
            value[3] = f64::NAN;
            value[4] = f64::NAN;
            value[5] = f64::NAN;
        }
        let mut covariance = vec![f64::NAN; NUM_OBS * 36];
        for row in 0..NUM_OBS {
            covariance[row * 36 + 7] = SIGMA_DEG * SIGMA_DEG;
            covariance[row * 36 + 14] = SIGMA_DEG * SIGMA_DEG;
        }
        CoordinateBatch::new(
            crate::CoordinateValues::Spherical(values),
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
        state[0] += 2e-4;
        state[1] -= 1e-4;
        state[2] += 5e-5;
        state[3] += 2e-6;
        state[4] -= 1e-6;
        state[5] += 1e-6;
        orbit(state)
    }

    fn fit(
        observed: &CoordinateBatch,
        ignore: &[bool],
        config: &WhitenedFitConfig,
    ) -> WhitenedFitOutput {
        fit_orbit_whitened_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            observed,
            &observers(),
            ignore,
            config,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// Deterministic non-Keplerian predictor porting the retired Python
    /// closed-form oracle behavior entirely into Rust:
    /// `prediction = base + A * (state - truth)`.
    struct LinearPredictor {
        base: Vec<[f64; 6]>,
        design: Vec<[f64; 6]>,
        truth: [f64; 6],
        batches: std::cell::RefCell<Vec<usize>>,
    }

    impl SphericalPredictor for LinearPredictor {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            _geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            self.batches.borrow_mut().push(states.len());
            let indices: PropagationResultValue<Vec<usize>> = observers
                .code
                .iter()
                .map(|code| {
                    code.0
                        .strip_prefix('T')
                        .and_then(|value| value.parse::<usize>().ok())
                        .filter(|&index| index < self.base.len())
                        .ok_or_else(|| {
                            PropagationError::InvalidRequest(format!(
                                "linear predictor received unknown observer code {:?}",
                                code.0
                            ))
                        })
                })
                .collect();
            let indices = indices?;
            let mut output = Vec::with_capacity(states.len() * observers.len() * 6);
            for state in states {
                let delta: [f64; 6] = std::array::from_fn(|axis| state[axis] - self.truth[axis]);
                for &row in &indices {
                    let mut predicted = self.base[row];
                    for component in 0..2 {
                        predicted[1 + component] += self.design[2 * row + component]
                            .iter()
                            .zip(delta)
                            .map(|(sensitivity, offset)| sensitivity * offset)
                            .sum::<f64>();
                    }
                    output.extend_from_slice(&predicted);
                }
            }
            Ok(Ok(output))
        }

        fn backend(&self) -> super::super::BackendIdentity {
            super::super::BackendIdentity {
                name: "test.linear.non_keplerian".to_string(),
                version: "1".to_string(),
                two_body_dynamics: false,
            }
        }
    }

    struct LinearFixture {
        observed: CoordinateBatch,
        predictor: LinearPredictor,
        /// Per-observation A^T R^-1 A matrices.
        normal_rows: Vec<[[f64; 6]; 6]>,
        /// Per-component whitened design rows (correlation-free fixtures).
        whitened_design: Vec<[f64; 6]>,
    }

    fn add_normal(target: &mut [[f64; 6]; 6], source: &[[f64; 6]; 6]) {
        for i in 0..6 {
            for j in 0..6 {
                target[i][j] += source[i][j];
            }
        }
    }

    fn invert_normal(normal: &[[f64; 6]; 6]) -> [f64; 36] {
        let inverse = inverse_6x6(normal).expect("deterministic design is full rank");
        std::array::from_fn(|index| inverse[index])
    }

    fn linear_normal_matrix(fixture: &LinearFixture, selected: &[bool]) -> [[f64; 6]; 6] {
        let mut normal = [[0.0_f64; 6]; 6];
        for (row, row_normal) in fixture.normal_rows.iter().enumerate() {
            if selected[row] {
                add_normal(&mut normal, row_normal);
            }
        }
        normal
    }

    fn expected_linear_covariance(fixture: &LinearFixture, selected: &[bool]) -> [f64; 36] {
        invert_normal(&linear_normal_matrix(fixture, selected))
    }

    fn assert_normal_times_covariance_is_identity(
        normal: &[[f64; 6]; 6],
        covariance: &[f64; 36],
        tolerance: f64,
    ) {
        for i in 0..6 {
            for j in 0..6 {
                let product: f64 = (0..6).map(|k| normal[i][k] * covariance[k * 6 + j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product - expected).abs() < tolerance,
                    "(normal * covariance)[{i},{j}] = {product}"
                );
            }
        }
    }

    fn central_jacobian_with_relative_step(
        problem: &WhitenedProblem<'_>,
        state: &[f64; 6],
        relative_step: f64,
    ) -> Vec<f64> {
        let mut states = Vec::with_capacity(12);
        let mut steps = [0.0_f64; 6];
        for axis in 0..6 {
            let step = relative_step * state[axis].abs().max(1e-3);
            steps[axis] = step;
            let mut plus = *state;
            let mut minus = *state;
            plus[axis] += step;
            minus[axis] -= step;
            states.push(plus);
            states.push(minus);
        }
        let batch = problem.residuals(&states).unwrap().unwrap();
        let rows = 2 * problem.m;
        let mut jacobian = vec![0.0_f64; rows * 6];
        for axis in 0..6 {
            for row in 0..rows {
                jacobian[row * 6 + axis] =
                    (batch[2 * axis][row] - batch[2 * axis + 1][row]) / (2.0 * steps[axis]);
            }
        }
        jacobian
    }

    fn covariance_from_unweighted_jacobian(jacobian: &[f64]) -> [f64; 36] {
        let mut normal = [[0.0_f64; 6]; 6];
        for row in jacobian.chunks_exact(6) {
            for i in 0..6 {
                for j in 0..6 {
                    normal[i][j] += row[i] * row[j];
                }
            }
        }
        invert_normal(&normal)
    }

    fn max_relative_difference(actual: &[f64], expected: &[f64]) -> f64 {
        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs() / expected.abs().max(1e-30))
            .fold(0.0, f64::max)
    }

    fn linear_fixture(correlation: f64) -> LinearFixture {
        let template = synthetic_observations(None);
        let mut base = vec![[f64::NAN; 6]; NUM_OBS];
        let mut covariance = vec![f64::NAN; NUM_OBS * 36];
        let mut design = vec![[0.0_f64; 6]; 2 * NUM_OBS];
        let mut seed = 7_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * 600.0
        };
        let mut normal_rows = Vec::with_capacity(NUM_OBS);
        let mut whitened_design = Vec::with_capacity(2 * NUM_OBS);
        for row in 0..NUM_OBS {
            base[row][1] = 10.0 + 0.01 * row as f64;
            base[row][2] = 0.0;
            let sigma_lon = SIGMA_DEG * (0.75 + 0.035 * row as f64);
            let sigma_lat = SIGMA_DEG * (1.20 - 0.025 * row as f64);
            let cov = correlation * sigma_lon * sigma_lat;
            covariance[row * 36 + 7] = sigma_lon * sigma_lon;
            covariance[row * 36 + 8] = cov;
            covariance[row * 36 + 13] = cov;
            covariance[row * 36 + 14] = sigma_lat * sigma_lat;
            design[2 * row] = std::array::from_fn(|_| next());
            design[2 * row + 1] = std::array::from_fn(|_| next());

            let determinant = sigma_lon * sigma_lon * sigma_lat * sigma_lat - cov * cov;
            let inverse = [
                sigma_lat * sigma_lat / determinant,
                -cov / determinant,
                -cov / determinant,
                sigma_lon * sigma_lon / determinant,
            ];
            let mut row_normal = [[0.0_f64; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    row_normal[i][j] = design[2 * row][i]
                        * (inverse[0] * design[2 * row][j] + inverse[1] * design[2 * row + 1][j])
                        + design[2 * row + 1][i]
                            * (inverse[2] * design[2 * row][j]
                                + inverse[3] * design[2 * row + 1][j]);
                }
            }
            normal_rows.push(row_normal);
            if correlation == 0.0 {
                whitened_design.push(design[2 * row].map(|value| value / sigma_lon));
                whitened_design.push(design[2 * row + 1].map(|value| value / sigma_lat));
            }
        }
        let observed = CoordinateBatch::new(
            crate::CoordinateValues::Spherical(base.clone()),
            template.frame,
            template.origins,
            template.times,
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
        .unwrap();
        LinearFixture {
            observed,
            predictor: LinearPredictor {
                base,
                design,
                truth: TRUTH_STATE,
                batches: std::cell::RefCell::new(Vec::new()),
            },
            normal_rows,
            whitened_design,
        }
    }

    fn linear_problem<'a>(
        fixture: &'a LinearFixture,
        config: &WhitenedFitConfig,
    ) -> WhitenedProblem<'a> {
        WhitenedProblem::new(
            &fixture.predictor,
            &orbit(TRUTH_STATE),
            &fixture.observed,
            &observers(),
            config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap()
    }

    #[test]
    fn linear_predictor_recovers_exact_wls_covariance_and_unit_quadratic_probe() {
        let fixture = linear_fixture(0.0);
        let selected = [true; NUM_OBS];
        let normal = linear_normal_matrix(&fixture, &selected);
        let expected = invert_normal(&normal);
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Central,
            ..WhitenedFitConfig::default()
        };
        let fit = fit_orbit_whitened_with(
            &fixture.predictor,
            &perturbed_start(),
            &fixture.observed,
            &observers(),
            &[false; NUM_OBS],
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(fit.converged, "status {}", fit.status_code);
        assert!(fit.warnings.is_empty(), "{:?}", fit.warnings);
        assert!(max_abs_diff(&fit.state, &TRUTH_STATE) < 1e-11);
        assert!(
            max_relative_difference(&fit.covariance, &expected) < 2e-6,
            "{}",
            max_relative_difference(&fit.covariance, &expected)
        );
        assert_normal_times_covariance_is_identity(&normal, &fit.covariance, 2e-6);

        let problem = linear_problem(&fixture, &config);
        let residuals = problem.residuals_one(&TRUTH_STATE).unwrap();
        let delta = weak_direction_delta_cost(
            &problem,
            &TRUTH_STATE,
            &expected,
            robust_cost(&residuals, LossType::Linear, config.f_scale),
            &config,
        )
        .unwrap()
        .unwrap();
        assert!((delta - 1.0).abs() < 1e-6, "{delta}");
    }

    #[test]
    fn linear_predictor_preserves_probe_warning_branches_and_analytic_fallback() {
        let fixture = linear_fixture(0.0);
        let expected = expected_linear_covariance(&fixture, &[true; NUM_OBS]);
        let fabricated = expected.map(|value| value / 100.0);
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Central,
            ..WhitenedFitConfig::default()
        };
        let problem = linear_problem(&fixture, &config);
        let residuals = problem.residuals_one(&TRUTH_STATE).unwrap();
        let cost = robust_cost(&residuals, LossType::Linear, config.f_scale);

        for (method, message) in [
            (JacobianMethod::TwoPoint, "noise floor"),
            (JacobianMethod::Central, "covariance may be unreliable"),
        ] {
            let mut warnings = Vec::new();
            let returned = validated_covariance(
                &problem,
                fabricated,
                method,
                &TRUTH_STATE,
                &residuals,
                cost,
                &config,
                &mut warnings,
            )
            .unwrap();
            assert_eq!(returned, fabricated);
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert!(warnings[0].contains(message), "{:?}", warnings);
        }

        let mut warnings = Vec::new();
        let recovered = validated_covariance(
            &problem,
            fabricated,
            JacobianMethod::Analytic,
            &TRUTH_STATE,
            &residuals,
            cost,
            &config,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("central-difference Jacobian"));
        assert!(max_relative_difference(&recovered, &expected) < 2e-6);
    }

    #[test]
    fn linear_predictor_huber_bounds_influence_reports_exact_weights_and_robust_covariance() {
        const OUTLIER: usize = 3;
        const OUTLIER_SIGMAS: f64 = 20.0;
        let mut fixture = linear_fixture(0.0);
        let sigma = fixture
            .observed
            .covariance
            .as_ref()
            .unwrap()
            .values_row_major[OUTLIER * 36 + 14]
            .sqrt();
        if let crate::CoordinateValues::Spherical(values) = &mut fixture.observed.values {
            values[OUTLIER][2] += OUTLIER_SIGMAS * sigma;
        }
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Central,
            loss: LossType::Huber,
            f_scale: 1.345,
            ..WhitenedFitConfig::default()
        };
        let linear = fit_orbit_whitened_with(
            &fixture.predictor,
            &orbit(TRUTH_STATE),
            &fixture.observed,
            &observers(),
            &[false; NUM_OBS],
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Central,
                validate_covariance: false,
                ..WhitenedFitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let fit = fit_orbit_whitened_with(
            &fixture.predictor,
            &orbit(TRUTH_STATE),
            &fixture.observed,
            &observers(),
            &[false; NUM_OBS],
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(fit.converged, "status {}", fit.status_code);
        assert!(fit.warnings.is_empty(), "{:?}", fit.warnings);
        assert!(fit.weights[OUTLIER] < 0.2, "{:?}", fit.weights);
        assert!(fit.evaluation.outlier.iter().all(|flag| !flag));

        let clean_normal = linear_normal_matrix(&fixture, &[true; NUM_OBS]);
        let shift = |state: &[f64; 6]| {
            let delta: [f64; 6] = std::array::from_fn(|axis| state[axis] - TRUTH_STATE[axis]);
            let quadratic: f64 = (0..6)
                .flat_map(|i| (0..6).map(move |j| delta[i] * clean_normal[i][j] * delta[j]))
                .sum();
            quadratic.max(0.0).sqrt()
        };
        let linear_shift = shift(&linear.state);
        let huber_shift = shift(&fit.state);
        assert!(linear_shift > 1.0, "linear shift = {linear_shift}");
        assert!(
            huber_shift < 0.3 * linear_shift,
            "Huber shift {huber_shift} was not bounded relative to {linear_shift}"
        );

        let tail_component = 2 * OUTLIER + 1;
        let mut normal = [[0.0_f64; 6]; 6];
        for (component, row) in fixture.whitened_design.iter().enumerate() {
            if component == tail_component {
                continue;
            }
            for i in 0..6 {
                for j in 0..6 {
                    normal[i][j] += row[i] * row[j];
                }
            }
        }
        let expected = invert_normal(&normal);
        assert!(max_relative_difference(&fit.covariance, &expected) < 2e-5);

        let problem = linear_problem(&fixture, &config);
        let residuals = problem.residuals_one(&fit.state).unwrap();
        assert!(residuals[tail_component].abs() > config.f_scale);

        // Independent Huber first-order condition: J^T psi(r) ~= 0, with
        // psi clipped directly rather than reusing the production IRLS weights.
        let robust_score_ratio = |state: &[f64; 6]| {
            let state_residuals = problem.residuals_one(state).unwrap();
            let jacobian = problem
                .jacobian(JacobianMethod::Central, state, &state_residuals)
                .unwrap();
            let mut robust_score = [0.0_f64; 6];
            let mut absolute_score = [0.0_f64; 6];
            for (component, &residual) in state_residuals.iter().enumerate() {
                let psi = residual.clamp(-config.f_scale, config.f_scale);
                for axis in 0..6 {
                    let term = jacobian[component * 6 + axis] * psi;
                    robust_score[axis] += term;
                    absolute_score[axis] += term.abs();
                }
            }
            (0..6)
                .map(|axis| robust_score[axis].abs() / absolute_score[axis].max(1.0))
                .fold(0.0_f64, f64::max)
        };
        let huber_score_ratio = robust_score_ratio(&fit.state);
        let linear_score_ratio = robust_score_ratio(&linear.state);
        assert!(huber_score_ratio < 2e-6, "{huber_score_ratio}");
        assert!(
            linear_score_ratio > 100.0 * huber_score_ratio,
            "Huber score {huber_score_ratio} was not stationary relative to linear {linear_score_ratio}"
        );

        for row in 0..NUM_OBS {
            let expected_component_weight = |residual: f64| {
                if residual.abs() <= config.f_scale {
                    1.0
                } else {
                    config.f_scale / residual.abs()
                }
            };
            let expected = expected_component_weight(residuals[2 * row])
                .min(expected_component_weight(residuals[2 * row + 1]));
            assert!(
                (fit.weights[row] - expected).abs() < 1e-12,
                "weight[{row}]={} expected {expected}",
                fit.weights[row]
            );
        }
        let delta = weak_direction_delta_cost(
            &problem,
            &fit.state,
            &fit.covariance,
            robust_cost(&residuals, config.loss, config.f_scale),
            &config,
        )
        .unwrap()
        .unwrap();
        assert!((delta - 1.0).abs() < 1e-3, "{delta}");
    }

    #[test]
    fn linear_predictor_central_step_covariance_is_stable_against_closed_form() {
        let fixture = linear_fixture(0.4);
        let expected = expected_linear_covariance(&fixture, &[true; NUM_OBS]);
        let problem = linear_problem(
            &fixture,
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Central,
                ..WhitenedFitConfig::default()
            },
        );
        let production_jacobian = problem
            .jacobian(JacobianMethod::Central, &TRUTH_STATE, &[])
            .unwrap();
        let production_covariance = covariance_from_unweighted_jacobian(&production_jacobian);
        assert!(max_relative_difference(&production_covariance, &expected) < 2e-6);

        for relative_step in [0.5e-6, 1.0e-6, 2.0e-6] {
            let jacobian =
                central_jacobian_with_relative_step(&problem, &TRUTH_STATE, relative_step);
            if relative_step == 1.0e-6 {
                assert!(max_abs_diff(&jacobian, &production_jacobian) < 1e-9);
            }
            let covariance = covariance_from_unweighted_jacobian(&jacobian);
            assert!(
                max_relative_difference(&covariance, &expected) < 2e-6,
                "relative step {relative_step}: {}",
                max_relative_difference(&covariance, &expected)
            );
        }
    }

    fn add_linear_observation_offset(
        observed: &mut CoordinateBatch,
        row: usize,
        lon: f64,
        lat: f64,
    ) {
        if let crate::CoordinateValues::Spherical(values) = &mut observed.values {
            values[row][1] += lon;
            values[row][2] += lat;
        } else {
            panic!("linear fixture is spherical");
        }
    }

    fn cmc_linear_config(
        jacobian: JacobianMethod,
        validate_covariance: bool,
        max_iterations: usize,
    ) -> Cmc2003FitConfig {
        Cmc2003FitConfig {
            max_iterations,
            fit: WhitenedFitConfig {
                jacobian,
                max_nfev: Some(1),
                validate_covariance,
                ..WhitenedFitConfig::default()
            },
            ..Cmc2003FitConfig::default()
        }
    }

    #[test]
    fn linear_predictor_cmc_finalization_uses_full_non_keplerian_sensitivities() {
        for jacobian in [
            JacobianMethod::Analytic,
            JacobianMethod::Central,
            JacobianMethod::TwoPoint,
        ] {
            for validate_covariance in [false, true] {
                let fixture = linear_fixture(0.4);
                let expected = expected_linear_covariance(&fixture, &[true; NUM_OBS]);
                let output = cmc2003_fit_with(
                    &fixture.predictor,
                    &orbit(TRUTH_STATE),
                    &fixture.observed,
                    &observers(),
                    &cmc_linear_config(jacobian, validate_covariance, 3),
                    &NoopProvider,
                    &ZeroTranslationProvider,
                )
                .unwrap();
                assert_eq!(output.n_iterations, 1);
                assert_eq!(output.n_rejected, 0);
                assert_eq!(output.n_recovered, 0);
                assert!(output.flags.is_empty(), "{:?}", output.flags);
                assert!(output.fit.warnings.is_empty(), "{:?}", output.fit.warnings);
                assert!(max_relative_difference(&output.fit.covariance, &expected) < 2e-6);
                assert_eq!(output.fit.iterations, 1); // exact solver budget
                let batches = fixture.predictor.batches.borrow();
                assert_eq!(batches.as_slice(), &[1, 12, 1, 2], "{jacobian:?}");
            }
        }
    }

    fn highest_leverage_offset(fixture: &LinearFixture, raw_chi2: f64) -> (usize, f64, f64, f64) {
        let covariance = expected_linear_covariance(fixture, &[true; NUM_OBS]);
        let covariance = covariance_matrix(&covariance);
        let covariances = &fixture
            .observed
            .covariance
            .as_ref()
            .unwrap()
            .values_row_major;
        let mut best = (f64::NEG_INFINITY, 0_usize, [0.0_f64; 2], [0.0_f64; 3]);
        for row in 0..NUM_OBS {
            let variance_lon = covariances[row * 36 + 7];
            let covariance_lonlat = covariances[row * 36 + 8];
            let variance_lat = covariances[row * 36 + 14];
            let l00 = variance_lon.sqrt();
            let l10 = covariance_lonlat / l00;
            let l11 = (variance_lat - l10 * l10).sqrt();
            let a0 = fixture.predictor.design[2 * row];
            let a1 = fixture.predictor.design[2 * row + 1];
            let b0 = a0.map(|value| value / l00);
            let b1 = std::array::from_fn(|axis| (a1[axis] - l10 * b0[axis]) / l11);
            let quadratic = |left: &[f64; 6], right: &[f64; 6]| {
                let mut value = 0.0;
                for i in 0..6 {
                    for j in 0..6 {
                        value += left[i] * covariance[i][j] * right[j];
                    }
                }
                value
            };
            let m00 = quadratic(&b0, &b0);
            let m01 = quadratic(&b0, &b1);
            let m11 = quadratic(&b1, &b1);
            let discriminant = ((m00 - m11) * (m00 - m11) + 4.0 * m01 * m01).sqrt();
            let eigenvalue = 0.5 * (m00 + m11 + discriminant);
            let mut direction = if m01.abs() > 1e-30 {
                [m01, eigenvalue - m00]
            } else if m00 >= m11 {
                [1.0, 0.0]
            } else {
                [0.0, 1.0]
            };
            let norm = direction[0].hypot(direction[1]);
            direction[0] /= norm;
            direction[1] /= norm;
            if eigenvalue > best.0 {
                best = (eigenvalue, row, direction, [l00, l10, l11]);
            }
        }
        let scale = raw_chi2.sqrt();
        let whitened = [scale * best.2[0], scale * best.2[1]];
        let lon = best.3[0] * whitened[0];
        let lat = best.3[1] * whitened[0] + best.3[2] * whitened[1];
        (best.1, best.0, lon, lat)
    }

    #[test]
    fn linear_predictor_cmc_near_threshold_recovery_and_final_mask_covariance() {
        for jacobian in [
            JacobianMethod::Analytic,
            JacobianMethod::Central,
            JacobianMethod::TwoPoint,
        ] {
            for (raw_chi2, max_iterations) in [(6.0, 1_usize), (10.0, 2_usize)] {
                let mut fixture = linear_fixture(0.4);
                let (row, leverage, lon, lat) = highest_leverage_offset(&fixture, raw_chi2);
                assert!(raw_chi2 / (1.0 - leverage) > 9.0, "{leverage}");
                assert!(raw_chi2 * (1.0 - leverage) < 7.0, "{leverage}");
                add_linear_observation_offset(&mut fixture.observed, row, lon, lat);
                let output = cmc2003_fit_with(
                    &fixture.predictor,
                    &orbit(TRUTH_STATE),
                    &fixture.observed,
                    &observers(),
                    &cmc_linear_config(jacobian, false, max_iterations),
                    &NoopProvider,
                    &ZeroTranslationProvider,
                )
                .unwrap();
                let expected_mask: Vec<bool> = (0..NUM_OBS)
                    .map(|index| index == row && max_iterations == 1)
                    .collect();
                assert_eq!(output.n_recovered, max_iterations - 1);
                assert_eq!(output.fit.evaluation.outlier, expected_mask);
                assert_eq!(output.flags, vec!["max_iterations".to_string()]);
                assert!(output.fit.warnings.is_empty(), "{:?}", output.fit.warnings);
                let selected: Vec<bool> = expected_mask.iter().map(|flag| !flag).collect();
                let expected = expected_linear_covariance(&fixture, &selected);
                assert!(max_relative_difference(&output.fit.covariance, &expected) < 2e-6);
                // Every decision pass also spends one two-state mandatory
                // covariance probe; the final selected-mask refit retains
                // the solver/Jacobian/evaluation three-batch finalization.
                assert_eq!(output.fit.iterations, 1);
                assert_eq!(
                    fixture.predictor.batches.borrow().len(),
                    4 * (max_iterations + 1),
                    "{jacobian:?} {raw_chi2}"
                );
            }
        }
    }

    #[test]
    fn linear_predictor_cmc_rejects_outlier_and_recovers_known_state_and_covariance() {
        const OUTLIER: usize = 6;
        let mut fixture = linear_fixture(0.4);
        let sigma = fixture
            .observed
            .covariance
            .as_ref()
            .unwrap()
            .values_row_major[OUTLIER * 36 + 14]
            .sqrt();
        add_linear_observation_offset(&mut fixture.observed, OUTLIER, 0.0, 40.0 * sigma);
        let output = cmc2003_fit_with(
            &fixture.predictor,
            &perturbed_start(),
            &fixture.observed,
            &observers(),
            &Cmc2003FitConfig {
                fit: WhitenedFitConfig {
                    jacobian: JacobianMethod::Central,
                    ..WhitenedFitConfig::default()
                },
                ..Cmc2003FitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(output.fit.converged, "status {}", output.fit.status_code);
        assert_eq!(output.n_rejected, 1);
        assert!(output.fit.evaluation.outlier[OUTLIER]);
        assert!(output.flags.is_empty(), "{:?}", output.flags);
        assert!(output.fit.warnings.is_empty(), "{:?}", output.fit.warnings);
        assert!(max_abs_diff(&output.fit.state, &TRUTH_STATE) < 1e-11);
        let selected: Vec<bool> = (0..NUM_OBS).map(|row| row != OUTLIER).collect();
        let expected = expected_linear_covariance(&fixture, &selected);
        assert!(max_relative_difference(&output.fit.covariance, &expected) < 2e-6);
    }

    #[derive(Clone, Copy, PartialEq)]
    enum PredictorMode {
        Normal,
        /// Every prediction after the first fails in-band.
        RejectTrials,
        /// Every prediction after the first is finite but far worse.
        WorseTrials,
        /// Any multi-state batch (a finite-difference Jacobian) is NaN.
        NanBatches,
    }

    struct CountingPredictor<'a> {
        inner: &'a dyn SphericalPredictor,
        batches: std::cell::RefCell<Vec<usize>>,
        mode: PredictorMode,
    }

    impl SphericalPredictor for CountingPredictor<'_> {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            self.batches.borrow_mut().push(states.len());
            let later = self.batches.borrow().len() > 1;
            match self.mode {
                PredictorMode::RejectTrials if later => {
                    return Ok(Err("rejected trial prediction".to_string()));
                }
                PredictorMode::NanBatches if states.len() > 1 => {
                    return Ok(Ok(vec![f64::NAN; states.len() * observers.len() * 6]));
                }
                _ => {}
            }
            let mut predicted = self.inner.predict_spherical(states, geometry, observers)?;
            if self.mode == PredictorMode::WorseTrials && later {
                if let Ok(values) = &mut predicted {
                    for row in values.chunks_exact_mut(6) {
                        row[1] += 0.5;
                    }
                }
            }
            Ok(predicted)
        }
    }

    fn solve_counted(config: &WhitenedFitConfig, reject_trials: bool) -> (Solution, Vec<usize>) {
        solve_counted_mode(
            config,
            if reject_trials {
                PredictorMode::RejectTrials
            } else {
                PredictorMode::Normal
            },
        )
    }

    fn solve_counted_mode(
        config: &WhitenedFitConfig,
        mode: PredictorMode,
    ) -> (Solution, Vec<usize>) {
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let predictor = CountingPredictor {
            inner: &inner,
            batches: std::cell::RefCell::new(Vec::new()),
            mode,
        };
        let problem = WhitenedProblem::new(
            &predictor,
            &perturbed_start(),
            &synthetic_observations(None),
            &observers(),
            config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let solution = solve(&problem, problem.geometry.state, config).unwrap();
        (solution, predictor.batches.into_inner())
    }

    #[test]
    fn evaluation_budget_bounds_every_residual_evaluation() {
        for (jacobian, columns) in [
            (JacobianMethod::Analytic, 0_usize),
            (JacobianMethod::Central, 12),
            (JacobianMethod::TwoPoint, 6),
        ] {
            for budget in [1, 2, 3, 7, 13, 25] {
                let config = WhitenedFitConfig {
                    jacobian,
                    max_nfev: Some(budget),
                    xtol: 0.0,
                    ftol: 0.0,
                    gtol: 0.0,
                    ..WhitenedFitConfig::default()
                };
                let (solution, batches) = solve_counted(&config, false);
                assert_eq!(solution.nfev, batches.iter().sum::<usize>());
                assert!(
                    solution.nfev <= budget,
                    "{jacobian:?} {budget}: {batches:?}"
                );
                assert_eq!(
                    solution.status, STATUS_LIMIT,
                    "{jacobian:?} {budget}: {batches:?}"
                );
                if budget == 1 {
                    assert_eq!(batches, vec![1]);
                    assert_eq!(solution.optimizer_iterations, 0);
                } else if columns > 0 && budget > columns {
                    assert!(
                        batches.contains(&columns),
                        "{jacobian:?} {budget}: {batches:?}"
                    );
                } else if columns > 0 {
                    assert_eq!(batches, vec![1], "{jacobian:?} {budget}: {batches:?}");
                }
            }
        }
    }

    #[test]
    fn propagator_jacobian_phases_share_one_solver_budget() {
        let output = fit_orbit_whitened_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &synthetic_observations(None),
            &observers(),
            &[false; NUM_OBS],
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Propagator,
                max_nfev: Some(1),
                max_iterations: 1,
                validate_covariance: false,
                ..WhitenedFitConfig::default()
            },
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(output.iterations, 1);
        assert_eq!(output.optimizer_iterations, 0);
        assert_eq!(output.status_code, STATUS_LIMIT);
        assert!(!output.converged);
    }

    #[test]
    fn iteration_budget_is_independent_of_evaluation_budget() {
        for (jacobian, expected_nfev) in [
            (JacobianMethod::Analytic, 2),
            (JacobianMethod::Central, 14),
            (JacobianMethod::TwoPoint, 8),
        ] {
            for max_nfev in [None, Some(1000)] {
                let config = WhitenedFitConfig {
                    jacobian,
                    max_iterations: 1,
                    max_nfev,
                    xtol: 0.0,
                    ftol: 0.0,
                    gtol: 0.0,
                    ..WhitenedFitConfig::default()
                };
                let (solution, batches) = solve_counted(&config, false);
                assert_eq!(solution.nfev, expected_nfev, "{jacobian:?}: {batches:?}");
                assert_eq!(solution.nfev, batches.iter().sum::<usize>());
                assert_eq!(solution.status, 0);
            }
        }
    }

    #[test]
    fn evaluation_budget_stops_rejected_trials_without_reporting_convergence() {
        // 25 is the initial evaluation plus all 24 damping attempts.
        for budget in [4, 25] {
            let config = WhitenedFitConfig {
                max_nfev: Some(budget),
                // Even very loose xtol must not turn budget exhaustion into success.
                xtol: 1.0,
                jacobian: JacobianMethod::Analytic,
                ..WhitenedFitConfig::default()
            };
            let (solution, batches) = solve_counted(&config, true);
            assert_eq!(batches, vec![1; budget]);
            assert_eq!(solution.nfev, budget);
            assert_eq!(solution.status, 0);
        }
    }

    #[test]
    fn convergence_on_last_permitted_evaluation_is_preserved() {
        for (jacobian, budget) in [
            (JacobianMethod::Analytic, 2),
            (JacobianMethod::Central, 14),
            (JacobianMethod::TwoPoint, 8),
        ] {
            let config = WhitenedFitConfig {
                jacobian,
                max_nfev: Some(budget),
                xtol: 1.0,
                ..WhitenedFitConfig::default()
            };
            let (solution, batches) = solve_counted(&config, false);
            assert_eq!(solution.nfev, budget);
            assert_eq!(solution.nfev, batches.iter().sum::<usize>());
            assert_eq!(solution.status, 3);
        }
    }

    #[test]
    fn fit_rejects_fewer_than_three_distinct_epochs_before_prediction() {
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let predictor = CountingPredictor {
            inner: &inner,
            batches: std::cell::RefCell::new(Vec::new()),
            mode: PredictorMode::Normal,
        };
        let mut observed = synthetic_observations(None);
        let epochs = observed.times.as_ref().unwrap().epochs.clone();
        let times = observed.times.as_mut().unwrap();
        for (row, epoch) in times.epochs.iter_mut().enumerate() {
            *epoch = epochs[row % 2];
        }
        let error = WhitenedProblem::new(
            &predictor,
            &perturbed_start(),
            &observed,
            &observers(),
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .err()
        .expect("two-epoch fit must be refused");
        assert!(error.to_string().contains("2 distinct observation epochs"));
        assert!(predictor.batches.borrow().is_empty());

        if let crate::CoordinateValues::Spherical(values) = &mut observed.values {
            values[0][1] = f64::NAN;
        }
        let error = WhitenedProblem::new(
            &predictor,
            &perturbed_start(),
            &observed,
            &observers(),
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .err()
        .expect("non-finite observation must be refused");
        assert!(error.to_string().contains("longitude"));
        assert!(predictor.batches.borrow().is_empty());

        let error = fit_orbit_whitened_with(
            &predictor,
            &perturbed_start(),
            &synthetic_observations(None),
            &observers(),
            &[false; NUM_OBS],
            &WhitenedFitConfig {
                max_iterations: 0,
                ..WhitenedFitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap_err();
        assert!(error.to_string().contains("max_iterations"));
        assert!(predictor.batches.borrow().is_empty());
    }

    #[test]
    fn iterative_fit_routes_arc_lengths_to_dynamical_time() {
        struct RoutingPredictor<'a> {
            inner: &'a dyn SphericalPredictor,
            civil_calls: std::cell::Cell<usize>,
            dynamical_calls: std::cell::Cell<usize>,
        }
        impl SphericalPredictor for RoutingPredictor<'_> {
            fn predict_spherical(
                &self,
                states: &[[f64; 6]],
                geometry: &OrbitGeometry,
                observers: &ObserverBatch,
            ) -> PropagationResultValue<Result<Vec<f64>, String>> {
                self.inner.predict_spherical(states, geometry, observers)
            }
            fn civil_times_mjd(
                &self,
                observed: &CoordinateBatch,
            ) -> PropagationResultValue<Vec<f64>> {
                self.civil_calls.set(self.civil_calls.get() + 1);
                self.inner.civil_times_mjd(observed)
            }
            fn dynamical_times_mjd(
                &self,
                observed: &CoordinateBatch,
            ) -> PropagationResultValue<Vec<f64>> {
                self.dynamical_calls.set(self.dynamical_calls.get() + 1);
                self.inner.dynamical_times_mjd(observed)
            }
        }

        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let predictor = RoutingPredictor {
            inner: &inner,
            civil_calls: std::cell::Cell::new(0),
            dynamical_calls: std::cell::Cell::new(0),
        };
        let output = super::iterative_fit_with(
            &predictor,
            &perturbed_start(),
            &synthetic_observations(None),
            &observers(),
            &IterativeFitConfig {
                rchi2_threshold: f64::MAX,
                min_arc_length: 0.0,
                contamination_percentage: 0.0,
                fit: WhitenedFitConfig {
                    jacobian: JacobianMethod::Central,
                    validate_covariance: false,
                    ..WhitenedFitConfig::default()
                },
                ..IterativeFitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(output.passes, 1);
        assert_eq!(predictor.civil_calls.get(), 0);
        assert_eq!(predictor.dynamical_calls.get(), 2);
    }

    #[test]
    fn remaining_solver_config_never_refreshes_either_budget() {
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Analytic,
            max_iterations: 7,
            max_nfev: Some(20),
            ..WhitenedFitConfig::default()
        };
        let remaining = remaining_fit_config(&config, 3, 4, JacobianMethod::Propagator).unwrap();
        assert_eq!(remaining.jacobian, JacobianMethod::Propagator);
        assert_eq!(remaining.max_iterations, 4);
        assert_eq!(remaining.max_nfev, Some(16));
        assert!(remaining_fit_config(&config, 7, 4, JacobianMethod::Central).is_none());
        assert!(remaining_fit_config(&config, 3, 20, JacobianMethod::Central).is_none());
    }

    #[test]
    fn propagator_phases_report_and_share_exact_combined_budgets() {
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let observed = synthetic_observations(None);
        let seed_config = WhitenedFitConfig {
            jacobian: JacobianMethod::Analytic,
            ..WhitenedFitConfig::default()
        };
        let problem = WhitenedProblem::new(
            &predictor,
            &perturbed_start(),
            &observed,
            &observers(),
            &seed_config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let seed = solve(&problem, problem.geometry.state, &seed_config).unwrap();
        assert!(seed.optimizer_iterations > 0);
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Propagator,
            max_iterations: seed.optimizer_iterations + 2,
            // Leave evaluation count unbounded so the iteration assertion
            // independently detects a fresh second-phase budget.
            max_nfev: None,
            ..WhitenedFitConfig::default()
        };
        let approximate = solve(
            &problem,
            problem.geometry.state,
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Analytic,
                ..config
            },
        )
        .unwrap();
        assert_eq!(approximate.optimizer_iterations, seed.optimizer_iterations);
        assert_eq!(approximate.nfev, seed.nfev);
        let exact_config = WhitenedFitConfig {
            max_iterations: config.max_iterations - approximate.optimizer_iterations,
            max_nfev: config.max_nfev.map(|limit| limit - approximate.nfev),
            ..config
        };
        let mut expected_warnings = Vec::new();
        let (exact, expected_method) = solve_with_nonfinite_fallback(
            &problem,
            approximate.state,
            &exact_config,
            &mut expected_warnings,
        )
        .unwrap();
        let mut warnings = Vec::new();
        let (combined, method) =
            solve_with_fallback(&problem, problem.geometry.state, &config, &mut warnings).unwrap();
        assert_eq!(method, expected_method);
        assert_eq!(warnings, expected_warnings);
        assert_eq!(combined.nfev, approximate.nfev + exact.nfev);
        assert_eq!(
            combined.optimizer_iterations,
            approximate.optimizer_iterations + exact.optimizer_iterations
        );
        assert!(combined.optimizer_iterations <= config.max_iterations);

        // If the analytic seed consumes the entire shared iteration budget,
        // the exact phase must not run at all. This distinguishes the real
        // phase hand-off from a fresh exact-phase max_iterations allowance.
        let exhausted = WhitenedFitConfig {
            max_iterations: approximate.optimizer_iterations,
            max_nfev: None,
            ..config
        };
        let mut exhausted_warnings = Vec::new();
        let (exhausted_solution, exhausted_method) = solve_with_fallback(
            &problem,
            problem.geometry.state,
            &exhausted,
            &mut exhausted_warnings,
        )
        .unwrap();
        assert_eq!(exhausted_method, JacobianMethod::Propagator);
        assert_eq!(exhausted_solution.status, STATUS_LIMIT);
        assert_eq!(
            exhausted_solution.optimizer_iterations,
            approximate.optimizer_iterations
        );
    }

    #[test]
    fn nonfinite_fallback_reports_and_shares_combined_iteration_budget() {
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let observed = synthetic_observations(None);
        let base = WhitenedFitConfig {
            jacobian: JacobianMethod::Analytic,
            ..WhitenedFitConfig::default()
        };
        let mut problem = WhitenedProblem::new(
            &predictor,
            &perturbed_start(),
            &observed,
            &observers(),
            &base,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        problem.analytic_terms.as_mut().unwrap().cos_lats[0] = f64::NAN;
        let plain = solve(&problem, problem.geometry.state, &base).unwrap();
        assert_eq!(plain.status, STATUS_NON_FINITE);
        assert!(plain.optimizer_iterations > 0);
        let config = WhitenedFitConfig {
            max_iterations: plain.optimizer_iterations + 1,
            // Leave evaluation count unbounded so the fallback cannot hide
            // a fresh iteration budget behind a tighter nfev cap.
            max_nfev: None,
            ..base
        };
        let retry_config = WhitenedFitConfig {
            jacobian: JacobianMethod::Central,
            max_iterations: 1,
            max_nfev: None,
            ..config
        };
        let retry = solve(&problem, plain.state, &retry_config).unwrap();
        let mut warnings = Vec::new();
        let (combined, method) =
            solve_with_nonfinite_fallback(&problem, problem.geometry.state, &config, &mut warnings)
                .unwrap();
        assert_eq!(method, JacobianMethod::Central);
        assert_eq!(warnings.len(), 1);
        assert_eq!(combined.nfev, plain.nfev + retry.nfev);
        assert_eq!(
            combined.optimizer_iterations,
            plain.optimizer_iterations + retry.optimizer_iterations
        );
        assert!(combined.optimizer_iterations <= config.max_iterations);

        let nfev_exhausted = WhitenedFitConfig {
            max_iterations: base.max_iterations,
            // Central differences need twelve perturbations plus their
            // initial residual. Leave exactly twelve evaluations after the
            // analytic failure: the retry can be entered, but cannot spend a
            // fresh max_nfev allowance on a Jacobian.
            max_nfev: Some(plain.nfev + 12),
            ..base
        };
        let mut exhausted_warnings = Vec::new();
        let (exhausted, exhausted_method) = solve_with_nonfinite_fallback(
            &problem,
            problem.geometry.state,
            &nfev_exhausted,
            &mut exhausted_warnings,
        )
        .unwrap();
        assert_eq!(exhausted_method, JacobianMethod::Central);
        assert_eq!(exhausted.status, STATUS_LIMIT);
        assert!(exhausted.nfev <= nfev_exhausted.max_nfev.unwrap());
    }

    #[test]
    fn non_finite_analytic_jacobian_continues_with_central_differences() {
        // Analytic terms poisoned (as the two-body STM is on very long arcs):
        // the plain solver stops with status -2, the fit continues from that
        // iterate with central differences through the predictor and reaches
        // the same minimum as a central-difference fit.
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let observed = synthetic_observations(None);
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Analytic,
            ..WhitenedFitConfig::default()
        };
        let mut problem = WhitenedProblem::new(
            &predictor,
            &perturbed_start(),
            &observed,
            &observers(),
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        problem.analytic_terms.as_mut().unwrap().cos_lats[0] = f64::NAN;
        let plain = solve(&problem, problem.geometry.state, &config).unwrap();
        assert_eq!(plain.status, STATUS_NON_FINITE);

        let mut warnings = Vec::new();
        let (solution, method) =
            solve_with_fallback(&problem, problem.geometry.state, &config, &mut warnings).unwrap();
        assert_eq!(method, JacobianMethod::Central);
        assert!(solution.status > 0, "status {}", solution.status);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("central differences"));
        let central = fit(
            &observed,
            &[false; NUM_OBS],
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Central,
                ..WhitenedFitConfig::default()
            },
        );
        assert!(max_abs_diff(&solution.state, &central.state) < 1e-9);

        // Central differences are not retried, and neither are non-finite
        // residuals (a NaN observation): status -2 stands.
        let mut warnings = Vec::new();
        let (_, method) = solve_with_fallback(
            &problem,
            problem.geometry.state,
            &WhitenedFitConfig {
                jacobian: JacobianMethod::TwoPoint,
                ..config
            },
            &mut warnings,
        )
        .unwrap();
        assert_eq!(method, JacobianMethod::TwoPoint);
        assert!(warnings.is_empty());
        let mut nan_observed = synthetic_observations(None);
        if let crate::CoordinateValues::Spherical(values) = &mut nan_observed.values {
            values[2][1] = f64::NAN;
        }
        let error = WhitenedProblem::new(
            &predictor,
            &perturbed_start(),
            &nan_observed,
            &observers(),
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .err()
        .expect("non-finite input must fail before the predictor runs");
        assert!(error.to_string().contains("longitude must be finite"));
    }

    #[test]
    fn centralized_fit_and_refinement_validation_rejects_invalid_families() {
        let invalid_fit = [
            WhitenedFitConfig {
                f_scale: f64::NAN,
                ..WhitenedFitConfig::default()
            },
            WhitenedFitConfig {
                xtol: f64::INFINITY,
                ..WhitenedFitConfig::default()
            },
            WhitenedFitConfig {
                max_iterations: 0,
                ..WhitenedFitConfig::default()
            },
            WhitenedFitConfig {
                two_body: TwoBodyModelConfig {
                    propagation_tol: -1.0,
                    ..TwoBodyModelConfig::default()
                },
                ..WhitenedFitConfig::default()
            },
        ];
        for config in invalid_fit {
            assert!(config.validate().is_err(), "accepted {config:?}");
        }
        let invalid_iterative = [
            IterativeFitConfig {
                min_obs: 2,
                ..IterativeFitConfig::default()
            },
            IterativeFitConfig {
                min_arc_length: -1.0,
                ..IterativeFitConfig::default()
            },
            IterativeFitConfig {
                contamination_percentage: f64::INFINITY,
                ..IterativeFitConfig::default()
            },
            IterativeFitConfig {
                rchi2_threshold: f64::NAN,
                ..IterativeFitConfig::default()
            },
        ];
        for config in invalid_iterative {
            assert!(config.validate().is_err(), "accepted {config:?}");
        }
        let invalid_cmc = [
            Cmc2003FitConfig {
                max_iterations: 0,
                ..Cmc2003FitConfig::default()
            },
            Cmc2003FitConfig {
                chi2_reject: f64::NAN,
                ..Cmc2003FitConfig::default()
            },
            Cmc2003FitConfig {
                chi2_reject: 1.0,
                chi2_recover: 2.0,
                ..Cmc2003FitConfig::default()
            },
            Cmc2003FitConfig {
                max_rejected_fraction: 1.1,
                ..Cmc2003FitConfig::default()
            },
            Cmc2003FitConfig {
                apparition_gap_days: -1.0,
                ..Cmc2003FitConfig::default()
            },
        ];
        for config in invalid_cmc {
            assert!(config.validate().is_err(), "accepted {config:?}");
        }
        assert!(IterativeFitConfig::default().validate().is_ok());
        assert!(Cmc2003FitConfig::default().validate().is_ok());
        assert!(RefinementConfig::default().validate().is_ok());

        // Full OD validates refinement even when IOD cannot produce a seed;
        // no control-flow branch may defer configuration validation.
        let observed = synthetic_observations(None);
        let observer_rows = observers();
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let invalid_full = FullOdConfig {
            iod: super::super::od::IodConfig {
                min_obs: NUM_OBS + 1,
                ..corrected_iod_config(50.0)
            },
            refinement: RefinementConfig::Cmc2003(Cmc2003FitConfig {
                max_iterations: 0,
                ..Cmc2003FitConfig::default()
            }),
        };
        let error = full_od_with(
            &predictor,
            &predictor,
            &observed,
            &observer_rows,
            &invalid_full,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("cmc2003.max_iterations"),
            "{error}"
        );
    }

    #[test]
    fn evaluation_budget_must_be_positive_when_present() {
        assert!(WhitenedFitConfig::default().validate().is_ok());
        assert!(WhitenedFitConfig {
            max_nfev: Some(0),
            ..WhitenedFitConfig::default()
        }
        .validate()
        .unwrap_err()
        .contains("max_nfev"));
    }

    #[test]
    fn trust_region_step_is_gauss_newton_inside_and_on_the_boundary_outside() {
        let eigenvalues = [4.0, 1.0, 0.25, 0.09, 0.01, 0.0025];
        let q = [2.0, -1.0, 0.5, 0.3, -0.1, 0.05];
        let gauss_newton: [f64; 6] = std::array::from_fn(|i| -q[i] / eigenvalues[i]);
        let gn_norm = norm6(&gauss_newton);
        let (inside, alpha) = trust_region_step(&eigenvalues, &q, 2.0 * gn_norm, 0.0);
        assert_eq!(alpha, 0.0);
        assert!(norm6(&std::array::from_fn(|i| inside[i] - gauss_newton[i])) < 1e-12);

        for radius in [0.5 * gn_norm, 0.1 * gn_norm, 0.01 * gn_norm] {
            let (step, alpha) = trust_region_step(&eigenvalues, &q, radius, 0.0);
            assert!(alpha > 0.0, "{radius}");
            assert!((norm6(&step) - radius).abs() <= 1e-12 * radius, "{radius}");
            // The damped step is the model minimizer on the boundary:
            // (Lambda + alpha I) p = -q up to the final 1 % renormalization.
            for i in 0..6 {
                let implied = -q[i] / (eigenvalues[i] + alpha);
                assert!((step[i] / implied - 1.0).abs() < 0.02, "{radius} {i}");
            }
        }

        // Rank deficiency with a gradient along the null direction: the
        // Gauss-Newton step is unbounded, the trust-region step is not.
        let deficient = [4.0, 1.0, 0.25, 0.09, 0.01, 0.0];
        let (step, alpha) = trust_region_step(&deficient, &q, 1.0, 0.0);
        assert!(alpha > 0.0);
        assert!((norm6(&step) - 1.0).abs() <= 1e-12);
        assert!(step.iter().all(|v| v.is_finite()));

        let (zero, alpha) = trust_region_step(&eigenvalues, &[0.0; 6], 1.0, 0.0);
        assert_eq!(zero, [0.0; 6]);
        assert_eq!(alpha, 0.0);
    }

    #[test]
    fn non_finite_jacobian_never_reports_convergence() {
        // A NaN finite-difference Jacobian makes the gradient NaN; a
        // NaN-ignoring norm would read that as zero and declare gtol
        // convergence (the 2000 FE8 defect). It must end with status -2,
        // the iterate untouched.
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Central,
            ..WhitenedFitConfig::default()
        };
        let (solution, batches) = solve_counted_mode(&config, PredictorMode::NanBatches);
        assert_eq!(batches, vec![1, 12]);
        assert_eq!(solution.nfev, 13);
        assert_eq!(solution.status, -2);
        assert!(solution.status <= 0);
        assert_eq!(
            solution.state,
            OrbitGeometry::from_orbit(&perturbed_start()).unwrap().state
        );
        assert!(solution.residuals.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn rejected_finite_trials_shrink_the_region_to_the_xtol_resolution() {
        // Every trial is finite but worse: the trust region shrinks fourfold
        // per rejection until the step is below the xtol resolution, which
        // ends the fit as a numerical minimum (status 3) at the start state,
        // never as a stall and well inside the attempt limit.
        let (solution, batches) = solve_counted_mode(
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Analytic,
                ..WhitenedFitConfig::default()
            },
            PredictorMode::WorseTrials,
        );
        assert_eq!(solution.status, 3, "{batches:?}");
        assert_eq!(
            solution.state,
            OrbitGeometry::from_orbit(&perturbed_start()).unwrap().state
        );
        assert!(
            batches.len() > 2 && batches.len() < MAX_STEP_ATTEMPTS,
            "{batches:?}"
        );
        assert!(batches.iter().all(|&size| size == 1));
    }

    #[test]
    fn central_difference_fit_converges_within_the_default_limits() {
        // The encounter-guard configuration of the walk-forward: central
        // differences through the predictor with the default budgets must
        // converge, not run out (the 2026 AC4 defect).
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Central,
            ..WhitenedFitConfig::default()
        };
        let (solution, batches) = solve_counted(&config, false);
        assert!(
            solution.status > 0,
            "status {} {batches:?}",
            solution.status
        );
        let jacobians = batches.iter().filter(|&&size| size == 12).count();
        assert!(jacobians <= 20, "{batches:?}");
        let error = max_abs_diff(&solution.state[..3], &TRUTH_STATE[..3]);
        assert!(error < 5e-5, "{error}");
    }

    #[test]
    fn analytic_fit_recovers_truth_with_validated_covariance() {
        let observed = synthetic_observations(None);
        let output = fit(&observed, &[false; NUM_OBS], &WhitenedFitConfig::default());
        assert!(output.converged, "status {}", output.status_code);
        assert!(output.warnings.is_empty(), "{:?}", output.warnings);
        // Sub-sigma scatter: the solution sits within ~1e-6 AU of the truth
        // and the reduced chi2 is of order one.
        let position_error = max_abs_diff(&output.state[..3], &TRUTH_STATE[..3]);
        assert!(position_error < 5e-5, "{position_error}");
        assert!(output.evaluation.reduced_chi2 < 2.0);
        // The truth lies within the fit covariance (6-dof Mahalanobis).
        let cov = covariance_matrix(&output.covariance);
        let mut precision = [[0.0_f64; 6]; 6];
        let inverse = inverse_6x6(&cov).unwrap();
        for i in 0..6 {
            precision[i].copy_from_slice(&inverse[i * 6..(i + 1) * 6]);
        }
        let mut mahalanobis = 0.0;
        for i in 0..6 {
            for j in 0..6 {
                mahalanobis += (output.state[i] - TRUTH_STATE[i])
                    * precision[i][j]
                    * (output.state[j] - TRUTH_STATE[j]);
            }
        }
        assert!(mahalanobis < 40.0, "{mahalanobis}");
        assert!((output.cost - output.fit_chi2).abs() < 1e-12);
        assert_eq!(output.weights, vec![1.0; NUM_OBS]);
        assert_eq!(output.evaluation.num_obs, NUM_OBS);
        assert!(output.covariance.iter().all(|v| v.is_finite()));
        let (eigenvalues, _) = symmetric_eigen_6x6(&covariance_matrix(&output.covariance));
        assert!(eigenvalues.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn finite_difference_jacobians_reach_the_same_minimum() {
        let observed = synthetic_observations(None);
        let analytic = fit(&observed, &[false; NUM_OBS], &WhitenedFitConfig::default());
        for method in [JacobianMethod::Central, JacobianMethod::TwoPoint] {
            let config = WhitenedFitConfig {
                jacobian: method,
                ..WhitenedFitConfig::default()
            };
            let output = fit(&observed, &[false; NUM_OBS], &config);
            assert!(output.converged, "{method:?} status {}", output.status_code);
            let diff = max_abs_diff(&output.state, &analytic.state);
            // Central differences are exact to round-off; forward differences
            // carry an O(h) Jacobian error, so their fixed point sits a few
            // 1e-9 au from the true minimum (scipy's "2-point" does the same).
            let tolerance = match method {
                JacobianMethod::TwoPoint => 1e-8,
                _ => 1e-9,
            };
            assert!(diff < tolerance, "{method:?}: {diff}");
            let cov_diff = max_abs_diff(&output.covariance, &analytic.covariance);
            let cov_scale = analytic
                .covariance
                .iter()
                .fold(0.0_f64, |acc, v| acc.max(v.abs()));
            assert!(
                cov_diff < 1e-4 * cov_scale,
                "{method:?}: {cov_diff} vs {cov_scale}"
            );
        }
    }

    #[test]
    fn ignored_observations_are_outliers_with_zero_weight() {
        let observed = synthetic_observations(None);
        let mut ignore = vec![false; NUM_OBS];
        ignore[0] = true;
        ignore[NUM_OBS - 1] = true;
        let output = fit(&observed, &ignore, &WhitenedFitConfig::default());
        assert!(output.converged);
        assert_eq!(output.evaluation.num_obs, NUM_OBS - 2);
        assert_eq!(output.evaluation.outlier, ignore);
        assert_eq!(output.weights[0], 0.0);
        assert_eq!(output.weights[NUM_OBS - 1], 0.0);
        assert!(output.weights[1..NUM_OBS - 1].iter().all(|&w| w == 1.0));
        assert_eq!(output.residuals_whitened.len(), 2 * (NUM_OBS - 2));
        assert_eq!(output.evaluation.residuals.len(), NUM_OBS * 6);
    }

    #[test]
    fn huber_loss_downweights_a_gross_outlier() {
        let observed = synthetic_observations(Some(5));
        let linear = fit(&observed, &[false; NUM_OBS], &WhitenedFitConfig::default());
        let huber = fit(
            &observed,
            &[false; NUM_OBS],
            &WhitenedFitConfig {
                loss: LossType::Huber,
                ..WhitenedFitConfig::default()
            },
        );
        assert!(huber.converged, "status {}", huber.status_code);
        assert!(huber.weights[5] < 0.5, "{:?}", huber.weights);
        assert!(huber
            .weights
            .iter()
            .enumerate()
            .all(|(row, &w)| row == 5 || w > 0.5));
        // The robust solution is closer to the truth than the linear one.
        let linear_error = max_abs_diff(&linear.state[..3], &TRUTH_STATE[..3]);
        let huber_error = max_abs_diff(&huber.state[..3], &TRUTH_STATE[..3]);
        assert!(
            huber_error < linear_error,
            "{huber_error} vs {linear_error}"
        );
        // Reported chi2 stays the plain chi2 of the included observations.
        assert!(huber.fit_chi2 > huber.cost);
        assert!((huber.fit_chi2 - huber.evaluation.orbit_chi2).abs() < 1e-9);
    }

    #[test]
    fn fabricated_covariance_fails_the_probe() {
        // A covariance claiming 1e-10 AU precision disagrees with the chi2
        // surface: the analytic branch falls back to central differences and
        // reports it.
        let observed = synthetic_observations(None);
        let config = WhitenedFitConfig::default();
        let observers = observers();
        let keep = vec![true; NUM_OBS];
        let observed_fit = filter_coordinate_batch(&observed, &keep).unwrap();
        let observers_fit = filter_observer_batch(&observers, &keep).unwrap();
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let truth = orbit(TRUTH_STATE);
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let problem = WhitenedProblem::new(
            &predictor,
            &truth,
            &observed_fit,
            &observers_fit,
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let solution = solve(&problem, TRUTH_STATE, &config).unwrap();
        let mut fabricated = [0.0_f64; 36];
        for k in 0..6 {
            fabricated[k * 7] = 1e-20;
        }
        let mut warnings = Vec::new();
        let validated = validated_covariance(
            &problem,
            fabricated,
            JacobianMethod::Analytic,
            &solution.state,
            &solution.residuals,
            solution.cost,
            &config,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("Falling back to a central-difference Jacobian"));
        assert!(validated[0] > 1e-15, "fallback covariance was not adopted");
    }

    /// Fails with a backend error only on the probe: with the analytic
    /// Jacobian every solver call predicts one candidate, the probe is the
    /// only two-candidate batch. `in_band` makes it a rejected trial
    /// prediction instead of a backend error.
    struct FailsOnProbe<'a> {
        inner: &'a dyn SphericalPredictor,
        in_band: bool,
        probes: std::cell::Cell<usize>,
    }

    impl SphericalPredictor for FailsOnProbe<'_> {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            if states.len() == 2 {
                self.probes.set(self.probes.get() + 1);
                if self.in_band {
                    return Ok(Err("probe prediction rejected".to_string()));
                }
                return Err(PropagationError::Backend(
                    "backend failed at the probe displacement".to_string(),
                ));
            }
            self.inner.predict_spherical(states, geometry, observers)
        }
    }

    struct DistortsOrRejectsCmcProbe<'a> {
        inner: &'a dyn SphericalPredictor,
        reject_probe: Option<usize>,
        distort: bool,
        probes: std::cell::Cell<usize>,
    }

    impl SphericalPredictor for DistortsOrRejectsCmcProbe<'_> {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            let result = self.inner.predict_spherical(states, geometry, observers)?;
            if states.len() != 2 {
                return Ok(result);
            }
            let probe = self.probes.get() + 1;
            self.probes.set(probe);
            if self.reject_probe == Some(probe) {
                return Ok(Err(format!("deliberately rejected CMC probe {probe}")));
            }
            let mut result = match result {
                Ok(result) => result,
                Err(message) => return Ok(Err(message)),
            };
            if self.distort {
                for prediction in result.chunks_exact_mut(6) {
                    prediction[1] += 1.0;
                }
            }
            Ok(Ok(result))
        }
    }

    #[test]
    fn cmc_fails_closed_on_probe_mismatch_and_probe_rejection() {
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let observed = synthetic_observations(None);
        let run = |predictor: &dyn SphericalPredictor| {
            cmc2003_fit_with(
                predictor,
                &perturbed_start(),
                &observed,
                &observers(),
                &Cmc2003FitConfig {
                    fit: WhitenedFitConfig {
                        jacobian: JacobianMethod::Central,
                        validate_covariance: false,
                        ..WhitenedFitConfig::default()
                    },
                    ..Cmc2003FitConfig::default()
                },
                &NoopProvider,
                &ZeroTranslationProvider,
            )
        };

        let distorted = DistortsOrRejectsCmcProbe {
            inner: &inner,
            reject_probe: None,
            distort: true,
            probes: std::cell::Cell::new(0),
        };
        let mismatch = run(&distorted).expect_err("CMC must reject a mismatched chi2 surface");
        assert!(mismatch.to_string().contains("consistency check failed"));
        assert_eq!(distorted.probes.get(), 1);

        let rejected = DistortsOrRejectsCmcProbe {
            inner: &inner,
            reject_probe: Some(1),
            distort: false,
            probes: std::cell::Cell::new(0),
        };
        let failure = run(&rejected).expect_err("CMC must reject a failed probe prediction");
        assert!(failure.to_string().contains("probe failed"));
        assert_eq!(rejected.probes.get(), 1);
    }

    #[test]
    fn cmc_regates_the_final_selection_refit() {
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let predictor = DistortsOrRejectsCmcProbe {
            inner: &inner,
            reject_probe: Some(2),
            distort: false,
            probes: std::cell::Cell::new(0),
        };
        let failure = cmc2003_fit_with(
            &predictor,
            &perturbed_start(),
            &synthetic_observations(Some(3)),
            &observers(),
            &Cmc2003FitConfig {
                max_iterations: 1,
                fit: WhitenedFitConfig {
                    jacobian: JacobianMethod::Central,
                    validate_covariance: false,
                    ..WhitenedFitConfig::default()
                },
                ..Cmc2003FitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .expect_err("the post-selection fit requires the same covariance gate");
        assert!(failure.to_string().contains("probe failed"));
        assert_eq!(predictor.probes.get(), 2);
    }

    #[test]
    fn probe_backend_error_propagates_and_in_band_failure_warns() {
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let observed = synthetic_observations(None);
        let config = WhitenedFitConfig::default();
        let run = |predictor: &dyn SphericalPredictor| {
            fit_orbit_whitened_with(
                predictor,
                &perturbed_start(),
                &observed,
                &observers(),
                &[false; NUM_OBS],
                &config,
                &NoopProvider,
                &ZeroTranslationProvider,
            )
        };

        let failing = FailsOnProbe {
            inner: &inner,
            in_band: false,
            probes: std::cell::Cell::new(0),
        };
        match run(&failing) {
            Err(PropagationError::Backend(message)) => {
                assert!(message.contains("probe displacement"), "{message}")
            }
            other => panic!("expected the probe backend error, got {other:?}"),
        }
        assert_eq!(failing.probes.get(), 1);

        let rejected = FailsOnProbe {
            inner: &inner,
            in_band: true,
            probes: std::cell::Cell::new(0),
        };
        let output = run(&rejected).unwrap();
        assert!(output.converged);
        assert_eq!(output.warnings.len(), 1, "{:?}", output.warnings);
        assert!(output.warnings[0].contains("could not be validated"));
        assert!(output.warnings[0].contains("probe prediction rejected"));
        assert!(output.covariance.iter().all(|v| v.is_finite()));

        // The standalone validation entry point follows the same rule.
        let validation = validate_fit_covariance_with(
            &failing,
            &orbit(output.state),
            &observed,
            &observers(),
            &[false; NUM_OBS],
            output.covariance,
            JacobianMethod::Analytic,
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        );
        assert!(matches!(validation, Err(PropagationError::Backend(_))));
        let validation = validate_fit_covariance_with(
            &rejected,
            &orbit(output.state),
            &observed,
            &observers(),
            &[false; NUM_OBS],
            output.covariance,
            JacobianMethod::Analytic,
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(validation.delta_cost.is_nan());
        assert_eq!(validation.warnings.len(), 1);
    }

    #[test]
    fn worst_residual_selection_breaks_ties_deterministically() {
        // Equal probabilities: the larger residual norm wins; equal norms
        // too: the first candidate row wins. A NaN probability is an error.
        let probability = [0.5, 0.1, 0.1, 0.9, 0.1, 0.1];
        let mut residuals = vec![0.0; 6 * 6];
        residuals[6 + 1] = 1.0;
        residuals[2 * 6 + 1] = 2.0;
        residuals[4 * 6 + 1] = 2.0;
        residuals[5 * 6 + 1] = 2.0;
        assert_eq!(
            lowest_probability_row(&[0, 1, 2, 3, 4, 5], &probability, &residuals).unwrap(),
            2
        );
        assert_eq!(
            lowest_probability_row(&[5, 4, 2], &probability, &residuals).unwrap(),
            5
        );
        assert_eq!(
            lowest_probability_row(&[0, 1, 3], &probability, &residuals).unwrap(),
            1
        );
        let mut with_nan = probability;
        with_nan[4] = f64::NAN;
        assert!(lowest_probability_row(&[0, 4], &with_nan, &residuals).is_err());
        assert!(lowest_probability_row(&[], &probability, &residuals).is_err());
    }

    #[test]
    fn iterative_fit_rejects_the_worst_observation_until_converged() {
        let observed = synthetic_observations(Some(3));
        let config = IterativeFitConfig {
            rchi2_threshold: 2.0,
            min_obs: 6,
            min_arc_length: 1.0,
            contamination_percentage: 20.0,
            fit: WhitenedFitConfig::default(),
        };
        let output = iterative_fit_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers(),
            &config,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(output.passes, 2);
        let outlier = &output.fit.evaluation.outlier;
        assert!(outlier[3], "{outlier:?}");
        assert_eq!(outlier.iter().filter(|&&flag| flag).count(), 1);
        assert_eq!(output.fit.weights[3], 0.0);
        assert!(output.fit.evaluation.reduced_chi2 <= 2.0);
        assert!(output.fit.converged);
        let error = max_abs_diff(&output.fit.state[..3], &TRUTH_STATE[..3]);
        assert!(error < 5e-5, "{error}");
        assert_eq!(max_outliers(12, 6, 20.0), 2);
        assert_eq!(max_outliers(7, 6, 50.0), 1);
        assert_eq!(max_outliers(5, 6, 50.0), 0);
    }

    #[test]
    fn iterative_fit_stops_at_the_contamination_budget() {
        let observed = synthetic_observations(Some(3));
        let config = IterativeFitConfig {
            rchi2_threshold: 1e-6,
            contamination_percentage: 0.0,
            ..IterativeFitConfig::default()
        };
        let output = iterative_fit_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers(),
            &config,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(output.passes, 1);
        assert!(output.fit.evaluation.outlier.iter().all(|&flag| !flag));
    }

    fn iterative_fit_with(
        observed: &CoordinateBatch,
        config: &IterativeFitConfig,
    ) -> IterativeFitOutput {
        iterative_fit_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            observed,
            &observers(),
            config,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap()
    }

    #[test]
    fn iterative_fit_min_obs_bounds_the_outlier_allowance() {
        // One gross outlier and a threshold no pass can meet: with
        // `min_obs` equal to the observation count nothing may be removed;
        // one below it exactly one observation may be, whatever the
        // contamination percentage allows.
        let observed = synthetic_observations(Some(3));
        let config = |min_obs: usize| IterativeFitConfig {
            rchi2_threshold: 1e-6,
            min_obs,
            contamination_percentage: 50.0,
            ..IterativeFitConfig::default()
        };
        let pinned = iterative_fit_with(&observed, &config(NUM_OBS));
        assert_eq!(pinned.passes, 1);
        assert!(pinned.fit.evaluation.outlier.iter().all(|&flag| !flag));
        let one = iterative_fit_with(&observed, &config(NUM_OBS - 1));
        assert_eq!(one.passes, 2);
        let outlier = &one.fit.evaluation.outlier;
        assert_eq!(
            outlier.iter().filter(|&&flag| flag).count(),
            1,
            "{outlier:?}"
        );
        assert!(outlier[3], "{outlier:?}");
    }

    #[test]
    fn iterative_fit_keeps_the_arc_above_min_arc_length() {
        // The gross outlier is the last observation (day 44 of the arc):
        // removing it would leave a 40-day arc. Above that `min_arc_length`
        // the loop stops without removing it; below, it is removed as usual.
        let observed = synthetic_observations(Some(NUM_OBS - 1));
        let config = |min_arc_length: f64| IterativeFitConfig {
            rchi2_threshold: 2.0,
            min_arc_length,
            ..IterativeFitConfig::default()
        };
        let kept = iterative_fit_with(&observed, &config(41.0));
        assert_eq!(kept.passes, 1);
        assert!(kept.fit.evaluation.outlier.iter().all(|&flag| !flag));
        assert!(kept.fit.evaluation.reduced_chi2 > 2.0);
        let removed = iterative_fit_with(&observed, &config(1.0));
        assert_eq!(removed.passes, 2);
        assert!(removed.fit.evaluation.outlier[NUM_OBS - 1]);
        assert!(removed.fit.evaluation.reduced_chi2 <= 2.0);
    }

    #[test]
    fn iterative_fit_returns_the_best_converged_pass() {
        // A clean arc with a threshold no pass can meet: every pass
        // converges and the loop spends its whole allowance. The returned
        // fit is the pass with the lowest reduced chi2, reproduced here pass
        // by pass with the loop's own worst-observation rule.
        let observed = synthetic_observations(None);
        let config = IterativeFitConfig {
            rchi2_threshold: 1e-9,
            ..IterativeFitConfig::default()
        };
        let output = iterative_fit_with(&observed, &config);
        let limit = max_outliers(NUM_OBS, config.min_obs, config.contamination_percentage);
        assert_eq!(limit, 2);
        assert_eq!(output.passes, limit + 1);

        let mut ignore = vec![false; NUM_OBS];
        let mut best: Option<WhitenedFitOutput> = None;
        for _ in 0..=limit {
            let pass = fit(&observed, &ignore, &config.fit);
            assert!(pass.converged);
            if best.as_ref().is_none_or(|current| {
                pass.evaluation.reduced_chi2 < current.evaluation.reduced_chi2
            }) {
                best = Some(pass.clone());
            }
            let solution: Vec<usize> = (0..NUM_OBS).filter(|&row| !ignore[row]).collect();
            let worst = lowest_probability_row(
                &solution,
                &pass.evaluation.probability,
                &pass.evaluation.residuals,
            )
            .unwrap();
            ignore[worst] = true;
        }
        let best = best.unwrap();
        assert_eq!(output.fit.state, best.state);
        assert_eq!(
            output.fit.evaluation.reduced_chi2,
            best.evaluation.reduced_chi2
        );
        assert_eq!(output.fit.evaluation.outlier, best.evaluation.outlier);
    }

    #[test]
    fn cmc2003_too_few_observations_fits_once_without_rejection() {
        // At or below `CMC2003_MIN_OBS` observations the loop fits once,
        // flags `too_few_observations` and rejects nothing, outlier or not.
        use crate::cmc2003::CMC2003_MIN_OBS;
        let rows: Vec<usize> = vec![0, 5, NUM_OBS - 1];
        assert!(rows.len() <= CMC2003_MIN_OBS);
        let observed =
            super::super::od::take_coordinate_batch(&synthetic_observations(Some(5)), &rows)
                .unwrap();
        let observers = super::super::od::take_observer_batch(&observers(), &rows).unwrap();
        let config = Cmc2003FitConfig {
            fit: WhitenedFitConfig {
                validate_covariance: false,
                ..WhitenedFitConfig::default()
            },
            ..Cmc2003FitConfig::default()
        };
        let output = cmc2003_fit_barycentric(
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
        assert_eq!(output.n_iterations, 1);
        assert_eq!(output.n_rejected, 0);
        assert_eq!(output.n_recovered, 0);
        assert_eq!(output.flags, vec!["too_few_observations".to_string()]);
        assert!(output.fit.evaluation.outlier.iter().all(|&flag| !flag));
    }

    #[test]
    fn cmc2003_max_iterations_refits_on_the_final_selection() {
        // A one-pass budget: the first pass rejects the gross outlier and
        // the budget is spent, so the loop flags `max_iterations` and refits
        // once more on the final selection rather than returning the
        // pre-rejection fit.
        let observed = synthetic_observations(Some(3));
        let config = Cmc2003FitConfig {
            max_iterations: 1,
            ..Cmc2003FitConfig::default()
        };
        let output = cmc2003_fit_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers(),
            &config,
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(output.n_iterations, 1);
        assert_eq!(output.flags, vec!["max_iterations".to_string()]);
        assert_eq!(output.n_rejected, 1);
        assert!(output.fit.evaluation.outlier[3]);
        assert_eq!(output.fit.weights[3], 0.0);
        assert_eq!(output.fit.evaluation.num_obs, NUM_OBS - 1);
        assert!(output.fit.evaluation.reduced_chi2 < 2.0);
    }

    #[test]
    fn cmc2003_rejects_gross_outliers_and_recovers_truth() {
        let mut observed = synthetic_observations(Some(3));
        // A second gross outlier on another row, in longitude.
        if let crate::CoordinateValues::Spherical(values) = &mut observed.values {
            values[8][1] += 15.0 * SIGMA_DEG / values[8][2].to_radians().cos();
        }
        let output = cmc2003_fit_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers(),
            &Cmc2003FitConfig::default(),
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let rejected: Vec<usize> = output
            .fit
            .evaluation
            .outlier
            .iter()
            .enumerate()
            .filter_map(|(row, &flag)| flag.then_some(row))
            .collect();
        assert_eq!(rejected, vec![3, 8], "flags {:?}", output.flags);
        assert_eq!(output.n_rejected, 2);
        assert!(output.n_iterations >= 2);
        assert!(output.flags.is_empty(), "{:?}", output.flags);
        assert!(output.fit.converged);
        let error = max_abs_diff(&output.fit.state[..3], &TRUTH_STATE[..3]);
        assert!(error < 5e-5, "{error}");
        assert_eq!(output.fit.weights[3], 0.0);
        assert_eq!(output.fit.weights[8], 0.0);
        assert_eq!(output.fit.evaluation.num_obs, NUM_OBS - 2);
    }

    #[test]
    fn cmc2003_clean_arc_converges_in_one_pass() {
        let observed = synthetic_observations(None);
        let output = cmc2003_fit_barycentric(
            &TwoBodyPropagator::default(),
            &perturbed_start(),
            &observed,
            &observers(),
            &Cmc2003FitConfig::default(),
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(output.n_iterations, 1);
        assert_eq!(output.n_rejected, 0);
        assert_eq!(output.n_recovered, 0);
        assert!(output.fit.evaluation.outlier.iter().all(|&flag| !flag));
    }

    #[test]
    fn cmc2003_finalization_uses_one_full_predictor_jacobian_outside_solver_budget() {
        let observed = synthetic_observations(None);
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        for jacobian in [
            JacobianMethod::Analytic,
            JacobianMethod::Central,
            JacobianMethod::TwoPoint,
        ] {
            for validate_covariance in [false, true] {
                let counting = CountingPredictor {
                    inner: &predictor,
                    batches: std::cell::RefCell::new(Vec::new()),
                    mode: PredictorMode::Normal,
                };
                let output = cmc2003_fit_with(
                    &counting,
                    &orbit(TRUTH_STATE),
                    &observed,
                    &observers(),
                    &Cmc2003FitConfig {
                        fit: WhitenedFitConfig {
                            jacobian,
                            validate_covariance,
                            max_nfev: Some(1),
                            ..WhitenedFitConfig::default()
                        },
                        ..Cmc2003FitConfig::default()
                    },
                    &NoopProvider,
                    &ZeroTranslationProvider,
                )
                .unwrap();
                assert_eq!(output.n_iterations, 1);
                assert_eq!(output.fit.iterations, 1);
                assert!(!output.fit.converged);
                // CMC always performs its fail-closed selected-mask
                // weak-direction probe, independent of the standalone fit
                // warning option.
                assert_eq!(*counting.batches.borrow(), vec![1, 12, 1, 2]);
            }
        }
    }

    #[test]
    fn general_fit_accepts_spd_ill_conditioning_but_cmc_refuses_it() {
        let mut fixture = linear_fixture(0.0);
        // Preserve full rank while making one state direction much more weakly
        // constrained than CMC permits. General differential correction must
        // not apply CMC's dimensionful conditioning heuristic.
        for row in &mut fixture.predictor.design {
            row[5] *= 1.0e-7;
        }
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Central,
            validate_covariance: false,
            ..WhitenedFitConfig::default()
        };
        let direct = fit_orbit_whitened_with(
            &fixture.predictor,
            &perturbed_start(),
            &fixture.observed,
            &observers(),
            &[false; NUM_OBS],
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(direct.converged, "{:?}", direct.warnings);
        assert!(covariance_validity(&direct.covariance).is_ok());
        let (minimum, maximum) = covariance_spectrum(&direct.covariance).unwrap();
        assert!(minimum / maximum < COVARIANCE_MIN_RELATIVE_EIGENVALUE);
        assert!(covariance_quality(&direct.covariance).is_err());

        let cmc = cmc2003_fit_with(
            &fixture.predictor,
            &perturbed_start(),
            &fixture.observed,
            &observers(),
            &Cmc2003FitConfig {
                fit: config,
                ..Cmc2003FitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .expect_err("CMC must retain its stricter conditioning gate");
        assert!(cmc.to_string().contains("ill-conditioned"), "{cmc}");
    }

    #[test]
    fn two_epoch_fit_is_rejected_before_numerical_work() {
        let fixture = linear_fixture(0.0);
        let rows = vec![0, 1];
        let observed = super::super::od::take_coordinate_batch(&fixture.observed, &rows).unwrap();
        let observers = super::super::od::take_observer_batch(&observers(), &rows).unwrap();
        let error = fit_orbit_whitened_with(
            &fixture.predictor,
            &perturbed_start(),
            &observed,
            &observers,
            &[false; 2],
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .expect_err("a two-epoch six-state fit is structurally underdetermined");
        assert!(error.to_string().contains("at least 3 are required"));
        assert!(fixture.predictor.batches.borrow().is_empty());
    }

    struct FailAtPredictor<'a> {
        inner: &'a dyn SphericalPredictor,
        calls: std::cell::Cell<usize>,
        fail_at: Vec<usize>,
        error: PropagationError,
    }

    impl SphericalPredictor for FailAtPredictor<'_> {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            if self.fail_at.contains(&call) {
                return Err(self.error.clone());
            }
            self.inner.predict_spherical(states, geometry, observers)
        }

        fn backend(&self) -> super::super::BackendIdentity {
            self.inner.backend()
        }
    }

    fn corrected_iod_cmc_call_fixture(
        error: PropagationError,
    ) -> PropagationResultValue<super::super::od::IodResult> {
        let observed = synthetic_observations(None);
        let observers = observers();
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let mut config = corrected_iod_config(1.0e-12);
        config.correction.min_stage_obs = observed.len();
        config.correction.max_corrections = 2;
        config.correction.max_converged = 1;
        config.correction.max_stages = 1;
        let mut result =
            super::super::od::iod_candidate_pool(&predictor, &observed, &observers, &config)?;
        let candidate = result.candidates.first().expect("fixture candidate");
        let seed_coordinates = CoordinateBatch::cartesian(
            vec![candidate.state],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::from_code("SUN"), 1),
            Some(crate::TimeArray::new(
                candidate.epoch_scale,
                vec![candidate.epoch],
            )?),
            None,
        )?;
        let seed = OrbitBatch::new(
            vec![crate::OrbitId("iod-candidate".to_string())],
            vec![None],
            seed_coordinates,
        )?;
        let counter = FailAtPredictor {
            inner: &predictor,
            calls: std::cell::Cell::new(0),
            fail_at: Vec::new(),
            error: error.clone(),
        };
        let stage_config = WhitenedFitConfig {
            loss: LossType::Linear,
            validate_covariance: false,
            ..WhitenedFitConfig::default()
        };
        let output = stage_fit(
            &counter,
            &seed,
            &observed,
            &observers,
            &vec![true; observed.len()],
            &stage_config,
            f64::INFINITY,
            &NoopProvider,
            &ZeroTranslationProvider,
        )?
        .expect("fixture whole-arc stage");
        assert!(
            output.evaluation.reduced_chi2 > config.rchi2_threshold,
            "fixture must enter the CMC call site"
        );
        let calls_per_stage = counter.calls.get();
        let failing = FailAtPredictor {
            inner: &predictor,
            calls: std::cell::Cell::new(0),
            fail_at: vec![calls_per_stage, 2 * calls_per_stage + 1],
            error,
        };
        // Two distinct records make continuation after the first local CMC
        // refusal observable at this exact call site.
        result.candidates.push(result.candidates[0].clone());
        correct_candidates(
            &failing,
            &observed,
            &observers,
            &config,
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
            &mut result,
            &[0, 1],
        )?;
        assert!(failing.calls.get() > 2 * calls_per_stage + 1);
        Ok(result)
    }

    #[test]
    fn corrected_iod_cmc_call_site_propagates_fatal_errors_and_localizes_only_cmc() {
        for fatal in [
            PropagationError::Backend("provider".to_string()),
            PropagationError::BackendProtocol("protocol".to_string()),
            PropagationError::InvalidRequest("settings".to_string()),
        ] {
            let propagated = corrected_iod_cmc_call_fixture(fatal.clone())
                .expect_err("fatal CMC call-site errors must propagate");
            assert_eq!(propagated, fatal);
        }

        let local = corrected_iod_cmc_call_fixture(PropagationError::CmcCovarianceUnavailable(
            "rank".to_string(),
        ))
        .expect("CMC covariance refusal must remain candidate-local");
        assert_eq!(local.candidates_failed, 2);
        assert_eq!(local.candidate_failures.len(), 2);
        assert!(local.candidate_failures.iter().all(|failure| failure
            .message
            .contains("CMC2003 expected covariance unavailable")));
    }

    #[test]
    fn corrected_iod_rejection_locality_records_only_candidate_failures() {
        let triplet = [2, 5, 9];
        let root_index = 3;
        for error in [
            PropagationError::CmcCovarianceUnavailable("weak direction".to_string()),
            PropagationError::RowFailure("one row".to_string()),
        ] {
            let mut result =
                super::super::od::IodResult::not_found(crate::TimeScale::Tdb, "test".to_string());
            let message = record_candidate_local_rejection_failure(
                error.clone(),
                4,
                triplet,
                root_index,
                &mut result,
            )
            .expect("the candidate-local refusal must continue the wider search");
            assert!(message.contains(&error.to_string()));
            assert_eq!(result.candidates_failed, 1);
            assert_eq!(result.candidate_failures.len(), 1);
            let diagnostic = &result.candidate_failures[0];
            assert_eq!(diagnostic.triplet, triplet);
            assert_eq!(diagnostic.root_index, root_index);
            assert_eq!(diagnostic.message, message);
        }
    }

    #[test]
    fn corrected_iod_rejection_propagates_provider_protocol_settings_and_other_errors() {
        let fatal = [
            PropagationError::Backend("provider".to_string()),
            PropagationError::BackendProtocol("protocol".to_string()),
            PropagationError::InvalidRequest("settings".to_string()),
            PropagationError::ThreadPool("global".to_string()),
            PropagationError::MissingOrbitTimes,
            PropagationError::UnsupportedCovarianceMode(
                super::super::CovariancePropagation::Monte {
                    samples: 4,
                    seed: 1,
                },
            ),
        ];
        for error in fatal {
            let mut result =
                super::super::od::IodResult::not_found(crate::TimeScale::Tdb, "test".to_string());
            let propagated = record_candidate_local_rejection_failure(
                error.clone(),
                4,
                [2, 5, 9],
                3,
                &mut result,
            )
            .expect_err("fatal errors must cross the corrected-IOD call site");
            assert_eq!(propagated, error);
            assert_eq!(result.candidates_failed, 0);
            assert!(result.candidate_failures.is_empty());
        }
    }

    #[test]
    fn corrected_iod_stage_keeps_only_state_from_rank_deficient_covariance() {
        let mut fixture = linear_fixture(0.0);
        for row in &mut fixture.predictor.design {
            row[5] = 0.0;
        }
        let staged = stage_fit(
            &fixture.predictor,
            &orbit(TRUTH_STATE),
            &fixture.observed,
            &observers(),
            &[true; NUM_OBS],
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Central,
                validate_covariance: false,
                ..WhitenedFitConfig::default()
            },
            1.0e30,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap()
        .expect("a finite intermediate state is a valid covariance-free seed");
        assert!(staged.evaluation.reduced_chi2 < 1.0e30);
        assert_eq!(staged.status_code, STATUS_INVALID_COVARIANCE);
        assert!(reportable_iod_covariance(&staged.covariance).is_none());
        assert!(staged.state.iter().all(|value| value.is_finite()));
        let seed = orbit_with_state(&orbit(TRUTH_STATE), staged.state).unwrap();
        assert!(seed.coordinates.covariance.is_none());
    }

    #[test]
    fn invalid_covariance_is_candidate_local_unsuccessful_output() {
        let mut fixture = linear_fixture(0.0);
        // Predictions remain finite and the optimizer can terminate, but one
        // unconstrained state axis makes the normal matrix exactly singular.
        for row in &mut fixture.predictor.design {
            row[5] = 0.0;
        }
        let output = fit_orbit_whitened_with(
            &fixture.predictor,
            &perturbed_start(),
            &fixture.observed,
            &observers(),
            &[false; NUM_OBS],
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Central,
                validate_covariance: false,
                ..WhitenedFitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .expect("candidate-local covariance failure must remain an output");
        assert!(!output.converged);
        assert_eq!(output.status_code, STATUS_INVALID_COVARIANCE);
        assert!(output.covariance.iter().any(|value| !value.is_finite()));
        assert!(output
            .warnings
            .iter()
            .any(|warning| warning.contains("usable state covariance")));
    }

    #[test]
    fn jacobian_covariance_symmetrization_is_non_vacuous() {
        let mut covariance = [0.0; 36];
        for axis in 0..6 {
            covariance[axis * 6 + axis] = 1.0;
        }
        covariance[1] = 2.0;
        covariance[6] = 4.0;
        let canonical = canonicalize_covariance_symmetry(covariance);
        assert_eq!(canonical[1], 3.0);
        assert_eq!(canonical[6], 3.0);
        for row in 0..6 {
            for col in 0..6 {
                assert_eq!(
                    canonical[row * 6 + col].to_bits(),
                    canonical[col * 6 + row].to_bits()
                );
            }
        }
    }

    #[test]
    fn invalid_covariance_does_not_erase_nonconvergence_status() {
        let mut fixture = linear_fixture(0.0);
        for row in &mut fixture.predictor.design {
            row[5] = 0.0;
        }
        let output = fit_orbit_whitened_with(
            &fixture.predictor,
            &perturbed_start(),
            &fixture.observed,
            &observers(),
            &[false; NUM_OBS],
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Central,
                validate_covariance: false,
                max_iterations: 1,
                ..WhitenedFitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .expect("non-converged fit remains a diagnostic output");
        assert!(!output.converged);
        assert_eq!(output.status_code, STATUS_LIMIT);
        assert!(output.covariance.iter().any(|value| !value.is_finite()));
        assert!(output
            .warnings
            .iter()
            .any(|warning| warning.contains("usable state covariance")));
    }

    #[test]
    fn cmc_covariance_refusal_retains_actionable_fit_diagnostics() {
        let fixture = linear_fixture(0.0);
        let output = fit_orbit_whitened_with(
            &fixture.predictor,
            &orbit(TRUTH_STATE),
            &fixture.observed,
            &observers(),
            &[false; NUM_OBS],
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Central,
                validate_covariance: false,
                ..WhitenedFitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let error = cmc_covariance_error("deliberate probe refusal", &output, &[true; NUM_OBS], 7);
        let message = error.to_string();
        for field in [
            "deliberate probe refusal",
            "pass:7",
            "selected:12",
            "status:",
            "converged:",
            "cost:",
            "state:",
            "covariance:",
        ] {
            assert!(message.contains(field), "missing {field}: {message}");
        }
    }

    #[test]
    fn cmc_covariance_quality_fails_closed_on_invalid_rank_and_shape() {
        let mut identity = [0.0; 36];
        for axis in 0..6 {
            identity[axis * 6 + axis] = 1.0;
        }
        assert!(covariance_validity(&identity).is_ok());
        assert!(covariance_quality(&identity).is_ok());

        let mut invalid = identity;
        invalid[0] = f64::NAN;
        assert!(covariance_validity(&invalid).is_err());
        assert!(covariance_quality(&invalid)
            .unwrap_err()
            .contains("non-finite"));

        invalid = identity;
        invalid[1] = 1.0e-3;
        assert!(covariance_validity(&invalid).is_err());
        assert!(covariance_quality(&invalid)
            .unwrap_err()
            .contains("symmetric"));

        invalid = identity;
        invalid[0] = -1.0;
        assert!(covariance_validity(&invalid).is_err());
        assert!(covariance_quality(&invalid)
            .unwrap_err()
            .contains("positive-definite"));

        invalid = identity;
        invalid[0] = 0.0;
        assert!(covariance_validity(&invalid).is_err());
        assert!(covariance_quality(&invalid)
            .unwrap_err()
            .contains("positive-definite"));

        invalid = identity;
        invalid[1] = 1.0;
        invalid[6] = 1.0;
        assert!(covariance_validity(&invalid).is_err());
        assert!(covariance_quality(&invalid).is_err());

        let correlation = |gap: f64| {
            let mut covariance = identity;
            covariance[1] = 1.0 - gap;
            covariance[6] = 1.0 - gap;
            covariance
        };
        assert!(covariance_validity(&correlation(4.0 * f64::EPSILON))
            .unwrap_err()
            .contains("numerically rank-deficient"));
        assert!(covariance_validity(&correlation(64.0 * f64::EPSILON)).is_ok());

        invalid = identity;
        invalid[0] = 1.0e-13;
        assert!(covariance_validity(&invalid).is_ok());
        assert!(covariance_quality(&invalid)
            .unwrap_err()
            .contains("ill-conditioned"));
    }

    fn full_od_config(refinement: RefinementConfig) -> FullOdConfig {
        FullOdConfig {
            iod: super::super::od::IodConfig {
                min_obs: 6,
                min_arc_length: 1.0,
                contamination_percentage: 20.0,
                // Gauss on the longest-arc triplets of this synthetic geometry
                // seeds the refinement coarsely; the threshold only gates it.
                rchi2_threshold: 1e10,
                observation_selection_method:
                    super::super::od::ObservationSelectionMethod::Combinations,
                light_time: true,
                mu: crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap(),
                speed_of_light: 173.144_632_674_240_3,
                max_triplets: None,
                max_pair_visits: 20_000,
                max_raw_pair_visits: 200_000,
                max_middle_visits: 200_000,
                velocity_method: super::super::od::VelocityMethod::Gibbs,
                spans: super::super::od::SpanSearchConfig::default(),
                acceptance: super::super::od::IodAcceptance::Prefit,
                correction: super::super::od::CorrectionConfig::default(),
                ranging: super::super::od_ranging::RangingConfig::default(),
            },
            refinement,
        }
    }

    fn corrected_iod_config(rchi2_threshold: f64) -> super::super::od::IodConfig {
        super::super::od::IodConfig {
            rchi2_threshold,
            contamination_percentage: 0.0,
            ..super::super::od::IodConfig::default()
        }
    }

    fn two_body() -> (TwoBodyPropagator, EphemerisOptions) {
        (TwoBodyPropagator::default(), ephemeris_options())
    }

    #[test]
    fn grow_stage_grows_geometrically_and_jumps_gaps() {
        let times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 40.0, 40.2, 41.0];
        let nights = super::super::od::observation_nights(&times, None);
        let mask =
            |rows: &[usize]| -> Vec<bool> { (0..times.len()).map(|r| rows.contains(&r)).collect() };
        // Span 2 days about 2..4 grown 4x: 1 .. 7 days -> rows 0..=6.
        assert_eq!(
            grow_stage(&times, &nights, &mask(&[2, 3, 4]), 4.0),
            mask(&[0, 1, 2, 3, 4, 5, 6])
        );
        // Nothing within 4x of the first week: the night of the nearest
        // observation beyond it (MJD 40) joins, both of its observations.
        assert_eq!(
            grow_stage(&times, &nights, &mask(&[0, 1, 2, 3, 4, 5, 6]), 4.0)
                .iter()
                .filter(|&&k| k)
                .count(),
            10 - 1
        );
        let jumped = grow_stage(&times, &nights, &mask(&[0, 1, 2, 3, 4, 5, 6]), 1.5);
        assert_eq!(jumped, mask(&[0, 1, 2, 3, 4, 5, 6, 7, 8]));
        // Everything included: unchanged.
        let all = vec![true; times.len()];
        assert_eq!(grow_stage(&times, &nights, &all, 4.0), all);
    }

    #[test]
    fn corrected_acceptance_seeds_where_the_prefit_gate_cannot() {
        // 44-day arc with one outlier: the raw Gauss orbits do not fit the
        // whole arc within reduced chi2 50, the corrected ones do.
        let observed = synthetic_observations(Some(3));
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let prefit = super::super::od::run_iod(
            &predictor,
            &observed,
            &observers(),
            &super::super::od::IodConfig {
                acceptance: super::super::od::IodAcceptance::Prefit,
                ..corrected_iod_config(50.0)
            },
        )
        .unwrap();
        assert!(!prefit.found(), "prefit unexpectedly accepted");

        let config = corrected_iod_config(50.0);
        let result = run_iod_with(
            &predictor,
            &observed,
            &observers(),
            &config,
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(result.found(), "{:?}", result.not_found_reason);
        assert!(result.corrections_run >= 1);
        assert!(result.corrections_run <= config.correction.max_corrections);
        let accepted = result.accepted_candidate().unwrap();
        assert!(accepted.converged && accepted.stages >= 2);
        let rchi2 = accepted.postfit_reduced_chi2.unwrap();
        assert!(rchi2 <= 50.0);
        // The lowest whole-arc score among the converged, index on ties.
        let score = accepted.whole_arc_score.unwrap();
        for candidate in result.candidates.iter().filter(|c| c.converged) {
            assert!(candidate.whole_arc_score.unwrap() >= score * (1.0 - 1e-6));
        }
        // The search stops at `max_converged`, but the sibling roots of the
        // triplet that reached it are still corrected (up to two more).
        assert!(
            result.candidates.iter().filter(|c| c.converged).count()
                <= config.correction.max_converged + 2
        );
        // The unrejected full-arc minimum: the same orbit a fit started from
        // the truth (propagated to the Gauss epoch) reaches on all
        // observations.
        let state = accepted.corrected_state.unwrap();
        let mu = crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap();
        let truth = crate::propagate_2body_row::<f64>(
            TRUTH_STATE,
            accepted.epoch.mjd() - 60_000.0,
            mu,
            1000,
            1e-14,
        );
        let truth_orbit = OrbitBatch::new(
            vec![OrbitId("truth".to_string())],
            vec![None],
            CoordinateBatch::cartesian(
                vec![truth],
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
                Some(TimeArray::new(TimeScale::Tdb, vec![accepted.epoch]).unwrap()),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let reference = fit_orbit_whitened_barycentric(
            &propagator,
            &truth_orbit,
            &observed,
            &observers(),
            &[false; NUM_OBS],
            &WhitenedFitConfig {
                validate_covariance: false,
                ..WhitenedFitConfig::default()
            },
            &options,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!((reference.evaluation.reduced_chi2 - rchi2).abs() < 1e-6 * rchi2);
        assert!(max_abs_diff(&state[..3], &reference.state[..3]) < 1e-7);
        let output = result.to_output(NUM_OBS);
        assert_eq!(output.state, state);
        assert!(
            output.outlier.iter().all(|&flag| !flag),
            "no rejection in IOD"
        );
    }

    #[test]
    fn corrected_acceptance_rejects_on_the_whole_arc_only() {
        // A gross outlier on the arc: a threshold below the unrejected
        // whole-arc fit (reduced chi2 6.2) is met only after the CMC2003
        // rejection of the whole-arc stage, which reports the outlier.
        let observed = synthetic_observations(Some(3));
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let config = super::super::od::IodConfig {
            // Intermediate stages see the outlier unrejected: allow them.
            correction: super::super::od::CorrectionConfig {
                max_corrections: 47,
                ..super::super::od::CorrectionConfig::default()
            },
            // Whole-arc rejection is bounded by the IOD's allowance.
            contamination_percentage: 20.0,
            ..corrected_iod_config(3.0)
        };
        let result = run_iod_with(
            &predictor,
            &observed,
            &observers(),
            &config,
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let accepted = result
            .accepted_candidate()
            .unwrap_or_else(|| panic!("no candidate converged: {:?}", result.not_found_reason));
        assert_eq!(accepted.removed, vec![3]);
        assert!(accepted.postfit_reduced_chi2.unwrap() <= 3.0);
        let output = result.to_output(NUM_OBS);
        assert!(output.outlier[3]);
        assert_eq!(output.outlier.iter().filter(|&&f| f).count(), 1);
    }

    #[test]
    fn huber_stages_carry_an_outlier_the_linear_stages_stall_on() {
        // The outlier (row 3) sits inside the early stages: with the linear
        // loss their plain reduced chi2 exceeds 3 and the candidates whose
        // stages hold it stall, so the search spends its whole budget on
        // the few that avoid it; Huber stages down-weight it, gate on the
        // robust cost, and reach the whole-arc CMC2003 stage (which rejects
        // it) in fewer corrections.
        let observed = synthetic_observations(Some(3));
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let run = |stage_loss| {
            let config = super::super::od::IodConfig {
                correction: super::super::od::CorrectionConfig {
                    stage_loss,
                    // Candidates whose stages hold the outlier come first in
                    // the whole-arc order; a local prefit ranks the ones
                    // that avoid it first, and both losses then succeed at
                    // once.
                    prefit_scope: super::super::od::PrefitScope::WholeArc,
                    ..super::super::od::CorrectionConfig::default()
                },
                contamination_percentage: 20.0,
                ..corrected_iod_config(3.0)
            };
            run_iod_with(
                &predictor,
                &observed,
                &observers(),
                &config,
                &WhitenedFitConfig::default(),
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
        };
        let linear = run(LossType::Linear);
        let huber = run(LossType::Huber);
        let accepted = huber
            .accepted_candidate()
            .unwrap_or_else(|| panic!("huber stages stalled: {:?}", huber.not_found_reason));
        assert_eq!(accepted.removed, vec![3]);
        assert!(accepted.postfit_reduced_chi2.unwrap() <= 3.0);
        assert!(huber.corrections_run < linear.corrections_run);
        assert!(linear.candidates.iter().any(|c| c
            .correction_note
            .as_deref()
            .is_some_and(|n| n.contains("stalled"))));
    }

    #[test]
    fn whole_arc_rejection_is_bounded_by_the_iod_contamination() {
        // The gross outlier needs one rejection to bring the whole arc
        // within 3: allowed at 20 % contamination (2 of 12), refused at 0 %.
        let observed = synthetic_observations(Some(3));
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let run = |contamination_percentage: f64| {
            run_iod_with(
                &predictor,
                &observed,
                &observers(),
                &super::super::od::IodConfig {
                    contamination_percentage,
                    ..corrected_iod_config(3.0)
                },
                &WhitenedFitConfig::default(),
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
        };
        let allowed = run(20.0);
        assert_eq!(allowed.accepted_candidate().unwrap().removed, vec![3]);
        let refused = run(0.0);
        assert!(!refused.found());
        assert!(refused
            .candidates
            .iter()
            .all(|c| c.removed.is_empty() || !c.converged));
    }

    #[test]
    fn candidates_rank_on_the_whole_arc_not_on_the_observations_they_kept() {
        // 2002 RV112: a wrong root rejected 22 of 44 observations and fit the
        // rest at reduced chi2 1.92; the right roots kept all 44 at 2.15.
        // Clipping every chi2 at CMC2003's rejection threshold, the wrong
        // root scores (1.92 x 38 + 22 x 8) / 82 = 3.04 and loses.
        let evaluation = |kept_rchi2: f64, rejected: usize| super::super::od::FitEvaluation {
            residuals: vec![0.0; 44 * 6],
            chi2: (0..44)
                .map(|row| {
                    if row < rejected {
                        1e4
                    } else {
                        kept_rchi2 * (2.0 * (44 - rejected) as f64 - 6.0) / (44 - rejected) as f64
                    }
                })
                .collect(),
            dof: vec![2; 44],
            probability: vec![0.5; 44],
            orbit_chi2: 0.0,
            reduced_chi2: kept_rchi2,
            arc_length: 30.0,
            num_obs: 44 - rejected,
            outlier: (0..44).map(|row| row < rejected).collect(),
        };
        let wrong = whole_arc_score(&evaluation(1.92, 22));
        let right = whole_arc_score(&evaluation(2.15, 0));
        assert!(
            (wrong - (1.92 * 38.0 + 22.0 * 8.0) / 82.0).abs() < 1e-9,
            "{wrong}"
        );
        assert!((right - 2.15).abs() < 1e-9, "{right}");
        assert!(right < wrong);
        // Over the margin: the wrong root is not an equal alternative.
        assert!((wrong - right) * 82.0 > AMBIGUITY_DELTA_CHI2);
    }

    /// Corrected IOD over a subset of the fixture's rows with 12-sigma
    /// outliers, 50 % contamination and the given `min_obs`.
    fn corrected_iod_on_rows(
        rows: &[usize],
        outliers: &[usize],
        min_obs: Option<usize>,
    ) -> super::super::od::IodResult {
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let defaults = corrected_iod_config(3.0);
        run_iod_with(
            &predictor,
            &super::super::od::take_coordinate_batch(
                &synthetic_observations_with_outliers(4, outliers),
                rows,
            )
            .unwrap(),
            &super::super::od::take_observer_batch(&observers(), rows).unwrap(),
            &super::super::od::IodConfig {
                min_obs: min_obs.unwrap_or(defaults.min_obs),
                contamination_percentage: 50.0,
                ..defaults
            },
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap()
    }

    fn kept(result: &super::super::od::IodResult) -> Option<usize> {
        result
            .accepted_candidate()
            .and_then(|candidate| candidate.evaluation.as_ref())
            .map(|evaluation| evaluation.num_obs)
    }

    #[test]
    fn corrected_acceptance_keeps_the_default_min_obs_after_rejection() {
        // Seven observations, two of them 12-sigma outliers, 50 % allowed.
        let rows = [0, 2, 3, 5, 6, 9, 11];
        let outliers = [3, 9];
        // With min_obs 4 the whole-arc rejection removes two and keeps five.
        let low = corrected_iod_on_rows(&rows, &outliers, Some(4));
        assert_eq!(kept(&low), Some(5));
        // The default min_obs (6) bounds that rejection: no orbit on five.
        assert_eq!(super::super::od::IodConfig::default().min_obs, 6);
        let default = corrected_iod_on_rows(&rows, &outliers, None);
        assert!(!default.found(), "kept {:?}", kept(&default));
        for candidate in &default.candidates {
            if let Some(evaluation) = &candidate.evaluation {
                assert!(evaluation.num_obs >= 6, "{}", evaluation.num_obs);
            }
        }
    }

    #[test]
    fn corrected_acceptance_keeps_a_configured_min_obs_after_rejection() {
        // Eight observations, three of them outliers: with min_obs 4 the
        // rejection keeps four; with min_obs 5 to 7 no orbit is accepted on
        // fewer observations than that.
        let rows = [0, 1, 3, 4, 6, 8, 9, 11];
        let outliers = [3, 6, 9];
        assert_eq!(
            kept(&corrected_iod_on_rows(&rows, &outliers, Some(4))),
            Some(4)
        );
        for min_obs in [5, 6, 7] {
            let result = corrected_iod_on_rows(&rows, &outliers, Some(min_obs));
            assert!(
                kept(&result).is_none_or(|kept| kept >= min_obs),
                "min_obs {min_obs}: kept {:?}",
                kept(&result)
            );
        }
    }

    #[test]
    fn cmc2003_rejection_never_leaves_fewer_than_min_kept() {
        let observed = synthetic_observations_with_outliers(4, &[2, 5, 8]);
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let run = |min_kept: usize| {
            cmc2003_fit_with(
                &predictor,
                &orbit(TRUTH_STATE),
                &observed,
                &observers(),
                &Cmc2003FitConfig {
                    min_kept,
                    ..Cmc2003FitConfig::default()
                },
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
            .fit
            .evaluation
            .num_obs
        };
        // The default floor lets the three outliers go; 11 keeps all but one.
        assert_eq!(run(Cmc2003FitConfig::default().min_kept), NUM_OBS - 3);
        assert!(run(11) >= 11);
    }

    #[test]
    fn corrected_acceptance_needs_min_arc_length_before_and_after_rejection() {
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let run = |observed: &CoordinateBatch, min_arc_length: f64| {
            run_iod_with(
                &predictor,
                observed,
                &observers(),
                &super::super::od::IodConfig {
                    min_arc_length,
                    contamination_percentage: 20.0,
                    ..corrected_iod_config(3.0)
                },
                &WhitenedFitConfig::default(),
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
        };
        // The 44-day arc is shorter than 50 days: refused up front.
        let short = run(&synthetic_observations(None), 50.0);
        assert!(!short.found());
        let reason = short.not_found_reason.unwrap();
        assert!(reason.contains("min_arc_length"), "{reason}");
        // The outlier is the last observation: rejecting it leaves 40 days,
        // shorter than 43.5, so no candidate converges.
        let rejected = run(&synthetic_observations(Some(NUM_OBS - 1)), 43.5);
        assert!(!rejected.found());
        assert!(
            rejected
                .candidates
                .iter()
                .filter_map(|c| c.correction_note.as_deref())
                .any(|note| note.contains("shorter than min_arc_length")),
            "{:?}",
            rejected.not_found_reason
        );
        // With 40 days enough, the same arc is accepted without the outlier.
        let accepted = run(&synthetic_observations(Some(NUM_OBS - 1)), 40.0);
        assert_eq!(
            accepted.accepted_candidate().unwrap().removed,
            vec![NUM_OBS - 1]
        );
    }

    #[test]
    fn local_prefit_scores_each_candidate_on_its_first_stage() {
        let observed = synthetic_observations(None);
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let pool = |prefit_scope| {
            super::super::od::iod_candidate_pool(
                &predictor,
                &observed,
                &observers(),
                &super::super::od::IodConfig {
                    correction: super::super::od::CorrectionConfig {
                        prefit_scope,
                        ..super::super::od::CorrectionConfig::default()
                    },
                    ..corrected_iod_config(50.0)
                },
            )
            .unwrap()
        };
        let local = pool(super::super::od::PrefitScope::Local);
        let whole = pool(super::super::od::PrefitScope::WholeArc);
        assert_eq!(local.candidates.len(), whole.candidates.len());
        let times = super::super::od::observed_times_mjd(&observed).unwrap();
        let nights = super::super::od::observation_nights(&times, Some(&observers().code));
        let correction = super::super::od::CorrectionConfig::default();
        for (l, w) in local.candidates.iter().zip(&whole.candidates) {
            assert_eq!((l.triplet, l.root_index), (w.triplet, w.root_index));
            assert_eq!(w.num_obs, NUM_OBS);
            let stage = first_stage_mask(&times, &nights, l.triplet, &correction);
            assert_eq!(l.num_obs, stage.iter().filter(|&&k| k).count());
            assert!(l.num_obs >= correction.min_stage_obs && l.num_obs <= NUM_OBS);
            // Same orbit, a subset of the observations: never a larger chi2.
            assert!(l.prefit_chi2 <= w.prefit_chi2 * (1.0 + 1e-12));
        }
        // Triplets from the dense windows are scored on part of the arc.
        assert!(local.candidates.iter().any(|c| c.num_obs < NUM_OBS));
    }

    /// The truth's topocentric range and range rate at the attributable's
    /// reference observation.
    fn truth_range(attributable: &super::super::od_ranging::Attributable) -> (f64, f64) {
        let mu = crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap();
        let truth = crate::propagate_2body_row::<f64>(
            TRUTH_STATE,
            attributable.epoch.mjd() - 60_000.0,
            mu,
            1000,
            1e-14,
        );
        let q = attributable.observer;
        let d: [f64; 6] = std::array::from_fn(|i| truth[i] - q[i]);
        let rho = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        (rho, (d[0] * d[3] + d[1] * d[4] + d[2] * d[5]) / rho)
    }

    #[test]
    fn ranging_samples_the_admissible_region_around_the_truth() {
        // Eleven days of noise-free observations: the attributable's rates
        // are the truth's, the truth (bound) lies in the admissible region,
        // and the best grid orbit sits next to it in log range.
        // Four observations over three days: a short arc, the regime
        // ranging serves.
        let rows: Vec<usize> = (4..8).collect();
        let observed =
            super::super::od::take_coordinate_batch(&synthetic_observations_spaced(1, None), &rows)
                .unwrap();
        let observers = super::super::od::take_observer_batch(&observers_spaced(1), &rows).unwrap();
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let mu = crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap();
        let config = super::super::od_ranging::RangingConfig::default();
        let result = super::super::od_ranging::systematic_ranging_with(
            &predictor,
            &observed,
            &observers,
            &config,
            mu,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let (rho, rho_dot) = truth_range(&result.attributable);
        let (lo, hi) =
            super::super::od_ranging::admissible_rho_dot_interval(&result.attributable, rho, mu)
                .unwrap();
        assert!(lo < rho_dot && rho_dot < hi, "{lo} {rho_dot} {hi}");
        assert_eq!(result.grid_points, config.n_rho * config.n_rho_dot);
        assert!(result.admissible_points > 0 && result.admissible_points <= result.grid_points);
        let total: f64 = result.samples.iter().map(|s| s.weight).sum();
        assert!((total - 1.0).abs() < 1e-12);
        let best = result.best_sample().unwrap();
        // The best node is the grid node nearest the truth's range (the
        // log grid steps by a factor ~1.26).
        assert!(
            (best.rho_au / rho).ln().abs() < 0.25,
            "best rho {} vs truth {rho}",
            best.rho_au
        );
        assert!((best.rho_dot_au_per_day - rho_dot).abs() < 1e-4);
        // The truth's own (rho, rho_dot) completes the attributable to an
        // orbit near the truth (within the light-time displacement, as the
        // truth's range here is geometric at reception).
        let state = super::super::od_ranging::ranging_state(&result.attributable, rho, rho_dot);
        let truth = crate::propagate_2body_row::<f64>(
            TRUTH_STATE,
            result.attributable.epoch.mjd() - 60_000.0,
            mu,
            1000,
            1e-14,
        );
        assert!(max_abs_diff(&state[..3], &truth[..3]) < 5e-5);
        assert!(max_abs_diff(&state[3..], &truth[3..]) < 1e-6);
    }

    #[test]
    fn short_arcs_get_the_ranging_sample_and_gauss_failures_its_corrections() {
        let observed = synthetic_observations(None);
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let run = |config: &super::super::od::IodConfig| {
            run_iod_with(
                &predictor,
                &observed,
                &observers(),
                config,
                &WhitenedFitConfig::default(),
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
        };
        // Shorter than min_arc_length: no orbit, the ranging sample instead.
        // The fixture spans 44 days; ranging's arc limit is raised to reach it.
        let long_ranging = super::super::od_ranging::RangingConfig {
            max_arc_days: 100.0,
            ..super::super::od_ranging::RangingConfig::default()
        };
        let short = run(&super::super::od::IodConfig {
            min_arc_length: 100.0,
            ranging: long_ranging,
            ..corrected_iod_config(50.0)
        });
        assert!(!short.found());
        assert!(short.candidates.is_empty());
        assert!(!short.ranging.as_ref().unwrap().samples.is_empty());
        let reason = short.not_found_reason.unwrap();
        assert!(reason.contains("ranging sample attached"), "{reason}");
        // No Gauss triplet at all (spans capped below every pair): the
        // ranging orbits are corrected like candidates and one is accepted.
        let no_gauss = run(&super::super::od::IodConfig {
            spans: super::super::od::SpanSearchConfig {
                max_span_days: 1e-3,
                ..super::super::od::SpanSearchConfig::default()
            },
            ranging: long_ranging,
            ..corrected_iod_config(50.0)
        });
        // At the default 5-day limit ranging does not run on this arc.
        let default_limit = run(&super::super::od::IodConfig {
            spans: super::super::od::SpanSearchConfig {
                max_span_days: 1e-3,
                ..super::super::od::SpanSearchConfig::default()
            },
            ..corrected_iod_config(50.0)
        });
        assert!(!default_limit.found() && default_limit.ranging.is_none());
        let accepted = no_gauss
            .accepted_candidate()
            .unwrap_or_else(|| panic!("{:?}", no_gauss.not_found_reason));
        assert_eq!(accepted.source, super::super::od::CandidateSource::Ranging);
        assert!(accepted.converged);
        let mu = crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap();
        let truth = crate::propagate_2body_row::<f64>(
            TRUTH_STATE,
            accepted.epoch.mjd() - 60_000.0,
            mu,
            1000,
            1e-14,
        );
        // The same least-squares minimum the Gauss path reaches (the
        // fixture's small noise puts it ~1e-5 au from the truth).
        assert!(max_abs_diff(&accepted.corrected_state.unwrap()[..3], &truth[..3]) < 1e-4);
        assert!(accepted.postfit_reduced_chi2.unwrap() < 0.05);
        // Disabled: the Gauss failure stands.
        let disabled = run(&super::super::od::IodConfig {
            spans: super::super::od::SpanSearchConfig {
                max_span_days: 1e-3,
                ..super::super::od::SpanSearchConfig::default()
            },
            ranging: super::super::od_ranging::RangingConfig {
                enabled: false,
                ..super::super::od_ranging::RangingConfig::default()
            },
            ..corrected_iod_config(50.0)
        });
        assert!(!disabled.found() && disabled.ranging.is_none());
    }

    #[test]
    fn linkage_batches_run_the_iod_stage_per_linkage_in_order() {
        let (propagator, options) = two_body();
        let observed = synthetic_observations(None);
        let iod_config = super::super::od::IodConfig {
            contamination_percentage: 20.0,
            ..corrected_iod_config(50.0)
        };
        let fit = WhitenedFitConfig::default();
        let all: Vec<usize> = (0..observed.len()).collect();
        let reversed: Vec<usize> = all.iter().rev().copied().collect();
        let too_few = vec![0, 1];
        let linkages = vec![all.clone(), too_few, reversed];
        let batch = run_iod_linkages_barycentric(
            &propagator,
            &observed,
            &observers(),
            &linkages,
            &iod_config,
            &fit,
            &options,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(batch.len(), 3);
        let single = run_iod_barycentric(
            &propagator,
            &observed,
            &observers(),
            &iod_config,
            &fit,
            &options,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let accepted = |result: &super::super::od::IodResult| {
            result
                .accepted_candidate()
                .unwrap()
                .corrected_state
                .unwrap()
        };
        assert_eq!(accepted(&batch[0]), accepted(&single));
        // A linkage the search cannot solve is a result without an accepted
        // candidate, not a batch failure.
        assert!(batch[1].accepted.is_none());
        // The stage is invariant to the row order of a linkage.
        assert_eq!(accepted(&batch[2]), accepted(&single));
        assert_eq!(
            batch[2]
                .accepted_candidate()
                .unwrap()
                .triplet
                .map(|row| linkages[2][row]),
            single.accepted_candidate().unwrap().triplet
        );
        let out_of_bounds = run_iod_linkages_barycentric(
            &propagator,
            &observed,
            &observers(),
            &[vec![0, observed.len()]],
            &iod_config,
            &fit,
            &options,
            &NoopProvider,
            &ZeroTranslationProvider,
        );
        assert!(matches!(
            out_of_bounds,
            Err(PropagationError::InvalidRequest(_))
        ));
    }

    #[test]
    fn external_propagators_run_the_od_stages_like_native_ones() {
        // adam-core's two-body propagator behind the C ABI (as a plugin such
        // as ASSIST exposes its own) gives the same IOD and refinement as
        // the propagator itself, and reports the external identity.
        let table = super::super::two_body_reference::table(TwoBodyPropagator::default());
        let external = unsafe { super::super::ExternalPropagator::from_raw(table) }.unwrap();
        let identity = external.backend();
        assert_eq!(identity.name, "adam_core.two_body.external");
        assert!(identity.two_body_dynamics);
        let (propagator, options) = two_body();
        let observed = synthetic_observations(Some(3));
        let config = FullOdConfig {
            iod: super::super::od::IodConfig {
                contamination_percentage: 20.0,
                ..corrected_iod_config(50.0)
            },
            refinement: RefinementConfig::default(),
        };
        let run = |iod: &dyn Fn() -> FullOdOutput| iod();
        let native = run(&|| {
            full_od_barycentric(
                &propagator,
                &propagator,
                &observed,
                &observers(),
                &config,
                &options,
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
        });
        let via_abi = run(&|| {
            full_od_barycentric(
                &external,
                &external,
                &observed,
                &observers(),
                &config,
                &options,
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
        });
        let (a, b) = (native.refined.unwrap(), via_abi.refined.unwrap());
        assert!(max_abs_diff(&a.fit.state, &b.fit.state) < 1e-12);
        assert_eq!(a.fit.evaluation.outlier, b.fit.evaluation.outlier);
        assert_eq!(native.iod.accepted, via_abi.iod.accepted);
        // The predictor reports the external backend.
        let predictor = PropagatorPredictor {
            propagator: &external,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        assert_eq!(predictor.backend(), identity);
    }

    /// A predictor with the two-body dynamics and STMs that does not declare
    /// them two-body (ASSIST's position: exact derivatives of non-two-body
    /// dynamics), or (`stm = false`) supplies no derivatives at all.
    struct UndeclaredBackend<'a> {
        inner: &'a dyn SphericalPredictor,
        stm: bool,
    }

    impl SphericalPredictor for UndeclaredBackend<'_> {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            self.inner.predict_spherical(states, geometry, observers)
        }

        fn supports_stm(&self) -> bool {
            self.stm
        }

        fn propagate_stm(
            &self,
            state: [f64; 6],
            non_gravitational_parameters: Option<crate::NonGravitationalParametersRow>,
            epoch_tdb: crate::Epoch,
            times_tdb: &[crate::Epoch],
        ) -> PropagationResultValue<Vec<super::super::StmRow>> {
            self.inner
                .propagate_stm(state, non_gravitational_parameters, epoch_tdb, times_tdb)
        }
    }

    #[test]
    fn propagator_jacobian_is_exact_and_auto_follows_the_dynamics() {
        let observed = synthetic_observations(None);
        let (propagator, options) = two_body();
        let native = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        // Two-body STMs chained through the projection equal the analytic
        // autodiff Jacobian.
        let problem = WhitenedProblem::new(
            &native,
            &perturbed_start(),
            &observed,
            &observers(),
            &WhitenedFitConfig {
                jacobian: JacobianMethod::Propagator,
                ..WhitenedFitConfig::default()
            },
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let state = problem.geometry.state;
        let analytic = problem
            .jacobian(JacobianMethod::Analytic, &state, &[])
            .unwrap();
        let from_stm = problem
            .jacobian(JacobianMethod::Propagator, &state, &[])
            .unwrap();
        let scale = analytic.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(max_abs_diff(&analytic, &from_stm) < 1e-9 * scale);

        // Auto: declared two-body -> analytic; exact derivatives of other
        // dynamics -> the propagator's; neither -> central differences.
        let with_stm = UndeclaredBackend {
            inner: &native,
            stm: true,
        };
        let without = UndeclaredBackend {
            inner: &native,
            stm: false,
        };
        assert_eq!(
            JacobianMethod::Auto.resolve(&native),
            JacobianMethod::Analytic
        );
        assert_eq!(
            JacobianMethod::Auto.resolve(&with_stm),
            JacobianMethod::Propagator
        );
        assert_eq!(
            JacobianMethod::Auto.resolve(&without),
            JacobianMethod::Central
        );
        // The fits reach the same minimum on every route.
        let fit_with = |predictor: &dyn SphericalPredictor| {
            fit_orbit_whitened_with(
                predictor,
                &perturbed_start(),
                &observed,
                &observers(),
                &[false; NUM_OBS],
                &WhitenedFitConfig::default(),
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
        };
        let reference = fit_with(&native);
        for predictor in [&with_stm as &dyn SphericalPredictor, &without] {
            let other = fit_with(predictor);
            assert!(other.converged);
            assert!(max_abs_diff(&other.state, &reference.state) < 1e-9);
        }
        // Through the external C ABI the two-body reference supplies STMs.
        let table = super::super::two_body_reference::table(TwoBodyPropagator::default());
        let external = unsafe { super::super::ExternalPropagator::from_raw(table) }.unwrap();
        assert!(external.supports_stm());
        let epoch = super::super::od::mjd_epoch(60_000.0);
        let times = [super::super::od::mjd_epoch(60_010.0)];
        let via_abi = external
            .propagate_stm(TRUTH_STATE, None, epoch, &times)
            .unwrap();
        let direct = propagator
            .propagate_stm(TRUTH_STATE, None, epoch, &times)
            .unwrap();
        assert_eq!(via_abi, direct);
    }

    fn orbit_with_non_gravitational(source: Option<&str>, a1: Option<f64>) -> OrbitBatch {
        orbit(TRUTH_STATE)
            .with_non_gravitational_parameters(crate::NonGravitationalParametersBatch {
                source: vec![source.map(str::to_string)],
                a1: vec![a1],
                a2: vec![None],
                a3: vec![None],
                aln: vec![Some(1.0)],
                nk: vec![None],
                nm: vec![Some(2.0)],
                nn: vec![Some(5.093)],
                r0: vec![Some(1.0)],
            })
            .unwrap()
    }

    #[test]
    fn inactive_non_gravitational_rows_have_in_tree_and_capsule_stm_parity() {
        let observed = synthetic_observations(None);
        let (propagator, options) = two_body();
        let direct = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let table = super::super::two_body_reference::table(TwoBodyPropagator::default());
        let external = unsafe { super::super::ExternalPropagator::from_raw(table) }.unwrap();
        let capsule = PropagatorPredictor {
            propagator: &external,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let config = WhitenedFitConfig {
            jacobian: JacobianMethod::Propagator,
            ..WhitenedFitConfig::default()
        };

        for inactive in [
            orbit_with_non_gravitational(Some("source-only"), None),
            orbit_with_non_gravitational(Some("zero-a1"), Some(0.0)),
        ] {
            let direct_problem = WhitenedProblem::new(
                &direct,
                &inactive,
                &observed,
                &observers(),
                &config,
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap();
            let capsule_problem = WhitenedProblem::new(
                &capsule,
                &inactive,
                &observed,
                &observers(),
                &config,
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap();
            for geometry in [&direct_problem.geometry, &capsule_problem.geometry] {
                assert!(geometry.non_gravitational_parameters.is_some());
                assert!(geometry.active_non_gravitational_parameters().is_none());
                assert_eq!(
                    geometry.non_gravitational_source.as_deref(),
                    inactive
                        .non_gravitational_parameters
                        .as_ref()
                        .unwrap()
                        .source[0]
                        .as_deref()
                );
                let candidate =
                    super::super::od::candidate_orbit_batch(&[geometry.state], geometry).unwrap();
                assert_eq!(
                    candidate.non_gravitational_parameters,
                    inactive.non_gravitational_parameters
                );
            }
            let state = direct_problem.geometry.state;
            let direct_jacobian = direct_problem
                .jacobian(JacobianMethod::Propagator, &state, &[])
                .unwrap();
            let capsule_jacobian = capsule_problem
                .jacobian(JacobianMethod::Propagator, &state, &[])
                .unwrap();
            assert!(max_abs_diff(&direct_jacobian, &capsule_jacobian) < 1e-12);
        }

        let active = orbit_with_non_gravitational(Some("active"), Some(1e-10));
        for predictor in [&direct as &dyn SphericalPredictor, &capsule] {
            let problem = WhitenedProblem::new(
                predictor,
                &active,
                &observed,
                &observers(),
                &config,
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap();
            assert!(problem.geometry.non_gravitational_parameters.is_some());
            let error = problem
                .jacobian(JacobianMethod::Propagator, &problem.geometry.state, &[])
                .unwrap_err();
            assert!(error.to_string().contains("non-gravitational"), "{error}");
        }
    }

    #[test]
    fn corrected_acceptance_reports_stalled_candidates() {
        let observed = synthetic_observations(None);
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let config = super::super::od::IodConfig {
            correction: super::super::od::CorrectionConfig {
                max_corrections: 3,
                ..super::super::od::CorrectionConfig::default()
            },
            // Ranging on this 44-day fixture, to exercise the fallback.
            ranging: super::super::od_ranging::RangingConfig {
                max_arc_days: 100.0,
                ..super::super::od_ranging::RangingConfig::default()
            },
            ..corrected_iod_config(1e-9)
        };
        let result = run_iod_with(
            &predictor,
            &observed,
            &observers(),
            &config,
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(!result.found());
        // Three Gauss candidates, then (nothing converged) the ranging
        // fallback's best orbits, which stall the same way.
        let gauss = |c: &&super::super::od::IodCandidate| {
            c.source == super::super::od::CandidateSource::Gauss
        };
        let gauss_corrected = result
            .candidates
            .iter()
            .filter(gauss)
            .filter(|c| c.stages > 0)
            .count();
        assert_eq!(gauss_corrected, 3);
        let ranging = result.ranging.as_ref().expect("ranging fallback ran");
        let ranging_candidates =
            result.candidates.len() - result.candidates.iter().filter(gauss).count();
        assert_eq!(
            ranging_candidates,
            config.ranging.max_corrections.min(ranging.samples.len())
        );
        assert_eq!(result.corrections_run, gauss_corrected + ranging_candidates);
        let noted = result
            .candidates
            .iter()
            .filter(|c| c.correction_note.is_some())
            .count();
        assert_eq!(noted, result.corrections_run);
        assert!(result
            .candidates
            .iter()
            .all(|c| !c.converged && c.corrected_state.is_none()));
        let reason = result.not_found_reason.unwrap();
        assert!(reason.contains("none reached the whole arc"), "{reason}");
        assert!(reason.contains("all roots failed correction"), "{reason}");
        assert!(reason.contains("stalled at stage 1"), "{reason}");
    }

    #[test]
    fn corrected_acceptance_corrects_every_root_of_a_converged_triplet() {
        // The late rows of this geometry give three Gauss roots per triplet
        // (Charlier's regime): every root of each converged triplet is
        // corrected and the accepted one is the lowest final reduced chi2
        // and the true orbit, whatever its root index.
        let rows: Vec<usize> = (6..NUM_OBS).collect();
        let observed =
            super::super::od::take_coordinate_batch(&synthetic_observations(None), &rows).unwrap();
        let observers = super::super::od::take_observer_batch(&observers(), &rows).unwrap();
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let result = run_iod_with(
            &predictor,
            &observed,
            &observers,
            &corrected_iod_config(50.0),
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let accepted = result
            .accepted_candidate()
            .unwrap_or_else(|| panic!("{:?}", result.not_found_reason));
        let siblings: Vec<&super::super::od::IodCandidate> = result
            .candidates
            .iter()
            .filter(|c| c.window == accepted.window && c.triplet == accepted.triplet)
            .collect();
        assert!(
            siblings.len() >= 2,
            "fixture no longer yields multiple roots"
        );
        assert!(siblings.iter().all(|c| c.root_count == siblings.len()));
        for sibling in &siblings {
            assert!(
                sibling.stages > 0,
                "root {} not corrected",
                sibling.root_index
            );
        }
        let rchi2 = accepted.postfit_reduced_chi2.unwrap();
        // Ranked on the whole-arc score (ties within a relative 1e-6).
        let score = accepted.whole_arc_score.unwrap();
        for candidate in result.candidates.iter().filter(|c| c.converged) {
            assert!(candidate.whole_arc_score.unwrap() >= score * (1.0 - 1e-6));
        }
        // The accepted orbit is the true one (within the scatter of six
        // observations), not the root-order first.
        let mu = crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap();
        let truth = crate::propagate_2body_row::<f64>(
            TRUTH_STATE,
            accepted.epoch.mjd() - 60_000.0,
            mu,
            1000,
            1e-14,
        );
        let state = accepted.corrected_state.unwrap();
        assert!(max_abs_diff(&state[..3], &truth[..3]) < 1e-3);
        // One root converges to a separate, worse local minimum (about
        // 0.4 au away, reduced chi2 3.7 against 0.03): distinct, but its
        // chi2 margin (about 22 on 6 dof) exceeds the 3-sigma 20.06, so the
        // IOD is not ambiguous.
        let distinct: Vec<_> = siblings
            .iter()
            .filter(|c| {
                c.converged && max_abs_diff(&c.corrected_state.unwrap()[..3], &state[..3]) > 0.1
            })
            .collect();
        assert_eq!(distinct.len(), 1);
        assert!(distinct[0].postfit_reduced_chi2.unwrap() > rchi2);
        assert!(!result.ambiguous(), "{:?}", result.ambiguous_with);

        // Force the accepted root's metadata to differ from its siblings and
        // make the first sibling fail in-band. The second sibling must still
        // be corrected successfully. This directly kills both the old
        // velocity-method filter and an early-return-on-sibling-failure.
        let primary = result.accepted.unwrap();
        let mut fresh = super::super::od::iod_candidate_pool(
            &predictor,
            &observed,
            &observers,
            &corrected_iod_config(50.0),
        )
        .unwrap();
        let mut group: Vec<usize> = fresh
            .candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                (candidate.triplet == fresh.candidates[primary].triplet
                    && candidate.window == fresh.candidates[primary].window)
                    .then_some(index)
            })
            .collect();
        group.sort_by_key(|&index| fresh.candidates[index].root_index);
        let siblings_after_primary: Vec<usize> = group
            .iter()
            .copied()
            .filter(|&index| index != primary)
            .collect();
        assert!(siblings_after_primary.len() >= 2);
        let failed_sibling = siblings_after_primary[0];
        let later_sibling = siblings_after_primary[1];
        fresh.candidates[primary].velocity_method = super::super::od::VelocityMethod::Gibbs;
        for &index in &siblings_after_primary {
            fresh.candidates[index].velocity_method =
                super::super::od::VelocityMethod::HerrickGibbs;
        }

        struct FailOnceAtState<'a> {
            inner: &'a dyn SphericalPredictor,
            target: [f64; 6],
            failed: std::cell::Cell<bool>,
        }
        impl SphericalPredictor for FailOnceAtState<'_> {
            fn predict_spherical(
                &self,
                states: &[[f64; 6]],
                geometry: &OrbitGeometry,
                observers: &ObserverBatch,
            ) -> PropagationResultValue<Result<Vec<f64>, String>> {
                if !self.failed.get()
                    && states.len() == 1
                    && max_abs_diff(&states[0], &self.target) < 1e-14
                {
                    self.failed.set(true);
                    return Ok(Err("forced sibling row failure".to_string()));
                }
                self.inner.predict_spherical(states, geometry, observers)
            }

            fn backend(&self) -> crate::propagation::BackendIdentity {
                self.inner.backend()
            }
        }
        let failing = FailOnceAtState {
            inner: &predictor,
            target: fresh.candidates[failed_sibling].state,
            failed: std::cell::Cell::new(false),
        };
        correct_candidates(
            &failing,
            &observed,
            &observers,
            &corrected_iod_config(50.0),
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
            &mut fresh,
            &[primary],
        )
        .unwrap();
        assert!(failing.failed.get());
        assert!(fresh.candidates[primary].converged);
        assert!(fresh.candidates[failed_sibling]
            .correction_note
            .as_deref()
            .is_some_and(|note| note.contains("forced sibling row failure")));
        assert!(fresh.candidate_failures.iter().any(|failure| {
            failure.triplet == fresh.candidates[failed_sibling].triplet
                && failure.root_index == fresh.candidates[failed_sibling].root_index
                && failure.message.contains("forced sibling row failure")
        }));
        assert!(
            fresh.candidates[later_sibling].converged,
            "later sibling was not continued: {:?}",
            fresh.candidates[later_sibling].correction_note
        );
    }

    #[test]
    fn full_od_is_run_iod_then_refine_on_the_typed_seed() {
        // The review's contract: run_od / full_od is the composition of the
        // run_iod and refine stages, with the IOD's typed seed (exact epoch).
        let observed = synthetic_observations(Some(3));
        let (propagator, options) = two_body();
        let config = FullOdConfig {
            iod: super::super::od::IodConfig {
                contamination_percentage: 20.0,
                ..corrected_iod_config(50.0)
            },
            refinement: RefinementConfig::default(),
        };
        let fit = match config.refinement {
            RefinementConfig::Cmc2003(loop_config) => loop_config.fit,
            RefinementConfig::WorstResidual(loop_config) => loop_config.fit,
        };
        let iod = run_iod_barycentric(
            &propagator,
            &observed,
            &observers(),
            &config.iod,
            &fit,
            &options,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        // The generic stage honours corrected acceptance.
        let accepted = iod.accepted_candidate().unwrap();
        assert!(accepted.converged && accepted.corrected_state.is_some());
        let seed = iod.seed_orbit().unwrap().unwrap();
        let seed_times = seed.coordinates.times.as_ref().unwrap();
        assert_eq!(seed_times.scale, accepted.epoch_scale);
        assert_eq!(seed_times.epochs[0], accepted.epoch);
        let refined = refine_barycentric(
            &propagator,
            &seed,
            &observed,
            &observers(),
            &config.refinement,
            &options,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let composed = full_od_barycentric(
            &propagator,
            &propagator,
            &observed,
            &observers(),
            &config,
            &options,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let full = composed.refined.unwrap();
        assert_eq!(full.fit.state, refined.fit.state);
        assert_eq!(full.fit.evaluation.outlier, refined.fit.evaluation.outlier);
        assert_eq!(full.n_rejected, refined.n_rejected);
    }

    struct StageCountingPredictor<'a> {
        stage: &'static str,
        inner: &'a dyn SphericalPredictor,
        calls: std::cell::Cell<usize>,
        states: std::cell::Cell<usize>,
    }

    impl StageCountingPredictor<'_> {
        fn reset(&self) {
            self.calls.set(0);
            self.states.set(0);
        }
    }

    impl SphericalPredictor for StageCountingPredictor<'_> {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            self.calls.set(self.calls.get() + 1);
            self.states.set(self.states.get() + states.len());
            self.inner.predict_spherical(states, geometry, observers)
        }

        fn supports_stm(&self) -> bool {
            self.inner.supports_stm()
        }

        fn propagate_stm(
            &self,
            state: [f64; 6],
            non_gravitational_parameters: Option<crate::NonGravitationalParametersRow>,
            epoch_tdb: crate::Epoch,
            times_tdb: &[crate::Epoch],
        ) -> PropagationResultValue<Vec<super::super::StmRow>> {
            self.inner
                .propagate_stm(state, non_gravitational_parameters, epoch_tdb, times_tdb)
        }

        fn backend(&self) -> super::super::BackendIdentity {
            let mut identity = self.inner.backend();
            identity.name = format!("test.stage.{}", self.stage);
            identity
        }
    }

    #[test]
    fn iod_refinement_full_od_and_run_od_use_only_their_configured_stage_predictors() {
        let observed = synthetic_observations_spaced(1, Some(3));
        let observers = observers_spaced(1);
        let (propagator, options) = two_body();
        let inner = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let iod_predictor = StageCountingPredictor {
            stage: "iod",
            inner: &inner,
            calls: std::cell::Cell::new(0),
            states: std::cell::Cell::new(0),
        };
        let refinement_predictor = StageCountingPredictor {
            stage: "refinement",
            inner: &inner,
            calls: std::cell::Cell::new(0),
            states: std::cell::Cell::new(0),
        };
        let config = FullOdConfig {
            iod: super::super::od::IodConfig {
                max_triplets: Some(3),
                contamination_percentage: 20.0,
                ..corrected_iod_config(50.0)
            },
            refinement: RefinementConfig::default(),
        };
        let fit = match config.refinement {
            RefinementConfig::Cmc2003(loop_config) => loop_config.fit,
            RefinementConfig::WorstResidual(loop_config) => loop_config.fit,
        };
        assert_eq!(iod_predictor.backend().name, "test.stage.iod");
        assert_eq!(refinement_predictor.backend().name, "test.stage.refinement");

        let iod = run_iod_with(
            &iod_predictor,
            &observed,
            &observers,
            &config.iod,
            &fit,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(iod_predictor.calls.get() > 0);
        assert_eq!(refinement_predictor.calls.get(), 0);
        assert!(iod.triplets_tried <= 3);
        let seed = iod
            .seed_orbit()
            .unwrap()
            .expect("configured IOD found a seed");
        let iod_evidence = (iod_predictor.calls.get(), iod_predictor.states.get());

        let refined = refine_with(
            &refinement_predictor,
            &seed,
            &observed,
            &observers,
            &config.refinement,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(refined.fit.converged);
        assert_eq!(
            (iod_predictor.calls.get(), iod_predictor.states.get()),
            iod_evidence
        );
        let refinement_evidence = (
            refinement_predictor.calls.get(),
            refinement_predictor.states.get(),
        );
        assert!(refinement_evidence.0 > 0);
        assert_ne!(iod_evidence, refinement_evidence);

        iod_predictor.reset();
        refinement_predictor.reset();
        let full = full_od_with(
            &iod_predictor,
            &refinement_predictor,
            &observed,
            &observers,
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(full.refined.is_some());
        assert_eq!(full.iod.triplets_tried, iod.triplets_tried);
        assert_eq!(full.iod.candidates.len(), iod.candidates.len());
        assert_eq!(
            (iod_predictor.calls.get(), iod_predictor.states.get()),
            iod_evidence
        );
        assert_eq!(
            (
                refinement_predictor.calls.get(),
                refinement_predictor.states.get(),
            ),
            refinement_evidence
        );
        assert_eq!(full.refined.as_ref().unwrap().fit.state, refined.fit.state);

        iod_predictor.reset();
        refinement_predictor.reset();
        let obs_code: Vec<String> = observers.code.iter().map(|code| code.0.clone()).collect();
        let none = vec![None; observed.len()];
        let run = run_od_with(
            &iod_predictor,
            &refinement_predictor,
            &observed,
            &observers,
            &obs_code,
            &none,
            &none,
            Vec::new(),
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(!run.models_changed);
        assert!(run.full_od.refined.is_some());
        assert_eq!(run.full_od.iod.triplets_tried, iod.triplets_tried);
        assert_eq!(
            (iod_predictor.calls.get(), iod_predictor.states.get()),
            iod_evidence
        );
        assert_eq!(
            (
                refinement_predictor.calls.get(),
                refinement_predictor.states.get(),
            ),
            refinement_evidence
        );
        assert_eq!(
            run.full_od.refined.as_ref().unwrap().fit.state,
            refined.fit.state
        );
    }

    #[test]
    fn ambiguity_needs_a_distinct_solution_with_a_comparable_fit() {
        let observed = synthetic_observations(None);
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let result = run_iod_with(
            &predictor,
            &observed,
            &observers(),
            &corrected_iod_config(50.0),
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let best = result.accepted_candidate().unwrap().clone();
        assert!(
            best.corrected_covariance.is_some(),
            "a reportable corrected-IOD covariance remains available to Rust callers"
        );
        let covariance = best.ambiguity_covariance.unwrap();
        let sigma_x = covariance[0].sqrt();
        assert!(sigma_x.is_finite() && sigma_x > 0.0);
        let shifted = |k: f64, score: f64| {
            let mut other = best.clone();
            let mut state = other.corrected_state.unwrap();
            state[0] += k * sigma_x;
            other.corrected_state = Some(state);
            other.whole_arc_score = Some(score);
            other
        };
        let rchi2 = best.whole_arc_score.unwrap();
        // The same orbit is not a second solution.
        assert!(!is_ambiguous_with(&best, &best));
        // Far outside the 3-sigma region, an equally good fit: ambiguous.
        assert!(is_ambiguous_with(&best, &shifted(1e3, rchi2)));
        // Far outside, but the fit is clearly worse: not ambiguous.
        assert!(!is_ambiguous_with(&best, &shifted(1e3, rchi2 + 10.0)));
        // Without a usable covariance the solutions are not compared.
        let mut blind = best.clone();
        blind.ambiguity_covariance = Some([f64::NAN; 36]);
        assert!(!is_ambiguous_with(&blind, &shifted(1e3, rchi2)));
    }

    #[test]
    fn rank_floor_covariance_is_internal_for_ambiguity_only() {
        let mut covariance = [0.0; 36];
        for axis in 0..6 {
            covariance[axis * 6 + axis] = 1.0;
        }
        covariance[1] = 1.0 - 2.0 * f64::EPSILON;
        covariance[6] = covariance[1];
        assert!(covariance_correlation_spectrum(&covariance).is_ok());
        assert!(covariance_validity(&covariance).is_err());
        assert!(reportable_iod_covariance(&covariance).is_none());
        assert_eq!(ambiguity_iod_covariance(&covariance), Some(covariance));

        covariance[0] = -1.0;
        assert!(ambiguity_iod_covariance(&covariance).is_none());
        covariance[0] = f64::NAN;
        assert!(ambiguity_iod_covariance(&covariance).is_none());
    }

    #[test]
    fn corrected_acceptance_is_deterministic() {
        let observed = synthetic_observations(Some(3));
        let (propagator, options) = two_body();
        let predictor = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &NoopProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let run = || {
            run_iod_with(
                &predictor,
                &observed,
                &observers(),
                &corrected_iod_config(50.0),
                &WhitenedFitConfig::default(),
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap()
        };
        let (a, b) = (run(), run());
        assert_eq!(a.accepted, b.accepted);
        assert_eq!(a.corrections_run, b.corrections_run);
        for (x, y) in a.candidates.iter().zip(&b.candidates) {
            assert_eq!(x.corrected_state, y.corrected_state);
            assert_eq!(x.stages, y.stages);
        }
    }

    #[test]
    fn full_od_runs_iod_then_cmc2003_refinement() {
        let observed = synthetic_observations_spaced(1, Some(3));
        let output = full_od_barycentric(
            &TwoBodyPropagator::default(),
            &TwoBodyPropagator::default(),
            &observed,
            &observers_spaced(1),
            &full_od_config(RefinementConfig::Cmc2003(Cmc2003FitConfig::default())),
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(output.iod.found());
        assert_eq!(output.iod.epoch_scale, TimeScale::Tdb);
        let accepted = output.iod.accepted_candidate().expect("accepted candidate");
        assert!(accepted.accepted && accepted.evaluation.is_some());
        assert_eq!(output.iod.candidates.len(), output.iod.candidates_evaluated);
        let iod_epoch = accepted.epoch;
        assert_eq!(iod_epoch, super::super::od::mjd_epoch(iod_epoch.mjd()));
        let refined = output.refined.expect("refined orbit");
        assert!(refined.fit.converged);
        assert!(refined.fit.evaluation.outlier[3]);
        assert_eq!(refined.n_rejected, 1);
        assert!(refined.fit.evaluation.reduced_chi2 < 2.0);
        // The refined orbit sits at the IOD epoch: compare with the truth
        // propagated there.
        let mu = crate::types::origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap();
        let truth_at_epoch = crate::propagate_2body_row::<f64>(
            TRUTH_STATE,
            iod_epoch.mjd() - 60_000.0,
            mu,
            1000,
            1e-14,
        );
        // An 11-day arc constrains range loosely: check the truth lies inside
        // the fit covariance (6-dof Mahalanobis) besides a coarse bound.
        let error = max_abs_diff(&refined.fit.state[..3], &truth_at_epoch[..3]);
        assert!(error < 1e-3, "{error}");
        let inverse = inverse_6x6(&covariance_matrix(&refined.fit.covariance)).unwrap();
        let mut mahalanobis = 0.0;
        for i in 0..6 {
            for j in 0..6 {
                mahalanobis += (refined.fit.state[i] - truth_at_epoch[i])
                    * inverse[i * 6 + j]
                    * (refined.fit.state[j] - truth_at_epoch[j]);
            }
        }
        assert!(mahalanobis < 40.0, "{mahalanobis}");
    }

    #[test]
    fn run_od_applies_models_and_records_provenance() {
        let observed = synthetic_observations_spaced(1, None);
        let obs_code: Vec<String> = (0..NUM_OBS).map(|_| "T00".to_string()).collect();
        let band = vec![None; NUM_OBS];
        let astcat = vec![None; NUM_OBS];
        let models: Vec<Box<dyn ObservationUncertaintyModel>> = vec![Box::new(
            crate::observation_uncertainty::NightBatchDeweightingModel::new(2).unwrap(),
        )];
        let output = run_od_barycentric(
            &TwoBodyPropagator::default(),
            &TwoBodyPropagator::default(),
            &observed,
            &observers_spaced(1),
            &obs_code,
            &band,
            &astcat,
            models,
            &full_od_config(RefinementConfig::WorstResidual(
                IterativeFitConfig::default(),
            )),
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        // Positions never move; every same-night batch here is one observation
        // per night, so the deweighting is a no-op and the used sigmas equal
        // the originals.
        assert_eq!(output.original.lon, output.used.lon);
        assert_eq!(output.original.sigma_lon.len(), NUM_OBS);
        assert!(output
            .original
            .sigma_lon
            .iter()
            .all(|s| (s - SIGMA_DEG).abs() < 1e-15));
        assert!(!output.models_changed);
        let refined = output.full_od.refined.expect("refined orbit");
        assert!(refined.fit.converged);
        assert_eq!(refined.passes, 1);
    }

    /// `synthetic_observations_spaced(1, None)` without any covariance.
    fn observations_without_covariance() -> CoordinateBatch {
        let observed = synthetic_observations_spaced(1, None);
        CoordinateBatch::new(
            observed.values.clone(),
            observed.frame,
            observed.origins.clone(),
            observed.times.clone(),
            None,
        )
        .unwrap()
    }

    fn run_od_without_covariance(
        models: Vec<Box<dyn ObservationUncertaintyModel>>,
    ) -> PropagationResultValue<RunOdOutput> {
        let obs_code: Vec<String> = (0..NUM_OBS).map(|_| "T00".to_string()).collect();
        run_od_barycentric(
            &TwoBodyPropagator::default(),
            &TwoBodyPropagator::default(),
            &observations_without_covariance(),
            &observers_spaced(1),
            &obs_code,
            &vec![None; NUM_OBS],
            &vec![None; NUM_OBS],
            models,
            &full_od_config(RefinementConfig::Cmc2003(Cmc2003FitConfig::default())),
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
    }

    #[test]
    fn run_od_lets_a_sigma_model_fill_a_missing_covariance() {
        let lookup = crate::veres2017::VeresSigmaLookup::new(
            &[crate::veres2017::VeresSigmaRow {
                obs_code: None,
                astcat: None,
                sigma_ra_arcsec: SIGMA_DEG * 3600.0,
                sigma_dec_arcsec: SIGMA_DEG * 3600.0,
            }],
            None,
        )
        .unwrap();
        let output =
            run_od_without_covariance(vec![Box::new(crate::veres2017::SigmaFillModel { lookup })])
                .unwrap();
        assert!(output.models_changed);
        assert!(output.original.sigma_lon.iter().all(|s| s.is_nan()));
        assert!(output.original.sigma_lat.iter().all(|s| s.is_nan()));
        assert!(output
            .used
            .sigma_lat
            .iter()
            .all(|s| (s - SIGMA_DEG).abs() < 1e-15));
        assert!(output.used.sigma_lon.iter().all(|s| s.is_finite()));
        assert!(output.full_od.iod.found());
        let refined = output.full_od.refined.expect("refined orbit");
        assert!(refined.fit.converged);
        assert_eq!(refined.n_rejected, 0);
    }

    #[test]
    fn run_od_without_covariance_or_a_filling_model_is_refused() {
        match run_od_without_covariance(Vec::new()) {
            Err(PropagationError::InvalidRequest(message)) => {
                assert!(message.contains("SigmaFillModel"), "{message}");
                assert!(message.contains("12 of 12"), "{message}");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
        // A model that does not touch the covariance does not help either.
        let models: Vec<Box<dyn ObservationUncertaintyModel>> = vec![Box::new(
            crate::observation_uncertainty::NightBatchDeweightingModel::new(2).unwrap(),
        )];
        assert!(matches!(
            run_od_without_covariance(models),
            Err(PropagationError::InvalidRequest(_))
        ));
    }

    #[test]
    fn jacobi_eigen_decomposition_reconstructs_the_matrix() {
        let a = [
            [4.0, 1.0, 0.5, 0.0, 0.2, 0.1],
            [1.0, 3.0, 0.3, 0.1, 0.0, 0.2],
            [0.5, 0.3, 2.0, 0.4, 0.1, 0.0],
            [0.0, 0.1, 0.4, 1.5, 0.3, 0.1],
            [0.2, 0.0, 0.1, 0.3, 1.0, 0.2],
            [0.1, 0.2, 0.0, 0.1, 0.2, 0.5],
        ];
        let (values, vectors) = symmetric_eigen_6x6(&a);
        for i in 0..6 {
            for j in 0..6 {
                let mut reconstructed = 0.0;
                for k in 0..6 {
                    reconstructed += vectors[i][k] * values[k] * vectors[j][k];
                }
                assert!((reconstructed - a[i][j]).abs() < 1e-12);
            }
        }
    }
}
