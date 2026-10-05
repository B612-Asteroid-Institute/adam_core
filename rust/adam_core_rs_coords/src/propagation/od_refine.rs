//! Backend-generic whitened-residual differential correction (the Rust
//! orchestration behind `adam_core.orbit_determination.fit_least_squares`).
//!
//! [`fit_orbit_whitened_barycentric`] runs the complete public work unit in
//! one crossing, generic over the [`Propagator`] trait like the drivers in
//! `od.rs`: residuals are the whitened (cos-lat corrected, inverse-Cholesky
//! scaled) angular residuals predicted through the backend-generic
//! barycentric ephemeris workflow; the Jacobian is either the exact 2-body
//! forward-mode autodiff Jacobian (`whitened_2body_jacobian`, the analytic
//! default), a central difference through the propagator, or scipy's
//! 2-point forward difference; the loss is linear or Huber (iteratively
//! reweighted, scipy's `least_squares` scaling convention); the covariance is
//! `inv(JᵀJ)` of the robust-cost Jacobian, validated by the weak-direction
//! probe with the central-difference fallback; and the final
//! `evaluate_orbits`-style pass over the full observation set is fused in.
//!
//! The optimizer is Levenberg-Marquardt with Marquardt (`x_scale="jac"`)
//! scaling and scipy's `ftol` / `xtol` / `gtol` stopping rules. It converges
//! to the same minimum as scipy's trust-region-reflective solver to solver
//! tolerance, not to bit-identical iterates; the covariance, the residuals
//! and every reported statistic are functions of the solution only.

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
    /// Exact 2-body state-transition Jacobian by forward-mode autodiff.
    Analytic,
    /// Central differences through the full residual pipeline.
    Central,
    /// scipy's forward-difference `"2-point"` Jacobian.
    TwoPoint,
}

impl JacobianMethod {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "analytic" => Ok(Self::Analytic),
            "central" => Ok(Self::Central),
            "2-point" => Ok(Self::TwoPoint),
            _ => Err(format!(
                "jacobian must be one of 'analytic', 'central', '2-point'; got {value:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Analytic => "analytic",
            Self::Central => "central",
            Self::TwoPoint => "2-point",
        }
    }
}

/// Settings of one whitened differential correction. Defaults mirror
/// `fit_least_squares` (`xtol = ftol = gtol = 1e-12`, linear loss,
/// `f_scale = 1.345`, analytic Jacobian, covariance validation on).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhitenedFitConfig {
    pub loss: LossType,
    pub f_scale: f64,
    pub jacobian: JacobianMethod,
    pub validate_covariance: bool,
    pub xtol: f64,
    pub ftol: f64,
    pub gtol: f64,
    /// Maximum optimizer iterations, independently of the evaluation budget.
    pub max_iterations: usize,
    /// Optional solver residual-vector evaluation budget. Includes the initial
    /// state, rejected trials and every finite-difference candidate (not just
    /// batch crossings). Final covariance/probe/reporting work is separate.
    pub max_nfev: Option<usize>,
    /// Universal-Kepler / light-time settings of the analytic 2-body model.
    pub two_body: TwoBodyModelConfig,
}

impl Default for WhitenedFitConfig {
    fn default() -> Self {
        Self {
            loss: LossType::Linear,
            f_scale: HUBER_F_SCALE_DEFAULT,
            jacobian: JacobianMethod::Analytic,
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
    pub iterations: usize,
    /// Whether a convergence stopping rule was met within both solver limits.
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
struct WhitenedProblem<'a> {
    predictor: &'a dyn SphericalPredictor,
    geometry: OrbitGeometry,
    epoch_mjd_tdb: f64,
    observed_flat: Vec<f64>,
    observers: ObserverBatch,
    whiteners: Vec<[f64; 4]>,
    analytic_terms: Option<TwoBodyJacobianTerms>,
    two_body: TwoBodyModelConfig,
    m: usize,
}

impl<'a> WhitenedProblem<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new<T: OriginTranslationProvider>(
        predictor: &'a dyn SphericalPredictor,
        orbit: &OrbitBatch,
        observed: &CoordinateBatch,
        observers: &ObserverBatch,
        config: &WhitenedFitConfig,
        provider: &dyn TimeScaleProvider,
        translation_provider: &T,
    ) -> PropagationResultValue<Self> {
        let m = observed.len();
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

        let analytic_terms = if config.jacobian == JacobianMethod::Analytic {
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
            two_body: config.two_body,
            m,
        })
    }

    /// Whitened residual vectors (`2M` each) of `states`, predicted in one
    /// ephemeris crossing. Outer `Err` is a request failure; inner `Err` a
    /// per-row numerical failure of the backend.
    fn residuals(
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
            .map_err(PropagationError::Backend)
    }

    /// `(2M, 6)` row-major Jacobian of the whitened residual vector.
    fn jacobian(
        &self,
        method: JacobianMethod,
        state: &[f64; 6],
        base: &[f64],
    ) -> PropagationResultValue<Vec<f64>> {
        match method {
            JacobianMethod::Analytic => {
                let terms = self.analytic_terms.as_ref().ok_or_else(|| {
                    PropagationError::InvalidRequest(
                        "analytic Jacobian terms were not prepared".to_string(),
                    )
                })?;
                whitened_2body_jacobian(*state, self.epoch_mjd_tdb, terms, &self.two_body)
                    .map_err(PropagationError::Backend)
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
                    .map_err(PropagationError::Backend)?;
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
                    .map_err(PropagationError::Backend)?;
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
    inverse_6x6(&a).filter(|cov| cov.iter().all(|v| v.is_finite()))
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

struct Solution {
    state: [f64; 6],
    residuals: Vec<f64>,
    cost: f64,
    nfev: usize,
    status: i64,
}

/// Levenberg-Marquardt on the robust cost with Marquardt scaling.
fn solve(
    problem: &WhitenedProblem<'_>,
    initial_state: [f64; 6],
    config: &WhitenedFitConfig,
) -> PropagationResultValue<Solution> {
    let loss = config.loss;
    let f_scale = config.f_scale;
    let mut state = initial_state;
    let mut residuals = problem.residuals_one(&state)?;
    let mut cost = robust_cost(&residuals, loss, f_scale);
    let mut nfev = 1_usize;
    let mut status = 0_i64;
    let mut lambda = 1e-3_f64;
    let max_nfev = config.max_nfev.unwrap_or(usize::MAX);
    let jacobian_nfev = match config.jacobian {
        JacobianMethod::Analytic => 0,
        JacobianMethod::Central => 12,
        JacobianMethod::TwoPoint => 6,
    };

    'iterations: for _iteration in 0..config.max_iterations {
        // Numerical Jacobians need a complete batch. Do not start one that
        // would exceed the remaining budget, even if a few evaluations remain.
        if nfev >= max_nfev || jacobian_nfev > max_nfev - nfev {
            break;
        }
        nfev += jacobian_nfev;
        let jac = problem.jacobian(config.jacobian, &state, &residuals)?;
        let (jac_weighted, f_weighted) = irls_problem(&jac, &residuals, loss, f_scale);
        let (a, g) = normal_equations(&jac_weighted, &f_weighted);
        // `g = Jᵀ psi(r)` is the gradient of half the robust cost; scipy's
        // gtol is on its infinity norm after x_scale scaling, which only
        // tightens the test.
        let g_norm = g.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if g_norm <= config.gtol {
            status = 1;
            break;
        }
        let mut d = [0.0_f64; 6];
        for k in 0..6 {
            d[k] = a[k][k].sqrt().max(1e-300);
        }

        let mut accepted = false;
        let mut last_step_norm = f64::INFINITY;
        for _attempt in 0..24 {
            if nfev >= max_nfev {
                // Budget exhaustion is not a stalled step or xtol convergence.
                break 'iterations;
            }
            let mut damped = a;
            for k in 0..6 {
                damped[k][k] += lambda * d[k] * d[k];
            }
            let mut rhs = [0.0_f64; 6];
            for k in 0..6 {
                rhs[k] = -g[k];
            }
            let delta = match crate::orbit_least_squares::solve_6x6(&damped, &rhs) {
                Some(delta) => delta,
                None => {
                    lambda *= 10.0;
                    continue;
                }
            };
            let mut trial = state;
            for k in 0..6 {
                trial[k] += delta[k];
            }
            last_step_norm = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
            nfev += 1;
            let trial_residuals = match problem.residuals(&[trial])? {
                Ok(mut batch) => batch.swap_remove(0),
                Err(_) => {
                    // A failed prediction (e.g. light time) rejects the step.
                    lambda *= 10.0;
                    continue;
                }
            };
            let trial_cost = robust_cost(&trial_residuals, loss, f_scale);
            if trial_cost.is_finite() && trial_cost <= cost {
                let step_norm = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
                let state_norm = state.iter().map(|v| v * v).sum::<f64>().sqrt();
                let cost_drop = cost - trial_cost;
                let ftol_met = cost_drop <= config.ftol * cost.max(f64::MIN_POSITIVE);
                let xtol_met = step_norm <= config.xtol * (config.xtol + state_norm);
                state = trial;
                residuals = trial_residuals;
                cost = trial_cost;
                lambda = (lambda / 3.0).max(1e-15);
                accepted = true;
                if ftol_met && xtol_met {
                    status = 4;
                } else if ftol_met {
                    status = 2;
                } else if xtol_met {
                    status = 3;
                }
                break;
            }
            lambda *= 10.0;
        }
        if !accepted {
            if nfev >= max_nfev {
                // The final allowed trial can also be the final damping
                // attempt; it must have the same budget status as an earlier
                // exhausted attempt, without falling through to xtol below.
                break;
            }
            // No damping produced a decrease. If the last (heavily damped)
            // step was already below the xtol resolution the iterate is a
            // numerical minimum; otherwise the solver stalled.
            let state_norm = state.iter().map(|v| v * v).sum::<f64>().sqrt();
            status = if last_step_norm <= config.xtol * (config.xtol + state_norm) {
                3
            } else {
                -1
            };
            break;
        }
        if status != 0 {
            break;
        }
    }

    Ok(Solution {
        state,
        residuals,
        cost,
        nfev,
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
    let delta = match weak_direction_delta_cost(problem, state, &covariance, cost_solution, config)
        .unwrap_or_else(|err| Err(err.to_string()))
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
        JacobianMethod::Central => {
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

    let solution = solve(&problem, problem.geometry.state, config)?;
    let mut warnings = Vec::new();

    let jac = problem.jacobian(config.jacobian, &solution.state, &solution.residuals)?;
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
            config.jacobian,
            &solution.state,
            &solution.residuals,
            solution.cost,
            config,
            &mut warnings,
        )?;
    }

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

    Ok(WhitenedFitOutput {
        state: solution.state,
        covariance,
        iterations: solution.nfev,
        converged: solution.status > 0,
        status_code: solution.status,
        cost: solution.cost,
        fit_chi2,
        weights,
        residuals_whitened: solution.residuals,
        evaluation,
        warnings,
    })
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
    let delta_cost =
        match weak_direction_delta_cost(&problem, &state, &covariance, cost, &probe_config) {
            Ok(Ok(delta)) => delta,
            _ => f64::NAN,
        };
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
/// non-gravitational parameter tables are not carried, matching the
/// candidate batches of the other drivers.
fn orbit_with_state(orbit: &OrbitBatch, state: [f64; 6]) -> PropagationResultValue<OrbitBatch> {
    let coordinates = CoordinateBatch::new(
        crate::CoordinateValues::Cartesian(vec![state]),
        orbit.coordinates.frame,
        orbit.coordinates.origins.clone(),
        orbit.coordinates.times.clone(),
        None,
    )?;
    OrbitBatch::new(orbit.orbit_id.clone(), orbit.object_id.clone(), coordinates)
        .map_err(Into::into)
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
    let n = observed.len();
    if observers.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed and observers must have equal length".to_string(),
        ));
    }
    let times_mjd = super::od::observed_times_mjd(observed)?;
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
            fit: WhitenedFitConfig::default(),
        }
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
/// residual covariance (analytic 2-body Jacobian at the fitted state),
/// warm-starting each pass from the previous solution.
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
    let geometry = OrbitGeometry::from_orbit(orbit)?;
    let epoch_mjd_tdb = crate::TimeArray::new(geometry.scale, vec![geometry.epoch])?
        .rescale_with_provider(TimeScale::Tdb, provider)?
        .mjd_values()[0];
    let observed_flat = spherical_flat(observed, "observed")?;
    let observed_cov = observed_covariance_flat(observed)?;
    let lat_deg: Vec<f64> = (0..n).map(|row| observed_flat[row * 6 + 2]).collect();
    let whiteners = observation_whitening_matrices(&observed_cov, &lat_deg)
        .map_err(|err| PropagationError::InvalidRequest(err.to_string()))?;
    let terms = analytic_jacobian_terms(
        &geometry,
        observers,
        &lat_deg,
        &whiteners,
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

    let run_fit =
        |seed: &OrbitBatch, selected: &[bool]| -> PropagationResultValue<WhitenedFitOutput> {
            let ignore: Vec<bool> = selected.iter().map(|&keep| !keep).collect();
            fit_orbit_whitened_with(
                predictor,
                seed,
                observed,
                observers,
                &ignore,
                &config.fit,
                provider,
                translation_provider,
            )
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
        let mut output = run_fit(&seed, &selected)?;
        warnings.append(&mut output.warnings);
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
        let residuals = whiten_residual_pairs(&whiteners, &residual_pairs);
        let jacobian =
            whitened_2body_jacobian(output.state, epoch_mjd_tdb, &terms, &config.fit.two_body)
                .map_err(PropagationError::Backend)?;
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
                min_obs: CMC2003_MIN_OBS,
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
        let mut output = run_fit(&seed, &selected)?;
        warnings.append(&mut output.warnings);
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

/// Settings of the complete native orbit determination: Gauss IOD followed
/// by differential correction with outlier treatment.
#[derive(Debug, Clone, Copy)]
pub struct FullOdConfig {
    pub iod: super::od::IodConfig,
    pub refinement: RefinementConfig,
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
    pub iod: super::od::IodOutput,
    /// Time scale of `iod.epoch_mjd` (that of the observation epochs).
    pub epoch_scale: TimeScale,
    pub refined: Option<RefinedOutput>,
}

/// One-crossing `OrbitFitter.full_od` for the native fitter: the complete
/// Gauss IOD decision loop (`iod_fit_barycentric`) followed by the
/// worst-residual or CMC2003 refinement of the accepted orbit against all
/// observations.
pub fn full_od_barycentric<P, T>(
    propagator: &P,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &FullOdConfig,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<FullOdOutput>
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
    full_od_with(
        &predictor,
        observed,
        observers,
        config,
        provider,
        translation_provider,
    )
}

/// [`full_od_barycentric`] over any [`SphericalPredictor`].
#[allow(clippy::too_many_arguments)]
pub fn full_od_with<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &FullOdConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<FullOdOutput> {
    let epoch_scale = observed
        .times
        .as_ref()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "orbit determination requires observation times".to_string(),
            )
        })?
        .scale;
    let iod = super::od::iod_fit(predictor, observed, observers, &config.iod)?;
    if !iod.found {
        return Ok(FullOdOutput {
            iod,
            epoch_scale,
            refined: None,
        });
    }
    let seed_coordinates = CoordinateBatch::cartesian(
        vec![iod.state],
        Frame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), 1),
        Some(crate::TimeArray::new(
            epoch_scale,
            vec![super::od::mjd_epoch(iod.epoch_mjd)],
        )?),
        None,
    )?;
    let seed = OrbitBatch::new(
        vec![crate::OrbitId("iod-orbit".to_string())],
        vec![None],
        seed_coordinates,
    )?;
    let refined = match &config.refinement {
        RefinementConfig::WorstResidual(loop_config) => {
            let output = iterative_fit_with(
                predictor,
                &seed,
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
                &seed,
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
    };
    Ok(FullOdOutput {
        iod,
        epoch_scale,
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
#[allow(clippy::too_many_arguments)]
pub fn run_od_barycentric<P, T>(
    propagator: &P,
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
    P: Propagator,
    T: OriginTranslationProvider,
{
    let predictor = PropagatorPredictor {
        propagator,
        options,
        provider,
        translation_provider,
    };
    run_od_with(
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

/// [`run_od_barycentric`] over any [`SphericalPredictor`].
#[allow(clippy::too_many_arguments)]
pub fn run_od_with<T: OriginTranslationProvider>(
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
    let n = observed.len();
    if observers.len() != n || obs_code.len() != n || band.len() != n || astcat.len() != n {
        return Err(PropagationError::InvalidRequest(
            "observed, observers, obs_code, band and astcat must have equal length".to_string(),
        ));
    }
    let observed_flat = spherical_flat(observed, "observed")?;
    let covariance = observed_covariance_flat(observed)?;
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

    let used = if models_changed {
        let mut values = observed.values.raw_values().to_vec();
        for (row, value) in values.iter_mut().enumerate() {
            value[1] = astrometry.lon[row];
            value[2] = astrometry.lat[row];
        }
        let source = observed.covariance.as_ref().expect("checked above");
        let mut used_covariance =
            crate::CovarianceBatch::new(n, 6, astrometry.covariance.clone(), source.units.clone())?;
        if let Some(validity) = &source.row_validity {
            used_covariance = used_covariance.with_row_validity(validity.clone())?;
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
            if Some(row) == outlier_row {
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

    struct CountingPredictor<'a> {
        inner: &'a dyn SphericalPredictor,
        batches: std::cell::RefCell<Vec<usize>>,
        reject_trials: bool,
    }

    impl SphericalPredictor for CountingPredictor<'_> {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            self.batches.borrow_mut().push(states.len());
            if self.reject_trials && self.batches.borrow().len() > 1 {
                return Ok(Err("rejected trial prediction".to_string()));
            }
            self.inner.predict_spherical(states, geometry, observers)
        }
    }

    fn solve_counted(config: &WhitenedFitConfig, reject_trials: bool) -> (Solution, Vec<usize>) {
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
            reject_trials,
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
    fn evaluation_budget_counts_candidates_without_overrunning_batches() {
        for jacobian in [
            JacobianMethod::Analytic,
            JacobianMethod::Central,
            JacobianMethod::TwoPoint,
        ] {
            for budget in [1, 2, 6, 7, 8, 12, 13, 14, 25] {
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
                if budget == 1 {
                    assert_eq!(batches, vec![1]);
                    assert_eq!(solution.status, 0);
                    assert_eq!(
                        solution.state,
                        OrbitGeometry::from_orbit(&perturbed_start()).unwrap().state
                    );
                }
                if jacobian == JacobianMethod::Central && budget <= 12 {
                    assert_eq!(batches, vec![1], "no room for a full central Jacobian");
                }
                if jacobian == JacobianMethod::TwoPoint && budget <= 6 {
                    assert_eq!(batches, vec![1], "no room for a full forward Jacobian");
                }
            }
        }
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
            assert!(diff < 1e-9, "{method:?}: {diff}");
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
            },
            refinement,
        }
    }

    #[test]
    fn full_od_runs_iod_then_cmc2003_refinement() {
        let observed = synthetic_observations_spaced(1, Some(3));
        let output = full_od_barycentric(
            &TwoBodyPropagator::default(),
            &observed,
            &observers_spaced(1),
            &full_od_config(RefinementConfig::Cmc2003(Cmc2003FitConfig::default())),
            &ephemeris_options(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(output.iod.found);
        assert_eq!(output.epoch_scale, TimeScale::Tdb);
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
            output.iod.epoch_mjd - 60_000.0,
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
