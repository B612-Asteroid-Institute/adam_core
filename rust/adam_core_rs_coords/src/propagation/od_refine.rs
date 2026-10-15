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
///   (with a warning, and a fresh `max_nfev` / `max_iterations` budget), and
///   the covariance uses `Central` too.
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
/// * **Limits.** `max_iterations` bounds optimizer iterations (one Jacobian
///   each). `max_nfev` bounds scipy's `nfev`: the initial residual
///   evaluation plus every trial step, accepted or rejected; evaluations
///   spent on finite-difference Jacobians are not budget (scipy `trf`
///   semantics, the Python baseline's). The `iterations` output still
///   reports every residual evaluation including Jacobian columns. Reaching
///   either limit returns the current iterate with status 0 unless a
///   stopping rule was met on the last permitted step.
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
    /// (`max_nfev` bounds trial evaluations only; see [`WhitenedFitConfig`].)
    pub iterations: usize,
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
    /// Every residual-vector evaluation, Jacobian columns included.
    nfev: usize,
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
    let solution = solve(problem, initial_state, config)?;
    if solution.status != STATUS_NON_FINITE
        || config.jacobian != JacobianMethod::Analytic
        || !solution.residuals.iter().all(|r| r.is_finite())
    {
        return Ok((solution, config.jacobian));
    }
    warnings.push(
        "The analytic two-body Jacobian was non-finite at the current iterate; the fit \
         continued with central differences through the propagator."
            .to_string(),
    );
    let central = WhitenedFitConfig {
        jacobian: JacobianMethod::Central,
        ..*config
    };
    let retry = solve(problem, solution.state, &central)?;
    Ok((
        Solution {
            nfev: solution.nfev + retry.nfev,
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
    // `nfev` counts every residual evaluation (the legacy `iterations`
    // column); `trials` counts the initial evaluation and each trial step,
    // scipy's `nfev`, which `max_nfev` bounds.
    let mut nfev = 1_usize;
    let mut trials = 1_usize;
    let max_trials = config.max_nfev.unwrap_or(usize::MAX);
    let jacobian_nfev = match config.jacobian {
        JacobianMethod::Analytic => 0,
        JacobianMethod::Central => 12,
        JacobianMethod::TwoPoint => 6,
    };
    let mut status = STATUS_LIMIT;
    // scipy `x_scale="jac"`: column norms, non-decreasing across iterations.
    let mut scale = [0.0_f64; 6];
    let mut radius = 0.0_f64;
    let mut alpha = 0.0_f64;

    'iterations: for iteration in 0..config.max_iterations {
        if trials >= max_trials {
            break;
        }
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
            if trials >= max_trials {
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
            trials += 1;
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
/// any other prediction: on the callback route it carries the Python
/// exception, which the next prediction would otherwise clear.
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
            converged: solution.status > 0,
            status_code: solution.status,
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

    let run_fit = |seed: &OrbitBatch, selected: &[bool]| {
        let ignore: Vec<bool> = selected.iter().map(|&keep| !keep).collect();
        fit_orbit_whitened_impl(
            predictor,
            seed,
            observed,
            observers,
            &ignore,
            &config.fit,
            provider,
            translation_provider,
            Some(&full_problem),
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
        let (mut output, rejection_jacobian) = run_fit(&seed, &selected)?;
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
        let (mut output, _) = run_fit(&seed, &selected)?;
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
) -> PropagationResultValue<Result<WhitenedFitOutput, String>> {
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
        // A failed prediction is local to this candidate.
        Err(
            err @ (PropagationError::Backend(_)
            | PropagationError::BackendProtocol(_)
            | PropagationError::ThreadPool(_)),
        ) => return Ok(Err(format!("prediction failed: {err}"))),
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
    let usable = output.status_code != STATUS_NON_FINITE;
    if !usable || !rchi2.is_finite() || rchi2 > rchi2_threshold {
        return Ok(Err(format!(
            "stage fit on {} observations: reduced chi2 {:.4}, status {}",
            mask.iter().filter(|&&keep| keep).count(),
            rchi2,
            output.status_code
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
        best.corrected_covariance,
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
/// candidate's triplet (Charlier: up to three per triplet) is corrected too
/// before the search moves on or stops, outside the `max_corrections`
/// budget. The accepted candidate is the converged one with the lowest
/// whole-arc score (chi2 over every observation, each clipped at the CMC2003
/// rejection threshold), the lowest index among those within a relative
/// `RCHI2_TIE_RTOL` of it, never the root order. Its orbit is that whole-arc fit at the Gauss epoch, rejected rows
/// reported as outliers. Converged candidates that are distinct solutions
/// fitting about as well (`is_ambiguous_with`) are listed in
/// `ambiguous_with`.
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
    let mut result = super::od::iod_candidate_pool(iod_predictor, observed, observers, config)?;
    if result.candidates.is_empty() {
        return Ok(result);
    }
    let n = observed.len();
    let times = super::od::observed_times_mjd(observed)?;
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
    let mut order: Vec<usize> = (0..result.candidates.len()).collect();
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
                Err(message) => {
                    note = Some(format!("stalled at stage {stages}: {message}"));
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
                        ..Cmc2003FitConfig::default()
                    },
                    provider,
                    translation_provider,
                ) {
                    Ok(rejected) => rejected.fit,
                    Err(
                        err @ (PropagationError::Backend(_)
                        | PropagationError::BackendProtocol(_)
                        | PropagationError::ThreadPool(_)),
                    ) => {
                        note = Some(format!(
                            "whole-arc rejection failed at stage {stages}: {err}"
                        ));
                        break;
                    }
                    Err(err) => return Err(err),
                }
            };
            let rchi2 = output.evaluation.reduced_chi2;
            if output.evaluation.arc_length < config.min_arc_length {
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
                record.corrected_covariance = Some(output.covariance);
                record.removed = (0..n)
                    .filter(|&row| output.evaluation.outlier[row])
                    .collect();
                record.evaluation = Some(output.evaluation);
                converged += 1;
                let (window, triplet, velocity_method) =
                    (record.window, record.triplet, record.velocity_method);
                let mut roots: Vec<usize> = (0..result.candidates.len())
                    .filter(|&other| {
                        let c = &result.candidates[other];
                        other != index
                            && c.stages == 0
                            && c.window == window
                            && c.triplet == triplet
                            && c.velocity_method == velocity_method
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
    Ok(result)
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
    let (state, epoch, epoch_scale) = match iod.accepted_candidate() {
        Some(candidate) => (
            candidate.corrected_state.unwrap_or(candidate.state),
            candidate.epoch,
            candidate.epoch_scale,
        ),
        None => return Ok(FullOdOutput { iod, refined: None }),
    };
    // The seed sits exactly at the IOD epoch (days and nanoseconds), not at
    // a float MJD round trip of it.
    let seed_coordinates = CoordinateBatch::cartesian(
        vec![state],
        Frame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), 1),
        Some(crate::TimeArray::new(epoch_scale, vec![epoch])?),
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
    fn evaluation_budget_bounds_trials_not_jacobian_columns() {
        // `max_nfev` is scipy's `nfev`: the initial evaluation plus every
        // trial step. Finite-difference Jacobian batches (6 or 12 predictions)
        // are reported in `nfev` but are not budget. Zero tolerances never
        // stop the solver, so every run spends its whole budget.
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
                let trials = batches.iter().filter(|&&size| size == 1).count();
                assert_eq!(trials, budget, "{jacobian:?} {budget}: {batches:?}");
                assert_eq!(solution.status, 0, "{jacobian:?} {budget}: {batches:?}");
                if budget == 1 {
                    assert_eq!(batches, vec![1]);
                    assert_eq!(
                        solution.state,
                        OrbitGeometry::from_orbit(&perturbed_start()).unwrap().state
                    );
                } else if columns > 0 {
                    assert!(
                        batches.contains(&columns),
                        "{jacobian:?} {budget}: a Jacobian batch is not budget: {batches:?}"
                    );
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
        let config = WhitenedFitConfig::default();
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
        let nan_problem = WhitenedProblem::new(
            &predictor,
            &perturbed_start(),
            &nan_observed,
            &observers(),
            &config,
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let mut warnings = Vec::new();
        let (solution, method) = solve_with_fallback(
            &nan_problem,
            nan_problem.geometry.state,
            &config,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(solution.status, STATUS_NON_FINITE);
        assert_eq!(method, JacobianMethod::Analytic);
        assert!(warnings.is_empty());
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
        let (solution, batches) =
            solve_counted_mode(&WhitenedFitConfig::default(), PredictorMode::WorseTrials);
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
                assert_eq!(
                    *counting.batches.borrow(),
                    if validate_covariance {
                        vec![1, 12, 2, 1]
                    } else {
                        vec![1, 12, 1]
                    }
                );
            }
        }
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
                velocity_method: super::super::od::VelocityMethod::Gibbs,
                spans: super::super::od::SpanSearchConfig::default(),
                acceptance: super::super::od::IodAcceptance::Prefit,
                correction: super::super::od::CorrectionConfig::default(),
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
        assert_eq!(result.corrections_run, 3.min(result.candidates.len()));
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
            .filter(|c| {
                c.window == accepted.window
                    && c.triplet == accepted.triplet
                    && c.velocity_method == accepted.velocity_method
            })
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
        let covariance = best.corrected_covariance.unwrap();
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
        blind.corrected_covariance = Some([f64::NAN; 36]);
        assert!(!is_ambiguous_with(&blind, &shifted(1e3, rchi2)));
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
