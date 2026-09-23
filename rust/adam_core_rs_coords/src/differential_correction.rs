//! Whitened-residual differential-correction kernels.
//!
//! Rust-canonical numerics behind `adam_core.orbit_determination.differential_correction`
//! on the whitened-residual formulation of `fit_least_squares`:
//!
//! * [`observation_whitening_matrices`] — per-observation inverse Cholesky
//!   factors of the cos(lat)-corrected (lon, lat) covariance block, so the two
//!   whitened components of one observation sum in quadrature to its chi2;
//! * [`whitened_2body_model_angles`] / [`whitened_2body_jacobian`] — the
//!   predicted whitened angles of the 2-body model (heliocentric universal
//!   Kepler propagation from the fit epoch, translation to the barycenter,
//!   topocentric equatorial projection with light-time iteration) and their
//!   exact Jacobian with respect to the epoch state by forward-mode automatic
//!   differentiation over [`Dual<6>`]. This is the observation partials chained
//!   through the 2-body state transition matrix that removes the
//!   finite-difference noise which fabricates confidence in weakly constrained
//!   (line-of-sight) directions;
//! * the robust-loss helpers ([`robust_cost`], [`robust_weights`],
//!   [`robust_jacobian_scale`]) for Huber's M-estimator in the convention of
//!   `scipy.optimize.least_squares`.
//!
//! The optimizer itself (scipy trust-region) and the N-body residual
//! evaluation through a `Propagator` stay in the Python veneer; these kernels
//! are what that veneer used JAX and NumPy for.

use crate::ephemeris::generate_ephemeris_2body_row;
use crate::propagate::propagate_2body_row;
use adam_core_rs_autodiff::{Dual, Scalar};
use std::fmt;

/// Huber transition point in units of whitened (1-sigma) residual components:
/// residuals within +/- f_scale are treated quadratically, larger residuals
/// linearly. 1.345 is the classical constant giving 95% asymptotic efficiency
/// for Gaussian errors (Huber 1981; Huber & Ronchetti 2009, sec. 4.5).
pub const HUBER_F_SCALE_DEFAULT: f64 = 1.345;

/// Robust loss functions supported by `fit_least_squares`. `Linear` is
/// ordinary weighted least squares (the cost is chi2); `Huber` is Huber's
/// M-estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossType {
    Linear,
    Huber,
}

impl LossType {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "linear" => Ok(Self::Linear),
            "huber" => Ok(Self::Huber),
            _ => Err(format!(
                "loss must be one of 'linear', 'huber'; got {value:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Huber => "huber",
        }
    }
}

/// Validate a `(loss, f_scale)` pair exactly like the Python `_validate_loss`.
pub fn validate_loss(loss: &str, f_scale: f64) -> Result<LossType, String> {
    let loss = LossType::parse(loss)?;
    if !f_scale.is_finite() || f_scale <= 0.0 {
        return Err(format!(
            "f_scale must be a positive finite number; got {f_scale:?}"
        ));
    }
    Ok(loss)
}

/// Objective minimized by `fit_least_squares` for the given loss, as a
/// function of the whitened residual components `r`:
///
/// * `Linear`: `sum(r**2)`, i.e. chi2;
/// * `Huber`: `f_scale**2 * sum(rho(r**2 / f_scale**2))` with `rho(z) = z` for
///   `z <= 1` and `rho(z) = 2 sqrt(z) - 1` otherwise, which is twice the
///   `cost` reported by `scipy.optimize.least_squares`.
pub fn robust_cost(residuals: &[f64], loss: LossType, f_scale: f64) -> f64 {
    match loss {
        LossType::Linear => residuals.iter().map(|r| r * r).sum(),
        LossType::Huber => {
            let total: f64 = residuals
                .iter()
                .map(|r| {
                    let z = (r / f_scale).powi(2);
                    if z <= 1.0 {
                        z
                    } else {
                        2.0 * z.sqrt() - 1.0
                    }
                })
                .sum();
            f_scale * f_scale * total
        }
    }
}

/// Iteratively-reweighted-least-squares weight `rho'(z)` of each whitened
/// residual component at the solution: 1 inside the quadratic core and
/// `f_scale / |r|` in the linear tail (Huber's psi(r) / r). All ones for
/// `Linear`.
pub fn robust_weights(residuals: &[f64], loss: LossType, f_scale: f64) -> Vec<f64> {
    match loss {
        LossType::Linear => vec![1.0; residuals.len()],
        LossType::Huber => residuals
            .iter()
            .map(|r| {
                let weight = (f_scale / r.abs()).min(1.0);
                if weight.is_finite() {
                    weight
                } else {
                    1.0
                }
            })
            .collect(),
    }
}

/// Row scaling that turns the Jacobian of the whitened residual vector into
/// the Gauss-Newton Jacobian of the robust cost, following the convention of
/// `scipy.optimize.least_squares` (`sqrt(rho' + 2 rho'' r**2)`, floored at
/// machine epsilon). For `Linear` this is all ones. For `Huber` the second
/// derivative of the loss vanishes in the linear tail, so components beyond
/// `f_scale` contribute (numerically) no curvature.
pub fn robust_jacobian_scale(residuals: &[f64], loss: LossType, f_scale: f64) -> Vec<f64> {
    match loss {
        LossType::Linear => vec![1.0; residuals.len()],
        LossType::Huber => residuals
            .iter()
            .map(|r| {
                if r.abs() <= f_scale {
                    1.0
                } else {
                    f64::EPSILON.sqrt()
                }
            })
            .collect(),
    }
}

/// Failure of [`observation_whitening_matrices`] for one observation row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhiteningError {
    NonFinite { row: usize },
    NotPositiveDefinite { row: usize },
}

impl fmt::Display for WhiteningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { row } => write!(
                f,
                "Observation {row} has non-finite (lon, lat) covariance entries; \
                 least-squares fitting requires finite angular uncertainties for every observation."
            ),
            Self::NotPositiveDefinite { row } => write!(
                f,
                "Observation {row} has a non-positive-definite (lon, lat) covariance block."
            ),
        }
    }
}

impl std::error::Error for WhiteningError {}

/// Per-observation whitening matrices for the angular residuals.
///
/// For each observation the 2x2 (lon, lat) block of the `(N, 36)` row-major
/// coordinate covariance (degrees²) is corrected for cos(latitude) on the
/// longitude axis (matching `Residuals.calculate`) and factored as
/// `C = L L^T`; the returned row-major 2x2 whitener is `L^-1`. Missing (NaN)
/// cross terms are treated as zero, matching `calculate_chi2`.
pub fn observation_whitening_matrices(
    covariances: &[f64],
    lat_deg: &[f64],
) -> Result<Vec<[f64; 4]>, WhiteningError> {
    let n = lat_deg.len();
    assert_eq!(
        covariances.len(),
        n * 36,
        "covariances must have shape (N, 6, 6)"
    );
    let mut whiteners = Vec::with_capacity(n);
    for row in 0..n {
        let block = &covariances[row * 36..(row + 1) * 36];
        let cos_lat = lat_deg[row].to_radians().cos();
        let cross = if block[8].is_finite() { block[8] } else { 0.0 };
        let a = block[7] * (cos_lat * cos_lat);
        let b = cross * cos_lat;
        let d = block[14];
        if !(a.is_finite() && b.is_finite() && d.is_finite()) {
            return Err(WhiteningError::NonFinite { row });
        }
        // Cholesky of [[a, b], [b, d]]: L = [[l11, 0], [l21, l22]].
        if a.is_nan() || a <= 0.0 {
            return Err(WhiteningError::NotPositiveDefinite { row });
        }
        let l11 = a.sqrt();
        let l21 = b / l11;
        let l22_sq = d - l21 * l21;
        if l22_sq.is_nan() || l22_sq <= 0.0 {
            return Err(WhiteningError::NotPositiveDefinite { row });
        }
        let l22 = l22_sq.sqrt();
        whiteners.push([1.0 / l11, 0.0, -l21 / (l11 * l22), 1.0 / l22]);
    }
    Ok(whiteners)
}

/// Whiten `(N, 2)` cos(lat)-corrected (lon, lat) residual pairs into the
/// flat `(2N,)` component vector `r` with `sum(r**2) == sum(chi2)`.
pub fn whiten_residual_pairs(whiteners: &[[f64; 4]], residual_pairs: &[f64]) -> Vec<f64> {
    assert_eq!(residual_pairs.len(), whiteners.len() * 2);
    let mut out = Vec::with_capacity(residual_pairs.len());
    for (w, pair) in whiteners.iter().zip(residual_pairs.chunks_exact(2)) {
        out.push(w[0] * pair[0] + w[1] * pair[1]);
        out.push(w[2] * pair[0] + w[3] * pair[1]);
    }
    out
}

/// Observation-dependent constants of the analytic 2-body Jacobian:
/// observation times (MJD TDB), barycentric ecliptic observer and Sun states
/// (AU, AU/day), cos(latitude) of the observed positions, the whitening
/// factors and the heliocentric / barycentric gravitational parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoBodyJacobianTerms {
    pub times_mjd_tdb: Vec<f64>,
    pub observer_states: Vec<[f64; 6]>,
    pub sun_states: Vec<[f64; 6]>,
    pub cos_lats: Vec<f64>,
    pub whiteners: Vec<[f64; 4]>,
    pub mu_helio: f64,
    pub mu_bary: f64,
}

impl TwoBodyJacobianTerms {
    pub fn len(&self) -> usize {
        self.times_mjd_tdb.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times_mjd_tdb.is_empty()
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.len();
        if self.observer_states.len() != n
            || self.sun_states.len() != n
            || self.cos_lats.len() != n
            || self.whiteners.len() != n
        {
            return Err("Jacobian terms must have one row per observation".to_string());
        }
        Ok(())
    }
}

/// Universal-Kepler and light-time iteration settings of the 2-body model.
/// Defaults mirror the legacy JAX `_propagate_2body` / `_generate_ephemeris_2body`
/// kernels differentiated by the Python implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBodyModelConfig {
    pub propagation_max_iter: usize,
    pub propagation_tol: f64,
    pub lt_tol: f64,
    pub ephemeris_max_iter: usize,
    pub ephemeris_tol: f64,
    pub max_lt_iter: usize,
}

impl Default for TwoBodyModelConfig {
    fn default() -> Self {
        Self {
            propagation_max_iter: 1000,
            propagation_tol: 1e-14,
            lt_tol: 1e-10,
            ephemeris_max_iter: 100,
            ephemeris_tol: 1e-15,
            max_lt_iter: 100,
        }
    }
}

fn whitened_model_angles_generic<T: Scalar>(
    state: [T; 6],
    epoch_mjd_tdb: f64,
    terms: &TwoBodyJacobianTerms,
    config: &TwoBodyModelConfig,
) -> Vec<T> {
    let mu_helio = T::from_f64(terms.mu_helio);
    let mu_bary = T::from_f64(terms.mu_bary);
    let mut out = Vec::with_capacity(2 * terms.len());
    for i in 0..terms.len() {
        let dt = T::from_f64(terms.times_mjd_tdb[i] - epoch_mjd_tdb);
        let propagated = propagate_2body_row::<T>(
            state,
            dt,
            mu_helio,
            config.propagation_max_iter,
            config.propagation_tol,
        );
        let mut barycentric = [T::from_f64(0.0); 6];
        let mut observer = [T::from_f64(0.0); 6];
        for k in 0..6 {
            barycentric[k] = propagated[k] + T::from_f64(terms.sun_states[i][k]);
            observer[k] = T::from_f64(terms.observer_states[i][k]);
        }
        let (spherical, _light_time, _aberrated) = generate_ephemeris_2body_row::<T>(
            barycentric,
            observer,
            mu_bary,
            config.lt_tol,
            config.ephemeris_max_iter,
            config.ephemeris_tol,
            false,
            config.max_lt_iter,
        );
        let corrected_lon = spherical[1] * T::from_f64(terms.cos_lats[i]);
        let lat = spherical[2];
        let w = terms.whiteners[i];
        out.push(T::from_f64(w[0]) * corrected_lon + T::from_f64(w[1]) * lat);
        out.push(T::from_f64(w[2]) * corrected_lon + T::from_f64(w[3]) * lat);
    }
    out
}

/// Predicted whitened (lon, lat) angles of the 2-body model as a flat `(2N,)`
/// vector: the heliocentric state is propagated from the fit epoch to each
/// observation time, translated to the barycenter, projected to topocentric
/// equatorial spherical coordinates including light time, then each (lon, lat)
/// pair is cos(latitude)-corrected and whitened like the residuals.
pub fn whitened_2body_model_angles(
    state: [f64; 6],
    epoch_mjd_tdb: f64,
    terms: &TwoBodyJacobianTerms,
    config: &TwoBodyModelConfig,
) -> Result<Vec<f64>, String> {
    terms.validate()?;
    Ok(whitened_model_angles_generic::<f64>(
        state,
        epoch_mjd_tdb,
        terms,
        config,
    ))
}

/// Jacobian of the whitened residual vector with respect to the epoch state,
/// `(2N, 6)` row-major: d(obs)/dx(t) chained through the 2-body state
/// transition matrix by forward-mode automatic differentiation. Residuals are
/// (observed - predicted), hence the negation of the model Jacobian.
pub fn whitened_2body_jacobian(
    state: [f64; 6],
    epoch_mjd_tdb: f64,
    terms: &TwoBodyJacobianTerms,
    config: &TwoBodyModelConfig,
) -> Result<Vec<f64>, String> {
    terms.validate()?;
    let seeded: [Dual<6>; 6] = Dual::seed(state);
    let model = whitened_model_angles_generic::<Dual<6>>(seeded, epoch_mjd_tdb, terms, config);
    let mut jacobian = Vec::with_capacity(model.len() * 6);
    for component in model {
        for k in 0..6 {
            jacobian.push(-component.du[k]);
        }
    }
    Ok(jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU_SUN: f64 = 0.000_295_912_208_284_119_56;

    fn terms(n: usize) -> TwoBodyJacobianTerms {
        // Geocentric-like observers on a 1 AU circle in the ecliptic, the Sun
        // offset slightly from the barycenter, arcsecond sigmas.
        let mut observer_states = Vec::with_capacity(n);
        let mut sun_states = Vec::with_capacity(n);
        let mut times = Vec::with_capacity(n);
        let v = (MU_SUN / 1.0_f64).sqrt();
        for i in 0..n {
            let day = 5.0 + 4.0 * i as f64;
            let theta = v * day;
            observer_states.push([
                theta.cos(),
                theta.sin(),
                0.0,
                -v * theta.sin(),
                v * theta.cos(),
                0.0,
            ]);
            sun_states.push([0.004, -0.003, 0.0001, 1e-6, 2e-6, 0.0]);
            times.push(61_000.0 + day);
        }
        let sigma = 0.5 / 3600.0;
        let mut covariances = vec![f64::NAN; n * 36];
        for block in covariances.chunks_exact_mut(36) {
            block[7] = sigma * sigma;
            block[14] = sigma * sigma;
        }
        let lats = vec![5.0; n];
        let whiteners = observation_whitening_matrices(&covariances, &lats).unwrap();
        TwoBodyJacobianTerms {
            times_mjd_tdb: times,
            observer_states,
            sun_states,
            cos_lats: lats.iter().map(|lat| lat.to_radians().cos()).collect(),
            whiteners,
            mu_helio: MU_SUN,
            mu_bary: MU_SUN * 1.0013,
        }
    }

    #[test]
    fn loss_helpers_match_closed_forms() {
        let k = HUBER_F_SCALE_DEFAULT;
        let r = [0.5, -2.0, 3.0];
        assert_eq!(robust_cost(&r, LossType::Linear, 1.0), 0.25 + 4.0 + 9.0);
        assert!(robust_weights(&r, LossType::Linear, 1.0)
            .iter()
            .all(|&w| w == 1.0));
        assert!(robust_jacobian_scale(&r, LossType::Linear, 1.0)
            .iter()
            .all(|&s| s == 1.0));

        let inside = [0.3 * k, -0.9 * k];
        let expected: f64 = inside.iter().map(|r| r * r).sum();
        assert!((robust_cost(&inside, LossType::Huber, k) - expected).abs() < 1e-12);
        let tail = [3.0 * k, -5.0 * k];
        let expected = k * k * ((2.0 * 3.0 - 1.0) + (2.0 * 5.0 - 1.0));
        assert!((robust_cost(&tail, LossType::Huber, k) - expected).abs() < 1e-12);
        assert!((robust_cost(&[k], LossType::Huber, k) - k * k).abs() < 1e-12);

        let r = [0.5 * k, -k, 2.0 * k, -10.0 * k, 0.0];
        let weights = robust_weights(&r, LossType::Huber, k);
        for (w, e) in weights.iter().zip([1.0, 1.0, 0.5, 0.1, 1.0]) {
            assert!((w - e).abs() < 1e-12);
        }
        let scale = robust_jacobian_scale(&r, LossType::Huber, k);
        assert_eq!(scale[0], 1.0);
        assert_eq!(scale[1], 1.0);
        assert_eq!(scale[4], 1.0);
        assert!(scale[2] < 1e-7 && scale[3] < 1e-7);

        assert!(validate_loss("cauchy", 1.0).is_err());
        assert!(validate_loss("huber", 0.0).is_err());
        assert!(validate_loss("huber", f64::NAN).is_err());
        assert_eq!(validate_loss("huber", 2.0).unwrap(), LossType::Huber);
    }

    #[test]
    fn whitening_inverts_the_cos_lat_corrected_block() {
        let mut covariances = vec![f64::NAN; 72];
        covariances[7] = 4e-8;
        covariances[14] = 1e-8;
        covariances[8] = 5e-9;
        covariances[13] = 5e-9;
        covariances[36 + 7] = 1e-8;
        covariances[36 + 14] = 1e-8; // NaN cross term -> zero
        let lats = [60.0, -20.0];
        let whiteners = observation_whitening_matrices(&covariances, &lats).unwrap();
        for (row, w) in whiteners.iter().enumerate() {
            let cos_lat = lats[row].to_radians().cos();
            let block = &covariances[row * 36..(row + 1) * 36];
            let cross = if block[8].is_finite() { block[8] } else { 0.0 };
            let c = [
                block[7] * cos_lat * cos_lat,
                cross * cos_lat,
                cross * cos_lat,
                block[14],
            ];
            // W C W^T must be the identity.
            let wc = [
                w[0] * c[0] + w[1] * c[2],
                w[0] * c[1] + w[1] * c[3],
                w[2] * c[0] + w[3] * c[2],
                w[2] * c[1] + w[3] * c[3],
            ];
            let identity = [
                wc[0] * w[0] + wc[1] * w[1],
                wc[0] * w[2] + wc[1] * w[3],
                wc[2] * w[0] + wc[3] * w[1],
                wc[2] * w[2] + wc[3] * w[3],
            ];
            for (value, expected) in identity.iter().zip([1.0, 0.0, 0.0, 1.0]) {
                assert!((value - expected).abs() < 1e-9, "{row}: {identity:?}");
            }
        }
        let whitened = whiten_residual_pairs(&whiteners, &[1e-4, 2e-4, 3e-4, -1e-4]);
        assert_eq!(whitened.len(), 4);

        let mut bad = covariances.clone();
        bad[7] = f64::NAN;
        assert_eq!(
            observation_whitening_matrices(&bad, &lats).unwrap_err(),
            WhiteningError::NonFinite { row: 0 }
        );
        let mut indefinite = covariances.clone();
        indefinite[8] = 1e-7;
        indefinite[13] = 1e-7;
        assert_eq!(
            observation_whitening_matrices(&indefinite, &lats).unwrap_err(),
            WhiteningError::NotPositiveDefinite { row: 0 }
        );
    }

    #[test]
    fn autodiff_jacobian_matches_central_differences() {
        let terms = terms(6);
        let config = TwoBodyModelConfig::default();
        let state = [0.9, 0.5, 0.05, -0.008, 0.012, 0.0005];
        let epoch = 61_000.0;
        let jacobian = whitened_2body_jacobian(state, epoch, &terms, &config).unwrap();
        assert_eq!(jacobian.len(), 12 * 6);
        let base = whitened_2body_model_angles(state, epoch, &terms, &config).unwrap();
        assert!(base.iter().all(|value| value.is_finite()));
        for k in 0..6 {
            let step = 1e-6 * state[k].abs().max(1e-3);
            let mut plus = state;
            let mut minus = state;
            plus[k] += step;
            minus[k] -= step;
            let f_plus = whitened_2body_model_angles(plus, epoch, &terms, &config).unwrap();
            let f_minus = whitened_2body_model_angles(minus, epoch, &terms, &config).unwrap();
            for i in 0..base.len() {
                let numeric = -(f_plus[i] - f_minus[i]) / (2.0 * step);
                let analytic = jacobian[i * 6 + k];
                let scale = analytic.abs().max(numeric.abs()).max(1e-3);
                assert!(
                    (numeric - analytic).abs() <= 1e-5 * scale,
                    "component {i} parameter {k}: numeric {numeric} analytic {analytic}"
                );
            }
        }
    }
}
