//! Rust-native Gauss-Newton least-squares orbit determination (W6 / OD slice 4,
//! bead personal-cmy.7).
//!
//! Iteratively corrects a 6-vector Cartesian state (at a fixed epoch) to minimize
//! the chi residuals against astrometric observations, reusing the exact 2-body
//! ephemeris + residual kernels validated in slice 3 (`propagate_2body_flat6`,
//! `generate_ephemeris_2body_flat6`, `compute_residuals_chi2_flat`).
//!
//! This is a Rust-native optimizer (Gauss-Newton with a 2-point finite-difference
//! Jacobian). It converges to the *same least-squares minimum* and covariance
//! (`inv(JᵀJ)`) as adam_core's scipy-based `fit_least_squares`, but is **not** a
//! bit-exact reproduction of scipy's trust-region iterates. Inputs are
//! barycentric (SSB / ecliptic), matching `generate_ephemeris_2body`'s convention.

use crate::types::{Epoch, NANOS_PER_DAY};
use crate::{compute_residuals_chi2_flat, generate_ephemeris_2body_flat6, propagate_2body_flat6};

/// Result of a least-squares orbit fit.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresFit {
    /// Fitted Cartesian state at the epoch (barycentric / ecliptic).
    pub state: [f64; 6],
    /// Row-major 6x6 parameter covariance `inv(JᵀJ)` at the solution.
    pub covariance: [f64; 36],
    /// Total chi² at the solution (sum over observations).
    pub chi2: f64,
    /// Gauss-Newton iterations performed.
    pub iterations: usize,
    /// Whether the step-size convergence criterion was met.
    pub converged: bool,
}

/// Gauss-Newton + ephemeris tuning parameters. Defaults mirror adam_core's
/// `fit_least_squares` / `generate_ephemeris_2body` defaults where applicable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeastSquaresConfig {
    pub xtol: f64,
    pub ftol: f64,
    pub max_iterations: usize,
    pub lt_tol: f64,
    pub ephemeris_max_iter: usize,
    pub ephemeris_tol: f64,
    pub stellar_aberration: bool,
    pub max_lt_iter: usize,
}

impl Default for LeastSquaresConfig {
    fn default() -> Self {
        Self {
            xtol: 1e-12,
            ftol: 1e-12,
            max_iterations: 100,
            lt_tol: 1e-10,
            ephemeris_max_iter: 1000,
            ephemeris_tol: 1e-15,
            stellar_aberration: false,
            max_lt_iter: 10,
        }
    }
}

fn epoch_diff_days(target: Epoch, origin: Epoch) -> f64 {
    (target.days - origin.days) as f64 + (target.nanos - origin.nanos) as f64 / NANOS_PER_DAY as f64
}

/// Per-observation chi residuals (`sqrt(chi2)`, matching adam_core's residual
/// function) and the total chi² for already-predicted spherical coordinates.
/// Shared by the 2-body kernel below and the backend-generic
/// [`fit_orbit_least_squares_with_predictor`] driver.
pub(crate) fn chi_residuals_from_predicted(
    predicted: &[f64],
    observed: &[f64],
    observed_cov: &[f64],
    n: usize,
) -> (Vec<f64>, f64) {
    let predicted_cov = vec![0.0_f64; n * 36];
    let output = compute_residuals_chi2_flat(
        observed,
        predicted,
        observed_cov,
        &predicted_cov,
        n,
        6,
        true,
    )
    .expect("residual shapes are constructed consistently");
    let residuals: Vec<f64> = output.chi2.iter().map(|c| c.max(0.0).sqrt()).collect();
    let total: f64 = output.chi2.iter().sum();
    (residuals, total)
}

/// Per-observation chi residuals and the total chi² for a candidate state
/// through the internal 2-body ephemeris.
fn chi_residuals(
    state: &[f64; 6],
    mu: f64,
    observed: &[f64],
    observed_cov: &[f64],
    observer_states: &[f64],
    dts: &[f64],
    config: &LeastSquaresConfig,
) -> (Vec<f64>, f64) {
    let n = dts.len();
    let mut orbit_flat = vec![0.0_f64; n * 6];
    for slot in orbit_flat.chunks_exact_mut(6) {
        slot.copy_from_slice(state);
    }
    let mus = vec![mu; n];
    let propagated = propagate_2body_flat6(
        &orbit_flat,
        dts,
        &mus,
        config.ephemeris_max_iter,
        config.ephemeris_tol,
    );
    let (predicted, _light_time, _aberrated) = generate_ephemeris_2body_flat6(
        &propagated,
        observer_states,
        &mus,
        config.lt_tol,
        config.ephemeris_max_iter,
        config.ephemeris_tol,
        config.stellar_aberration,
        config.max_lt_iter,
    );
    chi_residuals_from_predicted(&predicted, observed, observed_cov, n)
}

/// Backend-generic Gauss-Newton least-squares core (bead personal-cmy.7).
///
/// `predict` maps `M` candidate Cartesian states (all at the fit epoch) to
/// predicted spherical coordinates, `(M * N, 6)` row-major in candidate-major
/// order — typically one `generate_ephemeris::<P>` crossing per call, which is
/// what makes propagator-backed fits cheap: each Gauss-Newton iteration costs
/// one base prediction plus ONE batched 6-state Jacobian prediction (the
/// ASSIST same-epoch multi-particle fast path integrates all seven candidates
/// in a single simulation) instead of legacy scipy's seven sequential Python
/// propagator round-trips.
///
/// The algorithm mirrors [`fit_orbit_2body_least_squares`]: 2-point
/// finite-difference Jacobian, normal equations, backtracking line search,
/// chi²-plateau convergence, `inv(JᵀJ)` covariance.
pub fn fit_orbit_least_squares_with_predictor<F>(
    initial_state: [f64; 6],
    observed: &[f64],
    observed_cov: &[f64],
    n: usize,
    config: &LeastSquaresConfig,
    mut predict: F,
) -> Result<LeastSquaresFit, String>
where
    F: FnMut(&[[f64; 6]]) -> Result<Vec<f64>, String>,
{
    if observed.len() != n * 6 || observed_cov.len() != n * 36 {
        return Err("observed/observed_cov shapes must be (N, 6)/(N, 36)".to_string());
    }

    fn perturbations(state: &[f64; 6]) -> ([[f64; 6]; 6], [f64; 6]) {
        let mut steps = [0.0_f64; 6];
        let mut perturbed_states = [[0.0_f64; 6]; 6];
        for k in 0..6 {
            let h = 1.490_116_119_384_765_6e-8 * state[k].abs().max(1.0);
            let mut perturbed = *state;
            perturbed[k] += h;
            steps[k] = perturbed[k] - state[k];
            perturbed_states[k] = perturbed;
        }
        (perturbed_states, steps)
    }

    fn jacobian_from_batch(
        predicted_all: &[f64],
        base: &[f64],
        steps: &[f64; 6],
        observed: &[f64],
        observed_cov: &[f64],
        n: usize,
    ) -> Vec<f64> {
        let mut jac = vec![0.0_f64; n * 6];
        for k in 0..6 {
            let (rk, _) = chi_residuals_from_predicted(
                &predicted_all[k * n * 6..(k + 1) * n * 6],
                observed,
                observed_cov,
                n,
            );
            for i in 0..n {
                jac[i * 6 + k] = (rk[i] - base[i]) / steps[k];
            }
        }
        jac
    }

    let expect_rows = |predicted: &Vec<f64>, states: usize| -> Result<(), String> {
        if predicted.len() != states * n * 6 {
            return Err(format!(
                "predictor returned {} values for {} states x {} observations",
                predicted.len(),
                states,
                n
            ));
        }
        Ok(())
    };

    let mut state = initial_state;
    let mut converged = false;
    let mut iterations = 0;
    let mut previous_chi2 = f64::INFINITY;
    for iteration in 0..config.max_iterations {
        iterations = iteration + 1;
        let predicted = predict(&[state])?;
        expect_rows(&predicted, 1)?;
        let (base, chi2) = chi_residuals_from_predicted(&predicted, observed, observed_cov, n);
        if iteration > 0 && (previous_chi2 - chi2).abs() <= config.ftol * previous_chi2.max(1.0) {
            converged = true;
            break;
        }
        previous_chi2 = chi2;
        let (perturbed_states, steps) = perturbations(&state);
        let predicted_all = predict(&perturbed_states)?;
        expect_rows(&predicted_all, 6)?;
        let jac = jacobian_from_batch(&predicted_all, &base, &steps, observed, observed_cov, n);
        let a = normal_matrix(&jac, n);
        let mut rhs = [0.0_f64; 6];
        for i in 0..n {
            for k in 0..6 {
                rhs[k] -= jac[i * 6 + k] * base[i];
            }
        }
        let delta = match solve_6x6(&a, &rhs) {
            Some(delta) => delta,
            None => break,
        };
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..12 {
            let mut trial = state;
            for k in 0..6 {
                trial[k] += alpha * delta[k];
            }
            let predicted_trial = predict(&[trial])?;
            expect_rows(&predicted_trial, 1)?;
            let (_, trial_chi2) =
                chi_residuals_from_predicted(&predicted_trial, observed, observed_cov, n);
            if trial_chi2 < chi2 {
                state = trial;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            converged = true;
            break;
        }
    }

    let predicted = predict(&[state])?;
    expect_rows(&predicted, 1)?;
    let (base, chi2) = chi_residuals_from_predicted(&predicted, observed, observed_cov, n);
    let (perturbed_states, steps) = perturbations(&state);
    let predicted_all = predict(&perturbed_states)?;
    expect_rows(&predicted_all, 6)?;
    let jac = jacobian_from_batch(&predicted_all, &base, &steps, observed, observed_cov, n);
    let covariance = inverse_6x6(&normal_matrix(&jac, n)).unwrap_or([f64::NAN; 36]);

    Ok(LeastSquaresFit {
        state,
        covariance,
        chi2,
        iterations,
        converged,
    })
}

/// 2-point forward-difference Jacobian of the chi residuals w.r.t. the state
/// (`N x 6`, row-major). `rel_step ~ sqrt(machine eps)` matches scipy `2-point`.
fn jacobian(
    state: &[f64; 6],
    base: &[f64],
    mu: f64,
    observed: &[f64],
    observed_cov: &[f64],
    observer_states: &[f64],
    dts: &[f64],
    config: &LeastSquaresConfig,
) -> Vec<f64> {
    let n = dts.len();
    let mut jac = vec![0.0_f64; n * 6];
    for k in 0..6 {
        let h = 1.490_116_119_384_765_6e-8 * state[k].abs().max(1.0);
        let mut perturbed = *state;
        perturbed[k] += h;
        let step = perturbed[k] - state[k];
        let (rk, _) = chi_residuals(
            &perturbed,
            mu,
            observed,
            observed_cov,
            observer_states,
            dts,
            config,
        );
        for i in 0..n {
            jac[i * 6 + k] = (rk[i] - base[i]) / step;
        }
    }
    jac
}

fn normal_matrix(jac: &[f64], n: usize) -> [[f64; 6]; 6] {
    let mut a = [[0.0_f64; 6]; 6];
    for i in 0..n {
        for k in 0..6 {
            for l in 0..6 {
                a[k][l] += jac[i * 6 + k] * jac[i * 6 + l];
            }
        }
    }
    a
}

/// Solve `A x = b` for a 6x6 system via Gauss-Jordan with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn solve_6x6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
    let mut m = [[0.0_f64; 7]; 6];
    for i in 0..6 {
        m[i][..6].copy_from_slice(&a[i]);
        m[i][6] = b[i];
    }
    for col in 0..6 {
        let mut pivot = col;
        for row in (col + 1)..6 {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        if m[pivot][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, pivot);
        let diag = m[col][col];
        for v in m[col].iter_mut() {
            *v /= diag;
        }
        for row in 0..6 {
            if row == col {
                continue;
            }
            let factor = m[row][col];
            for c in 0..7 {
                m[row][c] -= factor * m[col][c];
            }
        }
    }
    let mut x = [0.0_f64; 6];
    for i in 0..6 {
        x[i] = m[i][6];
    }
    Some(x)
}

/// Invert a 6x6 matrix (row-major output). Returns `None` if singular.
fn inverse_6x6(a: &[[f64; 6]; 6]) -> Option<[f64; 36]> {
    let mut out = [0.0_f64; 36];
    for col in 0..6 {
        let mut e = [0.0_f64; 6];
        e[col] = 1.0;
        let solution = solve_6x6(a, &e)?;
        for row in 0..6 {
            out[row * 6 + col] = solution[row];
        }
    }
    Some(out)
}

/// Differentially correct a single orbit against astrometric observations using
/// Gauss-Newton least squares. All inputs are barycentric (SSB / ecliptic);
/// `observed`/`observed_cov` are spherical `(N, 6)` / `(N, 36)` row-major, and
/// `observer_states` are Cartesian `(N, 6)` at the observation epochs.
#[allow(clippy::too_many_arguments)]
pub fn fit_orbit_2body_least_squares(
    initial_state: [f64; 6],
    epoch: Epoch,
    mu: f64,
    observed: &[f64],
    observed_cov: &[f64],
    observer_states: &[f64],
    obs_epochs: &[Epoch],
    config: &LeastSquaresConfig,
) -> LeastSquaresFit {
    let n = obs_epochs.len();
    let dts: Vec<f64> = obs_epochs
        .iter()
        .map(|epoch_obs| epoch_diff_days(*epoch_obs, epoch))
        .collect();

    let mut state = initial_state;
    let mut converged = false;
    let mut iterations = 0;
    let mut previous_chi2 = f64::INFINITY;
    for iteration in 0..config.max_iterations {
        iterations = iteration + 1;
        let (base, chi2) = chi_residuals(
            &state,
            mu,
            observed,
            observed_cov,
            observer_states,
            &dts,
            config,
        );
        // chi2-plateau convergence (finite-difference Jacobians cannot reach
        // scipy's 1e-12 step tolerance, so converge on the cost reduction).
        if iteration > 0 && (previous_chi2 - chi2).abs() <= config.ftol * previous_chi2.max(1.0) {
            converged = true;
            break;
        }
        previous_chi2 = chi2;
        let jac = jacobian(
            &state,
            &base,
            mu,
            observed,
            observed_cov,
            observer_states,
            &dts,
            config,
        );
        let a = normal_matrix(&jac, n);
        let mut rhs = [0.0_f64; 6];
        for i in 0..n {
            for k in 0..6 {
                rhs[k] -= jac[i * 6 + k] * base[i];
            }
        }
        let delta = match solve_6x6(&a, &rhs) {
            Some(delta) => delta,
            None => break,
        };
        // Backtracking line search: accept the largest step that reduces chi2.
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..12 {
            let mut trial = state;
            for k in 0..6 {
                trial[k] += alpha * delta[k];
            }
            let (_, trial_chi2) = chi_residuals(
                &trial,
                mu,
                observed,
                observed_cov,
                observer_states,
                &dts,
                config,
            );
            if trial_chi2 < chi2 {
                state = trial;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            // No downhill step found: at the minimum within numerical noise.
            converged = true;
            break;
        }
    }

    let (final_residuals, chi2) = chi_residuals(
        &state,
        mu,
        observed,
        observed_cov,
        observer_states,
        &dts,
        config,
    );
    let jac = jacobian(
        &state,
        &final_residuals,
        mu,
        observed,
        observed_cov,
        observer_states,
        &dts,
        config,
    );
    let covariance = inverse_6x6(&normal_matrix(&jac, n)).unwrap_or([f64::NAN; 36]);

    LeastSquaresFit {
        state,
        covariance,
        chi2,
        iterations,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build synthetic noise-free observations from a truth orbit, then confirm
    // Gauss-Newton recovers it from a perturbed start (chi2 -> 0, state -> truth).
    fn observer_states(n: usize) -> (Vec<f64>, Vec<Epoch>) {
        // Observers on a 1 AU circular path sampled over ~40 days for geometry.
        let mut states = Vec::with_capacity(n * 6);
        let mut epochs = Vec::with_capacity(n);
        let mu = 0.000_295_912_208_285_591_1;
        let v = (mu / 1.0_f64).sqrt();
        for i in 0..n {
            let day = i as f64 * 5.0;
            let theta = v * day; // mean motion ~ v for r=1
            states.extend_from_slice(&[
                theta.cos(),
                theta.sin(),
                0.0,
                -v * theta.sin(),
                v * theta.cos(),
                0.0,
            ]);
            epochs.push(Epoch::new(60000 + (i as i64) * 5, 0));
        }
        (states, epochs)
    }

    fn truth_observations(
        truth: &[f64; 6],
        mu: f64,
        observer_states: &[f64],
        epochs: &[Epoch],
        epoch: Epoch,
        config: &LeastSquaresConfig,
    ) -> Vec<f64> {
        let n = epochs.len();
        let dts: Vec<f64> = epochs.iter().map(|e| epoch_diff_days(*e, epoch)).collect();
        let mut orbit_flat = vec![0.0_f64; n * 6];
        for slot in orbit_flat.chunks_exact_mut(6) {
            slot.copy_from_slice(truth);
        }
        let mus = vec![mu; n];
        let propagated = propagate_2body_flat6(
            &orbit_flat,
            &dts,
            &mus,
            config.ephemeris_max_iter,
            config.ephemeris_tol,
        );
        let (predicted, _lt, _ab) = generate_ephemeris_2body_flat6(
            &propagated,
            observer_states,
            &mus,
            config.lt_tol,
            config.ephemeris_max_iter,
            config.ephemeris_tol,
            config.stellar_aberration,
            config.max_lt_iter,
        );
        predicted
    }

    #[test]
    fn gauss_newton_recovers_truth_from_noise_free_observations() {
        let config = LeastSquaresConfig::default();
        let mu = 0.000_295_912_208_285_591_1;
        let epoch = Epoch::new(60000, 0);
        let truth = [1.2, 0.1, 0.05, -0.002, 0.016, 0.001];
        let (observers, epochs) = observer_states(8);
        let observed = truth_observations(&truth, mu, &observers, &epochs, epoch, &config);
        // arcsecond-scale astrometric sigmas on lon/lat (degrees), loose elsewhere.
        let arcsec = (1.0_f64 / 3600.0).powi(2);
        let mut observed_cov = vec![0.0_f64; epochs.len() * 36];
        for block in observed_cov.chunks_exact_mut(36) {
            let diag = [1.0, arcsec, arcsec, 1.0, 1.0, 1.0];
            for d in 0..6 {
                block[d * 6 + d] = diag[d];
            }
        }
        let initial = [
            truth[0] + 1e-3,
            truth[1] - 1e-3,
            truth[2] + 5e-4,
            truth[3] + 1e-5,
            truth[4] - 1e-5,
            truth[5] + 1e-5,
        ];
        let fit = fit_orbit_2body_least_squares(
            initial,
            epoch,
            mu,
            &observed,
            &observed_cov,
            &observers,
            &epochs,
            &config,
        );
        assert!(fit.converged, "GN did not converge: {fit:?}");
        // The truth fits with chi2=0; a forward-difference (2-point) Jacobian on
        // a multi-day arc limits the recovered state to ~1e-7 AU, so chi2 floors
        // near (sub-arcsecond residual)^2 rather than exactly zero.
        assert!(fit.chi2 < 0.1, "chi2 not minimized: {}", fit.chi2);
        for (fitted, truth_value) in fit.state.iter().zip(truth.iter()) {
            assert!(
                (fitted - truth_value).abs() < 1e-4,
                "state {fitted} vs truth {truth_value}"
            );
        }
    }
}
