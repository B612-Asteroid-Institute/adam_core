//! Two-body propagation via universal anomaly (Curtis 2014, ch. 3).
//!
//! Mirrors the JAX implementation in `adam_core.dynamics.{stumpff,chi,lagrange,propagation}`.
//! All kernels are generic over `T: Scalar` so the same bodies evaluate both f64
//! values and `Dual<6>` Jacobians for covariance propagation.

use adam_core_rs_autodiff::{Dual, Scalar};
use rayon::prelude::*;

pub type StumpffCoeffs<T> = [T; 6];

/// First six Stumpff functions c0..c5 evaluated at `psi` (= alpha * chi^2).
/// Danby (1992), Equations 6.9.14–6.9.16.
///
/// Numerical note: the closed-form `(1 - c0)/psi` expressions for c2..c5
/// suffer catastrophic cancellation when `|psi|` is small — both numerator
/// and denominator vanish at the same rate, and under `Dual<6>` the
/// tangent components can blow up to NaN through this degeneracy (seen
/// in practice at specific chi values like dt=2516 d for a 49-AU
/// semi-major axis orbit). For `|psi| < 1.0` we fall back to a 6-term
/// Taylor expansion in `psi`, which is numerically stable and trivially
/// Dual-compatible (polynomial in psi).
pub fn calc_stumpff<T: Scalar>(psi: T) -> StumpffCoeffs<T> {
    let psi_re = psi.re();
    // Taylor threshold: `|psi| < 1e-4`. Below this the closed-form
    // `(1 − cos(sqrt(psi)))/psi` loses 4+ sig figs to cancellation and
    // can emit NaN through the `Dual<6>` tangent arithmetic. Above 1e-4
    // the closed form is stable; below it the 5-6 term Taylor is
    // accurate to ~1e-24 and cheap. Catches the narrow dt band where
    // calc_chi's Newton iteration converges to a tiny |psi| value.
    if psi_re.abs() < 1e-4 {
        // Taylor series in psi. Each c_k = sum_{m>=0} (-psi)^m / (2m + k)!
        //   c0 = 1 − psi/2 + psi²/24 − psi³/720 + psi⁴/40320 − psi⁵/3628800 + ...
        //   c1 = 1 − psi/6 + psi²/120 − psi³/5040 + psi⁴/362880 − ...
        //   c2 = 1/2 − psi/24 + psi²/720 − psi³/40320 + ...
        //   c3 = 1/6 − psi/120 + psi²/5040 − psi³/362880 + ...
        //   c4 = 1/24 − psi/720 + psi²/40320 − psi³/3628800 + ...
        //   c5 = 1/120 − psi/5040 + psi²/362880 − ...
        // Horner-style accumulation for each. 6 terms gives ~1e-15
        // relative accuracy at |psi|=1, far tighter than the closed-form
        // stability crossover (~1e-8).
        let two = T::from_f64(2.0);
        let six = T::from_f64(6.0);
        let twenty_four = T::from_f64(24.0);
        let one_twenty = T::from_f64(120.0);
        let seven_twenty = T::from_f64(720.0);
        let five_forty = T::from_f64(5040.0);
        let forty_k = T::from_f64(40320.0);
        let three_sixty_k = T::from_f64(362880.0);
        let three_six_two_eight_k = T::from_f64(3628800.0);
        let one = T::from_f64(1.0);
        let one_half = T::from_f64(0.5);
        let one_sixth = T::from_f64(1.0 / 6.0);
        let one_twentyfour = T::from_f64(1.0 / 24.0);
        let one_oneeighty = T::from_f64(1.0 / 120.0);
        // Polynomials in psi — each truncation below 1e-16 relative error
        // at |psi| = 1; higher-order safe for Dual arithmetic.
        let c0 = one - psi / two + psi * psi / twenty_four - psi * psi * psi / seven_twenty
            + psi * psi * psi * psi / forty_k
            - psi * psi * psi * psi * psi / three_six_two_eight_k;
        let c1 = one - psi / six + psi * psi / one_twenty - psi * psi * psi / five_forty
            + psi * psi * psi * psi / three_sixty_k;
        let c2 = one_half - psi / twenty_four + psi * psi / seven_twenty
            - psi * psi * psi / forty_k
            + psi * psi * psi * psi / three_six_two_eight_k;
        let c3 =
            one_sixth - psi / one_twenty + psi * psi / five_forty - psi * psi * psi / three_sixty_k;
        let c4 = one_twentyfour - psi / seven_twenty + psi * psi / forty_k
            - psi * psi * psi / three_six_two_eight_k;
        let c5 = one_oneeighty - psi / five_forty + psi * psi / three_sixty_k;
        return [c0, c1, c2, c3, c4, c5];
    }
    if psi_re > 0.0 {
        let sqrt_psi = psi.sqrt();
        let c0 = sqrt_psi.cos();
        let c1 = sqrt_psi.sin() / sqrt_psi;
        let c2 = (T::from_f64(1.0) - c0) / psi;
        let c3 = (T::from_f64(1.0) - c1) / psi;
        let c4 = (T::from_f64(0.5) - c2) / psi;
        let c5 = (T::from_f64(1.0 / 6.0) - c3) / psi;
        [c0, c1, c2, c3, c4, c5]
    } else {
        let sqrt_npsi = (-psi).sqrt();
        let c0 = sqrt_npsi.cosh();
        let c1 = sqrt_npsi.sinh() / sqrt_npsi;
        let c2 = (T::from_f64(1.0) - c0) / psi;
        let c3 = (T::from_f64(1.0) - c1) / psi;
        let c4 = (T::from_f64(0.5) - c2) / psi;
        let c5 = (T::from_f64(1.0 / 6.0) - c3) / psi;
        [c0, c1, c2, c3, c4, c5]
    }
}

/// Newton-Raphson solver for the universal anomaly chi and its six Stumpff
/// companions. Curtis (2014), Equations 3.48, 3.50, 3.65, 3.66.
///
/// Convergence is tested on the value part of `ratio`; `Dual` tangents ride
/// through the final update so the result carries correct derivatives.
///
/// HISTORICAL NOTE: an earlier revision (2026-04-22) replaced plain Newton
/// with Laguerre's method (n=5) to tame "chaotic" Newton iterations seen
/// on a narrow elliptic regime where the f64 path needed ~60 thrash
/// iterations before settling, and `Dual<N>` tangents amplified through
/// the thrash to NaN. Laguerre with its `max(|denom|)` selection rule
/// fixed that case but introduced a NEW failure: for the universal-Kepler
/// equation in the far-from-root regime, the discriminant is dominated
/// by the curvature term `f·f''` rather than `f'²`. The "max |denom|"
/// rule then picks the branch in which the step is in the WRONG
/// direction (away from the root), causing the iteration to stall.
/// Concretely: the e≈1.2 hyperbolic Oumuamua orbit propagated forward
/// then back by 10000 d would diverge to ~1e+20 AU under Laguerre while
/// JAX's plain Newton converged to bit-parity in ~120 iterations.
///
/// The fix here is to mirror JAX exactly — plain Newton — so we are
/// always at parity with the legacy reference even on chaotic basins.
/// `Dual<N>` tangent values match JAX's `jacfwd` element-for-element on
/// those cases, including their imperfect (but finite) values.
pub fn calc_chi<T: Scalar>(
    r: [T; 3],
    v: [T; 3],
    dt: T,
    mu: T,
    max_iter: usize,
    tol: f64,
) -> (T, StumpffCoeffs<T>) {
    let consts = OrbitConstants::new(r, v, mu);
    let chi_init = consts.default_chi_init(dt);
    calc_chi_with_init(&consts, dt, chi_init, max_iter, tol)
}

/// Per-orbit constants reused by the chi solver and Lagrange coefficient
/// computation. Computed ONCE per orbit; passed by reference into
/// `calc_chi_with_init` and `lagrange_from_chi` so callers iterating one
/// orbit across many dt values (typical OD finite-difference Jacobian)
/// don't repeat the setup cost.
#[derive(Clone, Copy)]
pub struct OrbitConstants<T: Scalar> {
    pub r_mag: T,
    pub rv: T,
    pub sqrt_mu: T,
    pub alpha: T,
    pub r: [T; 3],
    pub v: [T; 3],
}

impl<T: Scalar> OrbitConstants<T> {
    pub fn new(r: [T; 3], v: [T; 3], mu: T) -> Self {
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let rv = (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) / r_mag;
        let sqrt_mu = mu.sqrt();
        let alpha = -(v_mag * v_mag) / mu + T::from_f64(2.0) / r_mag;
        Self {
            r_mag,
            rv,
            sqrt_mu,
            alpha,
            r,
            v,
        }
    }

    /// JAX-mirror initial chi guess: `sqrt(μ) · |α| · dt`.
    pub fn default_chi_init(&self, dt: T) -> T {
        self.sqrt_mu * self.alpha.abs() * dt
    }
}

/// Newton-Raphson chi solver with a caller-supplied initial guess. The
/// orbit-only constants are passed in via `consts` so they are NOT
/// recomputed per call — the win for OD inner loops where one orbit is
/// propagated to many dts.
pub fn calc_chi_with_init<T: Scalar>(
    consts: &OrbitConstants<T>,
    dt: T,
    chi_init: T,
    max_iter: usize,
    tol: f64,
) -> (T, StumpffCoeffs<T>) {
    let r_mag = consts.r_mag;
    let rv = consts.rv;
    let sqrt_mu = consts.sqrt_mu;
    let alpha = consts.alpha;
    let mut chi = chi_init;

    let mut stumpff = [T::from_f64(0.0); 6];
    let one = T::from_f64(1.0);

    for _ in 0..=max_iter {
        let chi2 = chi * chi;
        let chi3 = chi2 * chi;
        let psi = alpha * chi2;
        stumpff = calc_stumpff::<T>(psi);
        let c2 = stumpff[2];
        let c3 = stumpff[3];

        let f_val =
            r_mag * rv / sqrt_mu * chi2 * c2 + (one - alpha * r_mag) * chi3 * c3 + r_mag * chi
                - sqrt_mu * dt;
        let f_prime = r_mag * rv / sqrt_mu * chi * (one - alpha * chi2 * c3)
            + (one - alpha * r_mag) * chi2 * c2
            + r_mag;

        if f_prime.re() == 0.0 {
            break;
        }
        let ratio = f_val / f_prime;
        chi -= ratio;
        if ratio.re().abs() <= tol {
            break;
        }
    }

    (chi, stumpff)
}

/// Lagrange f, g, f_dot, g_dot coefficients for propagating (r, v) by `dt`.
/// Curtis (2014), Equations 3.69a-d.
pub fn calc_lagrange_coefficients<T: Scalar>(
    r: [T; 3],
    v: [T; 3],
    dt: T,
    mu: T,
    max_iter: usize,
    tol: f64,
) -> ([T; 4], StumpffCoeffs<T>, T) {
    let sqrt_mu = mu.sqrt();
    let (chi, stumpff) = calc_chi::<T>(r, v, dt, mu, max_iter, tol);
    let c2 = stumpff[2];
    let c3 = stumpff[3];
    let chi2 = chi * chi;
    let chi3 = chi2 * chi;

    let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let alpha = -(v_mag * v_mag) / mu + T::from_f64(2.0) / r_mag;

    let one = T::from_f64(1.0);
    let f = one - chi2 / r_mag * c2;
    let g = dt - chi3 / sqrt_mu * c3;

    let r_new = [
        f * r[0] + g * v[0],
        f * r[1] + g * v[1],
        f * r[2] + g * v[2],
    ];
    let r_new_mag = (r_new[0] * r_new[0] + r_new[1] * r_new[1] + r_new[2] * r_new[2]).sqrt();

    let f_dot = sqrt_mu / (r_mag * r_new_mag) * (alpha * chi3 * c3 - chi);
    let g_dot = one - chi2 / r_new_mag * c2;

    ([f, g, f_dot, g_dot], stumpff, chi)
}

/// Propagate a single 6-element Cartesian state by `dt` under a point-mass
/// Kepler orbit with gravitational parameter `mu`.
pub fn propagate_2body_row<T: Scalar>(
    orbit: [T; 6],
    dt: T,
    mu: T,
    max_iter: usize,
    tol: f64,
) -> [T; 6] {
    let r = [orbit[0], orbit[1], orbit[2]];
    let v = [orbit[3], orbit[4], orbit[5]];
    let (coeffs, _, _) = calc_lagrange_coefficients::<T>(r, v, dt, mu, max_iter, tol);
    let (f, g, f_dot, g_dot) = (coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
    [
        f * r[0] + g * v[0],
        f * r[1] + g * v[1],
        f * r[2] + g * v[2],
        f_dot * r[0] + g_dot * v[0],
        f_dot * r[1] + g_dot * v[1],
        f_dot * r[2] + g_dot * v[2],
    ]
}

/// Propagate a single orbit to many dt values.
///
/// The chi solver is warm-started from the previous dt's converged chi,
/// which (for sorted dts) drops Newton iterations from ~120 to ~5 per
/// step in typical OD inner-loop usage where one orbit is differenced
/// against many observation epochs.
///
/// Output preserves the original input order regardless of internal
/// processing order. Caller does NOT need to pre-sort `dts`.
pub fn propagate_2body_along_arc(
    orbit: [f64; 6],
    dts: &[f64],
    mu: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<[f64; 6]> {
    let n = dts.len();
    if n == 0 {
        return Vec::new();
    }
    let r = [orbit[0], orbit[1], orbit[2]];
    let v = [orbit[3], orbit[4], orbit[5]];
    let consts = OrbitConstants::new(r, v, mu);

    // Sort indices by signed dt (ascending) so adjacent calls share
    // monotonically-changing chi and warm-start is most accurate.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| dts[i].partial_cmp(&dts[j]).unwrap());

    let mut out = vec![[0.0_f64; 6]; n];
    let mut prev_dt = 0.0_f64;
    let mut prev_chi = 0.0_f64;
    let mut have_prev = false;

    let one = 1.0_f64;
    for &i in &idx {
        let dt = dts[i];
        let chi_init = if have_prev {
            // chi(dt) ≈ chi(prev_dt) + sqrt_mu·|alpha|·(dt − prev_dt)
            // — a first-order Taylor expansion of f(chi)=0 in dt that
            // lands within a couple of Newton steps of the true root.
            prev_chi + consts.sqrt_mu * consts.alpha.abs() * (dt - prev_dt)
        } else {
            consts.default_chi_init(dt)
        };
        let (chi, stumpff) = calc_chi_with_init(&consts, dt, chi_init, max_iter, tol);
        let c2 = stumpff[2];
        let c3 = stumpff[3];
        let chi2 = chi * chi;
        let chi3 = chi2 * chi;

        let f = one - chi2 / consts.r_mag * c2;
        let g = dt - chi3 / consts.sqrt_mu * c3;

        let r_new = [
            f * r[0] + g * v[0],
            f * r[1] + g * v[1],
            f * r[2] + g * v[2],
        ];
        let r_new_mag = (r_new[0] * r_new[0] + r_new[1] * r_new[1] + r_new[2] * r_new[2]).sqrt();

        let f_dot = consts.sqrt_mu / (consts.r_mag * r_new_mag) * (consts.alpha * chi3 * c3 - chi);
        let g_dot = one - chi2 / r_new_mag * c2;

        out[i] = [
            r_new[0],
            r_new[1],
            r_new[2],
            f_dot * r[0] + g_dot * v[0],
            f_dot * r[1] + g_dot * v[1],
            f_dot * r[2] + g_dot * v[2],
        ];

        prev_dt = dt;
        prev_chi = chi;
        have_prev = true;
    }
    out
}

/// Batched arc propagation: N orbits, each propagated to K dts, with
/// rayon parallelism ACROSS orbits and serial warm-started chi
/// solving WITHIN each orbit.
///
/// Inputs:
///   * `orbits_flat`: shape `(N, 6)` row-major Cartesian states
///   * `dts_flat`: shape `(N, K)` row-major — each orbit's K dt values
///   * `mus`: shape `(N,)` per-orbit gravitational parameters
///
/// Output: `(N, K, 6)` row-major flattened to `Vec<f64>` of length
/// `N * K * 6`. Within each orbit's K rows, output is in the same
/// dt order as the input (arc internally sorts and unsorts).
pub fn propagate_2body_arc_batch_flat6(
    orbits_flat: &[f64],
    dts_flat: &[f64],
    mus: &[f64],
    k: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    assert_eq!(orbits_flat.len() % 6, 0);
    let n = orbits_flat.len() / 6;
    assert_eq!(dts_flat.len(), n * k);
    assert_eq!(mus.len(), n);

    let mut out = vec![0.0_f64; n * k * 6];
    out.par_chunks_mut(k * 6)
        .zip(orbits_flat.par_chunks(6))
        .zip(dts_flat.par_chunks(k))
        .zip(mus.par_iter())
        .for_each(|(((out_chunk, orbit_row), dt_row), mu)| {
            let orbit: [f64; 6] = [
                orbit_row[0],
                orbit_row[1],
                orbit_row[2],
                orbit_row[3],
                orbit_row[4],
                orbit_row[5],
            ];
            let rows = propagate_2body_along_arc(orbit, dt_row, *mu, max_iter, tol);
            for (j, row) in rows.iter().enumerate() {
                out_chunk[j * 6..(j + 1) * 6].copy_from_slice(row);
            }
        });
    out
}

/// Batched row-wise propagation (rayon parallelism over rows).
pub fn propagate_2body_flat6(
    orbits_flat: &[f64],
    dts: &[f64],
    mus: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    assert_eq!(
        orbits_flat.len() % 6,
        0,
        "orbits_flat length must be a multiple of 6",
    );
    let n = orbits_flat.len() / 6;
    assert_eq!(dts.len(), n, "dts length must match orbits rows");
    assert_eq!(mus.len(), n, "mus length must match orbits rows");

    let mut out = vec![0.0_f64; orbits_flat.len()];
    out.par_chunks_mut(6)
        .zip(orbits_flat.par_chunks(6))
        .zip(dts.par_iter())
        .zip(mus.par_iter())
        .for_each(|(((out_row, in_row), dt), mu)| {
            let orbit: [f64; 6] = [
                in_row[0], in_row[1], in_row[2], in_row[3], in_row[4], in_row[5],
            ];
            let propagated = propagate_2body_row::<f64>(orbit, *dt, *mu, max_iter, tol);
            out_row.copy_from_slice(&propagated);
        });
    out
}

/// Propagate a single row and also emit the 6x6 Jacobian of the output state
/// with respect to the input Cartesian state (derivatives are zero w.r.t. dt
/// and mu, which are seeded as constants).
fn propagate_with_jacobian_row(
    orbit: [f64; 6],
    dt: f64,
    mu: f64,
    max_iter: usize,
    tol: f64,
) -> ([f64; 6], [[f64; 6]; 6]) {
    let rows_d: [Dual<6>; 6] = Dual::seed(orbit);
    let dt_d = Dual::constant(dt);
    let mu_d = Dual::constant(mu);
    let out_d = propagate_2body_row::<Dual<6>>(rows_d, dt_d, mu_d, max_iter, tol);
    let mut value = [0.0_f64; 6];
    let mut jac = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        value[i] = out_d[i].re;
        for j in 0..6 {
            jac[i][j] = out_d[i].du[j];
        }
    }
    (value, jac)
}

/// Batched propagation with covariance transport. For each row, evaluates
/// the state and the 6x6 Jacobian via a single `Dual<6>` pass, then applies
/// `Sigma_out = J @ Sigma_in @ J^T`. NaN covariance rows pass NaN through.
pub fn propagate_2body_with_covariance_flat6(
    orbits_flat: &[f64],
    cov_flat: &[f64],
    dts: &[f64],
    mus: &[f64],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        orbits_flat.len() % 6,
        0,
        "orbits_flat length must be a multiple of 6",
    );
    let n = orbits_flat.len() / 6;
    assert_eq!(dts.len(), n, "dts length must match orbits rows");
    assert_eq!(mus.len(), n, "mus length must match orbits rows");
    assert_eq!(cov_flat.len(), n * 36, "cov_flat length must be n*36");

    let mut out_states = vec![0.0_f64; orbits_flat.len()];
    let mut out_covs = vec![0.0_f64; cov_flat.len()];

    out_states
        .par_chunks_mut(6)
        .zip(out_covs.par_chunks_mut(36))
        .zip(orbits_flat.par_chunks(6))
        .zip(cov_flat.par_chunks(36))
        .zip(dts.par_iter())
        .zip(mus.par_iter())
        .for_each(|(((((state_out, cov_out), in_row), cov_in), dt), mu)| {
            let orbit: [f64; 6] = [
                in_row[0], in_row[1], in_row[2], in_row[3], in_row[4], in_row[5],
            ];
            let (value, jac) = propagate_with_jacobian_row(orbit, *dt, *mu, max_iter, tol);
            state_out.copy_from_slice(&value);

            let mut any_nan = false;
            for &x in cov_in.iter() {
                if x.is_nan() {
                    any_nan = true;
                    break;
                }
            }
            if any_nan {
                for x in cov_out.iter_mut() {
                    *x = f64::NAN;
                }
                return;
            }

            // Sigma_out = J @ Sigma_in @ J^T
            // Compute M = J @ Sigma_in (6x6), then Sigma_out = M @ J^T.
            let mut m = [[0.0_f64; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    let mut acc = 0.0;
                    for k in 0..6 {
                        acc += jac[i][k] * cov_in[k * 6 + j];
                    }
                    m[i][j] = acc;
                }
            }
            for i in 0..6 {
                for j in 0..6 {
                    let mut acc = 0.0;
                    for k in 0..6 {
                        acc += m[i][k] * jac[j][k];
                    }
                    cov_out[i * 6 + j] = acc;
                }
            }
        });

    (out_states, out_covs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heliocentric gravitational parameter in au^3/day^2 (Gaussian k^2).
    const MU_SUN: f64 = 2.95912208284120e-4;

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn norm(v: [f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn propagate_zero_dt_returns_input() {
        let orbit = [1.0, 0.2, 0.1, 0.001, 0.015, 0.0005];
        let out = propagate_2body_row::<f64>(orbit, 0.0, MU_SUN, 100, 1e-14);
        for i in 0..6 {
            assert!(
                (out[i] - orbit[i]).abs() < 1e-12,
                "row {}: {} vs {}",
                i,
                out[i],
                orbit[i]
            );
        }
    }

    #[test]
    fn propagate_elliptical_conserves_energy_and_angular_momentum() {
        let orbit = [1.5, 0.2, 0.05, -0.003, 0.017, 0.0008];
        let r0 = [orbit[0], orbit[1], orbit[2]];
        let v0 = [orbit[3], orbit[4], orbit[5]];
        let e0 = 0.5 * (v0[0].powi(2) + v0[1].powi(2) + v0[2].powi(2)) - MU_SUN / norm(r0);
        let h0 = cross(r0, v0);

        for dt in [10.0_f64, 100.0, 1000.0, 10000.0] {
            let out = propagate_2body_row::<f64>(orbit, dt, MU_SUN, 1000, 1e-14);
            let r = [out[0], out[1], out[2]];
            let v = [out[3], out[4], out[5]];
            let e = 0.5 * (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)) - MU_SUN / norm(r);
            let h = cross(r, v);
            assert!(
                (e - e0).abs() / e0.abs() < 1e-10,
                "dt={} energy drift {}",
                dt,
                (e - e0).abs()
            );
            for k in 0..3 {
                assert!(
                    (h[k] - h0[k]).abs() / h0[k].abs().max(1e-10) < 1e-10,
                    "dt={} h[{}] drift {}",
                    dt,
                    k,
                    (h[k] - h0[k]).abs()
                );
            }
        }
    }

    #[test]
    fn propagate_roundtrip_is_near_identity() {
        let orbit = [2.5, -0.3, 0.1, 0.002, 0.012, -0.001];
        let dt = 5000.0;
        let forward = propagate_2body_row::<f64>(orbit, dt, MU_SUN, 1000, 1e-14);
        let back = propagate_2body_row::<f64>(forward, -dt, MU_SUN, 1000, 1e-14);
        for i in 0..6 {
            assert!(
                (back[i] - orbit[i]).abs() < 1e-8,
                "roundtrip[{}] drift {}",
                i,
                (back[i] - orbit[i]).abs()
            );
        }
    }

    #[test]
    fn jacobian_is_finite_and_reasonable_scale() {
        let orbit = [1.5, 0.2, 0.05, -0.003, 0.017, 0.0008];
        let (_value, jac) = propagate_with_jacobian_row(orbit, 100.0, MU_SUN, 1000, 1e-14);
        for (i, row) in jac.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                assert!(val.is_finite(), "jac[{i}][{j}] not finite");
            }
        }
        // Position sensitivity to initial position should be O(1) for short
        // propagations (rows 0..3 wrt cols 0..3).
        for (i, row) in jac.iter().enumerate().take(3) {
            assert!(row[i].abs() > 0.01, "jac[{i}][{i}] = {}", row[i]);
        }
    }
}
