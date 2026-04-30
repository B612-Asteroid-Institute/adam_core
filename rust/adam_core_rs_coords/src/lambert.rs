//! Lambert's problem solver — Izzo's algorithm.
//!
//! Mirrors `adam_core.dynamics.lambert.izzo_lambert` exactly:
//!
//! - Hypergeometric 2F1(3, 1, 5/2, x) via power series (`_hyp2f1b`)
//! - Non-dimensional time-of-flight equation and its first three derivatives
//! - Initial guess with three regions (T ≥ T_0, T < T_1, middle interpolation)
//!   plus multi-revolution variants
//! - Householder (quartic) root-finding on the TOF equation
//! - Halley (cubic) minimum-finding used by multi-rev T_min refinement
//!
//! References:
//! - Izzo, D. (2015). Revisiting Lambert's problem. Celestial Mechanics and
//!   Dynamical Astronomy, 121(1), 1-15.
//!
//! Inputs/outputs are plain f64 — Lambert isn't in `Ephemeris` covariance
//! tracking, so no `Dual<N>` support is needed.

use rayon::prelude::*;
use std::f64::consts::PI;

#[inline]
fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Hypergeometric function 2F1(3, 1, 5/2, x) via power series. Matches JAX's
/// fixed-100-iteration evaluation bit-for-bit (stops at the same term count).
fn hyp2f1b(x: f64) -> f64 {
    if x >= 1.0 {
        return f64::INFINITY;
    }
    let mut term = 1.0_f64;
    let mut res = 1.0_f64;
    for i in 0..100 {
        let i_f = i as f64;
        term = term * (3.0 + i_f) * (1.0 + i_f) / (2.5 + i_f) * x / (i_f + 1.0);
        res += term;
    }
    res
}

#[inline]
fn compute_y(x: f64, ll: f64) -> f64 {
    (1.0 - ll * ll * (1.0 - x * x)).sqrt()
}

/// psi(x, y; ll) — three-branch: elliptic (x<1), hyperbolic (x>1), parabolic (x=1).
fn compute_psi(x: f64, y: f64, ll: f64) -> f64 {
    if x < 1.0 {
        let arg = (x * y + ll * (1.0 - x * x)).clamp(-1.0, 1.0);
        arg.acos()
    } else if x > 1.0 {
        ((y - x * ll) * (x * x - 1.0).sqrt()).asinh()
    } else {
        0.0
    }
}

/// Non-dimensional TOF equation with externally computed y. Matches JAX's
/// branching: use the small-M analytic form for M==0 ∧ sqrt(0.6)<x<sqrt(1.4),
/// otherwise the general form using psi.
fn tof_equation_y(x: f64, y: f64, t0: f64, ll: f64, m: u32) -> f64 {
    // Small-M analytic form: eta = y - ll·x, S_1 = (1 - ll - x·eta)/2,
    // Q = (4/3)·hyp2f1b(S_1), T_ = (eta³·Q + 4·ll·eta) / 2.
    let eta = y - ll * x;
    let s_1 = (1.0 - ll - x * eta) * 0.5;
    let q_h = 4.0 / 3.0 * hyp2f1b(s_1);
    let small_m = (eta.powi(3) * q_h + 4.0 * ll * eta) * 0.5;

    // General form.
    let psi = compute_psi(x, y, ll);
    let sqrt_term = (1.0 - x * x).abs().sqrt();
    let general = (psi + (m as f64) * PI) / sqrt_term - x + ll * y;
    let general = general / (1.0 - x * x);

    let use_small = m == 0 && x > 0.6_f64.sqrt() && x < 1.4_f64.sqrt();
    let t_ = if use_small { small_m } else { general };
    t_ - t0
}

#[inline]
fn tof_equation(x: f64, t0: f64, ll: f64, m: u32) -> f64 {
    let y = compute_y(x, ll);
    tof_equation_y(x, y, t0, ll, m)
}

#[inline]
fn tof_equation_p(x: f64, y: f64, t: f64, ll: f64) -> f64 {
    (3.0 * t * x - 2.0 + 2.0 * ll.powi(3) * x / y) / (1.0 - x * x)
}

#[inline]
fn tof_equation_p2(x: f64, y: f64, t: f64, dt: f64, ll: f64) -> f64 {
    (3.0 * t + 5.0 * x * dt + 2.0 * (1.0 - ll * ll) * ll.powi(3) / y.powi(3)) / (1.0 - x * x)
}

#[inline]
fn tof_equation_p3(x: f64, y: f64, dt: f64, ddt: f64, ll: f64) -> f64 {
    (7.0 * x * ddt + 8.0 * dt - 6.0 * (1.0 - ll * ll) * ll.powi(5) * x / y.powi(5)) / (1.0 - x * x)
}

/// Initial guess for x, per Izzo Section 2.1.
fn initial_guess(t: f64, ll: f64, m: u32, low_path: bool) -> f64 {
    let t_0 = ll.acos() + ll * (1.0 - ll * ll).sqrt() + (m as f64) * PI;
    let t_1 = 2.0 * (1.0 - ll.powi(3)) / 3.0;

    if m == 0 {
        if t >= t_0 {
            (t_0 / t).powf(2.0 / 3.0) - 1.0
        } else if t < t_1 {
            5.0 / 2.0 * t_1 / t * (t_1 - t) / (1.0 - ll.powi(5)) + 1.0
        } else {
            (2.0_f64.ln() * (t / t_0).ln() / (t_1 / t_0).ln()).exp() - 1.0
        }
    } else {
        let m_f = m as f64;
        let x_0l_num = ((m_f * PI + PI) / (8.0 * t)).powf(2.0 / 3.0);
        let x_0l = (x_0l_num - 1.0) / (x_0l_num + 1.0);
        let x_0r_num = ((8.0 * t) / (m_f * PI)).powf(2.0 / 3.0);
        let x_0r = (x_0r_num - 1.0) / (x_0r_num + 1.0);
        if low_path {
            x_0l.max(x_0r)
        } else {
            x_0l.min(x_0r)
        }
    }
}

/// Householder root-finder on the TOF equation (quartic convergence).
fn householder(p0_init: f64, t0: f64, ll: f64, m: u32, atol: f64, rtol: f64, maxiter: u32) -> f64 {
    let mut p0 = p0_init;
    // Seed p different from p0 so the convergence check doesn't short-circuit.
    let mut p = p0 * 1.1 + 0.01;
    for _ in 0..maxiter {
        if (p - p0).abs() < rtol * p0.abs() + atol {
            break;
        }
        p0 = p;
        let y = compute_y(p0, ll);
        let fval = tof_equation_y(p0, y, t0, ll, m);
        let t = fval + t0;
        let fder = tof_equation_p(p0, y, t, ll);
        let fder2 = tof_equation_p2(p0, y, t, fder, ll);
        let fder3 = tof_equation_p3(p0, y, fder, fder2, ll);

        let num = fder * fder - fval * fder2 / 2.0;
        let den = fder * (fder * fder - fval * fder2) + fder3 * fval * fval / 6.0;
        let safe_den = if den.abs() < 1e-15 {
            den.signum().max(1.0_f64.copysign(den)) * 1e-15
        } else {
            den
        };

        let mut delta = fval * (num / safe_den);
        // Step-size clamp to prevent divergence (matches JAX).
        let max_step = 0.1_f64.max(p0.abs());
        delta = delta.clamp(-max_step, max_step);
        p = p0 - delta;
    }
    p
}

/// Halley (cubic) minimum-finder for T_min computation in multi-rev paths.
fn halley(p0: f64, t0: f64, ll: f64, atol: f64, rtol: f64, maxiter: u32) -> f64 {
    let mut p_cur = p0;
    for _ in 0..maxiter {
        let y = compute_y(p_cur, ll);
        let fder = tof_equation_p(p_cur, y, t0, ll);
        let fder2 = tof_equation_p2(p_cur, y, t0, fder, ll);
        let fder3 = tof_equation_p3(p_cur, y, fder, fder2, ll);
        let p_new = p_cur - 2.0 * fder * fder2 / (2.0 * fder2 * fder2 - fder * fder3);
        if (p_new - p_cur).abs() < rtol * p_cur.abs() + atol {
            return p_new;
        }
        p_cur = p_new;
    }
    p_cur
}

/// Minimum T for a given ll, M. Three-way branch per Izzo.
fn compute_t_min(ll: f64, m: u32, maxiter: u32, atol: f64, rtol: f64) -> (f64, f64) {
    if (ll - 1.0).abs() < 1e-10 {
        let x = 0.0;
        (x, tof_equation(x, 0.0, ll, m))
    } else if m == 0 {
        (f64::INFINITY, 0.0)
    } else {
        let x_i = 0.1;
        let t_i = tof_equation(x_i, 0.0, ll, m);
        let x = halley(x_i, t_i, ll, atol, rtol, maxiter);
        (x, tof_equation(x, 0.0, ll, m))
    }
}

/// `_find_xy` — main driver. Returns (x, y) = NaN when no solution exists.
fn find_xy(
    ll: f64,
    t: f64,
    m: u32,
    maxiter: u32,
    atol: f64,
    rtol: f64,
    low_path: bool,
) -> (f64, f64) {
    let mut m_max = (t / PI).floor() as u32;
    let t_00 = ll.abs().acos() + ll.abs() * (1.0 - ll * ll).sqrt();

    let need_refine = t < t_00 + (m_max as f64) * PI && m_max > 0;
    if need_refine {
        let (_, t_min) = compute_t_min(ll, m_max, maxiter, atol, rtol);
        if t < t_min && m_max > 0 {
            m_max -= 1;
        }
    }

    if ll.abs() >= 1.0 || m > m_max {
        return (f64::NAN, f64::NAN);
    }

    let x_0 = initial_guess(t, ll, m, low_path);
    let x = householder(x_0, t, ll, m, atol, rtol, maxiter);
    let y = compute_y(x, ll);
    (x, y)
}

/// Izzo's Lambert solver for a single (r1, r2, tof) triplet.
///
/// Inputs:
/// - `r1`, `r2`: position vectors (AU)
/// - `tof`: time of flight (days; any consistent time unit works)
/// - `mu`: gravitational parameter (AU³/d² — or matching units)
/// - `m`: number of revolutions (0 for direct transfer)
/// - `prograde`: if true, prograde motion; else retrograde
/// - `low_path`: if two solutions available, select low vs high path
/// - `maxiter`, `atol`, `rtol`: convergence controls
///
/// Returns (v1, v2); each 3-vector velocity on the transfer orbit.
/// Returns NaN vectors if no solution exists.
#[allow(clippy::too_many_arguments)]
pub fn izzo_lambert(
    r1: [f64; 3],
    r2: [f64; 3],
    tof: f64,
    mu: f64,
    m: u32,
    prograde: bool,
    low_path: bool,
    maxiter: u32,
    atol: f64,
    rtol: f64,
) -> ([f64; 3], [f64; 3]) {
    let c = [r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]];
    let c_norm = norm3(c);
    let r1_norm = norm3(r1);
    let r2_norm = norm3(r2);
    let s = (r1_norm + r2_norm + c_norm) * 0.5;

    let i_r1 = [r1[0] / r1_norm, r1[1] / r1_norm, r1[2] / r1_norm];
    let i_r2 = [r2[0] / r2_norm, r2[1] / r2_norm, r2[2] / r2_norm];

    let i_h_raw = cross3(i_r1, i_r2);
    let i_h_norm = norm3(i_h_raw);
    let i_h_scale = if i_h_norm > 0.0 { i_h_norm } else { 1.0 };
    let i_h = [
        i_h_raw[0] / i_h_scale,
        i_h_raw[1] / i_h_scale,
        i_h_raw[2] / i_h_scale,
    ];

    let ll_base = (1.0 - (c_norm / s).min(1.0)).sqrt();
    let ll_sign = if i_h[2] < 0.0 { -1.0 } else { 1.0 };
    let mut ll = ll_base * ll_sign;

    let (mut i_t1, mut i_t2) = if i_h[2] < 0.0 {
        (cross3(i_r1, i_h), cross3(i_r2, i_h))
    } else {
        (cross3(i_h, i_r1), cross3(i_h, i_r2))
    };

    if !prograde {
        ll = -ll;
        i_t1 = [-i_t1[0], -i_t1[1], -i_t1[2]];
        i_t2 = [-i_t2[0], -i_t2[1], -i_t2[2]];
    }

    let t_ndim = (2.0 * mu / s.powi(3)).sqrt() * tof;

    let (x, y) = find_xy(ll, t_ndim, m, maxiter, atol, rtol, low_path);

    if x.is_nan() || y.is_nan() {
        let nan3 = [f64::NAN; 3];
        return (nan3, nan3);
    }

    let gamma = (mu * s / 2.0).sqrt();
    let rho = (r1_norm - r2_norm) / c_norm;
    let sigma = (1.0 - rho * rho).sqrt();

    let v_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1_norm;
    let v_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2_norm;
    let v_t1 = gamma * sigma * (y + ll * x) / r1_norm;
    let v_t2 = gamma * sigma * (y + ll * x) / r2_norm;

    let v1 = [
        v_r1 * i_r1[0] + v_t1 * i_t1[0],
        v_r1 * i_r1[1] + v_t1 * i_t1[1],
        v_r1 * i_r1[2] + v_t1 * i_t1[2],
    ];
    let v2 = [
        v_r2 * i_r2[0] + v_t2 * i_t2[0],
        v_r2 * i_r2[1] + v_t2 * i_t2[1],
        v_r2 * i_r2[2] + v_t2 * i_t2[2],
    ];

    (v1, v2)
}

/// Batched Lambert solver — rayon-parallel over rows.
///
/// Inputs all row-major N×3 flat arrays for r1, r2; tof is per-row.
/// Returns `(v1_flat, v2_flat)` each N×3.
#[allow(clippy::too_many_arguments)]
pub fn izzo_lambert_batch_flat(
    r1_flat: &[f64],
    r2_flat: &[f64],
    tof: &[f64],
    mu: f64,
    m: u32,
    prograde: bool,
    low_path: bool,
    maxiter: u32,
    atol: f64,
    rtol: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(r1_flat.len() % 3, 0);
    let n = r1_flat.len() / 3;
    assert_eq!(r2_flat.len(), n * 3);
    assert_eq!(tof.len(), n);

    let mut v1_out = vec![0.0_f64; n * 3];
    let mut v2_out = vec![0.0_f64; n * 3];

    v1_out
        .par_chunks_mut(3)
        .zip(v2_out.par_chunks_mut(3))
        .enumerate()
        .for_each(|(i, (v1_dst, v2_dst))| {
            let base = i * 3;
            let r1 = [r1_flat[base], r1_flat[base + 1], r1_flat[base + 2]];
            let r2 = [r2_flat[base], r2_flat[base + 1], r2_flat[base + 2]];
            let (v1, v2) = izzo_lambert(
                r1, r2, tof[i], mu, m, prograde, low_path, maxiter, atol, rtol,
            );
            v1_dst.copy_from_slice(&v1);
            v2_dst.copy_from_slice(&v2);
        });

    (v1_out, v2_out)
}

/// Porkchop trajectory grid: rayon-parallel batched Lambert over an N×M
/// meshgrid of (departure, arrival) pairs, filtered by arr_mjd > dep_mjd.
///
/// Inputs:
/// - `dep_states_flat`: (N*6,) row-major Cartesian state vectors at each
///   departure epoch (only the position columns are used for r1).
/// - `dep_mjds`: (N,) departure epoch MJDs.
/// - `arr_states_flat`: (M*6,) row-major arrival state vectors.
/// - `arr_mjds`: (M,) arrival epoch MJDs.
///
/// Returns `(dep_idx, arr_idx, v1_flat, v2_flat)` for the V valid pairs:
/// - `dep_idx[V]`, `arr_idx[V]`: u32 indices into the input grids.
/// - `v1_flat[V*3]`, `v2_flat[V*3]`: row-major Lambert velocity vectors.
///
/// The departure index varies fastest in the "natural" meshgrid layout
/// `(j, i)` (arrival outer, departure inner), matching numpy.meshgrid's
/// default behavior — but we sort pairs in `(i, j)` row-major order here
/// so callers iterate linearly.
#[allow(clippy::too_many_arguments)]
pub fn porkchop_grid_flat(
    dep_states_flat: &[f64],
    dep_mjds: &[f64],
    arr_states_flat: &[f64],
    arr_mjds: &[f64],
    mu: f64,
    prograde: bool,
    maxiter: u32,
    atol: f64,
    rtol: f64,
) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
    assert_eq!(dep_states_flat.len() % 6, 0);
    assert_eq!(arr_states_flat.len() % 6, 0);
    let n_dep = dep_states_flat.len() / 6;
    let n_arr = arr_states_flat.len() / 6;
    assert_eq!(dep_mjds.len(), n_dep);
    assert_eq!(arr_mjds.len(), n_arr);

    // Single sequential pass to enumerate valid pairs (small relative to
    // the Lambert work that follows).
    let mut valid_pairs: Vec<(u32, u32)> = Vec::with_capacity(n_dep * n_arr / 2);
    for (i, &dep_t) in dep_mjds.iter().enumerate() {
        for (j, &arr_t) in arr_mjds.iter().enumerate() {
            if arr_t > dep_t {
                valid_pairs.push((i as u32, j as u32));
            }
        }
    }

    let v = valid_pairs.len();
    let mut dep_idx_out = vec![0u32; v];
    let mut arr_idx_out = vec![0u32; v];
    let mut v1_out = vec![0.0_f64; v * 3];
    let mut v2_out = vec![0.0_f64; v * 3];

    v1_out
        .par_chunks_mut(3)
        .zip(v2_out.par_chunks_mut(3))
        .zip(dep_idx_out.par_iter_mut())
        .zip(arr_idx_out.par_iter_mut())
        .enumerate()
        .for_each(|(k, (((v1_dst, v2_dst), dep_dst), arr_dst))| {
            let (i, j) = valid_pairs[k];
            let i_us = i as usize;
            let j_us = j as usize;
            let r1 = [
                dep_states_flat[i_us * 6],
                dep_states_flat[i_us * 6 + 1],
                dep_states_flat[i_us * 6 + 2],
            ];
            let r2 = [
                arr_states_flat[j_us * 6],
                arr_states_flat[j_us * 6 + 1],
                arr_states_flat[j_us * 6 + 2],
            ];
            let tof = arr_mjds[j_us] - dep_mjds[i_us];
            let (v1, v2) = izzo_lambert(r1, r2, tof, mu, 0, prograde, true, maxiter, atol, rtol);
            v1_dst.copy_from_slice(&v1);
            v2_dst.copy_from_slice(&v2);
            *dep_dst = i;
            *arr_dst = j;
        });

    (dep_idx_out, arr_idx_out, v1_out, v2_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyp2f1b_at_zero_is_one() {
        // 2F1(3, 1, 5/2, 0) = 1.
        assert!((hyp2f1b(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn hyp2f1b_matches_jax_reference() {
        // Reference values from JAX's `_hyp2f1b` (100-term series).
        for (x, expected) in &[
            (0.0, 1.0),
            (0.1, 1.13542436662003),
            (0.3, 1.544424307841177),
            (0.5, 2.356194490192347),
            (0.7, 4.576573199723612),
        ] {
            let v = hyp2f1b(*x);
            assert!(
                (v - expected).abs() < 1e-12,
                "2F1(3,1,5/2,{x}): got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn lambert_single_rev_earth_to_mars_roundtrip() {
        // Famous Lambert example: Earth at t=0, Mars at t=tof (Curtis 5.6).
        // Position magnitudes in AU, tof in days, mu in AU³/d². Just check
        // that the solver produces finite, sensible-magnitude velocities.
        let r1 = [1.0, 0.0, 0.0];
        let r2 = [0.0, 1.524, 0.0];
        let tof = 200.0;
        let mu = 2.95912208284120e-4;
        let (v1, v2) = izzo_lambert(r1, r2, tof, mu, 0, true, true, 35, 1e-10, 1e-10);
        for v in v1.iter().chain(v2.iter()) {
            assert!(v.is_finite(), "velocity component {v} not finite");
        }
        // Rough speed ~ 0.02 AU/d (orbital speed scale).
        let v1_mag = norm3(v1);
        assert!(
            v1_mag > 0.005 && v1_mag < 0.1,
            "v1 magnitude {v1_mag} out of range"
        );
    }

    #[test]
    fn lambert_invalid_inputs_return_nan() {
        // tof = 0 is degenerate; should emit NaN.
        let r1 = [1.0, 0.0, 0.0];
        let r2 = [0.0, 1.0, 0.0];
        let (v1, _v2) = izzo_lambert(r1, r2, 0.0, 1e-4, 0, true, true, 35, 1e-10, 1e-10);
        // Either NaN or Infinity is acceptable for degenerate tof.
        assert!(!v1[0].is_finite() || v1[0].is_nan());
    }
}
