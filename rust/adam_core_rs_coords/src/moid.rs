//! Minimum Orbit Intersection Distance (MOID) between two orbits, fused
//! Rust kernel.
//!
//! Mirrors `adam_core.dynamics.moid.calculate_moid` — nested bounded
//! scalar minimization:
//!
//! 1. Outer loop over dt in [0, period] (or [0, 10000] for hyperbolic):
//!    propagate the primary orbit by dt via universal-anomaly 2-body,
//!    then compute distance from that point to the closest point on the
//!    secondary ellipse.
//!
//! 2. Inner loop over ellipse eccentric anomaly u in [0, 2π]: for a
//!    point P projected onto the secondary ellipse's plane, find the
//!    point on the ellipse that minimises ||P − r(u)||.
//!
//! Both minimizations use Brent's bounded method (matches scipy's
//! `minimize_scalar(method="bounded")`) so results are directly
//! comparable to the JAX/scipy reference.
//!
//! References:
//! - Hedo, J. M. et al. (2019). Minimum orbital intersection distance:
//!   an asymptotic approach. A&A 633, A22.

use crate::generic::cartesian_to_keplerian6;
use crate::propagate::propagate_2body_row;
use std::f64::consts::PI;

/// Brent's bounded scalar minimization — a stable, derivative-free
/// optimizer combining parabolic interpolation with golden-section
/// fallback. Matches scipy.optimize.minimize_scalar(method="bounded")
/// structurally (though not bit-identical due to different tolerance
/// wiring). Returns the minimizing x and its function value.
fn brent_bounded<F>(mut f: F, mut xa: f64, mut xb: f64, xtol: f64, max_iter: usize) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    // scipy's bounded routine uses a golden-section-seeded parabolic
    // interpolation; see SciPy _minimize_scalar_bounded. Constants match
    // that implementation for numerical consistency.
    let sqrt_eps = (f64::EPSILON).sqrt();
    let golden = 0.5 * (3.0 - 5.0_f64.sqrt());
    if xa > xb {
        std::mem::swap(&mut xa, &mut xb);
    }
    let mut fulc = xa + golden * (xb - xa);
    let mut nfc = fulc;
    let mut xf = fulc;
    let mut rat: f64 = 0.0;
    let mut e: f64 = 0.0;
    let x = xf;
    let mut fx = f(x);
    let mut fu = f64::INFINITY;
    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = 0.5 * (xa + xb);
    let mut tol1 = sqrt_eps * x.abs() + xtol / 3.0;
    let mut tol2 = 2.0 * tol1;

    for _ in 0..max_iter {
        if (xf - xm).abs() <= tol2 - 0.5 * (xb - xa) {
            break;
        }
        // Try parabolic step
        let mut golden_step = true;
        if e.abs() > tol1 {
            let r = (xf - nfc) * (fx - ffulc);
            let q_ = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q_ - (xf - nfc) * r;
            let mut q = 2.0 * (q_ - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let r_bak = e;
            e = rat;
            if p.abs() < (0.5 * q * r_bak).abs() && p > q * (xa - xf) && p < q * (xb - xf) {
                rat = p / q;
                let u = xf + rat;
                if (u - xa) < tol2 || (xb - u) < tol2 {
                    let si = if xm - xf >= 0.0 { 1.0 } else { -1.0 };
                    rat = tol1 * si;
                }
                golden_step = false;
            }
        }
        if golden_step {
            e = if xf >= xm { xa - xf } else { xb - xf };
            rat = golden * e;
        }
        let si = if rat >= 0.0 { 1.0 } else { -1.0 };
        let u = xf + si * rat.abs().max(tol1);
        fu = f(u);
        if fu <= fx {
            if u >= xf {
                xa = xf;
            } else {
                xb = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = u;
            fx = fu;
        } else {
            if u < xf {
                xa = u;
            } else {
                xb = u;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = u;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = u;
                ffulc = fu;
            }
        }
        xm = 0.5 * (xa + xb);
        tol1 = sqrt_eps * xf.abs() + xtol / 3.0;
        tol2 = 2.0 * tol1;
    }
    // Guard: if fu is still infinity (no iterations), fu == fx semantically.
    if !fu.is_finite() {
        fu = fx;
    }
    (xf, fx.min(fu))
}

/// Distance from coplanar point P to a point on the ellipse parameterized
/// by eccentric-anomaly-like u (as used in `coplanar_distance_to_ellipse`
/// in `adam_core.dynamics.moid`). Uses the Keplerian form where
/// `a, e, ω, Ω, M` are the last 5 of the 6 Keplerian elements but
/// `u` replaces mean-anomaly — matching the JAX code's convention.
#[inline]
fn coplanar_distance_to_ellipse(p: [f64; 3], a: f64, e: f64, u: f64) -> f64 {
    // Following the legacy JAX code exactly:
    //   E = 2 · atan(sqrt((1-e)/(1+e)) · tan(u/2))
    //   v = 2 · atan(sqrt((1+e)/(1-e)) · tan(u/2))
    //   r = a · (1 − e·cos(E))
    //   r_vec = [r·cos(v), r·sin(v), 0]
    //   distance = ||P − r_vec||
    let half_u = 0.5 * u;
    let tan_half_u = half_u.tan();
    let e_r1 = ((1.0 - e) / (1.0 + e)).sqrt();
    let e_r2 = ((1.0 + e) / (1.0 - e)).sqrt();
    let ecc_anom = 2.0 * (e_r1 * tan_half_u).atan();
    let true_anom = 2.0 * (e_r2 * tan_half_u).atan();
    let r = a * (1.0 - e * ecc_anom.cos());
    let rx = r * true_anom.cos();
    let ry = r * true_anom.sin();
    let dx = p[0] - rx;
    let dy = p[1] - ry;
    let dz = p[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Project a point P0 onto the plane of the primary ellipse. Requires
/// the primary ellipse's orbital-plane normal (h_hat).
#[inline]
fn project_point_on_plane(p0: [f64; 3], n_hat: [f64; 3]) -> [f64; 3] {
    let dot = p0[0] * n_hat[0] + p0[1] * n_hat[1] + p0[2] * n_hat[2];
    [
        p0[0] - dot * n_hat[0],
        p0[1] - dot * n_hat[1],
        p0[2] - dot * n_hat[2],
    ]
}

/// Full MOID kernel: minimum distance between two 2-body orbits.
///
/// Inputs:
/// - `primary_orbit`: 6-vector (x,y,z,vx,vy,vz) Cartesian state of the
///   primary orbit at its epoch.
/// - `secondary_orbit`: 6-vector of the secondary orbit at ITS epoch.
/// - `primary_epoch`, `secondary_epoch`: MJD epochs for propagation
///   reference. Primary epoch is used to bound the time-search range
///   via its orbital period.
/// - `mu`: gravitational parameter of the central body (AU³/d²).
/// - `max_iter`, `xtol`: bounded-minimization convergence controls.
///
/// Returns `(moid, dt_at_min)` — minimum distance and primary-orbit
/// propagation time (days since primary_epoch) where it occurs.
#[allow(clippy::too_many_arguments)]
pub fn calculate_moid(
    primary_orbit: [f64; 6],
    secondary_orbit: [f64; 6],
    mu: f64,
    max_iter: usize,
    xtol: f64,
) -> (f64, f64) {
    // Derive period and eccentricity of the PRIMARY orbit (used to bound
    // the outer dt search) from its Cartesian state. `cartesian_to_keplerian6`
    // returns the compact 6-tuple [a, e, i_deg, raan_deg, ap_deg, M_deg].
    let primary_kep = cartesian_to_keplerian6::<f64>(&primary_orbit, mu);
    let a_primary = primary_kep[0];
    let e_primary = primary_kep[1];
    // Period = 2π · sqrt(|a|³ / μ) for bound orbits; cap at 10000 days for
    // hyperbolic (matches legacy moid.py).
    let dt_upper = if e_primary < 1.0 && a_primary > 0.0 {
        2.0 * PI * (a_primary.powi(3) / mu).sqrt()
    } else {
        10_000.0
    };

    let sec_kep = cartesian_to_keplerian6::<f64>(&secondary_orbit, mu);
    let a_secondary = sec_kep[0];
    let e_secondary = sec_kep[1];

    // Orbital-plane normal of the SECONDARY orbit (for projecting points
    // onto its plane): h = r × v, normalized.
    let r_sec = [secondary_orbit[0], secondary_orbit[1], secondary_orbit[2]];
    let v_sec = [secondary_orbit[3], secondary_orbit[4], secondary_orbit[5]];
    let h_sec = [
        r_sec[1] * v_sec[2] - r_sec[2] * v_sec[1],
        r_sec[2] * v_sec[0] - r_sec[0] * v_sec[2],
        r_sec[0] * v_sec[1] - r_sec[1] * v_sec[0],
    ];
    let h_mag = (h_sec[0] * h_sec[0] + h_sec[1] * h_sec[1] + h_sec[2] * h_sec[2]).sqrt();
    let n_hat = if h_mag > 0.0 {
        [h_sec[0] / h_mag, h_sec[1] / h_mag, h_sec[2] / h_mag]
    } else {
        [0.0, 0.0, 1.0]
    };
    let _ = a_secondary; // silence unused-warning; only used for caller-facing docs

    // Distance from propagated primary point to secondary ellipse, as
    // a function of dt (the outer minimization variable).
    let distance_for_dt = |dt: f64| -> f64 {
        let propagated = propagate_2body_row::<f64>(primary_orbit, dt, mu, 100, 1e-14);
        let p0 = [propagated[0], propagated[1], propagated[2]];
        // Project P0 onto secondary plane.
        let p = project_point_on_plane(p0, n_hat);
        // Inner bounded-minimization over ellipse angle u in [0, 2π].
        let (_, d_parallel) = brent_bounded(
            |u| coplanar_distance_to_ellipse(p, a_secondary, e_secondary, u),
            0.0,
            2.0 * PI,
            xtol,
            max_iter,
        );
        // Recombine parallel + perpendicular distance (Hedo et al., Alg 1).
        let d_perp_sq = (p[0] - p0[0]).powi(2) + (p[1] - p0[1]).powi(2) + (p[2] - p0[2]).powi(2);
        (d_perp_sq + d_parallel * d_parallel).sqrt()
    };

    // Outer bounded minimization over dt in [0, period).
    let (dt_min, moid) = brent_bounded(distance_for_dt, 0.0, dt_upper, xtol, max_iter);
    let _ = a_primary; // silence unused-warning
    (moid, dt_min)
}

/// Batched MOID over N (primary, secondary) orbit pairs in parallel via rayon.
///
/// `primary_orbits`, `secondary_orbits`: flat (N*6,) row-major Cartesian states.
/// `mus`: (N,) gravitational parameters per primary.
/// Returns `(moids[N], dt_at_min[N])` as flat Vec<f64> pairs.
#[allow(clippy::too_many_arguments)]
pub fn calculate_moid_batch(
    primary_orbits: &[f64],
    secondary_orbits: &[f64],
    mus: &[f64],
    max_iter: usize,
    xtol: f64,
) -> (Vec<f64>, Vec<f64>) {
    use rayon::prelude::*;

    assert_eq!(primary_orbits.len() % 6, 0, "primary_orbits must be N*6");
    let n = primary_orbits.len() / 6;
    assert_eq!(secondary_orbits.len(), n * 6, "secondary_orbits must match");
    assert_eq!(mus.len(), n, "mus must have length N");

    let mut moids = vec![0.0_f64; n];
    let mut dts = vec![0.0_f64; n];
    moids
        .par_iter_mut()
        .zip(dts.par_iter_mut())
        .enumerate()
        .for_each(|(i, (moid_dst, dt_dst))| {
            let mut p = [0.0_f64; 6];
            let mut s = [0.0_f64; 6];
            p.copy_from_slice(&primary_orbits[i * 6..(i + 1) * 6]);
            s.copy_from_slice(&secondary_orbits[i * 6..(i + 1) * 6]);
            let (m, dt) = calculate_moid(p, s, mus[i], max_iter, xtol);
            *moid_dst = m;
            *dt_dst = dt;
        });
    (moids, dts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_finds_parabola_minimum() {
        // Minimize (x - 3)² + 5 on [-10, 10]; root at x = 3, value 5.
        let (x, fx) = brent_bounded(|x| (x - 3.0).powi(2) + 5.0, -10.0, 10.0, 1e-10, 100);
        assert!((x - 3.0).abs() < 1e-6, "x = {x}");
        assert!((fx - 5.0).abs() < 1e-10, "fx = {fx}");
    }

    #[test]
    fn coplanar_distance_is_nonneg_and_varies_with_u() {
        let p = [1.0, 0.5, 0.2];
        let d0 = coplanar_distance_to_ellipse(p, 1.5, 0.1, 0.0);
        let d1 = coplanar_distance_to_ellipse(p, 1.5, 0.1, 1.0);
        assert!(d0 >= 0.0 && d1 >= 0.0);
        assert!((d0 - d1).abs() > 1e-9, "distance should vary with u");
    }

    #[test]
    fn moid_is_finite_for_reasonable_orbits() {
        // Two asteroid-like orbits at different phases.
        let primary = [1.0, 0.0, 0.0, 0.0, 0.0172, 0.0];
        let secondary = [1.5, 0.3, 0.05, -0.002, 0.014, 0.001];
        let mu = 2.95912208284120e-4;
        let (moid, _) = calculate_moid(primary, secondary, mu, 100, 1e-10);
        assert!(
            moid.is_finite() && moid >= 0.0,
            "MOID {moid} not finite/nonneg"
        );
    }
}
