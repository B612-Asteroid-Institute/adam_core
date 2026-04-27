//! Fused Gauss IOD kernel — one Rust crossing for the entire public
//! `gaussIOD` body (equatorial-unit → ecliptic-unit rotation, coefficient
//! math, 8th-order polynomial roots, per-root orbit construction).
//!
//! Mirrors `adam_core.orbit_determination.gauss.gaussIOD` 1:1:
//!
//!   A  = |q2|^3 · (ρ̂1 × ρ̂3) · (t32·q1 − t31·q2 + t21·q3)
//!   B  = μ/6 · t32 · t21 · (ρ̂1 × ρ̂3) · ((t31+t32)·q1 + (t31+t21)·q3)
//!   V  = (ρ̂1 × ρ̂2) · ρ̂3
//!   C0 = V · t31 · |q2|^4 / B
//!   h0 = −A / B
//!   0  = C0² · r^8 − |q2|² (h0² + 2·C0·h0·cos(ε2) + C0²) · r^6
//!        + 2·|q2|^5 (h0 + C0·cos(ε2)) · r^3 − |q2|^8
//!
//! Positive real roots `r2_mag` feed
//! [`crate::gauss_iod_orbits_from_roots`] for per-root orbit assembly.
//!
//! Polynomial roots are found with Laguerre's method plus deflation
//! (globally convergent and robust to the wide coefficient dynamic
//! range of real-world gauss-IOD polynomials), followed by a real-axis
//! Newton polish so near-real roots snap to `im == 0.0` cleanly,
//! matching the numpy `np.isreal` filter semantics.

use crate::gauss_iod_orbits_from_roots;

const PI_F64: f64 = std::f64::consts::PI;
const OBLIQUITY_RAD: f64 = 84381.448_f64 * PI_F64 / (180.0_f64 * 3600.0_f64);
const DEG_TO_RAD: f64 = PI_F64 / 180.0_f64;

#[derive(Clone, Copy, Debug)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    #[inline]
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }

    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }

    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    #[inline]
    fn div(self, o: Self) -> Self {
        let denom = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / denom,
            (self.im * o.re - self.re * o.im) / denom,
        )
    }

    #[inline]
    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// Horner evaluation with first and second derivatives in a single pass
/// over complex coefficients. Returns `(p, p', p'')`.
#[inline]
fn poly_eval_with_derivs_c(coeffs: &[C64], z: C64) -> (C64, C64, C64) {
    let mut p = C64::new(0.0, 0.0);
    let mut pd = C64::new(0.0, 0.0);
    let mut pdd = C64::new(0.0, 0.0);
    for &c in coeffs {
        pdd = pdd.mul(z).add(pd);
        pd = pd.mul(z).add(p);
        p = p.mul(z).add(c);
    }
    pdd = C64::new(2.0, 0.0).mul(pdd);
    (p, pd, pdd)
}

/// Principal complex square root. Returns `w` such that `w * w == z`
/// with `w.re >= 0` (principal branch).
#[inline]
fn complex_sqrt(z: C64) -> C64 {
    let r = z.abs();
    if r == 0.0 {
        return C64::new(0.0, 0.0);
    }
    let re = ((r + z.re) * 0.5).max(0.0).sqrt();
    let mut im = ((r - z.re) * 0.5).max(0.0).sqrt();
    if z.im < 0.0 {
        im = -im;
    }
    C64::new(re, im)
}

/// Synthetic-divide `coeffs(x)` by `(x − root)`, returning the quotient
/// coefficients (one shorter). Works on complex coefficients so the
/// deflation can chain through complex roots.
fn synthetic_divide(coeffs: &[C64], root: C64) -> Vec<C64> {
    let mut out = Vec::with_capacity(coeffs.len() - 1);
    let mut carry = coeffs[0];
    for &c in &coeffs[1..] {
        out.push(carry);
        carry = carry.mul(root).add(c);
    }
    out
}

/// One root of a complex-coefficient polynomial via Laguerre's method.
/// Globally convergent for polynomials with real coefficients; cubic
/// near a simple root.
fn laguerre_one_root(coeffs: &[C64], z0: C64, max_iter: usize, tol: f64) -> C64 {
    let n = (coeffs.len() - 1) as f64;
    let mut z = z0;
    for _ in 0..max_iter {
        let (p, pd, pdd) = poly_eval_with_derivs_c(coeffs, z);
        if p.abs() == 0.0 {
            return z;
        }
        let g = pd.div(p);
        let h = g.mul(g).sub(pdd.div(p));
        // disc = (n − 1) · (n · h − g²)
        let n_c = C64::new(n, 0.0);
        let disc = n_c.mul(h).sub(g.mul(g)).mul(C64::new(n - 1.0, 0.0));
        let sq = complex_sqrt(disc);
        let denom_a = g.add(sq);
        let denom_b = g.sub(sq);
        let denom = if denom_a.abs() > denom_b.abs() {
            denom_a
        } else {
            denom_b
        };
        if denom.abs() == 0.0 {
            return z;
        }
        let a = n_c.div(denom);
        z = z.sub(a);
        if a.abs() < tol * z.abs().max(1.0) {
            return z;
        }
    }
    z
}

/// Find all 8 roots of a real-coefficient degree-8 polynomial via
/// Laguerre + deflation, with a final Newton polish on the original
/// (un-deflated) polynomial to re-tighten accuracy lost during deflation.
fn find_roots_laguerre_deg8(coeffs: &[f64]) -> [C64; 8] {
    debug_assert_eq!(coeffs.len(), 9);
    let mut working: Vec<C64> = coeffs.iter().map(|&c| C64::new(c, 0.0)).collect();
    let original: Vec<C64> = working.clone();
    let mut roots = [C64::new(0.0, 0.0); 8];
    // Starting guess: a small off-axis complex value that avoids the
    // real-axis stagnation of Laguerre on real-coefficient polynomials.
    let start = C64::new(0.1, 0.1);
    for root_slot in roots.iter_mut() {
        let root = laguerre_one_root(&working, start, 80, 1e-14);
        *root_slot = root;
        working = synthetic_divide(&working, root);
    }
    // Polish each root against the *original* polynomial — deflation
    // accumulates error (especially for the last few roots), so a few
    // Laguerre steps on the full polynomial restore full precision.
    for root_slot in roots.iter_mut() {
        *root_slot = laguerre_one_root(&original, *root_slot, 30, 1e-15);
    }
    roots
}

/// Evaluate a real polynomial and its derivative at a real point.
#[inline]
fn poly_eval_real_with_deriv(coeffs: &[f64], x: f64) -> (f64, f64) {
    // Horner for both value and derivative at once (classic trick).
    let mut val = 0.0_f64;
    let mut deriv = 0.0_f64;
    for &c in coeffs {
        deriv = deriv * x + val;
        val = val * x + c;
    }
    (val, deriv)
}

/// Polish a near-real root with a couple of Newton steps on the real
/// axis, then return the polished real root. Protects against 1/p'==0.
fn polish_real_root(coeffs: &[f64], x0: f64, max_iter: usize, tol: f64) -> f64 {
    let mut x = x0;
    for _ in 0..max_iter {
        let (p, pd) = poly_eval_real_with_deriv(coeffs, x);
        if pd.abs() < 1e-30 {
            break;
        }
        let step = p / pd;
        x -= step;
        if step.abs() < tol * x.abs().max(1.0) {
            break;
        }
    }
    x
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Fused gaussIOD kernel.
///
/// Inputs:
/// * `ra_deg`, `dec_deg`: three observation directions in equatorial
///   RA/Dec, degrees.
/// * `obs_times_mjd`: three observation times, MJD (any decimal-day
///   scale works as long as the three are consistent).
/// * `coords_obs`: three heliocentric observer Cartesian positions, in
///   AU, **ecliptic** frame. Caller is responsible for the frame
///   (matching the existing legacy gaussIOD contract where observer
///   coords come from `Observers` already in ecliptic).
/// * `velocity_method`: 0 = gauss, 1 = gibbs, 2 = herrick+gibbs.
/// * `light_time`: subtract light-time from epoch if true.
/// * `mu`, `c`: gravitational parameter (AU³/d²), speed of light (AU/d).
///
/// Returns `(epochs, orbits_flat)` where `orbits_flat` is 6-wide
/// Cartesian state per candidate.
#[allow(clippy::too_many_arguments)]
pub fn gauss_iod_fused(
    ra_deg: [f64; 3],
    dec_deg: [f64; 3],
    obs_times_mjd: [f64; 3],
    coords_obs: [[f64; 3]; 3],
    velocity_method: i32,
    light_time: bool,
    mu: f64,
    c: f64,
) -> (Vec<f64>, Vec<f64>) {
    // Step 1: RA/Dec → equatorial unit vector → ecliptic unit vector.
    //
    // Equatorial unit vector: [cos(δ)·cos(α), cos(δ)·sin(α), sin(δ)].
    // Then rotate equatorial → ecliptic via the J2000 obliquity matrix
    // (TRANSFORM_EQ2EC): [[1,0,0],[0,cos(ε),sin(ε)],[0,−sin(ε),cos(ε)]].
    let cos_o = OBLIQUITY_RAD.cos();
    let sin_o = OBLIQUITY_RAD.sin();

    let mut rho_hats = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        let ra = ra_deg[i] * DEG_TO_RAD;
        let dec = dec_deg[i] * DEG_TO_RAD;
        let cos_dec = dec.cos();
        let eq = [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()];
        let mut ec = [
            eq[0],
            cos_o * eq[1] + sin_o * eq[2],
            -sin_o * eq[1] + cos_o * eq[2],
        ];
        // Defensive renormalize (matches the python path).
        let n = norm(ec);
        if n > 0.0 {
            ec[0] /= n;
            ec[1] /= n;
            ec[2] /= n;
        }
        rho_hats[i] = ec;
    }

    let rho1_hat = rho_hats[0];
    let rho2_hat = rho_hats[1];
    let rho3_hat = rho_hats[2];
    let q1 = coords_obs[0];
    let q2 = coords_obs[1];
    let q3 = coords_obs[2];
    let q2_mag = norm(q2);

    let t1 = obs_times_mjd[0];
    let t2 = obs_times_mjd[1];
    let t3 = obs_times_mjd[2];
    let t31 = t3 - t1;
    let t21 = t2 - t1;
    let t32 = t3 - t2;

    // Step 2: Milani coefficients A, B, V and derived C0, h0.
    let q2_mag_cubed = q2_mag.powi(3);
    let cross_13 = cross(rho1_hat, rho3_hat);
    let vec_a = [
        t32 * q1[0] - t31 * q2[0] + t21 * q3[0],
        t32 * q1[1] - t31 * q2[1] + t21 * q3[1],
        t32 * q1[2] - t31 * q2[2] + t21 * q3[2],
    ];
    let a_coef = q2_mag_cubed * dot(cross_13, vec_a);

    let vec_b = [
        (t31 + t32) * q1[0] + (t31 + t21) * q3[0],
        (t31 + t32) * q1[1] + (t31 + t21) * q3[1],
        (t31 + t32) * q1[2] + (t31 + t21) * q3[2],
    ];
    let b_coef = (mu / 6.0) * t32 * t21 * dot(cross_13, vec_b);

    let v = dot(cross(rho1_hat, rho2_hat), rho3_hat);
    let coseps2 = dot(q2, rho2_hat) / q2_mag;
    let c0 = v * t31 * q2_mag.powi(4) / b_coef;
    let h0 = -a_coef / b_coef;

    if !c0.is_finite() || !h0.is_finite() {
        return (Vec::new(), Vec::new());
    }

    // Step 3: 8th-order polynomial coefficients (descending powers).
    //
    //   C0² r^8 + 0 r^7 − |q2|²·(h0² + 2·C0·h0·cos(ε2) + C0²) r^6
    //   + 0 r^5 + 0 r^4 + 2·|q2|^5·(h0 + C0·cos(ε2)) r^3
    //   + 0 r^2 + 0 r − |q2|^8.
    let coeffs = [
        c0 * c0,
        0.0,
        -q2_mag.powi(2) * (h0 * h0 + 2.0 * c0 * h0 * coseps2 + c0 * c0),
        0.0,
        0.0,
        2.0 * q2_mag.powi(5) * (h0 + c0 * coseps2),
        0.0,
        0.0,
        -q2_mag.powi(8),
    ];

    // Step 4: find roots via Laguerre + deflation. Laguerre is globally
    // convergent for real-coefficient polynomials and handles the wide
    // coefficient dynamic range of the gauss IOD polynomial (leading
    // coefficient C0² can be tiny relative to lower-order terms, yielding
    // roots at vastly different magnitudes — e.g. 0.86 and 37.6 AU).
    let roots = find_roots_laguerre_deg8(&coeffs);

    // Step 5: filter positive near-real roots, polishing each back to
    // the real axis so numerical residuals in im don't shift behaviour.
    //
    // Threshold: treat a root as real if |im| < 1e-8. This is tight
    // enough to reject genuinely complex roots (whose im comes from the
    // polynomial structure and is O(1) on well-posed inputs) yet loose
    // enough to admit all real roots even when Durand-Kerner only
    // reached 1e-12 convergence.
    let mut r2_mags: Vec<f64> = Vec::with_capacity(3);
    for root in roots.iter() {
        if root.im.abs() < 1e-8 && root.re >= 0.0 {
            let polished = polish_real_root(&coeffs, root.re, 6, 1e-14);
            if polished.is_finite() && polished >= 0.0 {
                r2_mags.push(polished);
            }
        }
    }

    if r2_mags.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Step 6: per-root orbit construction — reuse the already-Rust
    // implementation in gauss_iod_orbits_from_roots.
    gauss_iod_orbits_from_roots(
        &r2_mags,
        q1,
        q2,
        q3,
        rho1_hat,
        rho2_hat,
        rho3_hat,
        t1,
        t2,
        t3,
        v,
        velocity_method,
        light_time,
        mu,
        c,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_arithmetic_roundtrips() {
        let a = C64::new(3.0, 4.0);
        let b = C64::new(1.0, -2.0);
        let sum = a.add(b);
        assert!((sum.re - 4.0).abs() < 1e-15);
        assert!((sum.im - 2.0).abs() < 1e-15);
        let product = a.mul(b);
        // (3 + 4i)(1 − 2i) = 3 − 6i + 4i − 8i² = 11 − 2i
        assert!((product.re - 11.0).abs() < 1e-15);
        assert!((product.im + 2.0).abs() < 1e-15);
        let quotient = a.div(b);
        let back = quotient.mul(b);
        assert!((back.re - a.re).abs() < 1e-12);
        assert!((back.im - a.im).abs() < 1e-12);
    }

    #[test]
    fn polynomial_eval_matches_expansion() {
        // p(z) = z² − 3z + 2 = (z − 1)(z − 2). Roots at 1 and 2.
        let coeffs: Vec<C64> = [1.0, -3.0, 2.0].iter().map(|&c| C64::new(c, 0.0)).collect();
        let z1 = C64::new(1.0, 0.0);
        let z2 = C64::new(2.0, 0.0);
        let (p1, _, _) = poly_eval_with_derivs_c(&coeffs, z1);
        let (p2, _, _) = poly_eval_with_derivs_c(&coeffs, z2);
        assert!(p1.abs() < 1e-15);
        assert!(p2.abs() < 1e-15);
    }

    #[test]
    fn laguerre_finds_degree8_real_roots() {
        // (x − 1)(x − 2)(x − 3)(x − 4)(x² + 1)(x² + 4).
        // Roots: 1, 2, 3, 4 real; ±i, ±2i complex. Coefficients
        // expanded offline: [1, −10, 40, −100, 203, −290, 260, −200, 96].
        let coeffs = [1.0, -10.0, 40.0, -100.0, 203.0, -290.0, 260.0, -200.0, 96.0];
        let roots = find_roots_laguerre_deg8(&coeffs);
        let targets = [1.0, 2.0, 3.0, 4.0];
        let mut hits = [false; 4];
        for root in roots.iter() {
            if root.im.abs() < 1e-8 {
                for (idx, &t) in targets.iter().enumerate() {
                    if (root.re - t).abs() < 1e-8 {
                        hits[idx] = true;
                    }
                }
            }
        }
        assert!(
            hits.iter().all(|&h| h),
            "expected all four real roots recovered, got {roots:?}"
        );
    }

    #[test]
    fn polish_real_root_improves_accuracy() {
        // p(x) = x² − 2, root sqrt(2) ≈ 1.4142...
        let coeffs = [1.0, 0.0, -2.0];
        let x0 = 1.4;
        let x = polish_real_root(&coeffs, x0, 10, 1e-15);
        assert!((x - 2.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn laguerre_handles_real_world_gauss_iod_polynomial() {
        // Polynomial coefficients captured from an adam-core failing
        // triplet (RA=30.1, Dec=15.8 near ecliptic plane). Expected
        // positive real roots (via np.roots): {37.60, 0.861, 0.858}.
        let coeffs = [
            1.613481e-03_f64,
            0.0,
            -2.280941e+00,
            0.0,
            0.0,
            2.896208e+00,
            0.0,
            0.0,
            -9.198222e-01,
        ];
        let roots = find_roots_laguerre_deg8(&coeffs);
        let expected = [37.6_f64, 0.861, 0.858];
        let mut hits = [false; 3];
        for r in roots.iter() {
            if r.im.abs() < 1e-6 && r.re >= 0.0 {
                for (i, &e) in expected.iter().enumerate() {
                    if (r.re - e).abs() < 1e-2 {
                        hits[i] = true;
                    }
                }
            }
        }
        assert!(
            hits.iter().all(|&h| h),
            "expected roots {expected:?} all recovered; got {roots:?}"
        );
    }

    #[test]
    fn fused_path_handles_degenerate_inputs() {
        // All zero observer positions → q2_mag = 0 → h0 non-finite →
        // early return empty.
        let (epochs, orbits) = gauss_iod_fused(
            [10.0, 11.0, 12.0],
            [5.0, 5.1, 5.2],
            [59000.0, 59001.0, 59002.0],
            [[0.0; 3]; 3],
            1,
            true,
            2.95912208284120e-04,
            173.144632674,
        );
        assert!(epochs.is_empty());
        assert!(orbits.is_empty());
    }
}
