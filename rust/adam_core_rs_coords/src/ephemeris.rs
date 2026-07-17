//! Fused observer-frame ephemeris kernel.
//!
//! Mirrors `adam_core.dynamics.ephemeris._generate_ephemeris_2body` (light-time
//! Newton loop + optional stellar aberration + ec→eq rotation + cart→sph) as a
//! single `T: Scalar`-generic row body. The same code computes values with
//! `T = f64` and the full 6×6 output-wrt-input Jacobian with `T = Dual<6>` in
//! one forward pass — replacing the JAX `transform_covariances_jacobian` call
//! on the Rust path.
//!
//! All inputs are barycentric Cartesian in the ecliptic frame. The SUN→SSB
//! translation (when needed) is resolved Python-side before the crossing.
//!
//! Speed of light: `C = 299_792.458 km/s / KM_P_AU * S_P_DAY` au/day, matching
//! `adam_core.constants.Constants.C`.

use crate::propagate::{propagate_2body_from_consts, OrbitConstants};
use adam_core_rs_autodiff::{Dual, Scalar};
use rayon::prelude::*;

use crate::generic::{cartesian_to_spherical6, rotate_ecliptic_to_equatorial6};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::generic::{OBLIQUITY_COS_F64, OBLIQUITY_SIN_F64, RAD2DEG_F64, TWO_PI_F64};

/// Speed of light in AU/day — bit-identical to `Constants.C` in Python:
/// `299_792.458 / 149_597_870.700 * 86_400.0`.
pub const C_AU_PER_DAY: f64 = 299_792.458_f64 / 149_597_870.700_f64 * 86_400.0_f64;

/// Default maximum iterations for the outer light-time Newton loop, matching
/// the JAX default (`_add_light_time` uses `max_lt_iter=10`).
pub const DEFAULT_MAX_LT_ITER: usize = 10;

type EphemerisWarmStart<T> = ([T; 6], T, T, T);
type EphemerisRowResult<T> = ([T; 6], T, [T; 6], Option<EphemerisWarmStart<T>>);

#[inline]
fn rayon_is_single_threaded() -> bool {
    rayon::current_num_threads() == 1
}

#[inline]
fn stellar_aberrate<T: Scalar>(topo: &mut [T; 6], observer_state: &[T; 6]) {
    // Urban & Seidelmann (2013) eq. 7.40, matching `add_stellar_aberration`
    // in `adam_core.dynamics.aberrations`. Only the position component is
    // modified; velocity is untouched.
    let inv_c = T::from_f64(1.0 / C_AU_PER_DAY);
    let gx = observer_state[3] * inv_c;
    let gy = observer_state[4] * inv_c;
    let gz = observer_state[5] * inv_c;

    let gamma_sq = gx * gx + gy * gy + gz * gz;
    let beta_inv = (T::from_f64(1.0) - gamma_sq).sqrt();
    let delta = (topo[0] * topo[0] + topo[1] * topo[1] + topo[2] * topo[2]).sqrt();

    let inv_delta = T::from_f64(1.0) / delta;
    let rho_x = topo[0] * inv_delta;
    let rho_y = topo[1] * inv_delta;
    let rho_z = topo[2] * inv_delta;

    let rho_dot_gamma = rho_x * gx + rho_y * gy + rho_z * gz;
    let one = T::from_f64(1.0);
    let denom = one + rho_dot_gamma;
    let gamma_scale = rho_dot_gamma / (one + beta_inv);

    let num_x = beta_inv * rho_x + gx + gamma_scale * gx;
    let num_y = beta_inv * rho_y + gy + gamma_scale * gy;
    let num_z = beta_inv * rho_z + gz + gamma_scale * gz;

    topo[0] = delta * num_x / denom;
    topo[1] = delta * num_y / denom;
    topo[2] = delta * num_z / denom;
}

pub fn apply_stellar_aberration_row(topo: [f64; 6], observer_state: [f64; 6]) -> [f64; 6] {
    let mut out = topo;
    stellar_aberrate::<f64>(&mut out, &observer_state);
    out
}

/// Single-row light-time correction.
///
/// Iteratively back-propagates `orbit` by `lt = ||r_orbit - r_observer|| / C`
/// until the LT estimate stabilizes (matches Python `_add_light_time`).
/// Returns `(aberrated_orbit[6], light_time_days)`. NaN light_time on
/// non-convergence within `max_lt_iter` iterations.
///
/// `orbit` and `observer_pos` are barycentric Cartesian (AU); only the
/// position columns of the observer are used (this kernel does not apply
/// stellar aberration — that's a separable downstream step).
pub fn add_light_time_row<T: Scalar>(
    orbit: [T; 6],
    observer_pos: [T; 3],
    mu: T,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    max_lt_iter: usize,
) -> ([T; 6], T) {
    let orbit_consts = OrbitConstants::new(
        [orbit[0], orbit[1], orbit[2]],
        [orbit[3], orbit[4], orbit[5]],
        mu,
    );
    let mut orbit_i = orbit;
    let mut lt = T::from_f64(1.0e30);
    let mut dlt_val = 1.0e30_f64;
    let mut iter = 0_usize;
    let mut prev_dt = T::from_f64(0.0);
    let mut prev_chi = T::from_f64(0.0);
    let mut have_prev_chi = false;
    while dlt_val > lt_tol && iter < max_lt_iter {
        let dx = orbit_i[0] - observer_pos[0];
        let dy = orbit_i[1] - observer_pos[1];
        let dz = orbit_i[2] - observer_pos[2];
        let rho = (dx * dx + dy * dy + dz * dz).sqrt();
        let lt_new = rho / T::from_f64(C_AU_PER_DAY);
        dlt_val = (lt_new.re() - lt.re()).abs();
        let neg_lt = T::from_f64(0.0) - lt_new;
        let chi_init = if have_prev_chi {
            prev_chi + orbit_consts.sqrt_mu * orbit_consts.alpha.abs() * (neg_lt - prev_dt)
        } else {
            orbit_consts.default_chi_init(neg_lt)
        };
        let (next_orbit, chi) =
            propagate_2body_from_consts::<T>(&orbit_consts, neg_lt, chi_init, max_iter, tol);
        orbit_i = next_orbit;
        prev_dt = neg_lt;
        prev_chi = chi;
        have_prev_chi = true;
        lt = lt_new;
        iter += 1;
    }
    let lt_out = if iter >= max_lt_iter {
        T::from_f64(f64::NAN)
    } else {
        lt
    };
    (orbit_i, lt_out)
}

/// Batched light-time correction over N orbits/observer-position pairs.
/// Rayon-parallel.
///
/// Returns `(aberrated_flat[N*6], light_time[N])`.
#[allow(clippy::too_many_arguments)]
pub fn add_light_time_batch_flat(
    orbits_flat: &[f64],
    observer_pos_flat: &[f64],
    mus: &[f64],
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    max_lt_iter: usize,
) -> (Vec<f64>, Vec<f64>) {
    use rayon::prelude::*;

    assert_eq!(orbits_flat.len() % 6, 0);
    let n = orbits_flat.len() / 6;
    assert_eq!(observer_pos_flat.len(), n * 3);
    assert_eq!(mus.len(), n);

    let mut aberrated_flat = vec![0.0_f64; n * 6];
    let mut lts = vec![0.0_f64; n];
    if rayon_is_single_threaded() {
        for i in 0..n {
            let mut orbit = [0.0_f64; 6];
            orbit.copy_from_slice(&orbits_flat[i * 6..(i + 1) * 6]);
            let observer_pos = [
                observer_pos_flat[i * 3],
                observer_pos_flat[i * 3 + 1],
                observer_pos_flat[i * 3 + 2],
            ];
            let (ab, lt) = add_light_time_row::<f64>(
                orbit,
                observer_pos,
                mus[i],
                lt_tol,
                max_iter,
                tol,
                max_lt_iter,
            );
            aberrated_flat[i * 6..(i + 1) * 6].copy_from_slice(&ab);
            lts[i] = lt;
        }
        return (aberrated_flat, lts);
    }

    aberrated_flat
        .par_chunks_mut(6)
        .zip(lts.par_iter_mut())
        .enumerate()
        .for_each(|(i, (ab_dst, lt_dst))| {
            let mut orbit = [0.0_f64; 6];
            orbit.copy_from_slice(&orbits_flat[i * 6..(i + 1) * 6]);
            let observer_pos = [
                observer_pos_flat[i * 3],
                observer_pos_flat[i * 3 + 1],
                observer_pos_flat[i * 3 + 2],
            ];
            let (ab, lt) = add_light_time_row::<f64>(
                orbit,
                observer_pos,
                mus[i],
                lt_tol,
                max_iter,
                tol,
                max_lt_iter,
            );
            ab_dst.copy_from_slice(&ab);
            *lt_dst = lt;
        });
    (aberrated_flat, lts)
}

/// Single-row fused ephemeris kernel.
///
/// Returns `(spherical_equatorial[6], light_time_days, aberrated_cart[6])`.
/// If the light-time Newton loop fails to converge within `max_lt_iter`
/// iterations to `lt_tol`, `light_time_days` is set to NaN (matching JAX).
/// The spherical and aberrated outputs are still populated from the last
/// partial iterate so the batched wrapper can surface row-level context.
///
/// Inputs are barycentric Cartesian in the ecliptic frame. `observer_state`
/// must include velocity — stellar aberration uses `observer_state[3..6]`.
pub fn generate_ephemeris_2body_row<T: Scalar>(
    orbit: [T; 6],
    observer_state: [T; 6],
    mu: T,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> ([T; 6], T, [T; 6]) {
    let orbit_consts = OrbitConstants::new(
        [orbit[0], orbit[1], orbit[2]],
        [orbit[3], orbit[4], orbit[5]],
        mu,
    );
    let (spherical, lt, aberrated, _) = generate_ephemeris_2body_row_from_consts(
        &orbit_consts,
        observer_state,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
        None,
    );
    (spherical, lt, aberrated)
}

#[allow(clippy::too_many_arguments)]
fn generate_ephemeris_2body_row_from_consts<T: Scalar>(
    orbit_consts: &OrbitConstants<T>,
    observer_state: [T; 6],
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
    warm_start: Option<EphemerisWarmStart<T>>,
) -> EphemerisRowResult<T> {
    // Light-time Newton fixed point — mirrors `_add_light_time` exactly:
    //   p = [orbit_i, t_bookkeep, lt0=1e30, dlt=1e30, iter=0]
    //   while dlt > lt_tol and iter < max_lt_iter:
    //       rho = ||orbit_i[:3] - observer[:3]||
    //       lt  = rho / C
    //       dlt = |lt - lt0|
    //       orbit_i = propagate_2body(orbit_original, dt = -lt)   # always ORIGINAL orbit
    //       lt0 = lt;  iter += 1
    // Convergence is tested on `lt.re()` so `Dual` tangents ride the final iterate.
    let mut orbit_i = [
        orbit_consts.r[0],
        orbit_consts.r[1],
        orbit_consts.r[2],
        orbit_consts.v[0],
        orbit_consts.v[1],
        orbit_consts.v[2],
    ];
    let mut lt = T::from_f64(1.0e30);
    let mut dlt_val = 1.0e30_f64;
    let mut iter = 0_usize;
    let mut prev_dt = T::from_f64(0.0);
    let mut prev_chi = T::from_f64(0.0);
    let mut have_prev_chi = false;
    let mut updated_prev_chi = false;
    if let Some((warm_orbit, warm_lt, warm_dt, warm_chi)) = warm_start {
        orbit_i = warm_orbit;
        lt = warm_lt;
        prev_dt = warm_dt;
        prev_chi = warm_chi;
        have_prev_chi = true;
    }

    while dlt_val > lt_tol && iter < max_lt_iter {
        let dx = orbit_i[0] - observer_state[0];
        let dy = orbit_i[1] - observer_state[1];
        let dz = orbit_i[2] - observer_state[2];
        let rho = (dx * dx + dy * dy + dz * dz).sqrt();
        let lt_new = rho / T::from_f64(C_AU_PER_DAY);
        dlt_val = (lt_new.re() - lt.re()).abs();
        // Propagate the ORIGINAL orbit by dt = -lt_new. This matches the JAX
        // `_propagate_2body(orbit, t0, t0 - lt)` call: orbit is the closure
        // variable holding the original state, and the propagator uses only
        // `dt = t1 - t0 = -lt` as its time argument.
        let neg_lt = T::from_f64(0.0) - lt_new;
        let chi_init = if have_prev_chi {
            prev_chi + orbit_consts.sqrt_mu * orbit_consts.alpha.abs() * (neg_lt - prev_dt)
        } else {
            orbit_consts.default_chi_init(neg_lt)
        };
        let (next_orbit, chi) =
            propagate_2body_from_consts::<T>(orbit_consts, neg_lt, chi_init, max_iter, tol);
        orbit_i = next_orbit;
        prev_dt = neg_lt;
        prev_chi = chi;
        have_prev_chi = true;
        updated_prev_chi = true;
        lt = lt_new;
        iter += 1;
    }

    // Topocentric state (6-wide: pos - pos, vel - vel).
    let mut topo = [
        orbit_i[0] - observer_state[0],
        orbit_i[1] - observer_state[1],
        orbit_i[2] - observer_state[2],
        orbit_i[3] - observer_state[3],
        orbit_i[4] - observer_state[4],
        orbit_i[5] - observer_state[5],
    ];

    if stellar_aberration {
        stellar_aberrate::<T>(&mut topo, &observer_state);
    }

    let topo_eq = rotate_ecliptic_to_equatorial6::<T>(&topo);
    let spherical = cartesian_to_spherical6::<T>(&topo_eq);

    // Non-convergence NaN policy: matches JAX
    //   lt = jnp.where((dlt > lt_tol) | (iter >= max_lt_iter), NaN, lt).
    // With the loop exit condition `!(dlt > tol && iter < max)`, the only way
    // iter reaches max is via the second clause, so `iter >= max_lt_iter` is
    // the single test that captures both JAX branches.
    let converged = iter < max_lt_iter;
    let lt_out = if converged { lt } else { T::from_f64(f64::NAN) };
    let next_warm_start = if converged && updated_prev_chi {
        Some((orbit_i, lt, prev_dt, prev_chi))
    } else {
        None
    };

    (spherical, lt_out, orbit_i, next_warm_start)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod vforce {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn vvsqrt(out: *mut f64, input: *const f64, n: *const i32);
        fn vvasin(out: *mut f64, input: *const f64, n: *const i32);
        fn vvatan2(out: *mut f64, y: *const f64, x: *const f64, n: *const i32);
    }

    #[inline]
    fn len_i32(len: usize) -> i32 {
        i32::try_from(len).expect("vForce slice length must fit in i32")
    }

    #[inline]
    pub(super) fn sqrt_in_place(values: &mut [f64]) {
        let n = len_i32(values.len());
        // SAFETY: `values` is a unique mutable slice; vForce reads and writes
        // exactly `n` doubles in place.
        unsafe { vvsqrt(values.as_mut_ptr(), values.as_ptr(), &n) };
    }

    #[inline]
    pub(super) fn asin_in_place(values: &mut [f64]) {
        let n = len_i32(values.len());
        // SAFETY: see `sqrt_in_place`.
        unsafe { vvasin(values.as_mut_ptr(), values.as_ptr(), &n) };
    }

    #[inline]
    pub(super) fn atan2_into(y: &[f64], x: &[f64], output: &mut [f64]) {
        debug_assert_eq!(y.len(), x.len());
        debug_assert_eq!(y.len(), output.len());
        let n = len_i32(output.len());
        // SAFETY: all slices are valid for `n` doubles; the output slice is
        // distinct from the scratch input slices used by callers.
        unsafe { vvatan2(output.as_mut_ptr(), y.as_ptr(), x.as_ptr(), &n) };
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fill_spherical_from_topo_eq_flat_macos(topo_eq_sph_out: &mut [f64]) {
    let n = topo_eq_sph_out.len() / 6;
    let mut xs = vec![0.0_f64; n];
    let mut ys = vec![0.0_f64; n];
    let mut xy = vec![0.0_f64; n];
    let mut rho = vec![0.0_f64; n];
    let mut lat = vec![0.0_f64; n];
    let mut lon = vec![0.0_f64; n];

    for i in 0..n {
        let base = i * 6;
        let x = topo_eq_sph_out[base];
        let y = topo_eq_sph_out[base + 1];
        let z = topo_eq_sph_out[base + 2];
        xs[i] = x;
        ys[i] = y;
        let xy2 = x * x + y * y;
        xy[i] = xy2;
        rho[i] = xy2 + z * z;
    }

    vforce::sqrt_in_place(&mut xy);
    vforce::sqrt_in_place(&mut rho);

    for i in 0..n {
        let row_rho = rho[i];
        lat[i] = if row_rho == 0.0 {
            0.0
        } else {
            topo_eq_sph_out[i * 6 + 2] / row_rho
        };
    }

    vforce::asin_in_place(&mut lat);
    vforce::atan2_into(&ys, &xs, &mut lon);

    for i in 0..n {
        let base = i * 6;
        let x = xs[i];
        let y = ys[i];
        let z = topo_eq_sph_out[base + 2];
        let vx = topo_eq_sph_out[base + 3];
        let vy = topo_eq_sph_out[base + 4];
        let vz = topo_eq_sph_out[base + 5];
        let row_rho = rho[i];
        let row_xy = xy[i];
        let xy2 = row_xy * row_xy;
        let mut row_lon = lon[i];
        if row_lon < 0.0 {
            row_lon += TWO_PI_F64;
        }
        let vrho = if row_rho == 0.0 {
            0.0
        } else {
            (x * vx + y * vy + z * vz) / row_rho
        };
        let vlon = if xy2 == 0.0 {
            0.0
        } else {
            (vy * x - vx * y) / xy2
        };
        let vlat = if xy2 == 0.0 || row_rho == 0.0 {
            0.0
        } else {
            (vz - vrho * z / row_rho) / row_xy
        };
        topo_eq_sph_out[base] = row_rho;
        topo_eq_sph_out[base + 1] = row_lon * RAD2DEG_F64;
        topo_eq_sph_out[base + 2] = lat[i] * RAD2DEG_F64;
        topo_eq_sph_out[base + 3] = vrho;
        topo_eq_sph_out[base + 4] = vlon * RAD2DEG_F64;
        topo_eq_sph_out[base + 5] = vlat * RAD2DEG_F64;
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn generate_ephemeris_2body_row_topo_eq_no_stellar_f64(
    orbit_consts: &OrbitConstants<f64>,
    observer_state: [f64; 6],
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    max_lt_iter: usize,
    warm_start: Option<EphemerisWarmStart<f64>>,
) -> EphemerisRowResult<f64> {
    let mut orbit_i = [
        orbit_consts.r[0],
        orbit_consts.r[1],
        orbit_consts.r[2],
        orbit_consts.v[0],
        orbit_consts.v[1],
        orbit_consts.v[2],
    ];
    let mut lt = 1.0e30_f64;
    let mut dlt_val = 1.0e30_f64;
    let mut iter = 0_usize;
    let mut prev_dt = 0.0_f64;
    let mut prev_chi = 0.0_f64;
    let mut have_prev_chi = false;
    let mut updated_prev_chi = false;
    if let Some((warm_orbit, warm_lt, warm_dt, warm_chi)) = warm_start {
        orbit_i = warm_orbit;
        lt = warm_lt;
        prev_dt = warm_dt;
        prev_chi = warm_chi;
        have_prev_chi = true;
    }

    while dlt_val > lt_tol && iter < max_lt_iter {
        let dx = orbit_i[0] - observer_state[0];
        let dy = orbit_i[1] - observer_state[1];
        let dz = orbit_i[2] - observer_state[2];
        let rho = (dx * dx + dy * dy + dz * dz).sqrt();
        let lt_new = rho / C_AU_PER_DAY;
        dlt_val = (lt_new - lt).abs();
        let neg_lt = -lt_new;
        let chi_init = if have_prev_chi {
            prev_chi + orbit_consts.sqrt_mu * orbit_consts.alpha.abs() * (neg_lt - prev_dt)
        } else {
            orbit_consts.default_chi_init(neg_lt)
        };
        let (next_orbit, chi) =
            propagate_2body_from_consts::<f64>(orbit_consts, neg_lt, chi_init, max_iter, tol);
        orbit_i = next_orbit;
        prev_dt = neg_lt;
        prev_chi = chi;
        have_prev_chi = true;
        updated_prev_chi = true;
        lt = lt_new;
        iter += 1;
    }

    let topo_x = orbit_i[0] - observer_state[0];
    let topo_y = orbit_i[1] - observer_state[1];
    let topo_z = orbit_i[2] - observer_state[2];
    let topo_vx = orbit_i[3] - observer_state[3];
    let topo_vy = orbit_i[4] - observer_state[4];
    let topo_vz = orbit_i[5] - observer_state[5];
    let topo_eq = [
        topo_x,
        OBLIQUITY_COS_F64 * topo_y - OBLIQUITY_SIN_F64 * topo_z,
        OBLIQUITY_SIN_F64 * topo_y + OBLIQUITY_COS_F64 * topo_z,
        topo_vx,
        OBLIQUITY_COS_F64 * topo_vy - OBLIQUITY_SIN_F64 * topo_vz,
        OBLIQUITY_SIN_F64 * topo_vy + OBLIQUITY_COS_F64 * topo_vz,
    ];

    let converged = iter < max_lt_iter;
    let lt_out = if converged { lt } else { f64::NAN };
    let next_warm_start = if converged && updated_prev_chi {
        Some((orbit_i, lt, prev_dt, prev_chi))
    } else {
        None
    };

    (topo_eq, lt_out, orbit_i, next_warm_start)
}

/// Single-row kernel that also returns the 6×6 Jacobian of the spherical
/// output w.r.t. the input Cartesian orbit state. `observer_state`, `mu`,
/// and the observation time are treated as constants (zero tangents).
pub(crate) fn generate_ephemeris_2body_with_jacobian_row(
    orbit: [f64; 6],
    observer_state: [f64; 6],
    mu: f64,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> ([f64; 6], f64, [f64; 6], [[f64; 6]; 6]) {
    let orbit_d: [Dual<6>; 6] = Dual::seed(orbit);
    let observer_d: [Dual<6>; 6] = [
        Dual::<6>::constant(observer_state[0]),
        Dual::<6>::constant(observer_state[1]),
        Dual::<6>::constant(observer_state[2]),
        Dual::<6>::constant(observer_state[3]),
        Dual::<6>::constant(observer_state[4]),
        Dual::<6>::constant(observer_state[5]),
    ];
    let mu_d = Dual::<6>::constant(mu);
    let (sph_d, lt_d, aberrated_d) = generate_ephemeris_2body_row::<Dual<6>>(
        orbit_d,
        observer_d,
        mu_d,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
    );
    let mut sph = [0.0_f64; 6];
    let mut aberrated = [0.0_f64; 6];
    let mut jac = [[0.0_f64; 6]; 6];
    for (i, row) in jac.iter_mut().enumerate() {
        sph[i] = sph_d[i].re;
        aberrated[i] = aberrated_d[i].re;
        for (j, dst) in row.iter_mut().enumerate() {
            *dst = sph_d[i].du[j];
        }
    }
    (sph, lt_d.re, aberrated, jac)
}

fn rows_share_orbit_and_mu(orbits_flat: &[f64], mus: &[f64], left: usize, right: usize) -> bool {
    if mus[left].to_bits() != mus[right].to_bits() {
        return false;
    }
    let left_base = left * 6;
    let right_base = right * 6;
    (0..6).all(|k| orbits_flat[left_base + k].to_bits() == orbits_flat[right_base + k].to_bits())
}

fn has_consecutive_repeated_orbits(orbits_flat: &[f64], mus: &[f64], n: usize) -> bool {
    (1..n).any(|i| rows_share_orbit_and_mu(orbits_flat, mus, i - 1, i))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn generate_ephemeris_2body_grouped_single_thread_no_stellar_macos(
    orbits_flat: &[f64],
    observer_states_flat: &[f64],
    mus: &[f64],
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    max_lt_iter: usize,
    sph_out: &mut [f64],
    lt_out: &mut [f64],
    aberrated_out: &mut [f64],
) {
    let n = orbits_flat.len() / 6;
    let mut start = 0_usize;
    while start < n {
        let mut end = start + 1;
        while end < n && rows_share_orbit_and_mu(orbits_flat, mus, start, end) {
            end += 1;
        }

        let base = start * 6;
        let orbit_consts = OrbitConstants::new(
            [
                orbits_flat[base],
                orbits_flat[base + 1],
                orbits_flat[base + 2],
            ],
            [
                orbits_flat[base + 3],
                orbits_flat[base + 4],
                orbits_flat[base + 5],
            ],
            mus[start],
        );
        let mut warm_start: Option<EphemerisWarmStart<f64>> = None;
        for (offset, lt_slot) in lt_out[start..end].iter_mut().enumerate() {
            let i = start + offset;
            let row_base = i * 6;
            let observer: [f64; 6] = [
                observer_states_flat[row_base],
                observer_states_flat[row_base + 1],
                observer_states_flat[row_base + 2],
                observer_states_flat[row_base + 3],
                observer_states_flat[row_base + 4],
                observer_states_flat[row_base + 5],
            ];
            let (topo_eq, lt, aberrated, next_warm_start) =
                generate_ephemeris_2body_row_topo_eq_no_stellar_f64(
                    &orbit_consts,
                    observer,
                    lt_tol,
                    max_iter,
                    tol,
                    max_lt_iter,
                    warm_start,
                );
            sph_out[row_base..row_base + 6].copy_from_slice(&topo_eq);
            *lt_slot = lt;
            aberrated_out[row_base..row_base + 6].copy_from_slice(&aberrated);
            warm_start = next_warm_start;
        }
        start = end;
    }

    fill_spherical_from_topo_eq_flat_macos(sph_out);
}

#[allow(clippy::too_many_arguments)]
fn generate_ephemeris_2body_grouped_single_thread(
    orbits_flat: &[f64],
    observer_states_flat: &[f64],
    mus: &[f64],
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
    sph_out: &mut [f64],
    lt_out: &mut [f64],
    aberrated_out: &mut [f64],
) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if !stellar_aberration {
        generate_ephemeris_2body_grouped_single_thread_no_stellar_macos(
            orbits_flat,
            observer_states_flat,
            mus,
            lt_tol,
            max_iter,
            tol,
            max_lt_iter,
            sph_out,
            lt_out,
            aberrated_out,
        );
        return;
    }

    let n = orbits_flat.len() / 6;
    let mut start = 0_usize;
    while start < n {
        let mut end = start + 1;
        while end < n && rows_share_orbit_and_mu(orbits_flat, mus, start, end) {
            end += 1;
        }

        let base = start * 6;
        let orbit_consts = OrbitConstants::new(
            [
                orbits_flat[base],
                orbits_flat[base + 1],
                orbits_flat[base + 2],
            ],
            [
                orbits_flat[base + 3],
                orbits_flat[base + 4],
                orbits_flat[base + 5],
            ],
            mus[start],
        );
        let mut warm_start: Option<EphemerisWarmStart<f64>> = None;
        for (offset, lt_slot) in lt_out[start..end].iter_mut().enumerate() {
            let i = start + offset;
            let row_base = i * 6;
            let observer: [f64; 6] = [
                observer_states_flat[row_base],
                observer_states_flat[row_base + 1],
                observer_states_flat[row_base + 2],
                observer_states_flat[row_base + 3],
                observer_states_flat[row_base + 4],
                observer_states_flat[row_base + 5],
            ];
            let (sph, lt, aberrated, next_warm_start) =
                generate_ephemeris_2body_row_from_consts::<f64>(
                    &orbit_consts,
                    observer,
                    lt_tol,
                    max_iter,
                    tol,
                    stellar_aberration,
                    max_lt_iter,
                    warm_start,
                );
            sph_out[row_base..row_base + 6].copy_from_slice(&sph);
            *lt_slot = lt;
            aberrated_out[row_base..row_base + 6].copy_from_slice(&aberrated);
            warm_start = next_warm_start;
        }
        start = end;
    }
}

/// Batched state-only ephemeris. Rayon-parallel over rows.
///
/// Returns `(spherical_flat [N*6], light_time [N], aberrated_flat [N*6])`.
/// Rows whose LT Newton fails to converge emit NaN in `light_time[i]`; the
/// spherical / aberrated outputs contain whatever the last partial iterate
/// produced (mirrors JAX — the host-side error pass keys on NaN light-time).
#[allow(clippy::too_many_arguments)]
pub fn generate_ephemeris_2body_flat6(
    orbits_flat: &[f64],
    observer_states_flat: &[f64],
    mus: &[f64],
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(
        orbits_flat.len() % 6,
        0,
        "orbits_flat length must be a multiple of 6",
    );
    let n = orbits_flat.len() / 6;
    assert_eq!(
        observer_states_flat.len(),
        n * 6,
        "observer_states_flat length must be N * 6 for orbits shape (N, 6)",
    );
    assert_eq!(mus.len(), n, "mus length must match orbits rows");

    let mut sph_out = vec![0.0_f64; n * 6];
    let mut lt_out = vec![0.0_f64; n];
    let mut aberrated_out = vec![0.0_f64; n * 6];

    if rayon_is_single_threaded() {
        if has_consecutive_repeated_orbits(orbits_flat, mus, n) {
            generate_ephemeris_2body_grouped_single_thread(
                orbits_flat,
                observer_states_flat,
                mus,
                lt_tol,
                max_iter,
                tol,
                stellar_aberration,
                max_lt_iter,
                &mut sph_out,
                &mut lt_out,
                &mut aberrated_out,
            );
        } else {
            for i in 0..n {
                let base = i * 6;
                let orbit: [f64; 6] = [
                    orbits_flat[base],
                    orbits_flat[base + 1],
                    orbits_flat[base + 2],
                    orbits_flat[base + 3],
                    orbits_flat[base + 4],
                    orbits_flat[base + 5],
                ];
                let observer: [f64; 6] = [
                    observer_states_flat[base],
                    observer_states_flat[base + 1],
                    observer_states_flat[base + 2],
                    observer_states_flat[base + 3],
                    observer_states_flat[base + 4],
                    observer_states_flat[base + 5],
                ];
                let (sph, lt, aberrated) = generate_ephemeris_2body_row::<f64>(
                    orbit,
                    observer,
                    mus[i],
                    lt_tol,
                    max_iter,
                    tol,
                    stellar_aberration,
                    max_lt_iter,
                );
                sph_out[base..base + 6].copy_from_slice(&sph);
                lt_out[i] = lt;
                aberrated_out[base..base + 6].copy_from_slice(&aberrated);
            }
        }
        return (sph_out, lt_out, aberrated_out);
    }

    sph_out
        .par_chunks_mut(6)
        .zip(lt_out.par_iter_mut())
        .zip(aberrated_out.par_chunks_mut(6))
        .enumerate()
        .for_each(|(i, ((sph_dst, lt_dst), aberrated_dst))| {
            let base = i * 6;
            let orbit: [f64; 6] = [
                orbits_flat[base],
                orbits_flat[base + 1],
                orbits_flat[base + 2],
                orbits_flat[base + 3],
                orbits_flat[base + 4],
                orbits_flat[base + 5],
            ];
            let observer: [f64; 6] = [
                observer_states_flat[base],
                observer_states_flat[base + 1],
                observer_states_flat[base + 2],
                observer_states_flat[base + 3],
                observer_states_flat[base + 4],
                observer_states_flat[base + 5],
            ];
            let (sph, lt, aberrated) = generate_ephemeris_2body_row::<f64>(
                orbit,
                observer,
                mus[i],
                lt_tol,
                max_iter,
                tol,
                stellar_aberration,
                max_lt_iter,
            );
            sph_dst.copy_from_slice(&sph);
            *lt_dst = lt;
            aberrated_dst.copy_from_slice(&aberrated);
        });

    (sph_out, lt_out, aberrated_out)
}

/// Batched ephemeris with covariance transport. Rayon-parallel over rows.
///
/// For each row, evaluates (spherical, light-time, aberrated state) and the
/// 6×6 Jacobian `J = d spherical / d orbit` via a single `Dual<6>` pass, then
/// applies `Σ_out = J Σ_in J^T`. NaN-covariance input rows fall back to the
/// f64 kernel (state only) and fill Σ_out with NaN, mirroring the existing
/// `transform_with_covariance_flat6` policy.
///
/// Returns `(spherical_flat, light_time, aberrated_flat, spherical_cov_flat)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_ephemeris_2body_with_covariance_flat6(
    orbits_flat: &[f64],
    cov_flat: &[f64],
    observer_states_flat: &[f64],
    mus: &[f64],
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(
        orbits_flat.len() % 6,
        0,
        "orbits_flat length must be a multiple of 6",
    );
    let n = orbits_flat.len() / 6;
    assert_eq!(
        observer_states_flat.len(),
        n * 6,
        "observer_states_flat length must be N * 6 for orbits shape (N, 6)",
    );
    assert_eq!(mus.len(), n, "mus length must match orbits rows");
    assert_eq!(cov_flat.len(), n * 36, "cov_flat length must be N * 36");

    let mut sph_out = vec![0.0_f64; n * 6];
    let mut lt_out = vec![0.0_f64; n];
    let mut aberrated_out = vec![0.0_f64; n * 6];
    let mut cov_out = vec![0.0_f64; n * 36];

    sph_out
        .par_chunks_mut(6)
        .zip(lt_out.par_iter_mut())
        .zip(aberrated_out.par_chunks_mut(6))
        .zip(cov_out.par_chunks_mut(36))
        .enumerate()
        .for_each(|(i, (((sph_dst, lt_dst), aberrated_dst), cov_dst))| {
            let base = i * 6;
            let cov_base = i * 36;
            let orbit: [f64; 6] = [
                orbits_flat[base],
                orbits_flat[base + 1],
                orbits_flat[base + 2],
                orbits_flat[base + 3],
                orbits_flat[base + 4],
                orbits_flat[base + 5],
            ];
            let observer: [f64; 6] = [
                observer_states_flat[base],
                observer_states_flat[base + 1],
                observer_states_flat[base + 2],
                observer_states_flat[base + 3],
                observer_states_flat[base + 4],
                observer_states_flat[base + 5],
            ];
            let cov_slice = &cov_flat[cov_base..cov_base + 36];
            let cov_has_nan = cov_slice.iter().any(|v| v.is_nan());

            if cov_has_nan {
                let (sph, lt, aberrated) = generate_ephemeris_2body_row::<f64>(
                    orbit,
                    observer,
                    mus[i],
                    lt_tol,
                    max_iter,
                    tol,
                    stellar_aberration,
                    max_lt_iter,
                );
                sph_dst.copy_from_slice(&sph);
                *lt_dst = lt;
                aberrated_dst.copy_from_slice(&aberrated);
                for c in cov_dst.iter_mut() {
                    *c = f64::NAN;
                }
                return;
            }

            let (sph, lt, aberrated, jac) = generate_ephemeris_2body_with_jacobian_row(
                orbit,
                observer,
                mus[i],
                lt_tol,
                max_iter,
                tol,
                stellar_aberration,
                max_lt_iter,
            );
            sph_dst.copy_from_slice(&sph);
            *lt_dst = lt;
            aberrated_dst.copy_from_slice(&aberrated);

            // Σ_out = J Σ_in J^T via M = J Σ_in, Σ_out = M J^T (row-major 6×6).
            let mut m = [[0.0_f64; 6]; 6];
            for ii in 0..6 {
                for jj in 0..6 {
                    let mut s = 0.0_f64;
                    for kk in 0..6 {
                        s += jac[ii][kk] * cov_slice[kk * 6 + jj];
                    }
                    m[ii][jj] = s;
                }
            }
            for ii in 0..6 {
                for jj in 0..6 {
                    let mut s = 0.0_f64;
                    for kk in 0..6 {
                        s += m[ii][kk] * jac[jj][kk];
                    }
                    cov_dst[ii * 6 + jj] = s;
                }
            }
        });

    (sph_out, lt_out, aberrated_out, cov_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heliocentric gravitational parameter in au^3/day^2 (Gaussian k^2).
    const MU_SUN: f64 = 2.95912208284120e-4;

    fn sample_orbit() -> [f64; 6] {
        // Representative heliocentric asteroid-like orbit.
        [1.5, 0.2, 0.05, -0.003, 0.017, 0.0008]
    }

    fn sample_observer() -> [f64; 6] {
        // Offset barycentric observer (roughly Earth-like).
        [0.98, -0.15, 0.0, 0.002, 0.017, 0.0]
    }

    #[test]
    fn lt_is_consistent_with_topo_distance() {
        // After the LT Newton loop converges, the LT value must satisfy
        // rho_converged / C ≈ lt exactly (that IS the convergence condition).
        let (sph, lt, aberrated) = generate_ephemeris_2body_row::<f64>(
            sample_orbit(),
            sample_observer(),
            MU_SUN,
            1e-14,
            1000,
            1e-15,
            false,
            10,
        );
        assert!(lt.is_finite(), "LT did not converge");
        let dx = aberrated[0] - sample_observer()[0];
        let dy = aberrated[1] - sample_observer()[1];
        let dz = aberrated[2] - sample_observer()[2];
        let rho = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            (rho / C_AU_PER_DAY - lt).abs() < 1e-12,
            "rho/C ({}) vs lt ({})",
            rho / C_AU_PER_DAY,
            lt
        );
        assert!(
            sph[0] > 0.0 && sph[0].is_finite(),
            "spherical rho should be positive finite"
        );
    }

    #[test]
    fn jacobian_is_finite_and_nonzero_diagonal() {
        let (_sph, lt, _aberrated, jac) = generate_ephemeris_2body_with_jacobian_row(
            sample_orbit(),
            sample_observer(),
            MU_SUN,
            1e-14,
            1000,
            1e-15,
            false,
            10,
        );
        assert!(lt.is_finite());
        for (i, row) in jac.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                assert!(val.is_finite(), "jac[{i}][{j}] not finite: {val}");
            }
        }
        // Spherical ρ sensitivity to an input position component should be nonzero
        // (changes in observer-aligned position move the observed range).
        let any_nonzero = jac[0].iter().take(3).any(|v| v.abs() > 1e-6);
        assert!(any_nonzero, "d rho / d pos is all zero: {:?}", jac[0]);
    }

    #[test]
    fn stellar_aberration_flag_changes_output() {
        let (sph_off, _, _) = generate_ephemeris_2body_row::<f64>(
            sample_orbit(),
            sample_observer(),
            MU_SUN,
            1e-14,
            1000,
            1e-15,
            false,
            10,
        );
        let (sph_on, _, _) = generate_ephemeris_2body_row::<f64>(
            sample_orbit(),
            sample_observer(),
            MU_SUN,
            1e-14,
            1000,
            1e-15,
            true,
            10,
        );
        // Aberration should perturb lon / lat at ~10-20 arcsec scale
        // (observer velocity ~ 0.017 au/d, gamma ~ 1e-4 rad).
        let d_lon = (sph_on[1] - sph_off[1]).abs();
        let d_lat = (sph_on[2] - sph_off[2]).abs();
        assert!(d_lon > 0.0 && d_lat >= 0.0);
        // And rho (range) should be unchanged — aberration only rotates the unit vector.
        assert!(
            (sph_on[0] - sph_off[0]).abs() < 1e-12,
            "rho should not change under stellar aberration: {} vs {}",
            sph_on[0],
            sph_off[0]
        );
    }

    #[test]
    fn public_stellar_aberration_row_preserves_zero_velocity_case() {
        let topo = [1.0, 0.2, -0.1, 0.01, 0.02, 0.03];
        let got = apply_stellar_aberration_row(topo, [0.0; 6]);
        for i in 0..6 {
            assert!((got[i] - topo[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn non_converged_emits_nan_light_time() {
        // max_lt_iter = 0 guarantees the loop never runs; per the JAX policy
        // (iter >= max_lt_iter ⇒ NaN), light-time must be NaN.
        let (_sph, lt, _aberrated) = generate_ephemeris_2body_row::<f64>(
            sample_orbit(),
            sample_observer(),
            MU_SUN,
            1e-14,
            1000,
            1e-15,
            false,
            0,
        );
        assert!(lt.is_nan(), "expected NaN lt for max_lt_iter=0, got {lt}");
    }

    #[test]
    fn batched_matches_per_row() {
        let orbits = [sample_orbit(), sample_orbit()].concat();
        let observers = [sample_observer(), sample_observer()].concat();
        let mus = [MU_SUN, MU_SUN];
        let (sph_b, lt_b, aberrated_b) = generate_ephemeris_2body_flat6(
            &orbits, &observers, &mus, 1e-14, 1000, 1e-15, false, 10,
        );
        let (sph_r, lt_r, aberrated_r) = generate_ephemeris_2body_row::<f64>(
            sample_orbit(),
            sample_observer(),
            MU_SUN,
            1e-14,
            1000,
            1e-15,
            false,
            10,
        );
        for i in 0..6 {
            assert!((sph_b[i] - sph_r[i]).abs() < 1e-15);
            assert!((sph_b[6 + i] - sph_r[i]).abs() < 1e-15);
            assert!((aberrated_b[i] - aberrated_r[i]).abs() < 1e-15);
            assert!((aberrated_b[6 + i] - aberrated_r[i]).abs() < 1e-15);
        }
        assert!((lt_b[0] - lt_r).abs() < 1e-15);
        assert!((lt_b[1] - lt_r).abs() < 1e-15);
    }

    #[test]
    fn grouped_single_thread_matches_per_row_for_repeated_orbits() {
        let orbit = sample_orbit();
        let observers = [
            sample_observer(),
            [0.91, -0.31, 0.04, 0.004, 0.016, -0.0002],
            [1.04, 0.22, -0.03, -0.003, 0.017, 0.0001],
        ];
        let orbits = [orbit, orbit, orbit].concat();
        let observers_flat = observers.concat();
        let mus = [MU_SUN, MU_SUN, MU_SUN];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread rayon pool should build");

        pool.install(|| {
            assert!(rayon_is_single_threaded());
            let (sph_b, lt_b, aberrated_b) = generate_ephemeris_2body_flat6(
                &orbits,
                &observers_flat,
                &mus,
                1e-14,
                1000,
                1e-15,
                false,
                10,
            );

            for (row, observer) in observers.iter().enumerate() {
                let (sph_r, lt_r, aberrated_r) = generate_ephemeris_2body_row::<f64>(
                    orbit, *observer, MU_SUN, 1e-14, 1000, 1e-15, false, 10,
                );
                for i in 0..6 {
                    let got = sph_b[row * 6 + i];
                    let expected = sph_r[i];
                    assert!(
                        (got - expected).abs() < 1e-11,
                        "sph row {row} col {i}: {got} vs {expected}"
                    );
                    let got = aberrated_b[row * 6 + i];
                    let expected = aberrated_r[i];
                    assert!(
                        (got - expected).abs() < 1e-11,
                        "aberrated row {row} col {i}: {got} vs {expected}"
                    );
                }
                assert!(
                    (lt_b[row] - lt_r).abs() < 1e-14,
                    "lt row {row}: {} vs {lt_r}",
                    lt_b[row]
                );
            }
        });
    }

    #[test]
    fn nan_cov_short_circuits_to_nan_sigma() {
        let orbits = sample_orbit().to_vec();
        let observers = sample_observer().to_vec();
        let mus = [MU_SUN];
        let mut cov = vec![0.0_f64; 36];
        cov[0] = f64::NAN;
        let (_sph, lt, _aberrated, cov_out) = generate_ephemeris_2body_with_covariance_flat6(
            &orbits, &cov, &observers, &mus, 1e-14, 1000, 1e-15, false, 10,
        );
        assert!(lt[0].is_finite());
        for c in cov_out.iter() {
            assert!(c.is_nan());
        }
    }
}
