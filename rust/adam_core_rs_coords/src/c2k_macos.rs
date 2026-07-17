//! macOS-only batched-`acos` accelerator for `cartesian_to_keplerian_flat6`
//! and `cartesian_to_cometary_flat6`.
//!
//! The scalar per-row Rayon paths in `lib.rs` call `f64::acos` four times
//! per row (for inclination, RAAN, argument of perihelion, and true anomaly),
//! which dominates kernel cost on Apple Silicon at large workloads and
//! leaves the large-n lane below the 1.2x p50/p95 gate vs the JAX baseline.
//!
//! This module rewrites the kernel as a two-pass tile pipeline:
//!   1. Pass 1 (scalar per row): geometry through the four cosine arguments,
//!      packed contiguously into a `4 * tile` scratch buffer plus tile-sized
//!      arrays of the partials needed for the second pass.
//!   2. Single Apple `vvacos` call over the contiguous `4 * tile` buffer.
//!   3. Pass 2 (scalar per row): apply sign / degenerate-orbit branches
//!      matching the legacy reference, recompute endpoint-band rows via
//!      libm's `f64::acos` so vForce <-> libm rounding differences near ±1
//!      do not violate the parity tolerance, then run the kernel-specific
//!      mean-anomaly / output-row assembly.
//!
//! `cartesian_to_keplerian_flat6_into` and `cartesian_to_cometary_flat6_into`
//! share the geometry + acos pass plus the angle-correction step via
//! `fill_geometry_and_angles_tile` and `correct_angles`; only the
//! kernel-specific post-processing (output column count, period / tp branch
//! conventions) is per-API.

use std::cell::RefCell;

use rayon::prelude::*;

use crate::{calc_mean_anomaly, RAD2DEG, TWO_PI};

/// Tile size for the macOS path. Sized so each tile's scratch state fits in
/// L2 with room for the input/output streams while still amortizing one
/// `vvacos` dispatch.
const C2K_TILE_ROWS: usize = 8192;

/// Threshold above which the public flat6 entry points divert into the
/// macOS batched-`acos` path. Sized to be at or below the canonical
/// `parity_fuzz --n 128` workload so the macOS fast path is exercised by
/// every parity gate run, not just by the small-/large-n speed governance
/// at n>=2000. Below this threshold the existing scalar / Rayon path is
/// still the cross-platform fallback and is generally faster on per-row
/// dispatch overhead, but the cost is irrelevant: parity_fuzz is correctness-
/// only and there is no speed gate at n in [64, 2000).
pub const C2K_MACOS_MIN_ROWS: usize = 64;

const FLOAT_TOL: f64 = 1e-15;

/// `vvacos` and libm's `f64::acos` can disagree by ~1 ULP, which compounds
/// catastrophically when `acos` runs against a cosine very close to ±1
/// because `acos'(x) → ∞` as `|x| → 1`. The legacy reference uses libm-style
/// `acos` (via `jnp.arccos`); to keep the parity tolerance we recompute any
/// row whose clamped cosine sits inside this endpoint band using libm's
/// `f64::acos` directly.
const ACOS_ENDPOINT_GUARD: f64 = 1.0 - 1.0e-9;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vvacos(out: *mut f64, input: *const f64, n: *const i32);
}

struct C2KScratch {
    /// Packed `[i_args; raan_args; ap_args; nu_args]` of length `4 * tile`.
    /// Populated in pass 1 and consumed (read-only) by `vvacos`.
    acos_args: Vec<f64>,
    /// Output of `vvacos` over `acos_args`. Apple's `vvacos` requires the
    /// input and output buffers not to alias, so this is a separate slab.
    acos_out: Vec<f64>,
    a: Vec<f64>,
    p: Vec<f64>,
    e: Vec<f64>,
    n_vec_y: Vec<f64>,
    e_vec_z: Vec<f64>,
    rv: Vec<f64>,
    n_mag: Vec<f64>,
}

impl C2KScratch {
    fn new() -> Self {
        Self {
            acos_args: vec![0.0_f64; 4 * C2K_TILE_ROWS],
            acos_out: vec![0.0_f64; 4 * C2K_TILE_ROWS],
            a: vec![0.0_f64; C2K_TILE_ROWS],
            p: vec![0.0_f64; C2K_TILE_ROWS],
            e: vec![0.0_f64; C2K_TILE_ROWS],
            n_vec_y: vec![0.0_f64; C2K_TILE_ROWS],
            e_vec_z: vec![0.0_f64; C2K_TILE_ROWS],
            rv: vec![0.0_f64; C2K_TILE_ROWS],
            n_mag: vec![0.0_f64; C2K_TILE_ROWS],
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<C2KScratch> = RefCell::new(C2KScratch::new());
}

#[inline]
fn clamp_unit(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

/// Pass 1 + batched `vvacos` for one tile. Fills `scratch.acos_args` /
/// `scratch.acos_out` and the seven per-row partial vectors for the first
/// `m` rows starting at `tile_start`. Caller is responsible for the
/// kernel-specific output assembly (pass 2).
fn fill_geometry_and_angles_tile(
    scratch: &mut C2KScratch,
    flat_coords: &[f64],
    mu: &[f64],
    tile_start: usize,
    m: usize,
) {
    let (i_args, rest) = scratch.acos_args.split_at_mut(m);
    let (raan_args, rest) = rest.split_at_mut(m);
    let (ap_args, rest) = rest.split_at_mut(m);
    let (nu_args, _) = rest.split_at_mut(m);
    for k in 0..m {
        let i_global = tile_start + k;
        let base = i_global * 6;
        let rx = flat_coords[base];
        let ry = flat_coords[base + 1];
        let rz = flat_coords[base + 2];
        let vx = flat_coords[base + 3];
        let vy = flat_coords[base + 4];
        let vz = flat_coords[base + 5];
        let mu_i = mu[i_global];

        let r_mag = (rx * rx + ry * ry + rz * rz).sqrt();
        let v2 = vx * vx + vy * vy + vz * vz;
        let mu_over_r = mu_i / r_mag;
        let sme = v2 / 2.0 - mu_over_r;

        // h = r × v
        let hx = ry * vz - rz * vy;
        let hy = rz * vx - rx * vz;
        let hz = rx * vy - ry * vx;
        let h_mag = (hx * hx + hy * hy + hz * hz).sqrt();
        // n_vec = ẑ × h = (-hy, hx, 0); n_mag = sqrt(hx² + hy²).
        let n_vec_x = -hy;
        let n_vec_y_v = hx;
        let n_mag_v = (hx * hx + hy * hy).sqrt();

        let rv_v = rx * vx + ry * vy + rz * vz;
        let scale = v2 - mu_over_r;
        let e_vec_x = (scale * rx - rv_v * vx) / mu_i;
        let e_vec_y = (scale * ry - rv_v * vy) / mu_i;
        let e_vec_z_v = (scale * rz - rv_v * vz) / mu_i;
        let e_v = (e_vec_x * e_vec_x + e_vec_y * e_vec_y + e_vec_z_v * e_vec_z_v).sqrt();

        let p_v = h_mag * h_mag / mu_i;
        let near_parabolic = (e_v > (1.0 - FLOAT_TOL)) && (e_v < (1.0 + FLOAT_TOL));
        let a_v = if near_parabolic {
            f64::NAN
        } else {
            mu_i / (-2.0 * sme)
        };

        // The legacy NaN-on-degenerate handling lives in pass 2; here we
        // always feed `vvacos` a finite, clamped argument.
        let i_arg = clamp_unit(hz / h_mag);
        let raan_arg = if n_mag_v == 0.0 {
            0.0
        } else {
            clamp_unit(n_vec_x / n_mag_v)
        };
        let dot_n_e = n_vec_x * e_vec_x + n_vec_y_v * e_vec_y;
        let ap_arg = if n_mag_v == 0.0 || e_v == 0.0 {
            0.0
        } else {
            clamp_unit(dot_n_e / (n_mag_v * e_v))
        };
        let dot_e_r = e_vec_x * rx + e_vec_y * ry + e_vec_z_v * rz;
        let nu_arg = if e_v == 0.0 {
            0.0
        } else {
            clamp_unit(dot_e_r / (e_v * r_mag))
        };

        i_args[k] = i_arg;
        raan_args[k] = raan_arg;
        ap_args[k] = ap_arg;
        nu_args[k] = nu_arg;

        scratch.a[k] = a_v;
        scratch.p[k] = p_v;
        scratch.e[k] = e_v;
        scratch.n_vec_y[k] = n_vec_y_v;
        scratch.e_vec_z[k] = e_vec_z_v;
        scratch.rv[k] = rv_v;
        scratch.n_mag[k] = n_mag_v;
    }

    // Single batched `vvacos` over [i, raan, ap, nu] streams. Apple's
    // `vvacos` requires distinct input and output buffers.
    let total = (4 * m) as i32;
    // SAFETY: `acos_args` and `acos_out` are distinct unique-mutable slices
    // each at least `4 * m` long.
    unsafe {
        vvacos(
            scratch.acos_out.as_mut_ptr(),
            scratch.acos_args.as_ptr(),
            &total,
        );
    }
}

/// Apply legacy NaN/sign/zero branch corrections to the four acos angles for
/// a single row, including the libm-`acos` recompute for endpoint-band
/// arguments. Returns `(inc, raan, ap, nu)` in radians.
fn correct_angles(scratch: &C2KScratch, k: usize, m: usize) -> (f64, f64, f64, f64) {
    let (i_args_in, rest) = scratch.acos_args.split_at(m);
    let (raan_args_in, rest) = rest.split_at(m);
    let (ap_args_in, rest) = rest.split_at(m);
    let (nu_args_in, _) = rest.split_at(m);
    let (i_angles, rest) = scratch.acos_out.split_at(m);
    let (raan_angles, rest) = rest.split_at(m);
    let (ap_angles, rest) = rest.split_at(m);
    let (nu_angles, _) = rest.split_at(m);

    let inc = if i_args_in[k].abs() > ACOS_ENDPOINT_GUARD {
        i_args_in[k].acos()
    } else {
        i_angles[k]
    };
    let mut raan = if raan_args_in[k].abs() > ACOS_ENDPOINT_GUARD {
        raan_args_in[k].acos()
    } else {
        raan_angles[k]
    };
    let mut ap = if ap_args_in[k].abs() > ACOS_ENDPOINT_GUARD {
        ap_args_in[k].acos()
    } else {
        ap_angles[k]
    };
    let mut nu = if nu_args_in[k].abs() > ACOS_ENDPOINT_GUARD {
        nu_args_in[k].acos()
    } else {
        nu_angles[k]
    };

    let n_mag_v = scratch.n_mag[k];
    let n_vec_y_v = scratch.n_vec_y[k];
    let e_vec_z_v = scratch.e_vec_z[k];
    let e_v = scratch.e[k];
    let rv_v = scratch.rv[k];

    if n_mag_v == 0.0 {
        raan = f64::NAN;
    }
    if n_vec_y_v < 0.0 {
        raan = TWO_PI - raan;
    }
    if inc < FLOAT_TOL || (inc - TWO_PI).abs() < FLOAT_TOL {
        raan = 0.0;
    }

    if n_mag_v == 0.0 || e_v == 0.0 {
        ap = f64::NAN;
    }
    if e_vec_z_v < 0.0 {
        ap = TWO_PI - ap;
    }
    if e_v.abs() < FLOAT_TOL {
        ap = 0.0;
    }

    if e_v == 0.0 {
        nu = f64::NAN;
    }
    if rv_v < 0.0 {
        nu = TWO_PI - nu;
    }

    (inc, raan, ap, nu)
}

/// Tile-batched `acos` Cartesian-to-Keplerian conversion. `out` must have
/// length `13 * (flat_coords.len() / 6)`.
pub fn cartesian_to_keplerian_flat6_into(
    flat_coords: &[f64],
    t0: &[f64],
    mu: &[f64],
    out: &mut [f64],
) {
    let n = flat_coords.len() / 6;
    debug_assert_eq!(t0.len(), n);
    debug_assert_eq!(mu.len(), n);
    debug_assert_eq!(out.len(), n * 13);

    // Distribute tiles across Rayon workers for native multi-thread
    // throughput. With `RAYON_NUM_THREADS=1` (the canonical single-thread
    // gate) this collapses to a sequential walk so the gate timings are
    // unchanged; with unconstrained Rayon the work scales across cores
    // because each worker uses its own thread-local `SCRATCH` slab.
    out.par_chunks_mut(C2K_TILE_ROWS * 13)
        .enumerate()
        .for_each(|(tile_idx, out_tile)| {
            let tile_start = tile_idx * C2K_TILE_ROWS;
            let m = out_tile.len() / 13;
            SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                fill_geometry_and_angles_tile(&mut scratch, flat_coords, mu, tile_start, m);

                for k in 0..m {
                    let i_global = tile_start + k;
                    let (inc, raan, ap, nu) = correct_angles(&scratch, k, m);
                    let a_v = scratch.a[k];
                    let p_v = scratch.p[k];
                    let e_v = scratch.e[k];
                    let near_parabolic = (e_v > (1.0 - FLOAT_TOL)) && (e_v < (1.0 + FLOAT_TOL));
                    let q = if near_parabolic {
                        p_v / 2.0
                    } else {
                        a_v * (1.0 - e_v)
                    };
                    let q_apo = if e_v < 1.0 {
                        a_v * (1.0 + e_v)
                    } else {
                        f64::INFINITY
                    };
                    let m_anom = calc_mean_anomaly(nu, e_v);
                    let n_mean = if near_parabolic {
                        (mu[i_global] / (2.0 * q * q * q)).sqrt()
                    } else {
                        let abs_a = a_v.abs();
                        (mu[i_global] / (abs_a * abs_a * abs_a)).sqrt()
                    };
                    let period = if e_v < (1.0 - FLOAT_TOL) {
                        TWO_PI / n_mean
                    } else {
                        f64::INFINITY
                    };
                    let dtp = if (m_anom > std::f64::consts::PI) && (e_v < (1.0 - FLOAT_TOL)) {
                        period - m_anom / n_mean
                    } else {
                        -m_anom / n_mean
                    };
                    let tp = t0[i_global] + dtp;

                    let row = &mut out_tile[k * 13..k * 13 + 13];
                    row[0] = a_v;
                    row[1] = p_v;
                    row[2] = q;
                    row[3] = q_apo;
                    row[4] = e_v;
                    row[5] = inc * RAD2DEG;
                    row[6] = raan * RAD2DEG;
                    row[7] = ap * RAD2DEG;
                    row[8] = m_anom * RAD2DEG;
                    row[9] = nu * RAD2DEG;
                    row[10] = n_mean * RAD2DEG;
                    row[11] = period;
                    row[12] = tp;
                }
            });
        });
}

/// Tile-batched `acos` Cartesian-to-Cometary conversion. Output is
/// `[q, e, i_deg, raan_deg, ap_deg, tp]` per row (6 elements). Reuses the
/// shared geometry + `vvacos` + angle-correction helpers; the only kernel-
/// specific code is the cometary period / tp branch convention which uses
/// `e < 1.0` (no `FLOAT_TOL` margin) to match `cartesian_to_cometary6` in
/// `generic.rs`.
pub fn cartesian_to_cometary_flat6_into(
    flat_coords: &[f64],
    t0: &[f64],
    mu: &[f64],
    out: &mut [f64],
) {
    let n = flat_coords.len() / 6;
    debug_assert_eq!(t0.len(), n);
    debug_assert_eq!(mu.len(), n);
    debug_assert_eq!(out.len(), n * 6);

    out.par_chunks_mut(C2K_TILE_ROWS * 6)
        .enumerate()
        .for_each(|(tile_idx, out_tile)| {
            let tile_start = tile_idx * C2K_TILE_ROWS;
            let m = out_tile.len() / 6;
            SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                fill_geometry_and_angles_tile(&mut scratch, flat_coords, mu, tile_start, m);

                for k in 0..m {
                    let i_global = tile_start + k;
                    let (inc, raan, ap, nu) = correct_angles(&scratch, k, m);
                    let a_v = scratch.a[k];
                    let p_v = scratch.p[k];
                    let e_v = scratch.e[k];
                    let near_parabolic = (e_v > (1.0 - FLOAT_TOL)) && (e_v < (1.0 + FLOAT_TOL));
                    let q = if near_parabolic {
                        p_v / 2.0
                    } else {
                        a_v * (1.0 - e_v)
                    };
                    let m_anom = calc_mean_anomaly(nu, e_v);
                    // Cometary uses `e < 1.0` with no FLOAT_TOL margin
                    // (matches `cartesian_to_cometary6` in `generic.rs`).
                    let n_mean = (mu[i_global] / a_v.abs().powi(3)).sqrt();
                    let period = TWO_PI / n_mean;
                    let dtp = if m_anom > std::f64::consts::PI && e_v < 1.0 {
                        period - m_anom / n_mean
                    } else {
                        -(m_anom / n_mean)
                    };
                    let tp = t0[i_global] + dtp;

                    let row = &mut out_tile[k * 6..k * 6 + 6];
                    row[0] = q;
                    row[1] = e_v;
                    row[2] = inc * RAD2DEG;
                    row[3] = raan * RAD2DEG;
                    row[4] = ap * RAD2DEG;
                    row[5] = tp;
                }
            });
        });
}
