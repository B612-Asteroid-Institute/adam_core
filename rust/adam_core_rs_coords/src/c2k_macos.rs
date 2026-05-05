//! macOS-only batched-`acos` accelerator for `cartesian_to_keplerian_flat6`.
//!
//! The scalar per-row Rayon path in `lib.rs::cartesian_to_keplerian_flat6`
//! does four `f64::acos` calls per row (for inclination, RAAN, argument of
//! perihelion, and true anomaly). At 20k rows that is 80k scalar `acos` calls
//! through libm, which dominates the kernel cost on Apple Silicon and leaves
//! the kernel below the 1.2x p50/p95 gate vs the JAX baseline.
//!
//! This module rewrites the kernel as a two-pass tile pipeline:
//!   1. Pass 1 (scalar per row): geometry through the four cosine arguments,
//!      packed contiguously into a `4 * tile` scratch buffer plus tile-sized
//!      arrays of the partials needed for the second pass.
//!   2. Single Apple `vvacos` call over the contiguous `4 * tile` buffer.
//!   3. Pass 2 (scalar per row): read the four angles, apply sign /
//!      degenerate-orbit branches matching the legacy reference, compute the
//!      mean anomaly, and write the 13-element output row.
//!
//! All scratch buffers are reused via a thread-local pool so the public API
//! pays no per-call allocation. The cross-platform fallback in
//! `cartesian_to_keplerian_flat6` is unchanged for non-macOS targets.

use std::cell::RefCell;

use crate::{calc_mean_anomaly, RAD2DEG, TWO_PI};

/// Tile size for the macOS path. Sized so each tile's scratch state
/// (4 acos args + 7 partials × 8 bytes ≈ 88 KB) fits in L2 with room for
/// the input/output streams while still amortizing one `vvacos` dispatch.
const C2K_TILE_ROWS: usize = 8192;

/// Threshold above which the `cartesian_to_keplerian_flat6` entry point
/// diverts into the macOS batched-`acos` path. Below this the existing
/// scalar / Rayon path already wins on per-row dispatch overhead.
pub const C2K_MACOS_MIN_ROWS: usize = 1024;

const FLOAT_TOL: f64 = 1e-15;

/// `vvacos` and libm's `f64::acos` can disagree by ~1 ULP, which compounds
/// catastrophically when `acos` runs against a cosine very close to ±1
/// because `acos'(x) → ∞` as `|x| → 1`. The legacy reference uses libm-style
/// `acos` (via `jnp.arccos`); to keep the parity tolerance of `atol=1e-9` we
/// recompute any row whose clamped cosine sits inside this endpoint band
/// using libm's `f64::acos` directly. The threshold is sized so the residual
/// `vvacos` ULP error stays below ≈1e-10 degrees in the rest of the input
/// range, well under the parity gate.
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

/// Tile-batched batched-`acos` Cartesian-to-Keplerian conversion. `out` must
/// have length `13 * (flat_coords.len() / 6)`.
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

    SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let C2KScratch {
            acos_args,
            acos_out,
            a,
            p,
            e,
            n_vec_y,
            e_vec_z,
            rv,
            n_mag,
        } = &mut *scratch;

        for tile_start in (0..n).step_by(C2K_TILE_ROWS) {
            let m = C2K_TILE_ROWS.min(n - tile_start);

            // -- Pass 1: scalar per-row geometry, fill acos args + partials.
            let (i_args, rest) = acos_args.split_at_mut(m);
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

                // The legacy NaN-on-degenerate handling lives in pass 2;
                // here we always feed `vvacos` a finite, clamped argument.
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

                a[k] = a_v;
                p[k] = p_v;
                e[k] = e_v;
                n_vec_y[k] = n_vec_y_v;
                e_vec_z[k] = e_vec_z_v;
                rv[k] = rv_v;
                n_mag[k] = n_mag_v;
            }

            // -- Single batched `vvacos` over [i, raan, ap, nu] streams.
            // Apple's `vvacos` requires distinct input and output buffers, so
            // we write into the dedicated `acos_out` slab.
            let total = (4 * m) as i32;
            // SAFETY: `acos_args` and `acos_out` are distinct unique-mutable
            // slices each at least `4 * m` long.
            unsafe {
                vvacos(acos_out.as_mut_ptr(), acos_args.as_ptr(), &total);
            }

            // -- Pass 2: read angles, apply sign / degenerate branches,
            // mean anomaly, assemble 13-element output row. The four cosine
            // streams that landed in the endpoint band are recomputed via
            // libm `f64::acos` to match the legacy reference within the
            // c2k parity tolerance.
            let (i_args_in, rest) = acos_args.split_at(m);
            let (raan_args_in, rest) = rest.split_at(m);
            let (ap_args_in, rest) = rest.split_at(m);
            let (nu_args_in, _) = rest.split_at(m);
            let (i_angles, rest) = acos_out.split_at(m);
            let (raan_angles, rest) = rest.split_at(m);
            let (ap_angles, rest) = rest.split_at(m);
            let (nu_angles, _) = rest.split_at(m);
            for k in 0..m {
                let i_global = tile_start + k;
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

                let a_v = a[k];
                let p_v = p[k];
                let e_v = e[k];
                let n_vec_y_v = n_vec_y[k];
                let e_vec_z_v = e_vec_z[k];
                let rv_v = rv[k];
                let n_mag_v = n_mag[k];

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

                let row = &mut out[i_global * 13..i_global * 13 + 13];
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
        }
    });
}
