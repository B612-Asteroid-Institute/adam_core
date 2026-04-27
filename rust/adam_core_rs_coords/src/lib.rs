#![allow(clippy::too_many_arguments)]

use adam_core_rs_autodiff::Dual;
use rayon::prelude::*;

pub mod generic;
pub use generic::{
    cartesian_to_cometary6, cartesian_to_geodetic6, cartesian_to_keplerian6,
    cartesian_to_spherical6, cometary_to_cartesian6, keplerian_to_cartesian6,
    rotate_ecliptic_to_equatorial6, rotate_equatorial_to_ecliptic6, spherical_to_cartesian6,
};

pub mod propagate;
pub mod tisserand;
pub use tisserand::tisserand_parameter_flat;

pub mod classification;
pub use classification::classify_orbits_flat;

pub mod chi2;
pub use chi2::{calculate_chi2_flat, Chi2Error};

pub mod weighted;
pub use weighted::{weighted_covariance_flat, weighted_mean_flat};

pub mod abs_mag;
pub use abs_mag::{fit_absolute_magnitude_grouped, fit_absolute_magnitude_rows, AbsMagFit};

pub mod spherical_resid;
pub use spherical_resid::{
    apply_cosine_latitude_correction_flat, bound_longitude_residuals_flat,
};

pub use propagate::{
    calc_chi, calc_chi_with_init, calc_lagrange_coefficients, calc_stumpff,
    propagate_2body_along_arc, propagate_2body_arc_batch_flat6, propagate_2body_flat6,
    propagate_2body_row, propagate_2body_with_covariance_flat6, OrbitConstants,
};

pub mod ephemeris;
pub use ephemeris::{
    add_light_time_batch_flat, add_light_time_row, generate_ephemeris_2body_flat6,
    generate_ephemeris_2body_row, generate_ephemeris_2body_with_covariance_flat6, C_AU_PER_DAY,
    DEFAULT_MAX_LT_ITER,
};

pub mod photometry;
pub use photometry::{
    calculate_apparent_magnitude_v_and_phase_angle_flat, calculate_apparent_magnitude_v_flat,
    calculate_phase_angle_flat, predict_magnitudes_bandpass_flat,
};

pub mod lambert;
pub use lambert::{izzo_lambert, izzo_lambert_batch_flat, porkchop_grid_flat};

pub mod moid;
pub use moid::{calculate_moid, calculate_moid_batch};

pub fn cartesian_to_cometary_flat6(flat_coords: &[f64], t0: &[f64], mu: &[f64]) -> Vec<f64> {
    assert_eq!(
        flat_coords.len() % 6,
        0,
        "flat_coords length must be a multiple of 6",
    );
    let n = flat_coords.len() / 6;
    assert_eq!(t0.len(), n, "t0 length must match coords rows");
    assert_eq!(mu.len(), n, "mu length must match coords rows");
    let mut out = vec![0.0_f64; flat_coords.len()];
    out.par_chunks_mut(6).enumerate().for_each(|(i, dst)| {
        let base = i * 6;
        let row = [
            flat_coords[base],
            flat_coords[base + 1],
            flat_coords[base + 2],
            flat_coords[base + 3],
            flat_coords[base + 4],
            flat_coords[base + 5],
        ];
        let converted = cartesian_to_cometary6::<f64>(&row, t0[i], mu[i]);
        dst.copy_from_slice(&converted);
    });
    out
}

pub fn cometary_to_cartesian_flat6(
    flat_coords: &[f64],
    t0: &[f64],
    mu: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    assert_eq!(
        flat_coords.len() % 6,
        0,
        "flat_coords length must be a multiple of 6",
    );
    let n = flat_coords.len() / 6;
    assert_eq!(t0.len(), n, "t0 length must match coords rows");
    assert_eq!(mu.len(), n, "mu length must match coords rows");
    let mut out = vec![0.0_f64; flat_coords.len()];
    out.par_chunks_mut(6).enumerate().for_each(|(i, dst)| {
        let base = i * 6;
        let row = [
            flat_coords[base],
            flat_coords[base + 1],
            flat_coords[base + 2],
            flat_coords[base + 3],
            flat_coords[base + 4],
            flat_coords[base + 5],
        ];
        let converted = cometary_to_cartesian6::<f64>(&row, t0[i], mu[i], max_iter, tol);
        dst.copy_from_slice(&converted);
    });
    out
}

pub const TWO_PI: f64 = std::f64::consts::PI * 2.0;
pub const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;

#[inline]
pub fn normalize_lon_rad(lon: f64) -> f64 {
    if lon < 0.0 {
        lon + TWO_PI
    } else {
        lon
    }
}

pub fn calc_mean_motion_batch(a: &[f64], mu: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), mu.len(), "a and mu must have the same length");
    let mut out = vec![0.0_f64; a.len()];
    // Scalar-multiply instead of `.powi(3)` gives the compiler a clean
    // auto-vectorizable inner loop. Rayon parallelism doesn't help here
    // (the per-element work is ~5 flops, dwarfed by memory traffic —
    // thread-sync overhead dominates). At realistic call-site sizes
    // (1..~100k rows), this path beats JAX's XLA kernel by 1.5-7x; at
    // very large N (~400k+) XLA's SIMD width wins on memory-bandwidth
    // streaming, but that regime is not representative of production
    // gaussIOD/Keplerian.n call patterns.
    for ((o, &ai), &mi) in out.iter_mut().zip(a).zip(mu) {
        let abs_a = ai.abs();
        *o = (mi / (abs_a * abs_a * abs_a)).sqrt();
    }
    out
}

const OBLIQUITY_RAD: f64 = 84381.448_f64 * std::f64::consts::PI / (180.0_f64 * 3600.0_f64);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Frame {
    Ecliptic,
    Equatorial,
    Itrf93,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    Cartesian,
    Spherical,
    Geodetic,
    Keplerian,
    Cometary,
}

/// Apply the representation-in -> cartesian(frame_in) -> cartesian(frame_out) -> representation-out
/// chain with `T: Scalar` so the same body is reused both for f64 value evaluation and
/// Dual<6> Jacobian extraction. `t0`, `mu`, `a`, `f` are provided as `T::from_f64(...)` by the caller.
fn transform_chain<T: adam_core_rs_autodiff::Scalar>(
    coords: [T; 6],
    rep_in: Representation,
    rep_out: Representation,
    frame_in: Frame,
    frame_out: Frame,
    t0: T,
    mu: T,
    a: f64,
    f: f64,
    max_iter: usize,
    tol: f64,
    translation: Option<[T; 6]>,
) -> [T; 6] {
    let cart_in = match rep_in {
        Representation::Cartesian => coords,
        Representation::Spherical => spherical_to_cartesian6::<T>(&coords),
        Representation::Keplerian => keplerian_to_cartesian6::<T>(&coords, mu, max_iter, tol),
        Representation::Cometary => cometary_to_cartesian6::<T>(&coords, t0, mu, max_iter, tol),
        Representation::Geodetic => {
            // geodetic as input is not a supported rep-in for transforms in this pipeline
            // (matches legacy): mirror parse-level rejection by returning NaN.
            [T::from_f64(f64::NAN); 6]
        }
    };

    // Translation is a constant offset applied in the input frame (origin
    // change, covariance-invariant). The Jacobian wrt input state is the
    // identity, so the downstream J @ Σ @ J^T accumulates nothing for this
    // step — we just add the offset to the state.
    let cart_translated = if let Some(v) = translation {
        [
            cart_in[0] + v[0],
            cart_in[1] + v[1],
            cart_in[2] + v[2],
            cart_in[3] + v[3],
            cart_in[4] + v[4],
            cart_in[5] + v[5],
        ]
    } else {
        cart_in
    };

    let cart_out_frame = if frame_in == frame_out {
        cart_translated
    } else {
        match (frame_in, frame_out) {
            (Frame::Equatorial, Frame::Ecliptic) => {
                rotate_equatorial_to_ecliptic6::<T>(&cart_translated)
            }
            (Frame::Ecliptic, Frame::Equatorial) => {
                rotate_ecliptic_to_equatorial6::<T>(&cart_translated)
            }
            _ => [T::from_f64(f64::NAN); 6],
        }
    };

    match rep_out {
        Representation::Cartesian => cart_out_frame,
        Representation::Spherical => cartesian_to_spherical6::<T>(&cart_out_frame),
        Representation::Geodetic => cartesian_to_geodetic6::<T>(&cart_out_frame, a, f),
        Representation::Keplerian => {
            // Legacy returns (a, e, i, raan, ap, M) — 6 elements.
            cartesian_to_keplerian6::<T>(&cart_out_frame, mu)
        }
        Representation::Cometary => cartesian_to_cometary6::<T>(&cart_out_frame, t0, mu),
    }
}

/// Compute the 6-element output row AND its 6x6 Jacobian in a single Dual<6> pass.
///
/// The Jacobian rows correspond to output components; columns correspond to input
/// components. `jacobian[i][j] = d out_i / d in_j`.
pub fn transform_row_with_jacobian(
    coords: [f64; 6],
    rep_in: Representation,
    rep_out: Representation,
    frame_in: Frame,
    frame_out: Frame,
    t0: f64,
    mu: f64,
    a: f64,
    f: f64,
    max_iter: usize,
    tol: f64,
    translation: Option<[f64; 6]>,
) -> ([f64; 6], [[f64; 6]; 6]) {
    let seeded: [Dual<6>; 6] = Dual::seed(coords);
    let t0_d = Dual::<6>::constant(t0);
    let mu_d = Dual::<6>::constant(mu);
    let translation_d = translation.map(|v| {
        [
            Dual::<6>::constant(v[0]),
            Dual::<6>::constant(v[1]),
            Dual::<6>::constant(v[2]),
            Dual::<6>::constant(v[3]),
            Dual::<6>::constant(v[4]),
            Dual::<6>::constant(v[5]),
        ]
    });
    let out = transform_chain::<Dual<6>>(
        seeded,
        rep_in,
        rep_out,
        frame_in,
        frame_out,
        t0_d,
        mu_d,
        a,
        f,
        max_iter,
        tol,
        translation_d,
    );
    let mut values = [0.0_f64; 6];
    let mut jac = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        values[i] = out[i].re;
        for j in 0..6 {
            jac[i][j] = out[i].du[j];
        }
    }
    (values, jac)
}

/// Batch variant: compute propagated covariances `J @ Σ @ J^T` per row in parallel.
///
/// `covariance_in` is a flattened `[N * 36]` buffer (row-major 6x6 per entry).
/// Rows with any NaN in their input covariance propagate NaN through (no Jacobian
/// evaluation is performed for those rows). Returns `(coords_out_flat, cov_out_flat)`.
#[allow(clippy::too_many_arguments)]
pub fn transform_with_covariance_flat6(
    coords_flat: &[f64],
    covariance_flat: &[f64],
    rep_in: Representation,
    rep_out: Representation,
    frame_in: Frame,
    frame_out: Frame,
    t0: &[f64],
    mu: &[f64],
    a: f64,
    f: f64,
    max_iter: usize,
    tol: f64,
    translation_flat: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        coords_flat.len() % 6,
        0,
        "coords_flat length must be a multiple of 6"
    );
    let n = coords_flat.len() / 6;
    assert_eq!(
        covariance_flat.len(),
        n * 36,
        "covariance_flat length must be N * 36 for coords shape (N, 6)"
    );
    assert_eq!(t0.len(), n, "t0 length must match coords rows");
    assert_eq!(mu.len(), n, "mu length must match coords rows");
    if let Some(t) = translation_flat {
        assert_eq!(
            t.len(),
            n * 6,
            "translation_flat length must be N * 6 for coords shape (N, 6)"
        );
    }

    let mut coords_out = vec![0.0_f64; n * 6];
    let mut cov_out = vec![0.0_f64; n * 36];

    coords_out
        .par_chunks_mut(6)
        .zip(cov_out.par_chunks_mut(36))
        .enumerate()
        .for_each(|(i, (coords_dst, cov_dst))| {
            let base = i * 6;
            let row = [
                coords_flat[base],
                coords_flat[base + 1],
                coords_flat[base + 2],
                coords_flat[base + 3],
                coords_flat[base + 4],
                coords_flat[base + 5],
            ];
            let cov_base = i * 36;
            let cov_slice = &covariance_flat[cov_base..cov_base + 36];
            let cov_has_nan = cov_slice.iter().any(|v| v.is_nan());

            let translation_row: Option<[f64; 6]> = translation_flat.map(|t| {
                let tb = i * 6;
                [t[tb], t[tb + 1], t[tb + 2], t[tb + 3], t[tb + 4], t[tb + 5]]
            });

            if cov_has_nan {
                let values = match rep_in {
                    Representation::Cartesian
                    | Representation::Spherical
                    | Representation::Keplerian
                    | Representation::Cometary
                    | Representation::Geodetic => transform_chain::<f64>(
                        row,
                        rep_in,
                        rep_out,
                        frame_in,
                        frame_out,
                        t0[i],
                        mu[i],
                        a,
                        f,
                        max_iter,
                        tol,
                        translation_row,
                    ),
                };
                coords_dst.copy_from_slice(&values);
                for c in cov_dst.iter_mut() {
                    *c = f64::NAN;
                }
                return;
            }

            let (values, jac) = transform_row_with_jacobian(
                row,
                rep_in,
                rep_out,
                frame_in,
                frame_out,
                t0[i],
                mu[i],
                a,
                f,
                max_iter,
                tol,
                translation_row,
            );
            coords_dst.copy_from_slice(&values);

            // cov_out = J * Σ * J^T
            // Compute M = J * Σ (6x6) then cov_out = M * J^T.
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

    (coords_out, cov_out)
}

pub fn cartesian_to_spherical_batch(rows: &[[f64; 6]]) -> Vec<[f64; 6]> {
    rows.iter().map(cartesian_to_spherical_row).collect()
}

pub fn spherical_to_cartesian_batch(rows: &[[f64; 6]]) -> Vec<[f64; 6]> {
    rows.iter().map(spherical_to_cartesian_row).collect()
}

pub fn cartesian_to_geodetic_batch(
    rows: &[[f64; 6]],
    a: f64,
    f: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<[f64; 6]> {
    rows.iter()
        .map(|row| cartesian_to_geodetic_row(row, a, f, max_iter, tol))
        .collect()
}

pub fn cartesian_to_keplerian_batch(rows: &[[f64; 6]], t0: &[f64], mu: &[f64]) -> Vec<[f64; 13]> {
    assert_eq!(rows.len(), t0.len(), "t0 length must match coords rows");
    assert_eq!(rows.len(), mu.len(), "mu length must match coords rows");
    rows.iter()
        .zip(t0.iter().zip(mu.iter()))
        .map(|(row, (t0_i, mu_i))| cartesian_to_keplerian_row(row, *t0_i, *mu_i))
        .collect()
}

pub fn cartesian_to_spherical_flat6(flat_coords: &[f64]) -> Vec<f64> {
    assert_eq!(
        flat_coords.len() % 6,
        0,
        "flat_coords length must be a multiple of 6",
    );
    let mut out = vec![0.0_f64; flat_coords.len()];
    out.par_chunks_mut(6)
        .zip(flat_coords.par_chunks(6))
        .for_each(|(dst, src)| {
            let row = [src[0], src[1], src[2], src[3], src[4], src[5]];
            let converted = cartesian_to_spherical_row(&row);
            dst.copy_from_slice(&converted);
        });
    out
}

pub fn spherical_to_cartesian_flat6(flat_coords: &[f64]) -> Vec<f64> {
    assert_eq!(
        flat_coords.len() % 6,
        0,
        "flat_coords length must be a multiple of 6",
    );
    let mut out = vec![0.0_f64; flat_coords.len()];
    out.par_chunks_mut(6)
        .zip(flat_coords.par_chunks(6))
        .for_each(|(dst, src)| {
            let row = [src[0], src[1], src[2], src[3], src[4], src[5]];
            let converted = spherical_to_cartesian_row(&row);
            dst.copy_from_slice(&converted);
        });
    out
}

pub fn cartesian_to_geodetic_flat6(
    flat_coords: &[f64],
    a: f64,
    f: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    assert_eq!(
        flat_coords.len() % 6,
        0,
        "flat_coords length must be a multiple of 6",
    );
    let mut out = vec![0.0_f64; flat_coords.len()];
    out.par_chunks_mut(6)
        .zip(flat_coords.par_chunks(6))
        .for_each(|(dst, src)| {
            let row = [src[0], src[1], src[2], src[3], src[4], src[5]];
            let converted = cartesian_to_geodetic_row(&row, a, f, max_iter, tol);
            dst.copy_from_slice(&converted);
        });
    out
}

pub fn cartesian_to_keplerian_flat6(flat_coords: &[f64], t0: &[f64], mu: &[f64]) -> Vec<f64> {
    assert_eq!(
        flat_coords.len() % 6,
        0,
        "flat_coords length must be a multiple of 6",
    );
    let n = flat_coords.len() / 6;
    assert_eq!(t0.len(), n, "t0 length must match coords rows");
    assert_eq!(mu.len(), n, "mu length must match coords rows");

    let mut out = vec![0.0_f64; n * 13];
    out.par_chunks_mut(13).enumerate().for_each(|(i, dst)| {
        let base = i * 6;
        let row = [
            flat_coords[base],
            flat_coords[base + 1],
            flat_coords[base + 2],
            flat_coords[base + 3],
            flat_coords[base + 4],
            flat_coords[base + 5],
        ];
        let converted = cartesian_to_keplerian_row(&row, t0[i], mu[i]);
        dst.copy_from_slice(&converted);
    });
    out
}

pub fn keplerian_to_cartesian_flat6(
    flat_coords: &[f64],
    mu: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    assert_eq!(
        flat_coords.len() % 6,
        0,
        "flat_coords length must be a multiple of 6",
    );
    let n = flat_coords.len() / 6;
    assert_eq!(mu.len(), n, "mu length must match coords rows");

    let mut out = vec![0.0_f64; flat_coords.len()];
    out.par_chunks_mut(6).enumerate().for_each(|(i, dst)| {
        let base = i * 6;
        let row = [
            flat_coords[base],
            flat_coords[base + 1],
            flat_coords[base + 2],
            flat_coords[base + 3],
            flat_coords[base + 4],
            flat_coords[base + 5],
        ];
        let converted = keplerian_to_cartesian_a_row(&row, mu[i], max_iter, tol);
        dst.copy_from_slice(&converted);
    });
    out
}

pub fn rotate_cartesian_frame_flat6(
    flat_coords: &[f64],
    frame_in: Frame,
    frame_out: Frame,
) -> Result<Vec<f64>, &'static str> {
    assert_eq!(
        flat_coords.len() % 6,
        0,
        "flat_coords length must be a multiple of 6",
    );
    if frame_in == frame_out {
        return Ok(flat_coords.to_vec());
    }
    if !matches!(
        (frame_in, frame_out),
        (Frame::Equatorial, Frame::Ecliptic) | (Frame::Ecliptic, Frame::Equatorial)
    ) {
        return Err("unsupported frame transform");
    }

    let mut out = vec![0.0_f64; flat_coords.len()];
    out.par_chunks_mut(6)
        .zip(flat_coords.par_chunks(6))
        .for_each(|(dst, src)| {
            let row = [src[0], src[1], src[2], src[3], src[4], src[5]];
            let converted = match (frame_in, frame_out) {
                (Frame::Equatorial, Frame::Ecliptic) => rotate_equatorial_to_ecliptic_row(&row),
                (Frame::Ecliptic, Frame::Equatorial) => rotate_ecliptic_to_equatorial_row(&row),
                _ => unreachable!("validated supported frame transform before loop"),
            };
            dst.copy_from_slice(&converted);
        });

    Ok(out)
}

/// Apply per-row time-varying 6x6 rotation matrices to a batch of
/// Cartesian states (and optionally their 6x6 covariance matrices).
///
/// - `flat_coords`: row-major `[N*6]` states.
/// - `flat_cov`: row-major `[N*36]` covariances, or empty slice to skip.
/// - `time_index`: `[N]` per-row index into the matrix table. A single
///   6x6 matrix can be reused across many rows when the workload has
///   far fewer unique epochs than rows (the typical ephemeris case).
/// - `matrices_flat`: `[U*36]` row-major 6x6 matrices (already in the
///   caller's desired units — no unit conversion is applied here).
///
/// Returns `(rotated_coords_flat, rotated_cov_flat)`. When `flat_cov`
/// is empty the returned cov buffer is also empty. NaN covariance rows
/// pass through untouched (same convention as
/// [`transform_with_covariance_flat6`]).
pub fn rotate_cartesian_time_varying_flat6(
    flat_coords: &[f64],
    flat_cov: &[f64],
    time_index: &[usize],
    matrices_flat: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    if flat_coords.len() % 6 != 0 {
        return Err("flat_coords length must be a multiple of 6");
    }
    let n = flat_coords.len() / 6;
    if time_index.len() != n {
        return Err("time_index length must match coords rows");
    }
    if matrices_flat.len() % 36 != 0 {
        return Err("matrices_flat length must be a multiple of 36");
    }
    let u = matrices_flat.len() / 36;
    let has_cov = !flat_cov.is_empty();
    if has_cov && flat_cov.len() != n * 36 {
        return Err("flat_cov length must be N * 36 when provided");
    }

    // Validate the time-index bounds once up front so the hot loop
    // can index unchecked-safely without per-row Result handling.
    if let Some(&bad) = time_index.iter().find(|&&ti| ti >= u) {
        // Surface the offending value via a static message; detailed
        // row info is not retained to keep this kind-of-cheap error.
        let _ = bad;
        return Err("time_index contains value >= number of matrices");
    }

    let mut out_coords = vec![0.0_f64; flat_coords.len()];
    out_coords
        .par_chunks_mut(6)
        .zip(flat_coords.par_chunks(6))
        .zip(time_index.par_iter())
        .for_each(|((dst, src), &ti)| {
            let m = &matrices_flat[ti * 36..ti * 36 + 36];
            for r in 0..6 {
                let row = &m[r * 6..r * 6 + 6];
                dst[r] = row[0] * src[0]
                    + row[1] * src[1]
                    + row[2] * src[2]
                    + row[3] * src[3]
                    + row[4] * src[4]
                    + row[5] * src[5];
            }
        });

    let mut out_cov = if has_cov {
        vec![0.0_f64; flat_cov.len()]
    } else {
        Vec::new()
    };
    if has_cov {
        out_cov
            .par_chunks_mut(36)
            .zip(flat_cov.par_chunks(36))
            .zip(time_index.par_iter())
            .for_each(|((dst, src), &ti)| {
                // Match CartesianCoordinates.rotate's NaN policy exactly:
                // replace NaN with 0 for the rotation, then restore NaN
                // at the same positions in the output. An all-NaN input
                // yields an all-NaN output; a partial-NaN input yields
                // a matrix whose non-NaN cells are the rotated values
                // of the 0-filled input.
                let m = &matrices_flat[ti * 36..ti * 36 + 36];
                let mut filled = [0.0_f64; 36];
                let mut nan_mask = [false; 36];
                for i in 0..36 {
                    if src[i].is_nan() {
                        nan_mask[i] = true;
                    } else {
                        filled[i] = src[i];
                    }
                }
                // tmp = M @ Σ_filled (row-major 6x6).
                let mut tmp = [0.0_f64; 36];
                for r in 0..6 {
                    for c in 0..6 {
                        let mut acc = 0.0;
                        for k in 0..6 {
                            acc += m[r * 6 + k] * filled[k * 6 + c];
                        }
                        tmp[r * 6 + c] = acc;
                    }
                }
                // dst = tmp @ M^T, restoring NaN at original positions.
                for r in 0..6 {
                    for c in 0..6 {
                        if nan_mask[r * 6 + c] {
                            dst[r * 6 + c] = f64::NAN;
                            continue;
                        }
                        let mut acc = 0.0;
                        for k in 0..6 {
                            acc += tmp[r * 6 + k] * m[c * 6 + k];
                        }
                        dst[r * 6 + c] = acc;
                    }
                }
            });
    }

    Ok((out_coords, out_cov))
}

pub fn cartesian_to_spherical_row(v: &[f64; 6]) -> [f64; 6] {
    let x = v[0];
    let y = v[1];
    let z = v[2];
    let vx = v[3];
    let vy = v[4];
    let vz = v[5];

    let rho = (x * x + y * y + z * z).sqrt();
    let lon = normalize_lon_rad(y.atan2(x));

    let mut lat = if rho == 0.0 { 0.0 } else { (z / rho).asin() };
    if (3.0 * std::f64::consts::PI / 2.0..=TWO_PI).contains(&lat) {
        lat -= TWO_PI;
    }

    let vrho = if rho == 0.0 {
        0.0
    } else {
        (x * vx + y * vy + z * vz) / rho
    };

    let vlon = if x == 0.0 && y == 0.0 {
        0.0
    } else {
        (vy * x - vx * y) / (x * x + y * y)
    };

    let vlat = if (x == 0.0 && y == 0.0) || rho == 0.0 {
        0.0
    } else {
        (vz - vrho * z / rho) / (x * x + y * y).sqrt()
    };

    [
        rho,
        lon * RAD2DEG,
        lat * RAD2DEG,
        vrho,
        vlon * RAD2DEG,
        vlat * RAD2DEG,
    ]
}

pub fn rotate_equatorial_to_ecliptic_row(v: &[f64; 6]) -> [f64; 6] {
    let cos_o = OBLIQUITY_RAD.cos();
    let sin_o = OBLIQUITY_RAD.sin();
    [
        v[0],
        cos_o * v[1] + sin_o * v[2],
        -sin_o * v[1] + cos_o * v[2],
        v[3],
        cos_o * v[4] + sin_o * v[5],
        -sin_o * v[4] + cos_o * v[5],
    ]
}

pub fn rotate_ecliptic_to_equatorial_row(v: &[f64; 6]) -> [f64; 6] {
    let cos_o = OBLIQUITY_RAD.cos();
    let sin_o = OBLIQUITY_RAD.sin();
    [
        v[0],
        cos_o * v[1] - sin_o * v[2],
        sin_o * v[1] + cos_o * v[2],
        v[3],
        cos_o * v[4] - sin_o * v[5],
        sin_o * v[4] + cos_o * v[5],
    ]
}

pub fn spherical_to_cartesian_row(v: &[f64; 6]) -> [f64; 6] {
    let rho = v[0];
    let lon = v[1] / RAD2DEG;
    let lat = v[2] / RAD2DEG;
    let vrho = v[3];
    let vlon = v[4] / RAD2DEG;
    let vlat = v[5] / RAD2DEG;

    let cos_lat = lat.cos();
    let sin_lat = lat.sin();
    let cos_lon = lon.cos();
    let sin_lon = lon.sin();

    let x = rho * cos_lat * cos_lon;
    let y = rho * cos_lat * sin_lon;
    let z = rho * sin_lat;

    let vx =
        cos_lat * cos_lon * vrho - rho * cos_lat * sin_lon * vlon - rho * sin_lat * cos_lon * vlat;
    let vy =
        cos_lat * sin_lon * vrho + rho * cos_lat * cos_lon * vlon - rho * sin_lat * sin_lon * vlat;
    let vz = sin_lat * vrho + rho * cos_lat * vlat;

    [x, y, z, vx, vy, vz]
}

pub fn cartesian_to_geodetic_row(
    v: &[f64; 6],
    a: f64,
    f: f64,
    _max_iter: usize,
    _tol: f64,
) -> [f64; 6] {
    let x = v[0];
    let y = v[1];
    let z = v[2];
    let vx = v[3];
    let vy = v[4];
    let vz = v[5];

    let b = a * (1.0 - f);
    let e2 = (a * a - b * b) / (a * a);
    let xy = (x * x + y * y).sqrt();

    let lat0 = z.atan2(xy * (1.0 - e2));
    let sin_lat0 = lat0.sin();
    let cos_lat0 = lat0.cos();
    let n0 = a / (1.0 - e2 * sin_lat0 * sin_lat0).sqrt();
    let alt0 = xy / cos_lat0 - n0;

    // Keep behavior aligned with the current JAX path, which effectively
    // performs a single Bowring update step.
    let lat = z.atan2(xy * (1.0 - e2 * n0 / (n0 + alt0)));
    let lon = normalize_lon_rad(y.atan2(x));

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let alt = xy / cos_lat - n;

    let v_east = -sin_lon * vx + cos_lon * vy;
    let v_north = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz;
    let v_up = cos_lat * cos_lon * vx + cos_lat * sin_lon * vy + sin_lat * vz;

    [
        alt,
        lon * RAD2DEG,
        lat * RAD2DEG,
        v_east * RAD2DEG,
        v_north * RAD2DEG,
        v_up,
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn clamp_unit(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

fn calc_mean_anomaly(nu: f64, e: f64) -> f64 {
    let nu_norm = if nu >= TWO_PI { nu % TWO_PI } else { nu };
    if e < 1.0 {
        let ecc_anomaly = ((1.0 - e * e).sqrt() * nu_norm.sin()).atan2(e + nu_norm.cos());
        let mut mean = ecc_anomaly - e * ecc_anomaly.sin();
        if mean < 0.0 {
            mean += TWO_PI;
        }
        return mean;
    }

    if e > 1.0 {
        let hyp_anomaly = 2.0 * (((e - 1.0) / (e + 1.0)).sqrt() * (nu_norm / 2.0).tan()).atanh();
        let mut mean = e * hyp_anomaly.sinh() - hyp_anomaly;
        if mean < 0.0 {
            mean += TWO_PI;
        }
        return mean;
    }

    let d = (nu_norm / 2.0).atan();
    let mut mean = d + (d * d * d / 3.0);
    if mean < 0.0 {
        mean += TWO_PI;
    }
    mean
}

fn solve_barker(m: f64) -> f64 {
    let term = 3.0 * m + (9.0 * m * m + 1.0).sqrt();
    2.0 * (term.cbrt() - term.powf(-1.0 / 3.0)).atan()
}

fn solve_kepler_true_anomaly(e: f64, m: f64, max_iter: usize, tol: f64) -> f64 {
    let mut ratio = 1e15_f64;
    let mut iter = 0_usize;
    let anomaly = if e < 1.0 {
        let mut ecc_anomaly = m;
        while ratio.abs() > tol && iter <= max_iter {
            let f = ecc_anomaly - e * ecc_anomaly.sin() - m;
            let fp = 1.0 - e * ecc_anomaly.cos();
            ratio = f / fp;
            ecc_anomaly -= ratio;
            iter += 1;
        }
        2.0 * (((1.0 + e).sqrt() * (ecc_anomaly / 2.0).sin())
            .atan2((1.0 - e).sqrt() * (ecc_anomaly / 2.0).cos()))
    } else if e > 1.0 {
        let mut hyp_anomaly = m;
        while ratio.abs() > tol && iter <= max_iter {
            let f = e * hyp_anomaly.sinh() - hyp_anomaly - m;
            let fp = e * hyp_anomaly.cosh() - 1.0;
            ratio = f / fp;
            hyp_anomaly -= ratio;
            iter += 1;
        }
        2.0 * (((e + 1.0).sqrt() * (hyp_anomaly / 2.0).sinh())
            / ((e - 1.0).sqrt() * (hyp_anomaly / 2.0).cosh()))
        .atan()
    } else {
        solve_barker(m)
    };

    let mut nu = anomaly;
    if nu < 0.0 {
        nu += TWO_PI;
    }
    if nu >= TWO_PI {
        nu %= TWO_PI;
    }
    nu
}

pub fn keplerian_to_cartesian_a_row(v: &[f64; 6], mu: f64, max_iter: usize, tol: f64) -> [f64; 6] {
    let float_tol = 1e-15_f64;
    let a = v[0];
    let e = v[1];
    let i = v[2] / RAD2DEG;
    let raan = v[3] / RAD2DEG;
    let ap = v[4] / RAD2DEG;
    let m = v[5] / RAD2DEG;

    let p = if (e > (1.0 - float_tol)) && (e < (1.0 + float_tol)) {
        f64::NAN
    } else {
        a * (1.0 - e * e)
    };
    let nu = solve_kepler_true_anomaly(e, m, max_iter, tol);

    let denom = 1.0 + e * nu.cos();
    let r_pqw = [p * nu.cos() / denom, p * nu.sin() / denom, 0.0];
    let v_pqw = [
        -(mu / p).sqrt() * nu.sin(),
        (mu / p).sqrt() * (e + nu.cos()),
        0.0,
    ];

    let cos_raan = raan.cos();
    let sin_raan = raan.sin();
    let cos_ap = ap.cos();
    let sin_ap = ap.sin();
    let cos_i = i.cos();
    let sin_i = i.sin();

    let p1 = [
        [cos_ap, -sin_ap, 0.0],
        [sin_ap, cos_ap, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let p2 = [[1.0, 0.0, 0.0], [0.0, cos_i, -sin_i], [0.0, sin_i, cos_i]];
    let p3 = [
        [cos_raan, -sin_raan, 0.0],
        [sin_raan, cos_raan, 0.0],
        [0.0, 0.0, 1.0],
    ];

    let rotation = mul3x3(mul3x3(p3, p2), p1);
    let r = mul3x1(rotation, r_pqw);
    let vel = mul3x1(rotation, v_pqw);
    [r[0], r[1], r[2], vel[0], vel[1], vel[2]]
}

fn mul3x3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mul3x1(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2],
        a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2],
        a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2],
    ]
}

pub fn cartesian_to_keplerian_row(v: &[f64; 6], t0: f64, mu: f64) -> [f64; 13] {
    let float_tol = 1e-15_f64;
    let r = [v[0], v[1], v[2]];
    let vel = [v[3], v[4], v[5]];

    let r_mag = norm(r);
    let v_mag = norm(vel);
    let sme = v_mag * v_mag / 2.0 - mu / r_mag;

    let h = cross(r, vel);
    let h_mag = norm(h);
    let n_vec = cross([0.0, 0.0, 1.0], h);
    let n_mag = norm(n_vec);

    let rv = dot(r, vel);
    let scale = v_mag * v_mag - mu / r_mag;
    let e_vec = [
        (scale * r[0] - rv * vel[0]) / mu,
        (scale * r[1] - rv * vel[1]) / mu,
        (scale * r[2] - rv * vel[2]) / mu,
    ];
    let e = norm(e_vec);

    let p = h_mag * h_mag / mu;
    let i = (clamp_unit(h[2] / h_mag)).acos();

    let mut raan = if n_mag == 0.0 {
        f64::NAN
    } else {
        (clamp_unit(n_vec[0] / n_mag)).acos()
    };
    if n_vec[1] < 0.0 {
        raan = TWO_PI - raan;
    }
    if i < float_tol || (i - TWO_PI).abs() < float_tol {
        raan = 0.0;
    }

    let mut ap = if (n_mag == 0.0) || (e == 0.0) {
        f64::NAN
    } else {
        (clamp_unit(dot(n_vec, e_vec) / (n_mag * e))).acos()
    };
    if e_vec[2] < 0.0 {
        ap = TWO_PI - ap;
    }
    if e.abs() < float_tol {
        ap = 0.0;
    }

    let mut nu = if e == 0.0 {
        f64::NAN
    } else {
        (clamp_unit(dot(e_vec, r) / (e * r_mag))).acos()
    };
    if rv < 0.0 {
        nu = TWO_PI - nu;
    }

    let near_parabolic = (e > (1.0 - float_tol)) && (e < (1.0 + float_tol));
    let a = if near_parabolic {
        f64::NAN
    } else {
        mu / (-2.0 * sme)
    };
    let q = if near_parabolic {
        p / 2.0
    } else {
        a * (1.0 - e)
    };
    let q_apo = if e < 1.0 {
        a * (1.0 + e)
    } else {
        f64::INFINITY
    };
    let m_anom = calc_mean_anomaly(nu, e);

    let n_mean = if near_parabolic {
        (mu / (2.0 * q.powi(3))).sqrt()
    } else {
        (mu / a.abs().powi(3)).sqrt()
    };
    let period = if e < (1.0 - float_tol) {
        TWO_PI / n_mean
    } else {
        f64::INFINITY
    };
    let dtp = if (m_anom > std::f64::consts::PI) && (e < (1.0 - float_tol)) {
        period - m_anom / n_mean
    } else {
        -m_anom / n_mean
    };
    let tp = t0 + dtp;

    [
        a,
        p,
        q,
        q_apo,
        e,
        i * RAD2DEG,
        raan * RAD2DEG,
        ap * RAD2DEG,
        m_anom * RAD2DEG,
        nu * RAD2DEG,
        n_mean * RAD2DEG,
        period,
        tp,
    ]
}
