//! Helpers for spherical-coordinate residuals:
//!   * `bound_longitude_residuals_flat` — wrap the longitude residual to
//!     [-180, 180]° and flip sign on 0/360 boundary crossings.
//!   * `apply_cosine_latitude_correction_flat` — scale residuals[:, 1]
//!     and residuals[:, 4] by cos(lat); apply same scaling to covariance
//!     rows/cols 1 and 4 (NaN preserved).
//!
//! The covariance correction reduces from `D · cov · D.T` (with D
//! diagonal having 1, cos(lat), 1, 1, cos(lat), 1) to a per-element
//! scale: `out[i,j] = D[i] · cov[i,j] · D[j]`. Much simpler than the
//! full 3D einsum the legacy uses.

use rayon::prelude::*;

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// Wrap `residuals[:, 1]` (longitude residuals, degrees) to [-180, 180]
/// and flip sign on the 0°/360° boundary crossing per the convention:
/// 355 − 5 = +10, 5 − 355 = −10. Operates IN PLACE on `residuals_flat`.
///
/// `observed_flat` is the (N, D) observed values; only column 1 (the
/// observed longitude) is read.
pub fn bound_longitude_residuals_flat(
    observed_flat: &[f64],
    residuals_flat: &mut [f64],
    n: usize,
    d: usize,
) {
    assert_eq!(observed_flat.len(), n * d);
    assert_eq!(residuals_flat.len(), n * d);
    assert!(d >= 2, "spherical residuals require at least 2 dimensions");
    residuals_flat
        .par_chunks_mut(d)
        .zip(observed_flat.par_chunks(d))
        .for_each(|(row, obs)| {
            let lon_obs = obs[1];
            let mut lr = row[1];
            let lr_g180 = lr > 180.0;
            let lr_l180 = lr < -180.0;
            if lr_g180 {
                lr -= 360.0;
            } else if lr_l180 {
                lr += 360.0;
            }
            // Sign flip on boundary crossings (matches legacy logic).
            if lr_g180 && lon_obs > 180.0 {
                lr = -lr;
            } else if lr_l180 && lon_obs < 180.0 {
                lr = -lr;
            }
            row[1] = lr;
        });
}

/// Apply cos(latitude) factor to spherical residuals + covariance.
/// `lat_deg[N]` is the per-row latitude in degrees.
/// Operates in place on `residuals_flat[N*D]` (scales col 1 and col 4)
/// and `covariances_flat[N*D*D]` (scales rows/cols 1 and 4 of each
/// matrix; preserves NaN entries).
pub fn apply_cosine_latitude_correction_flat(
    lat_deg: &[f64],
    residuals_flat: &mut [f64],
    covariances_flat: &mut [f64],
    n: usize,
    d: usize,
) {
    assert_eq!(lat_deg.len(), n);
    assert_eq!(residuals_flat.len(), n * d);
    assert_eq!(covariances_flat.len(), n * d * d);
    assert!(d >= 5, "cos-lat correction requires D >= 5 (col 4 used)");

    // Per-row scale: D[i] = 1 except D[1] = D[4] = cos(lat).
    residuals_flat
        .par_chunks_mut(d)
        .zip(covariances_flat.par_chunks_mut(d * d))
        .zip(lat_deg.par_iter())
        .for_each(|((row, cov), &lat)| {
            let c = (lat * DEG2RAD).cos();
            // Scale residuals
            row[1] *= c;
            row[4] *= c;
            // Build per-dim D diagonal (size d). Only indices 1 and 4 are c.
            let scale = |i: usize| if i == 1 || i == 4 { c } else { 1.0 };
            // For each (i, j): cov_out[i, j] = D[i] · cov[i, j] · D[j].
            // NaN preserved (NaN * finite = NaN).
            for i in 0..d {
                let si = scale(i);
                for j in 0..d {
                    let sj = scale(j);
                    if si != 1.0 || sj != 1.0 {
                        let v = cov[i * d + j];
                        if !v.is_nan() {
                            cov[i * d + j] = si * v * sj;
                        }
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn longitude_no_wrap() {
        let obs = [1.0, 100.0, 0.0, 0.0, 0.0, 0.0];
        let mut res = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        bound_longitude_residuals_flat(&obs, &mut res, 1, 6);
        assert_eq!(res[1], 5.0);
    }

    #[test]
    fn longitude_wrap_355_minus_5_is_positive_10() {
        // observed = 355°, predicted = 5° → residual = 350°, should wrap to −10°
        // then sign-flip to +10° (since lr > 180 AND obs > 180).
        let obs = [1.0, 355.0, 0.0, 0.0, 0.0, 0.0];
        let mut res = [0.0, 350.0, 0.0, 0.0, 0.0, 0.0];
        bound_longitude_residuals_flat(&obs, &mut res, 1, 6);
        assert!(approx(res[1], 10.0, 1e-15));
    }

    #[test]
    fn longitude_wrap_5_minus_355_is_negative_10() {
        // observed = 5°, predicted = 355° → residual = −350°, wrap → +10°,
        // sign-flip → −10° (lr < -180 AND obs < 180).
        let obs = [1.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let mut res = [0.0, -350.0, 0.0, 0.0, 0.0, 0.0];
        bound_longitude_residuals_flat(&obs, &mut res, 1, 6);
        assert!(approx(res[1], -10.0, 1e-15));
    }

    #[test]
    fn cos_lat_zero_degrees_is_identity() {
        let lat = [0.0_f64];
        let mut res = [0.0, 5.0, 0.0, 0.0, 1.0, 0.0];
        let mut cov = [0.0_f64; 36];
        for i in 0..6 {
            cov[i * 6 + i] = 1.0;  // identity covariance
        }
        apply_cosine_latitude_correction_flat(&lat, &mut res, &mut cov, 1, 6);
        assert!(approx(res[1], 5.0, 1e-15));
        assert!(approx(res[4], 1.0, 1e-15));
        for i in 0..6 {
            assert!(approx(cov[i * 6 + i], 1.0, 1e-15));
        }
    }

    #[test]
    fn cos_lat_60_degrees_halves_lon_residual() {
        let lat = [60.0_f64];  // cos(60°) = 0.5
        let mut res = [0.0, 10.0, 0.0, 0.0, 4.0, 0.0];
        let mut cov = [0.0_f64; 36];
        for i in 0..6 {
            cov[i * 6 + i] = 1.0;
        }
        apply_cosine_latitude_correction_flat(&lat, &mut res, &mut cov, 1, 6);
        assert!(approx(res[1], 5.0, 1e-14));
        assert!(approx(res[4], 2.0, 1e-14));
        // Diagonal: cov[1,1] = 0.5·1·0.5 = 0.25; cov[4,4] = 0.25; others = 1.0.
        assert!(approx(cov[1 * 6 + 1], 0.25, 1e-14));
        assert!(approx(cov[4 * 6 + 4], 0.25, 1e-14));
        assert!(approx(cov[0 * 6 + 0], 1.0, 1e-15));
    }

    #[test]
    fn cos_lat_preserves_nan_in_cov() {
        let lat = [60.0_f64];
        let mut res = [0.0, 10.0, 0.0, 0.0, 4.0, 0.0];
        let mut cov = [0.0_f64; 36];
        cov[1 * 6 + 1] = f64::NAN;
        cov[1 * 6 + 4] = f64::NAN;
        for i in [0, 2, 3, 5] {
            cov[i * 6 + i] = 1.0;
        }
        cov[4 * 6 + 4] = 2.0;
        apply_cosine_latitude_correction_flat(&lat, &mut res, &mut cov, 1, 6);
        assert!(cov[1 * 6 + 1].is_nan());
        assert!(cov[1 * 6 + 4].is_nan());
        assert!(approx(cov[4 * 6 + 4], 0.5, 1e-14));  // 0.5 · 2.0 · 0.5
    }
}
