//! Fused observed-vs-predicted residuals + chi² over (N, D).
//!
//! One PyO3 crossing replaces what the Python `Residuals.calculate`
//! previously did via four sub-kernel crossings (`bound_longitude_residuals`,
//! `apply_cosine_latitude_correction` × 2, `calculate_chi2`) plus a
//! Python-side per-batch loop. This module is composition only: it calls
//! the existing `bound_longitude_residuals_flat`,
//! `apply_cosine_latitude_correction_flat`, and `calculate_chi2_flat` in
//! sequence and groups rows by NaN mask before chi² so each Cholesky-solve
//! sees a uniform-D batch.
//!
//! The component algorithms are unchanged from their direct-call form;
//! callers that exercise the sub-kernels directly continue to use them.

use crate::chi2::{calculate_chi2_flat, Chi2Error};
use crate::spherical_resid::{
    apply_cosine_latitude_correction_flat, bound_longitude_residuals_flat,
};

#[derive(Debug)]
pub enum ResidualsError {
    Chi2(Chi2Error),
    InvalidShape(&'static str),
}

impl From<Chi2Error> for ResidualsError {
    fn from(e: Chi2Error) -> Self {
        ResidualsError::Chi2(e)
    }
}

#[derive(Debug)]
pub struct ResidualsChi2Output {
    /// (N * D) row-major residuals (post longitude-wrap and cos-lat for
    /// the spherical case).
    pub residuals: Vec<f64>,
    /// (N,) chi² per row. NaN for rows whose batch covariance was entirely
    /// NaN within the active dimensions (matches the legacy Python skip).
    pub chi2: Vec<f64>,
    /// (N,) degrees of freedom = D − (NaN count in observed row).
    pub dof: Vec<i64>,
    /// `true` iff at least one batch had partial-NaN covariance within
    /// its active dimensions (mirrors the legacy per-batch UserWarning
    /// trigger). Callers surface the warning at the Python boundary.
    pub had_off_diagonal_nan: bool,
}

/// Fused residual + chi² pipeline.
///
/// `predicted_flat` must already be broadcast to `(N, D)` by the caller;
/// `predicted_cov_flat` must already have its NaN entries replaced with
/// 0 when the caller wants `use_predicted_covariance=False` semantics.
/// Both are cheap numpy ops and keep this kernel shape-uniform.
pub fn compute_residuals_chi2_flat(
    observed_flat: &[f64],
    predicted_flat: &[f64],
    observed_cov_flat: &[f64],
    predicted_cov_flat: &[f64],
    n: usize,
    d: usize,
    is_spherical: bool,
) -> Result<ResidualsChi2Output, ResidualsError> {
    if observed_flat.len() != n * d
        || predicted_flat.len() != n * d
        || observed_cov_flat.len() != n * d * d
        || predicted_cov_flat.len() != n * d * d
    {
        return Err(ResidualsError::InvalidShape(
            "observed/predicted/cov shapes must match (N, D) and (N, D, D)",
        ));
    }
    if d > 64 {
        return Err(ResidualsError::InvalidShape(
            "compute_residuals_chi2_flat supports D <= 64",
        ));
    }
    if is_spherical && d < 5 {
        return Err(ResidualsError::InvalidShape(
            "spherical residuals require D >= 5 (cols 1, 2, 4 used)",
        ));
    }

    // 1. residuals = observed - predicted
    let mut residuals = vec![0.0_f64; n * d];
    for i in 0..n * d {
        residuals[i] = observed_flat[i] - predicted_flat[i];
    }

    // 2. Mutable copies of the covariance buffers; the spherical branch
    //    writes back into them in place.
    let mut obs_cov = observed_cov_flat.to_vec();
    let mut pred_cov = predicted_cov_flat.to_vec();

    // 3. Spherical-only: longitude wrap + cos(lat) on residuals & both covs.
    if is_spherical {
        bound_longitude_residuals_flat(observed_flat, &mut residuals, n, d);

        // Latitude is column 2 of the (N, D) observed buffer.
        let mut lat_deg = vec![0.0_f64; n];
        for i in 0..n {
            lat_deg[i] = observed_flat[i * d + 2];
        }
        apply_cosine_latitude_correction_flat(&lat_deg, &mut residuals, &mut obs_cov, n, d);

        // Apply the same scaling to the predicted covariance only;
        // the legacy code passes a zeros buffer for residuals to
        // suppress the residuals path (which already ran above).
        let mut scratch = vec![0.0_f64; n * d];
        apply_cosine_latitude_correction_flat(&lat_deg, &mut scratch, &mut pred_cov, n, d);
    }

    // 4. total_cov = obs_cov + pred_cov
    let mut total_cov = obs_cov;
    for (slot, &p) in total_cov.iter_mut().zip(pred_cov.iter()) {
        *slot += p;
    }

    // 5. dof[i] = D - (NaN count in observed[i, :])
    let mut dof = vec![0i64; n];
    for i in 0..n {
        let mut nan_count = 0i64;
        for k in 0..d {
            if observed_flat[i * d + k].is_nan() {
                nan_count += 1;
            }
        }
        dof[i] = (d as i64) - nan_count;
    }

    // 6. Group rows by NaN mask of observed values and run chi² per group.
    //    chi² defaults to NaN where the active-dim covariance is all-NaN.
    let mut chi2 = vec![f64::NAN; n];
    let mut had_off_diagonal_nan = false;

    let mut masks = vec![0u64; n];
    for i in 0..n {
        let mut m: u64 = 0;
        for k in 0..d {
            if observed_flat[i * d + k].is_nan() {
                m |= 1u64 << k;
            }
        }
        masks[i] = m;
    }

    use std::collections::BTreeMap;
    let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (i, &m) in masks.iter().enumerate() {
        groups.entry(m).or_default().push(i);
    }

    for (mask, row_indices) in groups {
        let dims: Vec<usize> = (0..d).filter(|k| mask & (1u64 << *k) == 0).collect();
        let d_eff = dims.len();
        if d_eff == 0 {
            // No active dimensions in this batch: chi² stays NaN.
            continue;
        }
        let n_eff = row_indices.len();

        let mut packed_res = vec![0.0_f64; n_eff * d_eff];
        let mut packed_cov = vec![0.0_f64; n_eff * d_eff * d_eff];
        for (i_local, &row) in row_indices.iter().enumerate() {
            let res_off = row * d;
            let cov_off = row * d * d;
            for (j_local, &col) in dims.iter().enumerate() {
                packed_res[i_local * d_eff + j_local] = residuals[res_off + col];
                for (k_local, &col2) in dims.iter().enumerate() {
                    packed_cov[i_local * d_eff * d_eff + j_local * d_eff + k_local] =
                        total_cov[cov_off + col * d + col2];
                }
            }
        }

        // Match the legacy `if not np.all(np.isnan(covariances)):` skip
        // and the per-batch UserWarning emission.
        let any_nan = packed_cov.iter().any(|v| v.is_nan());
        let all_nan = any_nan && packed_cov.iter().all(|v| v.is_nan());
        if all_nan {
            // chi² stays NaN for these rows; no warning either (matches legacy).
            continue;
        }
        if any_nan {
            had_off_diagonal_nan = true;
        }

        let chi2_batch = calculate_chi2_flat(&packed_res, &packed_cov, n_eff, d_eff)?;
        for (i_local, &row) in row_indices.iter().enumerate() {
            chi2[row] = chi2_batch[i_local];
        }
    }

    Ok(ResidualsChi2Output {
        residuals,
        chi2,
        dof,
        had_off_diagonal_nan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn cartesian_unit_diag_chi2_matches_squared_residuals() {
        // observed[0] = [1, 1, 1], predicted[0] = [0, 0, 0], cov = I_3.
        // residual = (1, 1, 1); chi² = 3; dof = 3.
        let observed = [1.0, 1.0, 1.0];
        let predicted = [0.0, 0.0, 0.0];
        let mut obs_cov = [0.0_f64; 9];
        for i in 0..3 {
            obs_cov[i * 3 + i] = 1.0;
        }
        let pred_cov = [0.0_f64; 9];
        let out =
            compute_residuals_chi2_flat(&observed, &predicted, &obs_cov, &pred_cov, 1, 3, false)
                .unwrap();
        assert_eq!(out.residuals, vec![1.0, 1.0, 1.0]);
        assert!(approx(out.chi2[0], 3.0, 1e-14));
        assert_eq!(out.dof, vec![3]);
        assert!(!out.had_off_diagonal_nan);
    }

    #[test]
    fn cartesian_multi_batch_matches_legacy_dof() {
        // Mirror of test_Residuals_calculate fixture observed_array (D=6).
        // Row patterns produce dof = [1, 3, 2, 6] and chi² = [1, 3, 2, 0].
        let nan = f64::NAN;
        let observed = [
            0.2, nan, nan, nan, nan, nan, // row 0
            0.6, 1.0, 2.0, nan, nan, nan, // row 1
            nan, 3.0, nan, 4.0, nan, nan, // row 2
            0.5, 3.0, 0.5, 4.5, 0.1, 0.1, // row 3
        ];
        let predicted = [
            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // row 0
            0.5, 1.1, 1.9, 0.2, 0.1, 0.1, // row 1
            0.5, 2.9, 0.2, 4.1, 0.1, 0.1, // row 2
            0.5, 3.0, 0.5, 4.5, 0.1, 0.1, // row 3
        ];
        // Block-diagonal covariance per row: 0.01 on the active-dim diagonals,
        // 0 on the active-dim off-diagonals, NaN elsewhere.
        let mut obs_cov = vec![nan; 4 * 6 * 6];
        let active_dims_per_row: [&[usize]; 4] = [&[0], &[0, 1, 2], &[1, 3], &[0, 1, 2, 3, 4, 5]];
        for (row_idx, &active) in active_dims_per_row.iter().enumerate() {
            let off = row_idx * 36;
            for &di in active {
                for &dj in active {
                    obs_cov[off + di * 6 + dj] = if di == dj { 0.01 } else { 0.0 };
                }
            }
        }
        let pred_cov = vec![0.0_f64; 4 * 36];

        let out =
            compute_residuals_chi2_flat(&observed, &predicted, &obs_cov, &pred_cov, 4, 6, false)
                .unwrap();

        assert_eq!(out.dof, vec![1, 3, 2, 6]);
        // Expected chi² values from the legacy fixture (cov = 0.01 · I per
        // active dim, residual^2 · 100):
        //   row 0: r=(-0.1,)               → 0.01 / 0.01 = 1
        //   row 1: r=(0.1, -0.1, 0.1)      → (0.01+0.01+0.01)/0.01 = 3
        //   row 2: r=(0.1, -0.1)           → (0.01+0.01)/0.01 = 2
        //   row 3: r=zero                  → 0
        for (got, want) in out.chi2.iter().zip(&[1.0, 3.0, 2.0, 0.0]) {
            assert!(approx(*got, *want, 1e-12), "got {got} expected {want}");
        }
        // No NaN survives into a packed-cov off-diagonal in this case.
        assert!(!out.had_off_diagonal_nan);
    }

    #[test]
    fn off_diagonal_nan_sets_warning_flag() {
        // Single 3-D row with NaN on the active off-diagonal triggers the
        // warning flag but still computes chi² (NaN substituted with 0).
        let observed = [1.0, 1.0, 1.0];
        let predicted = [0.0, 0.0, 0.0];
        let obs_cov = [
            1.0,
            f64::NAN,
            0.0, // row 0
            f64::NAN,
            1.0,
            0.0, // row 1
            0.0,
            0.0,
            1.0, // row 2
        ];
        let pred_cov = [0.0_f64; 9];
        let out =
            compute_residuals_chi2_flat(&observed, &predicted, &obs_cov, &pred_cov, 1, 3, false)
                .unwrap();
        assert!(out.had_off_diagonal_nan);
        // chi² with off-diag NaN substituted: same as identity → 3.
        assert!(approx(out.chi2[0], 3.0, 1e-14));
    }

    #[test]
    fn diagonal_nan_returns_chi2_error() {
        let observed = [1.0, 1.0, 1.0];
        let predicted = [0.0, 0.0, 0.0];
        let mut obs_cov = [0.0_f64; 9];
        for i in 0..3 {
            obs_cov[i * 3 + i] = 1.0;
        }
        obs_cov[0] = f64::NAN; // diagonal
        let pred_cov = [0.0_f64; 9];
        let err =
            compute_residuals_chi2_flat(&observed, &predicted, &obs_cov, &pred_cov, 1, 3, false)
                .unwrap_err();
        match err {
            ResidualsError::Chi2(Chi2Error::NanDiagonal { .. }) => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn spherical_path_applies_longitude_wrap_and_cos_lat() {
        // 1 row, D=6, observed at lon=355°, lat=60° (cos=0.5).
        // observed - predicted in lon = 350 → wrap -> -10 → sign-flip -> +10.
        // After cos(lat=60°)=0.5 scaling, residual_col1 = +10 · 0.5 = 5.
        let observed = [1.0, 355.0, 60.0, 0.0, 4.0, 0.0];
        let predicted = [1.0, 5.0, 60.0, 0.0, 0.0, 0.0];
        let mut obs_cov = [0.0_f64; 36];
        for i in 0..6 {
            obs_cov[i * 6 + i] = 1.0;
        }
        let pred_cov = [0.0_f64; 36];
        let out =
            compute_residuals_chi2_flat(&observed, &predicted, &obs_cov, &pred_cov, 1, 6, true)
                .unwrap();
        // residuals[col 1] should be the wrapped+sign-flipped+cos-scaled +10*0.5 = 5.
        assert!(approx(out.residuals[1], 5.0, 1e-12));
        // residuals[col 4] = vlon_obs - vlon_pred = 4, scaled by cos(60°) = 2.
        assert!(approx(out.residuals[4], 2.0, 1e-12));
        assert_eq!(out.dof, vec![6]);
    }
}

/// Natural log of the gamma function (Lanczos approximation, g=7, n=9).
/// Absolute accuracy is ~1e-15 over the positive reals used here.
#[allow(clippy::excessive_precision)]
fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut accumulator = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        accumulator += coefficient / (x + index as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + accumulator.ln()
}

fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut term = 1.0 / a;
    let mut sum = term;
    let mut denominator = a;
    for _ in 0..500 {
        denominator += 1.0;
        term *= x / denominator;
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_q_continued_fraction(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for iteration in 1..500 {
        let an = -(iteration as f64) * (iteration as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Chi-squared survival function `P(X > chi2)` for `dof` degrees of freedom:
/// the regularized upper incomplete gamma `Q(dof/2, chi2/2)`. Mirrors
/// `1 - scipy.stats.chi2.cdf(chi2, dof)` to ~1e-15.
pub fn chi2_survival(chi2: f64, dof: f64) -> f64 {
    if !chi2.is_finite() || !dof.is_finite() || dof <= 0.0 || chi2 < 0.0 {
        return f64::NAN;
    }
    let a = dof / 2.0;
    let x = chi2 / 2.0;
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_continued_fraction(a, x)
    }
}

#[cfg(test)]
mod chi2_survival_tests {
    use super::chi2_survival;

    #[test]
    fn matches_scipy_reference_values() {
        // scipy.stats.chi2.sf reference values.
        let cases = [
            (1.0_f64, 1.0_f64, 0.31731050786291115_f64),
            (2.5, 2.0, 0.2865047968601901),
            (0.3, 2.0, 0.8607079764250578),
            (10.0, 4.0, 0.04042768199451279),
            (55.0, 6.0, 4.6354918724416246e-10),
            (0.05, 1.0, 0.8230632737581214),
        ];
        for (chi2, dof, expected) in cases {
            let actual = chi2_survival(chi2, dof);
            assert!(
                (actual - expected).abs() <= 1e-12 * expected.max(1e-300),
                "chi2_survival({chi2}, {dof}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn handles_edges() {
        assert!(chi2_survival(f64::NAN, 2.0).is_nan());
        assert!(chi2_survival(1.0, 0.0).is_nan());
        assert_eq!(chi2_survival(0.0, 2.0), 1.0);
    }
}
