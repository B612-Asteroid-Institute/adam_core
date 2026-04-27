//! Mahalanobis-style χ² per row: chi2 = r·Σ⁻¹·rᵀ for residual r and
//! covariance Σ. Assumes Σ is symmetric-positive-definite (true for
//! valid covariance matrices); solves L·Lᵀ = Σ via in-place Cholesky
//! and computes y·y where L·y = r — equivalent to r·Σ⁻¹·rᵀ but skips
//! the explicit inverse and is O(D³/6) per row instead of O(D³).
//!
//! NaN policy: a NaN on the diagonal of any covariance row triggers
//! `Err`. NaN off-diagonals are silently substituted with 0.0 (matches
//! the legacy `np.where(isnan, 0, cov)` behavior); the caller is
//! expected to surface that via a UserWarning at the Python boundary.
//!
//! Output: `Vec<f64>` of length N.

use rayon::prelude::*;

#[derive(Debug)]
pub enum Chi2Error {
    NanDiagonal { row: usize, dim: usize },
    NotPositiveDefinite { row: usize },
}

/// Per-row chi² over flat (N, D) residuals and (N, D, D) covariances.
pub fn calculate_chi2_flat(
    residuals_flat: &[f64],
    covariances_flat: &[f64],
    n: usize,
    d: usize,
) -> Result<Vec<f64>, Chi2Error> {
    assert_eq!(residuals_flat.len(), n * d);
    assert_eq!(covariances_flat.len(), n * d * d);

    // Pre-flight: scan diagonals for NaN. Cheap; surface a clean error
    // instead of letting Cholesky propagate a NaN through the result.
    for i in 0..n {
        let cov_off = i * d * d;
        for k in 0..d {
            if covariances_flat[cov_off + k * d + k].is_nan() {
                return Err(Chi2Error::NanDiagonal { row: i, dim: k });
            }
        }
    }

    let results: Result<Vec<f64>, Chi2Error> = (0..n)
        .into_par_iter()
        .map(|i| {
            let r_off = i * d;
            let cov_off = i * d * d;
            chi2_one_row(
                &residuals_flat[r_off..r_off + d],
                &covariances_flat[cov_off..cov_off + d * d],
                d,
                i,
            )
        })
        .collect();
    results
}

#[inline]
fn chi2_one_row(r: &[f64], cov: &[f64], d: usize, row_idx: usize) -> Result<f64, Chi2Error> {
    // Copy cov (substituting NaN off-diagonals with 0) into a stack-or-heap
    // buffer; small D (≤8) stays on the stack via a fixed-size array, larger
    // D (rare for chi²) falls to a heap Vec.
    if d <= 8 {
        let mut a = [0.0_f64; 64];
        for j in 0..d {
            for k in 0..d {
                let v = cov[j * d + k];
                a[j * d + k] = if v.is_nan() { 0.0 } else { v };
            }
        }
        cholesky_in_place_solve_dot(&mut a[..d * d], r, d, row_idx)
    } else {
        let mut a: Vec<f64> = cov.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect();
        cholesky_in_place_solve_dot(&mut a, r, d, row_idx)
    }
}

/// In-place Cholesky factorization of `a` (D×D, row-major), then forward-
/// substitution to solve L·y = r and return y·y.
///
/// Σ = L·Lᵀ where L is lower-triangular with positive diagonal.
/// chi² = r·Σ⁻¹·rᵀ = (L⁻¹·r)·(L⁻¹·r) = y·y where L·y = r.
#[inline]
fn cholesky_in_place_solve_dot(
    a: &mut [f64],
    r: &[f64],
    d: usize,
    row_idx: usize,
) -> Result<f64, Chi2Error> {
    // Cholesky: for j in 0..d, compute L[j,j] then L[i,j] for i>j.
    // We only fill the lower triangle (overwriting `a` row-major in place).
    for j in 0..d {
        // L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2)
        let mut diag = a[j * d + j];
        for k in 0..j {
            diag -= a[j * d + k] * a[j * d + k];
        }
        if diag <= 0.0 || !diag.is_finite() {
            return Err(Chi2Error::NotPositiveDefinite { row: row_idx });
        }
        let l_jj = diag.sqrt();
        a[j * d + j] = l_jj;
        // L[i,j] = (A[i,j] - sum_{k<j} L[i,k]·L[j,k]) / L[j,j], i > j.
        for i in (j + 1)..d {
            let mut sum = a[i * d + j];
            for k in 0..j {
                sum -= a[i * d + k] * a[j * d + k];
            }
            a[i * d + j] = sum / l_jj;
        }
    }

    // Forward substitution: solve L·y = r in place. Reuse a buffer.
    if d <= 8 {
        let mut y = [0.0_f64; 8];
        for i in 0..d {
            let mut s = r[i];
            for k in 0..i {
                s -= a[i * d + k] * y[k];
            }
            y[i] = s / a[i * d + i];
        }
        let mut chi2 = 0.0;
        for k in 0..d {
            chi2 += y[k] * y[k];
        }
        Ok(chi2)
    } else {
        let mut y = vec![0.0_f64; d];
        for i in 0..d {
            let mut s = r[i];
            for k in 0..i {
                s -= a[i * d + k] * y[k];
            }
            y[i] = s / a[i * d + i];
        }
        Ok(y.iter().map(|&v| v * v).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn diag_2d_matches_naive() {
        // Σ = diag(0.04, 0.09); r = (0.1, 0.2)
        // chi² = 0.01/0.04 + 0.04/0.09 = 0.25 + 0.44444 = 0.69444…
        let r = [0.1, 0.2];
        let cov = [0.04, 0.0, 0.0, 0.09];
        let out = calculate_chi2_flat(&r, &cov, 1, 2).unwrap();
        let expected = 0.01 / 0.04 + 0.04 / 0.09;
        assert!(approx(out[0], expected, 1e-14), "got {} expected {}", out[0], expected);
    }

    #[test]
    fn correlated_2d_matches_inverse() {
        // Σ = [[1, 0.5], [0.5, 1]] → Σ⁻¹ = [[4/3, -2/3], [-2/3, 4/3]]
        // r = (1, 1) → chi² = 1·(4/3 - 2/3) + 1·(-2/3 + 4/3) = 2/3 + 2/3 = 4/3
        let r = [1.0, 1.0];
        let cov = [1.0, 0.5, 0.5, 1.0];
        let out = calculate_chi2_flat(&r, &cov, 1, 2).unwrap();
        assert!(approx(out[0], 4.0 / 3.0, 1e-14));
    }

    #[test]
    fn nan_offdiagonal_treated_as_zero() {
        // Same as diag case but off-diagonals are NaN.
        let r = [0.1, 0.2];
        let cov = [0.04, f64::NAN, f64::NAN, 0.09];
        let out = calculate_chi2_flat(&r, &cov, 1, 2).unwrap();
        let expected = 0.01 / 0.04 + 0.04 / 0.09;
        assert!(approx(out[0], expected, 1e-14));
    }

    #[test]
    fn nan_diagonal_errors() {
        let r = [0.1, 0.2];
        let cov = [f64::NAN, 0.0, 0.0, 0.09];
        let err = calculate_chi2_flat(&r, &cov, 1, 2).unwrap_err();
        match err {
            Chi2Error::NanDiagonal { row: 0, dim: 0 } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn six_d_state_matches_inverse() {
        // SPD 6×6 = identity scaled. r = (1,…,1). chi² = 6.
        let r = [1.0_f64; 6];
        let mut cov = [0.0_f64; 36];
        for i in 0..6 {
            cov[i * 6 + i] = 1.0;
        }
        let out = calculate_chi2_flat(&r, &cov, 1, 6).unwrap();
        assert!(approx(out[0], 6.0, 1e-14));
    }
}
