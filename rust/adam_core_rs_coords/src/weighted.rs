//! Small linear-algebra kernels for weighted-sample statistics.
//!
//! `weighted_mean`:        mean[d]   = Σ_i W[i] · samples[i,:]
//! `weighted_covariance`:  cov[d,d]  = Σ_i W[i] · residual[i] ⊗ residual[i]
//!
//! Both written as tight rust loops with the row-sequential access
//! pattern. Auto-vectorizes well for typical D=6 (compiler unrolls the
//! inner dim loop into a SIMD FMA chain). Tried via `faer::matmul` but
//! its GEMM dispatch overhead is too high for the small-D shape — the
//! hand-rolled loop wins ~3-5× at n=100k. Faer remains in the workspace
//! for upcoming Cholesky/eigen needs (Wave E2 follow-up #148).
//!
//! Used by `coordinates/variants.py::create_coordinate_variants` and the
//! sigma-point reconstruction step in `covariances.py`.

/// Weighted mean: `mean[k] = Σ_i W[i] · samples[i, k]` for k in 0..d.
pub fn weighted_mean_flat(samples_flat: &[f64], w: &[f64], n: usize, d: usize) -> Vec<f64> {
    assert_eq!(samples_flat.len(), n * d);
    assert_eq!(w.len(), n);
    let mut mean = vec![0.0_f64; d];
    // Row-sequential pass: outer i, inner k. Inner loop has fixed-D
    // FMA chain that compiler unrolls into SIMD. No rayon — for D=6
    // and N up to ~10⁵ the work is memory-bandwidth-bound and rayon
    // dispatch overhead exceeds the win.
    for i in 0..n {
        let wi = w[i];
        let row_off = i * d;
        for k in 0..d {
            mean[k] += wi * samples_flat[row_off + k];
        }
    }
    mean
}

/// Weighted covariance: `cov[j, k] = Σ_i W_cov[i] · (s[i,j]−m[j]) · (s[i,k]−m[k])`.
///
/// Returns a flat `(d * d)` row-major matrix.
pub fn weighted_covariance_flat(
    mean: &[f64],
    samples_flat: &[f64],
    w_cov: &[f64],
    n: usize,
    d: usize,
) -> Vec<f64> {
    assert_eq!(samples_flat.len(), n * d);
    assert_eq!(w_cov.len(), n);
    assert_eq!(mean.len(), d);
    let mut cov = vec![0.0_f64; d * d];
    // Single pass over rows; per-row outer-product accumulation.
    // For D=6 the inner D×D FMA block fits in registers; compiler
    // emits SIMD FMA columns. Sequential samples access — cache-friendly.
    for i in 0..n {
        let wi = w_cov[i];
        let row_off = i * d;
        // Build residual on the stack for this row.
        let mut r = [0.0_f64; 64];  // upper bound for D ≤ 64 (typical D=6)
        let r = &mut r[..d];
        for j in 0..d {
            r[j] = samples_flat[row_off + j] - mean[j];
        }
        // cov[j, k] += wi · r[j] · r[k]
        for j in 0..d {
            let wir = wi * r[j];
            for k in 0..d {
                cov[j * d + k] += wir * r[k];
            }
        }
    }
    cov
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn weighted_mean_uniform_weights_recovers_arithmetic_mean() {
        // 4 samples in 2D; uniform W = 1/4
        let samples = [
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ];
        let w = [0.25, 0.25, 0.25, 0.25];
        let mean = weighted_mean_flat(&samples, &w, 4, 2);
        assert!(approx_eq(mean[0], 4.0, 1e-15));
        assert!(approx_eq(mean[1], 5.0, 1e-15));
    }

    #[test]
    fn weighted_covariance_uniform_recovers_zero_for_constant_samples() {
        // All identical samples → zero covariance.
        let samples = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let w = [1.0 / 3.0; 3];
        let mean = weighted_mean_flat(&samples, &w, 3, 2);
        let cov = weighted_covariance_flat(&mean, &samples, &w, 3, 2);
        assert!(approx_eq(cov[0], 0.0, 1e-15));
        assert!(approx_eq(cov[3], 0.0, 1e-15));
    }

    #[test]
    fn weighted_covariance_diagonal_case() {
        // Samples drawn from independent unit-variance: x ∈ {−1, +1}, y ∈ {−1, +1}.
        // Uniform weights (1/4 each), 4 samples covering all corners.
        let samples = [
            -1.0, -1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
        ];
        let w = [0.25; 4];
        let mean = weighted_mean_flat(&samples, &w, 4, 2);
        let cov = weighted_covariance_flat(&mean, &samples, &w, 4, 2);
        // E[x²] = E[y²] = 1; E[xy] = 0.
        assert!(approx_eq(cov[0], 1.0, 1e-15));  // var(x)
        assert!(approx_eq(cov[3], 1.0, 1e-15));  // var(y)
        assert!(approx_eq(cov[1], 0.0, 1e-15));  // cov(x,y)
        assert!(approx_eq(cov[2], 0.0, 1e-15));  // cov(y,x)
    }
}
