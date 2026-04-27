//! Absolute-magnitude H-fit kernels.
//!
//! Mirrors `photometry/absolute_magnitude.py::_fit_absolute_magnitude_rows`:
//! given per-detection (H_i, sigma_i) pairs (with H_i = m_observed - m_model0
//! and sigma_i either the reported `mag_sigma` or NaN), compute:
//!
//!   H_hat       — weighted mean if all sigmas finite, else arithmetic mean
//!   H_sigma     — propagated 1σ on H_hat (NaN if not computable)
//!   sigma_eff   — MAD-based robust scatter (NaN if N < 2)
//!   chi2_red    — reduced χ² (NaN if N < 2 or any sigma NaN)
//!   n_used      — # rows in the fit
//!
//! NaN-as-sentinel for the optional outputs; the Python wrapper translates
//! NaN → None to preserve the existing Optional[float] interface.

use rayon::prelude::*;

/// Per-group fit result. NaN where the metric isn't applicable.
#[derive(Debug, Clone, Copy)]
pub struct AbsMagFit {
    pub h_hat: f64,
    pub h_sigma: f64,
    pub sigma_eff: f64,
    pub chi2_red: f64,
    pub n_used: i64,
}

#[inline]
fn nanmedian(buf: &mut [f64]) -> f64 {
    // Filter NaN first, then partition-based O(N) median via
    // `select_nth_unstable_by` (introselect — partial sort that
    // places the k-th element in its sorted position without
    // sorting the rest). Even-N case takes the max of the lower
    // half, which is `slice[..n/2].iter().max()` — still O(N)
    // since the lower half is already partitioned ≤ pivot.
    let mut finite: Vec<f64> = buf.iter().copied().filter(|x| x.is_finite()).collect();
    let n = finite.len();
    if n == 0 {
        return f64::NAN;
    }
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
    if n.is_multiple_of(2) {
        // Place the upper-median element at index n/2; everything
        // strictly below at indices < n/2.
        let (lower, upper, _rest) = finite.select_nth_unstable_by(n / 2, cmp);
        let lower_max = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower_max + *upper)
    } else {
        let (_lower, mid, _upper) = finite.select_nth_unstable_by(n / 2, cmp);
        *mid
    }
}

#[inline]
fn mad_sigma(residuals: &[f64]) -> f64 {
    let med = {
        let mut buf: Vec<f64> = residuals.to_vec();
        nanmedian(&mut buf)
    };
    if !med.is_finite() {
        return f64::NAN;
    }
    let mut absdev: Vec<f64> = residuals.iter().map(|&x| (x - med).abs()).collect();
    let mad = nanmedian(&mut absdev);
    1.4826 * mad
}

/// Single-group fit. `h_rows` and `sigma_rows` are equal-length slices.
/// Returns NaN-bearing fields where a metric isn't applicable.
pub fn fit_absolute_magnitude_rows(h_rows: &[f64], sigma_rows: &[f64]) -> AbsMagFit {
    let n_used = h_rows.len() as i64;
    if n_used == 0 {
        return AbsMagFit {
            h_hat: f64::NAN,
            h_sigma: f64::NAN,
            sigma_eff: f64::NAN,
            chi2_red: f64::NAN,
            n_used: 0,
        };
    }

    let have_all_sigma = sigma_rows.iter().all(|s| s.is_finite());

    let h_hat = if have_all_sigma {
        // Weighted mean with w = 1/σ²
        let mut wsum = 0.0_f64;
        let mut wnum = 0.0_f64;
        for i in 0..h_rows.len() {
            let w = 1.0 / (sigma_rows[i] * sigma_rows[i]);
            wsum += w;
            wnum += w * h_rows[i];
        }
        if !wsum.is_finite() || wsum <= 0.0 {
            return AbsMagFit {
                h_hat: f64::NAN,
                h_sigma: f64::NAN,
                sigma_eff: f64::NAN,
                chi2_red: f64::NAN,
                n_used,
            };
        }
        wnum / wsum
    } else {
        // Arithmetic mean over finite h_rows.
        let mut s = 0.0_f64;
        let mut n = 0usize;
        for &h in h_rows {
            if h.is_finite() {
                s += h;
                n += 1;
            }
        }
        if n == 0 {
            return AbsMagFit {
                h_hat: f64::NAN,
                h_sigma: f64::NAN,
                sigma_eff: f64::NAN,
                chi2_red: f64::NAN,
                n_used,
            };
        }
        s / (n as f64)
    };

    let residual: Vec<f64> = h_rows.iter().map(|&h| h - h_hat).collect();

    let sigma_eff = if n_used >= 2 {
        let s = mad_sigma(&residual);
        if s.is_finite() { s } else { f64::NAN }
    } else {
        f64::NAN
    };

    let (h_sigma, chi2_red) = if have_all_sigma && n_used >= 2 {
        let mut wsum = 0.0_f64;
        let mut chi2 = 0.0_f64;
        for i in 0..h_rows.len() {
            let w = 1.0 / (sigma_rows[i] * sigma_rows[i]);
            wsum += w;
            chi2 += w * residual[i] * residual[i];
        }
        let chi2_red = chi2 / ((n_used - 1) as f64);
        let mut h_sigma = (1.0 / wsum).sqrt();
        if chi2_red.is_finite() && chi2_red > 1.0 {
            h_sigma *= chi2_red.sqrt();
        }
        (h_sigma, chi2_red)
    } else if sigma_eff.is_finite() && n_used >= 2 {
        (sigma_eff / (n_used as f64).sqrt(), f64::NAN)
    } else {
        (f64::NAN, f64::NAN)
    };

    AbsMagFit {
        h_hat,
        h_sigma,
        sigma_eff,
        chi2_red,
        n_used,
    }
}

/// Batched grouped fit. `group_offsets` is monotonically increasing with
/// `group_offsets[0] == 0` and `group_offsets[K] == h_rows.len()`. Returns
/// K fits, one per group, in input order.
pub fn fit_absolute_magnitude_grouped(
    h_rows: &[f64],
    sigma_rows: &[f64],
    group_offsets: &[usize],
) -> Vec<AbsMagFit> {
    assert_eq!(h_rows.len(), sigma_rows.len());
    assert!(group_offsets.first().copied() == Some(0));
    assert_eq!(group_offsets.last().copied().unwrap_or(0), h_rows.len());
    let k = group_offsets.len() - 1;
    (0..k)
        .into_par_iter()
        .map(|g| {
            let s = group_offsets[g];
            let e = group_offsets[g + 1];
            fit_absolute_magnitude_rows(&h_rows[s..e], &sigma_rows[s..e])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn weighted_mean_with_uniform_sigmas_recovers_arithmetic_mean() {
        let h = [15.0, 16.0, 17.0, 18.0];
        let sig = [0.1, 0.1, 0.1, 0.1];
        let fit = fit_absolute_magnitude_rows(&h, &sig);
        assert!(approx(fit.h_hat, 16.5, 1e-15));
        assert_eq!(fit.n_used, 4);
        assert!(fit.chi2_red.is_finite());
    }

    #[test]
    fn nan_sigma_falls_back_to_arithmetic_mean() {
        let h = [15.0, 16.0, 17.0];
        let sig = [0.1, f64::NAN, 0.2];
        let fit = fit_absolute_magnitude_rows(&h, &sig);
        assert!(approx(fit.h_hat, 16.0, 1e-15));
        assert!(fit.chi2_red.is_nan());  // not all sigmas finite
    }

    #[test]
    fn single_row_has_no_scatter_metric() {
        let fit = fit_absolute_magnitude_rows(&[16.5], &[0.1]);
        assert!(approx(fit.h_hat, 16.5, 1e-15));
        assert!(fit.sigma_eff.is_nan());
        assert!(fit.chi2_red.is_nan());
    }

    #[test]
    fn empty_returns_nan() {
        let fit = fit_absolute_magnitude_rows(&[], &[]);
        assert!(fit.h_hat.is_nan());
        assert_eq!(fit.n_used, 0);
    }

    #[test]
    fn grouped_three_groups() {
        let h = [15.0, 16.0, 16.5, 17.5, 19.0, 20.0, 21.0];
        let sig = [0.1, 0.1, 0.05, 0.05, 0.2, 0.2, 0.2];
        let offsets = [0, 2, 4, 7];
        let fits = fit_absolute_magnitude_grouped(&h, &sig, &offsets);
        assert_eq!(fits.len(), 3);
        assert!(approx(fits[0].h_hat, 15.5, 1e-15));
        assert!(approx(fits[1].h_hat, 17.0, 1e-15));
        assert!(approx(fits[2].h_hat, 20.0, 1e-15));
    }
}
