//! Outlier rejection with re-inclusion after Carpino, Milani & Chesley (2003).
//!
//! Rust-canonical decision kernels of `adam_core.orbit_determination.rejection`
//! ("CMC2003"): observations are rejected and re-included by testing each
//! post-fit residual against its EXPECTED residual covariance rather than
//! against its reported uncertainty alone, the scheme implemented in OrbFit
//! (`least_squares.f90` `reject_obs`, defaults in `lib/reject.def`) and used
//! by NEODyS/AstDyS. The fit loop that drives these kernels (refit, whitened
//! residuals, analytic Jacobian) is orchestrated by the Python
//! `cmc2003_fit_detailed`; the per-pass arithmetic lives here:
//!
//! * [`cmc2003_apparitions`] — time-sorted groups split at gaps longer than
//!   `apparition_gap_days`;
//! * [`cmc2003_expected_residual_chi2`] — `chi2_i = r_i^T (I ∓ P_i)^-1 r_i`
//!   with `P_i = J_i C J_i^T` the fit covariance projected onto the sky
//!   plane in whitened units (`I - P` inside the fit, `I + P` outside), with
//!   the eigenvalue floor protecting positive definiteness;
//! * [`cmc2003_select`] — one reject / re-include decision pass with the
//!   hysteresis thresholds, batch rule, small-sample fudge, minimum-sample,
//!   apparition and rejected-fraction guards.
//!
//! Constants are OrbFit's `reject.def` defaults.

use std::collections::BTreeSet;

pub const CMC2003_CHI2_REJECT: f64 = 8.0;
pub const CMC2003_CHI2_RECOVER: f64 = 7.0;
pub const CMC2003_CHI2_FRAC: f64 = 0.25;
pub const CMC2003_MAX_ITERATIONS: usize = 15;
pub const CMC2003_MAX_REJECTED_FRACTION: f64 = 0.5;
pub const CMC2003_APPARITION_GAP_DAYS: f64 = 180.0;
/// Eigenvalue floor on the expected residual covariance, as a fraction of the
/// (whitened, i.e. unit) observation variance.
pub const CMC2003_PSD_FLOOR_FRAC: f64 = 0.05;
/// OrbFit: no rejection at all when `n <= round(0.5 * n_params)` (6 parameters).
pub const CMC2003_MIN_OBS: usize = 3;

/// Diagnostic flags raised while computing or selecting; rendered with the
/// legacy snake_case names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Cmc2003Flag {
    TooFewObservations,
    MaxIterations,
    MaxRejectedFraction,
    KeptLastInApparition,
    PsdFloor,
    NoFitCovariance,
    SingularResidualCovariance,
    NonFiniteChi2,
}

impl Cmc2003Flag {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TooFewObservations => "too_few_observations",
            Self::MaxIterations => "max_iterations",
            Self::MaxRejectedFraction => "max_rejected_fraction",
            Self::KeptLastInApparition => "kept_last_in_apparition",
            Self::PsdFloor => "psd_floor",
            Self::NoFitCovariance => "no_fit_covariance",
            Self::SingularResidualCovariance => "singular_residual_covariance",
            Self::NonFiniteChi2 => "non_finite_chi2",
        }
    }
}

/// Apparition index per observation: time-sorted groups split at gaps longer
/// than `gap_days` (stable sort, so equal epochs keep input order).
pub fn cmc2003_apparitions(mjd: &[f64], gap_days: f64) -> Vec<i64> {
    let mut order: Vec<usize> = (0..mjd.len()).collect();
    order.sort_by(|&a, &b| mjd[a].total_cmp(&mjd[b]));
    let mut apparition = vec![0_i64; mjd.len()];
    let mut current = 0_i64;
    let mut previous: Option<f64> = None;
    for index in order {
        if let Some(previous) = previous {
            if mjd[index] - previous > gap_days {
                current += 1;
            }
        }
        apparition[index] = current;
        previous = Some(mjd[index]);
    }
    apparition
}

/// Eigen-decomposition of a symmetric 2x2 matrix `[[a, b], [b, d]]`:
/// ascending eigenvalues and the matching unit eigenvectors (columns).
fn symmetric_eigen_2x2(a: f64, b: f64, d: f64) -> ([f64; 2], [[f64; 2]; 2]) {
    if b == 0.0 {
        if a <= d {
            return ([a, d], [[1.0, 0.0], [0.0, 1.0]]);
        }
        return ([d, a], [[0.0, 1.0], [1.0, 0.0]]);
    }
    let half_trace = 0.5 * (a + d);
    let half_diff = 0.5 * (a - d);
    let radius = (half_diff * half_diff + b * b).sqrt();
    let lambda_low = half_trace - radius;
    let lambda_high = half_trace + radius;
    // Eigenvector for lambda_low: (b, lambda_low - a) is non-degenerate when b != 0.
    let (vx, vy) = (b, lambda_low - a);
    let norm = (vx * vx + vy * vy).sqrt();
    let low = [vx / norm, vy / norm];
    let high = [-low[1], low[0]];
    ([lambda_low, lambda_high], [low, high])
}

/// Per-observation chi2 against the expected post-fit residual covariance.
///
/// `residuals` are the `(N, 2)` whitened (lon, lat) residual components,
/// `jacobian` the `(2N, 6)` Jacobian of the whitened residual vector (rows
/// `2i`, `2i+1` belong to observation i) and `covariance` the row-major 6x6
/// fit covariance. `None`, or a non-finite covariance, is treated as zero
/// prediction uncertainty (chi2 reduces to the plain residual chi2) and raises
/// `NoFitCovariance`. Returns the non-negative chi2 per observation and the
/// diagnostic flags raised.
pub fn cmc2003_expected_residual_chi2(
    residuals: &[f64],
    jacobian: &[f64],
    covariance: Option<&[f64]>,
    selected: &[bool],
    psd_floor_frac: f64,
) -> Result<(Vec<f64>, BTreeSet<Cmc2003Flag>), String> {
    let n = selected.len();
    if residuals.len() != 2 * n {
        return Err("residuals must have shape (N, 2)".to_string());
    }
    let mut flags = BTreeSet::new();
    let covariance = match covariance {
        Some(values) if values.len() != 36 => {
            return Err("covariance must have shape (6, 6)".to_string())
        }
        Some(values) if values.iter().all(|value| value.is_finite()) => Some(values),
        _ => {
            flags.insert(Cmc2003Flag::NoFitCovariance);
            None
        }
    };
    if covariance.is_some() && jacobian.len() != 12 * n {
        return Err("jacobian must have shape (2N, 6)".to_string());
    }

    let mut chi2 = vec![0.0_f64; n];
    for i in 0..n {
        let r = [residuals[2 * i], residuals[2 * i + 1]];
        // expected = I -/+ J_i C J_i^T
        let (mut a, mut b, mut d) = (1.0_f64, 0.0_f64, 1.0_f64);
        if let Some(c) = covariance {
            let j0 = &jacobian[(2 * i) * 6..(2 * i + 1) * 6];
            let j1 = &jacobian[(2 * i + 1) * 6..(2 * i + 2) * 6];
            let mut jc0 = [0.0_f64; 6];
            let mut jc1 = [0.0_f64; 6];
            for col in 0..6 {
                for k in 0..6 {
                    jc0[col] += j0[k] * c[k * 6 + col];
                    jc1[col] += j1[k] * c[k * 6 + col];
                }
            }
            let p00: f64 = (0..6).map(|k| jc0[k] * j0[k]).sum();
            let p01: f64 = (0..6).map(|k| jc0[k] * j1[k]).sum();
            let p11: f64 = (0..6).map(|k| jc1[k] * j1[k]).sum();
            if selected[i] {
                a -= p00;
                b -= p01;
                d -= p11;
            } else {
                a += p00;
                b += p01;
                d += p11;
            }
        }
        let (eigenvalues, vectors) = symmetric_eigen_2x2(a, b, d);
        if eigenvalues[0] < psd_floor_frac {
            flags.insert(Cmc2003Flag::PsdFloor);
            let floored = [
                eigenvalues[0].max(psd_floor_frac),
                eigenvalues[1].max(psd_floor_frac),
            ];
            a = floored[0] * vectors[0][0] * vectors[0][0]
                + floored[1] * vectors[1][0] * vectors[1][0];
            b = floored[0] * vectors[0][0] * vectors[0][1]
                + floored[1] * vectors[1][0] * vectors[1][1];
            d = floored[0] * vectors[0][1] * vectors[0][1]
                + floored[1] * vectors[1][1] * vectors[1][1];
        }
        let det = a * d - b * b;
        let value = if det == 0.0 {
            flags.insert(Cmc2003Flag::SingularResidualCovariance);
            0.0
        } else {
            // r^T expected^-1 r with the explicit 2x2 inverse.
            let inv00 = d / det;
            let inv01 = -b / det;
            let inv11 = a / det;
            r[0] * (inv00 * r[0] + inv01 * r[1]) + r[1] * (inv01 * r[0] + inv11 * r[1])
        };
        let value = if value.is_finite() {
            value
        } else {
            flags.insert(Cmc2003Flag::NonFiniteChi2);
            0.0
        };
        chi2[i] = value.max(0.0);
    }
    Ok((chi2, flags))
}

/// Thresholds and guards for one [`cmc2003_select`] pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cmc2003SelectionOptions {
    pub chi2_reject: f64,
    pub chi2_recover: f64,
    pub chi2_frac: f64,
    pub max_rejected_fraction: f64,
    pub one_at_a_time: bool,
    pub min_obs: usize,
}

impl Default for Cmc2003SelectionOptions {
    fn default() -> Self {
        Self {
            chi2_reject: CMC2003_CHI2_REJECT,
            chi2_recover: CMC2003_CHI2_RECOVER,
            chi2_frac: CMC2003_CHI2_FRAC,
            max_rejected_fraction: CMC2003_MAX_REJECTED_FRACTION,
            one_at_a_time: false,
            min_obs: CMC2003_MIN_OBS,
        }
    }
}

/// Result of one [`cmc2003_select`] pass.
#[derive(Debug, Clone, PartialEq)]
pub struct Cmc2003Selection {
    pub selected: Vec<bool>,
    pub n_rejected: usize,
    pub n_recovered: usize,
    pub flags: BTreeSet<Cmc2003Flag>,
}

/// One CMC2003 reject / re-include decision pass.
///
/// Re-include an excluded observation when `chi2 <= chi2_recover + 0.75 *
/// fudge`; reject a selected one when `chi2 >= threshold` and `chi2 >
/// chi2_reject + fudge`, where `threshold` is `chi2_frac` times the worst
/// selected chi2 (batch rule) or the worst chi2 itself in one-at-a-time mode,
/// and the small-sample fudge is `400 * 3**(-n_selected)`. Guards: never below
/// `max(ceil(n * (1 - max_rejected_fraction)), min_obs + 1)` selected, and the
/// last selected observation of an apparition is never rejected.
pub fn cmc2003_select(
    chi2: &[f64],
    selected: &[bool],
    apparitions: &[i64],
    options: &Cmc2003SelectionOptions,
) -> Result<Cmc2003Selection, String> {
    let n = chi2.len();
    if selected.len() != n || apparitions.len() != n {
        return Err("chi2, selected and apparitions must have equal length".to_string());
    }
    let n_selected = selected.iter().filter(|&&flag| flag).count();
    let mut flags = BTreeSet::new();
    let mut new_selected = selected.to_vec();

    let chi2_max = if n_selected > 0 {
        chi2.iter()
            .zip(selected)
            .filter(|(_, &keep)| keep)
            .map(|(&value, _)| value)
            .fold(f64::NEG_INFINITY, f64::max)
    } else {
        0.0
    };
    let threshold = if options.one_at_a_time {
        chi2_max
    } else {
        chi2_max * options.chi2_frac
    };
    let fudge = 400.0 * 3.0_f64.powf(-(n_selected as f64));

    let mut n_recovered = 0_usize;
    for i in 0..n {
        if !selected[i] && chi2[i] <= options.chi2_recover + 0.75 * fudge {
            new_selected[i] = true;
            n_recovered += 1;
        }
    }

    let mut n_rejected = 0_usize;
    let min_keep = (n as f64 * (1.0 - options.max_rejected_fraction)).ceil() as usize;
    let mut candidates: Vec<usize> = (0..n)
        .filter(|&i| selected[i] && chi2[i] >= threshold && chi2[i] > options.chi2_reject + fudge)
        .collect();
    // Worst first; ties keep index order (stable sort on -chi2).
    candidates.sort_by(|&a, &b| chi2[b].total_cmp(&chi2[a]));
    for i in candidates {
        let currently_selected = new_selected.iter().filter(|&&flag| flag).count();
        if currently_selected.saturating_sub(1) < min_keep.max(options.min_obs + 1)
            || currently_selected == 0
        {
            flags.insert(Cmc2003Flag::MaxRejectedFraction);
            break;
        }
        let same_apparition = new_selected
            .iter()
            .zip(apparitions)
            .filter(|(&keep, &apparition)| keep && apparition == apparitions[i])
            .count();
        if same_apparition <= 1 {
            flags.insert(Cmc2003Flag::KeptLastInApparition);
            continue;
        }
        new_selected[i] = false;
        n_rejected += 1;
    }

    Ok(Cmc2003Selection {
        selected: new_selected,
        n_rejected,
        n_recovered,
        flags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn select(
        chi2: &[f64],
        selected: &[bool],
        apparitions: Option<&[i64]>,
        one_at_a_time: bool,
    ) -> Cmc2003Selection {
        let zeros = vec![0_i64; chi2.len()];
        let options = Cmc2003SelectionOptions {
            one_at_a_time,
            ..Cmc2003SelectionOptions::default()
        };
        cmc2003_select(chi2, selected, apparitions.unwrap_or(&zeros), &options).unwrap()
    }

    fn base(n: usize) -> (Vec<f64>, Vec<bool>) {
        (vec![1.0; n], vec![true; n])
    }

    #[test]
    fn nothing_happens_below_thresholds_and_rejects_above() {
        let (mut chi2, selected) = base(30);
        chi2[5] = 7.9;
        let out = select(&chi2, &selected, None, false);
        assert_eq!((out.n_rejected, out.n_recovered), (0, 0));
        assert!(out.flags.is_empty() && out.selected.iter().all(|&s| s));
        chi2[5] = 8.5;
        let out = select(&chi2, &selected, None, false);
        assert_eq!(out.n_rejected, 1);
        assert!(!out.selected[5]);
        assert_eq!(out.selected.iter().filter(|&&s| s).count(), 29);
    }

    #[test]
    fn batch_rule_one_at_a_time_and_recovery_hysteresis() {
        let (mut chi2, selected) = base(30);
        chi2[5] = 100.0;
        chi2[6] = 30.0;
        chi2[7] = 20.0;
        let out = select(&chi2, &selected, None, false);
        assert_eq!(out.n_rejected, 2);
        assert!(!out.selected[5] && !out.selected[6] && out.selected[7]);

        let (mut chi2, selected) = base(30);
        chi2[5] = 100.0;
        chi2[6] = 90.0;
        let out = select(&chi2, &selected, None, true);
        assert_eq!(out.n_rejected, 1);
        assert!(!out.selected[5] && out.selected[6]);

        let (mut chi2, mut selected) = base(30);
        selected[3] = false;
        selected[4] = false;
        chi2[3] = 6.9;
        chi2[4] = 7.5;
        let out = select(&chi2, &selected, None, false);
        assert_eq!((out.n_recovered, out.n_rejected), (1, 0));
        assert!(out.selected[3] && !out.selected[4]);
    }

    #[test]
    fn small_sample_fudge_and_guards() {
        // 5 selected: fudge = 400 * 3**-5 = 1.646, so 9 < 8 + fudge stays.
        let chi2 = [1.0, 1.0, 1.0, 1.0, 9.0];
        assert_eq!(select(&chi2, &[true; 5], None, true).n_rejected, 0);
        let chi2 = [1.0, 1.0, 1.0, 1.0, 10.0];
        assert_eq!(select(&chi2, &[true; 5], None, true).n_rejected, 1);

        let (mut chi2, selected) = base(10);
        for (i, value) in chi2.iter_mut().enumerate().take(8) {
            *value = 100.0 - i as f64;
        }
        let out = select(&chi2, &selected, None, false);
        assert_eq!(out.selected.iter().filter(|&&s| s).count(), 5);
        assert_eq!(out.n_rejected, 5);
        assert!(out.flags.contains(&Cmc2003Flag::MaxRejectedFraction));

        let (mut chi2, selected) = base(6);
        let apparitions = [0, 0, 0, 0, 0, 1];
        chi2[5] = 50.0;
        chi2[0] = 50.0;
        let out = select(&chi2, &selected, Some(&apparitions), false);
        assert!(out.selected[5] && !out.selected[0]);
        assert_eq!(out.n_rejected, 1);
        assert!(out.flags.contains(&Cmc2003Flag::KeptLastInApparition));
    }

    #[test]
    fn apparitions_split_at_gap() {
        assert_eq!(
            cmc2003_apparitions(&[10.0, 12.0, 400.0, 11.0, 401.0], 180.0),
            vec![0, 0, 1, 0, 1]
        );
        let mjd: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(cmc2003_apparitions(&mjd, 180.0).iter().all(|&a| a == 0));
    }

    #[test]
    fn expected_residual_chi2_rules() {
        let residuals = [1.0, 2.0, 0.5, 0.0];
        let (chi2, flags) =
            cmc2003_expected_residual_chi2(&residuals, &[0.0; 24], None, &[true, false], 0.05)
                .unwrap();
        assert_eq!(chi2, vec![5.0, 0.25]);
        assert_eq!(flags, BTreeSet::from([Cmc2003Flag::NoFitCovariance]));

        // P = 0.5 I for every observation: inside -> I - P = 0.5 I doubles the
        // chi2, outside -> I + P = 1.5 I shrinks it.
        let residuals = [1.0, 0.0, 1.0, 0.0];
        let mut jacobian = vec![0.0; 24];
        jacobian[0] = 1.0; // row 0, col 0
        jacobian[7] = 1.0; // row 1, col 1
        jacobian[12] = 1.0; // row 2, col 0
        jacobian[19] = 1.0; // row 3, col 1
        let mut covariance = [0.0; 36];
        covariance[0] = 0.5;
        covariance[7] = 0.5;
        let (chi2, flags) = cmc2003_expected_residual_chi2(
            &residuals,
            &jacobian,
            Some(&covariance),
            &[true, false],
            0.05,
        )
        .unwrap();
        assert!((chi2[0] - 2.0).abs() < 1e-12 && (chi2[1] - 1.0 / 1.5).abs() < 1e-12);
        assert!(flags.is_empty());

        // I - P = 0.01 I, floored to 0.05 I -> chi2 = 1 / 0.05
        let mut covariance = [0.0; 36];
        covariance[0] = 0.99;
        covariance[7] = 0.99;
        let (chi2, flags) = cmc2003_expected_residual_chi2(
            &[1.0, 0.0],
            &jacobian[..12],
            Some(&covariance),
            &[true],
            0.05,
        )
        .unwrap();
        assert!((chi2[0] - 20.0).abs() < 1e-9);
        assert!(flags.contains(&Cmc2003Flag::PsdFloor));

        // Off-diagonal projection exercises the eigen path.
        let (eigenvalues, vectors) = symmetric_eigen_2x2(2.0, 1.0, 2.0);
        assert!((eigenvalues[0] - 1.0).abs() < 1e-12 && (eigenvalues[1] - 3.0).abs() < 1e-12);
        for v in vectors {
            assert!((v[0] * v[0] + v[1] * v[1] - 1.0).abs() < 1e-12);
        }
    }
}
