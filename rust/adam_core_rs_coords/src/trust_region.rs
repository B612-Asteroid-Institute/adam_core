//! Shared trust-region Gauss-Newton step policy.

pub(crate) const MAX_STEP_ATTEMPTS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RankDeficientStepPolicy {
    /// Use the minimum-norm Gauss-Newton step when it lies inside the radius.
    MinimumNormInterior,
    /// Preserve the OD solver's established SciPy-compatible boundary step.
    Boundary,
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Exact step of a diagonalized Gauss-Newton model.
///
/// `eigenvalues` and `gradient` are the normal-matrix eigenvalues and gradient
/// components in the same eigenbasis. The returned step minimizes the quadratic
/// model inside `radius`; `initial_alpha` warm-starts the Moré-Sorensen solve.
/// `rank_deficient_policy` makes the only intentional difference between the
/// color solver and the pre-existing OD solver explicit.
pub(crate) fn trust_region_step(
    eigenvalues: &[f64],
    gradient: &[f64],
    radius: f64,
    initial_alpha: f64,
    rank_deficient_policy: RankDeficientStepPolicy,
) -> (Vec<f64>, f64) {
    debug_assert_eq!(eigenvalues.len(), gradient.len());
    let radius = radius.max(f64::MIN_POSITIVE);
    let gradient_norm = norm(gradient);
    if gradient_norm == 0.0 {
        return (vec![0.0; gradient.len()], 0.0);
    }
    let lambda_max = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let rank_cutoff = f64::EPSILON * f64::EPSILON * lambda_max;
    let full_rank = lambda_max > 0.0 && eigenvalues.iter().all(|&value| value > rank_cutoff);
    let step_at = |alpha: f64| -> Vec<f64> {
        gradient
            .iter()
            .zip(eigenvalues)
            .map(|(&value, &eigenvalue)| -value / (eigenvalue + alpha))
            .collect()
    };
    let gauss_newton = gradient
        .iter()
        .zip(eigenvalues)
        .map(|(&value, &eigenvalue)| {
            if eigenvalue > rank_cutoff {
                -value / eigenvalue
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let null_gradient = gradient
        .iter()
        .zip(eigenvalues)
        .filter(|(_, eigenvalue)| **eigenvalue <= rank_cutoff)
        .map(|(&value, _)| value * value)
        .sum::<f64>()
        .sqrt();
    let use_interior_step =
        full_rank || rank_deficient_policy == RankDeficientStepPolicy::MinimumNormInterior;
    if use_interior_step
        && null_gradient <= f64::EPSILON.sqrt() * gradient_norm
        && norm(&gauss_newton) <= radius
    {
        return (gauss_newton, 0.0);
    }
    let phi_and_derivative = |alpha: f64| -> (f64, f64) {
        let step_norm = norm(&step_at(alpha));
        let phi = step_norm - radius;
        let derivative = -gradient
            .iter()
            .zip(eigenvalues)
            .map(|(&value, &eigenvalue)| value * value / (eigenvalue + alpha).powi(3))
            .sum::<f64>()
            / step_norm;
        (phi, derivative)
    };
    let mut upper = gradient_norm / radius;
    let mut lower = if full_rank {
        let (phi, derivative) = phi_and_derivative(0.0);
        -phi / derivative
    } else {
        0.0
    };
    let fallback = |lower: f64, upper: f64| (0.001 * upper).max((lower * upper).sqrt());
    let mut alpha = initial_alpha;
    if alpha.is_nan() || alpha <= 0.0 || alpha < lower || alpha > upper {
        alpha = fallback(lower, upper);
    }
    for _ in 0..10 {
        if alpha < lower || alpha > upper {
            alpha = fallback(lower, upper);
        }
        let (phi, derivative) = phi_and_derivative(alpha);
        if phi < 0.0 {
            upper = alpha;
        }
        let ratio = phi / derivative;
        lower = lower.max(alpha - ratio);
        alpha -= (phi + radius) * ratio / radius;
        if phi.abs() < 0.01 * radius {
            break;
        }
    }
    let mut step = step_at(alpha);
    let step_norm = norm(&step);
    if step_norm > 0.0 {
        let factor = radius / step_norm;
        for value in &mut step {
            *value *= factor;
        }
    }
    (step, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_deficient_policy_separates_color_and_od_steps() {
        let eigenvalues = [4.0, 1.0, 0.0];
        let gradient = [2.0, -1.0, 0.0];
        let radius = 10.0;

        let (minimum_norm, alpha) = trust_region_step(
            &eigenvalues,
            &gradient,
            radius,
            0.0,
            RankDeficientStepPolicy::MinimumNormInterior,
        );
        assert_eq!(alpha, 0.0);
        assert_eq!(minimum_norm, vec![-0.5, 1.0, 0.0]);

        let (boundary, alpha) = trust_region_step(
            &eigenvalues,
            &gradient,
            radius,
            0.0,
            RankDeficientStepPolicy::Boundary,
        );
        assert!(alpha.is_finite());
        assert!(boundary.iter().all(|value| value.is_finite()));
        assert_ne!(boundary, minimum_norm);
        assert!((norm(&boundary) - radius).abs() <= 1.0e-12 * radius);
    }
}
