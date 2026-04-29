//! Tisserand parameter — small kernel.
//!
//! Tp = a_p/a + 2·cos(i)·sqrt((a/a_p)·(1−e²))
//!
//! where a_p is the semi-major axis of the perturbing body (e.g. Jupiter),
//! a is the semi-major axis of the orbit under test, e its eccentricity,
//! and i the inclination (radians).
//!
//! Used to classify Jupiter-family comets vs asteroids (Tp_J > 3 = asteroid,
//! 2 < Tp_J < 3 = JFC, Tp_J < 2 = Damocloid).

use rayon::prelude::*;

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// Per-row Tisserand parameter against a perturber with semi-major axis `ap`.
/// Inputs are flat arrays of length `n`; inclination is in DEGREES.
pub fn tisserand_parameter_flat(a: &[f64], e: &[f64], i_deg: &[f64], ap: f64) -> Vec<f64> {
    assert_eq!(a.len(), e.len());
    assert_eq!(a.len(), i_deg.len());
    let n = a.len();
    let mut out = vec![0.0_f64; n];
    out.par_iter_mut()
        .zip(a.par_iter())
        .zip(e.par_iter())
        .zip(i_deg.par_iter())
        .for_each(|(((dst, &a_i), &e_i), &i_d)| {
            let i_rad = i_d * DEG2RAD;
            *dst = ap / a_i + 2.0 * i_rad.cos() * ((a_i / ap) * (1.0 - e_i * e_i)).sqrt();
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy_formula_jupiter() {
        // Random asteroid-like orbit: a=2.5 AU, e=0.1, i=5°
        let a = vec![2.5_f64];
        let e = vec![0.1_f64];
        let i = vec![5.0_f64];
        let ap = 5.203719697535582_f64; // Jupiter
        let tp = tisserand_parameter_flat(&a, &e, &i, ap)[0];
        // Hand-computed: 5.2037/2.5 + 2*cos(5°·π/180)*sqrt(2.5/5.2037*(1-0.01))
        let expected = ap / 2.5 + 2.0 * (5.0_f64.to_radians()).cos() * ((2.5 / ap) * 0.99).sqrt();
        assert!((tp - expected).abs() < 1e-15, "tp={tp} expected={expected}");
    }
}
