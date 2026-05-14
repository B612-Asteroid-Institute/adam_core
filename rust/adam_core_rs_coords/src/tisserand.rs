//! Tisserand parameter — small kernel.
//!
//! Tp = a_p/a + 2·cos(i)·sqrt((a/a_p)·(1−e²))
//!
//! where a_p is the semi-major axis of the perturbing body (e.g. Jupiter),
//! a is the semi-major axis of the orbit under test, e its eccentricity,
//! and i the inclination (degrees).
//!
//! Used to classify Jupiter-family comets vs asteroids (Tp_J > 3 = asteroid,
//! 2 < Tp_J < 3 = JFC, Tp_J < 2 = Damocloid).

use rayon::prelude::*;

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;
const TISSERAND_PARALLEL_MIN_ROWS: usize = 4096;
#[cfg(target_os = "macos")]
const TISSERAND_VFORCE_MIN_ROWS: usize = 64;

#[cfg(target_os = "macos")]
mod vforce {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn vvcos(out: *mut f64, input: *const f64, n: *const i32);
        fn vvsqrt(out: *mut f64, input: *const f64, n: *const i32);
    }

    #[inline]
    fn len_i32(len: usize) -> i32 {
        i32::try_from(len).expect("vForce slice length must fit in i32")
    }

    #[inline]
    pub(super) fn cos_in_place(values: &mut [f64]) {
        let n = len_i32(values.len());
        // SAFETY: `values` is a unique mutable slice; vvcos reads and writes
        // exactly `n` doubles in place.
        unsafe { vvcos(values.as_mut_ptr(), values.as_ptr(), &n) };
    }

    #[inline]
    pub(super) fn sqrt_in_place(values: &mut [f64]) {
        let n = len_i32(values.len());
        // SAFETY: `values` is a unique mutable slice; vvsqrt reads and writes
        // exactly `n` doubles in place.
        unsafe { vvsqrt(values.as_mut_ptr(), values.as_ptr(), &n) };
    }
}

#[inline]
fn tisserand_parameter_row(a: f64, e: f64, i_deg: f64, ap: f64) -> f64 {
    let i_rad = i_deg * DEG2RAD;
    ap / a + 2.0 * i_rad.cos() * ((a / ap) * (1.0 - e * e)).sqrt()
}

fn fill_tisserand_parameter_serial(out: &mut [f64], a: &[f64], e: &[f64], i_deg: &[f64], ap: f64) {
    for (((dst, &a_i), &e_i), &i_d) in out.iter_mut().zip(a.iter()).zip(e.iter()).zip(i_deg) {
        *dst = tisserand_parameter_row(a_i, e_i, i_d, ap);
    }
}

#[cfg(target_os = "macos")]
fn tisserand_parameter_flat_vforce(a: &[f64], e: &[f64], i_deg: &[f64], ap: f64) -> Vec<f64> {
    let n = a.len();
    let mut cos_i = vec![0.0_f64; n];
    let mut root = vec![0.0_f64; n];
    for (((cos_dst, root_dst), &a_i), (&e_i, &i_d)) in cos_i
        .iter_mut()
        .zip(root.iter_mut())
        .zip(a.iter())
        .zip(e.iter().zip(i_deg.iter()))
    {
        *cos_dst = i_d * DEG2RAD;
        *root_dst = (a_i / ap) * (1.0 - e_i * e_i);
    }
    vforce::cos_in_place(&mut cos_i);
    vforce::sqrt_in_place(&mut root);

    let mut out = vec![0.0_f64; n];
    for (((dst, &a_i), &cos_i_i), &root_i) in out
        .iter_mut()
        .zip(a.iter())
        .zip(cos_i.iter())
        .zip(root.iter())
    {
        *dst = ap / a_i + 2.0 * cos_i_i * root_i;
    }
    out
}

/// Per-row Tisserand parameter against a perturber with semi-major axis `ap`.
/// Inputs are flat arrays of length `n`; inclination is in DEGREES.
pub fn tisserand_parameter_flat(a: &[f64], e: &[f64], i_deg: &[f64], ap: f64) -> Vec<f64> {
    assert_eq!(a.len(), e.len());
    assert_eq!(a.len(), i_deg.len());
    let n = a.len();

    #[cfg(target_os = "macos")]
    if n >= TISSERAND_VFORCE_MIN_ROWS {
        return tisserand_parameter_flat_vforce(a, e, i_deg, ap);
    }

    let mut out = vec![0.0_f64; n];
    if n < TISSERAND_PARALLEL_MIN_ROWS {
        fill_tisserand_parameter_serial(&mut out, a, e, i_deg, ap);
        return out;
    }

    out.par_iter_mut()
        .zip(a.par_iter())
        .zip(e.par_iter())
        .zip(i_deg.par_iter())
        .for_each(|(((dst, &a_i), &e_i), &i_d)| {
            *dst = tisserand_parameter_row(a_i, e_i, i_d, ap);
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
