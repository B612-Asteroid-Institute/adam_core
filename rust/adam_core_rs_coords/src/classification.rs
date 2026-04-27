//! Orbital dynamical class assignment over (a, e, q, Q).
//!
//! Mirrors the PDS Small Bodies Node classification scheme used in
//! `orbits/classification.py`. The rules apply in input order; later
//! matches overwrite earlier ones (this matches the legacy numpy-where
//! sequence-of-assignments pattern).
//!
//! Returns class codes (i32) per row; the caller maps codes → strings.
//! Codes:
//!   0  AST  (default)
//!   1  AMO  (a > 1.0, 1.017 < q < 1.3)
//!   2  APO  (a > 1.0, q < 1.017)
//!   3  ATE  (a < 1.0, Q > 0.983)
//!   4  CEN  (5.5 < a < 30.1)
//!   5  IEO  (Q < 0.983)
//!   6  IMB  (a < 2.0, q > 1.666)
//!   7  MBA  (2.0 < a < 3.2, q > 1.666)
//!   8  MCA  (a < 3.2, 1.3 < q < 1.666)
//!   9  OMB  (3.2 < a < 4.6)
//!   10 TJN  (4.6 < a < 5.5, e < 0.3)
//!   11 TNO  (a > 30.1)
//!   12 PAA  (e == 1)
//!   13 HYA  (e > 1)

use rayon::prelude::*;

pub const AST: i32 = 0;
pub const AMO: i32 = 1;
pub const APO: i32 = 2;
pub const ATE: i32 = 3;
pub const CEN: i32 = 4;
pub const IEO: i32 = 5;
pub const IMB: i32 = 6;
pub const MBA: i32 = 7;
pub const MCA: i32 = 8;
pub const OMB: i32 = 9;
pub const TJN: i32 = 10;
pub const TNO: i32 = 11;
pub const PAA: i32 = 12;
pub const HYA: i32 = 13;

#[inline]
fn classify_row(a: f64, e: f64, q: f64, q_apo: f64) -> i32 {
    let mut code = AST;
    if a > 1.0 && q > 1.017 && q < 1.3 { code = AMO; }
    if a > 1.0 && q < 1.017            { code = APO; }
    if a < 1.0 && q_apo > 0.983        { code = ATE; }
    if a > 5.5 && a < 30.1             { code = CEN; }
    if q_apo < 0.983                   { code = IEO; }
    if a < 2.0 && q > 1.666            { code = IMB; }
    if a > 2.0 && a < 3.2 && q > 1.666 { code = MBA; }
    if a < 3.2 && q > 1.3 && q < 1.666 { code = MCA; }
    if a > 3.2 && a < 4.6              { code = OMB; }
    if a > 4.6 && a < 5.5 && e < 0.3   { code = TJN; }
    if a > 30.1                        { code = TNO; }
    if e == 1.0                        { code = PAA; }
    if e > 1.0                         { code = HYA; }
    code
}

/// Per-row classification. All input arrays must have equal length.
pub fn classify_orbits_flat(a: &[f64], e: &[f64], q: &[f64], q_apo: &[f64]) -> Vec<i32> {
    assert_eq!(a.len(), e.len());
    assert_eq!(a.len(), q.len());
    assert_eq!(a.len(), q_apo.len());
    let n = a.len();
    let mut out = vec![0_i32; n];
    out.par_iter_mut()
        .zip(a.par_iter())
        .zip(e.par_iter())
        .zip(q.par_iter())
        .zip(q_apo.par_iter())
        .for_each(|((((dst, &a_i), &e_i), &q_i), &q_apo_i)| {
            *dst = classify_row(a_i, e_i, q_i, q_apo_i);
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_main_belt_asteroid() {
        // Ceres: a=2.77 AU, e=0.08, q=2.55, Q=2.99
        assert_eq!(classify_row(2.77, 0.08, 2.55, 2.99), MBA);
    }
    #[test]
    fn classifies_apollo() {
        // Apollo-class: a=1.5, e=0.5, q=0.75, Q=2.25
        assert_eq!(classify_row(1.5, 0.5, 0.75, 2.25), APO);
    }
    #[test]
    fn classifies_hyperbolic() {
        // 1I/Oumuamua: a=-1.27, e=1.20
        assert_eq!(classify_row(-1.27, 1.20, 0.26, 0.0), HYA);
    }
    #[test]
    fn classifies_tno() {
        // Pluto: a=39.4
        assert_eq!(classify_row(39.4, 0.25, 29.6, 49.3), TNO);
    }
}
