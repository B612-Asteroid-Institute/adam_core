//! Forward-mode automatic differentiation via dual numbers.
//!
//! A `Dual<N>` carries a value (`re`) and an N-component tangent vector (`du`).
//! Arithmetic on `Dual<N>` propagates derivatives via the chain rule, so that
//! evaluating a function `f: R^N -> R` with `N` inputs seeded as unit-tangent
//! dual variables yields `f(x)` in `re` and the gradient `df/dx_i` in `du[i]`.
//! For `f: R^N -> R^M`, the caller evaluates each output component with the
//! same seeded inputs; each output's `du` is one row of the Jacobian, giving
//! the full Jacobian in a single forward pass.
//!
//! The `Scalar` trait abstracts over `f64` and `Dual<N>` so the same kernel
//! code computes both values (with `T = f64`) and Jacobians (with `T = Dual<N>`).

// Dual-number arithmetic indexes into parallel tangent arrays; the
// `for i in 0..N` form is natural here and rewriting it as
// `iter_mut().enumerate()` would still need a separate index into the
// second tangent. Disable the lint at crate scope.
#![allow(clippy::needless_range_loop)]

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    fn from_f64(x: f64) -> Self;
    /// Returns the value component (for comparisons and branching).
    fn re(&self) -> f64;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: f64) -> Self;
    fn abs(self) -> Self;
    fn signum(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn cbrt(self) -> Self;
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}

impl Scalar for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn re(&self) -> f64 {
        *self
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f64::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f64::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        f64::atan(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    #[inline]
    fn powf(self, n: f64) -> Self {
        f64::powf(self, n)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn signum(self) -> Self {
        f64::signum(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        f64::sin_cos(self)
    }
    #[inline]
    fn sinh(self) -> Self {
        f64::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        f64::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        f64::asinh(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        f64::acosh(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        f64::atanh(self)
    }
    #[inline]
    fn cbrt(self) -> Self {
        f64::cbrt(self)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Dual<const N: usize> {
    pub re: f64,
    pub du: [f64; N],
}

impl<const N: usize> Dual<N> {
    #[inline]
    pub fn constant(v: f64) -> Self {
        Self {
            re: v,
            du: [0.0; N],
        }
    }
    /// Creates a dual with value `v` and a unit tangent in component `idx`.
    #[inline]
    pub fn variable(v: f64, idx: usize) -> Self {
        let mut du = [0.0_f64; N];
        du[idx] = 1.0;
        Self { re: v, du }
    }
    /// Seeds a length-N input vector so each component carries its own unit tangent.
    #[inline]
    pub fn seed(values: [f64; N]) -> [Self; N] {
        let mut out = [Self::constant(0.0); N];
        for (i, &v) in values.iter().enumerate() {
            out[i] = Self::variable(v, i);
        }
        out
    }
    #[inline]
    fn map_du<F: Fn(f64) -> f64>(mut self, f: F) -> Self {
        for x in self.du.iter_mut() {
            *x = f(*x);
        }
        self
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = self;
        out.re += rhs.re;
        for i in 0..N {
            out.du[i] += rhs.du[i];
        }
        out
    }
}
impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = self;
        out.re -= rhs.re;
        for i in 0..N {
            out.du[i] -= rhs.du[i];
        }
        out
    }
}
impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // (a + a'ε)(b + b'ε) = ab + (a'b + a b')ε
        let re = self.re * rhs.re;
        let mut du = [0.0_f64; N];
        for i in 0..N {
            du[i] = self.du[i] * rhs.re + self.re * rhs.du[i];
        }
        Self { re, du }
    }
}
impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // d(a/b) = (a' b - a b') / b^2
        let re = self.re / rhs.re;
        let inv_b2 = 1.0 / (rhs.re * rhs.re);
        let mut du = [0.0_f64; N];
        for i in 0..N {
            du[i] = (self.du[i] * rhs.re - self.re * rhs.du[i]) * inv_b2;
        }
        Self { re, du }
    }
}
impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.map_du(|x| -x).with_re(-self.re)
    }
}

impl<const N: usize> Dual<N> {
    #[inline]
    fn with_re(mut self, re: f64) -> Self {
        self.re = re;
        self
    }
}

impl<const N: usize> AddAssign for Dual<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<const N: usize> SubAssign for Dual<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<const N: usize> MulAssign for Dual<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<const N: usize> DivAssign for Dual<N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const N: usize> Scalar for Dual<N> {
    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::constant(x)
    }
    #[inline]
    fn re(&self) -> f64 {
        self.re
    }
    #[inline]
    fn sqrt(self) -> Self {
        let re = self.re.sqrt();
        // d(sqrt(x)) = 1/(2 sqrt(x))
        let k = 0.5 / re;
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.map_du(|d| d * c).with_re(s)
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.map_du(|d| -d * s).with_re(c)
    }
    #[inline]
    fn tan(self) -> Self {
        let t = self.re.tan();
        // d(tan(x)) = 1 + tan^2
        let k = 1.0 + t * t;
        self.map_du(|d| d * k).with_re(t)
    }
    #[inline]
    fn asin(self) -> Self {
        let re = self.re.asin();
        // d(asin(x)) = 1 / sqrt(1 - x^2)
        let k = 1.0 / (1.0 - self.re * self.re).sqrt();
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn acos(self) -> Self {
        let re = self.re.acos();
        // d(acos(x)) = -1 / sqrt(1 - x^2)
        let k = -1.0 / (1.0 - self.re * self.re).sqrt();
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn atan(self) -> Self {
        let re = self.re.atan();
        // d(atan(x)) = 1 / (1 + x^2)
        let k = 1.0 / (1.0 + self.re * self.re);
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        // atan2(y, x): dy = x/(x^2+y^2), dx = -y/(x^2+y^2)
        let re = self.re.atan2(other.re);
        let r2 = self.re * self.re + other.re * other.re;
        let dy = other.re / r2;
        let dx = -self.re / r2;
        let mut du = [0.0_f64; N];
        for i in 0..N {
            du[i] = self.du[i] * dy + other.du[i] * dx;
        }
        Self { re, du }
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        // hypot(x, y) = sqrt(x^2 + y^2)
        // d/dx = x/hypot, d/dy = y/hypot
        let re = self.re.hypot(other.re);
        let dx = self.re / re;
        let dy = other.re / re;
        let mut du = [0.0_f64; N];
        for i in 0..N {
            du[i] = self.du[i] * dx + other.du[i] * dy;
        }
        Self { re, du }
    }
    #[inline]
    fn exp(self) -> Self {
        let re = self.re.exp();
        self.map_du(|d| d * re).with_re(re)
    }
    #[inline]
    fn ln(self) -> Self {
        let re = self.re.ln();
        let k = 1.0 / self.re;
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::constant(1.0);
        }
        let re = self.re.powi(n);
        // d(x^n) = n x^{n-1}
        let k = (n as f64) * self.re.powi(n - 1);
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn powf(self, n: f64) -> Self {
        let re = self.re.powf(n);
        // d(x^n) = n x^{n-1}
        let k = n * self.re.powf(n - 1.0);
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn abs(self) -> Self {
        // d|x|/dx = sign(x); undefined at 0 (we pick 0).
        let s = self.re.signum();
        let re = self.re.abs();
        self.map_du(|d| d * s).with_re(re)
    }
    #[inline]
    fn signum(self) -> Self {
        // Derivative is 0 almost everywhere; undefined at 0. We return 0.
        Self::constant(self.re.signum())
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        // Branches on value. Nondifferentiable at equality; pick self side.
        if self.re <= other.re {
            self
        } else {
            other
        }
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self.re >= other.re {
            self
        } else {
            other
        }
    }
    #[inline]
    fn sinh(self) -> Self {
        let re = self.re.sinh();
        let k = self.re.cosh();
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn cosh(self) -> Self {
        let re = self.re.cosh();
        let k = self.re.sinh();
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn tanh(self) -> Self {
        let t = self.re.tanh();
        // d(tanh(x)) = 1 - tanh^2
        let k = 1.0 - t * t;
        self.map_du(|d| d * k).with_re(t)
    }
    #[inline]
    fn asinh(self) -> Self {
        let re = self.re.asinh();
        // d(asinh(x)) = 1 / sqrt(x^2 + 1)
        let k = 1.0 / (self.re * self.re + 1.0).sqrt();
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn acosh(self) -> Self {
        let re = self.re.acosh();
        // d(acosh(x)) = 1 / sqrt(x^2 - 1)
        let k = 1.0 / (self.re * self.re - 1.0).sqrt();
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn atanh(self) -> Self {
        let re = self.re.atanh();
        // d(atanh(x)) = 1 / (1 - x^2)
        let k = 1.0 / (1.0 - self.re * self.re);
        self.map_du(|d| d * k).with_re(re)
    }
    #[inline]
    fn cbrt(self) -> Self {
        let re = self.re.cbrt();
        // d(x^(1/3)) = 1 / (3 x^(2/3))
        let k = 1.0 / (3.0 * re * re);
        self.map_du(|d| d * k).with_re(re)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() <= TOL * (1.0 + b.abs())
    }

    #[test]
    fn chain_rule_sin_of_square() {
        // f(x) = sin(x^2); f'(x) = 2x cos(x^2)
        for x in [0.3_f64, 1.1, -0.5, 2.7] {
            let xd: Dual<1> = Dual::variable(x, 0);
            let y = (xd * xd).sin();
            assert!(approx(y.re, (x * x).sin()));
            assert!(approx(y.du[0], 2.0 * x * (x * x).cos()));
        }
    }

    #[test]
    fn jacobian_r2_to_r2_in_one_pass() {
        // f(x, y) = (x^2 + y, y * sin(x)); expected Jacobian = [[2x, 1], [y cos x, sin x]]
        let x = 0.7_f64;
        let y = 1.4_f64;
        let v: [Dual<2>; 2] = Dual::seed([x, y]);
        let f0 = v[0] * v[0] + v[1];
        let f1 = v[1] * v[0].sin();
        assert!(approx(f0.re, x * x + y));
        assert!(approx(f1.re, y * x.sin()));
        assert!(approx(f0.du[0], 2.0 * x));
        assert!(approx(f0.du[1], 1.0));
        assert!(approx(f1.du[0], y * x.cos()));
        assert!(approx(f1.du[1], x.sin()));
    }

    #[test]
    fn newton_solver_propagates_derivative() {
        // Solve f(E, M) = E - e sin(E) - M = 0 for E as a function of M,
        // then dE/dM = 1 / (1 - e cos(E)).
        fn kepler_solve<T: Scalar>(m: T, e: T) -> T {
            let mut e_val = m;
            for _ in 0..50 {
                let (s, c) = e_val.sin_cos();
                let f = e_val - e * s - m;
                let fp = T::from_f64(1.0) - e * c;
                let step = f / fp;
                e_val -= step;
                if step.re().abs() < 1e-14 {
                    break;
                }
            }
            e_val
        }
        let m = 1.3_f64;
        let e = 0.2_f64;
        let md: Dual<1> = Dual::variable(m, 0);
        let ed: Dual<1> = Dual::constant(e);
        let solved = kepler_solve(md, ed);
        let expected_de_dm = 1.0 / (1.0 - e * solved.re.cos());
        assert!(approx(solved.du[0], expected_de_dm));
    }

    #[test]
    fn f64_and_dual_value_paths_agree() {
        fn g<T: Scalar>(x: T, y: T) -> T {
            (x.sin() + y.cos() * T::from_f64(2.0)).sqrt()
        }
        let x = 0.9_f64;
        let y = -0.3_f64;
        let vf = g::<f64>(x, y);
        let xd: Dual<2> = Dual::variable(x, 0);
        let yd: Dual<2> = Dual::variable(y, 1);
        let vd = g(xd, yd);
        assert!(approx(vd.re, vf));
    }
}
