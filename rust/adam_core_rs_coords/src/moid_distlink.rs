// Pure-Rust elliptic DISTLINK 2.0 MOID kernel.
//
// Derived from the MIT-licensed DISTLINK 2.0 reference implementation,
// Copyright (c) 2018-2020 R.V. Baluev and D.V. Mikryukov. The complete MIT
// notice and port provenance are in
// THIRD_PARTY_LICENSES/distlink-2.0/LICENSE.txt and
// THIRD_PARTY_LICENSES/distlink-2.0/PORT_NOTES.md.
//
// Algorithm references:
// - Kholshevnikov & Vassiliev (1999), CMDA 75, 75-83.
// - Baluev & Mikryukov (2019), Astronomy and Computing 27, 11-22.

use num_complex::{Complex64, ComplexFloat};
use std::f64::consts::PI;
use std::sync::OnceLock;

const DEG: usize = 16;
const DIM: usize = 21;
const EPS: f64 = f64::EPSILON;
const TWO_PI: f64 = 2.0 * PI;

type C = Complex64;

#[inline]
fn sqr(x: f64) -> f64 {
    x * x
}

#[inline]
fn inverse(x: C) -> C {
    x.conj() / x.norm_sqr()
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a)
}

#[inline]
fn angle_wrap(x: f64) -> f64 {
    let x = (x + PI) % TWO_PI;
    if x < 0.0 {
        x + PI
    } else {
        x - PI
    }
}

fn roots_of_unity() -> &'static [C; DIM] {
    static EXPS: OnceLock<[C; DIM]> = OnceLock::new();
    EXPS.get_or_init(|| std::array::from_fn(|i| C::from_polar(1.0, TWO_PI * i as f64 / DIM as f64)))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Orbit {
    pub(crate) a: f64,
    pub(crate) e: f64,
    i: f64,
    node: f64,
    p: [f64; 3],
    q: [f64; 3],
}

impl Orbit {
    pub(crate) fn new(a: f64, e: f64, i: f64, w: f64, node: f64) -> Self {
        let (sw, cw) = w.sin_cos();
        let (so, co) = node.sin_cos();
        let (si, ci) = i.sin_cos();
        let p = [cw * co - ci * sw * so, cw * so + ci * sw * co, si * sw];
        let q = [-sw * co - ci * cw * so, ci * cw * co - sw * so, si * cw];
        Self {
            a,
            e,
            i,
            node,
            p,
            q,
        }
    }
}

#[derive(Clone, Debug)]
struct Aux {
    e1: f64,
    e2: f64,
    a1: f64,
    a2: f64,
    alpha1: f64,
    alpha2: f64,
    k: f64,
    pp: f64,
    ps: f64,
    sp: f64,
    ss: f64,
    p1: f64,
    p2: f64,
    abs_w: f64,
    p1w: f64,
    p2w: f64,
    q1w: f64,
    q2w: f64,
    mutual_i: f64,
    basis_p1: [f64; 3],
    basis_p2: [f64; 3],
    basis_q1: [f64; 3],
    basis_q2: [f64; 3],
}

impl Aux {
    fn new(o1: Orbit, o2: Orbit) -> Self {
        let e1 = o1.e.abs();
        let e2 = o2.e.abs();
        let a1 = o1.a.abs();
        let a2 = o2.a.abs();
        let p1 = a1 * (1.0 - e1 * e1);
        let p2 = a2 * (1.0 - e2 * e2);
        let alpha1 = a1 / a2;
        let alpha2 = a2 / a1;
        let k = alpha2 * e2 * e2;
        let eta1 = (1.0 - e1 * e1).abs().sqrt();
        let eta2 = (1.0 - e2 * e2).abs().sqrt();
        let (s1, c1) = o1.i.sin_cos();
        let (s2, c2) = o2.i.sin_cos();
        let w = [
            c1 * s2 * o2.node.cos() - s1 * c2 * o1.node.cos(),
            c1 * s2 * o2.node.sin() - s1 * c2 * o1.node.sin(),
            s1 * s2 * (o2.node - o1.node).sin(),
        ];
        let abs_w = norm(w).sqrt();
        let cos_i = c1 * c2 + s1 * s2 * (o2.node - o1.node).cos();
        let asin_w = abs_w.clamp(-1.0, 1.0).asin();
        let mutual_i = if cos_i > 0.0 { asin_w } else { PI - asin_w };
        let pp = dot(o1.p, o2.p);
        let ps = dot(o1.p, o2.q) * eta2;
        let sp = dot(o1.q, o2.p) * eta1;
        let ss = dot(o1.q, o2.q) * eta1 * eta2;
        Self {
            e1,
            e2,
            a1,
            a2,
            alpha1,
            alpha2,
            k,
            pp,
            ps,
            sp,
            ss,
            p1,
            p2,
            abs_w,
            p1w: dot(o1.p, w),
            p2w: dot(o2.p, w),
            q1w: dot(o1.q, w),
            q2w: dot(o2.q, w),
            mutual_i,
            basis_p1: o1.p,
            basis_p2: o2.p,
            basis_q1: o1.q,
            basis_q2: o2.q,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Result {
    pub(crate) good: bool,
    pub(crate) distance: f64,
    pub(crate) distance_error: f64,
    pub(crate) u1: f64,
    pub(crate) u2: f64,
    pub(crate) u1_error: f64,
    pub(crate) u2_error: f64,
    root_count: u16,
    min_delta: f64,
}

impl Default for Result {
    fn default() -> Self {
        Self {
            good: true,
            distance: -1.0,
            distance_error: 0.0,
            u1: 0.0,
            u2: 0.0,
            u1_error: 0.0,
            u2_error: 0.0,
            root_count: 0,
            min_delta: -1.0,
        }
    }
}

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    #[inline]
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let y = x.wrapping_mul(2_685_821_657_736_338_717);
        ((y >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[inline]
fn polynomial_valder(c: &[C], z: C, forward: bool) -> (C, C) {
    let n = c.len() - 1;
    if forward {
        let mut value = c[n];
        let mut derivative = c[n] * n as f64;
        if n > 2 {
            for i in (2..n).rev() {
                value = value * z + c[i];
                derivative = derivative * z + c[i] * i as f64;
            }
        }
        if n > 1 {
            value = value * z + c[1];
            derivative = derivative * z + c[1];
        }
        if n > 0 {
            value = value * z + c[0];
        }
        (value, derivative)
    } else {
        let mut value = c[0];
        let mut derivative = c[0] * n as f64;
        if n > 2 {
            for (i, coefficient) in c.iter().enumerate().take(n - 1).skip(1) {
                value = value * z + coefficient;
                derivative = derivative * z + coefficient * (n - i) as f64;
            }
        }
        if n > 1 {
            value = value * z + c[n - 1];
            derivative = derivative * z + c[n - 1];
        }
        if n > 0 {
            value = value * z + c[n];
        }
        (value, derivative)
    }
}

#[inline]
fn polynomial_valder2(c: &[C], z: C, forward: bool) -> (C, C, C) {
    let n = c.len() - 1;
    if forward {
        let mut value = c[n];
        let mut derivative = c[n] * n as f64;
        let mut second = c[n] * (n * (n - 1)) as f64;
        if n > 3 {
            for i in (3..n).rev() {
                value = value * z + c[i];
                derivative = derivative * z + c[i] * i as f64;
                second = second * z + c[i] * (i * (i - 1)) as f64;
            }
        }
        if n > 2 {
            value = value * z + c[2];
            derivative = derivative * z + 2.0 * c[2];
            second = second * z + 2.0 * c[2];
        }
        if n > 1 {
            value = value * z + c[1];
            derivative = derivative * z + c[1];
        }
        if n > 0 {
            value = value * z + c[0];
        }
        (value, derivative, second)
    } else {
        let mut value = c[0];
        let mut derivative = c[0] * n as f64;
        let mut second = c[0] * (n * (n - 1)) as f64;
        if n > 3 {
            for (i, coefficient) in c.iter().enumerate().take(n - 2).skip(1) {
                value = value * z + coefficient;
                derivative = derivative * z + coefficient * (n - i) as f64;
                second = second * z + coefficient * ((n - i) * (n - i - 1)) as f64;
            }
        }
        if n > 2 {
            value = value * z + c[n - 2];
            derivative = derivative * z + 2.0 * c[n - 2];
            second = second * z + 2.0 * c[n - 2];
        }
        if n > 1 {
            value = value * z + c[n - 1];
            derivative = derivative * z + c[n - 1];
        }
        if n > 0 {
            value = value * z + c[n];
        }
        (value, derivative, second)
    }
}

#[inline]
fn ratio(c: &[C], z: C, forward: bool) -> C {
    let (f, fd) = polynomial_valder(c, z, forward);
    f / fd
}

fn newton(c: &[C], z: &mut C, max_eps: f64, min_eps: f64, rng: &mut Rng) -> u64 {
    let max_eps = max_eps.abs().max(2.0 * min_eps.abs()).max(2.0 * EPS);
    let min_eps = min_eps.abs();
    let mut z1 = *z;
    let mut z2 = z1;
    let mut iterations = 0_u64;
    let mut error2: f64;
    let mut inner = true;
    loop {
        iterations += 1;
        let cycled = (iterations > 2 && (*z - z2).norm_sqr() < (*z - z1).norm_sqr() * 1e-6)
            || iterations % 300 == 0;
        let multiplier = if cycled {
            C::from_polar((2.0 * rng.next() + 1.0) / 3.0, rng.next() * TWO_PI)
        } else {
            C::new(1.0, 0.0)
        };
        let mut nrm = z.norm_sqr();
        if inner {
            if nrm > 5.0 {
                inner = false;
            }
        } else if 2.0 * nrm < 1.0 {
            inner = true;
        }
        z2 = z1;
        z1 = *z;
        let dz;
        if inner {
            dz = ratio(c, *z, true);
            *z -= dz * multiplier;
        } else {
            nrm = 1.0 / nrm;
            dz = ratio(c, z.conj() * nrm, false);
            *z /= C::new(1.0, 0.0) - *z * dz * multiplier;
        }
        error2 = dz.norm_sqr();
        if !(error2 > max_eps * max_eps * nrm && iterations < 3000) {
            break;
        }
    }
    if error2 > min_eps * min_eps * z.norm_sqr().min(1.0 / z.norm_sqr()) {
        let mut previous = error2;
        for _ in 0..=300 {
            let nrm = z.norm_sqr();
            let dz = if inner {
                let d = ratio(c, *z, true);
                *z -= d;
                d
            } else {
                let d = ratio(c, z.conj() / nrm, false);
                *z /= C::new(1.0, 0.0) - *z * d;
                d
            };
            let next = dz.norm_sqr();
            iterations += 1;
            if !(previous - next > EPS * next && next > min_eps * min_eps * nrm) {
                break;
            }
            previous = next;
        }
    }
    iterations
}

fn solve_quartic(c: &[C], roots: &mut [C]) {
    let r = 4.0 * c[4];
    let p = 2.0 * r * c[2] - 3.0 * c[3] * c[3];
    let q = 2.0 * c[3] * (c[3] * c[3] - r * c[2]) + r * r * c[1];
    let d0 = c[2] * c[2] - 3.0 * c[3] * c[1] + 3.0 * c[0] * r;
    let d1 = 2.0 * c[2] * c[2] * c[2] - 9.0 * c[3] * c[2] * c[1]
        + 27.0 * c[3] * c[3] * c[0]
        + 27.0 * c[1] * c[1] * c[4]
        - 18.0 * r * c[2] * c[0];
    let discriminant = (d1 * d1 - 4.0 * d0 * d0 * d0).sqrt();
    let qc = if d1.re * discriminant.re + d1.im * discriminant.im >= 0.0 {
        (d1 + discriminant) / 2.0
    } else {
        (d1 - discriminant) / 2.0
    };
    let q_abs = qc.norm().cbrt();
    let q_arg = qc.arg() / 3.0;
    let qs = [
        C::from_polar(q_abs, q_arg),
        C::from_polar(q_abs, q_arg + TWO_PI / 3.0),
        C::from_polar(q_abs, q_arg - TWO_PI / 3.0),
    ];
    let values = qs.map(|x| (x + d0 / x) * r - p);
    let selected = values
        .iter()
        .max_by(|a, b| a.norm_sqr().total_cmp(&b.norm_sqr()))
        .copied()
        .unwrap();
    let root = (selected / 3.0).sqrt();
    let s = if c[4].re >= 0.0 { root } else { -root };
    let t = (q / s - s * s - p).sqrt();
    roots[0] = (-c[3] - s + t) / r;
    roots[1] = (-c[3] - s - t) / r;
    let t = (-q / s - s * s - p).sqrt();
    roots[2] = (-c[3] + s + t) / r;
    roots[3] = (-c[3] + s - t) / r;
    for root in roots.iter_mut() {
        if root.norm_sqr() <= 1.0 {
            *root -= ratio(c, *root, true);
        } else {
            *root /= C::new(1.0, 0.0) - *root * ratio(c, inverse(*root), false);
        }
    }
}

fn polynomial_roots(
    mut c: [C; DEG + 1],
    roots: &mut [C; DEG],
    min_err: f64,
    max_err: f64,
    rng: &mut Rng,
) -> u64 {
    let mut start = 0_usize;
    let mut degree = DEG;
    let mut iterations = 0_u64;
    let mut i = 0_usize;
    while degree > 4 {
        iterations += newton(
            &c[start..=start + degree],
            &mut roots[i],
            max_err,
            min_err,
            rng,
        );
        let root = roots[i];
        if root.norm_sqr() <= 1.0 {
            for k in (1..degree).rev() {
                let next = c[start + k + 1];
                c[start + k] += next * root;
            }
            start += 1;
        } else {
            let inverse_root = inverse(root);
            for k in 1..degree {
                let previous = c[start + k - 1];
                c[start + k] += previous * inverse_root;
            }
        }
        degree -= 1;
        if degree > 4 && roots[i + 1] == C::new(0.0, 0.0) {
            roots[i + 1] = roots[i] / roots[i].norm_sqr();
        }
        i += 1;
    }
    solve_quartic(&c[start..=start + 4], &mut roots[DEG - 4..]);
    iterations
}

fn root_error(c: &[C; DEG + 1], coefficient_error: f64, z: C) -> f64 {
    let mut magnitude = z.norm_sqr();
    let forward = magnitude <= 1.0;
    if !forward {
        magnitude = 1.0 / magnitude;
    }
    let mut value_error2 = 1.0;
    for _ in 1..=DEG {
        value_error2 = value_error2 * magnitude + 1.0;
    }
    value_error2 *= coefficient_error * coefficient_error;
    let (f, fd, fd2) = polynomial_valder2(c, if forward { z } else { inverse(z) }, forward);
    let discriminant = (fd * fd - 2.0 * f * fd2).sqrt();
    let d2 = 4.0 * f.norm_sqr()
        / (fd + discriminant)
            .norm_sqr()
            .max((fd - discriminant).norm_sqr());
    ((d2 + value_error2 / discriminant.norm_sqr()) / magnitude).sqrt()
}

fn max_root_bound(c: &[C; DEG + 1]) -> f64 {
    let leading = c[DEG].norm_sqr();
    let mut result = c[DEG - 1].norm_sqr() / leading;
    for i in 2..=DEG {
        let candidate = (c[DEG - i].norm_sqr() / leading).powf(1.0 / i as f64);
        result = result.max(candidate);
    }
    result.sqrt()
}

#[inline]
fn g_fast(index: usize, data: &Aux) -> f64 {
    let z = roots_of_unity()[index];
    let x = z.re;
    let y = z.im;
    let a = data.ps * y - data.ss * x;
    let b = data.pp * y - data.sp * x;
    let c = data.e2 * b - data.alpha1 * data.e1 * y * (1.0 - data.e1 * x);
    let shifted_x = x - data.e1;
    let m = data.sp * y + data.pp * shifted_x + data.alpha2 * data.e2;
    let n = -data.ss * y - data.ps * shifted_x;
    let c2 = c * c;
    let a2 = a * a;
    let b2 = b * b;
    let ac = a2 - c2;
    let bc = b2 - c2;
    data.k * data.k * ac * bc + 2.0 * data.k * c * (n * a * ac + m * b * bc)
        - (a2 + b2) * (n * n * ac + m * m * bc - 2.0 * n * m * a * b)
}

fn create_polynomial(data: &Aux) -> ([C; DEG + 1], f64) {
    let factor = sqr(data.alpha1 * sqr(data.e1) / 16.0);
    let hc = factor
        * C::new(data.pp - data.ss - data.e1 * data.e2, data.sp + data.ps)
        * C::new(data.pp - data.ss + data.e1 * data.e2, data.sp + data.ps)
        * C::new(data.pp + data.ss - data.e1 * data.e2, data.sp - data.ps)
        * C::new(data.pp + data.ss + data.e1 * data.e2, data.sp - data.ps);
    let mut values = std::array::from_fn::<_, DIM, _>(|j| g_fast(j, data));
    for j in 1..=DIM / 2 {
        let sum = values[j] + values[DIM - j];
        let difference = values[j] - values[DIM - j];
        values[j] = sum;
        values[DIM - j] = difference;
    }
    let exps = roots_of_unity();
    let mut coefficients = [C::new(0.0, 0.0); DEG + 1];
    coefficients[DEG / 2] = C::new(values[0..=DIM / 2].iter().sum::<f64>() / DIM as f64, 0.0);
    let mut i = 1_usize;
    while i < DEG / 2 {
        let mut a = values[0];
        let mut b = 0.0;
        for j in 1..=DIM / 2 {
            a += values[j] * exps[(j * i) % DIM].re;
            b += values[DIM - j] * exps[(j * i) % DIM].im;
        }
        let value = C::new(a / DIM as f64, b / DIM as f64);
        coefficients[DEG / 2 - i] = value;
        coefficients[DEG / 2 + i] = value.conj();
        i += 1;
    }
    coefficients[0] = hc;
    coefficients[DEG] = hc.conj();
    let mut a = values[0];
    let mut b = 0.0;
    for j in 1..=DIM / 2 {
        a += values[j] * exps[(j * i) % DIM].re;
        b += values[DIM - j] * exps[(j * i) % DIM].im;
    }
    let mut coefficient_error = (C::new(a / DIM as f64, b / DIM as f64) - hc).norm_sqr();
    i += 1;
    while i <= DIM / 2 {
        let mut a = values[0];
        let mut b = 0.0;
        for j in 1..=DIM / 2 {
            a += values[j] * exps[(j * i) % DIM].re;
            b += values[DIM - j] * exps[(j * i) % DIM].im;
        }
        a /= DIM as f64;
        b /= DIM as f64;
        coefficient_error += a * a + b * b;
        i += 1;
    }
    coefficient_error /= (DIM - DEG).div_ceil(2) as f64;
    (coefficients, coefficient_error.sqrt().max(EPS * hc.norm()))
}

#[inline]
fn radius_vector(p: [f64; 3], q: [f64; 3], a: f64, e: f64, u: f64) -> [f64; 3] {
    let (sin_u, cos_u) = u.sin_cos();
    let x = cos_u - e;
    let y = sin_u * (1.0 - e * e).sqrt();
    [
        a * (p[0] * x + q[0] * y),
        a * (p[1] * x + q[1] * y),
        a * (p[2] * x + q[2] * y),
    ]
}

#[inline]
fn radius_valder2(
    p: [f64; 3],
    q: [f64; 3],
    a: f64,
    e: f64,
    u: f64,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let (sin_u, cos_u) = u.sin_cos();
    let eta = (1.0 - e * e).sqrt();
    let radius = radius_vector(p, q, a, e, u);
    let derivative = [
        -p[0] * sin_u + q[0] * cos_u * eta,
        -p[1] * sin_u + q[1] * cos_u * eta,
        -p[2] * sin_u + q[2] * cos_u * eta,
    ];
    let second = [
        -p[0] * cos_u - q[0] * sin_u * eta,
        -p[1] * cos_u - q[1] * sin_u * eta,
        -p[2] * cos_u - q[2] * sin_u * eta,
    ];
    (radius, derivative, second)
}

fn distance_between(data: &Aux, u1: f64, u2: f64) -> f64 {
    let r1 = radius_vector(data.basis_p1, data.basis_q1, data.a1, data.e1, u1);
    let r2 = radius_vector(data.basis_p2, data.basis_q2, data.a2, data.e2, u2);
    norm([r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]]).sqrt()
}

fn sqdist_vgh(data: &Aux, u1: f64, u2: f64) -> (f64, [f64; 2], [f64; 3]) {
    let (r1, rd1, rdd1) = radius_valder2(data.basis_p1, data.basis_q1, data.a1, data.e1, u1);
    let (r2, rd2, rdd2) = radius_valder2(data.basis_p2, data.basis_q2, data.a2, data.e2, u2);
    let dr = [r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]];
    let gradient = [-dot(dr, rd1) / data.a2, dot(dr, rd2) / data.a1];
    let hessian = [
        (-dot(dr, rdd1) + norm(rd1) * data.a1) / data.a2,
        (dot(dr, rdd2) + norm(rd2) * data.a2) / data.a1,
        -dot(rd1, rd2),
    ];
    (norm(dr) / (2.0 * data.a1 * data.a2), gradient, hessian)
}

fn sqdist_vg(data: &Aux, u1: f64, u2: f64) -> (f64, [f64; 2]) {
    let (r1, rd1, _) = radius_valder2(data.basis_p1, data.basis_q1, data.a1, data.e1, u1);
    let (r2, rd2, _) = radius_valder2(data.basis_p2, data.basis_q2, data.a2, data.e2, u2);
    let dr = [r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]];
    (
        norm(dr) / (2.0 * data.a1 * data.a2),
        [-dot(dr, rd1) / data.a2, dot(dr, rd2) / data.a1],
    )
}

fn sqdist_iteration(data: &Aux, u1: &mut f64, u2: &mut f64) -> (f64, [f64; 2], [f64; 3], f64, f64) {
    let (rho, gradient, hessian) = sqdist_vgh(data, *u1, *u2);
    let determinant = hessian[0] * hessian[1] - hessian[2] * hessian[2];
    if determinant == 0.0 {
        return (rho, gradient, hessian, determinant, 0.0);
    }
    let du1 = -(hessian[1] * gradient[0] - hessian[2] * gradient[1]) / determinant;
    let du2 = -(hessian[0] * gradient[1] - hessian[2] * gradient[0]) / determinant;
    *u1 += du1;
    *u2 += du2;
    (rho, gradient, hessian, determinant, du1 * du1 + du2 * du2)
}

fn eliminated_anomaly(data: &Aux, u1: f64, mut error: Option<&mut f64>) -> (bool, f64, f64) {
    let (y, x) = u1.sin_cos();
    let shifted_x = x - data.e1;
    let m = data.sp * y + data.pp * shifted_x + data.alpha2 * data.e2;
    let n = -data.ss * y - data.ps * shifted_x;
    let a = data.ps * y - data.ss * x;
    let b = data.pp * y - data.sp * x;
    let c = data.e2 * b - data.alpha1 * data.e1 * y * (1.0 - data.e1 * x);
    let v = a * a + b * b;
    let mut discriminant = v - c * c;
    let negative = discriminant < 0.0;
    let (mut u2, mut other);
    if negative {
        discriminant = 0.0;
        u2 = if c > 0.0 { a.atan2(b) } else { (-a).atan2(-b) };
        other = u2;
    } else {
        discriminant = discriminant.sqrt();
        let y1 = (a * c - b * discriminant) / v;
        let y2 = (a * c + b * discriminant) / v;
        let x1 = (b * c + a * discriminant) / v;
        let x2 = (b * c - a * discriminant) / v;
        u2 = y1.atan2(x1);
        other = y2.atan2(x2);
        let err1 = m * y1 + n * x1 - data.k * y1 * x1;
        let err2 = m * y2 + n * x2 - data.k * y2 * x2;
        if err1.abs().partial_cmp(&err2.abs()) != Some(std::cmp::Ordering::Less) {
            std::mem::swap(&mut u2, &mut other);
        }
    }
    if let Some(value) = error.as_mut() {
        let sigma = **value * (1.0 + sqr(data.e2) + sqr(data.alpha1 * data.e1)).sqrt();
        **value = sigma / (discriminant * discriminant + c.abs() * sigma / 2.0).sqrt();
    }
    (negative, u2, other)
}

fn min_distance_for(data: &Aux, u1: f64, error: Option<&mut f64>) -> (f64, f64) {
    let (negative, u2, other) = eliminated_anomaly(data, u1, error);
    if negative {
        return (-distance_between(data, u1, u2), u2);
    }
    let d1 = distance_between(data, u1, u2);
    let d2 = distance_between(data, u1, other);
    if d1 < d2 {
        (d1, u2)
    } else {
        (d2, other)
    }
}

fn refine_distance(
    data: &Aux,
    u1: &mut f64,
    u2: &mut f64,
    eps: f64,
) -> (i16, f64, f64, f64, f64, [f64; 3]) {
    let mut previous_error = f64::INFINITY;
    let mut hessian = [0.0; 3];
    let mut determinant = 0.0;
    for _ in 0..=30 {
        let (_, _, h, det, error) = sqdist_iteration(data, u1, u2);
        hessian = h;
        determinant = det;
        if u1.abs() > PI {
            *u1 = angle_wrap(*u1);
        }
        if u2.abs() > PI {
            *u2 = angle_wrap(*u2);
        }
        if det == 0.0 || !(previous_error - error > EPS * error && error > eps * eps) {
            break;
        }
        previous_error = error;
    }
    let (rho, gradient) = sqdist_vg(data, *u1, *u2);
    if determinant == 0.0 {
        let sign = if hessian[0] > 0.0 || hessian[1] > 0.0 {
            1
        } else if hessian[0] < 0.0 || hessian[1] < 0.0 {
            -1
        } else {
            0
        };
        return (sign, rho, 0.0, 0.0, 0.0, hessian);
    }
    let u1_error = ((hessian[1] * gradient[0] - hessian[2] * gradient[1]) / determinant).abs();
    let u2_error = ((hessian[0] * gradient[1] - hessian[2] * gradient[0]) / determinant).abs();
    let rho_error = ((hessian[1] * sqr(gradient[0]) + hessian[0] * sqr(gradient[1])
        - 2.0 * hessian[2] * gradient[0] * gradient[1])
        / (2.0 * determinant))
        .abs();
    let sign = if determinant > 0.0 {
        if hessian[0] > 0.0 {
            2
        } else if hessian[0] < 0.0 {
            -2
        } else {
            0
        }
    } else {
        0
    };
    (sign, rho, rho_error, u1_error, u2_error, hessian)
}

fn fast(o1: Orbit, o2: Orbit, seed: u64) -> Result {
    let data = Aux::new(o1, o2);
    let max_root_error = EPS.sqrt();
    let min_root_error = 2.0 * EPS;
    let (mut coefficients, mut coefficient_error) = create_polynomial(&data);
    let znorm = 1.0;
    let mut scale = 1.0;
    for i in 1..=DEG / 2 {
        scale *= znorm;
        coefficients[DEG / 2 + i] *= scale;
        coefficients[DEG / 2 - i] /= scale;
    }
    coefficient_error *= scale.max(1.0 / scale);
    let original = coefficients;
    let mut roots = [C::new(0.0, 0.0); DEG];
    let mut rng = Rng::new(seed);
    let max_abs_root = max_root_bound(&coefficients) + EPS;
    let min_abs_root = 1.0 / max_abs_root;
    roots[0] = C::from_polar(min_abs_root, rng.next() * TWO_PI);
    let eta1 = (1.0 - data.e1 * data.e1).sqrt();
    let eta2 = (1.0 - data.e2 * data.e2).sqrt();
    let wpq1 = data.p1w.hypot(data.q1w);
    let wpq2 = data.p2w.hypot(data.q2w);
    let u1 = (data.q1w * eta1).atan2(data.e1 * wpq1 + data.p1w);
    let u2 = (data.q2w * eta2).atan2(data.e2 * wpq2 + data.p2w);
    let u1_desc = (-data.q1w * eta1).atan2(data.e1 * wpq1 - data.p1w);
    let u2_desc = (-data.q2w * eta2).atan2(data.e2 * wpq2 - data.p2w);
    let mut index = 4_usize;
    let mut add_seed = |mut a: f64, mut b: f64| {
        let _ = sqdist_iteration(&data, &mut a, &mut b);
        roots[index] = C::from_polar(1.0, a);
        index += 1;
    };
    add_seed(u1, u2);
    add_seed(u1_desc, u2_desc);
    let plus_u1 = (-data.p1w * eta1).atan2(data.e1 * wpq1 + data.q1w);
    let mut plus_u2 = (-data.p2w * eta2).atan2(data.e2 * wpq2 + data.q2w);
    let mut minus_u2 = (data.p2w * eta2).atan2(data.e2 * wpq2 - data.q2w);
    if 2.0 * data.mutual_i > PI {
        std::mem::swap(&mut plus_u2, &mut minus_u2);
    }
    add_seed(plus_u1, plus_u2);
    let minus_u1 = (data.p1w * eta1).atan2(data.e1 * wpq1 - data.q1w);
    add_seed(minus_u1, minus_u2);
    roots[index] = C::from_polar(min_abs_root, rng.next() * TWO_PI);
    let _iterations = polynomial_roots(
        coefficients,
        &mut roots,
        min_root_error,
        max_root_error,
        &mut rng,
    );
    let mut result = Result::default();
    for (i, root) in roots.iter().copied().enumerate() {
        let error = root_error(&original, coefficient_error, root);
        let delta = root.norm_sqr().ln().abs() / (2.0 * error);
        if delta < 10.0 {
            result.good &= error < max_root_error;
            if delta < 3.0 {
                result.root_count += 1;
                let candidate_u1 = root.arg();
                let (distance, candidate_u2) = min_distance_for(&data, candidate_u1, None);
                let distance = distance.abs();
                if distance >= 0.0 && (result.distance < 0.0 || distance < result.distance) {
                    result.distance = distance;
                    result.u1 = candidate_u1;
                    result.u2 = candidate_u2;
                    let _ = i;
                }
            }
        }
        if delta >= 3.0 && (result.min_delta < 0.0 || delta < result.min_delta) {
            result.min_delta = delta;
        }
    }
    if result.distance < 0.0 || !result.distance.is_finite() {
        result.good = false;
        return result;
    }
    let old_u1 = result.u1;
    let old_u2 = result.u2;
    let (hessian_sign, rho, rho_error, u1_error, u2_error, hessian) =
        refine_distance(&data, &mut result.u1, &mut result.u2, min_root_error);
    result.u1_error = u1_error;
    result.u2_error = u2_error;
    result.distance = rho;
    result.distance_error = rho_error;
    let mut du1 = old_u1 - result.u1;
    let mut du2 = old_u2 - result.u2;
    if du1.abs() > PI {
        du1 = angle_wrap(du1);
    }
    if du2.abs() > PI {
        du2 = angle_wrap(du2);
    }
    let _du = du1.hypot(du2);
    let angle_error = EPS * PI;
    result.u1_error += angle_error;
    result.u2_error += angle_error;
    let factor = 2.0 * data.a1 * data.a2;
    let x1 = result.u1.cos();
    let x2 = result.u2.cos();
    let t1 = data.e1 * x1;
    let t2 = data.e2 * x2;
    let r1 = data.a1 * (1.0 - t1);
    let r2 = data.a2 * (1.0 - t2);
    let rd1 = data.a1.abs() * (1.0 - t1 * t1).abs().sqrt();
    let rd2 = data.a2.abs() * (1.0 - t2 * t2).abs().sqrt();
    let radial_error = r1.hypot(r2) * EPS;
    let gradient_error = radial_error * rd1.hypot(rd2) / factor.abs();
    let eigenvalue_max = ((hessian[0] + hessian[1]).abs() / 2.0)
        + ((hessian[0] - hessian[1]) / 2.0).hypot(hessian[2]);
    let determinant = (hessian[0] * hessian[1] - hessian[2] * hessian[2]).abs();
    let uerr3 = eigenvalue_max / determinant * gradient_error;
    result.u1_error += uerr3;
    result.u2_error += uerr3;
    result.distance_error += eigenvalue_max / 2.0
        * (angle_error * angle_error + gradient_error * gradient_error / determinant);
    result.distance *= factor;
    result.distance_error *= factor.abs();
    result.distance = result.distance.sqrt();
    result.distance_error += 2.0 * result.distance * radial_error + radial_error * radial_error;
    result.distance_error /=
        (result.distance * result.distance + result.distance_error / 2.0).sqrt();
    result.good = result.good
        && result.root_count >= 4
        && result.root_count % 2 == 0
        && result.min_delta > 10.0
        && hessian_sign == 2
        && result.u1_error.abs() <= max_root_error;
    result
}

fn restrict_search_range(data: &Aux) -> Vec<(f64, f64)> {
    let theta1 = data.q1w.atan2(data.p1w);
    let theta2 = data.q2w.atan2(data.p2w);
    let cos1 = theta1.cos();
    let cos2 = theta2.cos();
    let rp = data.p1 / (1.0 + data.e1 * cos1);
    let rm = data.p1 / (1.0 - data.e1 * cos1);
    let big_rp = data.p2 / (1.0 + data.e2 * cos2);
    let big_rm = data.p2 / (1.0 - data.e2 * cos2);
    let distances = [
        (rp - big_rp).abs(),
        (rp + big_rm).abs(),
        (rm + big_rp).abs(),
        (rm - big_rm).abs(),
    ];
    let flags = [
        rp > 0.0 && big_rp > 0.0,
        rp > 0.0 && big_rm > 0.0,
        rm > 0.0 && big_rp > 0.0,
        rm > 0.0 && big_rm > 0.0,
    ];
    let d0 = distances
        .iter()
        .zip(flags)
        .filter(|(_, valid)| *valid)
        .map(|(d, _)| *d)
        .fold(f64::INFINITY, f64::min);
    let a = (1.0 - sqr(data.e1 * cos1)).abs().sqrt();
    let k = d0 / (a * data.abs_w * data.a1.abs());
    let sin1 = theta1.sin();
    let eta1 = (1.0 - data.e1 * data.e1).sqrt();
    let mut phi = sin1.atan2(cos1 * eta1);
    let esth = data.e1 * sin1 / a;
    if esth.abs().min(1.0) <= (1.0 - k).abs() {
        if k < 1.0 {
            let minus = (esth - k).clamp(-1.0, 1.0).asin();
            let plus = (esth + k).clamp(-1.0, 1.0).asin();
            let first = (phi - plus, phi - minus);
            phi += PI;
            return vec![first, (phi + minus, phi + plus)];
        }
        return vec![(-PI, PI)];
    }
    if esth > 0.0 {
        let minus = (esth - k).clamp(-1.0, 1.0).asin();
        let mut range = (phi + minus, phi + PI - minus);
        if range.0 > range.1 {
            range = (phi + PI / 2.0, phi + PI / 2.0);
        }
        vec![range]
    } else {
        let plus = (esth + k).clamp(-1.0, 1.0).asin();
        let mut range = (phi - plus, phi + PI + plus);
        if range.0 > range.1 {
            range = (phi - PI / 2.0, phi - PI / 2.0);
        }
        vec![range]
    }
}

fn search_in_segment(data: &Aux, a: f64, b: f64, count: usize) -> (usize, f64) {
    let mut best_index = 0;
    let mut best = -1.0;
    for i in 0..=count {
        let u1 = (a * (count - i) as f64 + b * i as f64) / count as f64;
        let (distance, _) = min_distance_for(data, u1, None);
        if distance >= 0.0 && (best < 0.0 || distance < best) {
            best = distance;
            best_index = i;
        }
    }
    (best_index, best)
}

fn direct_search(o1: Orbit, o2: Orbit) -> Result {
    let mut data = Aux::new(o1, o2);
    let mut ranges = restrict_search_range(&data);
    let reverse = Aux::new(o2, o1);
    let reverse_ranges = restrict_search_range(&reverse);
    let length = ranges.iter().map(|(a, b)| (b - a).max(0.0)).sum::<f64>();
    let reverse_length = reverse_ranges
        .iter()
        .map(|(a, b)| (b - a).max(0.0))
        .sum::<f64>();
    let swapped = reverse_length < length;
    if swapped {
        data = reverse;
        ranges = reverse_ranges;
    }
    let mut candidates = Vec::with_capacity(ranges.len());
    for (mut a, mut b) in ranges {
        let mut half_width = (b - a) / 2.0;
        if half_width > 0.0 {
            let mut previous = 0.0;
            let mut delta: f64 = 0.0;
            let densities = [1000_usize, 30, 4, 0];
            let mut density = 3_usize;
            let mut use_densities = true;
            let mut iteration = 0_usize;
            while iteration < 1 || half_width > 1e-12 || delta.abs() > 1e-12 {
                if use_densities {
                    let configured = densities.get(iteration).copied().unwrap_or(0);
                    if configured != 0 {
                        density = if iteration == 0 {
                            (half_width * configured as f64 / PI).ceil() as usize
                        } else {
                            configured
                        };
                    } else {
                        use_densities = false;
                    }
                }
                density = density.max(3);
                half_width = 2.0 * half_width / density as f64;
                let (best_index, value) = search_in_segment(&data, a, b, density);
                a += half_width * best_index as f64;
                b = a + half_width;
                a -= half_width;
                if iteration > 0 {
                    delta = value - previous;
                }
                previous = value;
                iteration += 1;
                if iteration > 100 {
                    break;
                }
            }
        }
        let u1 = (a + b) / 2.0;
        let (distance, u2) = min_distance_for(&data, u1, None);
        candidates.push(Result {
            u1,
            u2,
            u1_error: half_width.abs(),
            distance: distance.abs(),
            ..Result::default()
        });
    }
    let mut result = candidates
        .into_iter()
        .min_by(|a, b| a.distance.total_cmp(&b.distance))
        .unwrap_or_default();
    let perturbation = EPS.sqrt();
    let (distance, u2) = min_distance_for(&data, result.u1 + perturbation, None);
    result.distance_error =
        (distance.abs() - result.distance).abs() * sqr(result.u1_error / perturbation);
    result.u2_error = (u2 - result.u2).abs() * result.u1_error / perturbation;
    if swapped {
        std::mem::swap(&mut result.u1, &mut result.u2);
        std::mem::swap(&mut result.u1_error, &mut result.u2_error);
    }
    result.good &= result.distance_error < 1e-12;
    result
}

pub(crate) fn solve(first: Orbit, second: Orbit, seed: u64) -> Result {
    let forward = fast(first, second, seed ^ 0xa076_1d64_78bd_642f);
    if forward.good
        && forward.distance.is_finite()
        && forward.u1.is_finite()
        && forward.u2.is_finite()
    {
        return forward;
    }
    let mut reverse = fast(second, first, seed ^ 0xe703_7ed1_a0b4_28db);
    if reverse.good
        && reverse.distance.is_finite()
        && reverse.u1.is_finite()
        && reverse.u2.is_finite()
    {
        std::mem::swap(&mut reverse.u1, &mut reverse.u2);
        std::mem::swap(&mut reverse.u1_error, &mut reverse.u2_error);
        return reverse;
    }
    direct_search(first, second)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_search_real_orientation_regression() {
        let first = Orbit::new(
            1.3873905893432101,
            0.27194969827994975,
            6.4372319666600175_f64.to_radians(),
            291.37818107796124_f64.to_radians(),
            314.3391686425961_f64.to_radians(),
        );
        let second = Orbit::new(
            1.0001874165116187,
            0.016619025003065148,
            0.005777978613270001_f64.to_radians(),
            304.0260276458503_f64.to_radians(),
            157.72026899495754_f64.to_radians(),
        );
        let result = direct_search(first, second);
        assert!((result.distance - 0.07224759404236622).abs() < 1e-8);
    }
}
