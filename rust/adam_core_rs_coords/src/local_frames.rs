//! Local orbital frames (SANA orbit-relative reference frame registry names)
//! for covariances: RSW (aliases RTN, RIC), TNW and VNC, each `_INERTIAL`
//! (pure rotation) or `_ROTATING` (velocity rows carry the two-body frame
//! rate). Jacobians come from forward-mode autodiff of the frame axes along
//! the two-body motion; the covariance product `J C J^T` is accumulated in
//! double-double arithmetic and rounded once.
//!
//! Interface fixed on 2026-10-08. Bodies are filled by the kernel task; the
//! OEM renderer calls these functions for its COV_REF_FRAME blocks.

use std::ops::{Add, Mul};

use adam_core_rs_autodiff::{Dual, Scalar};

use crate::types::{SchemaError, SchemaResult};

/// |r x v| (AU^2/day) below which a state has no orbit plane, as
/// `SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE` in `adam_core.coordinates.cartesian`.
const SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE: f64 = 1e-20;

/// The three axis families of the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalFrameFamily {
    /// x along position, z along the orbital angular momentum h.
    Rsw,
    /// x along velocity, z along h.
    Tnw,
    /// x along velocity, y along h.
    Vnc,
}

impl LocalFrameFamily {
    /// Index of the axis along the orbit normal h.
    fn normal_axis(self) -> usize {
        match self {
            Self::Rsw | Self::Tnw => 2,
            Self::Vnc => 1,
        }
    }
}

/// A local orbital frame: family plus whether velocity rows carry the frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalFrame {
    pub family: LocalFrameFamily,
    pub rotating: bool,
}

/// Every frame, in the order the canonical names are listed in errors.
const ALL_FRAMES: [LocalFrame; 6] = [
    LocalFrame {
        family: LocalFrameFamily::Rsw,
        rotating: true,
    },
    LocalFrame {
        family: LocalFrameFamily::Rsw,
        rotating: false,
    },
    LocalFrame {
        family: LocalFrameFamily::Tnw,
        rotating: true,
    },
    LocalFrame {
        family: LocalFrameFamily::Tnw,
        rotating: false,
    },
    LocalFrame {
        family: LocalFrameFamily::Vnc,
        rotating: true,
    },
    LocalFrame {
        family: LocalFrameFamily::Vnc,
        rotating: false,
    },
];

impl LocalFrame {
    /// Parse a registry name (`VNC_ROTATING`, `TNW_INERTIAL`, ...), an alias
    /// (`RTN`, `RIC` mean RSW) or a bare family name (meaning `_INERTIAL`),
    /// case-insensitively and ignoring surrounding whitespace.
    pub fn parse(name: &str) -> SchemaResult<Self> {
        let upper = name.trim().to_ascii_uppercase();
        // Aliases are bare names only, as in the Python parser.
        let canonical = match upper.as_str() {
            "RSW" | "RTN" | "RIC" => "RSW_INERTIAL",
            "TNW" => "TNW_INERTIAL",
            "VNC" => "VNC_INERTIAL",
            other => other,
        };
        ALL_FRAMES
            .into_iter()
            .find(|frame| frame.canonical_name() == canonical)
            .ok_or_else(|| {
                SchemaError::InvalidRecordBatch(format!(
                    "Unknown local orbital frame '{name}', expected one of {:?}",
                    ALL_FRAMES.map(|frame| frame.canonical_name())
                ))
            })
    }

    /// The canonical registry name, e.g. `VNC_ROTATING`.
    pub fn canonical_name(&self) -> &'static str {
        match (self.family, self.rotating) {
            (LocalFrameFamily::Rsw, true) => "RSW_ROTATING",
            (LocalFrameFamily::Rsw, false) => "RSW_INERTIAL",
            (LocalFrameFamily::Tnw, true) => "TNW_ROTATING",
            (LocalFrameFamily::Tnw, false) => "TNW_INERTIAL",
            (LocalFrameFamily::Vnc, true) => "VNC_ROTATING",
            (LocalFrameFamily::Vnc, false) => "VNC_INERTIAL",
        }
    }
}

/// Jacobians from inertial position and velocity to `frame`, one 6x6 per
/// state, returned row-major as `N * 36` values. `values` is `N * 6`
/// (x, y, z, vx, vy, vz in AU and AU/day), `mu` has one entry per state
/// (AU^3/day^2) and is only read for `_ROTATING` frames.
pub fn local_frame_jacobians(
    values: &[f64],
    mu: &[f64],
    frame: LocalFrame,
) -> SchemaResult<Vec<f64>> {
    let n = state_count(values, mu, frame)?;
    let mut out = Vec::with_capacity(n * 36);
    for (index, state) in values.chunks_exact(6).enumerate() {
        out.extend_from_slice(&state_jacobian(
            state,
            mu_of(mu, index, frame),
            frame,
            index,
        )?);
    }
    Ok(out)
}

/// `J C J^T` for every state, `covariances` and the result row-major `N * 36`
/// (AU, AU/day units). The input covariance is symmetrised exactly, the
/// product is accumulated in double-double arithmetic and rounded once, and
/// the result is symmetric to the bit. Rows whose covariance is all NaN stay NaN.
pub fn local_frame_covariances(
    values: &[f64],
    covariances: &[f64],
    mu: &[f64],
    frame: LocalFrame,
) -> SchemaResult<Vec<f64>> {
    let n = state_count(values, mu, frame)?;
    if covariances.len() != n * 36 {
        return Err(SchemaError::LengthMismatch {
            field: "covariances (36 entries per state)".to_string(),
            expected: n * 36,
            actual: covariances.len(),
        });
    }
    let mut out = vec![f64::NAN; n * 36];
    for (index, ((state, covariance), rotated)) in values
        .chunks_exact(6)
        .zip(covariances.chunks_exact(36))
        .zip(out.chunks_exact_mut(36))
        .enumerate()
    {
        // Nothing to rotate, so no Jacobian is built and the row stays NaN.
        if covariance.iter().all(|x| x.is_nan()) {
            continue;
        }
        // f64 entries are exact double-doubles, so each J_ik J_jl is exact.
        let jacobian = state_jacobian(state, mu_of(mu, index, frame), frame, index)?
            .map(|x| DoubleDouble::new((x, 0.0)));
        rotated.copy_from_slice(&rotate_covariance(&jacobian, covariance));
    }
    Ok(out)
}

/// Number of states, after checking `values` and (for rotating frames) `mu`.
fn state_count(values: &[f64], mu: &[f64], frame: LocalFrame) -> SchemaResult<usize> {
    if values.len() % 6 != 0 {
        return Err(SchemaError::InvalidRecordBatch(format!(
            "values must hold 6 entries per state (x, y, z, vx, vy, vz), got {}",
            values.len()
        )));
    }
    let n = values.len() / 6;
    if frame.rotating && mu.len() != n {
        return Err(SchemaError::LengthMismatch {
            field: format!("mu (one entry per state for {})", frame.canonical_name()),
            expected: n,
            actual: mu.len(),
        });
    }
    Ok(n)
}

fn mu_of(mu: &[f64], index: usize, frame: LocalFrame) -> f64 {
    if frame.rotating {
        mu[index]
    } else {
        0.0
    }
}

/// Row-major 6x6 Jacobian `[[R, 0], [dR/dt, R]]` of one state.
fn state_jacobian(
    state: &[f64],
    mu: f64,
    frame: LocalFrame,
    index: usize,
) -> SchemaResult<[f64; 36]> {
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let h = cross(r, v);
    if dot(h, h).sqrt() < SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE {
        return Err(SchemaError::InvalidRecordBatch(format!(
            "State {index} has no orbit plane."
        )));
    }
    let (rotation, rate) = if frame.rotating {
        // Forward mode in time along the two-body motion: r(t) = r + v t and
        // v(t) = v + a t, so each axis carries its time derivative as tangent.
        let r_mag = dot(r, r).sqrt();
        let a = r.map(|x| -mu * x / (r_mag * r_mag * r_mag));
        let t = Dual::<1>::variable(0.0, 0);
        let r_t: [Dual<1>; 3] =
            std::array::from_fn(|k| Dual::constant(r[k]) + Dual::constant(v[k]) * t);
        let v_t: [Dual<1>; 3] =
            std::array::from_fn(|k| Dual::constant(v[k]) + Dual::constant(a[k]) * t);
        let axes = triad(r_t, v_t, frame.family);
        let mut rate = axes.map(|axis| axis.map(|x| x.du[0]));
        // The frame turns about h under two-body motion, so the h row is exactly zero.
        rate[frame.family.normal_axis()] = [0.0; 3];
        (axes.map(|axis| axis.map(|x| x.re)), rate)
    } else {
        (triad(r, v, frame.family), [[0.0; 3]; 3])
    };
    let mut jacobian = [0.0; 36];
    for i in 0..3 {
        for j in 0..3 {
            jacobian[i * 6 + j] = rotation[i][j];
            jacobian[(i + 3) * 6 + j + 3] = rotation[i][j];
            jacobian[(i + 3) * 6 + j] = rate[i][j];
        }
    }
    Ok(jacobian)
}

/// The frame axes, i.e. the rows of the rotation from inertial to local.
fn triad<T: Scalar>(r: [T; 3], v: [T; 3], family: LocalFrameFamily) -> [[T; 3]; 3] {
    let h_hat = unit(cross(r, v));
    match family {
        LocalFrameFamily::Rsw => {
            let r_hat = unit(r);
            [r_hat, cross(h_hat, r_hat), h_hat]
        }
        LocalFrameFamily::Tnw => {
            let v_hat = unit(v);
            [v_hat, cross(h_hat, v_hat), h_hat]
        }
        LocalFrameFamily::Vnc => {
            let v_hat = unit(v);
            [v_hat, h_hat, cross(v_hat, h_hat)]
        }
    }
}

fn dot<T: Scalar>(a: [T; 3], b: [T; 3]) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross<T: Scalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit<T: Scalar>(a: [T; 3]) -> [T; 3] {
    let norm = dot(a, a).sqrt();
    a.map(|x| x / norm)
}

/// `hi + lo` with `|lo| <= ulp(hi) / 2`, about 32 significant digits.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

/// Exact `a + b = s + e` (Knuth).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Exact `a + b = s + e` when `|a| >= |b|`.
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Exact `a * b = p + e`.
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl DoubleDouble {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    fn new((hi, lo): (f64, f64)) -> Self {
        Self { hi, lo }
    }
}

impl Add for DoubleDouble {
    type Output = Self;
    /// Accurate sum: relative error near 2^-104 of the result, also under cancellation.
    fn add(self, other: Self) -> Self {
        let (s, e) = two_sum(self.hi, other.hi);
        let (t, f) = two_sum(self.lo, other.lo);
        let (s, e) = quick_two_sum(s, e + t);
        Self::new(quick_two_sum(s, e + f))
    }
}

impl Mul for DoubleDouble {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let (p, e) = two_prod(self.hi, other.hi);
        Self::new(quick_two_sum(
            p,
            e + (self.hi * other.lo + self.lo * other.hi),
        ))
    }
}

/// `J C J^T` of one state. `C` is symmetrised exactly, each upper element is
/// the double-double sum of its 36 products rounded once, and the lower
/// triangle mirrors the upper one.
fn rotate_covariance(jacobian: &[DoubleDouble; 36], covariance: &[f64]) -> [f64; 36] {
    // (C + C^T) / 2 is exact in double-double, halving being exact.
    let mut symmetric = [DoubleDouble::ZERO; 36];
    for k in 0..6 {
        for l in 0..6 {
            let (s, e) = two_sum(covariance[k * 6 + l], covariance[l * 6 + k]);
            symmetric[k * 6 + l] = DoubleDouble::new((0.5 * s, 0.5 * e));
        }
    }
    let mut out = [0.0; 36];
    for i in 0..6 {
        for j in i..6 {
            let mut sum = DoubleDouble::ZERO;
            for k in 0..6 {
                for l in 0..6 {
                    sum = sum + jacobian[i * 6 + k] * jacobian[j * 6 + l] * symmetric[k * 6 + l];
                }
            }
            out[i * 6 + j] = sum.hi + sum.lo;
            out[j * 6 + i] = out[i * 6 + j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU_SUN: f64 = 0.000_295_912_208_284_119_56;
    const FAMILIES: [LocalFrameFamily; 3] = [
        LocalFrameFamily::Rsw,
        LocalFrameFamily::Tnw,
        LocalFrameFamily::Vnc,
    ];

    fn frame(family: LocalFrameFamily, rotating: bool) -> LocalFrame {
        LocalFrame { family, rotating }
    }

    /// Heliocentric state from elements (angles in degrees).
    fn state_from_elements(a: f64, e: f64, i: f64, node: f64, peri: f64, nu: f64) -> [f64; 6] {
        let (i, node, peri, nu) = (
            i.to_radians(),
            node.to_radians(),
            peri.to_radians(),
            nu.to_radians(),
        );
        let p = a * (1.0 - e * e);
        let r = p / (1.0 + e * nu.cos());
        let s = (MU_SUN / p).sqrt();
        let pos = [r * nu.cos(), r * nu.sin(), 0.0];
        let vel = [-s * nu.sin(), s * (e + nu.cos()), 0.0];
        let (so, co) = node.sin_cos();
        let (si, ci) = i.sin_cos();
        let (sw, cw) = peri.sin_cos();
        let m = [
            [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
            [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
            [sw * si, cw * si, ci],
        ];
        let rotate = |x: [f64; 3]| m.map(|row| dot(row, x));
        let (pos, vel) = (rotate(pos), rotate(vel));
        [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]
    }

    fn sample_states() -> Vec<[f64; 6]> {
        [
            (0.0, 0.0, 0.0),
            (35.0, 80.0, 50.0),
            (120.0, 200.0, 170.0),
            (250.0, 310.0, 290.0),
        ]
        .iter()
        .map(|&(node, peri, nu)| state_from_elements(1.4, 0.25, 20.0, node, peri, nu))
        .collect()
    }

    fn jacobians(states: &[[f64; 6]], frame: LocalFrame) -> Vec<[f64; 36]> {
        let flat: Vec<f64> = states.iter().flatten().copied().collect();
        local_frame_jacobians(&flat, &vec![MU_SUN; states.len()], frame)
            .unwrap()
            .chunks_exact(36)
            .map(|j| j.try_into().unwrap())
            .collect()
    }

    /// 3x3 block of a row-major 6x6 starting at (row, col).
    fn block(m: &[f64; 36], row: usize, col: usize) -> [[f64; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| m[(row + i) * 6 + col + j]))
    }

    fn max_abs_diff(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
        (0..9)
            .map(|k| (a[k / 3][k % 3] - b[k / 3][k % 3]).abs())
            .fold(0.0, f64::max)
    }

    fn max_abs(a: [[f64; 3]; 3]) -> f64 {
        max_abs_diff(a, [[0.0; 3]; 3])
    }

    fn naive_product(j: &[f64; 36], c: &[f64]) -> [f64; 36] {
        std::array::from_fn(|ij| {
            let (i, jj) = (ij / 6, ij % 6);
            (0..36)
                .map(|kl| j[i * 6 + kl / 6] * c[kl] * j[jj * 6 + kl % 6])
                .sum()
        })
    }

    /// Covariance dominated by a timing error along the orbit, plus small noise.
    fn timing_covariance(state: &[f64; 6], sigma_t: f64) -> Vec<f64> {
        let r = [state[0], state[1], state[2]];
        let r3 = dot(r, r) * dot(r, r).sqrt();
        let s = [
            state[3],
            state[4],
            state[5],
            -MU_SUN * r[0] / r3,
            -MU_SUN * r[1] / r3,
            -MU_SUN * r[2] / r3,
        ];
        let noise = [1e-14, 2e-14, 3e-14, 1e-19, 2e-19, 3e-19];
        (0..36)
            .map(|kl| {
                let (k, l) = (kl / 6, kl % 6);
                sigma_t * sigma_t * s[k] * s[l] + if k == l { noise[k] } else { 0.0 }
            })
            .collect()
    }

    /// Well conditioned covariance L L^T from a fixed pseudo-random L.
    fn well_conditioned_covariance(seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let scale = [1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8];
        let mut l = [0.0; 36];
        for i in 0..6 {
            for j in 0..=i {
                l[i * 6 + j] = scale[i] * if i == j { 1.0 + next() } else { 0.3 * next() };
            }
        }
        (0..36)
            .map(|ij| {
                (0..6)
                    .map(|k| l[(ij / 6) * 6 + k] * l[(ij % 6) * 6 + k])
                    .sum()
            })
            .collect()
    }

    #[test]
    fn parse_accepts_registry_names_aliases_and_bare_families() {
        for frame in ALL_FRAMES {
            assert_eq!(LocalFrame::parse(frame.canonical_name()).unwrap(), frame);
        }
        let cases = [
            ("RTN", "RSW_INERTIAL"),
            ("ric", "RSW_INERTIAL"),
            (" rsw ", "RSW_INERTIAL"),
            ("tnw", "TNW_INERTIAL"),
            ("VNC", "VNC_INERTIAL"),
            ("vnc_rotating", "VNC_ROTATING"),
            ("\tTnw_Inertial\n", "TNW_INERTIAL"),
            ("  Rsw_Rotating", "RSW_ROTATING"),
        ];
        for (name, canonical) in cases {
            assert_eq!(LocalFrame::parse(name).unwrap().canonical_name(), canonical);
        }
    }

    #[test]
    fn parse_rejects_unknown_names() {
        assert_eq!(
            LocalFrame::parse("LVLH").unwrap_err(),
            SchemaError::InvalidRecordBatch(
                "Unknown local orbital frame 'LVLH', expected one of [\"RSW_ROTATING\", \
                 \"RSW_INERTIAL\", \"TNW_ROTATING\", \"TNW_INERTIAL\", \"VNC_ROTATING\", \
                 \"VNC_INERTIAL\"]"
                    .to_string()
            )
        );
        for name in [
            "",
            "VNC_",
            "VNC ROTATING",
            "QSW",
            "RTN_ROTATING",
            "RSW_INERTIAL_",
        ] {
            assert!(LocalFrame::parse(name).is_err(), "{name:?} parsed");
        }
    }

    #[test]
    fn axes_follow_the_registry_and_are_orthonormal() {
        let states = sample_states();
        for family in FAMILIES {
            for rotating in [false, true] {
                for (state, j) in states
                    .iter()
                    .zip(jacobians(&states, frame(family, rotating)))
                {
                    let r = [state[0], state[1], state[2]];
                    let v = [state[3], state[4], state[5]];
                    let (r_hat, v_hat, h_hat) = (unit(r), unit(v), unit(cross(r, v)));
                    let expected = match family {
                        LocalFrameFamily::Rsw => [r_hat, cross(h_hat, r_hat), h_hat],
                        LocalFrameFamily::Tnw => [v_hat, cross(h_hat, v_hat), h_hat],
                        LocalFrameFamily::Vnc => [v_hat, h_hat, cross(v_hat, h_hat)],
                    };
                    let rotation = block(&j, 0, 0);
                    assert!(max_abs_diff(rotation, expected) < 1e-14);
                    assert_eq!(block(&j, 3, 3), rotation);
                    assert_eq!(block(&j, 0, 3), [[0.0; 3]; 3]);
                    let gram = rotation.map(|a| rotation.map(|b| dot(a, b)));
                    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                    assert!(max_abs_diff(gram, identity) < 1e-14);
                    // right handed, and h maps onto the normal axis
                    let z = cross(rotation[0], rotation[1]);
                    assert!(max_abs_diff([z; 3], [rotation[2]; 3]) < 1e-14);
                    let h_local = rotation.map(|row| dot(row, h_hat));
                    assert!((h_local[family.normal_axis()] - 1.0).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn vnc_is_tnw_with_rows_t_w_minus_n() {
        let states = sample_states();
        for rotating in [false, true] {
            let tnw = jacobians(&states, frame(LocalFrameFamily::Tnw, rotating));
            let vnc = jacobians(&states, frame(LocalFrameFamily::Vnc, rotating));
            for (t, v) in tnw.iter().zip(&vnc) {
                for (row, col) in [(0, 0), (3, 0), (3, 3)] {
                    let (t, v) = (block(t, row, col), block(v, row, col));
                    let permuted = [t[0], t[2], t[1].map(|x| -x)];
                    assert!(max_abs_diff(v, permuted) <= 1e-15);
                }
            }
        }
    }

    #[test]
    fn inertial_jacobians_have_no_rate_block() {
        let states = sample_states();
        for family in FAMILIES {
            let inertial = jacobians(&states, frame(family, false));
            let rotating = jacobians(&states, frame(family, true));
            for (i, r) in inertial.iter().zip(&rotating) {
                assert_eq!(block(i, 3, 0), [[0.0; 3]; 3]);
                assert_eq!(block(i, 0, 0), block(r, 0, 0));
                assert!(max_abs(block(r, 3, 0)) > 0.0);
            }
        }
        // mu is not read for inertial frames
        let flat: Vec<f64> = states.iter().flatten().copied().collect();
        assert!(local_frame_jacobians(&flat, &[], frame(LocalFrameFamily::Vnc, false)).is_ok());
    }

    #[test]
    fn rate_block_matches_finite_differences_along_the_orbit() {
        let dt = 1e-3;
        let states = sample_states();
        // second order Taylor steps of the two-body motion
        let step = |state: &[f64; 6], h: f64| -> [f64; 6] {
            let r = [state[0], state[1], state[2]];
            let v = [state[3], state[4], state[5]];
            let r2 = dot(r, r);
            let r3 = r2 * r2.sqrt();
            let rv = dot(r, v);
            let a: [f64; 3] = std::array::from_fn(|k| -MU_SUN * r[k] / r3);
            let a_dot: [f64; 3] =
                std::array::from_fn(|k| -MU_SUN * (v[k] / r3 - 3.0 * r[k] * rv / (r3 * r2)));
            std::array::from_fn(|k| {
                if k < 3 {
                    r[k] + v[k] * h + 0.5 * a[k] * h * h
                } else {
                    v[k - 3] + a[k - 3] * h + 0.5 * a_dot[k - 3] * h * h
                }
            })
        };
        for family in FAMILIES {
            let rotating = jacobians(&states, frame(family, true));
            let plus: Vec<[f64; 6]> = states.iter().map(|s| step(s, dt)).collect();
            let minus: Vec<[f64; 6]> = states.iter().map(|s| step(s, -dt)).collect();
            let plus = jacobians(&plus, frame(family, false));
            let minus = jacobians(&minus, frame(family, false));
            for ((j, p), m) in rotating.iter().zip(&plus).zip(&minus) {
                let (p, m) = (block(p, 0, 0), block(m, 0, 0));
                let difference = std::array::from_fn(|i| {
                    std::array::from_fn(|k| (p[i][k] - m[i][k]) / (2.0 * dt))
                });
                let rate = block(j, 3, 0);
                assert!(max_abs_diff(rate, difference) <= 1e-8 * max_abs(rate));
            }
        }
    }

    #[test]
    fn circular_orbit_frame_rate_is_the_mean_motion() {
        let a = 1.3;
        let n = (MU_SUN / (a * a * a)).sqrt();
        let state = state_from_elements(a, 0.0, 20.0, 40.0, 0.0, 75.0);
        let h_hat = unit(cross(
            [state[0], state[1], state[2]],
            [state[3], state[4], state[5]],
        ));
        for family in FAMILIES {
            let j = jacobians(&[state], frame(family, true))[0];
            let (rotation, rate) = (block(&j, 0, 0), block(&j, 3, 0));
            // omega = 1/2 sum_i e_i x de_i/dt for an orthonormal triad
            let omega: [f64; 3] = std::array::from_fn(|k| {
                0.5 * (0..3).map(|i| cross(rotation[i], rate[i])[k]).sum::<f64>()
            });
            assert!((dot(omega, omega).sqrt() - n).abs() <= 1e-12 * n);
            assert!(max_abs_diff([omega; 3], [h_hat.map(|x| n * x); 3]) <= 1e-15);
            let expected = rotation.map(|e| cross(h_hat.map(|x| n * x), e));
            assert!(max_abs_diff(rate, expected) <= 1e-15);
        }
    }

    #[test]
    fn normal_row_of_the_rate_block_is_exactly_zero() {
        let states = sample_states();
        for family in FAMILIES {
            for j in jacobians(&states, frame(family, true)) {
                let rate = block(&j, 3, 0);
                for (i, row) in rate.iter().enumerate() {
                    if i == family.normal_axis() {
                        assert_eq!(*row, [0.0; 3]);
                    } else {
                        assert!(row.iter().any(|&x| x != 0.0));
                    }
                }
            }
        }
    }

    #[test]
    fn covariance_product_matches_f64_and_is_bit_symmetric() {
        let states = sample_states();
        let flat: Vec<f64> = states.iter().flatten().copied().collect();
        let covariances: Vec<f64> = (0..states.len() as u64)
            .flat_map(well_conditioned_covariance)
            .collect();
        let mu = vec![MU_SUN; states.len()];
        for frame in ALL_FRAMES {
            let rotated = local_frame_covariances(&flat, &covariances, &mu, frame).unwrap();
            let jacobians = jacobians(&states, frame);
            for ((p, c), j) in rotated
                .chunks_exact(36)
                .zip(covariances.chunks_exact(36))
                .zip(&jacobians)
            {
                let naive = naive_product(j, c);
                for i in 0..6 {
                    for k in 0..6 {
                        let scale = (p[i * 7] * p[k * 7]).sqrt();
                        assert!((p[i * 6 + k] - naive[i * 6 + k]).abs() <= 1e-12 * scale);
                        assert_eq!(p[i * 6 + k].to_bits(), p[k * 6 + i].to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn covariance_is_symmetrised_exactly() {
        let state = sample_states()[1];
        let mut c = well_conditioned_covariance(7);
        // ulp level asymmetry, as stored covariances carry
        for (k, l) in [(0, 3), (1, 4), (2, 5), (0, 1)] {
            c[k * 6 + l] = f64::from_bits(c[k * 6 + l].to_bits() + 3);
        }
        let transposed: Vec<f64> = (0..36).map(|kl| c[(kl % 6) * 6 + kl / 6]).collect();
        for frame in ALL_FRAMES {
            let a = local_frame_covariances(&state, &c, &[MU_SUN], frame).unwrap();
            let b = local_frame_covariances(&state, &transposed, &[MU_SUN], frame).unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }

    #[test]
    fn nan_covariance_rows_pass_through() {
        let mut states = sample_states();
        // a radial state without covariance does not need an orbit plane
        states[2] = [1.0, 0.0, 0.0, 0.01, 0.0, 0.0];
        let flat: Vec<f64> = states.iter().flatten().copied().collect();
        let mut covariances: Vec<f64> = (0..states.len() as u64)
            .flat_map(well_conditioned_covariance)
            .collect();
        covariances[72..108].fill(f64::NAN);
        let mu = vec![MU_SUN; states.len()];
        for frame in ALL_FRAMES {
            let rotated = local_frame_covariances(&flat, &covariances, &mu, frame).unwrap();
            for (index, row) in rotated.chunks_exact(36).enumerate() {
                if index == 2 {
                    assert!(row.iter().all(|x| x.is_nan()));
                } else {
                    assert!(row.iter().all(|x| x.is_finite()));
                }
            }
        }
    }

    #[test]
    fn product_is_correctly_rounded_for_an_f64_jacobian() {
        // Exact J C J^T for an f64 Jacobian from error free expansions,
        // against which the double-double product is within one ulp.
        fn exact(j: &[f64; 36], c: &[f64], i: usize, k: usize) -> f64 {
            let mut parts = Vec::new();
            for a in 0..6 {
                for b in 0..6 {
                    let (s, e) = two_sum(c[a * 6 + b], c[b * 6 + a]);
                    let (p, q) = two_prod(j[i * 6 + a], j[k * 6 + b]);
                    for x in [p, q] {
                        for y in [0.5 * s, 0.5 * e] {
                            let (u, w) = two_prod(x, y);
                            parts.extend([u, w]);
                        }
                    }
                }
            }
            // Shewchuk's exact summation into non-overlapping partials
            let mut partials: Vec<f64> = Vec::new();
            for mut x in parts {
                let mut kept = Vec::new();
                for y in partials {
                    let (s, e) = if x.abs() >= y.abs() {
                        quick_two_sum(x, y)
                    } else {
                        quick_two_sum(y, x)
                    };
                    if e != 0.0 {
                        kept.push(e);
                    }
                    x = s;
                }
                kept.push(x);
                partials = kept;
            }
            partials.iter().rev().fold(0.0, |acc, x| acc + x)
        }
        let states = sample_states();
        let mut worst_naive: f64 = 0.0;
        for (index, state) in states.iter().enumerate() {
            let noise = well_conditioned_covariance(index as u64);
            let timing = timing_covariance(state, 0.3);
            let c: Vec<f64> = timing.iter().zip(&noise).map(|(a, b)| a + b).collect();
            for family in FAMILIES {
                let frame = frame(family, true);
                let j = jacobians(&states[index..=index], frame)[0];
                let p = rotate_covariance(&j.map(|x| DoubleDouble::new((x, 0.0))), &c);
                let naive = naive_product(&j, &c);
                for i in 0..6 {
                    for k in 0..6 {
                        let reference = exact(&j, &c, i, k);
                        let ulp = f64::from_bits(reference.abs().to_bits() + 1) - reference.abs();
                        // plus the double-double bound, for elements that cancel by 50 bits
                        let terms: f64 = (0..36)
                            .map(|ab| (j[i * 6 + ab / 6] * c[ab] * j[k * 6 + ab % 6]).abs())
                            .sum();
                        let tolerance = ulp + terms * 2f64.powi(-100);
                        assert!((p[i * 6 + k] - reference).abs() <= tolerance);
                        worst_naive = worst_naive.max((naive[i * 6 + k] - reference).abs() / ulp);
                    }
                }
            }
        }
        // the plain f64 product is far off on these elements
        assert!(worst_naive > 1e3, "{worst_naive}");
    }

    #[test]
    fn invalid_inputs_are_reported() {
        let states = sample_states();
        let mut flat: Vec<f64> = states.iter().flatten().copied().collect();
        let vnc = frame(LocalFrameFamily::Vnc, true);
        let err = local_frame_jacobians(&flat[..7], &[MU_SUN], vnc).unwrap_err();
        assert!(err.to_string().contains("6 entries per state"));
        let err = local_frame_jacobians(&flat, &[MU_SUN], vnc).unwrap_err();
        assert!(err.to_string().contains("mu"));
        let covariances = vec![0.0; 36 * states.len() - 1];
        let mu = vec![MU_SUN; states.len()];
        let err = local_frame_covariances(&flat, &covariances, &mu, vnc).unwrap_err();
        assert!(err.to_string().contains("covariances"));
        // radial motion has no orbit plane
        let radial = [2.0 * flat[6], 2.0 * flat[7], 2.0 * flat[8]];
        flat[9..12].copy_from_slice(&radial);
        for frame in ALL_FRAMES {
            assert_eq!(
                local_frame_jacobians(&flat, &mu, frame).unwrap_err(),
                SchemaError::InvalidRecordBatch("State 1 has no orbit plane.".to_string())
            );
        }
    }
}
