//! Covariances in the local orbital frames of the SANA registry: RSW (aliases RTN,
//! RIC), TNW and VNC, each `_INERTIAL` or `_ROTATING` (velocity rows carry the
//! two-body frame rate). The Jacobian and `J C J^T` are evaluated in double-double
//! and rounded once, because the rotating frame velocity rows cancel deeply.

use std::ops::{Add, Div, Mul, Sub};

use crate::types::{SchemaError, SchemaResult};

/// |r x v| (AU^2/day) below which a state has no orbit plane, as in `CartesianCoordinates`.
const SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE: f64 = 1e-20;

/// Canonical names, indexed by `2 * family + inertial`.
#[rustfmt::skip]
const NAMES: [&str; 6] = [
    "RSW_ROTATING", "RSW_INERTIAL",
    "TNW_ROTATING", "TNW_INERTIAL",
    "VNC_ROTATING", "VNC_INERTIAL",
];

/// x along position (RSW) or velocity (TNW, VNC), the orbit normal z (RSW, TNW) or y (VNC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalFrameFamily {
    Rsw,
    Tnw,
    Vnc,
}

/// A local orbital frame: family plus whether velocity rows carry the frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalFrame {
    pub family: LocalFrameFamily,
    pub rotating: bool,
}

impl LocalFrame {
    /// Parse a registry name (`VNC_ROTATING`, `RTN_INERTIAL`, ...) or a bare family name
    /// meaning `_INERTIAL`, ignoring case and surrounding whitespace.
    pub fn parse(name: &str) -> SchemaResult<Self> {
        let upper = name.trim().to_ascii_uppercase();
        let (family, suffix) = upper.split_once('_').unwrap_or((&upper, "INERTIAL"));
        let family = match family {
            "RSW" | "RTN" | "RIC" => Some(LocalFrameFamily::Rsw),
            "TNW" => Some(LocalFrameFamily::Tnw),
            "VNC" => Some(LocalFrameFamily::Vnc),
            _ => None,
        };
        let rotating = suffix == "ROTATING";
        match family {
            Some(family) if rotating || suffix == "INERTIAL" => Ok(Self { family, rotating }),
            _ => Err(invalid(format!(
                "Unknown local orbital frame '{name}', expected one of {NAMES:?}"
            ))),
        }
    }

    /// The canonical registry name, e.g. `VNC_ROTATING`.
    pub fn canonical_name(&self) -> &'static str {
        NAMES[2 * self.family as usize + usize::from(!self.rotating)]
    }

    /// Row-major `[[R, 0], [dR/dt, R]]` of state `index`, the rows of R being the frame axes.
    fn jacobian(self, values: &[f64], mu: &[f64], index: usize) -> SchemaResult<[Dd; 36]> {
        let mu = if self.rotating { mu[index] } else { 0.0 };
        if self.rotating && !(mu.is_finite() && mu > 0.0) {
            return Err(invalid(format!(
                "mu must be finite and positive for a _ROTATING frame, got {mu} at state {index}"
            )));
        }
        let [x, y, z, vx, vy, vz]: [f64; 6] = values[index * 6..index * 6 + 6].try_into().unwrap();
        let h = [y * vz - z * vy, z * vx - x * vz, x * vy - y * vx];
        // NaN fails the comparison, and an infinite component makes the norm infinite or NaN.
        let norm = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
        if !(norm >= SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE && norm.is_finite()) {
            let message = format!("State {index} is not finite or has no orbit plane.");
            return Err(invalid(message));
        }

        use LocalFrameFamily::{Rsw, Vnc};
        let r = [x, y, z].map(|c| Dd(c, 0.0));
        let v = [vx, vy, vz].map(|c| Dd(c, 0.0));
        let (rsw, vnc) = (self.family == Rsw, self.family == Vnc);
        let h_hat = unit(cross(r, v)).0;
        let (x_hat, u_norm) = unit(if rsw { r } else { v });
        let zero = [Dd::ZERO; 3];
        let axes = match vnc {
            true => [x_hat, h_hat, cross(x_hat, h_hat)],
            false => [x_hat, cross(h_hat, x_hat), h_hat],
        };
        let mut rates = [zero; 3];
        if self.rotating {
            // d(u/|u|)/dt with du/dt the velocity (RSW) or the two-body acceleration (TNW, VNC).
            let r_norm = dot(r, r).sqrt();
            let scale = Dd(-mu, 0.0) / (r_norm * r_norm * r_norm);
            let du = if rsw { v } else { r.map(|c| c * scale) };
            let along = dot(du, x_hat);
            let dx: V = std::array::from_fn(|k| (du[k] - along * x_hat[k]) / u_norm);
            // h is constant under two-body motion, so the completing axis turns with x.
            let d_axes = match vnc {
                true => [dx, zero, cross(dx, h_hat)],
                false => [dx, cross(h_hat, dx), zero],
            };
            // omega = 1/2 sum e_i x de_i/dt; each axis turns as omega x e_i, and the
            // orbit normal row is exactly zero.
            let turns: [V; 3] = std::array::from_fn(|i| cross(axes[i], d_axes[i]));
            let omega: V =
                std::array::from_fn(|k| (turns[0][k] + turns[1][k] + turns[2][k]) * Dd(0.5, 0.0));
            rates = axes.map(|axis| cross(omega, axis));
            rates[if vnc { 1 } else { 2 }] = zero;
        }
        let mut jacobian = [Dd::ZERO; 36];
        for (i, j) in (0..9).map(|ij| (ij / 3, ij % 3)) {
            jacobian[i * 6 + j] = axes[i][j];
            jacobian[(i + 3) * 6 + j + 3] = axes[i][j];
            jacobian[(i + 3) * 6 + j] = rates[i][j];
        }
        Ok(jacobian)
    }
}

fn invalid(message: String) -> SchemaError {
    SchemaError::InvalidRecordBatch(message)
}

/// Row-major 6x6 Jacobians from inertial `values` (`N * 6`, AU and AU/day) to `frame`.
/// `mu` (AU^3/day^2, one per state) is read only for `_ROTATING` frames.
pub fn local_frame_jacobians(
    values: &[f64],
    mu: &[f64],
    frame: LocalFrame,
) -> SchemaResult<Vec<f64>> {
    let n = state_count(values, mu, frame)?;
    let mut out = Vec::with_capacity(n * 36);
    for index in 0..n {
        out.extend(frame.jacobian(values, mu, index)?.map(|x| x.0 + x.1));
    }
    Ok(out)
}

/// `J C J^T` for every state, row-major like `covariances` and symmetric to the bit.
/// Rows whose covariance is all NaN stay NaN and need neither an orbit plane nor `mu`.
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
    let rows = covariances.chunks_exact(36).zip(out.chunks_exact_mut(36));
    for (index, (covariance, rotated)) in rows.enumerate() {
        if !covariance.iter().all(|x| x.is_nan()) {
            let jacobian = frame.jacobian(values, mu, index)?;
            rotated.copy_from_slice(&rotate_covariance(&jacobian, covariance));
        }
    }
    Ok(out)
}

fn state_count(values: &[f64], mu: &[f64], frame: LocalFrame) -> SchemaResult<usize> {
    if values.len() % 6 != 0 {
        return Err(invalid(format!(
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

/// `J C J^T` of one state: each upper element is the double-double sum of
/// its 36 products over the exactly symmetrised `C`, rounded once and mirrored.
fn rotate_covariance(jacobian: &[Dd; 36], covariance: &[f64]) -> [f64; 36] {
    // (C + C^T) / 2 is exact in double-double, halving being exact.
    let symmetric: [Dd; 36] = std::array::from_fn(|kl| {
        let (s, e) = two_sum(covariance[kl], covariance[kl % 6 * 6 + kl / 6]);
        Dd(0.5 * s, 0.5 * e)
    });
    let mut out = [0.0; 36];
    for i in 0..6 {
        for j in i..6 {
            let (a, b) = (&jacobian[i * 6..i * 6 + 6], &jacobian[j * 6..j * 6 + 6]);
            let terms = (0..36).map(|kl| a[kl / 6] * b[kl % 6] * symmetric[kl]);
            let sum = terms.fold(Dd::ZERO, |sum, term| sum + term);
            out[i * 6 + j] = sum.0 + sum.1;
            out[j * 6 + i] = out[i * 6 + j];
        }
    }
    out
}

type V = [Dd; 3];

fn dot(a: V, b: V) -> Dd {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The unit vector along `a` and the norm of `a`.
fn unit(a: V) -> (V, Dd) {
    let norm = dot(a, a).sqrt();
    (a.map(|x| x / norm), norm)
}

/// A double-double `hi + lo` with `|lo| <= ulp(hi) / 2`, about 32 significant digits.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Dd(f64, f64);

/// Exact `a + b = s + e` (Knuth).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Exact `a + b` when `|a| >= |b|`.
fn quick_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    Dd(s, b - (s - a))
}

/// Exact `a * b = p + e`.
fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    Dd(p, a.mul_add(b, -p))
}

impl Dd {
    const ZERO: Self = Self(0.0, 0.0);
    /// One Newton step from the f64 root.
    fn sqrt(self) -> Self {
        let s = self.0.sqrt();
        let remainder = self - two_prod(s, s);
        quick_two_sum(s, remainder.0 / (2.0 * s))
    }
}

impl Add for Dd {
    type Output = Self;
    /// Accurate sum: relative error near 2^-104 of the result, also under cancellation.
    fn add(self, other: Self) -> Self {
        let (s, e) = two_sum(self.0, other.0);
        let (t, f) = two_sum(self.1, other.1);
        let Dd(s, e) = quick_two_sum(s, e + t);
        quick_two_sum(s, e + f)
    }
}

impl Sub for Dd {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + Self(-other.0, -other.1)
    }
}

impl Mul for Dd {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let Dd(p, e) = two_prod(self.0, other.0);
        quick_two_sum(p, e + (self.0 * other.1 + self.1 * other.0))
    }
}

impl Div for Dd {
    type Output = Self;
    /// Two quotient digits, the second from the double-double remainder.
    fn div(self, other: Self) -> Self {
        let q = self.0 / other.0;
        let remainder = self - other * Self(q, 0.0);
        quick_two_sum(q, remainder.0 / other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU_SUN: f64 = 0.000_295_912_208_284_119_56;
    const STATES: [[f64; 6]; 3] = [
        [1.1, -0.3, 0.05, 0.001, 0.017, -0.0003],
        [0.9, 0.4, -0.02, -0.009, 0.015, 0.0002],
        [-1.3, 0.1, 0.27, 0.004, -0.012, 0.0031],
    ];
    const ZERO: [[f64; 3]; 3] = [[0.0; 3]; 3];

    fn frames() -> [LocalFrame; 6] {
        NAMES.map(|name| LocalFrame::parse(name).unwrap())
    }

    /// The 3x3 block at (row, col) of the Jacobian of `state`.
    fn block(state: &[f64; 6], frame: LocalFrame, row: usize, col: usize) -> [[f64; 3]; 3] {
        let jacobian = local_frame_jacobians(state, &[MU_SUN], frame).unwrap();
        std::array::from_fn(|i| std::array::from_fn(|k| jacobian[(row + i) * 6 + col + k]))
    }

    fn max_diff(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> f64 {
        let differences = (0..9).map(|k| (a[k / 3][k % 3] - b[k / 3][k % 3]).abs());
        differences.fold(0.0, f64::max)
    }

    #[test]
    fn parse_accepts_registry_names_and_aliases() {
        let aliases = [" rtn ", "Vnc", "\tric_Rotating\n", "RIC_INERTIAL"];
        let names = NAMES.into_iter().chain(aliases);
        for (name, index) in names.zip([0, 1, 2, 3, 4, 5, 1, 5, 0, 1]) {
            let frame = LocalFrame::parse(name).unwrap();
            assert_eq!(frame.canonical_name(), NAMES[index]);
        }
        let error = "Unknown local orbital frame 'LVLH', expected one of [\"RSW_ROTATING\", \
                     \"RSW_INERTIAL\", \"TNW_ROTATING\", \"TNW_INERTIAL\", \"VNC_ROTATING\", \
                     \"VNC_INERTIAL\"]";
        assert_eq!(LocalFrame::parse("LVLH"), Err(invalid(error.to_string())));
        for name in ["", "VNC_", "VNC ROTATING", "RTN_", "RICE", "RSW_INERTIAL_"] {
            assert!(LocalFrame::parse(name).is_err(), "{name:?} parsed");
        }
    }

    #[test]
    fn axes_follow_the_registry_and_vnc_is_tnw_permuted() {
        for state in &STATES {
            let r = [state[0], state[1], state[2]].map(|x| Dd(x, 0.0));
            let v = [state[3], state[4], state[5]].map(|x| Dd(x, 0.0));
            let h = unit(cross(r, v)).0;
            for (index, frame) in frames().into_iter().enumerate() {
                let x = unit([r, v][usize::from(index > 1)]).0;
                let axes = match frame.family {
                    LocalFrameFamily::Vnc => [x, h, cross(x, h)],
                    _ => [x, cross(h, x), h],
                };
                let rotation = block(state, frame, 0, 0);
                assert!(max_diff(rotation, axes.map(|axis| axis.map(|x| x.0))) < 1e-15);
                assert_eq!(block(state, frame, 3, 3), rotation);
                assert_eq!(block(state, frame, 0, 3), ZERO);
                assert_eq!(block(state, frame, 3, 0) == ZERO, !frame.rotating);
            }
            // VNC rows are the TNW rows T, W, -N, in the rotation and in the rate block.
            for (tnw, vnc, row) in [(2, 4, 0), (2, 4, 3), (3, 5, 0), (3, 5, 3)] {
                let t = block(state, frames()[tnw], row, 0);
                let permuted = [t[0], t[2], t[1].map(|x| -x)];
                assert!(max_diff(block(state, frames()[vnc], row, 0), permuted) <= 1e-15);
            }
        }
    }

    #[test]
    fn rate_block_matches_finite_differences_along_the_orbit() {
        for state in &STATES {
            // Steps along the two-body flow, whose tangent is (v, a).
            let r3 = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).powf(1.5);
            let scale = [1.0, -MU_SUN / r3];
            let step = |dt: f64| {
                std::array::from_fn(|k| state[k] + dt * scale[k / 3] * state[(k + 3) % 6])
            };
            // frames() pairs each _ROTATING frame with its _INERTIAL one.
            for pair in frames().chunks(2) {
                let plus = block(&step(1e-3), pair[1], 0, 0);
                let minus = block(&step(-1e-3), pair[1], 0, 0);
                let difference = std::array::from_fn(|i| {
                    std::array::from_fn(|k| (plus[i][k] - minus[i][k]) / 2e-3)
                });
                let rate = block(state, pair[0], 3, 0);
                assert!(max_diff(rate, difference) <= 1e-8 * max_diff(rate, ZERO));
            }
        }
    }

    #[test]
    fn product_is_bit_symmetric_and_nan_rows_pass_through() {
        // Asymmetric covariances, so C and C^T must give the same bits; row 1 is all NaN,
        // on a radial state without mu.
        let mut covariances: Vec<f64> = (0..108).map(|i| 1e-12 * (i as f64).sin()).collect();
        covariances[36..72].fill(f64::NAN);
        let mirror = |i: usize| i / 36 * 36 + i % 6 * 6 + i % 36 / 6;
        let transposed: Vec<f64> = (0..3 * 36).map(|i| covariances[mirror(i)]).collect();
        let mut states = STATES.concat();
        states[6..12].copy_from_slice(&[1.0, 0.0, 0.0, 0.01, 0.0, 0.0]);
        let mu = [MU_SUN, f64::NAN, MU_SUN];
        for frame in frames() {
            let a = local_frame_covariances(&states, &covariances, &mu, frame).unwrap();
            let b = local_frame_covariances(&states, &transposed, &mu, frame).unwrap();
            for (i, x) in a.iter().enumerate() {
                assert_eq!([b[i], a[mirror(i)]].map(f64::to_bits), [x.to_bits(); 2]);
                assert_eq!(x.is_nan(), (36..72).contains(&i));
            }
        }
    }

    #[test]
    fn invalid_inputs_are_reported() {
        let (states, covariances, vnc) = (STATES.concat(), vec![0.0; 3 * 36], frames()[4]);
        let message = |result: SchemaResult<Vec<f64>>| result.unwrap_err().to_string();
        let short = local_frame_jacobians(&states[..7], &[MU_SUN], vnc);
        assert!(message(short).contains("must hold 6 entries per state"));
        let mu = local_frame_jacobians(&states, &[MU_SUN], vnc);
        assert!(message(mu).contains("mu (one entry per state for VNC_ROTATING)"));
        let short = local_frame_covariances(&states, &covariances[1..], &[MU_SUN; 3], vnc);
        assert!(message(short).contains("covariances (36 entries per state)"));
        // mu must be finite and positive for rotating frames and is not read otherwise.
        for value in [f64::NAN, f64::INFINITY, 0.0, -MU_SUN] {
            let mu = [MU_SUN, MU_SUN, value];
            let text = format!("must be finite and positive for a _ROTATING frame, got {value}");
            let error = invalid(format!("mu {text} at state 2"));
            for frame in frames() {
                let expected = frame.rotating.then(|| error.clone());
                assert_eq!(local_frame_jacobians(&states, &mu, frame).err(), expected);
                let rotated = local_frame_covariances(&states, &covariances, &mu, frame);
                assert_eq!(rotated.err(), expected);
            }
        }
        // Radial motion has no orbit plane, and NaN or infinite states fail the same check.
        let error = invalid("State 1 is not finite or has no orbit plane.".to_string());
        let radial = [1.8, 0.8, -0.04];
        for velocity in [radial, [f64::NAN; 3], [f64::INFINITY, 0.0, 0.0]] {
            let mut states = states.clone();
            states[9..12].copy_from_slice(&velocity);
            for frame in frames() {
                let jacobians = local_frame_jacobians(&states, &[MU_SUN; 3], frame);
                assert_eq!(jacobians, Err(error.clone()));
            }
        }
    }
}
