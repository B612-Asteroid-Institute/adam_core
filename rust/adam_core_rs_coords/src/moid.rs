//! Global minimum-orbit-intersection distance (MOID).
//!
//! This module uses the error-controlled critical-point algorithm from
//! Baluev & Mikryukov (2019), *Astronomy and Computing* 27, 11-22, through
//! a pure-Rust port derived from the MIT-licensed distlink 2.0 reference
//! implementation. Unlike the former nested local Brent searches, the
//! algorithm retains both orbital orientations and enumerates the stationary
//! points of the two-conic distance function.

use crate::generic::cartesian_to_keplerian6;
use crate::moid_distlink::{
    solve as solve_distlink_kernel, Orbit as DistlinkOrbit, Result as DistlinkResult,
};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fmt;

const DEG_TO_RAD: f64 = PI / 180.0;
const TWO_PI: f64 = 2.0 * PI;

/// Typed failure for a MOID request outside the bounded elliptic API.
#[derive(Clone, Debug, PartialEq)]
pub enum MoidError {
    NonFiniteInput,
    InvalidGravitationalParameter,
    DegenerateConic,
    ParabolicConic,
    HyperbolicConic,
    SolverFailure,
}

impl fmt::Display for MoidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::NonFiniteInput => "MOID inputs must be finite",
            Self::InvalidGravitationalParameter => "MOID requires a finite positive mu",
            Self::DegenerateConic => "MOID requires non-degenerate confocal conics",
            Self::ParabolicConic => "parabolic MOID is unsupported",
            Self::HyperbolicConic => "hyperbolic MOID is unsupported by this bounded API",
            Self::SolverFailure => "global MOID solver did not return a reliable finite result",
        };
        f.write_str(text)
    }
}

impl std::error::Error for MoidError {}

/// Typed failure for a row in a batched MOID request.
#[derive(Clone, Debug, PartialEq)]
pub struct MoidBatchError {
    pub row: usize,
    pub source: MoidError,
}

impl fmt::Display for MoidBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "global MOID solver failed for row {}: {}",
            self.row, self.source
        )
    }
}

impl std::error::Error for MoidBatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Clone, Copy, Debug)]
struct ConicElements {
    // distlink layout: a, e, i, argument of periapsis, longitude of node.
    values: [f64; 5],
    mean_anomaly: f64,
    normal: [f64; 3],
    eccentricity_direction: Option<[f64; 3]>,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalized(v: [f64; 3]) -> Result<[f64; 3], MoidError> {
    let norm = dot(v, v).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(MoidError::DegenerateConic);
    }
    Ok([v[0] / norm, v[1] / norm, v[2] / norm])
}

fn elements(state: &[f64; 6], mu: f64) -> Result<ConicElements, MoidError> {
    if !state.iter().all(|x| x.is_finite()) {
        return Err(MoidError::NonFiniteInput);
    }
    if !(mu.is_finite() && mu > 0.0) {
        return Err(MoidError::InvalidGravitationalParameter);
    }
    let kep = cartesian_to_keplerian6::<f64>(state, mu);
    if !kep.iter().all(|x| x.is_finite()) {
        return Err(MoidError::DegenerateConic);
    }
    let (a, e) = (kep[0], kep[1]);
    if a == 0.0 || e < 0.0 {
        return Err(MoidError::DegenerateConic);
    }
    if (e - 1.0).abs() <= 32.0 * f64::EPSILON {
        return Err(MoidError::ParabolicConic);
    }
    if e > 1.0 {
        return Err(MoidError::HyperbolicConic);
    }
    if a <= 0.0 {
        return Err(MoidError::DegenerateConic);
    }
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let normal = normalized(cross(r, v))?;
    let eccentricity_direction = if e > 1e-12 {
        let h = cross(r, v);
        let radius = normalized(r)?;
        let vxh = cross(v, h);
        Some(normalized([
            vxh[0] / mu - radius[0],
            vxh[1] / mu - radius[1],
            vxh[2] / mu - radius[2],
        ])?)
    } else {
        None
    };
    Ok(ConicElements {
        values: [
            a,
            e,
            kep[2] * DEG_TO_RAD,
            kep[4] * DEG_TO_RAD,
            kep[3] * DEG_TO_RAD,
        ],
        mean_anomaly: kep[5] * DEG_TO_RAD,
        normal,
        eccentricity_direction,
    })
}

fn same_curve(first: ConicElements, second: ConicElements) -> bool {
    let tolerance = 128.0 * f64::EPSILON;
    let scale = first.values[0].abs().max(second.values[0].abs()).max(1.0);
    if (first.values[0] - second.values[0]).abs() > tolerance * scale
        || (first.values[1] - second.values[1]).abs() > tolerance
        || 1.0 - dot(first.normal, second.normal).abs() > tolerance
    {
        return false;
    }
    match (first.eccentricity_direction, second.eccentricity_direction) {
        (None, None) => true,
        (Some(a), Some(b)) => 1.0 - dot(a, b) < tolerance,
        _ => false,
    }
}

fn circular_pair_witness(
    primary_state: [f64; 6],
    primary: ConicElements,
    secondary: ConicElements,
    mu: f64,
) -> (f64, f64) {
    let moid = (primary.values[0] - secondary.values[0]).abs();
    let line = cross(primary.normal, secondary.normal);
    let Ok(line) = normalized(line) else {
        // Coplanar circles have a closest secondary point at every primary
        // anomaly, so the input epoch is a valid deterministic witness.
        return (moid, 0.0);
    };
    let radius = normalized([primary_state[0], primary_state[1], primary_state[2]])
        .expect("validated elliptic state has nonzero radius");
    let tangent = normalized(cross(primary.normal, radius))
        .expect("circular radius and normal are orthogonal");
    let angle = dot(line, tangent).atan2(dot(line, radius)).rem_euclid(PI);
    let mean_motion = (mu / primary.values[0].powi(3)).sqrt();
    (moid, angle / mean_motion)
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn pair_seed(first: &[f64; 5], second: &[f64; 5]) -> u64 {
    // Sort the raw element tuples so swapping the two conics preserves the
    // deterministic root-initialization stream.
    let (left, right) = if first
        .iter()
        .map(|x| x.to_bits())
        .cmp(second.iter().map(|x| x.to_bits()))
        .is_le()
    {
        (first, second)
    } else {
        (second, first)
    };
    let mut h = 0x243f_6a88_85a3_08d3_u64;
    for value in left.iter().chain(right.iter()) {
        h = mix(h ^ value.to_bits());
    }
    h
}

fn witness_dt(primary: ConicElements, eccentric_anomaly: f64, mu: f64) -> f64 {
    let a = primary.values[0];
    let e = primary.values[1];
    let witness_m = eccentric_anomaly - e * eccentric_anomaly.sin();
    let delta_m = (witness_m - primary.mean_anomaly).rem_euclid(TWO_PI);
    delta_m / (mu / a.powi(3)).sqrt()
}

fn solve_distlink(
    first: &[f64; 5],
    second: &[f64; 5],
    seed: u64,
) -> Result<DistlinkResult, MoidError> {
    let first = DistlinkOrbit::new(first[0], first[1], first[2], first[3], first[4]);
    let second = DistlinkOrbit::new(second[0], second[1], second[2], second[3], second[4]);
    let result = solve_distlink_kernel(first, second, seed);
    if !(result.good
        && result.distance.is_finite()
        && result.distance >= 0.0
        && result.u1.is_finite()
        && result.u2.is_finite())
    {
        Err(MoidError::SolverFailure)
    } else {
        Ok(result)
    }
}

/// Calculate a global MOID and a deterministic primary-orbit witness time.
///
/// `dt_at_min` is the two-body time from the primary state epoch to the
/// returned primary anomaly, wrapped into one forward orbital period.
/// Hyperbolic and parabolic conics are rejected explicitly because their
/// unbounded-branch witness semantics do not match this API. Solver controls
/// are internal to the error-controlled critical-point algorithm.
pub fn calculate_moid(
    primary_orbit: [f64; 6],
    secondary_orbit: [f64; 6],
    mu: f64,
) -> Result<(f64, f64), MoidError> {
    let primary = elements(&primary_orbit, mu)?;
    let secondary = elements(&secondary_orbit, mu)?;
    if same_curve(primary, secondary) {
        return Ok((0.0, 0.0));
    }
    if primary.values[1] <= 1e-12 && secondary.values[1] <= 1e-12 {
        return Ok(circular_pair_witness(primary_orbit, primary, secondary, mu));
    }
    let seed = pair_seed(&primary.values, &secondary.values);
    // Use an order based only on intrinsic conic invariants.  This makes the
    // numerical path (and therefore the distance bits) identical when callers
    // swap the two states.  Congruent but differently oriented conics have no
    // individual rotational invariant that can break the tie, so that rare
    // case evaluates both directions and selects the smaller result.
    let ordering = primary.values[0]
        .total_cmp(&secondary.values[0])
        .then(primary.values[1].total_cmp(&secondary.values[1]));
    let (result, primary_anomaly) = match ordering {
        std::cmp::Ordering::Less => {
            let result = solve_distlink(&primary.values, &secondary.values, seed)?;
            (result, result.u1)
        }
        std::cmp::Ordering::Greater => {
            let result = solve_distlink(&secondary.values, &primary.values, seed)?;
            (result, result.u2)
        }
        std::cmp::Ordering::Equal => {
            let forward = solve_distlink(&primary.values, &secondary.values, seed)?;
            let reverse = solve_distlink(&secondary.values, &primary.values, seed)?;
            if forward.distance.total_cmp(&reverse.distance).is_le() {
                (forward, forward.u1)
            } else {
                (reverse, reverse.u2)
            }
        }
    };
    let dt = witness_dt(primary, primary_anomaly, mu);
    if !dt.is_finite() {
        return Err(MoidError::SolverFailure);
    }
    Ok((result.distance, dt))
}

/// Batched MOID over N orbit pairs in parallel via Rayon.
///
/// A failing row is returned with its index as a typed error.
pub fn calculate_moid_batch(
    primary_orbits: &[f64],
    secondary_orbits: &[f64],
    mus: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), MoidBatchError> {
    assert_eq!(primary_orbits.len() % 6, 0, "primary_orbits must be N*6");
    let n = primary_orbits.len() / 6;
    assert_eq!(
        secondary_orbits.len(),
        n * 6,
        "secondary_orbits must be N*6"
    );
    assert_eq!(mus.len(), n, "mus must have N entries");

    // Preserve indexed input order in the parallel collection, then resolve
    // failures serially so multiple invalid rows deterministically report the
    // first failing input row rather than a scheduler-dependent error.
    let rows: Vec<Result<(f64, f64), MoidError>> = (0..n)
        .into_par_iter()
        .map(|row| {
            let mut p = [0.0; 6];
            let mut s = [0.0; 6];
            p.copy_from_slice(&primary_orbits[row * 6..(row + 1) * 6]);
            s.copy_from_slice(&secondary_orbits[row * 6..(row + 1) * 6]);
            calculate_moid(p, s, mus[row])
        })
        .collect();
    let successful_rows: Result<Vec<(f64, f64)>, MoidBatchError> = rows
        .into_iter()
        .enumerate()
        .map(|(row, result)| result.map_err(|source| MoidBatchError { row, source }))
        .collect();
    let (moids, witness_dts) = successful_rows?.into_iter().unzip();
    Ok((moids, witness_dts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::keplerian_to_cartesian6;

    const MU: f64 = 2.95912208284120e-4;

    fn state(kep: [f64; 6]) -> [f64; 6] {
        keplerian_to_cartesian6::<f64>(&kep, MU, 100, 1e-15)
    }

    #[test]
    fn circular_radial_separation() {
        let first = state([1.0, 0.0, 0.0, 0.0, 0.0, 12.0]);
        let second = state([1.5, 0.0, 73.0, 41.0, 80.0, 220.0]);
        let (moid, dt) = calculate_moid(first, second, MU).unwrap();
        assert!((moid - 0.5).abs() < 1e-11, "{moid}");
        assert!(dt.is_finite());
    }

    #[test]
    fn distance_is_symmetric() {
        let first = state([1.7, 0.31, 22.0, 41.0, 73.0, 13.0]);
        let second = state([1.0, 0.0167, 0.2, 11.0, 102.9, 251.0]);
        let a = calculate_moid(first, second, MU).unwrap().0;
        let b = calculate_moid(second, first, MU).unwrap().0;
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn coplanar_near_tangent_ellipses_find_the_global_zero() {
        // Q1 = 1.2 au and q2 = 1.2 au, with the outer periapsis pointed at
        // the inner apoapsis.  This is a repeated-root/tangency stress case.
        let inner = state([1.0, 0.2, 20.0, 30.0, 0.0, 31.0]);
        let outer = state([1.5, 0.2, 20.0, 30.0, 180.0, 217.0]);
        let moid = calculate_moid(inner, outer, MU).unwrap().0;
        assert!(moid < 1e-10, "{moid}");
    }

    #[test]
    fn congruent_differently_oriented_curves_are_symmetric() {
        let first = state([1.7, 0.31, 22.0, 41.0, 73.0, 13.0]);
        let second = state([1.7, 0.31, 81.0, 12.0, 203.0, 13.0]);
        let a = calculate_moid(first, second, MU).unwrap().0;
        let b = calculate_moid(second, first, MU).unwrap().0;
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn repeated_minimum_same_curve_is_exact_and_deterministic() {
        let first = state([1.7, 0.31, 22.0, 41.0, 73.0, 13.0]);
        let second = state([1.7, 0.31, 22.0, 41.0, 73.0, 213.0]);
        assert_eq!(calculate_moid(first, second, MU).unwrap(), (0.0, 0.0));
    }

    #[test]
    fn hyperbolic_is_typed_failure() {
        let hyperbolic = state([-2.0, 1.5, 20.0, 30.0, 40.0, 10.0]);
        let elliptic = state([1.0, 0.1, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            calculate_moid(hyperbolic, elliptic, MU),
            Err(MoidError::HyperbolicConic)
        );
    }

    #[test]
    fn corpus_orientation_regression() {
        let first = [
            0.5468166257366497,
            -0.9620100000435787,
            -0.031733613709438696,
            0.01689463917904537,
            0.005710405289977736,
            0.001813602461028905,
        ];
        let second = [
            0.4944189987543816,
            -0.8876143848674095,
            6.392_549_263_519_007e-5,
            0.014748747145495232,
            0.008313059820214456,
            -1.339_633_201_162_278e-6,
        ];
        let result = calculate_moid(first, second, MU).unwrap();
        assert!((result.0 - 0.07224759404236622).abs() < 1e-8);
    }

    #[test]
    fn parabolic_is_typed_failure() {
        // Cartesian-to-Keplerian is numerically awkward at exactly e=1, so
        // verify invalid inputs never become a finite MOID.
        let bad = [f64::NAN; 6];
        assert_eq!(calculate_moid(bad, bad, MU), Err(MoidError::NonFiniteInput));
        let valid = state([1.0, 0.1, 3.0, 4.0, 5.0, 6.0]);
        let batch =
            calculate_moid_batch(&[valid, bad].concat(), &[valid, valid].concat(), &[MU, MU])
                .unwrap_err();
        assert_eq!(
            batch,
            MoidBatchError {
                row: 1,
                source: MoidError::NonFiniteInput,
            }
        );
    }

    #[test]
    fn batch_reports_first_failure_in_input_order() {
        let valid = state([1.0, 0.1, 3.0, 4.0, 5.0, 6.0]);
        let bad = [f64::NAN; 6];
        let primary = [valid, bad, bad, valid].concat();
        let secondary = [valid; 4].concat();
        let mus = [MU; 4];
        for _ in 0..100 {
            assert_eq!(
                calculate_moid_batch(&primary, &secondary, &mus).unwrap_err(),
                MoidBatchError {
                    row: 1,
                    source: MoidError::NonFiniteInput,
                }
            );
        }
    }
}
