//! Systematic ranging for arcs Gauss cannot handle (phase 1).
//!
//! A short arc fixes an *attributable* — the sky position and its rates at a
//! reference epoch — but not the topocentric range ρ or range rate ρ̇
//! (Milani et al. 2004). Ranging samples that missing pair: every
//! (ρ, ρ̇) in the *admissible region* (bound to the Sun, not a satellite
//! of the observer's planet) completes the attributable to an orbit, and
//! each orbit is scored against all the observations (systematic ranging,
//! Farnocchia, Chesley & Micheli 2015; statistical ranging, Virtanen et al.
//! 2001). The result is a weighted orbit sample, `exp(-Δχ²/2)`, and its
//! best member seeds the corrected IOD's staged differential correction.
//!
//! Phase 1 keeps the attributable at its fitted value at every node (the
//! four-parameter fit per node of the full method is phase 2), uses a
//! log-spaced ρ grid and a uniform ρ̇ grid inside the exact bound-orbit
//! interval, and places each orbit at the reference epoch (the observed
//! position advanced by the light time along the orbit's velocity).

use super::od::{gauss_geometry, observed_times_mjd, OrbitGeometry, GAUSS_OBLIQUITY_RAD};
use super::od_refine::{JacobianMethod, WhitenedFitConfig, WhitenedProblem};
use super::{PropagationError, PropagationResultValue, SphericalPredictor};
use crate::translation::OriginTranslationProvider;
use crate::types::time::TimeScaleProvider;
use crate::types::Frame;
use crate::{CoordinateBatch, ObserverBatch, OrbitBatch, OrbitId, OriginArray, OriginId};

/// Earth's gravitational parameter (au³/day², GM⊕ = 398600.4418 km³/s²): the
/// admissible region excludes orbits bound to the observer's planet.
pub const RANGING_EARTH_MU: f64 = 8.887_692_445_125_634e-10;

/// Settings of systematic ranging (defaults after Farnocchia, Chesley &
/// Micheli 2015, whose grids span the admissible region in log ρ and
/// uniform ρ̇ at comparable resolution).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangingConfig {
    /// Run ranging when the arc is too short for Gauss (`min_arc_length`)
    /// or no Gauss candidate converges.
    pub enabled: bool,
    /// Range grid (au), log-spaced: from 1e-4 au (`GAUSS_MIN_TOPOCENTRIC_
    /// RANGE_AU`, where the heliocentric two-body model stops applying) to
    /// 100 au.
    pub rho_min_au: f64,
    pub rho_max_au: f64,
    pub n_rho: usize,
    /// Range-rate nodes per range, uniform inside the bound-orbit interval.
    pub n_rho_dot: usize,
    /// Best ranging orbits corrected by the staged correction (in
    /// increasing χ²) when the arc is long enough to accept an orbit.
    pub max_corrections: usize,
    /// States scored per prediction crossing.
    pub chunk_size: usize,
    /// Longest arc (days) ranging runs on: 5 days, a few nights. Ranging is
    /// a short-arc method, and phase 1 holds the attributable fixed, which
    /// stops describing the arc within about a week (on an 11-day test arc
    /// the best node already sat a grid step off the truth); on a long arc
    /// it would also propagate every node across the whole arc.
    pub max_arc_days: f64,
}

impl Default for RangingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rho_min_au: adam_core_rs_orbit_determination::GAUSS_MIN_TOPOCENTRIC_RANGE_AU,
            rho_max_au: 100.0,
            n_rho: 60,
            n_rho_dot: 40,
            max_corrections: 3,
            chunk_size: 512,
            max_arc_days: 5.0,
        }
    }
}

/// The attributable of an arc at its reference observation: equatorial
/// position (degrees) and rates (degrees/day), and the observer's
/// heliocentric ecliptic state there.
#[derive(Debug, Clone, PartialEq)]
pub struct Attributable {
    /// Row of the reference observation (nearest the arc's midpoint).
    pub reference_row: usize,
    pub epoch: crate::Epoch,
    pub epoch_scale: crate::TimeScale,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub ra_rate_deg_per_day: f64,
    pub dec_rate_deg_per_day: f64,
    pub observer: [f64; 6],
    /// Arc length (days) the rates were fitted on.
    pub arc_days: f64,
}

/// One scored ranging orbit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangingSample {
    /// Heliocentric ecliptic state at the attributable's epoch.
    pub state: [f64; 6],
    pub rho_au: f64,
    pub rho_dot_au_per_day: f64,
    pub chi2: f64,
    /// `exp(-(chi2 - chi2_min) / 2)`, normalized over the sample.
    pub weight: f64,
}

/// Product of systematic ranging: the weighted orbit sample over the
/// admissible region.
#[derive(Debug, Clone, PartialEq)]
pub struct RangingResult {
    pub attributable: Attributable,
    /// Admissible, successfully scored samples, in grid order.
    pub samples: Vec<RangingSample>,
    /// Index of the lowest-χ² sample.
    pub best: Option<usize>,
    /// Grid nodes considered, and how many lay in the admissible region.
    pub grid_points: usize,
    pub admissible_points: usize,
    pub num_obs: usize,
}

impl RangingResult {
    pub fn best_sample(&self) -> Option<&RangingSample> {
        self.best.map(|index| &self.samples[index])
    }

    /// Indices of the samples in increasing χ² (grid order on ties).
    pub fn ranked(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.samples.len()).collect();
        order.sort_by(|&a, &b| {
            self.samples[a]
                .chi2
                .total_cmp(&self.samples[b].chi2)
                .then(a.cmp(&b))
        });
        order
    }
}

fn rotate_eq_to_ec(v: [f64; 3]) -> [f64; 3] {
    let (s, c) = GAUSS_OBLIQUITY_RAD.sin_cos();
    [v[0], c * v[1] + s * v[2], -s * v[1] + c * v[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Weighted least-squares polynomial (degree 1 to 3) through `(dt, y)`;
/// returns the value and first derivative at `dt = 0`.
fn value_and_rate(dt: &[f64], y: &[f64], w: &[f64], degree: usize) -> Option<(f64, f64)> {
    let k = degree + 1;
    let mut ata = nalgebra::DMatrix::<f64>::zeros(k, k);
    let mut atb = nalgebra::DVector::<f64>::zeros(k);
    for ((&t, &v), &weight) in dt.iter().zip(y).zip(w) {
        let basis: Vec<f64> = (0..k).map(|p| t.powi(p as i32)).collect();
        for i in 0..k {
            atb[i] += weight * basis[i] * v;
            for j in 0..k {
                ata[(i, j)] += weight * basis[i] * basis[j];
            }
        }
    }
    let solution = ata.lu().solve(&atb)?;
    Some((solution[0], solution[1]))
}

/// The attributable of `observed` (see [`Attributable`]): rates from a
/// weighted polynomial in time — cubic with six observations at four or
/// more distinct epochs, quadratic with four at three, linear otherwise —
/// at the observation nearest the arc's midpoint (earliest row on ties).
pub fn attributable(
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
) -> PropagationResultValue<Attributable> {
    let (directions, observer_states) = gauss_geometry(observed, observers)?;
    let times = observed_times_mjd(observed)?;
    let n = times.len();
    let mut epochs: Vec<f64> = times.clone();
    epochs.sort_by(f64::total_cmp);
    epochs.dedup();
    if epochs.len() < 2 {
        return Err(PropagationError::InvalidRequest(
            "ranging needs observations at two distinct epochs at least".to_string(),
        ));
    }
    let (t_min, t_max) = (epochs[0], epochs[epochs.len() - 1]);
    let midpoint = 0.5 * (t_min + t_max);
    let reference_row = (0..n)
        .min_by(|&a, &b| {
            (times[a] - midpoint)
                .abs()
                .total_cmp(&(times[b] - midpoint).abs())
                .then(a.cmp(&b))
        })
        .expect("non-empty");
    let t0 = times[reference_row];
    let covariance = observed
        .covariance
        .as_ref()
        .map(|c| c.values_row_major.as_slice());
    let sigma2 = |row: usize, index: usize| -> f64 {
        covariance
            .and_then(|values| values.get(row * 36 + index).copied())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0)
    };
    let ra0 = directions[reference_row][0];
    let dt: Vec<f64> = times.iter().map(|&t| t - t0).collect();
    let ra: Vec<f64> = directions
        .iter()
        .map(|d| (d[0] - ra0 + 540.0).rem_euclid(360.0) - 180.0)
        .collect();
    let dec: Vec<f64> = directions.iter().map(|d| d[1]).collect();
    let w_ra: Vec<f64> = (0..n).map(|row| 1.0 / sigma2(row, 7)).collect();
    let w_dec: Vec<f64> = (0..n).map(|row| 1.0 / sigma2(row, 14)).collect();
    let degree = if n >= 6 && epochs.len() >= 4 {
        3
    } else if n >= 4 && epochs.len() >= 3 {
        2
    } else {
        1
    };
    let fit = |y: &[f64], w: &[f64]| {
        value_and_rate(&dt, y, w, degree).ok_or_else(|| {
            PropagationError::InvalidRequest("ranging: singular attributable fit".to_string())
        })
    };
    let (ra_offset, ra_rate) = fit(&ra, &w_ra)?;
    let (dec_deg, dec_rate) = fit(&dec, &w_dec)?;
    let epoch_scale = observed.times.as_ref().map(|t| t.scale).ok_or_else(|| {
        PropagationError::InvalidRequest("ranging needs observation times".to_string())
    })?;
    let epoch = observed
        .times
        .as_ref()
        .map(|t| t.epochs[reference_row])
        .unwrap();
    Ok(Attributable {
        reference_row,
        epoch,
        epoch_scale,
        ra_deg: (ra0 + ra_offset).rem_euclid(360.0),
        dec_deg,
        ra_rate_deg_per_day: ra_rate,
        dec_rate_deg_per_day: dec_rate,
        observer: observer_states[reference_row],
        arc_days: t_max - t_min,
    })
}

/// Unit direction (ecliptic) and its time derivative (1/day) of an
/// attributable.
fn direction_and_rate(attributable: &Attributable) -> ([f64; 3], [f64; 3]) {
    let (a, d) = (
        attributable.ra_deg.to_radians(),
        attributable.dec_deg.to_radians(),
    );
    let (a_dot, d_dot) = (
        attributable.ra_rate_deg_per_day.to_radians(),
        attributable.dec_rate_deg_per_day.to_radians(),
    );
    let (sa, ca) = a.sin_cos();
    let (sd, cd) = d.sin_cos();
    let rho_hat = [cd * ca, cd * sa, sd];
    let d_alpha = [-cd * sa, cd * ca, 0.0];
    let d_delta = [-sd * ca, -sd * sa, cd];
    let rate = std::array::from_fn(|i| a_dot * d_alpha[i] + d_dot * d_delta[i]);
    (rotate_eq_to_ec(rho_hat), rotate_eq_to_ec(rate))
}

/// The bound-orbit range-rate interval at range `rho`: heliocentric energy
/// below zero is `|w + ρ̇ ρ̂|² < 2μ/|r|`, a quadratic in ρ̇; `None` when
/// no range rate gives a bound orbit there.
pub fn admissible_rho_dot_interval(
    attributable: &Attributable,
    rho: f64,
    mu: f64,
) -> Option<(f64, f64)> {
    let (rho_hat, rho_hat_dot) = direction_and_rate(attributable);
    let q = &attributable.observer;
    let r: [f64; 3] = std::array::from_fn(|i| q[i] + rho * rho_hat[i]);
    let w: [f64; 3] = std::array::from_fn(|i| q[3 + i] + rho * rho_hat_dot[i]);
    let b = dot(w, rho_hat);
    let discriminant = b * b - dot(w, w) + 2.0 * mu / dot(r, r).sqrt();
    if discriminant.is_nan() || discriminant <= 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    Some((-b - root, -b + root))
}

/// The heliocentric ecliptic state for range `rho` and range rate
/// `rho_dot` completing `attributable`, at the reference epoch: the observed
/// direction points to where the object was a light time `rho / c` earlier,
/// so the position is advanced by that light time along the velocity.
pub fn ranging_state(attributable: &Attributable, rho: f64, rho_dot: f64) -> [f64; 6] {
    let (rho_hat, rho_hat_dot) = direction_and_rate(attributable);
    let q = &attributable.observer;
    let v: [f64; 3] =
        std::array::from_fn(|i| q[3 + i] + rho_dot * rho_hat[i] + rho * rho_hat_dot[i]);
    let light_time = rho / super::od::IOD_DEFAULT_SPEED_OF_LIGHT;
    [
        q[0] + rho * rho_hat[0] + light_time * v[0],
        q[1] + rho * rho_hat[1] + light_time * v[1],
        q[2] + rho * rho_hat[2] + light_time * v[2],
        v[0],
        v[1],
        v[2],
    ]
}

/// Systematic ranging of `observed` scored through `predictor` (see the
/// module documentation). `mu` is the heliocentric gravitational
/// parameter (au³/day²).
#[allow(clippy::too_many_arguments)]
pub fn systematic_ranging_with<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    config: &RangingConfig,
    mu: f64,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<RangingResult> {
    if config.n_rho == 0
        || config.n_rho_dot == 0
        || config.rho_min_au.is_nan()
        || config.rho_max_au.is_nan()
        || config.rho_min_au <= 0.0
        || config.rho_max_au <= config.rho_min_au
    {
        return Err(PropagationError::InvalidRequest(
            "ranging needs n_rho, n_rho_dot > 0 and rho_max_au > rho_min_au > 0".to_string(),
        ));
    }
    let attributable = attributable(observed, observers)?;
    let (_, rho_hat_dot) = direction_and_rate(&attributable);
    let angular_rate2 = dot(rho_hat_dot, rho_hat_dot);
    let mut nodes: Vec<(f64, f64, [f64; 6])> = Vec::new();
    let mut grid_points = 0_usize;
    let log_min = config.rho_min_au.ln();
    let log_step = if config.n_rho > 1 {
        (config.rho_max_au.ln() - log_min) / (config.n_rho - 1) as f64
    } else {
        0.0
    };
    for i in 0..config.n_rho {
        let rho = (log_min + log_step * i as f64).exp();
        grid_points += config.n_rho_dot;
        let Some((lo, hi)) = admissible_rho_dot_interval(&attributable, rho, mu) else {
            continue;
        };
        let step = (hi - lo) / config.n_rho_dot as f64;
        for j in 0..config.n_rho_dot {
            let rho_dot = lo + (j as f64 + 0.5) * step;
            // Not bound to the observer's planet: positive two-body energy
            // about it.
            let planet_energy =
                0.5 * (rho_dot * rho_dot + rho * rho * angular_rate2) - RANGING_EARTH_MU / rho;
            if planet_energy < 0.0 {
                continue;
            }
            nodes.push((rho, rho_dot, ranging_state(&attributable, rho, rho_dot)));
        }
    }
    let admissible_points = nodes.len();
    let num_obs = observed.len();
    if nodes.is_empty() {
        return Ok(RangingResult {
            attributable,
            samples: Vec::new(),
            best: None,
            grid_points,
            admissible_points,
            num_obs,
        });
    }
    // One whitened problem at the reference epoch scores every node.
    let seed = OrbitBatch::new(
        vec![OrbitId("ranging".to_string())],
        vec![None],
        CoordinateBatch::cartesian(
            vec![nodes[0].2],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::from_code("SUN"), 1),
            Some(crate::TimeArray::new(
                attributable.epoch_scale,
                vec![attributable.epoch],
            )?),
            None,
        )?,
    )?;
    let _ = OrbitGeometry::from_orbit(&seed)?;
    let problem = WhitenedProblem::new(
        predictor,
        &seed,
        observed,
        observers,
        &WhitenedFitConfig {
            jacobian: JacobianMethod::Central,
            validate_covariance: false,
            ..WhitenedFitConfig::default()
        },
        provider,
        translation_provider,
    )?;
    let chi2_of = |residuals: &[f64]| -> f64 {
        let chi2: f64 = residuals.iter().map(|r| r * r).sum();
        if chi2.is_finite() {
            chi2
        } else {
            f64::INFINITY
        }
    };
    let mut chi2 = vec![f64::INFINITY; nodes.len()];
    for (start, chunk) in nodes
        .chunks(config.chunk_size.max(1))
        .enumerate()
        .map(|(k, c)| (k * config.chunk_size.max(1), c))
    {
        let states: Vec<[f64; 6]> = chunk.iter().map(|node| node.2).collect();
        match problem.residuals(&states)? {
            Ok(batch) => {
                for (offset, residuals) in batch.iter().enumerate() {
                    chi2[start + offset] = chi2_of(residuals);
                }
            }
            // A row failed somewhere in the chunk: score its states one by
            // one so the failure stays local to its node.
            Err(_) => {
                for (offset, state) in states.iter().enumerate() {
                    if let Ok(Ok(single)) = problem.residuals(std::slice::from_ref(state)) {
                        chi2[start + offset] = chi2_of(&single[0]);
                    }
                }
            }
        }
    }
    let minimum = chi2.iter().copied().fold(f64::INFINITY, f64::min);
    let mut samples: Vec<RangingSample> = nodes
        .iter()
        .zip(&chi2)
        .filter(|(_, c)| c.is_finite())
        .map(|((rho, rho_dot, state), &c)| RangingSample {
            state: *state,
            rho_au: *rho,
            rho_dot_au_per_day: *rho_dot,
            chi2: c,
            weight: (-(c - minimum) / 2.0).exp(),
        })
        .collect();
    let total: f64 = samples.iter().map(|s| s.weight).sum();
    if total > 0.0 {
        for sample in samples.iter_mut() {
            sample.weight /= total;
        }
    }
    let best = (0..samples.len())
        .min_by(|&a, &b| samples[a].chi2.total_cmp(&samples[b].chi2).then(a.cmp(&b)));
    Ok(RangingResult {
        attributable,
        samples,
        best,
        grid_points,
        admissible_points,
        num_obs,
    })
}
