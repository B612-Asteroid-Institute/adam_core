//! Independent review oracle for systematic-ranging posterior weights
//! (included into od_ranging.rs via #[path]).
use super::*;
use crate::types::{SchemaError, SchemaResult};
use crate::{CovarianceBatch, CovarianceUnits, Epoch, ObservatoryCode, TimeArray, TimeScale};

struct NoopProvider;
impl TimeScaleProvider for NoopProvider {
    fn rescale(&self, _t: &TimeArray, _s: TimeScale) -> SchemaResult<TimeArray> {
        Err(SchemaError::InvalidRecordBatch("no provider".to_string()))
    }
}
struct ZeroTranslation;
impl OriginTranslationProvider for ZeroTranslation {
    fn origin_translation_vectors(
        &self,
        origins: &OriginArray,
        _t: &OriginId,
        _f: Frame,
        _times: &TimeArray,
    ) -> SchemaResult<Vec<[f64; 6]>> {
        Ok(vec![[0.0; 6]; origins.len()])
    }
}

const SIGMA: f64 = 1.0 / 3600.0;

fn arc(observer_pos: [f64; 3]) -> (CoordinateBatch, ObserverBatch) {
    let t = [0.0, 0.1, 0.4, 0.5];
    let n = t.len();
    let times = TimeArray::new(
        TimeScale::Tdb,
        t.iter()
            .map(|&dt| Epoch::new(60_000, (dt * 86_400e9) as i64))
            .collect(),
    )
    .unwrap();
    let values: Vec<[f64; 6]> = t
        .iter()
        .map(|&dt| {
            [
                f64::NAN,
                40.0 + 0.25 * dt,
                10.0 + 0.08 * dt,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ]
        })
        .collect();
    let mut cov = vec![0.0; n * 36];
    for row in 0..n {
        cov[row * 36 + 7] = SIGMA * SIGMA;
        cov[row * 36 + 14] = SIGMA * SIGMA;
    }
    let observed = CoordinateBatch::new(
        crate::CoordinateValues::Spherical(values),
        Frame::Equatorial,
        OriginArray::repeat(OriginId::from_code("SUN"), n),
        Some(times.clone()),
        Some(
            CovarianceBatch::new(
                n,
                6,
                cov,
                CovarianceUnits::Coordinate(crate::CoordinateRepresentation::Spherical),
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let r = (observer_pos[0].powi(2) + observer_pos[1].powi(2)).sqrt();
    let speed = (0.000_295_912_208_3 / r).sqrt();
    let states: Vec<[f64; 6]> = t
        .iter()
        .map(|_| {
            [
                observer_pos[0],
                observer_pos[1],
                observer_pos[2],
                -speed * observer_pos[1] / r,
                speed * observer_pos[0] / r,
                0.0,
            ]
        })
        .collect();
    let observers = ObserverBatch::new(
        vec![ObservatoryCode("X05".to_string()); n],
        CoordinateBatch::cartesian(
            states,
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::from_code("SUN"), n),
            Some(times),
            None,
        )
        .unwrap(),
    )
    .unwrap();
    (observed, observers)
}

/// Predictor whose chi2 is an analytic function g(rho, rho_dot) of the
/// ranging node, recovered exactly from the state.
struct AnalyticPredictor {
    lonlat: Vec<[f64; 2]>,
    attributable: Attributable,
    g: Box<dyn Fn(f64, f64) -> f64>,
}

impl AnalyticPredictor {
    fn node(&self, s: &[f64; 6]) -> (f64, f64) {
        let (rho_hat, _) = direction_and_rate(&self.attributable);
        let q = &self.attributable.observer;
        let c = super::super::od::IOD_DEFAULT_SPEED_OF_LIGHT;
        let v = [s[3], s[4], s[5]];
        let u = [s[0] - q[0], s[1] - q[1], s[2] - q[2]];
        let d = [
            rho_hat[0] + v[0] / c,
            rho_hat[1] + v[1] / c,
            rho_hat[2] + v[2] / c,
        ];
        let rho = dot(u, d) / dot(d, d);
        let rho_dot = dot([v[0] - q[3], v[1] - q[4], v[2] - q[5]], rho_hat);
        (rho, rho_dot)
    }
}

impl SphericalPredictor for AnalyticPredictor {
    fn predict_spherical(
        &self,
        states: &[[f64; 6]],
        _geometry: &super::super::od::OrbitGeometry,
        observers: &ObserverBatch,
    ) -> PropagationResultValue<Result<Vec<f64>, String>> {
        let rows = observers.len();
        let mut out = Vec::with_capacity(states.len() * rows * 6);
        for s in states {
            let (rho, rho_dot) = self.node(s);
            let g = (self.g)(rho, rho_dot);
            for row in 0..rows {
                let [lon, lat] = self.lonlat[row];
                let lat = if row == 0 {
                    lat + g.sqrt() * SIGMA
                } else {
                    lat
                };
                out.extend_from_slice(&[1.0, lon, lat, 0.0, 0.0, 0.0]);
            }
        }
        Ok(Ok(out))
    }
}

const MU_SUN: f64 = 0.000_295_912_208_285_591_1;

struct Moments {
    mean_rho: f64,
    mean_rho_dot: f64,
    p_low: f64,
}

/// Independent oracle: uniform prior in physical (rho, rho_dot) truncated to
/// the admissible region, midpoint quadrature uniform in rho (not ln rho).
fn oracle(
    att: &Attributable,
    cfg: &RangingConfig,
    g: &dyn Fn(f64, f64) -> f64,
    split: f64,
) -> Moments {
    let (_, rho_hat_dot) = direction_and_rate(att);
    let w2 = dot(rho_hat_dot, rho_hat_dot);
    let (nr, nd) = (3000_usize, 3000_usize);
    let h = (cfg.rho_max_au - cfg.rho_min_au) / nr as f64;
    let (mut m0, mut m1, mut m2, mut ml) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..nr {
        let rho = cfg.rho_min_au + (i as f64 + 0.5) * h;
        let Some((lo, hi)) = admissible_rho_dot_interval(att, rho, MU_SUN) else {
            continue;
        };
        let k = (hi - lo) / nd as f64;
        for j in 0..nd {
            let rd = lo + (j as f64 + 0.5) * k;
            if 0.5 * (rd * rd + rho * rho * w2) - RANGING_EARTH_MU / rho < 0.0 {
                continue;
            }
            let w = (-0.5 * g(rho, rd)).exp() * h * k;
            m0 += w;
            m1 += w * rho;
            m2 += w * rd;
            if rho < split {
                ml += w;
            }
        }
    }
    Moments {
        mean_rho: m1 / m0,
        mean_rho_dot: m2 / m0,
        p_low: ml / m0,
    }
}

fn sample_moments(r: &RangingResult, split: f64, likelihood_only: bool) -> Moments {
    let min = r
        .samples
        .iter()
        .map(|s| s.chi2)
        .fold(f64::INFINITY, f64::min);
    let w: Vec<f64> = r
        .samples
        .iter()
        .map(|s| {
            if likelihood_only {
                (-(s.chi2 - min) / 2.0).exp()
            } else {
                s.weight
            }
        })
        .collect();
    let m0: f64 = w.iter().sum();
    let m1: f64 = r.samples.iter().zip(&w).map(|(s, w)| s.rho_au * w).sum();
    let m2: f64 = r
        .samples
        .iter()
        .zip(&w)
        .map(|(s, w)| s.rho_dot_au_per_day * w)
        .sum();
    let ml: f64 = r
        .samples
        .iter()
        .zip(&w)
        .filter(|(s, _)| s.rho_au < split)
        .map(|(_, w)| w)
        .sum();
    Moments {
        mean_rho: m1 / m0,
        mean_rho_dot: m2 / m0,
        p_low: ml / m0,
    }
}

fn run(
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    cfg: &RangingConfig,
    g: Box<dyn Fn(f64, f64) -> f64>,
) -> RangingResult {
    let att = attributable(observed, observers, &NoopProvider).unwrap();
    let lonlat = observed
        .values
        .raw_values()
        .iter()
        .map(|v| [v[1], v[2]])
        .collect();
    let p = AnalyticPredictor {
        lonlat,
        attributable: att,
        g,
    };
    systematic_ranging_with(
        &p,
        observed,
        observers,
        cfg,
        MU_SUN,
        &NoopProvider,
        &ZeroTranslation,
    )
    .unwrap()
}

#[test]
fn review_ranging_posterior_matches_independent_uniform_physical_oracle() {
    let (observed, observers) = arc([1.0, 0.0, 0.0]);
    let att = attributable(&observed, &observers, &NoopProvider).unwrap();
    let base = RangingConfig {
        rho_min_au: 0.2,
        rho_max_au: 3.0,
        ..RangingConfig::default()
    };
    let (lo0, hi0) = admissible_rho_dot_interval(&att, 1.0, MU_SUN).unwrap();
    let d0 = lo0 + 0.6 * (hi0 - lo0);
    let sd = 0.15 * (hi0 - lo0);
    type Chi2Shape = std::rc::Rc<dyn Fn(f64, f64) -> f64>;
    let shapes: Vec<(&str, Chi2Shape)> = vec![
        ("flat", std::rc::Rc::new(|_r, _d| 0.0)),
        (
            "gaussian",
            std::rc::Rc::new(move |r, d| ((r - 1.0) / 0.35).powi(2) + ((d - d0) / sd).powi(2)),
        ),
        (
            "range-only",
            std::rc::Rc::new(|r, _d| ((r - 0.6) / 0.4).powi(2)),
        ),
    ];
    let mut report = Vec::new();
    for (name, g) in shapes {
        let truth = oracle(&att, &base, &*g, 1.0);
        let mut errors = Vec::new();
        for (nr, nd) in [
            (15, 10),
            (30, 20),
            (60, 40),
            (120, 80),
            (240, 160),
            (480, 320),
            (960, 640),
        ] {
            let cfg = RangingConfig {
                n_rho: nr,
                n_rho_dot: nd,
                ..base
            };
            let g2 = g.clone();
            let r = run(&observed, &observers, &cfg, Box::new(move |a, b| g2(a, b)));
            // chi2 reported must equal g at the node (oracle wiring check).
            let g3 = g.clone();
            for s in r.samples.iter().take(50) {
                let expect = g3(s.rho_au, s.rho_dot_au_per_day);
                assert!(
                    (s.chi2 - expect).abs() <= 1e-6 * (1.0 + expect),
                    "{} vs {}",
                    s.chi2,
                    expect
                );
            }
            let total: f64 = r.samples.iter().map(|s| s.weight).sum();
            assert!((total - 1.0).abs() < 1e-12);
            let m = sample_moments(&r, 1.0, false);
            let old = sample_moments(&r, 1.0, true);
            let e = (
                (m.mean_rho - truth.mean_rho).abs() / truth.mean_rho,
                (m.mean_rho_dot - truth.mean_rho_dot).abs() / truth.mean_rho_dot.abs().max(1e-12),
                (m.p_low - truth.p_low).abs(),
            );
            report.push(format!(
                "{name} grid={nr}x{nd} new: E[rho]={:.6} E[rdot]={:.6e} P(rho<1)={:.4} | oracle: {:.6} {:.6e} {:.4} | rel.err rho={:.2e} rdot={:.2e} dP={:.2e} | likelihood-only E[rho]={:.6} P={:.4}",
                m.mean_rho, m.mean_rho_dot, m.p_low, truth.mean_rho, truth.mean_rho_dot, truth.p_low,
                e.0, e.1, e.2, old.mean_rho, old.p_low
            ));
            errors.push(e);
        }
        let last = errors.last().unwrap();
        assert!(last.0 < 5e-3 && last.1 < 2e-2, "{name}: {report:#?}");
    }
    eprintln!("{}", report.join("\n"));
}

#[test]
fn review_ranging_weights_are_stable_under_huge_chi2_offsets() {
    let (observed, observers) = arc([1.0, 0.0, 0.0]);
    let cfg = RangingConfig {
        rho_min_au: 0.2,
        rho_max_au: 3.0,
        n_rho: 30,
        n_rho_dot: 20,
        ..RangingConfig::default()
    };
    let a = run(
        &observed,
        &observers,
        &cfg,
        Box::new(|r, _| ((r - 1.0) / 0.3).powi(2)),
    );
    let b = run(
        &observed,
        &observers,
        &cfg,
        Box::new(|r, _| ((r - 1.0) / 0.3).powi(2) + 1.0e6),
    );
    assert_eq!(a.samples.len(), b.samples.len());
    for (x, y) in a.samples.iter().zip(&b.samples) {
        assert!(x.weight.is_finite() && y.weight.is_finite());
        assert!(
            (x.weight - y.weight).abs() <= 1e-6 * x.weight.max(1e-300),
            "{} {}",
            x.weight,
            y.weight
        );
    }
    // Deterministic ordering: ranked() is by chi2 then grid index.
    let ranked = a.ranked();
    for w in ranked.windows(2) {
        let (p, q) = (&a.samples[w[0]], &a.samples[w[1]]);
        assert!(p.chi2 < q.chi2 || (p.chi2 == q.chi2 && w[0] < w[1]));
    }
}

/// Policy evidence: the planet-bound exclusion uses Earth's GM around the
/// observer regardless of where the observer is.
#[test]
fn review_ranging_planet_exclusion_is_applied_in_supported_observer_context() {
    for pos in [[1.0, 0.0, 0.0]] {
        let (observed, observers) = arc(pos);
        let att = attributable(&observed, &observers, &NoopProvider).unwrap();
        let cfg = RangingConfig {
            rho_min_au: 1e-4,
            rho_max_au: 1e-2,
            n_rho: 40,
            n_rho_dot: 40,
            ..RangingConfig::default()
        };
        let r = run(&observed, &observers, &cfg, Box::new(|_, _| 0.0));
        let mut helio = 0;
        let log_min = cfg.rho_min_au.ln();
        let step = (cfg.rho_max_au.ln() - log_min) / (cfg.n_rho - 1) as f64;
        for i in 0..cfg.n_rho {
            let rho = (log_min + step * i as f64).exp();
            if admissible_rho_dot_interval(&att, rho, MU_SUN).is_some() {
                helio += cfg.n_rho_dot;
            }
        }
        eprintln!(
            "REVIEW-PLANET observer at {pos:?} au: heliocentric-admissible nodes={helio} kept={} excluded as Earth-bound={}",
            r.admissible_points,
            helio - r.admissible_points
        );
    }
}
