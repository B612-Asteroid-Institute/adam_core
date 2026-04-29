//! Photometry kernels for H-G apparent magnitudes and solar phase angle.
//!
//! Mirrors `adam_core.photometry.magnitude._calculate_*_core_jax` exactly:
//!
//!   r                = ||object_pos||                      (heliocentric)
//!   delta            = ||object_pos - observer_pos||       (topocentric)
//!   observer_sun_dist= ||observer_pos||                    (observer→Sun)
//!   cos(alpha)       = (r² + delta² − observer_sun_dist²) / (2 · r · delta)
//!                      clamped to [-1, 1]
//!
//! Phase angle in degrees is recovered from `cos(alpha)` via a stable
//! half-angle identity (no direct `acos`):
//!
//!   alpha_rad = 2 · atan2(sqrt(max(0, 1 − cos)), sqrt(max(0, 1 + cos)))
//!
//! H-G apparent V-band magnitude reuses the same geometry:
//!
//!   tan(alpha/2) = sqrt((1 − cos) / (1 + cos))
//!   phi1         = exp(-3.33 · tan_half^0.63)
//!   phi2         = exp(-1.87 · tan_half^1.22)
//!   phase_fn     = (1 − G)·phi1 + G·phi2
//!   mag_v        = H_v + 5·log10(r·delta) − 2.5·log10(phase_fn)
//!
//! All kernels are f64-only (photometry is not part of `Ephemeris`'s
//! covariance tracking; no `Dual<6>` path is needed).

use rayon::prelude::*;

const RAD2DEG: f64 = 180.0_f64 / std::f64::consts::PI;

/// Rayon chunk size for batched photometry kernels. Per-row work is ~10
/// flops, so per-row `par_iter_mut` task dispatch dominates the math.
/// Chunking gives each worker a substantial batch — small n becomes one
/// chunk (effectively serial), large n distributes across cores.
const PHOT_CHUNK: usize = 1024;

#[inline]
fn row_geometry(obj: [f64; 3], obs: [f64; 3]) -> (f64, f64, f64, f64) {
    let r = (obj[0] * obj[0] + obj[1] * obj[1] + obj[2] * obj[2]).sqrt();
    let dx = obj[0] - obs[0];
    let dy = obj[1] - obs[1];
    let dz = obj[2] - obs[2];
    let delta = (dx * dx + dy * dy + dz * dz).sqrt();
    let obs_sun = (obs[0] * obs[0] + obs[1] * obs[1] + obs[2] * obs[2]).sqrt();
    let numer = r * r + delta * delta - obs_sun * obs_sun;
    let denom = 2.0 * r * delta;
    let cos_alpha = (numer / denom).clamp(-1.0, 1.0);
    (r, delta, obs_sun, cos_alpha)
}

#[inline]
fn cos_to_alpha_deg(cos_alpha: f64) -> f64 {
    // Stable conversion from cos(alpha) → alpha without `acos`; matches
    // `_calculate_phase_angle_core_jax` byte-for-byte.
    let y = (0.0_f64.max(1.0 - cos_alpha)).sqrt();
    let x = (0.0_f64.max(1.0 + cos_alpha)).sqrt();
    let alpha_rad = 2.0 * y.atan2(x);
    alpha_rad * RAD2DEG
}

#[inline]
fn cos_to_mag_v(h_v: f64, g: f64, r: f64, delta: f64, cos_phase: f64) -> f64 {
    // tan(phase/2) = sqrt((1 - cos) / (1 + cos)); the law-of-cosines clamp
    // guarantees the denominator > 0 for any finite, positive r & delta.
    let tan_half = ((1.0 - cos_phase) / (1.0 + cos_phase)).sqrt();
    let phi1 = (-3.33_f64 * tan_half.powf(0.63)).exp();
    let phi2 = (-1.87_f64 * tan_half.powf(1.22)).exp();
    let phase_fn = (1.0 - g) * phi1 + g * phi2;
    h_v + 5.0 * (r * delta).log10() - 2.5 * phase_fn.log10()
}

/// Batched solar phase angle in degrees.
///
/// Inputs are flattened `N × 3` heliocentric Cartesian positions (AU) —
/// caller's responsibility to transform to SUN origin before calling.
/// Invalid rows (non-finite or non-positive `r`/`delta`) yield NaN.
pub fn calculate_phase_angle_flat(object_pos: &[f64], observer_pos: &[f64]) -> Vec<f64> {
    assert_eq!(
        object_pos.len() % 3,
        0,
        "object_pos length must be a multiple of 3",
    );
    let n = object_pos.len() / 3;
    assert_eq!(
        observer_pos.len(),
        n * 3,
        "observer_pos must have the same length as object_pos",
    );

    let mut out = vec![0.0_f64; n];
    // Chunked rayon: per-row work is ~10 flops, so per-row `par_iter_mut`
    // task dispatch dominates the math at typical n. `par_chunks_mut(1024)`
    // gives each worker a substantial batch — small n becomes one chunk
    // (effectively serial, no overhead); large n distributes across cores.
    out.par_chunks_mut(PHOT_CHUNK)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base_i = ci * PHOT_CHUNK;
            for (k, dst) in chunk.iter_mut().enumerate() {
                let i = base_i + k;
                let base = i * 3;
                let obj = [object_pos[base], object_pos[base + 1], object_pos[base + 2]];
                let obs = [
                    observer_pos[base],
                    observer_pos[base + 1],
                    observer_pos[base + 2],
                ];
                let (r, delta, _, cos_alpha) = row_geometry(obj, obs);
                if !r.is_finite() || !delta.is_finite() || r <= 0.0 || delta <= 0.0 {
                    *dst = f64::NAN;
                } else {
                    *dst = cos_to_alpha_deg(cos_alpha);
                }
            }
        });
    out
}

/// Batched apparent V-band magnitude under the H-G phase function.
///
/// `h_v` and `g` are per-row (length `N`); positions are `N × 3`.
pub fn calculate_apparent_magnitude_v_flat(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
) -> Vec<f64> {
    assert_eq!(
        object_pos.len() % 3,
        0,
        "object_pos length must be a multiple of 3",
    );
    let n = object_pos.len() / 3;
    assert_eq!(
        observer_pos.len(),
        n * 3,
        "observer_pos must have the same length as object_pos",
    );
    assert_eq!(h_v.len(), n, "h_v must have length N");
    assert_eq!(g.len(), n, "g must have length N");

    let mut out = vec![0.0_f64; n];
    out.par_chunks_mut(PHOT_CHUNK)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base_i = ci * PHOT_CHUNK;
            for (k, dst) in chunk.iter_mut().enumerate() {
                let i = base_i + k;
                let base = i * 3;
                let obj = [object_pos[base], object_pos[base + 1], object_pos[base + 2]];
                let obs = [
                    observer_pos[base],
                    observer_pos[base + 1],
                    observer_pos[base + 2],
                ];
                let (r, delta, _, cos_phase) = row_geometry(obj, obs);
                if !r.is_finite() || !delta.is_finite() || r <= 0.0 || delta <= 0.0 {
                    *dst = f64::NAN;
                } else {
                    *dst = cos_to_mag_v(h_v[i], g[i], r, delta, cos_phase);
                }
            }
        });
    out
}

/// Fused batched (V-band magnitude, phase-angle deg). Faster than two
/// separate calls because the shared `row_geometry` runs once per row.
pub fn calculate_apparent_magnitude_v_and_phase_angle_flat(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        object_pos.len() % 3,
        0,
        "object_pos length must be a multiple of 3",
    );
    let n = object_pos.len() / 3;
    assert_eq!(
        observer_pos.len(),
        n * 3,
        "observer_pos must have the same length as object_pos",
    );
    assert_eq!(h_v.len(), n, "h_v must have length N");
    assert_eq!(g.len(), n, "g must have length N");

    let mut mag_out = vec![0.0_f64; n];
    let mut alpha_out = vec![0.0_f64; n];
    mag_out
        .par_chunks_mut(PHOT_CHUNK)
        .zip(alpha_out.par_chunks_mut(PHOT_CHUNK))
        .enumerate()
        .for_each(|(ci, (mag_chunk, alpha_chunk))| {
            let base_i = ci * PHOT_CHUNK;
            for (k, (mag_dst, alpha_dst)) in
                mag_chunk.iter_mut().zip(alpha_chunk.iter_mut()).enumerate()
            {
                let i = base_i + k;
                let base = i * 3;
                let obj = [object_pos[base], object_pos[base + 1], object_pos[base + 2]];
                let obs = [
                    observer_pos[base],
                    observer_pos[base + 1],
                    observer_pos[base + 2],
                ];
                let (r, delta, _, cos_phase) = row_geometry(obj, obs);
                if !r.is_finite() || !delta.is_finite() || r <= 0.0 || delta <= 0.0 {
                    *mag_dst = f64::NAN;
                    *alpha_dst = f64::NAN;
                } else {
                    *mag_dst = cos_to_mag_v(h_v[i], g[i], r, delta, cos_phase);
                    *alpha_dst = cos_to_alpha_deg(cos_phase);
                }
            }
        });
    (mag_out, alpha_out)
}

/// Fused H-G apparent V-band magnitude + per-row target-filter delta lookup.
///
/// Computes `mag_v + delta_table[target_ids[i]]` per row. Invalid geometry
/// rows (r ≤ 0, δ ≤ 0) produce NaN. Out-of-range target_ids produce NaN
/// (caller should pre-validate; NaN surfaces downstream).
pub fn predict_magnitudes_bandpass_flat(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    target_ids: &[i32],
    delta_table: &[f64],
) -> Vec<f64> {
    assert_eq!(
        object_pos.len() % 3,
        0,
        "object_pos length must be a multiple of 3",
    );
    let n = object_pos.len() / 3;
    assert_eq!(
        observer_pos.len(),
        n * 3,
        "observer_pos must have the same length as object_pos",
    );
    assert_eq!(h_v.len(), n, "h_v must have length N");
    assert_eq!(g.len(), n, "g must have length N");
    assert_eq!(target_ids.len(), n, "target_ids must have length N");

    let k = delta_table.len() as i32;
    let mut out = vec![0.0_f64; n];
    out.par_chunks_mut(PHOT_CHUNK)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base_i = ci * PHOT_CHUNK;
            for (kk, dst) in chunk.iter_mut().enumerate() {
                let i = base_i + kk;
                let base = i * 3;
                let obj = [object_pos[base], object_pos[base + 1], object_pos[base + 2]];
                let obs = [
                    observer_pos[base],
                    observer_pos[base + 1],
                    observer_pos[base + 2],
                ];
                let (r, delta, _, cos_phase) = row_geometry(obj, obs);
                let tid = target_ids[i];
                if !r.is_finite()
                    || !delta.is_finite()
                    || r <= 0.0
                    || delta <= 0.0
                    || tid < 0
                    || tid >= k
                {
                    *dst = f64::NAN;
                } else {
                    let mag_v = cos_to_mag_v(h_v[i], g[i], r, delta, cos_phase);
                    *dst = mag_v + delta_table[tid as usize];
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ([f64; 3], [f64; 3]) {
        // Asteroid-like heliocentric position + Earth-like observer.
        let object = [2.2, 0.5, 0.1];
        let observer = [1.0, -0.15, 0.0];
        (object, observer)
    }

    #[test]
    fn phase_angle_is_finite_and_reasonable() {
        let (obj, obs) = sample();
        let alpha = calculate_phase_angle_flat(&obj, &obs);
        assert_eq!(alpha.len(), 1);
        let a = alpha[0];
        assert!(a.is_finite());
        // Roughly Sun-object-observer geometry for an asteroid visible near
        // quadrature; phase angle should be well under 90°.
        assert!((0.0..90.0).contains(&a));
    }

    #[test]
    fn magnitude_matches_expected_scale() {
        let (obj, obs) = sample();
        let h_v = [18.0];
        let g = [0.15];
        let mag = calculate_apparent_magnitude_v_flat(&h_v, &obj, &obs, &g);
        assert_eq!(mag.len(), 1);
        // H_v=18 asteroid at r≈2.25, delta≈1.27 should be ~2-3 mag fainter
        // than absolute (5 log10(r·delta) term dominates).
        assert!((mag[0] - 18.0).abs() < 5.0);
        assert!(mag[0] > 18.0);
    }

    #[test]
    fn fused_matches_separate() {
        let (obj, obs) = sample();
        let h_v = [18.0];
        let g = [0.15];
        let alpha = calculate_phase_angle_flat(&obj, &obs);
        let mag = calculate_apparent_magnitude_v_flat(&h_v, &obj, &obs, &g);
        let (mag_fused, alpha_fused) =
            calculate_apparent_magnitude_v_and_phase_angle_flat(&h_v, &obj, &obs, &g);
        assert!((mag[0] - mag_fused[0]).abs() < 1e-15);
        assert!((alpha[0] - alpha_fused[0]).abs() < 1e-15);
    }

    #[test]
    fn invalid_geometry_emits_nan() {
        // r = 0 (object at Sun) is invalid.
        let obj = [0.0, 0.0, 0.0];
        let obs = [1.0, 0.0, 0.0];
        let alpha = calculate_phase_angle_flat(&obj, &obs);
        assert!(alpha[0].is_nan());
        let mag = calculate_apparent_magnitude_v_flat(&[18.0], &obj, &obs, &[0.15]);
        assert!(mag[0].is_nan());
    }

    #[test]
    fn batched_matches_per_row() {
        let (obj, obs) = sample();
        let obj_b: Vec<f64> = obj.iter().chain(obj.iter()).copied().collect();
        let obs_b: Vec<f64> = obs.iter().chain(obs.iter()).copied().collect();
        let h_v = [18.0, 18.0];
        let g = [0.15, 0.15];
        let mag = calculate_apparent_magnitude_v_flat(&h_v, &obj_b, &obs_b, &g);
        let ref_mag = calculate_apparent_magnitude_v_flat(&[18.0], &obj, &obs, &[0.15]);
        assert_eq!(mag.len(), 2);
        assert!((mag[0] - ref_mag[0]).abs() < 1e-15);
        assert!((mag[1] - ref_mag[0]).abs() < 1e-15);
    }
}
