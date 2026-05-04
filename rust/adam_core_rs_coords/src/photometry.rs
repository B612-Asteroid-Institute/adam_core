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
const TWO_POINT_FIVE_OVER_LN10: f64 = 2.5_f64 / std::f64::consts::LN_10;

/// Rayon chunk size for batched photometry kernels. Per-row work is tiny,
/// so each worker gets a block of rows instead of one row.
const PHOT_CHUNK: usize = 1024;

/// Avoid Rayon scheduling overhead for small batches. The canonical baseline
/// gates (`parity_main --speed-n 2000` and `rust-parity-speed-cold`) currently
/// use n=2000, where two chunks add p95 jitter but do not amortize
/// work-stealing reliably. Re-sweep this threshold if the canonical gate size
/// rises above roughly 4k rows.
const PHOT_PHASE_SERIAL_THRESHOLD_ROWS: usize = 4096;

/// Switch to the tiled vForce/NEON pipeline once we have enough rows for the
/// batched transcendentals to amortize their per-call setup. Below this we
/// stay on the scalar Rayon path which already wins on n=2000 small-call
/// timings via lower per-row overhead.
#[cfg(target_os = "macos")]
const PHOT_VFORCE_MIN_ROWS: usize = 4096;

/// Tile size for the vForce/NEON path. Sized so that all per-tile scratch
/// buffers fit comfortably in L2 cache (~64 KB each × 5 buffers ≈ 320 KB) on
/// Apple Silicon while still amortizing vForce per-call dispatch costs.
#[cfg(target_os = "macos")]
const PHOT_VFORCE_TILE_ROWS: usize = 8192;

/// Phase-angle slow-path threshold: when the clamped `cos(α)` is closer than
/// `1 - PHOT_ENDPOINT_COS_ABS_MAX` to ±1, the half-angle identity gives
/// strictly better precision than `acos`, matching the legacy JAX reference.
/// Rows that hit this branch are recomputed via the scalar half-angle path.
#[cfg(target_os = "macos")]
const PHOT_ENDPOINT_COS_ABS_MAX: f64 = 0.999_999_99;

#[cfg(target_os = "macos")]
mod vforce {
    //! Thin Apple Accelerate vForce wrappers used by the tiled photometry
    //! pipeline. Each call dispatches a hand-tuned NEON kernel for the
    //! whole-slice transcendental, which is the structural lever that lets us
    //! beat the JAX vmap'd baseline at large workloads.
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn vvsqrt(out: *mut f64, input: *const f64, n: *const i32);
        fn vvacos(out: *mut f64, input: *const f64, n: *const i32);
        fn vvatan(out: *mut f64, input: *const f64, n: *const i32);
        fn vvexp(out: *mut f64, input: *const f64, n: *const i32);
        fn vvlog(out: *mut f64, input: *const f64, n: *const i32);
    }

    #[inline]
    fn len_i32(len: usize) -> i32 {
        i32::try_from(len).expect("vForce slice length must fit in i32")
    }

    #[inline]
    pub(super) fn sqrt_in_place(values: &mut [f64]) {
        let n = len_i32(values.len());
        // SAFETY: `values` is a unique mutable slice; vvsqrt reads and writes
        // exactly `n` doubles in place.
        unsafe { vvsqrt(values.as_mut_ptr(), values.as_ptr(), &n) };
    }

    #[inline]
    pub(super) fn acos(input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());
        let n = len_i32(input.len());
        // SAFETY: input/output have the same length; vvacos reads `n` from
        // input and writes `n` to output.
        unsafe { vvacos(output.as_mut_ptr(), input.as_ptr(), &n) };
    }

    #[inline]
    pub(super) fn atan_in_place(values: &mut [f64]) {
        let n = len_i32(values.len());
        // SAFETY: see `sqrt_in_place`.
        unsafe { vvatan(values.as_mut_ptr(), values.as_ptr(), &n) };
    }

    #[inline]
    pub(super) fn exp_in_place(values: &mut [f64]) {
        let n = len_i32(values.len());
        // SAFETY: see `sqrt_in_place`.
        unsafe { vvexp(values.as_mut_ptr(), values.as_ptr(), &n) };
    }

    #[inline]
    pub(super) fn ln_in_place(values: &mut [f64]) {
        let n = len_i32(values.len());
        // SAFETY: see `sqrt_in_place`.
        unsafe { vvlog(values.as_mut_ptr(), values.as_ptr(), &n) };
    }
}

#[inline]
fn load3(flat: &[f64], i: usize) -> [f64; 3] {
    let base = i * 3;
    [flat[base], flat[base + 1], flat[base + 2]]
}

#[inline]
fn invalid_geometry(r: f64, delta: f64) -> bool {
    !r.is_finite() || !delta.is_finite() || r <= 0.0 || delta <= 0.0
}

#[inline]
fn row_geometry(obj: [f64; 3], obs: [f64; 3]) -> (f64, f64, f64) {
    // The JAX reference uses (r² + δ² − obs_sun²)/(2 r δ). Algebraic identity:
    //     r² + δ² − obs_sun² = 2(r² − obj·obs)
    // The factor of 2 cancels the 2 in the denominator, leaving
    //     cosα = (r² − obj·obs) / (r · δ),
    // which keeps Rust bit-parity with the legacy formula while saving a multiply.
    let r = (obj[0] * obj[0] + obj[1] * obj[1] + obj[2] * obj[2]).sqrt();
    let dx = obj[0] - obs[0];
    let dy = obj[1] - obs[1];
    let dz = obj[2] - obs[2];
    let delta = (dx * dx + dy * dy + dz * dz).sqrt();
    let obs_sun = (obs[0] * obs[0] + obs[1] * obs[1] + obs[2] * obs[2]).sqrt();
    let numer = r * r + delta * delta - obs_sun * obs_sun;
    let denom = 2.0 * r * delta;
    let cos_alpha = (numer / denom).clamp(-1.0, 1.0);
    (r, delta, cos_alpha)
}

#[inline]
fn cos_to_half_angle_terms(cos_alpha: f64) -> (f64, f64) {
    (
        (0.0_f64.max(1.0 - cos_alpha)).sqrt(),
        (0.0_f64.max(1.0 + cos_alpha)).sqrt(),
    )
}

#[inline]
fn half_angle_terms_to_alpha_deg(y: f64, x: f64) -> f64 {
    2.0 * y.atan2(x) * RAD2DEG
}

#[inline]
fn cos_to_alpha_deg(cos_alpha: f64) -> f64 {
    // Stable conversion from cos(alpha) → alpha without `acos`; matches
    // `_calculate_phase_angle_core_jax` and preserves endpoint precision.
    let (y, x) = cos_to_half_angle_terms(cos_alpha);
    half_angle_terms_to_alpha_deg(y, x)
}

#[inline]
fn mag_v_from_tan_half(h_v: f64, g: f64, r: f64, delta: f64, tan_half: f64) -> f64 {
    let phi1 = (-3.33_f64 * tan_half.powf(0.63)).exp();
    let phi2 = (-1.87_f64 * tan_half.powf(1.22)).exp();
    let phase_fn = (1.0 - g) * phi1 + g * phi2;
    h_v + 5.0 * (r * delta).log10() - 2.5 * phase_fn.log10()
}

#[inline]
fn cos_to_mag_v(h_v: f64, g: f64, r: f64, delta: f64, cos_phase: f64) -> f64 {
    // tan(phase/2) = sqrt((1 - cos) / (1 + cos)); the clamp guarantees the
    // denominator > 0 for finite, positive geometry except the physical 180°
    // limit, where this naturally becomes +inf.
    let tan_half = ((1.0 - cos_phase) / (1.0 + cos_phase)).sqrt();
    mag_v_from_tan_half(h_v, g, r, delta, tan_half)
}

#[inline]
fn phase_angle_row(object_pos: &[f64], observer_pos: &[f64], i: usize) -> f64 {
    let obj = load3(object_pos, i);
    let obs = load3(observer_pos, i);
    let (r, delta, cos_alpha) = row_geometry(obj, obs);
    if invalid_geometry(r, delta) {
        f64::NAN
    } else {
        cos_to_alpha_deg(cos_alpha)
    }
}

#[inline]
fn apparent_magnitude_v_row(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    i: usize,
) -> f64 {
    let obj = load3(object_pos, i);
    let obs = load3(observer_pos, i);
    let (r, delta, cos_phase) = row_geometry(obj, obs);
    if invalid_geometry(r, delta) {
        f64::NAN
    } else {
        cos_to_mag_v(h_v[i], g[i], r, delta, cos_phase)
    }
}

#[inline]
fn apparent_magnitude_v_and_phase_angle_row(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    i: usize,
) -> (f64, f64) {
    let obj = load3(object_pos, i);
    let obs = load3(observer_pos, i);
    let (r, delta, cos_phase) = row_geometry(obj, obs);
    if invalid_geometry(r, delta) {
        (f64::NAN, f64::NAN)
    } else {
        let (y, x) = cos_to_half_angle_terms(cos_phase);
        let tan_half = y / x;
        (
            mag_v_from_tan_half(h_v[i], g[i], r, delta, tan_half),
            half_angle_terms_to_alpha_deg(y, x),
        )
    }
}

#[inline]
fn predict_magnitude_row(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    target_ids: &[i32],
    delta_table: &[f64],
    i: usize,
) -> f64 {
    let obj = load3(object_pos, i);
    let obs = load3(observer_pos, i);
    let (r, delta, cos_phase) = row_geometry(obj, obs);
    let tid = target_ids[i];
    if invalid_geometry(r, delta) || tid < 0 || tid >= delta_table.len() as i32 {
        f64::NAN
    } else {
        cos_to_mag_v(h_v[i], g[i], r, delta, cos_phase) + delta_table[tid as usize]
    }
}

// =============================================================================
// macOS vForce/NEON tiled photometry pipeline
// -----------------------------------------------------------------------------
// Replaces the per-row scalar `pow`/`acos` chain with a whole-tile pipeline of
// hand-tuned NEON glue (squared geometry, clamps, finishing FMAs) and Apple
// vForce transcendentals (vvsqrt, vvacos, vvexp, vvlog, vvatan).
//
// Structural wins versus the scalar JAX-vmap'd legacy:
//   * Compute tan²(α/2) directly from `(numer, denom)` without the leading
//     `sqrt((1−cos)/(1+cos))`.
//   * Replace `tan_half.powf(0.63)` and `.powf(1.22)` with
//     `exp(0.315·ln(tan_half_sq))` and `exp(0.61·ln(tan_half_sq))`, batching
//     all log/exp calls through vForce.
//   * For the phase-angle kernel use `vvacos(cos)` directly (1 transcendental
//     per row) and only fall back to the half-angle reference for rows whose
//     clamped `cos` lies in the slow-path band near ±1.
//   * Fuse `predict_magnitudes`'s bandpass-delta lookup into the final
//     magnitude write so it costs an indexed load + FMA per NEON pair instead
//     of a separate pass.
// =============================================================================

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fill_geometry_squared_terms_tile(
    object_pos: &[f64],
    observer_pos: &[f64],
    row_offset: usize,
    r_sq: &mut [f64],
    delta_sq: &mut [f64],
    numer: &mut [f64],
) {
    use core::arch::aarch64::*;

    let mut k = 0;
    while k + 2 <= r_sq.len() {
        let i = row_offset + k;
        // SAFETY: callers pass tile slices sized within the asserted N×3
        // object/observer rows; vld3q_f64 reads two interleaved xyz rows.
        unsafe {
            let obj = vld3q_f64(object_pos.as_ptr().add(i * 3));
            let obs = vld3q_f64(observer_pos.as_ptr().add(i * 3));

            let dx = vsubq_f64(obj.0, obs.0);
            let dy = vsubq_f64(obj.1, obs.1);
            let dz = vsubq_f64(obj.2, obs.2);

            let r_sq_v = vfmaq_f64(
                vfmaq_f64(vmulq_f64(obj.0, obj.0), obj.1, obj.1),
                obj.2,
                obj.2,
            );
            let delta_sq_v = vfmaq_f64(vfmaq_f64(vmulq_f64(dx, dx), dy, dy), dz, dz);
            let dot_v = vfmaq_f64(
                vfmaq_f64(vmulq_f64(obj.0, obs.0), obj.1, obs.1),
                obj.2,
                obs.2,
            );
            // numer = 2*(r² − dot); the matching `denom = 2 r δ` is computed
            // separately, and the factor of 2 cancels in cos = numer/denom.
            let numer_v = vsubq_f64(r_sq_v, dot_v);

            vst1q_f64(r_sq.as_mut_ptr().add(k), r_sq_v);
            vst1q_f64(delta_sq.as_mut_ptr().add(k), delta_sq_v);
            vst1q_f64(numer.as_mut_ptr().add(k), numer_v);
        }
        k += 2;
    }
    while k < r_sq.len() {
        let i = row_offset + k;
        let base = i * 3;
        let ox = object_pos[base];
        let oy = object_pos[base + 1];
        let oz = object_pos[base + 2];
        let bx = observer_pos[base];
        let by = observer_pos[base + 1];
        let bz = observer_pos[base + 2];
        let dx = ox - bx;
        let dy = oy - by;
        let dz = oz - bz;
        let r_sq_v = ox * ox + oy * oy + oz * oz;
        let delta_sq_v = dx * dx + dy * dy + dz * dz;
        let dot = ox * bx + oy * by + oz * bz;
        r_sq[k] = r_sq_v;
        delta_sq[k] = delta_sq_v;
        numer[k] = r_sq_v - dot;
        k += 1;
    }
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn fill_geometry_squared_terms_tile(
    object_pos: &[f64],
    observer_pos: &[f64],
    row_offset: usize,
    r_sq: &mut [f64],
    delta_sq: &mut [f64],
    numer: &mut [f64],
) {
    for k in 0..r_sq.len() {
        let i = row_offset + k;
        let base = i * 3;
        let ox = object_pos[base];
        let oy = object_pos[base + 1];
        let oz = object_pos[base + 2];
        let bx = observer_pos[base];
        let by = observer_pos[base + 1];
        let bz = observer_pos[base + 2];
        let dx = ox - bx;
        let dy = oy - by;
        let dz = oz - bz;
        let r_sq_v = ox * ox + oy * oy + oz * oz;
        let delta_sq_v = dx * dx + dy * dy + dz * dz;
        let dot = ox * bx + oy * by + oz * bz;
        r_sq[k] = r_sq_v;
        delta_sq[k] = delta_sq_v;
        numer[k] = r_sq_v - dot;
    }
}

/// Combine `r_sq` and `delta_sq` into `rd_sq = r²·δ²` (left in `r_sq`) and
/// `denom_sq = rd_sq` (left in `delta_sq`) so a single `vvsqrt` over the
/// `delta_sq` slice produces `denom = r·δ`. Splitting into two slots lets
/// us preserve `rd_sq` for the final `mag = h_v + (5/2 ln10)·ln(rd_sq/phase_fn)`
/// step without re-multiplying.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fill_rd_sq_and_denom_sq(r_sq: &mut [f64], delta_sq: &mut [f64]) {
    use core::arch::aarch64::*;
    let mut k = 0;
    while k + 2 <= r_sq.len() {
        // SAFETY: equal-length slices stepped two doubles at a time.
        unsafe {
            let r_v = vld1q_f64(r_sq.as_ptr().add(k));
            let d_v = vld1q_f64(delta_sq.as_ptr().add(k));
            let rd = vmulq_f64(r_v, d_v);
            vst1q_f64(r_sq.as_mut_ptr().add(k), rd);
            vst1q_f64(delta_sq.as_mut_ptr().add(k), rd);
        }
        k += 2;
    }
    while k < r_sq.len() {
        let rd = r_sq[k] * delta_sq[k];
        r_sq[k] = rd;
        delta_sq[k] = rd;
        k += 1;
    }
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn fill_rd_sq_and_denom_sq(r_sq: &mut [f64], delta_sq: &mut [f64]) {
    for k in 0..r_sq.len() {
        let rd = r_sq[k] * delta_sq[k];
        r_sq[k] = rd;
        delta_sq[k] = rd;
    }
}

/// Clamp `cos = numer / denom` to `[-1, 1]` and detect the slow-path band
/// `|cos| > PHOT_ENDPOINT_COS_ABS_MAX` and any non-finite raw cos. Returns
/// `true` if any row needs the half-angle slow-path fix-up.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn clamp_cos_in_place_detect_endpoint(numer: &mut [f64], denom: &[f64]) -> bool {
    use core::arch::aarch64::*;
    let mut k = 0;
    let one = unsafe { vdupq_n_f64(1.0) };
    let neg_one = unsafe { vdupq_n_f64(-1.0) };
    let endpoint_threshold = unsafe { vdupq_n_f64(PHOT_ENDPOINT_COS_ABS_MAX) };
    let max_finite = unsafe { vdupq_n_f64(f64::MAX) };
    let all_bits = unsafe { vdupq_n_u64(u64::MAX) };
    let mut bad = unsafe { vdupq_n_u64(0) };
    while k + 2 <= numer.len() {
        // SAFETY: numer/denom equal length, two-lane stride.
        unsafe {
            let nu = vld1q_f64(numer.as_ptr().add(k));
            let de = vld1q_f64(denom.as_ptr().add(k));
            let raw = vdivq_f64(nu, de);
            let clamped = vmaxq_f64(neg_one, vminq_f64(one, raw));
            vst1q_f64(numer.as_mut_ptr().add(k), clamped);

            let raw_finite = vcleq_f64(vabsq_f64(raw), max_finite);
            let endpoint = vcgtq_f64(vabsq_f64(clamped), endpoint_threshold);
            let row_bad = vorrq_u64(veorq_u64(raw_finite, all_bits), endpoint);
            bad = vorrq_u64(bad, row_bad);
        }
        k += 2;
    }
    let mut needs_slow = unsafe { (vgetq_lane_u64::<0>(bad) | vgetq_lane_u64::<1>(bad)) != 0 };
    while k < numer.len() {
        let raw = numer[k] / denom[k];
        let clamped = raw.clamp(-1.0, 1.0);
        numer[k] = clamped;
        needs_slow |= !raw.is_finite() || clamped.abs() > PHOT_ENDPOINT_COS_ABS_MAX;
        k += 1;
    }
    needs_slow
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn clamp_cos_in_place_detect_endpoint(numer: &mut [f64], denom: &[f64]) -> bool {
    let mut needs_slow = false;
    for k in 0..numer.len() {
        let raw = numer[k] / denom[k];
        let clamped = raw.clamp(-1.0, 1.0);
        numer[k] = clamped;
        needs_slow |= !raw.is_finite() || clamped.abs() > PHOT_ENDPOINT_COS_ABS_MAX;
    }
    needs_slow
}

/// Compute `tan²(α/2) = (1−cos)/(1+cos)` directly from `(numer, denom)` with
/// the same algebraic clamp the JAX path uses on `cos`. This skips the
/// `sqrt` that the legacy formula does on `(1−cos)/(1+cos)` before raising
/// to fractional powers.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fill_tan_half_sq_from_numer_denom(numer: &mut [f64], denom: &[f64]) {
    use core::arch::aarch64::*;
    let mut k = 0;
    while k + 2 <= numer.len() {
        // SAFETY: equal-length slices, two-lane stride.
        unsafe {
            let nu = vld1q_f64(numer.as_ptr().add(k));
            let de = vld1q_f64(denom.as_ptr().add(k));
            // clamp(numer, -denom, +denom) matches clamp(cos, ±1) since denom = r·δ ≥ 0.
            let clamped = vmaxq_f64(vnegq_f64(de), vminq_f64(de, nu));
            let tan_sq = vdivq_f64(vsubq_f64(de, clamped), vaddq_f64(de, clamped));
            vst1q_f64(numer.as_mut_ptr().add(k), tan_sq);
        }
        k += 2;
    }
    while k < numer.len() {
        let de = denom[k];
        let clamped = numer[k].clamp(-de, de);
        numer[k] = (de - clamped) / (de + clamped);
        k += 1;
    }
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn fill_tan_half_sq_from_numer_denom(numer: &mut [f64], denom: &[f64]) {
    for k in 0..numer.len() {
        let de = denom[k];
        let clamped = numer[k].clamp(-de, de);
        numer[k] = (de - clamped) / (de + clamped);
    }
}

/// Detect any row whose `rd_sq` is non-finite or non-positive so the magnitude
/// kernel can stamp NaN at the end. Cheap because rd_sq is already computed.
#[cfg(target_os = "macos")]
fn rd_sq_has_invalid(rd_sq: &[f64]) -> bool {
    rd_sq.iter().any(|&v| !v.is_finite() || v <= 0.0)
}

/// `phi1 = 0.315 * ln_tan_sq`; `phi2 = 0.61 * ln_tan_sq`. Multiplications by
/// `0.5 * 0.63` and `0.5 * 1.22` fold the `tan_half = sqrt(tan_half_sq)`
/// raise-to-power into the surrounding `vvln`/`vvexp` pair.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fill_phi_logs_from_ln_tan_sq(ln_tan_sq: &[f64], phi1: &mut [f64], phi2: &mut [f64]) {
    use core::arch::aarch64::*;
    let mut k = 0;
    let s1 = unsafe { vdupq_n_f64(0.315) };
    let s2 = unsafe { vdupq_n_f64(0.61) };
    while k + 2 <= ln_tan_sq.len() {
        // SAFETY: equal-length slices, two-lane stride.
        unsafe {
            let lt = vld1q_f64(ln_tan_sq.as_ptr().add(k));
            vst1q_f64(phi1.as_mut_ptr().add(k), vmulq_f64(s1, lt));
            vst1q_f64(phi2.as_mut_ptr().add(k), vmulq_f64(s2, lt));
        }
        k += 2;
    }
    while k < ln_tan_sq.len() {
        phi1[k] = 0.315_f64 * ln_tan_sq[k];
        phi2[k] = 0.61_f64 * ln_tan_sq[k];
        k += 1;
    }
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn fill_phi_logs_from_ln_tan_sq(ln_tan_sq: &[f64], phi1: &mut [f64], phi2: &mut [f64]) {
    for k in 0..ln_tan_sq.len() {
        phi1[k] = 0.315_f64 * ln_tan_sq[k];
        phi2[k] = 0.61_f64 * ln_tan_sq[k];
    }
}

/// Scale the `phi1`/`phi2` buffers by `−3.33` and `−1.87` ahead of the second
/// `vvexp` pass.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn scale_phi_outer(phi1: &mut [f64], phi2: &mut [f64]) {
    use core::arch::aarch64::*;
    let mut k = 0;
    let s1 = unsafe { vdupq_n_f64(-3.33) };
    let s2 = unsafe { vdupq_n_f64(-1.87) };
    while k + 2 <= phi1.len() {
        // SAFETY: equal-length slices, two-lane stride.
        unsafe {
            let p1 = vld1q_f64(phi1.as_ptr().add(k));
            let p2 = vld1q_f64(phi2.as_ptr().add(k));
            vst1q_f64(phi1.as_mut_ptr().add(k), vmulq_f64(s1, p1));
            vst1q_f64(phi2.as_mut_ptr().add(k), vmulq_f64(s2, p2));
        }
        k += 2;
    }
    while k < phi1.len() {
        phi1[k] *= -3.33_f64;
        phi2[k] *= -1.87_f64;
        k += 1;
    }
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn scale_phi_outer(phi1: &mut [f64], phi2: &mut [f64]) {
    for k in 0..phi1.len() {
        phi1[k] *= -3.33_f64;
        phi2[k] *= -1.87_f64;
    }
}

/// `log_arg[k] = rd_sq[k] / ((1−g[i]) * phi1[k] + g[i] * phi2[k])`. Folded
/// into a single NEON FMA + divide pair so the whole step is two vector ops
/// per pair plus one indexed `g` load.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fill_mag_log_arg(
    g: &[f64],
    row_offset: usize,
    rd_sq: &[f64],
    phi1: &[f64],
    phi2: &[f64],
    log_arg: &mut [f64],
) {
    use core::arch::aarch64::*;
    let mut k = 0;
    let one = unsafe { vdupq_n_f64(1.0) };
    while k + 2 <= log_arg.len() {
        // SAFETY: equal-length slices, two-lane stride.
        unsafe {
            let g_v = vld1q_f64(g.as_ptr().add(row_offset + k));
            let p1 = vld1q_f64(phi1.as_ptr().add(k));
            let p2 = vld1q_f64(phi2.as_ptr().add(k));
            let rd = vld1q_f64(rd_sq.as_ptr().add(k));
            let phase_fn = vfmaq_f64(vmulq_f64(vsubq_f64(one, g_v), p1), g_v, p2);
            vst1q_f64(log_arg.as_mut_ptr().add(k), vdivq_f64(rd, phase_fn));
        }
        k += 2;
    }
    while k < log_arg.len() {
        let i = row_offset + k;
        let phase_fn = (1.0 - g[i]) * phi1[k] + g[i] * phi2[k];
        log_arg[k] = rd_sq[k] / phase_fn;
        k += 1;
    }
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn fill_mag_log_arg(
    g: &[f64],
    row_offset: usize,
    rd_sq: &[f64],
    phi1: &[f64],
    phi2: &[f64],
    log_arg: &mut [f64],
) {
    for k in 0..log_arg.len() {
        let i = row_offset + k;
        let phase_fn = (1.0 - g[i]) * phi1[k] + g[i] * phi2[k];
        log_arg[k] = rd_sq[k] / phase_fn;
    }
}

/// Final magnitude write: `out[k] = h_v[i] + (2.5/ln10) * log_arg[k]`,
/// vectorized as a NEON FMA so the most-frequent finishing step costs one
/// vector load + FMA + store per pair.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn finish_magnitude_neon(h_v: &[f64], row_offset: usize, log_arg: &[f64], out: &mut [f64]) {
    use core::arch::aarch64::*;
    let mut k = 0;
    let scale = unsafe { vdupq_n_f64(TWO_POINT_FIVE_OVER_LN10) };
    while k + 2 <= out.len() {
        // SAFETY: equal-length slices, two-lane stride.
        unsafe {
            let h = vld1q_f64(h_v.as_ptr().add(row_offset + k));
            let la = vld1q_f64(log_arg.as_ptr().add(k));
            vst1q_f64(out.as_mut_ptr().add(k), vfmaq_f64(h, scale, la));
        }
        k += 2;
    }
    while k < out.len() {
        let i = row_offset + k;
        out[k] = h_v[i] + TWO_POINT_FIVE_OVER_LN10 * log_arg[k];
        k += 1;
    }
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn finish_magnitude_neon(h_v: &[f64], row_offset: usize, log_arg: &[f64], out: &mut [f64]) {
    for k in 0..out.len() {
        let i = row_offset + k;
        out[k] = h_v[i] + TWO_POINT_FIVE_OVER_LN10 * log_arg[k];
    }
}

/// Build the full magnitude pipeline for a single tile: consume
/// `tan_half_sq` and `rd_sq`, leave `out` filled with V-band magnitudes.
/// Invalid rows are NaN-stamped at the end. Used by both the magnitude and
/// fused entry points.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn magnitude_pipeline_tile(
    h_v: &[f64],
    g: &[f64],
    row_offset: usize,
    rd_sq: &[f64],
    tan_half_sq: &mut [f64],
    phi1: &mut [f64],
    phi2: &mut [f64],
    out: &mut [f64],
    has_invalid: bool,
) {
    let m = out.len();
    vforce::ln_in_place(tan_half_sq);
    fill_phi_logs_from_ln_tan_sq(tan_half_sq, phi1, phi2);
    vforce::exp_in_place(&mut phi1[..m]);
    vforce::exp_in_place(&mut phi2[..m]);
    scale_phi_outer(&mut phi1[..m], &mut phi2[..m]);
    vforce::exp_in_place(&mut phi1[..m]);
    vforce::exp_in_place(&mut phi2[..m]);

    // Reuse `tan_half_sq` as the log-arg scratch; its prior contents
    // (ln_tan_sq) have been consumed by the phi pipeline above.
    fill_mag_log_arg(g, row_offset, rd_sq, phi1, phi2, &mut tan_half_sq[..m]);
    vforce::ln_in_place(&mut tan_half_sq[..m]);
    finish_magnitude_neon(h_v, row_offset, &tan_half_sq[..m], out);

    if has_invalid {
        for k in 0..m {
            if !rd_sq[k].is_finite() || rd_sq[k] <= 0.0 {
                out[k] = f64::NAN;
            }
        }
    }
}

/// Phase-angle finishing: `vvacos` over the clamped `cos` slice gives radians;
/// scale to degrees and replace endpoint-band rows with the JAX half-angle
/// reference for the precision the legacy gate expects.
#[cfg(target_os = "macos")]
fn finish_phase_angle_tile(
    object_pos: &[f64],
    observer_pos: &[f64],
    row_offset: usize,
    cos_phase: &[f64],
    rd_sq: &[f64],
    out_tile: &mut [f64],
    needs_slow: bool,
) {
    vforce::acos(cos_phase, out_tile);
    if needs_slow {
        for k in 0..out_tile.len() {
            if !rd_sq[k].is_finite() || rd_sq[k] <= 0.0 {
                out_tile[k] = f64::NAN;
                continue;
            }
            if cos_phase[k].abs() > PHOT_ENDPOINT_COS_ABS_MAX {
                let i = row_offset + k;
                let obj = load3(object_pos, i);
                let obs = load3(observer_pos, i);
                let (_, _, cos_alpha) = row_geometry(obj, obs);
                out_tile[k] = cos_to_alpha_deg(cos_alpha);
            } else {
                out_tile[k] *= RAD2DEG;
            }
        }
    } else {
        for v in out_tile.iter_mut() {
            *v *= RAD2DEG;
        }
    }
}

#[cfg(target_os = "macos")]
fn calculate_phase_angle_macos(object_pos: &[f64], observer_pos: &[f64], out: &mut [f64]) {
    let n = out.len();
    let mut rd_sq = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut denom = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut cos_phase = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    for row_offset in (0..n).step_by(PHOT_VFORCE_TILE_ROWS) {
        let m = PHOT_VFORCE_TILE_ROWS.min(n - row_offset);
        let rd_tile = &mut rd_sq[..m];
        let denom_tile = &mut denom[..m];
        let cos_tile = &mut cos_phase[..m];
        let out_tile = &mut out[row_offset..row_offset + m];
        fill_geometry_squared_terms_tile(
            object_pos,
            observer_pos,
            row_offset,
            rd_tile,
            denom_tile,
            cos_tile,
        );
        fill_rd_sq_and_denom_sq(rd_tile, denom_tile);
        vforce::sqrt_in_place(denom_tile);
        let needs_slow = clamp_cos_in_place_detect_endpoint(cos_tile, denom_tile);
        finish_phase_angle_tile(
            object_pos,
            observer_pos,
            row_offset,
            cos_tile,
            rd_tile,
            out_tile,
            needs_slow,
        );
    }
}

#[cfg(target_os = "macos")]
fn calculate_apparent_magnitude_v_macos(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    out: &mut [f64],
) {
    let n = out.len();
    let mut rd_sq = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut denom = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut tan_half_sq = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut phi1 = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut phi2 = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    for row_offset in (0..n).step_by(PHOT_VFORCE_TILE_ROWS) {
        let m = PHOT_VFORCE_TILE_ROWS.min(n - row_offset);
        let rd_tile = &mut rd_sq[..m];
        let denom_tile = &mut denom[..m];
        let tan_tile = &mut tan_half_sq[..m];
        let out_tile = &mut out[row_offset..row_offset + m];
        fill_geometry_squared_terms_tile(
            object_pos,
            observer_pos,
            row_offset,
            rd_tile,
            denom_tile,
            tan_tile,
        );
        fill_rd_sq_and_denom_sq(rd_tile, denom_tile);
        vforce::sqrt_in_place(denom_tile);
        fill_tan_half_sq_from_numer_denom(tan_tile, denom_tile);
        let has_invalid = rd_sq_has_invalid(rd_tile);
        magnitude_pipeline_tile(
            h_v,
            g,
            row_offset,
            rd_tile,
            tan_tile,
            &mut phi1[..m],
            &mut phi2[..m],
            out_tile,
            has_invalid,
        );
    }
}

#[cfg(target_os = "macos")]
fn calculate_apparent_magnitude_v_and_phase_angle_macos(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    mag_out: &mut [f64],
    alpha_out: &mut [f64],
) {
    let n = mag_out.len();
    let mut rd_sq = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut denom = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut tan_half_sq = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut phi1 = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut phi2 = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    for row_offset in (0..n).step_by(PHOT_VFORCE_TILE_ROWS) {
        let m = PHOT_VFORCE_TILE_ROWS.min(n - row_offset);
        let rd_tile = &mut rd_sq[..m];
        let denom_tile = &mut denom[..m];
        let tan_tile = &mut tan_half_sq[..m];
        let mag_tile = &mut mag_out[row_offset..row_offset + m];
        let alpha_tile = &mut alpha_out[row_offset..row_offset + m];
        fill_geometry_squared_terms_tile(
            object_pos,
            observer_pos,
            row_offset,
            rd_tile,
            denom_tile,
            tan_tile,
        );
        fill_rd_sq_and_denom_sq(rd_tile, denom_tile);
        vforce::sqrt_in_place(denom_tile);
        fill_tan_half_sq_from_numer_denom(tan_tile, denom_tile);
        // Save tan_half_sq for the alpha branch before the magnitude pipeline
        // consumes it. alpha_tile becomes our tan_half_sq scratch from here.
        alpha_tile.copy_from_slice(tan_tile);
        let has_invalid = rd_sq_has_invalid(rd_tile);
        magnitude_pipeline_tile(
            h_v,
            g,
            row_offset,
            rd_tile,
            tan_tile,
            &mut phi1[..m],
            &mut phi2[..m],
            mag_tile,
            has_invalid,
        );
        // alpha = 2 * atan(sqrt(tan_half_sq)) * RAD2DEG.
        vforce::sqrt_in_place(alpha_tile);
        vforce::atan_in_place(alpha_tile);
        if has_invalid {
            for k in 0..m {
                alpha_tile[k] = if !rd_tile[k].is_finite() || rd_tile[k] <= 0.0 {
                    f64::NAN
                } else {
                    2.0 * alpha_tile[k] * RAD2DEG
                };
            }
        } else {
            for v in alpha_tile.iter_mut() {
                *v *= 2.0 * RAD2DEG;
            }
        }
    }
}

/// True iff every `target_ids[i]` is in-range for `delta_table` and
/// non-negative. The bandpass-aware fast path can use unchecked NEON loads
/// when this holds.
#[cfg(target_os = "macos")]
fn target_ids_all_valid(target_ids: &[i32], delta_table_len: usize) -> bool {
    target_ids
        .iter()
        .all(|&tid| tid >= 0 && (tid as usize) < delta_table_len)
}

/// Final write for `predict_magnitudes` when every `target_ids[i]` is
/// in-range for `delta_table`. Identical NEON FMA layout as the magnitude
/// finish, but adds the per-row bandpass delta.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn finish_predict_neon(
    h_v: &[f64],
    row_offset: usize,
    target_ids: &[i32],
    delta_table: &[f64],
    log_arg: &[f64],
    out: &mut [f64],
) {
    use core::arch::aarch64::*;
    let mut k = 0;
    let scale = unsafe { vdupq_n_f64(TWO_POINT_FIVE_OVER_LN10) };
    while k + 2 <= out.len() {
        // SAFETY: caller pre-validated target_ids in [0, delta_table.len()),
        // h_v and out share the same per-row layout.
        unsafe {
            let i = row_offset + k;
            let h = vld1q_f64(h_v.as_ptr().add(i));
            let la = vld1q_f64(log_arg.as_ptr().add(k));
            let mag = vfmaq_f64(h, scale, la);
            let t0 = *target_ids.get_unchecked(i) as usize;
            let t1 = *target_ids.get_unchecked(i + 1) as usize;
            let d0 = *delta_table.get_unchecked(t0);
            let d1 = *delta_table.get_unchecked(t1);
            let delta_v = vsetq_lane_f64::<1>(d1, vdupq_n_f64(d0));
            vst1q_f64(out.as_mut_ptr().add(k), vaddq_f64(mag, delta_v));
        }
        k += 2;
    }
    while k < out.len() {
        let i = row_offset + k;
        let tid = target_ids[i] as usize;
        out[k] = h_v[i] + TWO_POINT_FIVE_OVER_LN10 * log_arg[k] + delta_table[tid];
        k += 1;
    }
}

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
fn finish_predict_neon(
    h_v: &[f64],
    row_offset: usize,
    target_ids: &[i32],
    delta_table: &[f64],
    log_arg: &[f64],
    out: &mut [f64],
) {
    for k in 0..out.len() {
        let i = row_offset + k;
        let tid = target_ids[i] as usize;
        out[k] = h_v[i] + TWO_POINT_FIVE_OVER_LN10 * log_arg[k] + delta_table[tid];
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn predict_magnitudes_macos_valid_targets(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    target_ids: &[i32],
    delta_table: &[f64],
    out: &mut [f64],
) {
    let n = out.len();
    let mut rd_sq = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut denom = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut tan_half_sq = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut phi1 = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    let mut phi2 = vec![0.0_f64; PHOT_VFORCE_TILE_ROWS];
    for row_offset in (0..n).step_by(PHOT_VFORCE_TILE_ROWS) {
        let m = PHOT_VFORCE_TILE_ROWS.min(n - row_offset);
        let rd_tile = &mut rd_sq[..m];
        let denom_tile = &mut denom[..m];
        let tan_tile = &mut tan_half_sq[..m];
        let phi1_tile = &mut phi1[..m];
        let phi2_tile = &mut phi2[..m];
        let out_tile = &mut out[row_offset..row_offset + m];
        fill_geometry_squared_terms_tile(
            object_pos,
            observer_pos,
            row_offset,
            rd_tile,
            denom_tile,
            tan_tile,
        );
        fill_rd_sq_and_denom_sq(rd_tile, denom_tile);
        vforce::sqrt_in_place(denom_tile);
        fill_tan_half_sq_from_numer_denom(tan_tile, denom_tile);
        let has_invalid = rd_sq_has_invalid(rd_tile);

        // Magnitude pipeline through the final `vvln(log_arg)`, leaving the
        // log-arg in `tan_tile`. Then add the bandpass delta in the NEON
        // finish step.
        vforce::ln_in_place(tan_tile);
        fill_phi_logs_from_ln_tan_sq(tan_tile, phi1_tile, phi2_tile);
        vforce::exp_in_place(phi1_tile);
        vforce::exp_in_place(phi2_tile);
        scale_phi_outer(phi1_tile, phi2_tile);
        vforce::exp_in_place(phi1_tile);
        vforce::exp_in_place(phi2_tile);
        fill_mag_log_arg(g, row_offset, rd_tile, phi1_tile, phi2_tile, tan_tile);
        vforce::ln_in_place(tan_tile);
        finish_predict_neon(h_v, row_offset, target_ids, delta_table, tan_tile, out_tile);

        if has_invalid {
            for k in 0..m {
                if !rd_tile[k].is_finite() || rd_tile[k] <= 0.0 {
                    out_tile[k] = f64::NAN;
                }
            }
        }
    }
}

/// Batched solar phase angle in degrees, writing into a caller-owned buffer.
///
/// Inputs are flattened `N × 3` heliocentric Cartesian positions (AU) —
/// caller's responsibility to transform to SUN origin before calling.
/// Invalid rows (non-finite or non-positive `r`/`delta`) yield NaN.
pub fn calculate_phase_angle_into(object_pos: &[f64], observer_pos: &[f64], out: &mut [f64]) {
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
    assert_eq!(out.len(), n, "out must have length N");

    #[cfg(target_os = "macos")]
    if n >= PHOT_VFORCE_MIN_ROWS {
        calculate_phase_angle_macos(object_pos, observer_pos, out);
        return;
    }

    if n <= PHOT_PHASE_SERIAL_THRESHOLD_ROWS {
        for (i, dst) in out.iter_mut().enumerate() {
            *dst = phase_angle_row(object_pos, observer_pos, i);
        }
        return;
    }

    out.par_chunks_mut(PHOT_CHUNK)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base_i = ci * PHOT_CHUNK;
            for (k, dst) in chunk.iter_mut().enumerate() {
                *dst = phase_angle_row(object_pos, observer_pos, base_i + k);
            }
        });
}

/// Allocating wrapper around [`calculate_phase_angle_into`].
pub fn calculate_phase_angle_flat(object_pos: &[f64], observer_pos: &[f64]) -> Vec<f64> {
    let n = object_pos.len() / 3;
    let mut out = vec![0.0_f64; n];
    calculate_phase_angle_into(object_pos, observer_pos, &mut out);
    out
}

/// Batched apparent V-band magnitude under the H-G phase function,
/// writing into a caller-owned buffer.
///
/// `h_v` and `g` are per-row (length `N`); positions are `N × 3`.
pub fn calculate_apparent_magnitude_v_into(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    out: &mut [f64],
) {
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
    assert_eq!(out.len(), n, "out must have length N");

    #[cfg(target_os = "macos")]
    if n >= PHOT_VFORCE_MIN_ROWS {
        calculate_apparent_magnitude_v_macos(h_v, object_pos, observer_pos, g, out);
        return;
    }

    out.par_chunks_mut(PHOT_CHUNK)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base_i = ci * PHOT_CHUNK;
            for (k, dst) in chunk.iter_mut().enumerate() {
                *dst = apparent_magnitude_v_row(h_v, object_pos, observer_pos, g, base_i + k);
            }
        });
}

/// Allocating wrapper around [`calculate_apparent_magnitude_v_into`].
pub fn calculate_apparent_magnitude_v_flat(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
) -> Vec<f64> {
    let n = object_pos.len() / 3;
    let mut out = vec![0.0_f64; n];
    calculate_apparent_magnitude_v_into(h_v, object_pos, observer_pos, g, &mut out);
    out
}

/// Fused batched (V-band magnitude, phase-angle deg) writing into
/// caller-owned buffers. Faster than two separate calls because the shared
/// `row_geometry` runs once per row.
pub fn calculate_apparent_magnitude_v_and_phase_angle_into(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    mag_out: &mut [f64],
    alpha_out: &mut [f64],
) {
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
    assert_eq!(mag_out.len(), n, "mag_out must have length N");
    assert_eq!(alpha_out.len(), n, "alpha_out must have length N");

    #[cfg(target_os = "macos")]
    if n >= PHOT_VFORCE_MIN_ROWS {
        calculate_apparent_magnitude_v_and_phase_angle_macos(
            h_v,
            object_pos,
            observer_pos,
            g,
            mag_out,
            alpha_out,
        );
        return;
    }

    mag_out
        .par_chunks_mut(PHOT_CHUNK)
        .zip(alpha_out.par_chunks_mut(PHOT_CHUNK))
        .enumerate()
        .for_each(|(ci, (mag_chunk, alpha_chunk))| {
            let base_i = ci * PHOT_CHUNK;
            for (k, (mag_dst, alpha_dst)) in
                mag_chunk.iter_mut().zip(alpha_chunk.iter_mut()).enumerate()
            {
                let (mag, alpha) = apparent_magnitude_v_and_phase_angle_row(
                    h_v,
                    object_pos,
                    observer_pos,
                    g,
                    base_i + k,
                );
                *mag_dst = mag;
                *alpha_dst = alpha;
            }
        });
}

/// Allocating wrapper around
/// [`calculate_apparent_magnitude_v_and_phase_angle_into`].
pub fn calculate_apparent_magnitude_v_and_phase_angle_flat(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = object_pos.len() / 3;
    let mut mag_out = vec![0.0_f64; n];
    let mut alpha_out = vec![0.0_f64; n];
    calculate_apparent_magnitude_v_and_phase_angle_into(
        h_v,
        object_pos,
        observer_pos,
        g,
        &mut mag_out,
        &mut alpha_out,
    );
    (mag_out, alpha_out)
}

/// Fused H-G apparent V-band magnitude + per-row target-filter delta lookup.
///
/// Computes `mag_v + delta_table[target_ids[i]]` per row. Invalid geometry
/// rows (r ≤ 0, δ ≤ 0) produce NaN. Out-of-range target_ids produce NaN
/// (caller should pre-validate; NaN surfaces downstream).
pub fn predict_magnitudes_bandpass_into(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    target_ids: &[i32],
    delta_table: &[f64],
    out: &mut [f64],
) {
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
    assert_eq!(out.len(), n, "out must have length N");

    #[cfg(target_os = "macos")]
    if n >= PHOT_VFORCE_MIN_ROWS && target_ids_all_valid(target_ids, delta_table.len()) {
        predict_magnitudes_macos_valid_targets(
            h_v,
            object_pos,
            observer_pos,
            g,
            target_ids,
            delta_table,
            out,
        );
        return;
    }

    out.par_chunks_mut(PHOT_CHUNK)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base_i = ci * PHOT_CHUNK;
            for (kk, dst) in chunk.iter_mut().enumerate() {
                *dst = predict_magnitude_row(
                    h_v,
                    object_pos,
                    observer_pos,
                    g,
                    target_ids,
                    delta_table,
                    base_i + kk,
                );
            }
        });
}

/// Allocating wrapper around [`predict_magnitudes_bandpass_into`].
pub fn predict_magnitudes_bandpass_flat(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    target_ids: &[i32],
    delta_table: &[f64],
) -> Vec<f64> {
    let n = object_pos.len() / 3;
    let mut out = vec![0.0_f64; n];
    predict_magnitudes_bandpass_into(
        h_v,
        object_pos,
        observer_pos,
        g,
        target_ids,
        delta_table,
        &mut out,
    );
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
    fn phase_angle_matches_small_angle_geometry() {
        let alpha_deg = 0.001_f64;
        let offset = alpha_deg.to_radians().tan();
        let object = [2.0, 0.0, 0.0];
        let observer = [1.0, offset, 0.0];
        let alpha = calculate_phase_angle_flat(&object, &observer);
        assert!((alpha[0] - alpha_deg).abs() < 1e-10);
    }

    #[test]
    fn phase_angle_matches_near_one_eighty_geometry() {
        let miss_deg = 0.001_f64;
        let offset = miss_deg.to_radians().tan();
        let object = [2.0, 0.0, 0.0];
        let observer = [3.0, offset, 0.0];
        let alpha = calculate_phase_angle_flat(&object, &observer);
        assert!((alpha[0] - (180.0 - miss_deg)).abs() < 3e-9);
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
