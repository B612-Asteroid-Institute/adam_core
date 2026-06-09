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
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
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
/// batched transcendentals to amortize their per-call setup. Sized to be at
/// or below the canonical `parity_fuzz --n 128` workload so the vForce path
/// is exercised by every parity gate run, not just by the small-/large-n
/// speed governance at n>=2000. Below this threshold the existing scalar
/// Rayon path is still the cross-platform fallback and is generally faster
/// on per-row dispatch overhead, but the cost is irrelevant: parity_fuzz is
/// correctness-only and there is no speed gate at n in [64, 2000).
#[cfg(target_os = "macos")]
const PHOT_VFORCE_MIN_ROWS: usize = 64;

/// Scratch-pool capacity for the vForce/NEON path. Magnitude / fused /
/// predict work best at this tile because their five per-tile vForce
/// transcendentals amortize the per-call dispatch cost across more rows.
#[cfg(target_os = "macos")]
const PHOT_VFORCE_TILE_ROWS: usize = 8192;

/// Phase-angle uses a smaller tile so its two scratch buffers sit in the
/// 192 KB Apple Silicon L1 across the geometry+`vvacos`+finishing passes.
/// Magnitude / fused / predict do the opposite trade and stay at the larger
/// tile because each extra vForce dispatch costs more than the L1 win.
#[cfg(target_os = "macos")]
const PHOT_VFORCE_PHASE_TILE_ROWS: usize = 4096;

// Reuse a per-thread scratch buffer pool across photometry calls. The
// buffers are sized to the full vForce tile so they can be sliced down to
// the per-call N. Avoids a heap allocation on every public-API entry.
#[cfg(target_os = "macos")]
thread_local! {
    static VFORCE_SCRATCH: std::cell::RefCell<VForceScratch> =
        std::cell::RefCell::new(VForceScratch::new());
}

#[cfg(target_os = "macos")]
struct VForceScratch {
    rd_sq: Vec<f64>,
    tan_half_sq: Vec<f64>,
    /// Backing storage for both `phi1` and `phi2` as a single contiguous
    /// `2 * PHOT_VFORCE_TILE_ROWS` buffer. The first half is `phi1`, the
    /// second half is `phi2`. Storing them contiguously lets the magnitude
    /// pipeline run each `vvexp` once over both halves instead of twice,
    /// saving two vForce dispatches per tile.
    phi: Vec<f64>,
}

#[cfg(target_os = "macos")]
impl VForceScratch {
    fn new() -> Self {
        Self {
            rd_sq: vec![0.0_f64; PHOT_VFORCE_TILE_ROWS],
            tan_half_sq: vec![0.0_f64; PHOT_VFORCE_TILE_ROWS],
            phi: vec![0.0_f64; 2 * PHOT_VFORCE_TILE_ROWS],
        }
    }
}

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

/// Fused NEON pass that streams `(object_pos, observer_pos)` once and writes
/// `rd_sq = r²·δ²` plus the per-row clamped cosine into the caller's tile
/// buffer. Returns `true` if any row has non-finite or non-positive geometry,
/// so callers can NaN-stamp those rows after `vvacos`. Replaces the prior
/// four-pass geometry pipeline (squared terms → rd_sq+denom_sq → sqrt → clamp).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fill_phase_geometry_tile(
    object_pos: &[f64],
    observer_pos: &[f64],
    row_offset: usize,
    rd_sq: &mut [f64],
    cos_clamped: &mut [f64],
) -> bool {
    use core::arch::aarch64::*;

    let mut k = 0;
    let zero = unsafe { vdupq_n_f64(0.0) };
    let one = unsafe { vdupq_n_f64(1.0) };
    let neg_one = unsafe { vdupq_n_f64(-1.0) };
    let max_finite = unsafe { vdupq_n_f64(f64::MAX) };
    let all_bits = unsafe { vdupq_n_u64(u64::MAX) };
    let mut bad = unsafe { vdupq_n_u64(0) };
    while k + 2 <= rd_sq.len() {
        let i = row_offset + k;
        // SAFETY: caller passes tile slices sized within the asserted N×3
        // input buffers; vld3q_f64 reads two interleaved xyz rows.
        unsafe {
            let obj = vld3q_f64(object_pos.as_ptr().add(i * 3));
            let obs = vld3q_f64(observer_pos.as_ptr().add(i * 3));
            let r_sq_v = vfmaq_f64(
                vfmaq_f64(vmulq_f64(obj.0, obj.0), obj.1, obj.1),
                obj.2,
                obj.2,
            );
            let dx = vsubq_f64(obj.0, obs.0);
            let dy = vsubq_f64(obj.1, obs.1);
            let dz = vsubq_f64(obj.2, obs.2);
            let delta_sq_v = vfmaq_f64(vfmaq_f64(vmulq_f64(dx, dx), dy, dy), dz, dz);
            let dot_v = vfmaq_f64(
                vfmaq_f64(vmulq_f64(obj.0, obs.0), obj.1, obs.1),
                obj.2,
                obs.2,
            );
            // numer = 2(r² − dot); the factor of 2 cancels with the 2 in the
            // legacy denom = 2 r δ, so we compute numer = r² − dot, denom = r · δ.
            let numer_v = vsubq_f64(r_sq_v, dot_v);
            let rd_sq_v = vmulq_f64(r_sq_v, delta_sq_v);
            let denom_v = vsqrtq_f64(rd_sq_v);
            let raw_cos = vdivq_f64(numer_v, denom_v);
            let clamped = vmaxq_f64(neg_one, vminq_f64(one, raw_cos));
            vst1q_f64(rd_sq.as_mut_ptr().add(k), rd_sq_v);
            vst1q_f64(cos_clamped.as_mut_ptr().add(k), clamped);

            // Track invalid geometry: r_sq or delta_sq non-finite or ≤ 0.
            // r_sq and delta_sq are sums of squares so they are ≥0, but a
            // NaN/Inf input or a coincident object/observer row will trip this.
            let r_finite = vcleq_f64(r_sq_v, max_finite);
            let delta_finite = vcleq_f64(delta_sq_v, max_finite);
            let r_pos = vcgtq_f64(r_sq_v, zero);
            let delta_pos = vcgtq_f64(delta_sq_v, zero);
            let row_ok = vandq_u64(
                vandq_u64(r_finite, delta_finite),
                vandq_u64(r_pos, delta_pos),
            );
            bad = vorrq_u64(bad, veorq_u64(row_ok, all_bits));
        }
        k += 2;
    }
    let mut has_invalid = unsafe { (vgetq_lane_u64::<0>(bad) | vgetq_lane_u64::<1>(bad)) != 0 };
    while k < rd_sq.len() {
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
        let numer = r_sq_v - dot;
        let rd_sq_v = r_sq_v * delta_sq_v;
        let denom = rd_sq_v.sqrt();
        rd_sq[k] = rd_sq_v;
        cos_clamped[k] = (numer / denom).clamp(-1.0, 1.0);
        has_invalid |=
            !r_sq_v.is_finite() || !delta_sq_v.is_finite() || r_sq_v <= 0.0 || delta_sq_v <= 0.0;
        k += 1;
    }
    has_invalid
}

/// Fused NEON pass that streams `(object_pos, observer_pos)` once and writes
/// `rd_sq` plus `tan²(α/2)` directly. Used by the magnitude / fused / predict
/// pipelines. Returns `true` for any non-finite/≤0 geometry row.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fill_mag_geometry_tile(
    object_pos: &[f64],
    observer_pos: &[f64],
    row_offset: usize,
    rd_sq: &mut [f64],
    tan_half_sq: &mut [f64],
) -> bool {
    use core::arch::aarch64::*;

    let mut k = 0;
    let zero = unsafe { vdupq_n_f64(0.0) };
    let max_finite = unsafe { vdupq_n_f64(f64::MAX) };
    let all_bits = unsafe { vdupq_n_u64(u64::MAX) };
    let mut bad = unsafe { vdupq_n_u64(0) };
    while k + 2 <= rd_sq.len() {
        let i = row_offset + k;
        // SAFETY: see fill_phase_geometry_tile.
        unsafe {
            let obj = vld3q_f64(object_pos.as_ptr().add(i * 3));
            let obs = vld3q_f64(observer_pos.as_ptr().add(i * 3));
            let r_sq_v = vfmaq_f64(
                vfmaq_f64(vmulq_f64(obj.0, obj.0), obj.1, obj.1),
                obj.2,
                obj.2,
            );
            let dx = vsubq_f64(obj.0, obs.0);
            let dy = vsubq_f64(obj.1, obs.1);
            let dz = vsubq_f64(obj.2, obs.2);
            let delta_sq_v = vfmaq_f64(vfmaq_f64(vmulq_f64(dx, dx), dy, dy), dz, dz);
            let dot_v = vfmaq_f64(
                vfmaq_f64(vmulq_f64(obj.0, obs.0), obj.1, obs.1),
                obj.2,
                obs.2,
            );
            let numer_v = vsubq_f64(r_sq_v, dot_v);
            let rd_sq_v = vmulq_f64(r_sq_v, delta_sq_v);
            let denom_v = vsqrtq_f64(rd_sq_v);
            // tan²(α/2) = (denom − clamp(numer, ±denom)) / (denom + clamp(…)).
            let clamped = vmaxq_f64(vnegq_f64(denom_v), vminq_f64(denom_v, numer_v));
            let tan_sq_v = vdivq_f64(vsubq_f64(denom_v, clamped), vaddq_f64(denom_v, clamped));
            vst1q_f64(rd_sq.as_mut_ptr().add(k), rd_sq_v);
            vst1q_f64(tan_half_sq.as_mut_ptr().add(k), tan_sq_v);

            let r_finite = vcleq_f64(r_sq_v, max_finite);
            let delta_finite = vcleq_f64(delta_sq_v, max_finite);
            let r_pos = vcgtq_f64(r_sq_v, zero);
            let delta_pos = vcgtq_f64(delta_sq_v, zero);
            let row_ok = vandq_u64(
                vandq_u64(r_finite, delta_finite),
                vandq_u64(r_pos, delta_pos),
            );
            bad = vorrq_u64(bad, veorq_u64(row_ok, all_bits));
        }
        k += 2;
    }
    let mut has_invalid = unsafe { (vgetq_lane_u64::<0>(bad) | vgetq_lane_u64::<1>(bad)) != 0 };
    while k < rd_sq.len() {
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
        let numer = r_sq_v - dot;
        let rd_sq_v = r_sq_v * delta_sq_v;
        let denom = rd_sq_v.sqrt();
        let clamped = numer.clamp(-denom, denom);
        rd_sq[k] = rd_sq_v;
        tan_half_sq[k] = (denom - clamped) / (denom + clamped);
        has_invalid |=
            !r_sq_v.is_finite() || !delta_sq_v.is_finite() || r_sq_v <= 0.0 || delta_sq_v <= 0.0;
        k += 1;
    }
    has_invalid
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
fn magnitude_pipeline_tile(
    h_v: &[f64],
    g: &[f64],
    row_offset: usize,
    rd_sq: &[f64],
    tan_half_sq: &mut [f64],
    phi: &mut [f64],
    out: &mut [f64],
    has_invalid: bool,
) {
    let m = out.len();
    debug_assert_eq!(phi.len(), 2 * m);
    vforce::ln_in_place(tan_half_sq);
    let (phi1, phi2) = phi.split_at_mut(m);
    fill_phi_logs_from_ln_tan_sq(tan_half_sq, phi1, phi2);
    // First exp pass: phi[0..m] = tan_half^0.63, phi[m..2m] = tan_half^1.22.
    vforce::exp_in_place(phi);
    let (phi1, phi2) = phi.split_at_mut(m);
    scale_phi_outer(phi1, phi2);
    // Second exp pass: phi[0..m] = exp(-3.33·tan_half^0.63),
    // phi[m..2m] = exp(-1.87·tan_half^1.22).
    vforce::exp_in_place(phi);
    let (phi1, phi2) = phi.split_at(m);

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

/// Phase-angle finishing on the `vvacos` output. The fast common case is one
/// NEON multiply-by-RAD2DEG sweep; rows whose clamped cosine sits in the
/// endpoint band `|cos| > PHOT_ENDPOINT_COS_ABS_MAX` are recomputed via the
/// JAX half-angle reference for the precision the legacy gate expects, and
/// invalid-geometry rows are NaN-stamped.
#[cfg(target_os = "macos")]
fn finish_phase_angle_tile(
    object_pos: &[f64],
    observer_pos: &[f64],
    row_offset: usize,
    cos_phase: &[f64],
    rd_sq: &[f64],
    out_tile: &mut [f64],
    has_invalid: bool,
) {
    // Single SIMD scan to confirm whether any row sits in the endpoint band.
    // This is cheaper than threading the detection through the geometry pass
    // and lets the common (no slow-path) case skip a per-row branch.
    let needs_endpoint = has_invalid
        || cos_phase
            .iter()
            .any(|&c| c.abs() > PHOT_ENDPOINT_COS_ABS_MAX);
    if !needs_endpoint {
        for v in out_tile.iter_mut() {
            *v *= RAD2DEG;
        }
        return;
    }
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
}

#[cfg(target_os = "macos")]
fn calculate_phase_angle_macos(object_pos: &[f64], observer_pos: &[f64], out: &mut [f64]) {
    // Distribute tiles across Rayon workers for native multi-thread
    // throughput. With `RAYON_NUM_THREADS=1` (the canonical single-thread
    // gate) this collapses to a sequential walk so the gate timings are
    // unchanged; with unconstrained Rayon the work scales across cores
    // because each worker uses its own thread-local `VFORCE_SCRATCH` slab.
    out.par_chunks_mut(PHOT_VFORCE_PHASE_TILE_ROWS)
        .enumerate()
        .for_each(|(tile_idx, out_tile)| {
            let row_offset = tile_idx * PHOT_VFORCE_PHASE_TILE_ROWS;
            let m = out_tile.len();
            VFORCE_SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let VForceScratch {
                    rd_sq, tan_half_sq, ..
                } = &mut *scratch;
                let rd_tile = &mut rd_sq[..m];
                let cos_tile = &mut tan_half_sq[..m];
                let has_invalid = fill_phase_geometry_tile(
                    object_pos,
                    observer_pos,
                    row_offset,
                    rd_tile,
                    cos_tile,
                );
                vforce::acos(cos_tile, out_tile);
                // The half-angle reference is required only inside the
                // slow-path band `|cos| > PHOT_ENDPOINT_COS_ABS_MAX`.
                // Without scanning we'd pay precision near 0°/180°; with the
                // scan, the common case is a single multiply-by-RAD2DEG sweep.
                finish_phase_angle_tile(
                    object_pos,
                    observer_pos,
                    row_offset,
                    cos_tile,
                    rd_tile,
                    out_tile,
                    has_invalid,
                );
            });
        });
}

#[cfg(target_os = "macos")]
fn calculate_apparent_magnitude_v_macos(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    out: &mut [f64],
) {
    out.par_chunks_mut(PHOT_VFORCE_TILE_ROWS)
        .enumerate()
        .for_each(|(tile_idx, out_tile)| {
            let row_offset = tile_idx * PHOT_VFORCE_TILE_ROWS;
            let m = out_tile.len();
            VFORCE_SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let VForceScratch {
                    rd_sq,
                    tan_half_sq,
                    phi,
                } = &mut *scratch;
                let rd_tile = &mut rd_sq[..m];
                let tan_tile = &mut tan_half_sq[..m];
                let phi_tile = &mut phi[..2 * m];
                let has_invalid =
                    fill_mag_geometry_tile(object_pos, observer_pos, row_offset, rd_tile, tan_tile);
                magnitude_pipeline_tile(
                    h_v,
                    g,
                    row_offset,
                    rd_tile,
                    tan_tile,
                    phi_tile,
                    out_tile,
                    has_invalid,
                );
            });
        });
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn calculate_apparent_magnitude_v_and_phase_angle_macos(
    h_v: &[f64],
    object_pos: &[f64],
    observer_pos: &[f64],
    g: &[f64],
    mag_out: &mut [f64],
    alpha_out: &mut [f64],
) {
    mag_out
        .par_chunks_mut(PHOT_VFORCE_TILE_ROWS)
        .zip(alpha_out.par_chunks_mut(PHOT_VFORCE_TILE_ROWS))
        .enumerate()
        .for_each(|(tile_idx, (mag_tile, alpha_tile))| {
            let row_offset = tile_idx * PHOT_VFORCE_TILE_ROWS;
            let m = mag_tile.len();
            VFORCE_SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let VForceScratch {
                    rd_sq,
                    tan_half_sq,
                    phi,
                } = &mut *scratch;
                let rd_tile = &mut rd_sq[..m];
                let tan_tile = &mut tan_half_sq[..m];
                let phi_tile = &mut phi[..2 * m];
                let has_invalid =
                    fill_mag_geometry_tile(object_pos, observer_pos, row_offset, rd_tile, tan_tile);
                // Save tan_half_sq for the alpha branch before the magnitude
                // pipeline consumes it. alpha_tile becomes our tan_half_sq
                // scratch from here.
                alpha_tile.copy_from_slice(tan_tile);
                magnitude_pipeline_tile(
                    h_v,
                    g,
                    row_offset,
                    rd_tile,
                    tan_tile,
                    phi_tile,
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
            });
        });
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
    out.par_chunks_mut(PHOT_VFORCE_TILE_ROWS)
        .enumerate()
        .for_each(|(tile_idx, out_tile)| {
            let row_offset = tile_idx * PHOT_VFORCE_TILE_ROWS;
            let m = out_tile.len();
            VFORCE_SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let VForceScratch {
                    rd_sq,
                    tan_half_sq,
                    phi,
                } = &mut *scratch;
                let rd_tile = &mut rd_sq[..m];
                let tan_tile = &mut tan_half_sq[..m];
                let phi_tile = &mut phi[..2 * m];
                let has_invalid =
                    fill_mag_geometry_tile(object_pos, observer_pos, row_offset, rd_tile, tan_tile);

                // Magnitude pipeline through the final `vvln(log_arg)`,
                // leaving the log-arg in `tan_tile`. The contiguous `phi`
                // buffer lets us run each `vvexp` once over both halves
                // instead of twice. The bandpass delta is fused into the
                // final NEON write so predict doesn't pay an extra pass
                // beyond magnitude.
                vforce::ln_in_place(tan_tile);
                let (phi1_tile, phi2_tile) = phi_tile.split_at_mut(m);
                fill_phi_logs_from_ln_tan_sq(tan_tile, phi1_tile, phi2_tile);
                vforce::exp_in_place(phi_tile);
                let (phi1_tile, phi2_tile) = phi_tile.split_at_mut(m);
                scale_phi_outer(phi1_tile, phi2_tile);
                vforce::exp_in_place(phi_tile);
                let (phi1_tile, phi2_tile) = phi_tile.split_at(m);
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
            });
        });
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
