# Scope — `dynamics.generate_ephemeris_2body` Rust port

Status: scoped, not started. Driven by the 2026-04-22 decision that
surface-level (public-API) performance is the priority and that JAX
sub-kernel shape parity is explicitly **not** a constraint.

## 1. Surface API recap

```python
def generate_ephemeris_2body(
    propagated_orbits: Orbits,          # N barycentric Cartesian (or auto-transformed)
    observers: Observers,               # N observer states (same frame/origin)
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
    predict_magnitudes: bool = True,
    *,
    predict_phase_angle: bool = False,
    max_processes: int | None = 1,
    chunk_size: int = 100,
) -> Ephemeris                           # N topocentric spherical rows (RA/Dec/...)
```

Produced columns per row: topocentric (ρ, lon, lat, vρ, vlon, vlat) in the
equatorial J2000 frame, plus light-time days, aberrated Cartesian state
at emission time, 6×6 output covariance (when input covariance is finite),
and optional `alpha` (phase angle) and `predicted_magnitude_v`.

## 2. What's already Rust-default (reused)

- `coordinates.transform_coordinates` with single-crossing Rust dispatch
  — used for the SUN→SSB barycentric lift when the fast-path is missed
  (Cartesian + origin change + frame change, all fused).
- `dynamics.propagate_2body` / `…_with_covariance` — single-row Rust kernel
  `propagate_2body_row<T: Scalar>` already generic over `Dual<6>` for
  covariance transport (87× / 414× p50 vs legacy). The fused ephemeris
  kernel will **call this row kernel directly**, not the batched flat6
  entrypoint, because each observation row needs its own LT Newton loop
  around it.
- `rotate_ecliptic_to_equatorial6<T>` and `cartesian_to_spherical6<T>` in
  `adam_core_rs_coords::generic` — both already `T: Scalar`-generic,
  usable inside the fused chain under `Dual<6>`.

## 3. Fused kernel design

One Rust row function that takes the fully barycentric inputs and emits
(spherical output, light time, aberrated Cartesian state) plus, under
`Dual<6>` seeding, the full 6×6 Jacobian of the spherical output w.r.t.
the input Cartesian state — all in one pass. Two batched wrappers on top
(state-only, and with-covariance) follow the `propagate_2body_flat6` /
`…_with_covariance_flat6` pattern.

```rust
// rust/adam_core_rs_coords/src/ephemeris.rs (new file)

/// Single-row fused ephemeris kernel. All inputs barycentric Cartesian
/// in the ecliptic frame (caller's responsibility — the Python wrapper
/// does any required SUN→SSB translation before the Rust crossing).
///
/// Returns (spherical_equatorial[6], light_time_days, aberrated_cart[6]).
/// Both outputs are `T`, so seeding `orbit` with `Dual<6>::seed(...)` and
/// `observer_position` as `Dual::<6>::constant(...)` recovers the Jacobian
/// of the spherical output w.r.t. the input orbit state — i.e. the full
/// legacy `transform_covariances_jacobian(_generate_ephemeris_2body)`
/// replacement in one forward pass.
pub fn generate_ephemeris_2body_row<T: Scalar>(
    orbit: [T; 6],                  // barycentric Cartesian ecliptic
    observation_time_mjd_tdb: T,    // seeded as constant
    observer_state: [T; 6],         // barycentric Cartesian ecliptic; pos + vel
    mu: T,                          // attractor GM, AU^3/d^2
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,             // default 10, matches JAX _add_light_time
) -> ([T; 6], T, [T; 6]);
```

Body (pseudo-Rust; single file, no new sub-crates):

1. **Light-time Newton loop** (replaces JAX `_add_light_time`): `lt ← 0`,
   `orbit_i ← orbit`, `t_i ← observation_time`. Iterate up to
   `max_lt_iter`:
   - `rho = ‖orbit_i[0..3] − observer_state[0..3]‖`  (carries Dual tangents)
   - `lt_new = rho / C`
   - stop when `(lt_new − lt).re().abs() ≤ lt_tol`
   - else `t_new = observation_time − lt_new`,
     `orbit_i = propagate_2body_row::<T>(orbit, t_new − observation_time, mu, max_iter, tol)` —
     reuse the already-Dual-ready propagate kernel; the dt argument is
     relative, matching how `propagate_2body_row` treats it.
   - `lt = lt_new`
   - Same non-convergence policy as JAX: if the loop exits with
     `dlt > lt_tol` or iterations == max, set `lt = T::from_f64(NaN)` so
     host-side code can fail fast with row-level context. (The row
     kernel returns the values; the batched wrapper's error pass in
     Python keeps the existing row-index/orbit-id error messages.)
2. **Topocentric translation**: `topo = orbit_i − observer_state` (6-wide).
3. **Optional stellar aberration** (replaces `add_stellar_aberration` —
   closed-form, easy to port):
   - `γ = observer_state[3..6] / C`
   - `β⁻¹ = sqrt(1 − ‖γ‖²)`
   - `δ = ‖topo[0..3]‖`
   - `ρ̂ = topo[0..3] / δ`
   - `ρ̂·γ = dot(ρ̂, γ)`
   - `ρ_ab = δ · (β⁻¹ · ρ̂ + γ + (ρ̂·γ) · γ / (1 + β⁻¹)) / (1 + ρ̂·γ)`
   - overwrite `topo[0..3] = ρ_ab` (velocity untouched, matching legacy)
4. **Ec → Eq rotation** (call the existing `T`-generic
   `rotate_ecliptic_to_equatorial6::<T>(&topo)`).
5. **Cartesian → spherical** (call the existing `T`-generic
   `cartesian_to_spherical6::<T>(&topo_eq)`).
6. Return `(spherical, lt, orbit_i)`.

Batched wrappers:

```rust
pub fn generate_ephemeris_2body_flat6(
    orbits_flat: &[f64],           // (N, 6) barycentric ecliptic Cart
    observation_times: &[f64],
    observer_states_flat: &[f64],  // (N, 6)
    mus: &[f64],
    lt_tol: f64, max_iter: usize, tol: f64,
    stellar_aberration: bool, max_lt_iter: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>);  // (spherical N*6, lt N, aberrated N*6)

pub fn generate_ephemeris_2body_with_covariance_flat6(
    orbits_flat: &[f64],
    covariance_flat: &[f64],       // (N, 36)
    observation_times: &[f64],
    observer_states_flat: &[f64],
    mus: &[f64],
    lt_tol: f64, max_iter: usize, tol: f64,
    stellar_aberration: bool, max_lt_iter: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
//  (spherical N*6, lt N, aberrated N*6, spherical_cov N*36)
```

Both rayon-parallel over rows (`par_chunks_mut`), matching
`propagate_2body_flat6`. Covariance path seeds `Dual<6>` once per row
and does `Σ_out = J @ Σ_in @ J^T` in-loop — the exact pattern from
`propagate_2body_with_covariance_flat6`. NaN-covariance rows short-circuit
to an f64 state-only evaluation and fill Σ_out with NaN (also matching
the existing pattern).

**Critical fusion point**: the one JAX call the legacy wrapper makes to
`transform_covariances_jacobian(_generate_ephemeris_2body)` is expensive
(builds a 6-Jacobian per row via JAX's `vmap`+`jacfwd` on the full
chain). The fused Rust kernel replaces that entire call with a single
Dual<6> evaluation of the same chain — one forward pass, no JIT compile,
no Python round-trip.

## 4. Python integration

1. `src/adam_core/_rust/api.py` — two new thin wrappers returning
   `None` when `RUST_BACKEND_AVAILABLE is False`:
   ```python
   def generate_ephemeris_2body_numpy(orbits, times, observer_states, mus,
       lt_tol=1e-10, max_iter=1000, tol=1e-15,
       stellar_aberration=False, max_lt_iter=10,
   ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]: ...

   def generate_ephemeris_2body_with_covariance_numpy(...
   ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: ...
   ```
2. `rust/adam_core_py/src/lib.rs` — two new `#[pyfunction]` shims,
   following the `propagate_2body_numpy` /
   `propagate_2body_with_covariance_numpy` shape (shape validation in Rust,
   `PyReadonlyArray2` in, `PyArray2` out). Register in
   `#[pymodule] fn _rust_native`.
3. `src/adam_core/_rust/status.py` — two new `ApiMigration` entries:
   - `dynamics.generate_ephemeris_2body` (state + light-time + aberrated)
   - `dynamics.generate_ephemeris_2body_with_covariance` (adds spherical cov)
   Both `status="dual"`, `default="legacy"`, waiver
   `waiver-20260422-generate-ephemeris-2body-perf-pending` until the
   benchmark gate runs.
4. `src/adam_core/dynamics/ephemeris.py` — refactor
   `generate_ephemeris_2body` so the chunked body dispatches through a
   single Rust crossing when the extension is available:
   - Keep the existing SUN/ecliptic barycentric fast-path (uses
     `get_perturber_state` once, already Rust-backed via spicekit).
   - Replace the current `process_in_chunks` loop over
     `_generate_ephemeris_2body_vmap` + the separate
     `transform_covariances_jacobian` call with one call to the Rust
     batched fn (with-covariance variant when covariance is finite,
     state-only otherwise). Chunking is no longer needed — rayon handles
     parallelism — but keep a large-N watchdog to emit the same
     `DynamicsNumericalError` row context on NaN rows.
   - Legacy JAX path kept behind `if rust_result is None:` for review
     period (as per 2026-04-16 deviation from fail-loudly).
   - Ray multi-process outer fan-out stays Python-side (already
     process-parallel; Rust kernel is rayon thread-parallel *within*
     each Ray worker — the two compose cleanly).
5. `src/adam_core/_rust/__init__.py` — re-export the two new wrappers.

## 5. Covariance transport

Entire chain (LT-Newton → topo translate → opt-aberration → ec→eq → cart→sph)
runs once under `Dual<6>`. The resulting 6×6 Jacobian is the Jacobian of
spherical-output w.r.t. Cartesian-input — the exact quantity legacy's
`transform_covariances_jacobian(_generate_ephemeris_2body)` computes via
JAX `jacfwd`. Then `Σ_out = J @ Σ_in @ J^T` as usual. No JAX on the
Rust path.

The LT loop's convergence test already uses `.re().abs() ≤ tol` (same
pattern as `calc_chi` in `propagate.rs`), so the final Dual state carries
correct derivatives through the fixed point.

## 6. Parity test plan

New file `src/adam_core/dynamics/tests/test_rust_ephemeris_parity.py`:

- States + light-time + aberrated: compare Rust batched kernel output
  against `_generate_ephemeris_2body_vmap(...)` on the same inputs.
  Tolerances: `atol=1e-11` on spherical (matches propagate parity),
  `atol=1e-12` on light-time days, `atol=1e-11` on aberrated state.
- Covariance: compare Rust-computed spherical Σ against
  `transform_covariances_jacobian(_generate_ephemeris_2body, ...)` at
  `rtol=1e-6, atol=1e-14` (matches propagate cov parity tolerance —
  Jacobian products amplify roundoff).
- Parametrized over n ∈ {1, 8, 64}, dt_scales equivalent to observation
  lead times {1, 30, 365, 3650} days from orbit epoch, and
  `stellar_aberration ∈ {False, True}`.
- Edge cases: partial-NaN covariance input → NaN Σ output, aberrated
  state finite; all-NaN covariance input → NaN Σ output, warn-free
  happy path for state-only callers.

## 7. Benchmark + promotion plan

Add a `generate_ephemeris_2body` entry to
`migration/scripts/rust_backend_benchmark_gate.py`:
- Input builder: N_orbits × N_times grid (start at 1000×20 = 20k rows,
  same shape as propagate gate). Barycentric Cartesian orbits via
  Keplerian→Cart with scatter; observer states sampled from a
  DE440-realistic observatory spread.
- Reference: `_generate_ephemeris_2body_vmap(...)` + (for cov path)
  `transform_covariances_jacobian(_generate_ephemeris_2body, ...)` —
  matches production's code path exactly.
- Register both api_ids in `benchmark_to_api_id`.
- Expect ≫ 1.2× p50/p95 given propagate_2body's 87× is a subset of this
  chain. Gate promotion to `rust-default` on the harness artifact, same
  as propagate.

## 8. Out-of-scope for v1

- SPICE origin resolution stays Python (`get_perturber_state` already
  Rust-backed via spicekit; the 1–2 translate calls are cheap).
- Photometry (`calculate_apparent_magnitude_v*`, `calculate_phase_angle`)
  stays JAX/NumPy — separate follow-up.
- Ray multi-process fan-out stays Python (coarse parallelism over
  chunks; rayon inside each chunk).
- Non-barycentric observers / non-SUN attractors on the fast path —
  pre-transform stays in `transform_coordinates` (already Rust-default).

## 9. Retroactive fusion candidates (per 2026-04-22 rule)

Review pass to run after this port lands. Each needs a microbench before
committing time:

1. **`propagate_2body` + `transform_coordinates` composed through Python**:
   callers that propagate then immediately change representation/frame
   currently cross Rust twice. A fused `propagate_and_transform_row`
   would save one crossing and unlock output-in-Keplerian directly from
   Cartesian inputs via Dual<6>. Check call sites — if the combined
   pattern is common, fuse.
2. **`transform_coordinates` with origin translation + representation
   change**: already fused at the row level (per 2026-04-20 lever 4).
   No action.
3. **`calc_gauss` + `propagate_2body` in `gaussIOD` refinement loop**:
   the IOD orbits go through propagate_2body for residual scoring.
   Currently gaussIOD is `default="legacy"` pending perf — a fused
   candidate-gen + propagate kernel could close the gap in a single
   revisit.

## 10. Rollout phases

1. **Land kernel + bindings + parity tests** (one PR). Ship as
   `status="dual" / default="legacy"` with the perf-pending waiver.
2. **Run benchmark gate**, capture artifact, and — assuming >1.2×
   p50/p95 — flip `default="rust"` and wire dispatch in
   `generate_ephemeris_2body`'s chunk loop.
3. **Mark waiver resolved** in `migration/waivers.yaml` with the gate
   artifact as resolution evidence, matching the propagate_2body pattern.

## TODO.md delta

```markdown
- [ ] Port `dynamics.generate_ephemeris_2body` fused kernel (LT Newton
      + optional stellar aberration + ec→eq + cart→sph) as a single
      `T: Scalar`-generic row kernel, reusing the existing
      `propagate_2body_row`. Ship as `status="dual"` / `default="legacy"`
      with waiver `waiver-20260422-generate-ephemeris-2body-perf-pending`.
- [ ] Add parity tests (states + light-time + aberrated + covariance)
      vs `_generate_ephemeris_2body_vmap` + `transform_covariances_jacobian`.
- [ ] Add benchmark-gate entry `generate_ephemeris_2body` +
      `generate_ephemeris_2body_with_covariance`; promote to `rust-default`
      and resolve the waiver once the +20% p50/p95 gate clears.
```
