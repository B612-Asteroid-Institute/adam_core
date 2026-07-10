# Parity + Baseline Speed Gate

Baseline-main enforcement for the Rust migration. Three gates run side-by-side
for APIs wired into this harness:

1. **Parity-fuzz** — randomized inputs, current Rust output must match the
   upstream `main` implementation within the per-API tolerance defined in
   `tolerances.py`.
2. **Fixed-fixture parity** — deterministic baseline-main fixtures for APIs
   where randomized fuzz is intentionally misleading (currently
   `orbit_determination.gaussIOD`).
3. **Speedup** — current Rust must meet lane-specific p50/p95 thresholds
   versus the upstream `main` implementation on identical workloads, unless an
   explicit lane-scoped waiver is attached. Speed artifacts include `tiny-n`
   for quick one-off calls, `small-n` for the historical `n=2000` promotion
   gate, and enforced API-shaped `large-n` workloads with structured axes.

This harness does not time a current-branch Python fallback as "legacy".
The legacy side is the separate baseline-main checkout installed in
`.legacy-venv`, so it cannot accidentally call migration-branch Rust
fallthroughs. When a fair baseline-main oracle does not exist for an API, use
fixed trusted vectors or the Rust-only latency gate described in
[`migration/benchmark_governance.md`](../benchmark_governance.md).

## Legacy oracle

The legacy implementation lives in a **dedicated, main-pinned** sibling
checkout `/Users/aleck/Code/adam-core-legacy-main`. Keep this checkout
separate from any working checkout you develop in: the speed baseline is
fingerprinted on the legacy checkout's git commit (plus its venv
`pip freeze` and the harness source), so a checkout that drifts onto a
feature branch silently invalidates `parity_legacy_speed_baseline.json`.
The dedicated checkout should stay on the commit the baseline was captured
against (`4c1fbc4cd1a67b1e8527f20dce0b853b9a4022ac`, upstream `main` at
capture time); the speed gate asserts this pin before cache validation so
checkout drift fails with a clear error. Bump it deliberately and recapture.

Because both repos export the package name `adam_core`, they cannot coexist
in one Python venv. We invoke the legacy implementation through a
**subprocess** running inside `.legacy-venv` (gitignored, set up once):

```bash
# Dedicated legacy checkout pinned to upstream main.
git clone git@github.com:B612-Asteroid-Institute/adam_core.git \
    /Users/aleck/Code/adam-core-legacy-main
git -C /Users/aleck/Code/adam-core-legacy-main checkout main

python3.13 -m venv .legacy-venv
.legacy-venv/bin/pip install -e /Users/aleck/Code/adam-core-legacy-main
```

To bump the legacy baseline to a newer upstream `main`:

```bash
git -C /Users/aleck/Code/adam-core-legacy-main fetch origin
git -C /Users/aleck/Code/adam-core-legacy-main reset --hard origin/main
pdm run rust-parity-legacy-cache-refresh   # recapture the baseline
```

The legacy checkout path, legacy venv Python, and expected legacy commit are
overridable via `ADAM_CORE_LEGACY_REPO_ROOT`, `ADAM_CORE_LEGACY_VENV_PYTHON`,
and `ADAM_CORE_LEGACY_EXPECTED_GIT_COMMIT` (defaults:
`/Users/aleck/Code/adam-core-legacy-main`, `.legacy-venv/bin/python`, and the
pinned commit above). Only override the expected commit when intentionally
bumping and recapturing the baseline.

Each parity/speed row is labeled with a **comparison mode** (`raw kernel`,
`thin wrapper`, `public facade`, `rust native`, or `impl candidate`) so the
tables make explicit whether the current side is measured as a raw Rust/PyO3
kernel, a thin NumPy wrapper, or a composed public Python facade -- all
against the same legacy public Python.

Performance rows report three distinct columns:

1. **legacy adam_core**: the pinned legacy checkout in its isolated Python
   runtime;
2. **current through Python** (`current_python_*`; historical `rust_*` keys are
   retained as aliases): the compatible current Python/public entrypoint users
   call; and
3. **native Rust** (`native_rust_*`): the underlying implementation called
   directly in Rust and timed by Rust `std::time::Instant`. Python may launch
   the benchmark and pass prepared inputs once, but no Python or PyO3 boundary
   is inside any native-Rust sample.

Legacy/current-Python remains the enforced speed gate. Native Rust is a
performance decomposition diagnostic. A Python→PyO3 call duration is **not** a
native-Rust measurement. When no Rust-internal timer exists, the native column
is `null` and carries an explicit reason/TODO rather than a proxy value. The
Arrow-native observer reference implements this contract; each
`personal-cmy.36.3`–`.36.9` conversion must add its Rust-internal timer as the
surface moves from NumPy to native Arrow.

To preserve historical artifact readers without duplicating raw sample
matrices, old keys remain the sample owners. In adam_core speed artifacts,
current-through-Python samples live under `rust_sample_trials_s`; the explicit
`current_python_*` fields are scalar summaries. In nested ASSIST timing
payloads, `python.values` and `rust.values` own legacy and current samples,
while `legacy_adam_core.samples_alias` and `current_python.samples_alias` are
dotted JSON paths to those arrays. The flat impacts artifact uses sibling-key
aliases (`python_samples_s` / `rust_samples_s`) instead. Consumers should
resolve `samples_alias` when they need raw samples.

Verify it's reachable:

```bash
.venv/bin/python -m migration.parity._oracle
# → [oracle smoke] cart→sph OK, ...
```

## Legacy adam_assist oracle

The GPL `adam_assist_rust` parity suite
(`rust/adam_core_rs_assist/python/tests/`) compares the Rust
`adam_assist_rust.ASSISTPropagator` against the legacy, composition-based
downstream `adam_assist.ASSISTPropagator` with the **same two-runtime
pattern**. adam_core here has deleted the base `Propagator` composition, so
the legacy propagator is no longer instantiable in the main runtime; it runs
in a dedicated `.legacy-assist-venv` (gitignored) pinning legacy adam_core
(composition) + downstream `adam_assist`:

```bash
python3.13 -m venv .legacy-assist-venv
.legacy-assist-venv/bin/pip install 'assist==1.2.3' 'rebound>=4.4.10'
.legacy-assist-venv/bin/pip install -e /Users/aleck/Code/adam-core-legacy-main
.legacy-assist-venv/bin/pip install naif-de440 jpl-small-bodies-de441-n16
.legacy-assist-venv/bin/pip install 'adam-assist==0.3.9' --no-deps
```

`migration/parity/_assist_oracle.py` exposes a `LegacyAssistPropagator`
drop-in proxy: each `propagate_orbits` / `generate_ephemeris` /
`detect_collisions` call serializes its quivr inputs to Arrow IPC, runs the
legacy propagator in the isolated runtime
(`migration/parity/_assist_legacy_runner.py`), and reconstructs the result
under the main runtime's adam_core. Legacy outputs are cached under
`migration/artifacts/assist_parity_cache/` (gitignored) keyed by a stable
hash of the request, so the expensive ASSIST integrations run once per
distinct input (tests with non-deterministic monte-carlo inputs re-run the
legacy runtime live, like the adam_core fuzz gate). Set
`ADAM_CORE_ASSIST_PARITY_REFRESH=1` to force re-running the legacy runtime;
the venv Python is overridable via `ADAM_CORE_LEGACY_ASSIST_VENV_PYTHON`.

ASSIST artifacts use the same column names. Until its package-level Rust
implementations expose Rust-internal `Instant` benchmark adapters, the native
Rust field is intentionally null and references `personal-98v.1`; timing the
`NativeAssistPropagator` PyO3 method from Python would not satisfy the native
column.

The parity tests skip gracefully when `.legacy-assist-venv` is absent. The
frozen public-semantics fixture
(`migration/artifacts/assist_public_semantics_fixture_2026-05-20.json`, the
Rust crate's compile-time oracle) is likewise regenerated in this legacy
runtime by `test_assist_public_semantics_fixture_is_current`.

## Running the gates

Full baseline-main gate (writes `migration/artifacts/parity_gate.json`):

```bash
.venv/bin/python -m migration.parity.parity_main \
    --threads multi-thread \
    --speed-tiny --speed-tiny-reps 101 --speed-tiny-warmup 3 \
    --speed-n 2000 --speed-reps 21 --speed-warmup 3 \
    --speed-large --speed-large-reps 7 --speed-large-warmup 1 \
    --speed-legacy-cache migration/artifacts/parity_legacy_speed_baseline.json
```

These canonical warm speed loops are non-cached compute comparisons. Before
each warmup and timed legacy/current-Python call, the harness clears semantic
observer-state, origin-translation, and SPKEZ result caches outside the timer.
Imports, loaded kernels/readers, JIT state, and thread pools remain warm.
Production caches are unchanged; the policy applies only inside the benchmark
harness and is recorded in emitted artifacts.

For a compact review view containing only canonical surface name and absolute
legacy/current-Python/native-Rust p50/p95 columns (blank when a native adapter
does not exist), render the committed artifacts with:

```bash
.venv/bin/python -m migration.scripts.parity_table \
    --parity-artifact migration/artifacts/parity_gate.json \
    --speed-artifact migration/artifacts/parity_gate.json \
    --simple-timings
```

Temporary `bridge.*` candidate IDs are shown under their canonical public
surface name plus an implementation qualifier in this view; they remain
separate diagnostic lanes until `personal-cmy.36.10` retires them after their
coverage is absorbed.

The canonical thread mode is **multi-thread**: both Rust Rayon and the legacy
NumPy/JAX/XLA/BLAS pools run uncapped, giving a production-realistic
best-effort comparison. The historical `--threads single` mode is retained as
an ad-hoc diagnostic flag on `parity_speed`/`parity_main`, but it is no longer
used in canonical scripts because legacy JAX/XLA on macOS Apple Silicon does
not honor known thread-cap env vars (verified 2026-05-07: cpu/wall ~3.6 cores
in legacy JAX photometry kernels even with all known caps applied), so the
single-thread comparison is asymmetric Rust-1-core vs legacy-uncapped on
macOS. Rust-only single-thread regression detection lives in
`pdm run rust-latency-gate`, where there is no legacy comparison and capped
Rayon gives a stable baseline.

Detailed speed-lane enforcement, tiny-n p95 policy, raw-kernel diagnostic
status, and built-in timing-trial aggregation are governed in
[`migration/benchmark_governance.md`](../benchmark_governance.md). The speed
artifacts repeat the active policy as machine-readable metadata.

Refresh the serialized baseline-main timing cache once after adding benchmark
APIs, changing workload shapes, changing reps/warmup/thread policy, or updating
the baseline checkout. Refreshes merge newly captured entries into the existing
cache by default, so API-scoped recaptures do not wipe unrelated rows. A
benchmark-source hash change from adding adapters or workloads can be refreshed
additively; each cache entry records the identity it was captured under, and
normal cache lookups fail loudly on legacy checkout, tracked-code or relevant
untracked-code status, Python/environment, machine, timing-process, or
process-version drift while allowing benchmark-source-only drift from additive
coverage updates. Use
`--replace-legacy-cache` / `--speed-replace-legacy-cache` only for intentional
full recapture after benchmark process or baseline-identity changes.

For harness experiments that touch timing-process files such as
`parity_speed.py` or `_threading.py`, use a throwaway branch or an output/cache
path outside the committed artifacts. Timing-process hash mismatches are
intentional red flags; revert experimental harness edits before reusing the
canonical cache, or explicitly replace the cache after an intentional benchmark
process change.

```bash
pdm run rust-parity-legacy-cache-refresh
```

Normal canonical gates reuse that cache and fail loudly if a requested legacy
entry is missing or stale.

Single-API iteration during a port:

```bash
.venv/bin/python -m migration.parity.parity_main \
    --apis dynamics.propagate_2body --speed-n 5000
```

Just the parity-fuzz half:

```bash
.venv/bin/python -m migration.parity.parity_fuzz \
    --seeds 8 --n 128 --output migration/artifacts/parity_fuzz.json
```

Just fixed fixtures:

```bash
.venv/bin/python -m migration.parity.parity_fixed \
    --output migration/artifacts/parity_fixed_fixtures.json
```

The `--n` value is the default per-seed workload. Expensive scalar optimizer
surfaces may declare smaller canonical per-seed overrides in `_inputs.py` so
coverage remains direct and randomized without making every full fuzz run
optimizer-bound. NaN positions are part of the parity contract: one mismatched
NaN cell (`rust` NaN / legacy finite, or the reverse) fails the gate even when
all finite cells satisfy the numeric tolerance.

Broader tolerance-envelope sweeps can be run outside the deterministic review
gate with larger and rotating seeds, for example:

```bash
pdm run python -m migration.parity.parity_main \
    --skip-speed --threads multi-thread \
    --fuzz-seeds 64 --fuzz-n 1024 \
    --base-seed $(date +%Y%m%d) \
    --output /tmp/parity_gate_large_sweep.json
```

Use these large sweeps as periodic/nightly evidence for future tolerance
tightening; keep the canonical CI/review seed fixed for reproducibility. Do not
automatically rewrite tolerances from a single green artifact: proposed
tightening should be reviewed with physical-unit rationale, headroom, and
large-sweep evidence so validity policy does not drift silently.

Just the speedup half (warm only — default, small lane only):

```bash
.venv/bin/python -m migration.parity.parity_speed \
    --threads single --n 2000 --reps 7 \
    --output migration/artifacts/parity_speed.json
```

Active performance waivers are read from `migration/waivers.yaml` and legacy
registry waiver fields. Current waivers, if explicitly approved, must include
the failing lane (for example `lane: large-n`) so a tiny/small pass cannot hide
a large-workload miss. Waived rows still record their raw p50/p95 miss in the
JSON artifact, but no active large-n waiver should be used for merge-readiness
without a fresh user decision.

Add `--cold` to additionally measure cold-call latency. Cold timing
spawns a fresh Python subprocess per measurement (so each call pays
process startup + module import + JIT compile cost). This matters for
one-shot CLIs / mission-design scripts where rust wins 18-76× over
legacy.

```bash
.venv/bin/python -m migration.parity.parity_speed \
    --threads single \
    --tiny --tiny-reps 101 --tiny-warmup 3 \
    --n 2000 --reps 21 --warmup 3 --cold \
    --large --large-reps 7 --large-warmup 1 --large-cold \
    --legacy-cache migration/artifacts/parity_legacy_speed_baseline.json \
    --output migration/artifacts/parity_speed_cold_warm.json
```

This canonical cold/warm artifact contains enforced `tiny-n`, `small-n`, and
`large-n` lanes. The large lane records structured shape labels such as
`orbits=400 × epochs=50 (20000 rows)` and collects cold-call timing when
`--large-cold` is present.

The cold/warm review gate intentionally uses more warm timing samples than the
quick warm-only command. The tiny lane uses 101 reps because microsecond-scale
p95 is otherwise dominated by a single scheduler outlier; the historical small
lane uses 21 reps, while large workloads are millisecond-scale and keep 7 reps.
Those repetitions are per trial; the built-in serial trial aggregation is always
applied on top of them for canonical pass/fail evidence.

## Pretty-Printing Review Tables

When presenting parity/performance tables for review, use the canonical
pretty-printer rather than ad hoc JSON extraction. It joins
`parity_gate.json` with `tolerances.py`, so the parity table includes
the tolerance rationale, physical magnitude, root cause, and verdict.

```bash
.venv/bin/python -m migration.scripts.parity_table \
    --parity-artifact migration/artifacts/parity_gate.json \
    --speed-artifact migration/artifacts/parity_speed_cold_warm.json \
    --json-output migration/artifacts/parity_table_rca.json \
    --markdown-output migration/artifacts/parity_report.md
```

Markdown output keeps the full untruncated rationale/RCA text by default.
Use `--max-text 120` only for compact console summaries. The JSON output
always keeps the full text.

## Adding a new rust-default API

1. Add an `ApiMigration` row in `src/adam_core/_rust/status.py`.
2. Add a `ToleranceSpec` entry in
   `migration/parity/tolerances.py`. Lead with the rationale pulled from
   your port's journal entry.
3. Add a dispatch entry in `_rust_runner.py` (rust side) and
   `_legacy_runner.py` (legacy side). Output keys must match
   `tolerances.py` output names.
4. Add an input generator in `_inputs.py` that returns a `Sample` with
   `rust_kwargs` and `legacy_kwargs` (often the same dict).
5. Run `parity_main --apis <your.api>` and confirm both gates pass.
6. If the legacy implementation is being removed or the current-branch legacy
   path would call Rust, preserve dated evidence first, then track future
   performance with `pdm run rust-latency-gate`.

## Coverage

The harness covers the baseline-main scope: rust-default APIs wired in
`_inputs.GENERATORS`, plus selected orchestration functions whose correctness
is bounded by underlying kernel parity. Higher-level Python wrappers
(e.g. `Orbits.propagate_to`, `EphemerisMixin.generate_ephemeris`, OD
`LeastSquares`, `Variants`, `Residuals`) are pure-Python compositions over
rust-default kernels; their parity follows from the wired primitives.

`src/adam_core/_rust/status.py` is the source of truth for coverage state. The
pretty-printer joins this registry to `tolerances.py` so reports distinguish
direct randomized fuzz, fixed-fixture coverage, orchestration-implied coverage,
targeted-test-only coverage, and intentional exclusions.

Currently wired directly in randomized fuzz (42):
- 14 coordinates/transform/residual surfaces
- 15 dynamics/missions primitives and public orchestration surfaces, including
  direct MOID, batch MOID, perturber MOIDs, Tisserand parameter, propagation
  arc helpers, LT-correction, Lambert, and raw/public porkchop grids
- 6 photometry surfaces, including H-fit row and grouped kernels
- 1 orbit-classification rule surface
- 4 OD primitives, including constrained shared-root `gaussIOD` fuzz
- 2 raw statistics kernels, with speed tracked as diagnostic Rust-vs-BLAS
  comparisons because public wrappers remain NumPy/BLAS-backed

Supplemental fixed-fixture parity:
- `dynamics.calculate_moid` — randomized fuzz covers non-degenerate optimizer
  rows; a deterministic identical-circular fixture covers the flat-minimum
  regime where the MOID distance is unique but the returned argmin time is only
  an optimizer witness within the orbital period. A second non-degenerate
  fixture pins a unique-minimum `dt_at_min` to `1e-6` day so optimizer time
  regressions cannot hide behind the randomized slack.
- `dynamics.propagate_2body_with_covariance` — randomized fuzz covers typical
  covariance propagation against baseline-main Jacobian covariance propagation;
  a deterministic high-a, 2516-day fixture compares Rust Dual covariance output
  to a central finite-difference witness from the scalar state map.
- `dynamics.generate_ephemeris_2body_with_covariance` — randomized fuzz covers
  typical ephemeris covariance propagation; a deterministic distant-object,
  stellar-aberration fixture compares Rust covariance output to a central
  finite-difference witness from the scalar ephemeris map.
- `orbit_determination.gaussIOD` — constrained randomized fuzz covers
  well-conditioned low-e, main-belt-like, multi-day triplets where Rust
  Laguerre+deflation and legacy `np.roots`/LAPACK share a physical best root.
  The canonical gate also enforces eight deterministic well-conditioned
  triplets as a stable supplement; unconstrained ill-conditioned/multi-root
  triplets remain excluded because solver root-subset policy can differ.

`coordinates.transform_coordinates` has two complementary gates. Direct
randomized fuzz covers a public quivr-object dispatcher matrix: Cartesian
constant-frame inverse directions, Spherical/Keplerian/Cometary non-Cartesian
inputs, representative covariance-bearing Cartesian/Keplerian dispatcher paths,
SUN↔EARTH origin translations, and Earth-centered ITRF93 time-varying rotations
at vetted PCK epochs. The fixed legacy-frozen branch fixture
`migration/artifacts/transform_coordinates_branch_fixture_2026-07-06.json`
then pins every public dispatcher branch that is too specific for random fuzz:
identity/no-op returns, validation errors, geodetic output, Cartesian
frame-only and Cartesian-origin fallback paths, non-Cartesian ITRF93 fallback,
finite-covariance ITRF93, mixed-origin arrays, and MPC observatory-origin
translation.

The ITRF93 rows intentionally compare asymmetric implementations: the
baseline-main oracle uses the legacy CSPICE/spiceypy PCK path, while the
migration path uses spicekit's pure-Rust PCK evaluator. The `3e-8` tolerance is
therefore scoped only to the known Earth-rotation backend divergence documented
and independently spec-validated in the SPICE/spicekit tests. The underlying
rotation-matrix disagreement is last-ULP scale, but the time-varying rotation
feeds r×ω and the spherical velocity-angle Jacobian, amplifying it into
~1e-8 deg/day vlon/vlat drift. It is not a blanket SPICE tolerance. SUN↔EARTH
origin-translation rows do not use that budget and are held to `1e-11` in the
transform matrix.

Arbitrary user-furnished SPICE body coverage beyond the SUN/EARTH/MPC-observatory
fixture matrix remains a fixture-extension task when new kernels/bodies are
added; it is no longer a transform dispatcher branch-coverage gap.

## Files

| file | role |
|------|------|
| `tolerances.py` | per-API tolerance table + rationale |
| `_inputs.py` | randomized input generators per API |
| `_rust_runner.py` | rust dispatch — main-venv side |
| `_legacy_runner.py` | legacy dispatch — runs inside `.legacy-venv` via subprocess |
| `_oracle.py` | subprocess helper that talks to `_legacy_runner` |
| `parity_fuzz.py` | randomized parity gate (rust vs legacy outputs) |
| `parity_fixed.py` | deterministic fixed-fixture parity gate |
| `parity_speed.py` | 1.2× speedup gate against baseline-main timing |
| `parity_main.py` | orchestrator running parity + speed gates → JSON artifact |
