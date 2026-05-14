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

The legacy implementation lives in a sibling repo
`/Users/aleck/Code/adam-core` pinned to upstream `main`. Because both
repos export the package name `adam_core`, they cannot coexist in one
Python venv. We invoke the legacy implementation through a
**subprocess** running inside `.legacy-venv` (gitignored, set up
once):

```bash
python3.13 -m venv .legacy-venv
.legacy-venv/bin/pip install -e /Users/aleck/Code/adam-core
```

Verify it's reachable:

```bash
.venv/bin/python -m migration.parity._oracle
# → [oracle smoke] cart→sph OK, ...
```

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

Currently wired directly in randomized fuzz (38):
- 14 coordinates/transform/residual surfaces
- 11 dynamics primitives and public orchestration surfaces, including direct
  MOID, perturber MOIDs, Tisserand parameter, LT-correction, and porkchop grids
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

`coordinates.transform_coordinates` is marked partial in the registry, but
direct randomized fuzz now covers a public quivr-object dispatcher subcase
matrix: Cartesian constant-frame inverse directions, Spherical/Keplerian/
Cometary non-Cartesian inputs, representative covariance-bearing
Cartesian/Keplerian dispatcher paths, SUN↔EARTH origin translations, and
Earth-centered ITRF93 time-varying rotations at vetted PCK epochs. The
intentionally excluded subcases remain explicit: Cartesian-to-Cartesian
frame-only fallthrough,
covariance-bearing ITRF93 public dispatcher cases, mixed-origin arrays,
observatory origins, and user-furnished SPICE body coverage beyond the
SUN/EARTH matrix.

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

These remaining excluded/indirect cases need a different harness style, fixed
fixtures, or quivr round-trips rather than numpy-boundary random subprocess
hand-off.

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
