# Parity + Baseline Speed Gate

Baseline-main enforcement for the Rust migration. Two gates run side-by-side
for APIs wired into this harness:

1. **Parity-fuzz** — randomized inputs, current Rust output must match the
   upstream `main` implementation within the per-API tolerance defined in
   `tolerances.py`.
2. **Speedup** — current Rust must meet lane-specific p50/p95 thresholds
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
    --threads single \
    --speed-tiny --speed-tiny-reps 101 --speed-tiny-warmup 3 \
    --speed-n 2000 --speed-reps 21 --speed-warmup 3 \
    --speed-large --speed-large-reps 7 --speed-large-warmup 1 \
    --speed-legacy-cache migration/artifacts/parity_legacy_speed_baseline.json
```

The `tiny-n`, `small-n`, and `large-n` speed lanes are enforced by default at
1.2× p50/p95. Large-workload misses stay red under RM-P1-020 unless the user
makes an explicit structural-acceptance decision; `--speed-large-diagnostic` is
only for ad-hoc local probes.

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

The `--n` value is the default per-seed workload. Expensive scalar optimizer
surfaces may declare smaller canonical per-seed overrides in `_inputs.py` so
coverage remains direct and randomized without making every full fuzz run
optimizer-bound.

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
direct randomized fuzz, orchestration-implied coverage, targeted-test-only
coverage, and intentional exclusions.

Currently wired directly in randomized fuzz (25):
- 9 coordinates/transform/residual surfaces
- 8 dynamics primitives including direct MOID and LT-correction orchestration
- 4 photometry
- 1 orbit-classification rule surface
- 3 OD primitives

Covered indirectly through underlying kernel parity:
- `dynamics.calculate_perturber_moids` — orchestration over Orbits quivr table
- `dynamics.generate_porkchop_data` — orchestration over Orbits quivr table

Declared but intentionally unwired in randomized fuzz:
- `orbit_determination.gaussIOD` — variable-length output (0-3 orbits) and
  root-subset differences between Rust Laguerre+deflation and legacy
  `np.roots` make random byte-by-byte parity misleading.

`coordinates.transform_coordinates` is marked partial in the registry: direct
randomized fuzz now covers a public quivr-object dispatcher case
(`CartesianCoordinates` ecliptic -> equatorial into `SphericalCoordinates`).
The intentionally excluded subcases remain explicit: Cartesian-to-Cartesian
frame-only fallthrough, ITRF93/time-varying rotations, origin translation, and
remaining non-Cartesian representation combinations.

These excluded/indirect cases need a different harness style, fixed fixtures,
or quivr round-trips rather than numpy-boundary random subprocess hand-off.

## Files

| file | role |
|------|------|
| `tolerances.py` | per-API tolerance table + rationale |
| `_inputs.py` | randomized input generators per API |
| `_rust_runner.py` | rust dispatch — main-venv side |
| `_legacy_runner.py` | legacy dispatch — runs inside `.legacy-venv` via subprocess |
| `_oracle.py` | subprocess helper that talks to `_legacy_runner` |
| `parity_fuzz.py` | randomized parity gate (rust vs legacy outputs) |
| `parity_speed.py` | 1.2× speedup gate against baseline-main timing |
| `parity_main.py` | orchestrator running both gates → JSON artifact |
