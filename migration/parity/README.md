# Parity + Baseline Speed Gate

Baseline-main enforcement for the Rust migration. Two gates run side-by-side
for APIs wired into this harness:

1. **Parity-fuzz** — randomized inputs, current Rust output must match the
   upstream `main` implementation within the per-API tolerance defined in
   `tolerances.py`.
2. **Speedup** — current Rust must be >= 1.2x faster than the upstream `main`
   implementation at p50 and p95 latency on identical workloads, unless an
   explicit waiver is attached.

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
.venv/bin/python -m migration.parity.parity_main
```

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

Just the speedup half (warm only — default):

```bash
.venv/bin/python -m migration.parity.parity_speed \
    --n 2000 --reps 7 --output migration/artifacts/parity_speed.json
```

Active performance waivers are read from `src/adam_core/_rust/status.py`.
Waived APIs still record their raw p50/p95 miss in the JSON artifact, but
the gate reports them as `WAIVED` and exits successfully. Every waiver must
also be recorded in `migration/waivers.yaml` with an owner, review date,
and exit criteria.

Add `--cold` to additionally measure cold-call latency. Cold timing
spawns a fresh Python subprocess per measurement (so each call pays
process startup + module import + JIT compile cost). This matters for
one-shot CLIs / mission-design scripts where rust wins 18-76× over
legacy.

```bash
.venv/bin/python -m migration.parity.parity_speed \
    --n 2000 --reps 21 --warmup 3 --cold \
    --output migration/artifacts/parity_speed_cold_warm.json
```

The cold/warm review gate intentionally uses more warm timing samples than the
quick warm-only command. Several Rust APIs complete in tens of microseconds, so
7 reps makes p95 behave like a single scheduler-outlier detector rather than a
stable latency estimate.

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

Currently wired directly in randomized fuzz (22):
- 8 coordinates/transform surfaces
- 7 dynamics primitives including LT-correction orchestration
- 4 photometry
- 3 OD primitives

Covered indirectly through underlying kernel parity:
- `dynamics.calculate_perturber_moids` — orchestration over Orbits quivr table
- `dynamics.generate_porkchop_data` — orchestration over Orbits quivr table

Declared but intentionally unwired in randomized fuzz:
- `orbit_determination.gaussIOD` — variable-length output (0-3 orbits) and
  root-subset differences between Rust Laguerre+deflation and legacy
  `np.roots` make random byte-by-byte parity misleading.

`coordinates.transform_coordinates` is marked partial in the registry: direct
randomized fuzz covers only the raw Cartesian ecliptic/equatorial frame
rotation kernel. RM-P1-009 owns public-dispatch parity for broader quivr
call shapes, ITRF93/time-varying rotations, origin translation, and composed
representation conversions.

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
