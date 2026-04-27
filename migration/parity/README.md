# Parity + speedup gate

Constitutional rust-vs-legacy enforcement for the Rust migration. Two
gates run side-by-side:

1. **Parity-fuzz** — randomized inputs, every rust-default API must
   match the upstream JAX/numba legacy implementation within the
   per-API tolerance defined in `tolerances.py`.
2. **Speedup** — every rust-default API must be ≥ 1.2× faster than
   legacy at p50 and p95 latency on identical workloads.

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

Full constitutional gate (writes `migration/artifacts/parity_gate.json`):

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

Add `--cold` to additionally measure cold-call latency. Cold timing
spawns a fresh Python subprocess per measurement (so each call pays
process startup + module import + JIT compile cost). This matters for
one-shot CLIs / mission-design scripts where rust wins 18-76× over
legacy.

```bash
.venv/bin/python -m migration.parity.parity_speed \
    --n 2000 --reps 7 --cold \
    --output migration/artifacts/parity_speed_cold_warm.json
```

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

## Coverage

The harness covers the constitutional scope: every rust-default API in
`src/adam_core/_rust/status.py` plus orchestration functions that
compose them. Higher-level Python wrappers (e.g. `Orbits.propagate_to`,
`EphemerisMixin.generate_ephemeris`, OD `LeastSquares`, `Variants`,
`Residuals`) are pure-Python compositions over rust-default kernels;
their parity follows from the wired primitives.

Currently wired (21):
- 7 coordinates representation transforms
- 6 dynamics primitives + 1 LT-correction orchestration
- 4 photometry
- 3 OD primitives

Known gaps (5):
- `coordinates.transform_coordinates` — needs Python+pyarrow path through quivr
- `orbit_determination.gaussIOD` — variable-length output (0–3 orbits) makes byte-by-byte parity tricky
- `dynamics.calculate_perturber_moids` — orchestration over Orbits quivr table
- `dynamics.generate_porkchop_data` — orchestration over Orbits quivr table

These need a different harness style (Python+pyarrow round-trip rather
than numpy-boundary subprocess hand-off). Track as follow-up work.

## Files

| file | role |
|------|------|
| `tolerances.py` | per-API tolerance table + rationale |
| `_inputs.py` | randomized input generators per API |
| `_rust_runner.py` | rust dispatch — main-venv side |
| `_legacy_runner.py` | legacy dispatch — runs inside `.legacy-venv` via subprocess |
| `_oracle.py` | subprocess helper that talks to `_legacy_runner` |
| `parity_fuzz.py` | randomized parity gate (rust vs legacy outputs) |
| `parity_speed.py` | 1.2× speedup gate (rust vs legacy timing) |
| `parity_main.py` | orchestrator running both gates → JSON artifact |
