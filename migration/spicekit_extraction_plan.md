# spicekit — Standalone Rust SPICE Kernel Toolkit Extraction Plan

Status: Rust crate extracted and consumed as a pinned git dependency;
  CSpice parity oracle carved out into `crates/spicekit-bench`; Python
  bindings carved out into `crates/spicekit-py` and consumed by
  adam-core as the `spicekit` Python package. Still pending:
  crates.io and PyPI publish.
Drafted: 2026-04-20
Last updated: 2026-04-21

## Motivation

The pure-Rust NAIF reader built for adam-core is a generally-useful piece
of infrastructure. It is a from-scratch reimplementation of the CSPICE
subset that adam-core actually consumes — no CSPICE linkage, no C FFI,
no global state — and nothing in its design is adam-core-specific.
Extracting it into a standalone MIT-licensed project (`spicekit`) gives
the broader Rust ecosystem a modern DAF/SPK/PCK reader and lets
adam-core consume it as a normal dependency instead of carrying it
internally.

The spicekit repo hosts three crates:

1. `crates/spicekit/` — the pure-Rust reader library (public Rust API,
   goes to crates.io).
2. `crates/spicekit-py/` — the PyO3 bindings (published to PyPI as the
   `spicekit` Python package; `publish = false` on crates.io).
3. `crates/spicekit-bench/` — the CSpice FFI parity oracle and
   performance benchmark harness (`publish = false`; links `cspice-sys`
   behind a feature flag and compares every code path against CSpice at
   machine-epsilon tolerance).

## Scope — what has moved

Everything that was in `rust/adam_core_rs_naif/` is now in
`crates/spicekit/src/` (Rust library) and `crates/spicekit-py/src/lib.rs`
(Python bindings):

| Module | What it is | Location |
|---|---|---|
| `daf.rs` | DAF container reader (memory-mapped, shared by SPK and PCK) | `crates/spicekit/` |
| `spk.rs` | Pure-Rust SPK reader (Type 2, Type 3, Type 9, Type 13) | `crates/spicekit/` |
| `spk_writer.rs` | Pure-Rust SPK writer (Type 3, Type 9) | `crates/spicekit/` |
| `pck.rs` | Binary PCK reader (time-varying ITRF93 Euler angles) | `crates/spicekit/` |
| `naif_ids.rs` | NAIF body-name ↔ code built-in table + `normalize_name` | `crates/spicekit/` |
| `naif_builtin_table.rs` | 692-entry const array mirrored from CSpice `zzidmap.c` | `crates/spicekit/` |
| `text_kernel.rs` | Text-kernel parser for `NAIF_BODY_NAME` / `NAIF_BODY_CODE` | `crates/spicekit/` |
| `frame.rs` | Frame constants (`OBLIQUITY_J2000_RAD`), sxform helpers | `crates/spicekit/` |
| `NaifSpk`, `NaifPck`, `NaifSpkWriter`, `naif_bodn2c`, `naif_bodc2n`, `naif_parse_text_kernel_bindings` PyO3 wrappers | | `crates/spicekit-py/` |
| CSpice FFI parity oracle (`cspice_wrap.rs`, `kernels.rs`, `tests/parity.rs`, `src/main.rs`) | | `crates/spicekit-bench/` |

Transitively: `memmap2`, `rayon`, and `thiserror` stay as spicekit's
Rust dependencies. spicekit-py adds `pyo3`, `numpy`, `ndarray`.

## What stays in adam-core

- `adam_core_py` — the PyO3 bindings layer for adam-core-specific
  coordinate / orbit-determination kernels. No longer depends on
  spicekit: all NAIF PyO3 wrappers have moved to `spicekit-py`.
- `adam_core_rs_coords`, `adam_core_rs_orbit_determination`,
  `adam_core_rs_autodiff` — unrelated to NAIF work.
- `src/adam_core/utils/spice_backend.py::RustBackend` — the backend
  adapter that composes `spicekit` Python primitives into adam-core's
  `SpiceBackend` protocol. Stays in adam-core; it's adam-core's backend
  wiring, not spicekit API.
- `src/adam_core/_rust/api.py::naif_*` — thin Python wrappers that
  delegate to the `spicekit` package and return `None` when spicekit is
  unavailable (review-period fallback contract).

## Public Rust API stabilized for v0.1

Stabilize only what `spicekit-py` and any external consumer actually
need. Anything not re-exported at the crate root stays internal.

- `daf::{DafFile, Summary, DafError}`
- `spk::{SpkFile, SpkSegment, SpkError}`
- `spk_writer::{SpkWriter, SpkWriterError, Type3Record, Type3Segment, Type9Segment}`
- `pck::{PckFile, PckSegment, PckError}`
- `naif_ids::{bodn2c, bodc2n, NaifIdError}`
- `text_kernel::{parse_body_bindings, parse_body_bindings_from_str, BodyBinding, TextKernelError}`
- `frame::{NaifFrame, OBLIQUITY_J2000_RAD, rotate_state, apply_sxform,
  invert_sxform, j2000_to_eclipj2000, pck_euler_rotation_and_derivative,
  sxform_from_rotation}`

## Public Python API stabilized for v0.1

Surface exposed by the `spicekit` Python package:

- `NaifSpk(path)`: `state`, `state_batch`, `state_batch_in_frame`,
  `segments`.
- `NaifPck(path)`: `euler_state`, `sxform`, `pxform`, `sxform_batch`,
  `pxform_batch`, `rotate_state_batch`, `segments`.
- `NaifSpkWriter(locifn)`: `add_type3`, `add_type9`, `write`.
- `naif_bodn2c(name)` / `naif_bodc2n(code)`.
- `naif_parse_text_kernel_bindings(path)`.

## Design decisions made

1. **Crate name on crates.io**: `spicekit` — confirmed available as of
   2026-04-21. README carries an explicit first-paragraph disclaimer
   distancing from NAIF/JPL.

2. **PyPI package name**: `spicekit` — confirmed available. Published
   from `crates/spicekit-py/` via maturin (`module-name =
   spicekit._rust_native`, `python-source = python`). The Python-side
   shim at `python/spicekit/__init__.py` re-exports the native symbols.

3. **License**: MIT. CSpice attribution for `naif_builtin_table.rs`
   lives in `LICENSE-NOTICES` per NAIF distribution terms.

4. **SPK types in v0.1**: reader supports Types 2, 3, 9, 13; writer
   supports Types 3, 9. Same surface as adam-core had pre-extraction.

5. **Binary PCK writer**: deferred; spicekit v0.1 is read-only for PCK.

6. **CSpice parity oracle location**: `crates/spicekit-bench/`, not
   adam-core. Keeps the assertion next to the code being asserted.

7. **MSRV**: not yet pinned; document on tag.

## Consumption model in adam-core

Phase-in history:

1. **Local path dep.** [DONE 2026-04-20] adam-core's
   `rust/adam_core_py/Cargo.toml` pointed at `~/Code/spicekit` via a
   local path dep. The in-tree `rust/adam_core_rs_naif` directory was
   removed and all `use adam_core_rs_naif::*` lines in
   `rust/adam_core_py/src/lib.rs` retargeted to `spicekit::*`.

2. **GitHub remote + git dep pin.** [DONE 2026-04-20] spicekit pushed
   to `ssh://git@github.com/B612-Asteroid-Institute/spicekit.git`
   (private). adam-core pinned to
   `rev = "afd16c2158b6ceee3571d1c320214d103268d4d2"`.

3. **spicekit-bench parity oracle carve-out.** [DONE 2026-04-20] The
   CSpice FFI wrapper, kernel-path resolution, and side-by-side parity
   integration tests moved into `crates/spicekit-bench/` in the
   spicekit repo. The `adam_core_rs_spice` crate was deleted from
   adam-core.

4. **spicekit-py carve-out.** [DONE 2026-04-21] `NaifSpk`, `NaifPck`,
   `NaifSpkWriter`, `naif_bodn2c`, `naif_bodc2n`,
   `naif_parse_text_kernel_bindings` and their helpers
   (`sxform_matrix`, `static_inter_inertial`, `matmul6`,
   `body_frame_code`, `is_inertial`, `parse_naif_frame`,
   `spk_err_to_py`, `pck_err_to_py`, `spk_writer_err_to_py`) moved
   from `rust/adam_core_py/src/lib.rs` into
   `crates/spicekit-py/src/lib.rs`. adam-core dropped the `spicekit`
   Rust git dep from `adam_core_py/Cargo.toml` (no longer needed).
   adam-core's `src/adam_core/_rust/api.py` naif_* wrappers now import
   from the `spicekit` Python package; `SPICEKIT_AVAILABLE` is the new
   capability flag, set at module import from `import spicekit`.
   `RustBackend.__init__` and `get_backend()` now check
   `SPICEKIT_AVAILABLE`. Full adam-core test suite (762 passing)
   confirmed green.

5. **crates.io publish** [TODO]. Once the git dep has been stable in
   adam-core for ~2 weeks, publish `spicekit = "0.1.0"` to crates.io.
   Switch adam-core's `Cargo.toml` from git dep to a versioned
   crates.io dep.

6. **PyPI publish** [TODO]. Publish `spicekit` Python wheel via
   `maturin publish` once the native API has been stable in adam-core
   for ~2 weeks. Switch adam-core from `maturin develop` of the local
   path to `pip install spicekit>=0.1`.

7. **Regression policy.** Any future adam-core change that needs a new
   spicekit API first lands in spicekit (tagged release, git rev bump,
   or new maturin-built wheel), then adam-core upgrades. Prevents
   adam-core from accumulating local spicekit patches that drift the
   two projects apart.

## Parity-test oracle — where it lives

The split is three layers deep.

**spicekit library (`crates/spicekit/`)** is self-contained: DAF-reader
roundtrip, Chebyshev polynomial exactness, Type 9 Lagrange
exact-at-knots, text-kernel parser, Euler-angle rotation orthogonality,
etc. No CSpice linkage anywhere in the library or its tests — that is
its selling point.

**spicekit-bench (`crates/spicekit-bench/`)** is a separate workspace
member with `publish = false` that feature-gates `cspice-sys`
(`default = ["cspice"]`) and runs the side-by-side comparison. It owns:

- `src/lib.rs` — minimal spicekit-side `Backend` dispatcher that owns
  a list of `SpkFile` + `PckFile` + text-kernel bindings, replicating
  the furnsh/unload/spkez_batch/pxform_batch/sxform_batch/bodn2c
  semantics of adam-core's Python `RustBackend` so the comparison is
  apples-to-apples with what CSpice does after a batch of `furnsh_c`
  calls.
- `src/cspice_wrap.rs` — Mutex-guarded CSpice FFI wrapper (furnsh,
  unload, spkez, pxform, sxform, bodn2c).
- `src/kernels.rs` — resolves the six canonical kernel paths
  (mirroring adam-core's `DEFAULT_KERNELS`) from the `naif-*` PyPI
  packages, with env-var fallback so CI doesn't spawn Python
  subprocesses during tests.
- `src/main.rs` — microbench binary that loads both backends with
  identical kernels and prints a markdown table across the case
  matrix.
- `tests/parity.rs` — cargo integration tests ported verbatim from
  adam-core's `test_spice_backend.py` at the same tolerances
  (`rtol=1e-14, atol=1e-7` for spkez, `atol=1e-12` for pxform,
  `atol=1e-11` for sxform, plus text-kernel binding semantics).

**spicekit-py (`crates/spicekit-py/`)** exposes the same surface to
Python. No CSpice linkage. Correctness is inherited transitively from
the underlying spicekit crate — the Python bindings are
stateless marshaling only.

**CI** (`.github/workflows/bench.yml`) runs on Ubuntu,
`uv pip install`s the six `naif-*` packages, resolves their paths into
env vars, runs `cargo test --release -p spicekit-bench` for parity,
then `cargo run --release --bin spicekit-bench` for the performance
table (uploaded as artifact).

**What stays in adam-core.** The adam-core-specific parity harness —
the Python `RustBackend` unit tests (`test_spice_backend.py`) —
continues to verify that adam-core's backend adapter correctly composes
the spicekit primitives. The raw spicekit-vs-CSpice comparison has
moved to `spicekit-bench`, so that assertion lives with the code being
asserted.

## Repo state

- [x] `crates/spicekit/` Cargo crate with full NAIF reader surface
- [x] `crates/spicekit-bench/` parity oracle + benchmark harness
- [x] `crates/spicekit-py/` PyO3 bindings + maturin packaging
- [x] `README.md` with scope, usage, NAIF disclaimer, Python-bindings
      section
- [x] `LICENSE` (MIT) + `LICENSE-NOTICES` (CSpice attribution for
      `naif_builtin_table.rs`)
- [x] GitHub remote at
      `ssh://git@github.com/B612-Asteroid-Institute/spicekit.git` (private)
- [ ] GitHub Actions: `cargo test`, `cargo fmt --check`, `cargo clippy`,
      MSRV check, maturin wheel build
- [ ] Git tag `v0.1.0` once adam-core has run clean against the git dep
      for two weeks
- [ ] `cargo publish` (spicekit, spicekit-py is `publish = false`)
- [ ] `maturin publish` (spicekit-py → PyPI as `spicekit`)

## Explicit non-goals for v0.1

- No async I/O. Memory-mapped sync reads only.
- No binary PCK writer.
- No additional SPK types beyond 2/3/9/13.
- No CLI tool.
- No support for text-kernel keys other than `NAIF_BODY_NAME` /
  `NAIF_BODY_CODE`. Leapseconds, SCLK, IK / FK files parse without
  error (the parser skips everything else) but their contents are not
  exposed.

These are all reasonable v0.2+ additions driven by external demand.

## What finishes this task

adam-core's `Cargo.toml` depends on `spicekit = "0.1"` from crates.io;
adam-core's pyproject depends on `spicekit>=0.1` from PyPI; full
adam-core test suite and benchmark are green against the published
crate and published wheel.
