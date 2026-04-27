# Historical Rust-vs-Legacy Benchmark Snapshots

This directory preserves benchmark artifacts captured while a pure-JAX/Numba
legacy path was still callable and measurable alongside the Rust fast path.
Once legacy code was deleted from production, the live bench stopped being
able to produce a meaningful "legacy" column — these snapshots are the
historical record of what Rust was compared against during the migration.

## Files

### `rust_vs_legacy_final_snapshot_2026-04-23.json`

Verbatim copy of `migration/artifacts/rust_benchmark_gate.json` as of
2026-04-23, on the reference dev machine (Apple M1, Darwin 23.6). Captured
via `migration/scripts/rust_backend_benchmark_gate.py` immediately before
the gate script was rewritten to Rust-only.

Each entry has `speedup_p50` / `speedup_p95` = `legacy_p50 / rust_p50`.
Higher is better for Rust.

## Per-API validity notes

Most entries are legitimate JAX-vmap-vs-Rust or Numba-vs-Rust measurements:
they call a pure-JAX `_*_vmap` (or numba `_calc_*_numpy_legacy`) for the
legacy side and the corresponding Rust `*_numpy` kernel for the Rust side.
See the table below for exceptions.

| API | speedup p50 / p95 | note |
|---|---|---|
| `cartesian_to_spherical` | 4.35 / 3.66 | valid (JAX vmap vs Rust numpy) |
| `cartesian_to_geodetic` | 11.34 / 7.95 | valid |
| `cartesian_to_keplerian` | 3.47 / 3.24 | valid |
| `keplerian_to_cartesian` | 34.37 / 10.75 | valid |
| `cartesian_to_cometary` | 4.10 / 2.88 | valid |
| `cometary_to_cartesian` | 43.54 / 48.91 | valid |
| `spherical_to_cartesian` | 7.40 / 6.21 | valid |
| `calc_mean_motion` | 2.35 / 2.10 | valid |
| `propagate_2body` | 50.29 / 37.37 | valid |
| `propagate_2body_with_covariance` | 487.34 / 369.19 | valid (JAX `_propagate_2body` + `transform_covariances_jacobian` vs Rust Dual<6>) |
| `generate_ephemeris_2body` | 982.52 / 857.76 | valid |
| `generate_ephemeris_2body_with_covariance` | 912.46 / 698.06 | valid (JAX + `transform_covariances_jacobian` vs Rust Dual<6>) |
| `calculate_phase_angle` | 6.09 / 1.43 | valid |
| `calculate_apparent_magnitude_v` | 3.22 / 2.14 | valid |
| `calculate_apparent_magnitude_v_and_phase_angle` | 6.05 / 5.80 | valid |
| `calc_gibbs` | 42.68 / 42.56 | valid (Numba vs Rust) |
| `calc_herrick_gibbs` | 3.89 / 3.81 | valid (Numba vs Rust) |
| `calc_gauss` | 2.34 / 2.34 | valid (Numba vs Rust) |
| `spherical_from_cartesian` | 1.77 / 0.82 | **deprecated API** — referred to the `cartesian_to_spherical_arrow` wrapper, which was removed on 2026-04-23. Use `cartesian_to_spherical` for the numpy-boundary equivalent (4.35× / 3.66×). |
| `transform_coordinates` | 1.00 / 0.97 | **CONTAMINATED — do not cite.** The bench harness patched `_try_transform_coordinates_rust` to force the fallthrough, but the fallthrough path calls `coord_class.to_cartesian()` / `.from_cartesian()`, which internally dispatch to `rust_covariance_transform` (Rust). Both "legacy" and "Rust" sides were measuring Rust work. See clean historical number below. |

## Clean `transform_coordinates` historical number

From `journal.md` entry `2026-04-17 (promotion): Phase 5`:

> Measured end-to-end rust-vs-legacy on a 10k Cartesian-with-covariance →
> Keplerian workload: rust p50 11.3 ms vs legacy 20.8 ms
> (**1.84x p50 / 1.70x p95**). Representation-change variants
> (cart→spherical, cart→cometary, cart→kep with frame change) all in
> 1.44x–2.77x p50 range.

This measurement was taken after Phase 4 wired `transform_coordinates_with_covariance_numpy`
into the top-level dispatcher on 2026-04-17, but **before** 2026-04-20
when the coord-class `.to_cartesian()` / `.from_cartesian()` methods gained
their own `rust_covariance_transform` fast path. At that point, the
"legacy" dispatcher fall-through was genuinely pure JAX via
`coord_class.to_cartesian()` → `jax.jit`-compiled `_keplerian_to_cartesian_a_vmap`
+ `transform_covariances_jacobian`. That's a fair JAX-vs-Rust comparison.

**Authoritative `transform_coordinates` speedup for the migration record:
1.84× p50 / 1.70× p95** on 10k Cartesian-with-covariance → Keplerian.

## Why we stopped live legacy measurement

By 2026-04-23 the fallback branches that routed through JAX/`transform_covariances_jacobian`
in production coord classes had been deleted as unreachable (the "no rustless
environment" decision, task #121). That made the `transform_coordinates`
"legacy" bench path route through Rust internally, contaminating the
comparison. The live legacy measurement also gratuitously imported the
JAX `_*_vmap` kernels and Numba `_calc_*_numpy_legacy` helpers, blocking
the code-sweep cleanup of those files.

The bench was then rewritten to Rust-only regression tracking against a
pinned `migration/artifacts/rust_latency_baseline.json`. These history
snapshots are the permanent record of the JAX/Numba baseline.
