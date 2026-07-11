# Remaining Rust support surface

Updated after the Arrow-native epic (`personal-cmy.36`) and ASSIST ownership
handoff (`personal-yio`). “Unsupported” is separated from missing benchmark
instrumentation: a blank native timing is not a runtime support failure.

## Runtime support summary

- **Canonical adam-core parity registry:** 44/44 surfaces have a Rust-backed
  compatible Python path (34 public Rust defaults, 9 Rust raw kernels, and
  `Observers.from_codes` with a Rust default plus fallback).
- **Typed-table Arrow audit:** complete; every audited typed-table surface is a
  one-crossing Arrow facade or explicitly classified as a NumPy-flat numeric
  kernel.
- **adam-assist package semantics:** Rust-backed propagation, sampled covariance,
  ephemeris, collision/impact workflows, and per-orbit least-squares are
  available through `adam_assist.ASSISTPropagator`.

## Supported only as Python + Rust composition (not yet Rust-only)

1. **Top-level OD batch / differential correction** (`personal-cmy.33.4`).
   The ASSIST per-orbit Gauss-Newton fit is Rust-native, but adam-core still
   performs post-fit `evaluate_orbits` and multi-orbit scheduling in Python/Ray.
   Fit + evaluation + batch distribution are not one Rust crossing.
2. **Top-level IOD orchestration** (`personal-cmy.33.10`).
   `calcGauss` and `gaussIOD` are Rust-backed, but `iod_worker` candidate
   selection, iterative refinement/differential correction, linkage batching,
   and Ray distribution remain Python orchestration.
3. **Space-based/custom-kernel observer fallback** (`personal-cmy.33.7`).
   Ground MPC-code `Observers.from_codes` is Arrow-native Rust. Space-based or
   custom-kernel cases can still fall back to Python rather than staying in the
   single Rust crossing.
4. **Network and file-product facades (intentional I/O boundary).** SBDB,
   Horizons, NEOCC, Scout, OEM, and OpenSpace-facing Python APIs retain Python
   HTTP/file orchestration; deterministic numeric parsing/rendering is Rust
   where migrated. A standalone Rust HTTP/product client is not currently a
   migration requirement.

## Not yet independently deployable as a pure-Rust stack

1. **Kernel/data discovery** (`personal-3uy`). Rust SPICE/ASSIST still relies on
   BSP assets distributed by Python data packages or explicit paths. Dedicated
   Rust data crates / first-class asset discovery are not complete.
2. **Published Rust crates/API.** The internal adam-core crates and downstream
   `adam_assist_rs` are buildable, but are not yet a stable, versioned standalone
   Rust SDK replacing the Python distributions.
3. **Upstream ASSIST primitive release** (`personal-cmy.8`). The downstream
   backend remains pinned to assist-rs PR #11 revision pending merge/publication.

## Runtime-supported surfaces missing qualifying native-Rust timing

These APIs run Rust today, but lack a Rust-owned `Instant` adapter, so the
native column remains blank under `personal-98v.1`:

- `coordinates.cartesian_to_cometary`
- `coordinates.cartesian_to_geodetic`
- `coordinates.cartesian_to_keplerian`
- `coordinates.cartesian_to_spherical`
- `coordinates.cometary.to_cartesian`
- `coordinates.keplerian.to_cartesian`
- `coordinates.residuals.apply_cosine_latitude_correction`
- `coordinates.residuals.bound_longitude_residuals`
- `coordinates.residuals.calculate_chi2`
- `coordinates.rotate_cartesian_time_varying`
- `coordinates.spherical.to_cartesian`
- `coordinates.transform_coordinates_with_covariance`
- `dynamics.add_light_time`
- `dynamics.calc_mean_motion`
- `dynamics.calculate_moid`
- `dynamics.calculate_moid_batch`
- `dynamics.propagate_2body_along_arc`
- `dynamics.propagate_2body_arc_batch`
- `dynamics.propagate_2body_with_covariance`
- `dynamics.solve_lambert`
- `dynamics.tisserand_parameter`
- `missions.porkchop_grid`
- `orbit_determination.calcGauss`
- `orbit_determination.calcGibbs`
- `orbit_determination.calcHerrickGibbs`
- `orbits.classify_orbits`
- `photometry.calculate_apparent_magnitude_v`
- `photometry.calculate_apparent_magnitude_v_and_phase_angle`
- `photometry.calculate_phase_angle`
- `photometry.fit_absolute_magnitude_grouped`
- `photometry.fit_absolute_magnitude_rows`
- `photometry.predict_magnitudes`
- `statistics.weighted_covariance`
- `statistics.weighted_mean`

All adam-assist benchmark lanes (17 propagation, 6 covariance, and 3 collision
lanes) likewise lack Rust-owned `Instant` adapters. Their public veneer speedups
are valid; native-Rust speedups must remain blank until downstream adapters are
implemented.

## Surfaces with qualifying native-Rust timing today

- `coordinates.residuals.Residuals.calculate`
- `coordinates.transform_coordinates`
- `dynamics.calculate_perturber_moids`
- `dynamics.generate_ephemeris_2body`
- `dynamics.generate_ephemeris_2body_with_covariance`
- `dynamics.generate_porkchop_data`
- `dynamics.propagate_2body`
- `observers.Observers.from_codes`
- `orbit_determination.gaussIOD`
- `orbits.VariantOrbits.create`
