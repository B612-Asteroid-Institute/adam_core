# Remaining Rust support surface

Updated 2026-07-14 after integrating upstream main at
`936cc636096fcfefcee3e1310c21528444f39546`.

## Runtime support summary

- The authoritative complete inventory is
  `migration/public_surface/manifest.json`: **576 symbols / 66 constants**.
  Domain dispositions are recorded under `migration/public_surface/`.
- The 44-row parity registry remains a selected migration benchmark set, not a
  public-API count. All enforced rows pass; raw-kernel diagnostic rows do not
  control public promotion.
- Every adam-core-owned non-plotting operation is Rust-backed, a thin compatible
  veneer over a native crossing, generic quivr/dataclass infrastructure, static
  data, or an explicit provider boundary.
- The complete latest-main `photometry.rotation` surface is present. Rust owns
  the estimator, frequency fits, confidence/alias policy, apparition selection,
  grouped solves, default observation assembly, and native timing.
- `adam-assist` owns ASSIST propagation, sampled covariance, ephemeris,
  collision/impact, OD/IOD, and scheduling orchestration. It consumes
  `libassist-sys` and `librebound-sys` directly; no `assist-rs` v2 is planned.

## Explicit compatibility and provider boundaries

These are supported boundaries, not hidden default backends:

- plotting/display and `Orbits.preview`;
- Astropy object conversion and UT1/IERS data;
- the historical Astroquery `SBDB.query` monkeypatch seam;
- Healpy compatibility helpers;
- the opt-in historical JAX batch-fit module (the public rotation estimator
  always uses Rust);
- abstract/external propagator calls and optional SPK propagation;
- live network providers and user-supplied/custom observer providers; and
- generic quivr projection, linkage, table construction, and serialization
  infrastructure where no adam-core algorithm is performed.

Ordinary TAI/TT/UTC/TDB conversion, ISO formatting, observing-night calculation,
geographic timezone lookup, SBDB access, HEALPix operations, MPC observers,
photometry, and default product generation do not require those optional
providers.

## Pure-Rust distribution work

The workspace now contains publishable permissive domain crates plus the public
`adam_core` umbrella crate. Crate metadata includes versions, licenses,
repository provenance, MSRV, and versioned path dependencies. Release-candidate
CI tests the workspace and packages crates in dependency order; trusted
publishing is prepared but cannot be enabled until crates.io receives each
crate's first manual release. No crate is published as part of this migration.

Large BSP assets remain external because they exceed crates.io limits.
`adam_core_rs_kernel_data` resolves them automatically in this order: explicit
override, installed Python package, cache, then verified wheel download. Offline
mode and installed-package discovery are tested without duplicate files.

## Native artifact targets

The supported wheel matrix is CPython 3.11-3.13 on manylinux 2.17 x86-64 and
AArch64, macOS Apple silicon and Intel, and Windows x86-64. Musllinux is
unsupported. Release publication downloads the exact clean-room-tested
artifacts and does not rebuild wheels.

## Deferred

Serialized Arrow `RecordBatch` transport remains deferred (`personal-cmy.11`)
until a concrete second process/consumer needs it; adding serialization now
would add copies without improving the in-process Rust APIs.
