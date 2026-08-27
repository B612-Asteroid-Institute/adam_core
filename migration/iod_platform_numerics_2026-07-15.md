# Gauss IOD platform-numerics investigation (2026-07-15)

## Scope

The pinned legacy orchestration fixture in
`migration/artifacts/iod_orchestration_fixture_2026-07-12.json` was captured on
macOS arm64. Hosted Linux x86-64 CI produced the same selected observations and
candidate ordering, but the fitted state differed by as much as
`1.7443380073700610e-11 AU`, beyond the original scalar `4e-12` state bound.
This note records the numerical investigation performed before changing that
bound.

## Findings

There were two independent effects.

### 1. The fused provider inherited the wrong Gauss constant

The original Python orchestration owned candidate generation in adam-core:
`iod()` called `gaussIOD()` without a `mu` argument, so Gauss used adam-core's
DE44x value:

```text
0.00029591220828411956 AU^3/day^2
```

Adam-assist only propagated and scored the resulting candidates. The migrated
fused provider hook had instead inherited Gaussian `k^2`:

```text
0.00029591220828559115 AU^3/day^2
```

With otherwise identical Linux inputs, the composed path matched
`gaussIOD(mu=Constants.MU)` and the fused path matched
`gaussIOD(mu=0.01720209895^2)`. The fix is ownership-preserving: adam-core now
passes `Constants.MU` and `Constants.C` explicitly through the existing fused
provider crossing. This does not modify ASSIST propagation physics or add a
crossing.

### 2. The remaining difference is inherited platform sensitivity

The same byte-identical RA/Dec, MJD, and observer-position arrays were fed to
the current Rust kernel on macOS arm64 and to the accepted manylinux x86-64
wheel in an amd64 container. The first candidate's ecliptic `y` differed by
`1.3401724174857412e-11 AU` even though the inputs were bit-identical.

The first divergence occurs in platform trigonometry. For two declinations,
macOS and Linux cosine results differ by one ULP:

```text
input 0x1.d773344edb880p-2:
  macOS 0x1.cab0158d14323p-1
  Linux 0x1.cab0158d14322p-1

input 0x1.d8b536373e422p-2:
  macOS 0x1.ca6875e6d6286p-1
  Linux 0x1.ca6875e6d6287p-1
```

After equatorial-to-ecliptic rotation and normalization, cancellation changes
one direction-vector component by up to ten ULPs. The scalar triple product
and largest positive Gauss root then become:

```text
                         macOS arm64                 Linux x86-64
V                        8.465164707711914e-7         8.465164707735767e-7
largest positive root    2.621585784693498            2.6215857846802537
```

The root difference is `1.3244516596784277e-11 AU`. This preliminary-orbit
geometry is therefore mildly ill-conditioned with respect to last-bit
trigonometric differences.

This is not a Rust-only divergence. A standalone execution of the verbatim
legacy NumPy formulas on Linux produced largest root
`2.6215857846802604`, only `6.7e-15 AU` from Rust Linux. Their first-candidate
positions agreed within `9e-16 AU`; velocity differences from operation order
were below `3e-14 AU/day`. On macOS, the actual legacy adam-core `gaussIOD`
and current Rust kernel produced the same first-candidate state for the fixed
inputs.

## Acceptance policy

The same-platform fused-versus-composed test remains the primary check that
native orchestration preserves the current public computation. The historical
macOS-arm64 fixture continues to check selected observations, ordering,
metadata, residuals, chi-squared values, and the state, but its state uses
component-specific cross-platform bounds:

```text
[4e-12, 2e-11, 4e-12, 4e-13, 4e-13, 4e-13]
```

The only substantially expanded component is ecliptic `y`, and its `2e-11 AU`
bound directly covers the measured legacy-compatible platform envelope rather
than masking a candidate-selection or orchestration difference. No production
algorithm, root solver, or trigonometric implementation was changed to force
one platform's last bits onto another.
