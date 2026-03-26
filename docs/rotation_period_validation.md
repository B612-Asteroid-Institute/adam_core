# Rotation Period Analysis and Validation

## What Was Added

`adam_core.photometry` now includes a rotation-period solver based on a high-order
Fourier frequency search, along with detection wrappers and optional real-data
regression fixtures.

The main public entry points are:

- `estimate_rotation_period(...)`
- `build_rotation_period_observations_from_detections(...)`
- `estimate_rotation_period_from_detections(...)`
- `estimate_rotation_period_from_detections_grouped(...)`

The observation model now supports an optional `session_id` field. When present,
the solver fits within-filter session offsets in addition to the existing filter
offsets, phase-function terms, and Fourier harmonics. This is important for
real lightcurve archives where different nights or submitted lightcurve blocks
often have different zero points.

## Validation Strategy

The project now uses two layers of validation:

1. Synthetic unit tests

- period recovery for a known signal
- multi-filter offset handling
- session-offset handling
- single-peaked period doubling
- outlier rejection
- minimum-sufficient Fourier-order selection
- detection-wrapper and grouped-wrapper behavior

2. Optional real-data regression tests

- legacy MPC-backed path retained as a heterogeneous stress test
- new PDS-backed path added as the primary scientific regression path
- LCDB supplies the expected published period
- ALCDEF supplies the raw time-series lightcurve sessions

The PDS regression suite is opt-in:

```bash
pdm run pytest --benchmark-skip src/adam_core/photometry/tests/test_rotation_period_pds.py -q --run-rotation-period-pds
```

## Current PDS Gold Fixtures

The initial frozen fixtures are:

- `289 Nenetta`
- `1011 Laodamia`

These fixtures include:

- times
- magnitudes
- magnitude errors when present
- filters
- `session_id`
- geometry arrays (`r_au`, `delta_au`, `phase_angle_deg`)
- LCDB expected period and quality metadata
- per-session metadata copied from ALCDEF for auditability

## Why This Is Not Just Overfitting To Two Objects

The current evidence is stronger than a two-example smoke test for three reasons.

First, the core solver behavior is constrained by synthetic tests that are
independent of the PDS fixtures. Those tests verify the algorithmic properties
we actually rely on: harmonics, nuisance offsets, clipping, order selection,
and the single-peak/double-peak ambiguity. The PDS cases are not the only tests
exercising the implementation.

Second, the PDS fixtures are not hand-shaped to the output of the solver. They
come from LCDB and ALCDEF, two external archives with different purposes:
published periods on one side and raw time-series sessions on the other. The
solver must recover the LCDB period from the ALCDEF photometry; the expected
answer is not computed by the code under test.

Third, the session-aware extension is justified by the data model, not by the
specific identities of `289` and `1011`. ALCDEF metadata explicitly preserves
lightcurve-block structure, and raw multi-session asteroid photometry commonly
needs per-session zero-point freedom. The feature is therefore a model change
driven by archive structure, not a special-case tweak for one asteroid.

## What We Know So Far

With the current implementation and frozen PDS fixtures, the solver recovers:

- `1011 Laodamia`: `5.173269 h` vs `5.17247 h`
- `289 Nenetta`: `6.91669 h` vs `6.902 h`

That is materially tighter than the earlier MPC-backed validation path, which
was dominated by heterogeneous photometric systematics.

## Current Limits

- The gold PDS suite currently covers two objects, not the full challenge set.
- The optional PDS suite is slower than the synthetic suite; the two-fixture run
  is on the order of two minutes.
- The challenge objects identified in `PDS_CANDIDATES.md` are not yet frozen
  into fixtures, so the current real-data suite is intentionally conservative.
