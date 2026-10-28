# OD API migration for Core 0.7

## Foundation status

This change does **not** publish the final batch `run_iod`, `run_od`, `refine`,
or `evaluate_orbits` functions. Those names are intentionally absent until the
`personal-eack9` Associations/FittedOrbitMembers grouping and typed-failure
contract is implemented. The future batch return is expected to be
`FittedOrbits`, `FittedOrbitMembers`, and `OrbitDeterminationFailures`; the
focused one-group IOD report is not a third top-level batch return.

## Configuration migration

Replace `models=`, `models="default"`, fitter objects, callback models, and
free-form settings dictionaries with one explicit `OdConfig`. Use a bound
native profile or construct complete typed `IodRunConfig`,
`RefinementRunConfig`, and `EvaluationConfig` declarations plus an ordered
`ObservationPipeline`. Rust is the sole parser, validator, and canonical
provenance source.

The old `ObservationUncertaintyModel`, `CompositeModel`,
`apply_observation_models`, `initial_orbit_determination`, `iod`,
`differential_correction`, `NativeOrbitFitter`, `OrbitFitter`, and Python model
provenance modules have no aliases. Native `run_od_ipc(models=...)` and the open
Rust transform trait-object route are also removed.

Transform names are:

- `FillMissingAstrometricUncertainty`
- `Efcc18CatalogBiasCorrection(exclude_astcats=...)` (the exclusion list is
  required explicitly)
- `StationBandResidualCovarianceAdjustment(operation=..., minimum_residual_count=...)`
- `NightlyBatchDeweighting(effective_observation_cap=...)`

Use `ObservationPipeline.as_supplied()` for deliberate no-transform behavior.
An empty ordered list is rejected. Transform order and repetition are retained.

Profiles bind explicit local paths by semantic slot. No package discovery,
runtime acquisition, fallback profile, unused resource, or absolute-path
provenance is permitted. `CORE_OD_V1` currently does not fill missing
uncertainties; it fails closed on such inputs pending public-profile
qualification.

The foundation deliberately exports no interim Python `_one_group` workflows:
configured composition and provenance remain Rust-owned until the canonical
batch functions land. Expert numerical kernels remain under
`adam_core.orbit_determination.algorithms`; standalone Rust predictor-level
configured composition lives under `adam_core_rs_coords::od::algorithms`. The
disposition of every Python OD test function removed since the accepted
numerical baseline is recorded in `od_test_migration_0_7.md`.
