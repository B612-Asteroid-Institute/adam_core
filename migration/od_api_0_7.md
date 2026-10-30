# OD API migration for Core 0.7

## Canonical batch API

Core 0.7 exports four explicit-config batch workflows from
`adam_core.orbit_determination`:

- `run_iod(observations, associations, propagator, config)`;
- `run_od(observations, associations, propagator, config)`;
- `refine(orbits, members, observations, propagator, config)`;
- `evaluate_orbits(orbits, members, observations, propagator, config)`.

Every function returns exactly `FittedOrbits`, `FittedOrbitMembers`, and
`OrbitDeterminationFailures`. Raw IOD/OD consumes `SourceCatalog` plus explicit
`Associations`; refinement and evaluation consume explicit
`FittedOrbitMembers`. A one-object call is one group through the same Rust
batch path. There are no primary `_one_group` or linkage aliases.

Rust validates complete schemas and memberships before adopting a propagation
provider or executing an observation transform. Null group IDs, unknown
references, duplicate membership pairs, non-unique observation/orbit IDs, and
orbits without explicit members abort the invocation atomically. Group order is
first appearance and rows within each group are restored to source-table order;
overlapping hypotheses are supported.

Safely isolated scientific failures continue per group and are typed columns in
`OrbitDeterminationFailures`. Contract, resource, provider, provenance, and
invariant failures are invocation-fatal and never become failure rows. IOD
failures produce no placeholder state. A non-converged refinement is retained
with `success=False` only when the native solver returned its actual best state;
a local error that produced no state emits only a failure row. Evaluation errors
likewise emit no unchanged seed placeholder. Failure `group_id` is the original
association hypothesis (`object_id`) when available. `orbit_id` identifies an
actually emitted retained/best state and is null when no state exists;
`input_orbit_id` separately identifies the refine/evaluate input orbit.

## Configuration migration

Replace `models=`, `models="default"`, fitter objects, callback models, and
free-form settings dictionaries with typed Rust-owned configuration. `run_od`
takes a complete bound `OdConfig`; `run_iod` takes its bound `IodRunConfig`
projection from `config.for_iod()`; `refine` takes `RefinementRunConfig`; and
`evaluate_orbits` takes the bound `EvaluationConfig` projection from
`config.for_evaluation()`. Profiles or manual
`OdConfig.explicit(...)` construction bind every resource path before an IOD/OD
workflow begins. Canonical IOD orbit identity deliberately includes the complete
`OdConfig`, so changing later-stage or evaluation policy changes the seed ID even
when IOD numerics are otherwise unchanged. Evaluation uses the two-body/predictor
settings of the complete configuration's refinement fit; the evaluation policy
itself supplies the covariance dimension.

The old `ObservationUncertaintyModel`, `CompositeModel`,
`apply_observation_models`, `initial_orbit_determination`, `iod`,
`differential_correction`, `NativeOrbitFitter`, `OrbitFitter`, and Python model
provenance modules have no aliases. Native `run_od_ipc(models=...)` and the open
Rust transform trait-object route are removed.

Transform names are:

- `FillMissingAstrometricUncertainty`;
- `Efcc18CatalogBiasCorrection`;
- `StationBandResidualCovarianceAdjustment`;
- `NightlyBatchDeweighting`.

Use `ObservationPipeline.as_supplied()` for deliberate no-transform behavior.
An empty ordered list is rejected. Transform order and repetition are retained.

Profiles bind explicit local paths by semantic slot. No package discovery,
runtime acquisition, fallback profile, unused resource, or absolute-path
provenance is permitted. `CORE_OD_V1` does not fill missing uncertainties and
fails closed on such input.

The pipeline executes once per group. `run_od` gives refinement the exact
prepared IOD rows without reapplying transforms. Fitted members persist original
and used astrometry, including the exact diagonal variances rather than relying
on a lossy sigma-square round trip; standalone `refine` consumes that prepared
handoff. A
successful `run_od` result is the same state, members, diagnostics, and canonical
provenance as `run_iod(..., config.for_iod())` followed by
`refine(..., config.for_refinement())`. Standalone refinement/evaluation rejects
legacy members with missing snapshots and rejects source astrometry changed since
the persisted original snapshot; rerun canonical IOD to produce a complete
handoff. Exact composition is intentionally strict: a `run_iod` seed may be
refined only with the refinement projection embedded in its complete
configuration and with its original complete membership. To try a different
refinement strategy, run a distinct complete configuration and therefore obtain
a distinct canonical identity. Expert numerical kernels remain under
`adam_core.orbit_determination.algorithms` and
standalone Rust predictor seams remain under
`adam_core_rs_coords::od::algorithms`.

Canonical provenance is emitted by Rust and includes clean Core build identity,
ordered complete source/member fingerprints, exact orbit-state fingerprint,
route, native provider metadata, stable canonical stage diagnostics, and
canonical path-free configuration. A separate provenance fingerprint binds the
route, validated configuration SHA-256, and input fingerprint, so relabelled
routes or mixed configuration/input claims fail before provider binding.
Unknown/dirty build identity, mutated source
or observer geometry, changed membership/state, and missing provider
fingerprints fail closed before provider binding.
