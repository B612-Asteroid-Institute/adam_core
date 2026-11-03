# Core 0.7 canonical OD Rust/Python API crosswalk

Baseline: `72870d84` (successor to `483be7e2`). This is a design/validation artifact, not narrative user documentation.

## Required public contract

| Workflow | Rust input | Rust config | Rust provider | Python input | Python config | Return |
|---|---|---|---|---|---|---|
| `run_iod` | `&AssociatedObservations` | bound, self-contained `&IodRunConfig` | `&OdPropagator` | `SourceCatalog`, `Associations` | bound, self-contained `IodRunConfig` | fitted orbits, memberships, typed failures |
| `run_od` | `&AssociatedObservations` | bound, self-contained `&OdConfig` | `&OdPropagator` | `SourceCatalog`, `Associations` | bound, self-contained `OdConfig` | fitted orbits, memberships, typed failures |
| `refine` | `&MemberedOrbits` | self-contained `&RefinementRunConfig` | `&OdPropagator` | `FittedOrbits`, `FittedOrbitMembers`, `SourceCatalog` | self-contained `RefinementRunConfig` | fitted orbits, memberships, typed failures |
| `evaluate_orbits` | `&MemberedOrbits` | self-contained `&EvaluationConfig` | `&OdPropagator` | `FittedOrbits`, `FittedOrbitMembers`, `SourceCatalog` | self-contained `EvaluationConfig` | fitted orbits, memberships, typed failures |

Python's separate typed table arguments and tuple return are intentional binding/marshalling differences. Provider acquisition from a Python object/capsule is also intentionally binding-private. Scientific configuration, validation, preparation, provider metadata, execution, failures, fingerprints, provenance, identifiers, ordering, and route composition remain Rust-owned.

## Baseline mismatch inventory

| Workflow | Rust at baseline | Python at baseline | Mismatch |
|---|---|---|---|
| `run_iod` | takes full `&OdConfig` plus generic public `OdPropagatorSource` | accepts `IodRunConfig`, but passes hidden `_workflow_native: OdConfig` | stage type is not bound/self-contained; public signature disagrees; hidden coupling |
| `run_od` | takes `&OdConfig` plus generic public `OdPropagatorSource` | takes `OdConfig` | provider-source abstraction leaks into primary Rust API |
| `refine` | takes `&RefinementRunConfig` plus generic public `OdPropagatorSource` | takes `RefinementRunConfig` | provider-source abstraction leaks into primary Rust API |
| `evaluate_orbits` | takes full `&OdConfig` plus generic public `OdPropagatorSource` | accepts `EvaluationConfig`, but passes hidden `_workflow_native: OdConfig` | stage type omits prediction settings; public signature disagrees; hidden coupling |
| all four | `OdBatchResult` publicly owns `groups`/`providers`; orbit/member/failure products are iterators over group internals | returns the documented three typed products directly | Rust result exposes planning/provider details as primary instead of the three products |

The baseline projection methods are copies of declarations, not bound stage values: `OdConfig::for_iod()` returns only search/correction/ranging plus candidate-fit values; `OdConfig::for_evaluation()` returns only covariance dimension. Observation pipeline, profile/resource binding, and evaluation prediction settings remain in the full `OdConfig` and reach Python only through `_workflow_native`.

## Coherent split selected for implementation

### Configuration types

1. Rename the current declaration-only `IodRunConfig` to `IodStageConfig` in Rust and Python.
2. Define `IodRunConfig` as an owned bound value containing exactly:
   - the bound `ObservationPipeline` (therefore exact resource identities and data),
   - the `IodStageConfig`, and
   - optional immutable profile-selection identity.
3. Keep `RefinementRunConfig`; it already owns every numerical value used by refinement.
4. Make `EvaluationConfig` own both:
   - `covariance_dimension`, and
   - the exact `WhitenedFitConfig` used for provider prediction/residual evaluation.
5. Make `OdConfig` own the bound observation pipeline, `IodStageConfig`, `RefinementRunConfig`, and optional profile identity. Evaluation is not part of `run_od` and is configured independently.
6. `OdConfig::for_iod()` returns the complete bound `IodRunConfig`; `for_refinement()` remains a complete copy. `for_evaluation()` returns a new self-contained value derived from the explicitly selected refinement prediction settings; it does not retain or embed `OdConfig`. Named profiles also expose their explicit evaluation config separately.
7. Python values mirror these public fields. Remove `_workflow_native` entirely. Native PyO3 wrappers are implementation handles only and cannot carry undeclared scientific state. A bound named profile is declared as immutable `profile_identity` value state (`name`, `version`, and definition SHA-256); explicit constructors always produce `None` and cannot accept a caller-forged identity.

### Composition and provenance

A minimal IOD stage value cannot carry unrelated refinement/evaluation declarations merely to reproduce the old full-`OdConfig` provenance. Doing so would only rename the hidden coupling. Therefore:

- standalone/fused IOD uses a versioned route-specific canonical IOD configuration containing profile identity, transforms/resources, and IOD settings;
- `run_od` uses the complete bound OD configuration containing profile identity, transforms/resources, IOD settings, and refinement settings;
- refinement of a `run_iod` seed composes the seed's canonical IOD configuration with the explicit `RefinementRunConfig`, producing exactly the same canonical OD configuration as direct `run_od`; a named profile identity is retained only when the requested refinement exactly matches that verified profile definition, otherwise composition deliberately drops the profile and equals an explicit unprofiled `OdConfig`;
- standalone refinement uses only its explicit refinement configuration;
- evaluation uses only its self-contained evaluation configuration;
- route configuration fingerprints/domains are versioned and tested; no compiled build identity is reintroduced;
- default `orbit_id` generation uses the IOD-stage configuration fingerprint in both `run_iod` and `run_od`, preserving exact composition while correctly excluding later-stage settings from IOD identity.

The fixed six-dimensional evaluation invariant remains mandatory, but it is no longer represented as a configurable `run_od` input. This is an intentional schema correction, not a numerical change.

### Provider boundary

Primary Rust functions take `&OdPropagator` directly. Static validation/preparation must still precede Python capsule acquisition. To preserve both requirements without duplicating preparation:

- keep a private internal source/callback helper in `adam_core`;
- expose a `#[doc(hidden)]`, Python-feature-gated bridge only when building `adam_core_py`;
- implement deferred Python object/capsule acquisition solely in `adam_core_py` through that bridge;
- do not export a generic provider-source trait from the normal Rust API or use it in any primary signature.

Ordinary Rust callers construct `OdPropagator::two_body()` or `OdPropagator::external(...)` and pass a reference.

### Result ownership

`OdBatchResult` privately owns deterministic vectors for fitted orbits, fitted memberships, and typed failures. Public accessors return immutable slices, while `into_products()` moves all three vectors out together. Raw group handles and the separate `groups()` / `group_*` association accessors are absent from the ordinary public API, so a handle from one batch cannot panic, go stale, or select records from another batch. `to_membered_orbits()` and `to_membered_groups()` are the safe lossless handoffs, and mutable relational labels are edited only on the resulting `MemberedOrbits` values.

The Python-feature bridge alone exposes an opaque borrowed group view for Quivr assembly. Each view borrows its originating `OdBatchResult` together with that group's orbit, membership, failure, preparation, and diagnostics; no API accepts a standalone view alongside a separate batch. Ordinary Rust callers use the three primary slices without traversing group planning state.

## Intentional language differences

1. Python accepts/returns typed Quivr tables; Rust accepts/returns Core Rust batches/records.
2. Python propagator arguments may be `None` (the explicit built-in two-body backend) or capsule-providing objects; Rust uses explicit `OdPropagator` constructors.
3. Python obtains an external provider lazily after Rust static preparation; ordinary Rust already has a provider reference and does not need acquisition.
4. Python returns the established tuple `(FittedOrbits, FittedOrbitMembers, OrbitDeterminationFailures)`; Rust returns `OdBatchResult` with immutable views and an owning `into_products()` path.
5. Python resource paths are bound while constructing configs; canonical Rust configuration/provenance stores verified identities and path-free values. Native-to-Python stage projections reuse validated native state and never reopen those paths.
6. Rust names the IOD declaration accessor `stage()` / `iod_stage()` while Python exposes `.iod`; Python additionally forwards `.search_correction_ranging` and `.candidate_correction_fit` on `IodRunConfig`.
7. Rust provides `IodRunConfig::explicit`; Python provides the positional `IodRunConfig(...)` constructor. Python supports both positional `OdConfig(...)` and keyword-only `OdConfig.explicit(...)` over the same fields.
8. Predictor-level focused seams intentionally changed with the stage split: `prepare_observations` takes `&ObservationPipeline`, `run_iod_with_predictor` takes `&IodStageConfig`, and `refine_with_predictor` takes `&RefinementRunConfig`.
9. Rust `EvaluationConfig` contains floating-point fit settings and therefore is copyable but not `Eq`; Python dataclass value equality compares the same declared fields.
10. `OdProfile.evaluation` currently decodes the profile's Core-v1 evaluation declaration through `EvaluationConfig.core_v1()`; both shipped definitions are pinned to that exact value.

Outside these enumerated language and route-schema/default-label changes, defaults, ordering, failure classification, provider identity, scientific fingerprints, and numerical outcomes are required to remain equal.

## Required drift guards and examples

1. Rust compile-time signature assertions for all four primary functions using concrete function-pointer types.
2. Source guards forbidding `_workflow_native`, public `OdPropagatorSource`, and full-`OdConfig` parameters on `run_iod`/`evaluate_orbits`.
3. Rust/Python canonical configuration JSON equality tests for each stage projection.
4. Exact direct-`run_od` versus projected-`run_iod` + `refine` tests covering success, typed scientific failure, empty input, provider identity, profile/resource binding, and provider-call count.
5. Standalone Rust examples/tests for all four functions with the built-in provider and at least one external-provider path.
6. Python tests for direct construction of each self-contained stage config; no projection-only hidden acceptance path.
7. `OdBatchResult` tests that read immutable product slices, consume `into_products()`, preserve safe membered conversion, and compile-reject direct field mutation and raw group-handle access.
8. Package/profile/resource and public-surface audits updated for the deliberate breaking API.
