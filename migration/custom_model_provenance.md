# Migrating custom observation models for Core 0.7

Core 0.7 does not execute third-party Python observation-model behavior.
`ObservationUncertaintyModel`, `ModelProvenance`, `ProvenancedModel`, and the
native-spec callback path have been removed rather than retained as
compatibility interfaces. Package metadata, source paths, bytecode, and
provider-selected fingerprints cannot attest mutable Python behavior.

Use an explicit `ObservationPipeline` made only from Core's declarative
transform variants. Core Rust validates the full declaration, binds any
required resource by semantic slot, executes transforms in declaration order,
and records canonical path-free provenance. Repeated transforms are preserved;
`ObservationPipeline.as_supplied()` is the only deliberate no-transform policy.

A deployment-specific data product may be supplied through a profile's
`bind(...)` operation when its bytes and schema satisfy that profile's pinned
`ResourceRequirement`. It cannot add executable transform semantics or select
a propagator. Unsupported transform behavior requires a new reviewed Core Rust
variant and, when it changes a predefined profile, a new profile version and
fingerprint.

The former installable live-Python example has been removed because it imported
APIs that do not exist in the 0.7 surface. There is no external observation
model plugin ABI or Python runtime-attestation fallback.
