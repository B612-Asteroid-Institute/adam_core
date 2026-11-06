"""Rust-owned configuration identity and result-schema provenance contracts."""

import json

from ..config import (
    IodStageConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
)
from ..fitted_orbits import OdProvenance, OrbitDeterminationFailures
from ..observation_pipeline import ObservationPipeline


def config() -> OdConfig:
    return OdConfig.explicit(
        observation_pipeline=ObservationPipeline.as_supplied(),
        iod=IodStageConfig.core_v1(),
        refinement=RefinementRunConfig.core_v1(),
        resources=ResourceBindings({}),
    )


def test_complete_configuration_provenance_is_canonical_and_path_free() -> None:
    value = config().provenance
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    assert value["transforms"] == []
    assert "configuration_sha256" in value
    assert "/Users/" not in encoded and "/tmp/" not in encoded


def test_runtime_provenance_schema_contains_scientific_identity_fields_only() -> None:
    names = set(OdProvenance.schema.names)
    assert "adam_core_version" not in names
    assert "adam_core_build" not in names
    assert {
        "input_fingerprint",
        "state_fingerprint",
        "provenance_fingerprint",
        "route",
        "iod_provider",
        "iod_diagnostics",
        "refinement_provider",
        "refinement_diagnostics",
        "evaluation_provider",
        "evaluation_diagnostics",
        "configuration",
        "num_rejected",
        "rejected_fraction",
    } <= names


def test_batch_scientific_failure_schema_is_typed_and_not_opaque() -> None:
    assert OrbitDeterminationFailures.schema.names == [
        "group_id",
        "orbit_id",
        "input_orbit_id",
        "stage",
        "code",
        "message",
        "recoverable",
        "num_obs",
        "arc_length",
        "triplets_tried",
        "candidates_evaluated",
        "candidates_failed",
        "corrections_run",
        "passes",
        "n_rejected",
        "n_recovered",
        "status_code",
        "best_state_retained",
    ]
