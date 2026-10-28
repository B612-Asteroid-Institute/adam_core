"""Rust-owned configuration identity and result-schema provenance contracts."""

import json

from ..config import (
    EvaluationConfig,
    IodRunConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
)
from ..fitted_orbits import OdProvenance
from ..observation_pipeline import ObservationPipeline


def config() -> OdConfig:
    return OdConfig.explicit(
        observation_pipeline=ObservationPipeline.as_supplied(),
        iod=IodRunConfig.core_v1(),
        refinement=RefinementRunConfig.core_v1(),
        evaluation=EvaluationConfig.six_dimensional(),
        resources=ResourceBindings({}),
    )


def test_complete_configuration_provenance_is_canonical_and_path_free() -> None:
    value = config().provenance
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    assert value["transforms"] == []
    assert "configuration_sha256" in value
    assert "/Users/" not in encoded and "/tmp/" not in encoded


def test_runtime_provenance_schema_contains_complete_identity_fields() -> None:
    names = set(OdProvenance.schema.names)
    assert {
        "adam_core_version",
        "adam_core_build",
        "input_fingerprint",
        "route",
        "iod_provider",
        "iod_diagnostics",
        "refinement_provider",
        "refinement_diagnostics",
        "configuration",
        "num_rejected",
        "rejected_fraction",
    } <= names
