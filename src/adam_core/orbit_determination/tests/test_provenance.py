"""Fail-closed configured-workflow provenance contracts."""

import json

import pytest

from ..algorithms import _provenance
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


def test_dirty_or_unknown_native_identity_is_refused(monkeypatch) -> None:
    for build in ("dirty:" + "a" * 40, "unknown", ""):
        monkeypatch.setattr(
            _provenance._rust_native,
            "build_identity",
            lambda build=build: ("0.7.0", build),
        )
        with pytest.raises(RuntimeError, match="exact clean"):
            _provenance.core_build_identity()


def test_clean_native_identity_is_accepted(monkeypatch) -> None:
    commit = "a" * 40
    monkeypatch.setattr(
        _provenance._rust_native, "build_identity", lambda: ("0.7.0", commit)
    )
    assert _provenance.core_build_identity() == ("0.7.0", commit)


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
