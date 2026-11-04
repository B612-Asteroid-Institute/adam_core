"""Static guards for the four canonical OD workflow docstrings."""

from __future__ import annotations

import inspect
import runpy
from importlib.resources import files
from pathlib import Path

import pytest

from ...observations import Associations, SourceCatalog
from .. import workflows
from ..config import (
    EvaluationConfig,
    IodRunConfig,
    IodStageConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
)
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..observation_pipeline import ObservationPipeline
from ..profiles import OdProfile
from ..workflows import evaluate_orbits, refine, run_iod, run_od
from .test_batch_workflows import _two_body_source


@pytest.mark.parametrize("workflow", [run_iod, run_od, refine, evaluate_orbits])
def test_canonical_workflow_docstrings_keep_complete_contract(workflow: object) -> None:
    doc = inspect.getdoc(workflow)
    assert doc is not None
    for section in ["Parameters", "Returns", "Raises", "Notes", "Examples"]:
        assert f"\n{section}\n" in doc
    for concept in [
        "FittedOrbits",
        "FittedOrbitMembers",
        "OrbitDeterminationFailures",
        "provenance",
        "provider",
        "empty",
    ]:
        assert concept.lower() in doc.lower()


def test_canonical_workflow_docstrings_keep_route_specific_guidance() -> None:
    expectations = {
        run_iod: ["IodRunConfig", "Associations", "first-appearance", "no-solution"],
        run_od: ["OdConfig", "run_iod", "refine", "only once"],
        refine: [
            "RefinementRunConfig",
            "original_astrometry",
            "used_astrometry",
            "mutable relational data",
        ],
        evaluate_orbits: [
            "EvaluationConfig",
            "subset",
            "without changing their states",
            "does not refit",
        ],
    }
    for workflow, phrases in expectations.items():
        doc = inspect.getdoc(workflow)
        assert doc is not None
        for phrase in phrases:
            assert phrase in doc


def _source_catalog() -> SourceCatalog:
    data = files("adam_core.orbit_determination.tests.data")
    return _two_body_source(
        (
            FittedOrbits.from_parquet(str(data.joinpath("pure_iod_orbit.parquet"))),
            FittedOrbitMembers.from_parquet(
                str(data.joinpath("pure_iod_orbit_members.parquet"))
            ),
            OrbitDeterminationObservations.from_parquet(
                str(data.joinpath("pure_iod_orbit_observations.parquet"))
            ),
        )
    )


def _iod_config() -> IodRunConfig:
    return IodRunConfig(
        ObservationPipeline.as_supplied(),
        IodStageConfig.core_v1(),
        ResourceBindings({}),
    )


def test_workflow_raises_sections_match_binding_boundaries() -> None:
    for workflow in [run_iod, run_od, refine, evaluate_orbits]:
        doc = inspect.getdoc(workflow)
        assert doc is not None
        raises = doc.split("\nRaises\n", 1)[1].split("\nNotes\n", 1)[0]
        assert "TypeError" in raises
        assert "ValueError" in raises
        assert "RuntimeError" in raises
        assert "OdConfigurationError" not in raises
        assert "OdResourceValidationError" not in raises

    for config_type in [
        IodStageConfig,
        IodRunConfig,
        OdConfig,
        RefinementRunConfig,
        EvaluationConfig,
    ]:
        assert "OdConfigurationError" in (inspect.getdoc(config_type) or "")
    assert "OdResourceValidationError" in (inspect.getdoc(IodRunConfig) or "")
    assert "OdResourceValidationError" in (inspect.getdoc(OdConfig) or "")
    assert "OdResourceValidationError" in (inspect.getdoc(OdProfile.bind) or "")

    support = (
        Path(__file__).resolve().parents[4]
        / "rust"
        / "adam_core_py"
        / "src"
        / "od_support.rs"
    ).read_text()
    runtime_arm = support.split("PropagationError::CmcCovarianceUnavailable", 1)[
        1
    ].split("=> PyRuntimeError", 1)[0]
    for category in ["RowFailure", "Backend", "BackendProtocol", "ThreadPool"]:
        assert f"PropagationError::{category}" in runtime_arm


def test_checked_in_narrative_examples_execute() -> None:
    runner = (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "scripts"
        / "run_orbit_determination_examples.py"
    )
    namespace = runpy.run_path(str(runner))
    namespace["main"]()


def test_documented_workflow_exception_boundaries_are_exercised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="IodRunConfig"):
        run_iod(object(), object(), None, object())  # type: ignore[arg-type]

    source = _source_catalog()
    dangling = Associations.from_kwargs(
        detection_id=["not-a-source-id"], object_id=["group"]
    )
    with pytest.raises(ValueError, match="not-a-source-id"):
        run_iod(source, dangling, None, _iod_config())

    associations = Associations.from_kwargs(
        detection_id=source.id, object_id=["group"] * len(source)
    )

    def backend_failure(*_args: object) -> object:
        raise RuntimeError("documented backend/protocol failure")

    native = getattr(workflows, "_rust_native")
    monkeypatch.setattr(native, "run_iod_batch_ipc", backend_failure)
    with pytest.raises(RuntimeError, match="backend/protocol failure"):
        run_iod(source, associations, None, _iod_config())
