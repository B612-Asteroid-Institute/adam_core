"""Static guards for the four canonical OD workflow docstrings."""

from __future__ import annotations

import inspect
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from ...observations import Associations, SourceCatalog
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
        normalized_raises = " ".join(raises.split())
        for phrase in [
            "non-capsule",
            "adam_core.propagator.v2",
            "malformed descriptor",
            "NativeBackend",
            "complete provenance suffix",
            "capability bits are unknown to this Core",
            "ABI-v2 contract",
            "partial provenance suffix",
            "alignment",
            "reserved words",
            "null context or callback",
            "THREAD_SAFE",
            "empty identity strings",
            "malformed fingerprints",
            "STM capability/callback slot",
            "ERROR",
            "PANIC",
            "INVALID",
            "unknown status",
            "malformed response",
            "before callback execution",
            "validation layer",
        ]:
            assert phrase in normalized_raises

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


def test_native_abi_reference_scopes_unknown_capability_bits_to_od() -> None:
    reference = (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "reference"
        / "native_propagator_abi.rst"
    ).read_text()
    normalized = " ".join(reference.split())
    assert "General propagation ignores unknown capability bits" in normalized
    assert "record the provider's complete capability set in provenance" in normalized
    assert "reject any bit unknown to this Core with ``ValueError``" in normalized
    assert "Unknown bits advertise optional future" not in normalized


def test_guide_enumerates_the_validation_layer_exception_split() -> None:
    guide = (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "use_cases"
        / "orbit_determination.rst"
    ).read_text()
    normalized = " ".join(guide.split())
    for phrase in [
        "``ValueError`` covers a non-capsule",
        "capsule not named ``adam_core.propagator.v2``",
        "malformed descriptor",
        "``NativeBackend`` declaration/table identity mismatch",
        "without the complete provenance suffix",
        "capability bits unknown to this Core",
        "``RuntimeError`` covers an ABI-v2 table-contract violation",
        "base size or partial suffix",
        "null context or callback",
        "missing ``THREAD_SAFE``",
        "empty identity strings",
        "malformed fingerprints",
        "STM capability/callback slot",
        "callback ``ERROR``, ``PANIC``, ``INVALID``, or an unknown status",
        "Both classes can therefore arise before callback execution",
        "validation layer, not acquisition-versus-execution timing",
    ]:
        assert phrase in normalized


def test_rendered_od_rustdoc_has_complete_public_member_docs() -> None:
    root = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment["RUSTDOCFLAGS"] = "-D warnings"
    subprocess.run(
        [
            environment.get("CARGO", "cargo"),
            "doc",
            "--locked",
            "-p",
            "adam_core",
            "--no-deps",
            "--all-features",
        ],
        cwd=root / "rust",
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "docs" / "scripts" / "check_od_rustdoc_coverage.py"),
            str(root / "target" / "doc" / "adam_core" / "od"),
            "--expected-total",
            "378",
        ],
        check=True,
    )

    # The pinned count is a rendered-coverage sentinel, not an API-size
    # promise. Prove specifically that an enum-variant field regression is
    # detected rather than merely asserting that the unmodified tree passes.
    with tempfile.TemporaryDirectory() as temporary:
        mutated_root = Path(temporary) / "od"
        shutil.copytree(root / "target" / "doc" / "adam_core" / "od", mutated_root)
        page = mutated_root / "enum.ObservationTransform.html"
        rendered = page.read_text()
        field = "variant.FillMissingAstrometricUncertainty.field.resource"
        anchor = rendered.index(f'id="{field}"')
        docblock = rendered.index('<div class="docblock">', anchor)
        rendered = (
            rendered[:docblock]
            + '<div class="docblock-removed">'
            + rendered[docblock + len('<div class="docblock">') :]
        )
        page.write_text(rendered)
        control = subprocess.run(
            [
                sys.executable,
                str(root / "docs" / "scripts" / "check_od_rustdoc_coverage.py"),
                str(mutated_root),
                "--expected-total",
                "378",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert control.returncode != 0
        assert f"MISSING od/enum.ObservationTransform.html: {field}" in control.stdout


def _narrative_runner_namespace() -> dict[str, Any]:
    runner = (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "scripts"
        / "run_orbit_determination_examples.py"
    )
    return runpy.run_path(str(runner))


def test_checked_in_narrative_examples_execute() -> None:
    _narrative_runner_namespace()["main"]()


def test_narrative_parser_and_contract_fail_closed_on_mutation() -> None:
    namespace = _narrative_runner_namespace()
    code_blocks = namespace["code_blocks"]
    assert_contract = namespace["assert_block_contract"]
    contract = namespace["GUIDE_BLOCK_CONTRACT"]
    guide = (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "use_cases"
        / "orbit_determination.rst"
    ).read_text()

    with pytest.raises(AssertionError, match="language changed"):
        assert_contract(
            code_blocks(
                guide.replace(".. code-block:: python", ".. code-block:: python3", 1)
            ),
            contract,
            "mutated guide",
        )
    with pytest.raises(AssertionError, match="count changed"):
        assert_contract(
            code_blocks(guide.replace(".. code-block:: python", ".. note::", 1)),
            contract,
            "mutated guide",
        )
    with pytest.raises(AssertionError, match="lost expected identity"):
        assert_contract(
            code_blocks(guide.replace("one_group =", "one_group_broken =", 1)),
            contract,
            "mutated guide",
        )
    with pytest.raises(AssertionError, match="malformed code-block directive"):
        code_blocks(".. code-block: python\n\n   pass")


def test_documented_workflow_type_and_value_boundaries_are_exercised() -> None:
    with pytest.raises(TypeError, match="IodRunConfig"):
        run_iod(object(), object(), None, object())  # type: ignore[arg-type]

    source = _source_catalog()
    dangling = Associations.from_kwargs(
        detection_id=["not-a-source-id"], object_id=["group"]
    )
    with pytest.raises(ValueError, match="not-a-source-id"):
        run_iod(source, dangling, None, _iod_config())
