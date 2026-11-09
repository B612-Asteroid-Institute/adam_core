"""Static guards for the four canonical OD workflow docstrings."""

from __future__ import annotations

import inspect
import re
import textwrap
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from ...observations import Associations, SourceCatalog
from ..config import (
    EvaluationConfig,
    IodRunConfig,
    IodStageConfig,
    OdConfig,
    OdResourceValidationError,
    RefinementRunConfig,
    ResourceBindings,
)
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import (
    FittedOrbitMembers,
    FittedOrbits,
    OrbitDeterminationFailures,
)
from ..observation_pipeline import ObservationPipeline
from ..profiles import OdProfile
from ..workflows import evaluate_orbits, refine, run_iod, run_od
from .test_batch_workflows import (
    _assert_exact_results,
    _concat_failures,
    _two_body_source,
)

_ROOT = Path(__file__).resolve().parents[4]


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


def test_non_gravitational_provider_boundary_matches_native_capability() -> None:
    reference = (_ROOT / "docs" / "source" / "non_grav_support.rst").read_text()
    normalized = " ".join(reference.split())
    assert "``Propagator::supports_non_gravitational_parameters``" in normalized
    assert "``EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL``" in normalized
    assert (
        "ABI-v2 boundary and Core's built-in two-body provider fail closed"
        in normalized
    )
    assert "supports_non_gravitational_forces" not in normalized
    assert "log a warning" not in normalized


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


@dataclass(frozen=True)
class CodeBlock:
    """One parsed reStructuredText code-block directive and body."""

    language: str
    body: str


_GUIDE_BLOCK_CONTRACT = (
    ("python", "one_group = Associations.from_kwargs"),
    ("python", "config = CORE_OD_V1.bind"),
    ("python", "overlapping = Associations.from_kwargs"),
    ("python", "iod_orbits, iod_members, iod_failures = run_iod"),
    ("python", "selected_id = fitted_orbits.orbit_id[0].as_py()"),
    ("python", 'new_id = "analyst-label"'),
    ("python", "institute_config = ASTEROID_INSTITUTE_OD_V1.bind(resources)"),
    ("python", "explicit_config = OdConfig.explicit"),
    ("rust", "fn workflows("),
    ("console", "cargo run -p adam_core --example od_canonical_workflows"),
    ("console", "cargo run -p adam_core --example od_external_provider"),
    ("rust", "let resources = ResourceBindings::new().bind("),
)
_WORKFLOW_BLOCK_CONTRACT = {
    "run_iod": (("python", "one_group = Associations.from_kwargs"),),
    "run_od": (("python", "config = CORE_OD_V1.bind"),),
    "refine": (("python", "iod_orbits, iod_members, iod_failures = run_iod"),),
    "evaluate_orbits": (("python", "evaluated_orbits, evaluated_members, failures"),),
}


def _code_blocks(text: str) -> list[CodeBlock]:
    """Parse every code-block directive, rejecting malformed directives."""

    lines = text.splitlines()
    blocks: list[CodeBlock] = []
    index = 0
    while index < len(lines):
        directive = lines[index].strip()
        if not directive.startswith(".. code-block"):
            index += 1
            continue
        match = re.fullmatch(r"\.\. code-block:: ([A-Za-z0-9_+.-]+)", directive)
        if match is None:
            raise AssertionError(f"malformed code-block directive: {directive!r}")
        language = match.group(1)
        index += 1
        body: list[str] = []
        while index < len(lines) and (
            not lines[index].strip() or lines[index].startswith("   ")
        ):
            body.append(lines[index])
            index += 1
        blocks.append(CodeBlock(language, textwrap.dedent("\n".join(body)).strip("\n")))
    return blocks


def _assert_block_contract(
    blocks: list[CodeBlock], contract: tuple[tuple[str, str], ...], label: str
) -> None:
    """Require exact directive count, language order, and stable block identities."""

    assert len(blocks) == len(
        contract
    ), f"{label} code-block count changed: expected {len(contract)}, got {len(blocks)}"
    for index, (block, (language, identity)) in enumerate(
        zip(blocks, contract, strict=True)
    ):
        assert block.language == language, (
            f"{label} block {index} language changed: "
            f"expected {language!r}, got {block.language!r}"
        )
        assert (
            identity in block.body
        ), f"{label} block {index} lost expected identity {identity!r}"


def _assert_products(namespace: dict[str, Any], names: tuple[str, str, str]) -> None:
    orbits, members, failures = (namespace[name] for name in names)
    assert isinstance(orbits, FittedOrbits)
    assert isinstance(members, FittedOrbitMembers)
    assert isinstance(failures, OrbitDeterminationFailures)


def _execute(code: str, namespace: dict[str, Any], label: str) -> None:
    exec(compile(code, label, "exec"), namespace)


def _execute_documented_examples() -> None:
    source_catalog = _source_catalog()
    ids = source_catalog.id.to_pylist()
    overlap_count = max(3, len(ids) // 2)
    associations = Associations.from_kwargs(
        detection_id=ids + ids[:overlap_count],
        object_id=["full-arc"] * len(ids) + ["half-arc"] * overlap_count,
    )
    namespace: dict[str, Any] = {
        "source_catalog": source_catalog,
        "associations": associations,
    }

    guide = _ROOT / "docs" / "source" / "use_cases" / "orbit_determination.rst"
    guide_blocks = _code_blocks(guide.read_text())
    _assert_block_contract(guide_blocks, _GUIDE_BLOCK_CONTRACT, str(guide))
    python_blocks = guide_blocks[:8]
    assert all(block.language == "python" for block in python_blocks)
    for index, block in enumerate(python_blocks):
        label = f"{guide.name}:python[{index}]"
        if index == 2:
            local = dict(namespace)
            local["source_catalog"] = source_catalog.set_column(
                "id",
                pa.array(
                    [f"obs-{row + 1}" for row in range(len(source_catalog))],
                    type=source_catalog.id.type,
                ),
            )
            _execute(block.body, local, label)
            _assert_products(local, ("orbits", "members", "failures"))
        elif index in (6, 7):
            with pytest.raises(OdResourceValidationError):
                _execute(block.body, dict(namespace), label)
        else:
            _execute(block.body, namespace, label)
            expected_products = {
                0: [("fitted_orbits", "fitted_members", "failures")],
                1: [("fitted_orbits", "fitted_members", "failures")],
                3: [
                    ("iod_orbits", "iod_members", "iod_failures"),
                    ("refined_orbits", "refined_members", "refine_failures"),
                ],
                4: [
                    (
                        "evaluated_orbits",
                        "evaluated_members",
                        "evaluation_failures",
                    )
                ],
            }.get(index, [])
            for names in expected_products:
                _assert_products(namespace, names)

    relabeled = namespace["fitted_orbits"]
    assert "analyst-label" in relabeled.orbit_id.to_pylist()
    evaluated = evaluate_orbits(
        relabeled,
        namespace["fitted_members"],
        source_catalog,
        namespace["propagator"],
        namespace["config"].for_evaluation(),
    )
    refined = refine(
        relabeled,
        namespace["fitted_members"],
        source_catalog,
        namespace["propagator"],
        namespace["config"].for_refinement(),
    )
    assert "analyst-label" in evaluated[0].orbit_id.to_pylist()
    assert "analyst-label" in refined[0].orbit_id.to_pylist()

    direct = run_od(source_catalog, associations, None, namespace["config"])
    iod = run_iod(source_catalog, associations, None, namespace["config"].for_iod())
    staged = refine(
        iod[0],
        iod[1],
        source_catalog,
        None,
        namespace["config"].for_refinement(),
    )
    _assert_exact_results(
        direct,
        (staged[0], staged[1], _concat_failures(iod[2], staged[2])),
    )

    for workflow in [run_iod, run_od, refine, evaluate_orbits]:
        name = workflow.__name__
        blocks = _code_blocks(inspect.getdoc(workflow) or "")
        _assert_block_contract(
            blocks, _WORKFLOW_BLOCK_CONTRACT[name], f"{name} docstring"
        )
        local = {
            "source_catalog": source_catalog,
            "associations": associations,
            "config": namespace["config"],
            "fitted_orbits": namespace["iod_orbits"],
            "fitted_members": namespace["iod_members"],
            "run_iod": run_iod,
            "refine": refine,
            "evaluate_orbits": evaluate_orbits,
        }
        _execute(blocks[0].body, local, f"{name}.__doc__[0]")


def test_checked_in_narrative_examples_execute() -> None:
    _execute_documented_examples()


def test_narrative_parser_and_contract_fail_closed_on_mutation() -> None:
    guide = (
        _ROOT / "docs" / "source" / "use_cases" / "orbit_determination.rst"
    ).read_text()

    with pytest.raises(AssertionError, match="language changed"):
        _assert_block_contract(
            _code_blocks(
                guide.replace(".. code-block:: python", ".. code-block:: python3", 1)
            ),
            _GUIDE_BLOCK_CONTRACT,
            "mutated guide",
        )
    with pytest.raises(AssertionError, match="count changed"):
        _assert_block_contract(
            _code_blocks(guide.replace(".. code-block:: python", ".. note::", 1)),
            _GUIDE_BLOCK_CONTRACT,
            "mutated guide",
        )
    with pytest.raises(AssertionError, match="lost expected identity"):
        _assert_block_contract(
            _code_blocks(guide.replace("one_group =", "one_group_broken =", 1)),
            _GUIDE_BLOCK_CONTRACT,
            "mutated guide",
        )
    with pytest.raises(AssertionError, match="malformed code-block directive"):
        _code_blocks(".. code-block: python\n\n   pass")


def test_documented_workflow_type_and_value_boundaries_are_exercised() -> None:
    with pytest.raises(TypeError, match="IodRunConfig"):
        run_iod(object(), object(), None, object())  # type: ignore[arg-type]

    source = _source_catalog()
    dangling = Associations.from_kwargs(
        detection_id=["not-a-source-id"], object_id=["group"]
    )
    with pytest.raises(ValueError, match="not-a-source-id"):
        run_iod(source, dangling, None, _iod_config())
