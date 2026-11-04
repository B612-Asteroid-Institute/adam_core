"""Execute every intended Python snippet in the canonical OD documentation.

The prose deliberately assumes caller-owned tables. This validation runner
supplies the repository's deterministic two-body fixture, first proves the
exact directive/order/identity contract, and then executes every Python block
in narrative order. Run it from a source environment containing the built
native extension::

    python docs/scripts/run_orbit_determination_examples.py
"""

from __future__ import annotations

import inspect
import re
import textwrap
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import pyarrow as pa

from adam_core.observations import Associations, SourceCatalog
from adam_core.orbit_determination import (
    FittedOrbitMembers,
    FittedOrbits,
    OdResourceValidationError,
    OrbitDeterminationFailures,
    evaluate_orbits,
    refine,
    run_iod,
    run_od,
)
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbit_determination.tests.test_batch_workflows import (
    _assert_exact_results,
    _concat_failures,
    _two_body_source,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CodeBlock:
    """One parsed reStructuredText code-block directive and body."""

    language: str
    body: str


GUIDE_BLOCK_CONTRACT = (
    ("python", "one_group = Associations.from_kwargs"),
    ("python", "config = CORE_OD_V1.bind"),
    ("python", "overlapping = Associations.from_kwargs"),
    ("python", "iod_orbits, iod_members, iod_failures = run_iod"),
    ("python", "selected_id = fitted_orbits.orbit_id[0].as_py()"),
    ("python", 'new_id = "analyst-label"'),
    ("python", "institute_config = ASTEROID_INSTITUTE_OD_V1.bind(resources)"),
    ("rust", "fn workflows("),
    ("console", "cargo run -p adam_core --example od_canonical_workflows"),
    ("console", "cargo run -p adam_core --example od_external_provider"),
    ("console", "python docs/scripts/run_orbit_determination_examples.py"),
    ("rust", "let resources = ResourceBindings::new().bind("),
)
WORKFLOW_BLOCK_CONTRACT = {
    "run_iod": (("python", "one_group = Associations.from_kwargs"),),
    "run_od": (("python", "config = CORE_OD_V1.bind"),),
    "refine": (("python", "iod_orbits, iod_members, iod_failures = run_iod"),),
    "evaluate_orbits": (("python", "evaluated_orbits, evaluated_members, failures"),),
}


def fixture_source() -> SourceCatalog:
    """Load deterministic observations used by the native two-body OD tests."""

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


def code_blocks(text: str) -> list[CodeBlock]:
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


def assert_block_contract(
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


def assert_products(namespace: dict[str, Any], names: tuple[str, str, str]) -> None:
    """Assert one documented result tuple has the three canonical table types."""

    orbits, members, failures = (namespace[name] for name in names)
    assert isinstance(orbits, FittedOrbits)
    assert isinstance(members, FittedOrbitMembers)
    assert isinstance(failures, OrbitDeterminationFailures)


def execute(code: str, namespace: dict[str, Any], label: str) -> None:
    exec(compile(code, label, "exec"), namespace)


def main() -> None:
    source_catalog = fixture_source()
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

    guide = ROOT / "docs" / "source" / "use_cases" / "orbit_determination.rst"
    guide_blocks = code_blocks(guide.read_text())
    assert_block_contract(guide_blocks, GUIDE_BLOCK_CONTRACT, str(guide))
    python_blocks = guide_blocks[:7]
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
            execute(block.body, local, label)
            assert_products(local, ("orbits", "members", "failures"))
        elif index == 6:
            try:
                execute(block.body, dict(namespace), label)
            except OdResourceValidationError:
                pass
            else:
                raise AssertionError(
                    "illustrative missing resources did not fail closed"
                )
        else:
            execute(block.body, namespace, label)
            expected_products = {
                0: [("fitted_orbits", "fitted_members", "failures")],
                1: [("fitted_orbits", "fitted_members", "failures")],
                3: [
                    ("iod_orbits", "iod_members", "iod_failures"),
                    ("refined_orbits", "refined_members", "refine_failures"),
                ],
                4: [("evaluated_orbits", "evaluated_members", "evaluation_failures")],
            }.get(index, [])
            for names in expected_products:
                assert_products(namespace, names)

    # Complete the relabel recipe with both downstream workflows.
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

    # Exercise exact documented composition, including carried IOD failures.
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
        blocks = code_blocks(inspect.getdoc(workflow) or "")
        assert_block_contract(
            blocks, WORKFLOW_BLOCK_CONTRACT[name], f"{name} docstring"
        )
        block = blocks[0]
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
        execute(block.body, local, f"{name}.__doc__[0]")

    print("canonical OD Python narrative examples passed")


if __name__ == "__main__":
    main()
