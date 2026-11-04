"""Execute the Python snippets in the canonical OD guide and docstrings.

The prose deliberately assumes caller-owned tables. This validation runner
supplies the repository's deterministic two-body fixture and executes snippets
in narrative order. Run it from a source environment containing the built
native extension::

    python docs/scripts/run_orbit_determination_examples.py
"""

from __future__ import annotations

import inspect
import textwrap
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
from adam_core.orbit_determination.tests.test_batch_workflows import _two_body_source

ROOT = Path(__file__).resolve().parents[2]


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


def code_blocks(text: str) -> list[str]:
    """Extract indented ``code-block:: python`` bodies."""

    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != ".. code-block:: python":
            index += 1
            continue
        index += 1
        body: list[str] = []
        while index < len(lines) and (
            not lines[index].strip() or lines[index].startswith("   ")
        ):
            body.append(lines[index])
            index += 1
        blocks.append(textwrap.dedent("\n".join(body)).strip("\n"))
    return blocks


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
    for index, block in enumerate(code_blocks(guide.read_text())):
        label = f"{guide.name}:python[{index}]"
        if "overlapping = Associations.from_kwargs" in block:
            local = dict(namespace)
            local["source_catalog"] = source_catalog.set_column(
                "id",
                pa.array(
                    [f"obs-{row + 1}" for row in range(len(source_catalog))],
                    type=source_catalog.id.type,
                ),
            )
            execute(block, local, label)
            assert_products(local, ("orbits", "members", "failures"))
            continue
        if "ASTEROID_INSTITUTE_OD_V1.bind(resources)" in block:
            try:
                execute(block, dict(namespace), label)
            except OdResourceValidationError:
                continue
            raise AssertionError("illustrative missing resources did not fail closed")
        execute(block, namespace, label)
        for names in [
            ("fitted_orbits", "fitted_members", "failures"),
            ("iod_orbits", "iod_members", "iod_failures"),
            ("refined_orbits", "refined_members", "refine_failures"),
            ("evaluated_orbits", "evaluated_members", "evaluation_failures"),
        ]:
            if names[0] in block:
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

    # Exercise the documented exact composition, including carried IOD failures.
    direct = run_od(source_catalog, associations, None, namespace["config"])
    iod = run_iod(source_catalog, associations, None, namespace["config"].for_iod())
    staged = refine(
        iod[0],
        iod[1],
        source_catalog,
        None,
        namespace["config"].for_refinement(),
    )
    assert len(direct[0]) == len(staged[0])
    assert len(direct[1]) == len(staged[1])
    assert len(direct[2]) == len(iod[2]) + len(staged[2])

    for workflow in [run_iod, run_od, refine, evaluate_orbits]:
        for index, block in enumerate(code_blocks(inspect.getdoc(workflow) or "")):
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
            execute(block, local, f"{workflow.__name__}.__doc__[{index}]")

    print("canonical OD Python narrative examples passed")


if __name__ == "__main__":
    main()
