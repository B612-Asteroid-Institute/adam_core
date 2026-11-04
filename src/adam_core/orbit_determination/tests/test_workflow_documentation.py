"""Static guards for the four canonical OD workflow docstrings."""

from __future__ import annotations

import inspect

import pytest

from ..workflows import evaluate_orbits, refine, run_iod, run_od


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
