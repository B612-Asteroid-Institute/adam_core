from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import adam_core.orbit_determination as od
from adam_core import _rust_native
from adam_core.observations import Associations, SourceCatalog
from adam_core.orbit_determination import algorithms


def _as_supplied() -> od.OdConfig:
    return od.OdConfig.explicit(
        observation_pipeline=od.ObservationPipeline.as_supplied(),
        iod=od.IodRunConfig.core_v1(),
        refinement=od.RefinementRunConfig.core_v1(),
        evaluation=od.EvaluationConfig.six_dimensional(),
        resources=od.ResourceBindings({}),
    )


def test_primary_batch_names_are_exported_without_one_group_aliases() -> None:
    for name in ("run_iod", "run_od", "refine", "evaluate_orbits"):
        assert name in od.__all__
        assert callable(getattr(od, name))
    assert "OrbitDeterminationFailures" in od.__all__
    for name in (
        "run_iod_one_group",
        "run_od_one_group",
        "refine_one_group",
        "evaluate_orbits_one_group",
    ):
        assert name not in algorithms.__all__
        assert not hasattr(algorithms, name)


def test_iod_projection_retains_the_explicit_bound_workflow_context() -> None:
    config = _as_supplied()
    assert config.for_iod()._workflow_native is config._native
    assert config.for_evaluation()._workflow_native is config._native
    assert od.IodRunConfig.core_v1()._workflow_native is None
    assert od.EvaluationConfig.six_dimensional()._workflow_native is None


def test_empty_batch_returns_three_typed_empty_tables_without_provider_binding() -> (
    None
):
    fitted, members, failures = od.run_iod(
        SourceCatalog.empty(),
        Associations.empty(),
        object(),
        _as_supplied().for_iod(),
    )
    assert isinstance(fitted, od.FittedOrbits)
    assert isinstance(members, od.FittedOrbitMembers)
    assert isinstance(failures, od.OrbitDeterminationFailures)
    assert len(fitted) == len(members) == len(failures) == 0


def test_removed_legacy_modules_and_native_routes_have_no_aliases() -> None:
    removed = (
        "adam_core.orbit_determination.iod",
        "adam_core.orbit_determination.od_orchestration",
        "adam_core.orbit_determination.observation_uncertainty",
        "adam_core.orbit_determination.veres2017",
        "adam_core.orbit_determination.native_orbit_fitter",
        "adam_core.orbit_determination.orbit_fitter",
        "adam_core.orbit_determination.od",
        "adam_core.orbit_determination.model_provenance",
        "adam_core.orbit_determination.provenance",
    )
    for module in removed:
        with pytest.raises(ModuleNotFoundError):
            __import__(module)
    for symbol in (
        "run_od_ipc",
        "run_od_2body_ipc",
        "full_od_ipc",
        "iod_fit_ipc",
        "run_iod_linkages_ipc",
        "validate_iod_settings",
        "validate_full_od_settings",
        "iod_defaults",
        "sort_linkages_by_id_and_time_numpy",
    ):
        assert not hasattr(_rust_native, symbol)


def test_complete_stage_configs_are_explicitly_constructible() -> None:
    core_iod = od.IodRunConfig.core_v1()
    search = dataclasses.replace(core_iod.search_correction_ranging, min_obs=7)
    two_body = dataclasses.replace(
        core_iod.candidate_correction_fit.two_body,
        lt_tol=2.0e-12,
        ephemeris_max_iter=777,
        max_lt_iter=11,
    )
    fit = dataclasses.replace(
        core_iod.candidate_correction_fit,
        loss="huber",
        f_scale=1.2,
        two_body=two_body,
    )
    explicit_iod = od.IodRunConfig(search, fit)
    assert explicit_iod.search_correction_ranging.min_obs == 7
    native_iod = json.loads(explicit_iod._native.canonical_json)
    assert native_iod["candidate_correction_fit"]["loss"] == "huber"
    assert (
        native_iod["candidate_correction_fit"]["two_body"]["ephemeris_max_iter"] == 777
    )

    core_refinement = od.RefinementRunConfig.core_v1().refinement
    assert isinstance(core_refinement, od.Cmc2003Refinement)
    worst = od.WorstResidualRefinement(
        method="worst_residual",
        rchi2_threshold=12.0,
        min_obs=7,
        min_arc_length=0.5,
        contamination_percentage=15.0,
        fit=fit,
    )
    explicit_refinement = od.RefinementRunConfig(worst)
    assert (
        json.loads(explicit_refinement._native.canonical_json)["method"]
        == "worst_residual"
    )
    assert od.EvaluationConfig(6).covariance_dimension == 6
    with pytest.raises(od.OdConfigurationError):
        od.EvaluationConfig(5)


def test_core_constants_match_d21_effective_values_exactly() -> None:
    iod = od.IodRunConfig.core_v1().candidate_correction_fit.two_body
    refinement = od.RefinementRunConfig.core_v1().refinement
    assert isinstance(refinement, od.Cmc2003Refinement)
    for two_body in (iod, refinement.fit.two_body):
        assert np.float64(two_body.lt_tol).view(np.uint64) == np.float64(1e-12).view(
            np.uint64
        )
        assert two_body.ephemeris_max_iter == 1000
        assert two_body.max_lt_iter == 10


def test_pipeline_requires_explicit_as_supplied_and_efcc_declaration() -> None:
    with pytest.raises(ValueError, match="as_supplied"):
        od.ObservationPipeline.ordered([])
    with pytest.raises(TypeError):
        od.Efcc18CatalogBiasCorrection()  # type: ignore[call-arg]
    supplied = od.ObservationPipeline.as_supplied()
    assert supplied.is_as_supplied
    assert supplied.transforms == ()


def test_profiles_are_native_sourced_and_not_constructible_or_replaceable() -> None:
    assert (
        od.CORE_OD_V1.definition_sha256
        == "fb921274a3daf4bb09885a4247a3f2598b141e355883dc2e2815cfbd5153810b"
    )
    assert (
        od.ASTEROID_INSTITUTE_OD_V1.definition_sha256
        == "47cf72508a4c593cd69aefa6f37f1f5d65991a9c76f72975e182c7592bc08b8c"
    )
    with pytest.raises(TypeError):
        od.OdProfile({}, (), _token=object())
    with pytest.raises(TypeError):
        dataclasses.replace(od.CORE_OD_V1)  # type: ignore[type-var]
    assert [
        type(value).__name__
        for value in od.ASTEROID_INSTITUTE_OD_V1.observation_pipeline.transforms
    ] == [
        "FillMissingAstrometricUncertainty",
        "Efcc18CatalogBiasCorrection",
        "StationBandResidualCovarianceAdjustment",
        "NightlyBatchDeweighting",
    ]


def test_resource_errors_are_typed_aggregate_and_slot_correct(tmp_path: Path) -> None:
    bindings = od.ResourceBindings(
        {
            requirement.slot: tmp_path / requirement.slot
            for requirement in od.ASTEROID_INSTITUTE_OD_V1.requirements
        }
    )
    with pytest.raises(od.OdResourceValidationError) as caught:
        od.ASTEROID_INSTITUTE_OD_V1.bind(bindings)
    message = str(caught.value)
    for requirement in od.ASTEROID_INSTITUTE_OD_V1.requirements:
        assert requirement.slot in message

    with pytest.raises(od.OdResourceValidationError, match="does not use"):
        od.OdConfig.explicit(
            observation_pipeline=od.ObservationPipeline.ordered(
                [od.NightlyBatchDeweighting(4)]
            ),
            iod=od.IodRunConfig.core_v1(),
            refinement=od.RefinementRunConfig.core_v1(),
            evaluation=od.EvaluationConfig(6),
            resources=od.ResourceBindings(
                {"efcc18_catalog_bias_table": tmp_path / "unused"}
            ),
        )


def test_native_transform_json_is_strict() -> None:
    iod = od.IodRunConfig.core_v1()._native
    refinement = od.RefinementRunConfig.core_v1()._native
    evaluation = od.EvaluationConfig(6)._native
    with pytest.raises(ValueError, match="exactly"):
        _rust_native.explicit_od_config(
            json.dumps(
                [
                    {
                        "kind": "nightly_batch_deweighting",
                        "effective_observation_cap": 4,
                        "bogus": 1,
                    }
                ]
            ),
            {},
            iod,
            refinement,
            evaluation,
        )
    with pytest.raises(ValueError, match="exactly"):
        _rust_native.explicit_od_config(
            json.dumps([{"kind": "efcc18_catalog_bias_correction"}]),
            {},
            iod,
            refinement,
            evaluation,
        )
