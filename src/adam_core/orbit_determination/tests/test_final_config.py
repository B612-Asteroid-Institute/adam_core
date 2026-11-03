from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import adam_core.orbit_determination as od
from adam_core import _rust_native
from adam_core.observations import Associations, SourceCatalog
from adam_core.orbit_determination import algorithms
from adam_core.orbit_determination import config as config_module
from adam_core.orbit_determination import workflows as workflows_module


def _as_supplied() -> od.OdConfig:
    return od.OdConfig.explicit(
        observation_pipeline=od.ObservationPipeline.as_supplied(),
        iod=od.IodStageConfig.core_v1(),
        refinement=od.RefinementRunConfig.core_v1(),
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


def test_python_workflow_signatures_and_config_state_do_not_drift() -> None:
    expected = {
        "run_iod": ["observations", "associations", "propagator", "config"],
        "run_od": ["observations", "associations", "propagator", "config"],
        "refine": ["orbits", "members", "observations", "propagator", "config"],
        "evaluate_orbits": [
            "orbits",
            "members",
            "observations",
            "propagator",
            "config",
        ],
    }
    for name, parameters in expected.items():
        assert list(inspect.signature(getattr(od, name)).parameters) == parameters
    assert "_workflow_native" not in inspect.getsource(config_module)
    assert "_workflow_native" not in inspect.getsource(workflows_module)


def test_stage_configs_are_self_contained_without_hidden_workflow_state() -> None:
    config = _as_supplied()
    iod = config.for_iod()
    assert isinstance(iod, od.IodRunConfig)
    assert iod.observation_pipeline == config.observation_pipeline
    assert iod.iod == config.iod
    assert iod.resources == config.resources
    assert not hasattr(iod, "_workflow_native")
    evaluation = config.for_evaluation()
    assert evaluation.prediction_fit == config.refinement.refinement.fit
    assert not hasattr(evaluation, "_workflow_native")

    direct = od.IodRunConfig(
        od.ObservationPipeline.as_supplied(),
        od.IodStageConfig.core_v1(),
        od.ResourceBindings({}),
    )
    assert direct.provenance["schema"] == "adam-core-iod-run-config/v1"


def test_empty_batch_returns_three_typed_empty_tables_without_provider_binding() -> (
    None
):
    fitted, members, failures = od.run_iod(
        SourceCatalog.empty(),
        Associations.empty(),
        None,
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
    core_iod = od.IodStageConfig.core_v1()
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
    explicit_iod = od.IodStageConfig(search, fit)
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
    evaluation = od.EvaluationConfig(6, fit)
    assert evaluation.covariance_dimension == 6
    assert evaluation.prediction_fit == fit
    with pytest.raises(od.OdConfigurationError):
        od.EvaluationConfig(5, fit)


def test_core_constants_match_d21_effective_values_exactly() -> None:
    iod = od.IodStageConfig.core_v1().candidate_correction_fit.two_body
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


def test_core_public_resource_binding_uses_installed_data_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bias_dat = tmp_path / "bias.dat"
    bias_dat.write_bytes(b"provisional-path-contract")
    package = types.ModuleType("jpl_debias_2018")
    package.bias_dat = str(bias_dat)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jpl_debias_2018", package)

    bindings = od.core_od_v1_resource_bindings()
    assert bindings.paths == {"efcc18_catalog_bias_table": bias_dat}


def test_published_core_resource_binds_and_validates_natively() -> None:
    import jpl_debias_2018

    bias_dat = Path(jpl_debias_2018.bias_dat)
    assert importlib.metadata.version("jpl-debias-2018") == "2023.3.1.1"
    assert bias_dat.is_absolute() and bias_dat.is_file()
    assert bias_dat.stat().st_size == 37_161_151
    with bias_dat.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == "ef6a50830bb83d8b1161e7acb391eed988ea9708e9fbcccc0c834da6109476e3"
        )

    bindings = od.core_od_v1_resource_bindings()
    assert bindings.paths == {"efcc18_catalog_bias_table": bias_dat}
    config = od.CORE_OD_V1.bind(bindings)
    assert config.profile_name == "CORE_OD_V1"
    resource = config.provenance["transforms"][0]["resource"]  # type: ignore[index]
    assert resource["actual_byte_size"] == 37_161_151
    assert (
        resource["actual_sha256"]
        == "ef6a50830bb83d8b1161e7acb391eed988ea9708e9fbcccc0c834da6109476e3"
    )


def test_profiles_are_native_sourced_and_not_constructible_or_replaceable() -> None:
    assert (
        od.CORE_OD_V1.definition_sha256
        == "fb921274a3daf4bb09885a4247a3f2598b141e355883dc2e2815cfbd5153810b"
    )
    assert (
        od.ASTEROID_INSTITUTE_OD_V1.definition_sha256
        == "7cf341740994fc918f3d4186fdbe4db75e8860698c99d85407ab12dc448b60c9"
    )
    with pytest.raises(TypeError):
        od.OdProfile({}, (), _token=object())
    with pytest.raises(TypeError):
        dataclasses.replace(od.CORE_OD_V1)  # type: ignore[type-var]
    requirements = {
        requirement.slot: requirement
        for requirement in od.ASTEROID_INSTITUTE_OD_V1.requirements
    }
    sigma = requirements["station_catalog_uncertainty_table"]
    assert (sigma.sha256, sigma.byte_size, sigma.row_count) == (
        "95a6d12c0dc1f6278a869e2e0d7d356b54683773d536d286f00326a3d868495b",
        52_288,
        2_480,
    )
    residual = requirements["station_band_residual_covariance_table"]
    assert (residual.sha256, residual.byte_size, residual.row_count) == (
        "ed0083b041fe523e6cfbabd3bff80bb0688b3056982de48d7789e59fdf370bac",
        158_965,
        1_126,
    )
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
            iod=od.IodStageConfig.core_v1(),
            refinement=od.RefinementRunConfig.core_v1(),
            resources=od.ResourceBindings(
                {"efcc18_catalog_bias_table": tmp_path / "unused"}
            ),
        )


def test_native_transform_json_is_strict() -> None:
    iod = od.IodStageConfig.core_v1()._native
    refinement = od.RefinementRunConfig.core_v1()._native
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
        )
    with pytest.raises(ValueError, match="exactly"):
        _rust_native.explicit_od_config(
            json.dumps([{"kind": "efcc18_catalog_bias_correction"}]),
            {},
            iod,
            refinement,
        )
