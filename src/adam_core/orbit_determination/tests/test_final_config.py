from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import adam_core.orbit_determination as od
from adam_core import _rust_native
from adam_core.orbit_determination import algorithms
from adam_core.orbit_determination.algorithms import _provenance
from adam_core.orbit_determination.algorithms.workflows import run_od_one_group
from adam_core.orbit_determination.tests.test_differential_correction import (
    TwoBodyPropagator,
    make_synthetic_observations,
)
from adam_core.propagator import Propagator


def _as_supplied() -> od.OdConfig:
    return od.OdConfig.explicit(
        observation_pipeline=od.ObservationPipeline.as_supplied(),
        iod=od.IodRunConfig.core_v1(),
        refinement=od.RefinementRunConfig.core_v1(),
        evaluation=od.EvaluationConfig.six_dimensional(),
        resources=od.ResourceBindings({}),
    )


def test_primary_batch_names_are_intentionally_not_exported() -> None:
    for name in ("run_iod", "run_od", "refine", "evaluate_orbits"):
        assert name not in od.__all__
        assert not hasattr(od, name)
    assert hasattr(algorithms, "run_iod_one_group")
    assert hasattr(algorithms, "run_od_one_group")
    assert hasattr(algorithms, "refine_one_group")
    assert hasattr(algorithms, "evaluate_orbits_one_group")


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


def test_duplicate_ids_fail_before_provider_or_pipeline(monkeypatch) -> None:
    observations = make_synthetic_observations(n_obs=12, seed=10).set_column(
        "id", pa.array(["same"] * 12, type=pa.large_string())
    )
    calls: list[str] = []
    monkeypatch.setattr(_provenance, "core_build_identity", lambda: ("0.7.0", "a" * 40))
    monkeypatch.setattr(
        "adam_core.orbit_determination.algorithms.workflows.apply_pipeline_once",
        lambda *_: calls.append("pipeline"),
    )
    with pytest.raises(ValueError, match="observation IDs must be unique"):
        run_od_one_group(observations, object(), config=_as_supplied())  # type: ignore[arg-type]
    assert calls == []


def test_empty_group_is_noop_before_provider_and_pipeline(monkeypatch) -> None:
    observations = make_synthetic_observations(n_obs=12, seed=10)[0:0]
    monkeypatch.setattr(_provenance, "core_build_identity", lambda: ("0.7.0", "a" * 40))
    monkeypatch.setattr(
        "adam_core.orbit_determination.algorithms.workflows.apply_pipeline_once",
        lambda *_: (_ for _ in ()).throw(AssertionError("pipeline called")),
    )
    fitted, members = run_od_one_group(observations, object(), config=_as_supplied())  # type: ignore[arg-type]
    assert len(fitted) == len(members) == 0


def test_provider_is_rejected_before_pipeline(monkeypatch) -> None:
    observations = make_synthetic_observations(n_obs=12, seed=10)
    calls: list[str] = []
    monkeypatch.setattr(_provenance, "core_build_identity", lambda: ("0.7.0", "a" * 40))
    monkeypatch.setattr(
        "adam_core.orbit_determination.algorithms.workflows.apply_pipeline_once",
        lambda *_: calls.append("pipeline"),
    )

    class PyOnly(Propagator):
        def propagate_orbits(self, *args, **kwargs):
            raise AssertionError("Python propagation must not run")

        def generate_ephemeris(self, *args, **kwargs):
            raise AssertionError("Python ephemeris must not run")

    with pytest.raises(TypeError, match="native"):
        run_od_one_group(observations, PyOnly(), config=_as_supplied())
    assert calls == []


def test_focused_run_od_is_bitwise_explicit_composition(monkeypatch) -> None:
    observations = make_synthetic_observations(n_obs=12, seed=10)
    config = _as_supplied()
    propagator = TwoBodyPropagator()
    monkeypatch.setattr(_provenance, "core_build_identity", lambda: ("0.7.0", "a" * 40))
    iod_orbits, _, report = algorithms.run_iod_one_group(
        observations, propagator, config=config
    )
    assert report.found
    composed, _ = algorithms.refine_one_group(
        iod_orbits, observations, propagator, config=config
    )
    direct, members = algorithms.run_od_one_group(
        observations, propagator, config=config, object_id="synthetic"
    )
    assert np.array_equal(direct.coordinates.values, composed.coordinates.values)
    assert np.array_equal(
        direct.coordinates.covariance.to_matrix(),
        composed.coordinates.covariance.to_matrix(),
        equal_nan=True,
    )
    provenance = direct.provenance
    values = {
        field: provenance.table.column(field)[0].as_py()
        for field in provenance.schema.names
    }
    assert values["adam_core_version"] == "0.7.0"
    assert values["adam_core_build"] == "a" * 40
    assert values["input_fingerprint"] == _provenance.input_fingerprint(
        observations, members
    )
    assert values["route"] == "algorithms.run_od_one_group"
    assert values["fitter"] is None
    assert values["settings"] is None

    iod_provider = json.loads(values["iod_provider"])
    refinement_provider = json.loads(values["refinement_provider"])
    provider_fields = {
        "name",
        "version",
        "abi_version",
        "identity_validated_by_capsule",
        "capabilities",
        "known_capabilities",
        "integration_time_scale",
        "two_body_dynamics",
        "configuration_fingerprint",
        "data_fingerprint",
    }
    assert set(iod_provider) == provider_fields
    assert set(refinement_provider) == provider_fields
    assert values["iod_backend"] == _provenance._backend(values["iod_provider"])
    assert values["refinement_backend"] == _provenance._backend(
        values["refinement_provider"]
    )

    iod_diagnostics = json.loads(values["iod_diagnostics"])
    assert iod_diagnostics == {
        "triplets_tried": report.triplets_tried,
        "candidates_evaluated": report.candidates_evaluated,
        "candidates_failed": report.candidates_failed,
        "corrections_run": report.corrections_run,
        "pair_visits": report.pair_visits,
        "raw_pair_visits": report.raw_pair_visits,
        "middle_visits": report.middle_visits,
        "search_cap_exhausted": report.search_cap_exhausted,
        "not_found_reason": report.not_found_reason,
    }
    refinement_diagnostics = json.loads(values["refinement_diagnostics"])
    assert set(refinement_diagnostics) == {
        "passes",
        "n_rejected",
        "n_recovered",
        "flags",
    }
    rejected = sum(members.outlier.to_pylist())
    assert refinement_diagnostics["n_rejected"] == rejected
    assert json.loads(values["configuration"]) == dict(config.provenance)
    assert values["iod_ambiguous"] is report.ambiguous
    assert values["iod_alternatives"] == len(report.ambiguous_with)
    assert values["num_rejected"] == rejected
    assert values["rejected_fraction"] == rejected / len(members)
    assert len(members) == len(observations)


def test_focused_signatures_require_complete_config() -> None:
    for function in (
        algorithms.run_iod_one_group,
        algorithms.run_od_one_group,
        algorithms.refine_one_group,
        algorithms.evaluate_orbits_one_group,
    ):
        parameter = inspect.signature(function).parameters["config"]
        assert parameter.default is inspect.Signature.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
