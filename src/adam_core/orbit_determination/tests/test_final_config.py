from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import FrozenInstanceError

import pytest

import adam_core.orbit_determination as od


def test_public_surface_is_final_and_legacy_names_are_absent() -> None:
    expected = {
        "OdConfig",
        "IodRunConfig",
        "RefinementRunConfig",
        "EvaluationConfig",
        "ObservationPipeline",
        "FillMissingAstrometricUncertainty",
        "Efcc18CatalogBiasCorrection",
        "StationBandResidualCovarianceAdjustment",
        "NightlyBatchDeweighting",
        "CORE_OD_V1",
        "ASTEROID_INSTITUTE_OD_V1",
        "run_iod",
        "run_od",
        "refine",
        "evaluate_orbits",
    }
    assert expected <= set(od.__all__)
    for removed in (
        "ObservationUncertaintyModel",
        "CompositeModel",
        "apply_observation_models",
        "initial_orbit_determination",
        "iod",
        "differential_correction",
        "NativeOrbitFitter",
        "OrbitFitter",
    ):
        assert removed not in od.__all__
        assert not callable(getattr(od, removed, None))


@pytest.mark.parametrize(
    "module",
    [
        "adam_core.orbit_determination.observation_uncertainty",
        "adam_core.orbit_determination.veres2017",
        "adam_core.orbit_determination.native_orbit_fitter",
        "adam_core.orbit_determination.orbit_fitter",
        "adam_core.orbit_determination.od",
        "adam_core.orbit_determination.model_provenance",
        "adam_core.orbit_determination.provenance",
    ],
)
def test_removed_legacy_modules_have_no_compatibility_alias(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_all_high_level_workflows_require_config_without_legacy_kwargs() -> None:
    for function in (od.run_iod, od.run_od, od.refine, od.evaluate_orbits):
        parameters = inspect.signature(function).parameters
        assert "config" in parameters
        assert parameters["config"].default is inspect.Parameter.empty
        assert "models" not in parameters
        assert "observatory_bias_model" not in parameters


def test_profiles_are_native_mirrors_with_stable_order() -> None:
    assert od.CORE_OD_V1.name == "CORE_OD_V1"
    assert [
        type(value).__name__ for value in od.CORE_OD_V1.observation_pipeline.transforms
    ] == [
        "Efcc18CatalogBiasCorrection",
        "NightlyBatchDeweighting",
    ]
    assert [
        type(value).__name__
        for value in od.ASTEROID_INSTITUTE_OD_V1.observation_pipeline.transforms
    ] == [
        "FillMissingAstrometricUncertainty",
        "Efcc18CatalogBiasCorrection",
        "StationBandResidualCovarianceAdjustment",
        "NightlyBatchDeweighting",
    ]
    assert len(od.CORE_OD_V1.definition_sha256) == 64
    assert len(od.ASTEROID_INSTITUTE_OD_V1.definition_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        od.CORE_OD_V1.name = "changed"  # type: ignore[misc]


def test_missing_profile_resources_are_aggregated_before_science() -> None:
    with pytest.raises(ValueError) as caught:
        od.ASTEROID_INSTITUTE_OD_V1.bind(od.ResourceBindings({}))
    message = str(caught.value)
    assert "station_catalog_uncertainty_table" in message
    assert "efcc18_catalog_bias_table" in message
    assert "station_band_residual_covariance_table" in message


def test_explicit_as_supplied_config_has_empty_transform_provenance() -> None:
    config = od.OdConfig.explicit(
        observation_pipeline=od.ObservationPipeline.as_supplied(),
        iod=od.IodRunConfig.core_v1(),
        refinement=od.RefinementRunConfig.core_v1(),
        evaluation=od.EvaluationConfig.six_dimensional(),
        resources=od.ResourceBindings({}),
    )
    assert config.profile_name is None
    assert config.observation_pipeline.is_as_supplied
    provenance = config.provenance
    assert provenance["transforms"] == []
    assert provenance["iod"]["search_correction_ranging"]["min_obs"] == 6
    assert provenance["iod"]["candidate_correction_fit"]["max_iterations"] == 100
    assert provenance["refinement"]["method"] == "cmc2003"
    assert provenance["refinement"]["fit"]["jacobian"] == "auto"
    assert "WhitenedFitConfig" not in json.dumps(dict(provenance))
    assert config.for_evaluation().covariance_dimension == 6


def test_manual_pipeline_preserves_repeated_transform_order() -> None:
    pipeline = od.ObservationPipeline.ordered(
        (
            od.NightlyBatchDeweighting(4),
            od.NightlyBatchDeweighting(1),
        )
    )
    config = od.OdConfig.explicit(
        observation_pipeline=pipeline,
        iod=od.IodRunConfig.core_v1(),
        refinement=od.RefinementRunConfig.core_v1(),
        evaluation=od.EvaluationConfig.six_dimensional(),
        resources=od.ResourceBindings({}),
    )
    assert config._native.transform_kinds() == [
        "nightly_batch_deweighting",
        "nightly_batch_deweighting",
    ]


def _observations(count: int = 5):
    import numpy as np

    from adam_core.coordinates import (
        CoordinateCovariances,
        Origin,
        SphericalCoordinates,
    )
    from adam_core.observers import Observers
    from adam_core.time import Timestamp

    from ..evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry

    times = Timestamp.from_mjd(60000.0 + 0.01 * np.arange(count), scale="utc")
    covariance = np.full((count, 6, 6), np.nan)
    covariance[:, 1, 1] = 1.0e-8
    covariance[:, 2, 2] = 4.0e-8
    covariance[:, 1, 2] = covariance[:, 2, 1] = 1.0e-8
    coordinates = SphericalCoordinates.from_kwargs(
        lon=10.0 + np.arange(count),
        lat=20.0 + np.arange(count),
        covariance=CoordinateCovariances.from_matrix(covariance),
        time=times,
        origin=Origin.from_kwargs(code=["500"] * count),
        frame="equatorial",
    )
    return OrbitDeterminationObservations.from_kwargs(
        id=[f"obs-{index}" for index in range(count)],
        coordinates=coordinates,
        observers=Observers.from_codes(times=times, codes=["500"] * count),
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=[None] * count,
            rmsmag=[None] * count,
            band=[None] * count,
        ),
        astcat=[None] * count,
    )


def _config(pipeline: od.ObservationPipeline) -> od.OdConfig:
    return od.OdConfig.explicit(
        observation_pipeline=pipeline,
        iod=od.IodRunConfig.core_v1(),
        refinement=od.RefinementRunConfig.core_v1(),
        evaluation=od.EvaluationConfig.six_dimensional(),
        resources=od.ResourceBindings({}),
    )


def test_as_supplied_pipeline_preserves_exact_caller_astrometry() -> None:
    import numpy as np

    from .._pipeline import apply_pipeline_once

    original = _observations()
    used, output = apply_pipeline_once(
        original, _config(od.ObservationPipeline.as_supplied())
    )
    np.testing.assert_array_equal(original.coordinates.values, used.coordinates.values)
    np.testing.assert_array_equal(
        original.coordinates.covariance.to_matrix(),
        used.coordinates.covariance.to_matrix(),
    )
    assert output["pipeline_changed"] is False


def test_nightly_pipeline_scales_complete_covariance_block_once() -> None:
    import numpy as np

    from .._pipeline import apply_pipeline_once

    original = _observations(5)
    pipeline = od.ObservationPipeline.ordered((od.NightlyBatchDeweighting(4),))
    used, output = apply_pipeline_once(original, _config(pipeline))
    before = original.coordinates.covariance.to_matrix()
    after = used.coordinates.covariance.to_matrix()
    np.testing.assert_allclose(after[:, 1:3, 1:3], before[:, 1:3, 1:3] * 1.25)
    np.testing.assert_array_equal(original.coordinates.values, used.coordinates.values)
    assert output["pipeline_changed"] is True


def test_run_od_matches_explicit_run_iod_then_refine() -> None:
    import numpy as np

    from .test_differential_correction import (
        TwoBodyPropagator,
        make_synthetic_observations,
    )

    observations = make_synthetic_observations(n_obs=12, seed=10)
    config = _config(od.ObservationPipeline.as_supplied())
    propagator = TwoBodyPropagator()
    iod_orbits, _iod_members, report = od.run_iod(
        observations, propagator, config=config
    )
    assert report.found
    composed, composed_members = od.refine(
        iod_orbits, observations, propagator, config=config
    )
    fused, fused_members = od.run_od(
        observations, propagator, config=config, object_id="synthetic"
    )
    np.testing.assert_allclose(
        fused.coordinates.values, composed.coordinates.values, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(fused_members.residuals.values.to_pylist(), dtype=float),
        np.asarray(composed_members.residuals.values.to_pylist(), dtype=float),
        rtol=0,
        atol=1e-11,
    )
    for name in fused_members.used_astrometry.schema.names:
        np.testing.assert_allclose(
            np.asarray(fused_members.used_astrometry.table.column(name).to_pylist()),
            np.asarray(composed_members.used_astrometry.table.column(name).to_pylist()),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
