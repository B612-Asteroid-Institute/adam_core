"""Closed observation-pipeline declaration and validation regressions."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ..config import (
    IodStageConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
)
from ..observation_pipeline import (
    Efcc18CatalogBiasCorrection,
    FillMissingAstrometricUncertainty,
    MeasuredBiasAstrometricUncertaintyFloor,
    NightlyBatchDeweighting,
    ObservationPipeline,
    PerformanceWeightedCovarianceScaling,
    ResidualCovarianceOperation,
    StationBandResidualCovarianceAdjustment,
    Vfc2017AstrometricUncertaintyFloor,
    Vfc2017AstrometricUncertaintyReplace,
)


def _explicit(
    pipeline: ObservationPipeline,
    resources: ResourceBindings = ResourceBindings({}),
) -> OdConfig:
    return OdConfig.explicit(
        observation_pipeline=pipeline,
        iod=IodStageConfig.core_v1(),
        refinement=RefinementRunConfig.core_v1(),
        resources=resources,
    )


def test_as_supplied_is_the_only_empty_pipeline() -> None:
    assert ObservationPipeline.as_supplied().is_as_supplied
    with pytest.raises(ValueError, match="as_supplied"):
        ObservationPipeline.ordered([])
    assert _explicit(ObservationPipeline.as_supplied()).provenance["transforms"] == []


def test_order_and_repetition_are_authoritative() -> None:
    pipeline = ObservationPipeline.ordered(
        [
            NightlyBatchDeweighting(4),
            NightlyBatchDeweighting(2),
            NightlyBatchDeweighting(4),
        ]
    )
    config = _explicit(pipeline)
    transforms = config.provenance["transforms"]
    assert isinstance(transforms, list)
    assert all(isinstance(item, dict) for item in transforms)
    assert [item["effective_observation_cap"] for item in transforms] == [4, 2, 4]


def test_transform_declarations_are_immutable() -> None:
    transform = NightlyBatchDeweighting(4)
    with pytest.raises(FrozenInstanceError):
        transform.effective_observation_cap = 5  # type: ignore[misc]


def test_efcc_exclusions_are_explicit_and_preserved() -> None:
    transform = Efcc18CatalogBiasCorrection(("Gaia2", "UCAC4"))
    assert transform.exclude_astcats == ("Gaia2", "UCAC4")
    with pytest.raises(TypeError):
        Efcc18CatalogBiasCorrection()  # type: ignore[call-arg]


def test_resource_transforms_fail_without_exact_bindings() -> None:
    for transform, slot in (
        (FillMissingAstrometricUncertainty(), "station_catalog_uncertainty_table"),
        (
            Vfc2017AstrometricUncertaintyFloor(fill_missing=False),
            "station_catalog_uncertainty_table",
        ),
        (
            Vfc2017AstrometricUncertaintyReplace(),
            "station_catalog_uncertainty_table",
        ),
        (Efcc18CatalogBiasCorrection(()), "efcc18_catalog_bias_table"),
        (
            StationBandResidualCovarianceAdjustment(
                ResidualCovarianceOperation.ADD, 30
            ),
            "station_band_residual_covariance_table",
        ),
        (
            PerformanceWeightedCovarianceScaling(30),
            "station_band_residual_covariance_table",
        ),
        (
            MeasuredBiasAstrometricUncertaintyFloor(30),
            "station_band_residual_covariance_table",
        ),
    ):
        with pytest.raises(Exception, match=slot):
            _explicit(ObservationPipeline.ordered([transform]))


def test_optional_transform_descriptors_round_trip_in_exact_order() -> None:
    default_floor = Vfc2017AstrometricUncertaintyFloor()
    assert default_floor.fill_missing is False
    assert default_floor._descriptor()["fill_missing"] is False

    pipeline = ObservationPipeline.ordered(
        [
            Vfc2017AstrometricUncertaintyFloor(fill_missing=False),
            Vfc2017AstrometricUncertaintyFloor(fill_missing=True),
            Vfc2017AstrometricUncertaintyReplace(),
            PerformanceWeightedCovarianceScaling(31),
            MeasuredBiasAstrometricUncertaintyFloor(47),
        ]
    )
    descriptors = pipeline._descriptors()
    assert descriptors == [
        {
            "kind": "vfc2017_astrometric_uncertainty_floor",
            "fill_missing": False,
        },
        {"kind": "vfc2017_astrometric_uncertainty_floor", "fill_missing": True},
        {"kind": "vfc2017_astrometric_uncertainty_replace"},
        {
            "kind": "performance_weighted_covariance_scaling",
            "minimum_residual_count": 31,
        },
        {
            "kind": "measured_bias_astrometric_uncertainty_floor",
            "minimum_residual_count": 47,
        },
    ]
    assert ObservationPipeline._from_descriptors(descriptors or []) == pipeline


def test_optional_transform_descriptor_schema_is_closed() -> None:
    with pytest.raises(RuntimeError, match="invalid fields"):
        ObservationPipeline._from_descriptors(
            [
                {
                    "kind": "vfc2017_astrometric_uncertainty_replace",
                    "unknown": True,
                }
            ]
        )
    with pytest.raises(RuntimeError, match="invalid fields"):
        ObservationPipeline._from_descriptors(
            [{"kind": "performance_weighted_covariance_scaling"}]
        )
    with pytest.raises(TypeError, match="fill_missing"):
        Vfc2017AstrometricUncertaintyFloor(fill_missing=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        PerformanceWeightedCovarianceScaling(0)


def test_unused_binding_is_rejected_before_file_access() -> None:
    with pytest.raises(Exception, match="unused|bogus"):
        _explicit(
            ObservationPipeline.ordered([NightlyBatchDeweighting(4)]),
            ResourceBindings({"bogus": Path("/does/not/exist")}),
        )


def test_covariance_operation_is_closed() -> None:
    add = StationBandResidualCovarianceAdjustment(ResidualCovarianceOperation.ADD, 30)
    replace = StationBandResidualCovarianceAdjustment(
        ResidualCovarianceOperation.REPLACE, 12
    )
    assert add.operation.value == "add"
    assert replace.operation.value == "replace"
