"""Closed observation-pipeline declaration and validation regressions."""

from dataclasses import FrozenInstanceError

import pytest

from ..config import (
    EvaluationConfig,
    IodRunConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
)
from ..observation_pipeline import (
    Efcc18CatalogBiasCorrection,
    FillMissingAstrometricUncertainty,
    NightlyBatchDeweighting,
    ObservationPipeline,
    ResidualCovarianceOperation,
    StationBandResidualCovarianceAdjustment,
)


def _explicit(pipeline, resources=ResourceBindings({})):
    return OdConfig.explicit(
        observation_pipeline=pipeline,
        iod=IodRunConfig.core_v1(),
        refinement=RefinementRunConfig.core_v1(),
        evaluation=EvaluationConfig.six_dimensional(),
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
        (Efcc18CatalogBiasCorrection(()), "efcc18_catalog_bias_table"),
        (
            StationBandResidualCovarianceAdjustment(
                ResidualCovarianceOperation.ADD, 30
            ),
            "station_band_residual_covariance_table",
        ),
    ):
        with pytest.raises(Exception, match=slot):
            _explicit(ObservationPipeline.ordered([transform]))


def test_unused_binding_is_rejected_before_file_access() -> None:
    with pytest.raises(Exception, match="unused|bogus"):
        _explicit(
            ObservationPipeline.ordered([NightlyBatchDeweighting(4)]),
            ResourceBindings({"bogus": "/does/not/exist"}),
        )


def test_covariance_operation_is_closed() -> None:
    add = StationBandResidualCovarianceAdjustment(ResidualCovarianceOperation.ADD, 30)
    replace = StationBandResidualCovarianceAdjustment(
        ResidualCovarianceOperation.REPLACE, 12
    )
    assert add.operation.value == "add"
    assert replace.operation.value == "replace"
