"""The Institute profile is a fixed ordered native resource pipeline."""

from ..observation_pipeline import ResidualCovarianceOperation
from ..profiles import ASTEROID_INSTITUTE_OD_V1


def test_institute_profile_pipeline_order_and_values() -> None:
    transforms = ASTEROID_INSTITUTE_OD_V1.observation_pipeline.transforms
    assert [type(item).__name__ for item in transforms] == [
        "FillMissingAstrometricUncertainty",
        "Efcc18CatalogBiasCorrection",
        "StationBandResidualCovarianceAdjustment",
        "NightlyBatchDeweighting",
    ]
    covariance = transforms[2]
    assert covariance.operation is ResidualCovarianceOperation.ADD
    assert covariance.minimum_residual_count == 30
    assert transforms[3].effective_observation_cap == 4


def test_institute_profile_declares_all_resource_slots() -> None:
    assert {item.slot for item in ASTEROID_INSTITUTE_OD_V1.requirements} == {
        "efcc18_catalog_bias_table",
        "station_catalog_uncertainty_table",
        "station_band_residual_covariance_table",
    }
