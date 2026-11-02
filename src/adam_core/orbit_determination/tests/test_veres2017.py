"""Station/catalog uncertainty data is consumed only by a native transform."""

import pytest

from ..config import (
    EvaluationConfig,
    IodRunConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
)
from ..observation_pipeline import (
    FillMissingAstrometricUncertainty,
    ObservationPipeline,
)
from ..profiles import ASTEROID_INSTITUTE_OD_V1, CORE_OD_V1


def test_fill_transform_requires_versioned_station_catalog_resource() -> None:
    with pytest.raises(Exception, match="station_catalog_uncertainty_table"):
        OdConfig.explicit(
            observation_pipeline=ObservationPipeline.ordered(
                [FillMissingAstrometricUncertainty()]
            ),
            iod=IodRunConfig.core_v1(),
            refinement=RefinementRunConfig.core_v1(),
            evaluation=EvaluationConfig.six_dimensional(),
            resources=ResourceBindings({}),
        )


def test_institute_but_not_core_profile_declares_sigma_fill() -> None:
    institute = [
        type(item).__name__
        for item in ASTEROID_INSTITUTE_OD_V1.observation_pipeline.transforms
    ]
    core = [type(item).__name__ for item in CORE_OD_V1.observation_pipeline.transforms]
    assert "FillMissingAstrometricUncertainty" in institute
    assert "FillMissingAstrometricUncertainty" not in core


def test_no_hidden_fallback_uncertainty_is_in_profile_provenance() -> None:
    descriptors = [
        item._descriptor()
        for item in ASTEROID_INSTITUTE_OD_V1.observation_pipeline.transforms
    ]
    fill = next(
        item
        for item in descriptors
        if item["kind"] == "fill_missing_astrometric_uncertainty"
    )
    assert set(fill) == {"kind"}
