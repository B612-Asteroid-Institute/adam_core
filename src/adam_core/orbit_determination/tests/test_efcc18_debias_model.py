"""EFCC18 is a closed, resource-bound native transform."""

import pytest

from ..config import (
    IodStageConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
)
from ..observation_pipeline import (
    Efcc18CatalogBiasCorrection,
    ObservationPipeline,
)


def test_efcc18_requires_its_semantic_resource_slot() -> None:
    with pytest.raises(Exception, match="efcc18_catalog_bias_table"):
        OdConfig.explicit(
            observation_pipeline=ObservationPipeline.ordered(
                [Efcc18CatalogBiasCorrection(())]
            ),
            iod=IodStageConfig.core_v1(),
            refinement=RefinementRunConfig.core_v1(),
            resources=ResourceBindings({}),
        )


def test_efcc18_exclusion_order_is_canonical() -> None:
    transform = Efcc18CatalogBiasCorrection(("UCAC4", "Gaia2"))
    assert transform._descriptor() == {
        "kind": "efcc18_catalog_bias_correction",
        "exclude_astcats": ["UCAC4", "Gaia2"],
    }
