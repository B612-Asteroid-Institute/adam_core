"""A live Python model that reproducible run_od intentionally refuses."""

from importlib.metadata import version

from adam_core.orbit_determination import ModelProvenance
from adam_core.orbit_determination.observation_uncertainty import (
    ObservationUncertaintyModel,
)


class CalibratedModel(ObservationUncertaintyModel):
    """No-op live behavior used to prove package metadata is insufficient."""

    def __init__(self, scale: float = 1.5, mode: str = "calibrated") -> None:
        self.scale = scale
        self.mode = mode

    def apply(self, observations):
        return observations

    def model_provenance(self) -> ModelProvenance:
        return ModelProvenance(
            package="acme-orbit-models",
            version=version("acme-orbit-models"),
            implementation_fingerprint=(
                "ed8ab4b58f6c4b2bbf6c4c4d736942aa" "dddc774f5ac86b7b53c50a1072f900a1"
            ),
            configuration={"mode": self.mode, "scale": self.scale},
        )
