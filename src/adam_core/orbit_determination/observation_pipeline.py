"""Declarative, Rust-executed observation pipelines for orbit determination.

These immutable Python values contain no executable science.  They are
validated, bound to explicitly supplied resources, and executed by Core's
native Rust implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Iterable


class ResidualCovarianceOperation(str, Enum):
    """How a station/band residual covariance adjusts supplied covariance."""

    ADD = "add"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class FillMissingAstrometricUncertainty:
    """Fill missing RA/Dec marginal uncertainty from the bound station table."""

    kind: Final[str] = field(
        default="fill_missing_astrometric_uncertainty", init=False, repr=False
    )

    def _descriptor(self) -> dict[str, object]:
        return {"kind": self.kind}


@dataclass(frozen=True, slots=True)
class Efcc18CatalogBiasCorrection:
    """Apply the EFCC18 catalog-dependent astrometric bias correction."""

    exclude_astcats: tuple[str, ...]
    kind: Final[str] = field(
        default="efcc18_catalog_bias_correction", init=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "exclude_astcats", tuple(self.exclude_astcats))
        if any(not isinstance(value, str) for value in self.exclude_astcats):
            raise TypeError("exclude_astcats must contain only strings")

    def _descriptor(self) -> dict[str, object]:
        return {"kind": self.kind, "exclude_astcats": list(self.exclude_astcats)}


@dataclass(frozen=True, slots=True)
class StationBandResidualCovarianceAdjustment:
    """Add or replace covariance using station/band residual statistics."""

    operation: ResidualCovarianceOperation
    minimum_residual_count: int
    kind: Final[str] = field(
        default="station_band_residual_covariance_adjustment",
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.operation, ResidualCovarianceOperation):
            raise TypeError("operation must be ResidualCovarianceOperation")
        if (
            isinstance(self.minimum_residual_count, bool)
            or self.minimum_residual_count <= 0
        ):
            raise ValueError("minimum_residual_count must be a positive integer")

    def _descriptor(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "operation": self.operation.value,
            "minimum_residual_count": self.minimum_residual_count,
        }


@dataclass(frozen=True, slots=True)
class NightlyBatchDeweighting:
    """Deweight station/UTC-night batches above an effective observation cap.

    For ``N`` observations from the same observatory and ``floor(UTC MJD)``,
    Rust scales the complete RA/Dec covariance block by ``max(N / cap, 1)``.
    """

    effective_observation_cap: int
    kind: Final[str] = field(
        default="nightly_batch_deweighting", init=False, repr=False
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.effective_observation_cap, bool)
            or self.effective_observation_cap <= 0
        ):
            raise ValueError("effective_observation_cap must be a positive integer")

    def _descriptor(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "effective_observation_cap": self.effective_observation_cap,
        }


ObservationTransform = (
    FillMissingAstrometricUncertainty
    | Efcc18CatalogBiasCorrection
    | StationBandResidualCovarianceAdjustment
    | NightlyBatchDeweighting
)


@dataclass(frozen=True, slots=True)
class ObservationPipeline:
    """An explicit as-supplied policy or an ordered sequence of transforms.

    Order and repeated transform instances are scientifically significant.
    No transform is sorted, combined, or deduplicated.
    """

    transforms: tuple[ObservationTransform, ...]
    _uses_as_supplied: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if self._uses_as_supplied and self.transforms:
            raise ValueError("as-supplied pipelines cannot contain transforms")
        if not self._uses_as_supplied and not self.transforms:
            raise ValueError(
                "an empty transform list is ambiguous; use "
                "ObservationPipeline.as_supplied()"
            )
        valid = (
            FillMissingAstrometricUncertainty,
            Efcc18CatalogBiasCorrection,
            StationBandResidualCovarianceAdjustment,
            NightlyBatchDeweighting,
        )
        if any(not isinstance(transform, valid) for transform in self.transforms):
            raise TypeError("pipeline contains an unsupported observation transform")

    @classmethod
    def as_supplied(cls) -> ObservationPipeline:
        """Explicitly preserve caller astrometry without transformation."""

        return cls((), _uses_as_supplied=True)

    @classmethod
    def ordered(cls, transforms: Iterable[ObservationTransform]) -> ObservationPipeline:
        return cls(tuple(transforms))

    @property
    def is_as_supplied(self) -> bool:
        return self._uses_as_supplied

    def _descriptors(self) -> list[dict[str, object]] | None:
        if self._uses_as_supplied:
            return None
        return [transform._descriptor() for transform in self.transforms]

    @classmethod
    def _from_descriptors(
        cls, descriptors: Iterable[dict[str, object]]
    ) -> ObservationPipeline:
        transforms: list[ObservationTransform] = []
        for descriptor in descriptors:
            kind = descriptor.get("kind")
            if kind == "fill_missing_astrometric_uncertainty":
                transforms.append(FillMissingAstrometricUncertainty())
            elif kind == "efcc18_catalog_bias_correction":
                values = descriptor.get("exclude_astcats")
                if not isinstance(values, list) or any(
                    not isinstance(value, str) for value in values
                ):
                    raise RuntimeError("native EFCC18 declaration is malformed")
                transforms.append(
                    Efcc18CatalogBiasCorrection(exclude_astcats=tuple(values))
                )
            elif kind == "station_band_residual_covariance_adjustment":
                operation = descriptor.get("operation")
                count = descriptor.get("minimum_residual_count")
                if not isinstance(operation, str) or not isinstance(count, int):
                    raise RuntimeError(
                        "native residual covariance declaration is malformed"
                    )
                transforms.append(
                    StationBandResidualCovarianceAdjustment(
                        operation=ResidualCovarianceOperation(operation),
                        minimum_residual_count=count,
                    )
                )
            elif kind == "nightly_batch_deweighting":
                cap = descriptor.get("effective_observation_cap")
                if not isinstance(cap, int):
                    raise RuntimeError("native nightly declaration is malformed")
                transforms.append(NightlyBatchDeweighting(cap))
            else:  # pragma: no cover - native/Python version skew guard
                raise RuntimeError(f"unsupported native transform {kind!r}")
        return cls.as_supplied() if not transforms else cls.ordered(transforms)


__all__ = [
    "Efcc18CatalogBiasCorrection",
    "FillMissingAstrometricUncertainty",
    "NightlyBatchDeweighting",
    "ObservationPipeline",
    "ObservationTransform",
    "ResidualCovarianceOperation",
    "StationBandResidualCovarianceAdjustment",
]
