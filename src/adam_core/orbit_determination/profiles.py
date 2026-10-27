"""Immutable path-free orbit-determination profiles.

Core exposes semantic resource slots only.  Callers explicitly bind local
files; profiles never inspect installed packages, download data, choose a
propagator, or fall back to another profile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from adam_core import _rust_native as _rust

from .config import OdConfig, ResourceBindings
from .observation_pipeline import (
    Efcc18CatalogBiasCorrection,
    FillMissingAstrometricUncertainty,
    NightlyBatchDeweighting,
    ObservationPipeline,
    ObservationTransform,
    ResidualCovarianceOperation,
    StationBandResidualCovarianceAdjustment,
)


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    slot: str
    product: str
    product_version: str
    schema_version: str
    sha256: str
    byte_size: int
    row_count: int | None
    catalog_date: str | None
    format: str


@dataclass(frozen=True, slots=True)
class OdProfile:
    name: str
    version: int
    definition_sha256: str
    observation_pipeline: ObservationPipeline
    requirements: tuple[ResourceRequirement, ...]

    def bind(self, resources: ResourceBindings) -> OdConfig:
        """Validate all required bytes, parse them in Rust, and expand config."""

        if not isinstance(resources, ResourceBindings):
            raise TypeError("resources must be ResourceBindings")
        native = _rust.bind_od_profile(self.name, resources._native_paths())
        return OdConfig(native, self.observation_pipeline)


def _requirements(name: str) -> tuple[ResourceRequirement, ...]:
    raw = json.loads(_rust.od_profile_requirements(name))
    return tuple(ResourceRequirement(**requirement) for requirement in raw)


def _profile(name: str) -> OdProfile:
    payload = json.loads(_rust.od_profile_definition(name))
    identity = payload["identity"]
    definitions = payload["definition"]["pipeline"]
    transforms: list[ObservationTransform] = []
    for definition in definitions:
        kind = definition["kind"]
        if kind == "fill_missing_astrometric_uncertainty":
            transforms.append(FillMissingAstrometricUncertainty())
        elif kind == "efcc18_catalog_bias_correction":
            transforms.append(
                Efcc18CatalogBiasCorrection(
                    exclude_astcats=tuple(definition["exclude_astcats"])
                )
            )
        elif kind == "station_band_residual_covariance_adjustment":
            transforms.append(
                StationBandResidualCovarianceAdjustment(
                    operation=ResidualCovarianceOperation(definition["operation"]),
                    minimum_residual_count=definition["minimum_residual_count"],
                )
            )
        elif kind == "nightly_batch_deweighting":
            transforms.append(
                NightlyBatchDeweighting(
                    effective_observation_cap=definition["effective_observation_cap"]
                )
            )
        else:  # pragma: no cover - native/Python version skew guard
            raise RuntimeError(f"unsupported native profile transform {kind!r}")
    return OdProfile(
        name=identity["name"],
        version=identity["version"],
        definition_sha256=identity["definition_sha256"],
        observation_pipeline=ObservationPipeline.ordered(transforms),
        requirements=_requirements(name),
    )


CORE_OD_V1 = _profile("CORE_OD_V1")
ASTEROID_INSTITUTE_OD_V1 = _profile("ASTEROID_INSTITUTE_OD_V1")

PROFILES: Mapping[str, OdProfile] = MappingProxyType(
    {
        CORE_OD_V1.name: CORE_OD_V1,
        ASTEROID_INSTITUTE_OD_V1.name: ASTEROID_INSTITUTE_OD_V1,
    }
)

__all__ = [
    "ASTEROID_INSTITUTE_OD_V1",
    "CORE_OD_V1",
    "OdProfile",
    "PROFILES",
    "ResourceRequirement",
]
