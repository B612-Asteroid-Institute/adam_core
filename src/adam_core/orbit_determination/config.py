"""Complete explicit Core-owned orbit-determination configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

from adam_core import _rust_native as _rust

from .observation_pipeline import ObservationPipeline


@dataclass(frozen=True, slots=True)
class ResourceBindings:
    """Concrete local paths keyed by stable semantic resource slot.

    Paths select bytes at deployment time and are never part of scientific
    configuration identity or provenance.
    """

    paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        normalized: dict[str, Path] = {}
        for slot, value in self.paths.items():
            if not isinstance(slot, str):
                raise TypeError("resource slots must be strings")
            normalized[slot] = Path(os.fspath(value))
        object.__setattr__(self, "paths", MappingProxyType(normalized))

    def _native_paths(self) -> dict[str, str]:
        return {slot: os.fspath(path) for slot, path in self.paths.items()}


@dataclass(frozen=True, slots=True)
class IodRunConfig:
    """Complete IOD/search and candidate-correction settings."""

    _native: Any

    @classmethod
    def core_v1(cls) -> IodRunConfig:
        """Select the named Core-v1 IOD settings explicitly."""

        return cls(_rust.IodRunConfig.core_v1())


@dataclass(frozen=True, slots=True)
class RefinementRunConfig:
    """Complete fit, Jacobian, covariance and rejection settings."""

    _native: Any

    @classmethod
    def core_v1(cls) -> RefinementRunConfig:
        """Select the named Core-v1 CMC2003 refinement settings explicitly."""

        return cls(_rust.RefinementRunConfig.core_v1())


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Complete orbit-evaluation settings."""

    _native: Any

    @classmethod
    def six_dimensional(cls) -> EvaluationConfig:
        """Require the supported six-dimensional covariance contract."""

        return cls(_rust.EvaluationConfig.six_dimensional())

    @property
    def covariance_dimension(self) -> int:
        return cast(int, self._native.covariance_dimension)


@dataclass(frozen=True, slots=True)
class OdConfig:
    """Validated complete Core-owned OD configuration.

    Construct manually with :meth:`explicit` or bind an immutable named
    profile. There is no implicit/default constructor.
    """

    _native: Any
    observation_pipeline: ObservationPipeline

    @classmethod
    def explicit(
        cls,
        *,
        observation_pipeline: ObservationPipeline,
        iod: IodRunConfig,
        refinement: RefinementRunConfig,
        evaluation: EvaluationConfig,
        resources: ResourceBindings,
    ) -> OdConfig:
        if not isinstance(observation_pipeline, ObservationPipeline):
            raise TypeError("observation_pipeline must be ObservationPipeline")
        if not isinstance(iod, IodRunConfig):
            raise TypeError("iod must be IodRunConfig")
        if not isinstance(refinement, RefinementRunConfig):
            raise TypeError("refinement must be RefinementRunConfig")
        if not isinstance(evaluation, EvaluationConfig):
            raise TypeError("evaluation must be EvaluationConfig")
        if not isinstance(resources, ResourceBindings):
            raise TypeError("resources must be ResourceBindings")
        descriptors = observation_pipeline._descriptors()
        transforms_json = (
            None
            if descriptors is None
            else json.dumps(descriptors, sort_keys=True, separators=(",", ":"))
        )
        native = _rust.explicit_od_config(
            transforms_json,
            resources._native_paths(),
            iod._native,
            refinement._native,
            evaluation._native,
        )
        return cls(native, observation_pipeline)

    def for_iod(self) -> IodRunConfig:
        return IodRunConfig(self._native.for_iod())

    def for_refinement(self) -> RefinementRunConfig:
        return RefinementRunConfig(self._native.for_refinement())

    def for_evaluation(self) -> EvaluationConfig:
        return EvaluationConfig(self._native.for_evaluation())

    @property
    def provenance(self) -> Mapping[str, object]:
        value = cast(dict[str, object], json.loads(self._native.provenance_json))
        return MappingProxyType(value)

    @property
    def profile_name(self) -> str | None:
        return cast(str | None, self._native.profile_name)


__all__ = [
    "EvaluationConfig",
    "IodRunConfig",
    "OdConfig",
    "RefinementRunConfig",
    "ResourceBindings",
]
