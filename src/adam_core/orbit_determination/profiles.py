"""Immutable path-free native orbit-determination profiles.

Profile values are decoded only from Rust's canonical definition. Profile
objects cannot be constructed or replaced by callers, and binding returns a
configuration whose Python view is decoded back from the bound native state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, cast

from adam_core import _rust_native as _rust  # type: ignore[attr-defined]

from .config import (
    EvaluationConfig,
    IodRunConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
)
from .observation_pipeline import ObservationPipeline


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


_PROFILE_TOKEN = object()


class OdProfile:
    """Read-only view of one authoritative Rust profile definition."""

    __slots__ = ("_payload", "_requirements")

    def __init__(
        self,
        payload: Mapping[str, Any],
        requirements: tuple[ResourceRequirement, ...],
        *,
        _token: object,
    ) -> None:
        if _token is not _PROFILE_TOKEN:
            raise TypeError("OdProfile values are created only by Core Rust")
        self._payload = MappingProxyType(dict(payload))
        self._requirements = requirements

    @property
    def name(self) -> str:
        return cast(str, self._payload["identity"]["name"])

    @property
    def version(self) -> int:
        return cast(int, self._payload["identity"]["version"])

    @property
    def definition_sha256(self) -> str:
        return cast(str, self._payload["identity"]["definition_sha256"])

    @property
    def requirements(self) -> tuple[ResourceRequirement, ...]:
        return self._requirements

    @property
    def observation_pipeline(self) -> ObservationPipeline:
        definitions = cast(
            list[dict[str, object]], self._payload["definition"]["pipeline"]
        )
        return ObservationPipeline._from_descriptors(definitions)

    @property
    def iod(self) -> IodRunConfig:
        value = self._payload["definition"]["iod"]
        native = _rust.iod_run_config_from_json(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
        return IodRunConfig._from_native(native)

    @property
    def refinement(self) -> RefinementRunConfig:
        value = self._payload["definition"]["refinement"]
        native = _rust.refinement_run_config_from_json(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
        return RefinementRunConfig._from_native(native)

    @property
    def evaluation(self) -> EvaluationConfig:
        value = self._payload["definition"]["evaluation"]
        return EvaluationConfig(cast(int, value["covariance_dimension"]))

    @property
    def definition(self) -> Mapping[str, object]:
        return MappingProxyType(dict(self._payload["definition"]))

    def bind(self, resources: ResourceBindings) -> OdConfig:
        """Validate every binding and expand through the native definition."""

        if not isinstance(resources, ResourceBindings):
            raise TypeError("resources must be ResourceBindings")
        native = _rust.bind_od_profile(self.name, resources._native_paths())
        return OdConfig(native)

    def __repr__(self) -> str:
        return f"OdProfile(name={self.name!r}, version={self.version})"


def _requirements(name: str) -> tuple[ResourceRequirement, ...]:
    raw = json.loads(_rust.od_profile_requirements(name))
    return tuple(ResourceRequirement(**requirement) for requirement in raw)


def _profile(name: str) -> OdProfile:
    payload = json.loads(_rust.od_profile_definition(name))
    return OdProfile(payload, _requirements(name), _token=_PROFILE_TOKEN)


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
