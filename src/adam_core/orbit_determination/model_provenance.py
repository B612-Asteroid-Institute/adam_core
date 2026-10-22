"""Explicit provenance contract for custom observation models.

adam-core's own observation models are identified by the exact native build
commit and behavior graph. This value can describe a provider declaration, but
it cannot authenticate live executable Python behavior and does not make a
custom model acceptable to ``run_od``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Immutable identity and configuration declared by a custom model.

    Parameters
    ----------
    package
        Distribution/project name asserted by the provider.
    version
        Version asserted by the provider.
    implementation_fingerprint
        Lowercase 64-hex SHA-256 (or equivalent content fingerprint) chosen by
        the provider. A science-affecting implementation change must use a new
        fingerprint.
    configuration
        Complete model configuration. adam-core canonicalizes supported values
        deterministically and rejects unsupported values before model work.
    """

    package: str
    version: str
    implementation_fingerprint: str
    configuration: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _BehaviorSpec:
    """Exact shipped class behavior used by built-in provenance checks."""

    model_type: type[Any]
    class_members: tuple[tuple[str, object], ...]
    exact_mro: tuple[type[Any], ...]
    class_surfaces: tuple[tuple[type[Any], tuple[tuple[str, object], ...]], ...]
    instance_fields: tuple[str, ...] = ()
    collaborators: tuple[tuple[str, "_BehaviorSpec"], ...] = ()
    module_delegates: tuple[tuple[Mapping[str, object], str, object], ...] = ()


def _capture_behavior_spec(
    model_type: type[Any],
    member_names: tuple[str, ...],
    *,
    instance_fields: tuple[str, ...] = (),
    collaborators: tuple[tuple[str, _BehaviorSpec], ...] = (),
    module_delegates: tuple[tuple[Mapping[str, object], str, object], ...] = (),
) -> _BehaviorSpec:
    """Capture explicitly named shipped members before callers can patch them."""
    members: list[tuple[str, object]] = []
    for name in member_names:
        for owner in model_type.__mro__:
            if name in vars(owner):
                members.append((name, vars(owner)[name]))
                break
        else:  # pragma: no cover - import-time registry programming error
            raise RuntimeError(
                f"registered behavior member {model_type.__qualname__}.{name} is missing"
            )
    surface_owners = tuple(
        owner
        for owner in model_type.__mro__
        if owner.__module__.startswith("adam_core.orbit_determination")
    )
    surfaces = tuple((owner, tuple(vars(owner).items())) for owner in surface_owners)
    return _BehaviorSpec(
        model_type,
        tuple(members),
        model_type.__mro__,
        surfaces,
        instance_fields,
        collaborators,
        module_delegates,
    )


@runtime_checkable
class ProvenancedModel(Protocol):
    """Structural provenance hook implemented by custom observation models.

    This protocol describes only a provider declaration. It does not
    authenticate executable Python behavior; ``run_od`` accepts only exact
    shipped Core models with declarative Core-Rust execution.
    """

    def model_provenance(self) -> ModelProvenance:
        """Return this model instance's explicit immutable provenance."""
