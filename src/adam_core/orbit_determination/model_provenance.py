"""Explicit provenance contract for custom observation models.

adam-core's own observation models are identified by the exact native build
commit. Third-party and user-defined models cannot be inferred reliably from
Python runtime state, so they provide this small immutable contract instead.
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
        Installed distribution name that owns the implementation class.
    version
        Exact immutable installed distribution version for this implementation.
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
    instance_fields: tuple[str, ...] = ()
    collaborators: tuple[tuple[str, "_BehaviorSpec"], ...] = ()


def _capture_behavior_spec(
    model_type: type[Any],
    member_names: tuple[str, ...],
    *,
    instance_fields: tuple[str, ...] = (),
    collaborators: tuple[tuple[str, _BehaviorSpec], ...] = (),
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
    return _BehaviorSpec(model_type, tuple(members), instance_fields, collaborators)


@runtime_checkable
class ProvenancedModel(Protocol):
    """Structural provenance hook implemented by custom observation models.

    This protocol describes only the provenance facet. A model accepted by
    ``run_od`` must also subclass ``ObservationUncertaintyModel`` and be a
    source-declared class owned by the claimed installed distribution.
    """

    def model_provenance(self) -> ModelProvenance:
        """Return this model instance's explicit immutable provenance."""
