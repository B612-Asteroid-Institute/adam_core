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
        Stable distribution or project name that owns the implementation.
    version
        Exact package/project version for this implementation.
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


@runtime_checkable
class ProvenancedModel(Protocol):
    """Structural hook required for custom observation models used by run_od."""

    def model_provenance(self) -> ModelProvenance:
        """Return this model instance's explicit immutable provenance."""
