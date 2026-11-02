"""Invocation-fatal validation shared by focused one-group OD algorithms."""

from __future__ import annotations

from collections.abc import Sequence

from adam_core.orbits import Orbits

from ..evaluate import OrbitDeterminationObservations


def _require_unique_non_null(values: Sequence[str | None], label: str) -> None:
    if any(value is None for value in values):
        raise ValueError(f"{label} must be non-null")
    non_null = [value for value in values if value is not None]
    if len(set(non_null)) != len(non_null):
        raise ValueError(f"{label} must be unique")


def validate_observations(observations: OrbitDeterminationObservations) -> bool:
    """Validate static identity semantics; return whether the group is empty."""

    if not isinstance(observations, OrbitDeterminationObservations):
        raise TypeError("observations must be OrbitDeterminationObservations")
    if len(observations) == 0:
        return True
    _require_unique_non_null(observations.id.to_pylist(), "observation IDs")
    if len(observations.coordinates) != len(observations.observers):
        raise ValueError("observation coordinates and observers must have equal length")
    return False


def validate_seed_orbits(orbits: Orbits, *, exactly_one: bool = False) -> bool:
    """Validate seed identity semantics; return whether the seed table is empty."""

    if not isinstance(orbits, Orbits):
        raise TypeError("orbit must be Orbits or FittedOrbits")
    if len(orbits) == 0:
        return True
    _require_unique_non_null(orbits.orbit_id.to_pylist(), "orbit IDs")
    if exactly_one and len(orbits) != 1:
        raise ValueError("focused refinement requires exactly one seed orbit")
    return False
