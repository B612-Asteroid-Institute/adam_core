"""Invocation-fatal validation shared by focused one-group OD algorithms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from adam_core.orbits import Orbits

from ..evaluate import OrbitDeterminationObservations


def three_position_vectors(
    r1: npt.ArrayLike, r2: npt.ArrayLike, r3: npt.ArrayLike
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Normalize and validate the three position vectors used by IOD kernels."""

    r1_array = np.ascontiguousarray(np.asarray(r1, dtype=np.float64))
    r2_array = np.ascontiguousarray(np.asarray(r2, dtype=np.float64))
    r3_array = np.ascontiguousarray(np.asarray(r3, dtype=np.float64))
    vectors = (r1_array, r2_array, r3_array)
    if any(vector.shape != (3,) for vector in vectors):
        raise ValueError("r1, r2, and r3 must each have shape (3,)")
    return r1_array, r2_array, r3_array


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
