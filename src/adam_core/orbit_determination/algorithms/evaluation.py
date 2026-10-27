"""Focused one-group orbit evaluation pending the canonical batch surface."""

from __future__ import annotations

from adam_core.orbits import Orbits
from adam_core.propagator import Propagator

from .._native_dispatch import native_propagator
from .._pipeline import apply_pipeline_once, attach_original_and_used
from ..config import OdConfig
from ..evaluate import OrbitDeterminationObservations, _evaluate_orbits_prepared
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ._provenance import core_build_identity, stamp_complete_provenance
from ._validation import validate_observations, validate_seed_orbits


def evaluate_orbits_one_group(
    orbits: Orbits | FittedOrbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    *,
    config: OdConfig,
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    """Expert one-group evaluation used internally until batch workflows land."""

    if not isinstance(config, OdConfig):
        raise TypeError("config must be an OdConfig")
    empty_observations = validate_observations(observations)
    seed = orbits.to_orbits() if isinstance(orbits, FittedOrbits) else orbits
    empty_seed = validate_seed_orbits(seed)
    core_build_identity()
    if empty_observations or empty_seed:
        return FittedOrbits.empty(), FittedOrbitMembers.empty()
    # Fail before pipeline execution when the provider has no native capsule.
    native_propagator(propagator)
    evaluation = config.for_evaluation()
    used, _pipeline_output = apply_pipeline_once(observations, config)
    fitted, members = _evaluate_orbits_prepared(
        orbits,
        used,
        propagator,
        parameters=evaluation.covariance_dimension,
        ignore=None,
    )
    members = attach_original_and_used(members, observations, used)
    fitted = stamp_complete_provenance(
        fitted,
        members,
        observations,
        config,
        route="algorithms.evaluate_orbits_one_group",
    )
    return fitted, members


__all__ = ["evaluate_orbits_one_group"]
