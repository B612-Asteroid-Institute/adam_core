"""Canonical explicit-configuration one-group OD composition.

The batch-native grouping and typed-failure layer is intentionally implemented
by the separate Core 0.7 batch-workflow task.  This module establishes the
one-group stage contract used by that layer: prepare once, IOD, then refine the
same prepared observations.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

from .. import _rust_native
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from ._pipeline import (
    apply_pipeline_once,
    attach_configuration_provenance,
    attach_original_and_used,
    provenance_with_fields,
)
from .config import IodRunConfig, OdConfig, RefinementRunConfig
from .evaluate import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers, FittedOrbits
from .iod import _run_iod_prepared

__all__ = ["refine", "run_od"]


def _stamp_object_id(fitted: FittedOrbits, object_id: str | None) -> FittedOrbits:
    if len(fitted) == 0:
        return fitted
    return fitted.set_column(
        "object_id",
        pa.array([object_id] * len(fitted), type=pa.large_string()),
    )


def _refine_prepared(
    orbit: Orbits | FittedOrbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    stage_config: RefinementRunConfig,
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    """Refine against observations whose configured pipeline already ran."""

    from ._native_dispatch import native_propagator
    from .algorithms.fitting import (
        _emit_native_warnings,
        _fitted_tables_from_native_output,
        _native_problem,
    )

    if not isinstance(stage_config, RefinementRunConfig):
        raise TypeError("stage_config must be a RefinementRunConfig")
    seed = orbit.to_orbits() if isinstance(orbit, FittedOrbits) else orbit
    output = _rust_native.refine_config_ipc(
        native_propagator(propagator),
        *_native_problem(seed, observations),
        stage_config._native,
    )
    _emit_native_warnings(output)
    refined, members = _fitted_tables_from_native_output(
        seed,
        observations,
        output,
        np.asarray(output["weights"], dtype=np.float64),
    )
    if len(refined):
        diagnostics = json.dumps(
            {
                "passes": int(output["passes"]),
                "n_rejected": int(output["n_rejected"]),
                "n_recovered": int(output["n_recovered"]),
                "flags": sorted(output["flags"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        provider = json.dumps(
            output["refinement_provider"],
            sort_keys=True,
            separators=(",", ":"),
        )
        refined = refined.set_column(
            "provenance",
            provenance_with_fields(
                refined.provenance,
                refinement_provider=[provider] * len(refined),
                refinement_diagnostics=[diagnostics] * len(refined),
            ),
        )
    if isinstance(orbit, FittedOrbits) and len(orbit) == 1 and len(refined):
        updates = {}
        for name in (
            "iod_provider",
            "iod_diagnostics",
            "iod_ambiguous",
            "iod_alternatives",
        ):
            value = orbit.provenance.table.column(name)[0].as_py()
            if value is not None:
                updates[name] = [value] * len(refined)
        if updates:
            refined = refined.set_column(
                "provenance", provenance_with_fields(refined.provenance, **updates)
            )
    return refined, members


def refine(
    orbit: Orbits | FittedOrbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    *,
    config: OdConfig,
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    """Refine one orbit group with an explicit complete configuration."""

    if not isinstance(config, OdConfig):
        raise TypeError("config must be an OdConfig")
    stage_config = config.for_refinement()
    used, _pipeline_output = apply_pipeline_once(observations, config)
    refined, members = _refine_prepared(orbit, used, propagator, stage_config)
    refined = attach_configuration_provenance(refined, config)
    return refined, attach_original_and_used(members, observations, used)


def run_od(
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    *,
    config: OdConfig,
    object_id: str | None = None,
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    """Run exact ``run_iod + refine`` composition with one pipeline execution.

    Caller astrometry is retained for member provenance.  The configured Rust
    pipeline runs exactly once, and both numerical stages consume that same
    prepared table; refinement never reapplies transforms.
    """

    if not isinstance(config, OdConfig):
        raise TypeError("config must be an OdConfig")
    iod_config: IodRunConfig = config.for_iod()
    refinement_config: RefinementRunConfig = config.for_refinement()
    original = observations
    used, _pipeline_output = apply_pipeline_once(original, config)
    iod_orbits, iod_members, _report = _run_iod_prepared(used, propagator, iod_config)
    if len(iod_orbits) == 0:
        return FittedOrbits.empty(), attach_original_and_used(
            iod_members, original, used
        )
    refined, members = _refine_prepared(iod_orbits, used, propagator, refinement_config)
    refined = _stamp_object_id(refined, object_id)
    refined = attach_configuration_provenance(refined, config)
    return refined, attach_original_and_used(members, original, used)
