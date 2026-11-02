"""Private Python marshalling for the single authoritative Rust pipeline stage."""

from __future__ import annotations

from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]

from adam_core import _rust_native as _rust  # type: ignore[attr-defined]
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.orbits.arrow_bridge import coordinates_from_ipc, coordinates_to_ipc

from .config import OdConfig
from .evaluate import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers, ObservationAstrometry


def apply_pipeline_once(
    observations: OrbitDeterminationObservations, config: OdConfig
) -> tuple[OrbitDeterminationObservations, dict[str, Any]]:
    """Marshal one group to Rust and apply its configured pipeline exactly once."""

    if not isinstance(config, OdConfig):
        raise TypeError("config must be an OdConfig")
    observed_ipc = coordinates_to_ipc(observations.coordinates, "spherical")
    output = _rust.apply_observation_pipeline_ipc(
        observed_ipc,
        observations.observers.code.to_pylist(),
        observations.photometry.band.to_pylist(),
        observations.astcat.to_pylist(),
        config._native,
    )
    coordinates = coordinates_from_ipc(
        output["used_coordinates_ipc"], SphericalCoordinates
    )
    return observations.set_column("coordinates", coordinates), output


def attach_original_and_used(
    members: FittedOrbitMembers,
    original: OrbitDeterminationObservations,
    used: OrbitDeterminationObservations,
) -> FittedOrbitMembers:
    """Attach caller and prepared astrometry without executing Python science."""

    if len(members) == 0:
        return members
    member_ids = (
        members.obs_id.combine_chunks()
        if isinstance(members.obs_id, pa.ChunkedArray)
        else members.obs_id
    )
    original_ids = (
        original.id.combine_chunks()
        if isinstance(original.id, pa.ChunkedArray)
        else original.id
    )
    used_ids = (
        used.id.combine_chunks() if isinstance(used.id, pa.ChunkedArray) else used.id
    )
    original_indices = pc.index_in(member_ids, value_set=original_ids)
    used_indices = pc.index_in(member_ids, value_set=used_ids)
    if original_indices.null_count or used_indices.null_count:
        raise ValueError("fitted members reference unknown observation IDs")
    original_rows = original.take(original_indices)
    used_rows = used.take(used_indices)
    return (
        members.set_column(
            "original_astrometry",
            ObservationAstrometry.from_spherical(original_rows.coordinates),
        )
        .set_column(
            "used_astrometry",
            ObservationAstrometry.from_spherical(used_rows.coordinates),
        )
        .set_column("astcat", original_rows.astcat)
    )
