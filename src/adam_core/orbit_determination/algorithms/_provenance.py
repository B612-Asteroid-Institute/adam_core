"""Fail-closed runtime provenance for focused one-group OD algorithms."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import quivr as qv

from adam_core import _rust_native  # type: ignore[attr-defined]

from ..config import OdConfig
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits, OdProvenance

_CLEAN_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def core_build_identity() -> tuple[str, str]:
    version, build = _rust_native.build_identity()
    if not isinstance(version, str) or not version:
        raise RuntimeError("native adam-core build has no version identity")
    if not isinstance(build, str) or _CLEAN_COMMIT.fullmatch(build) is None:
        raise RuntimeError(
            "orbit determination requires an exact clean native adam-core build identity"
        )
    return version, build


def _table_bytes(table: pa.Table | qv.Table) -> bytes:
    arrow = table.table if isinstance(table, qv.Table) else table
    arrow = arrow.combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, arrow.schema) as writer:
        writer.write_table(arrow)
    return bytes(sink.getvalue().to_pybytes())


def input_fingerprint(
    observations: OrbitDeterminationObservations, members: FittedOrbitMembers
) -> str:
    used = members.table.select(["obs_id", "used_astrometry"])
    digest = hashlib.sha256()
    for label, payload in (
        (b"ordered-observations-v1", _table_bytes(observations)),
        (b"ordered-used-astrometry-v1", _table_bytes(used)),
    ):
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _backend(provider: str | None) -> str | None:
    if provider is None:
        return None
    value = json.loads(provider)
    name = value.get("name") or value.get("backend_name") or value.get("provider")
    version = value.get("version") or value.get("backend_version")
    if name is None:
        return "native-provider"
    return str(name) if version is None else f"{name} {version}"


def _replace(current: OdProvenance, **updates: Any) -> OdProvenance:
    fields = {name: current.table.column(name) for name in OdProvenance.schema.names}
    fields.update(updates)
    return OdProvenance.from_kwargs(**fields)


def stamp_complete_provenance(
    fitted: FittedOrbits,
    members: FittedOrbitMembers,
    original: OrbitDeterminationObservations,
    config: OdConfig,
    *,
    route: str,
) -> FittedOrbits:
    """Populate every applicable runtime identity and rejection field."""

    if len(fitted) == 0:
        return fitted
    version, build = core_build_identity()
    if len(members) == 0:
        raise ValueError("a fitted orbit must have member rows for provenance")
    rejected = sum(bool(value) for value in members.outlier.to_pylist())
    fraction = rejected / len(members)
    iod_provider = fitted.provenance.iod_provider[0].as_py()
    refinement_provider = fitted.provenance.refinement_provider[0].as_py()
    provenance = _replace(
        fitted.provenance,
        adam_core_version=[version] * len(fitted),
        adam_core_build=[build] * len(fitted),
        input_fingerprint=[input_fingerprint(original, members)] * len(fitted),
        route=[route] * len(fitted),
        iod_backend=[_backend(iod_provider)] * len(fitted),
        refinement_backend=[_backend(refinement_provider)] * len(fitted),
        configuration=[config._native.provenance_json] * len(fitted),
        num_rejected=[rejected] * len(fitted),
        rejected_fraction=[fraction] * len(fitted),
    )
    return fitted.set_column("provenance", provenance)


__all__ = ["core_build_identity", "stamp_complete_provenance"]
