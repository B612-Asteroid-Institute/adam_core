"""
Compact orbit-determination provenance (`OdProvenance` on `FittedOrbits`).

The IOD stage records whether its solution was ambiguous; `run_od` completes
the record on either route: adam-core version, route, fitter, IOD and
refinement backends, settings, the observation models (with a content hash
of each table they carry, not the tables) and the outliers the final fit
rejected.
"""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from typing import Any, Optional, Sequence

import numpy as np
import pyarrow as pa
import quivr as qv

from ._native_dispatch import native_backend
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, OdProvenance


def _package_version(module: str) -> Optional[str]:
    top = module.split(".")[0]
    for name in (top, top.replace("_", "-")):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def backend_identity(propagator: object) -> Optional[str]:
    """``"name version"`` of a propagator's `NativeBackend`, else
    ``"python:module.qualname[ version]"``; None for None. A class is
    identified as an instance of it would be."""
    if propagator is None:
        return None
    instance = (
        object.__new__(propagator) if isinstance(propagator, type) else propagator
    )
    backend = native_backend(instance)
    if backend is not None:
        return f"{backend.name} {backend.version}"
    cls = type(instance)
    version = _package_version(cls.__module__)
    identity = f"python:{cls.__module__}.{cls.__qualname__}"
    return identity if version is None else f"{identity} {version}"


def _table_sha256(table: pa.Table | qv.Table) -> str:
    if isinstance(table, qv.Table):
        table = table.table
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def model_descriptor(model: object) -> dict[str, Any]:
    """Type, scalar parameters, table hashes and nested models of an
    observation model."""
    cls = type(model)
    descriptor: dict[str, Any] = {"type": f"{cls.__module__}.{cls.__qualname__}"}
    parameters: dict[str, Any] = {}
    tables: dict[str, str] = {}
    nested: list[dict[str, Any]] = []
    for name, value in sorted(vars(model).items()):
        if isinstance(value, (pa.Table, qv.Table)):
            tables[name.lstrip("_")] = _table_sha256(value)
        elif value is None or isinstance(value, (bool, int, float, str)):
            parameters[name.lstrip("_")] = value
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(hasattr(item, "apply") for item in value)
        ):
            nested.extend(model_descriptor(item) for item in value)
    if parameters:
        descriptor["parameters"] = parameters
    if tables:
        descriptor["tables_sha256"] = tables
    if nested:
        descriptor["models"] = nested
    return descriptor


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def with_iod_provenance(
    fitted_orbits: FittedOrbits, ambiguous_with: Optional[Sequence[int]]
) -> FittedOrbits:
    """Record the IOD's ambiguity (``ambiguous_with`` of its report) on every
    row of ``fitted_orbits``."""
    n = len(fitted_orbits)
    if n == 0 or ambiguous_with is None:
        return fitted_orbits
    alternatives = len(ambiguous_with)
    return fitted_orbits.set_column(
        "provenance",
        OdProvenance.from_kwargs(
            iod_ambiguous=[alternatives > 0] * n,
            iod_alternatives=[alternatives] * n,
        ),
    )


def carry_iod_provenance(
    iod_orbits: FittedOrbits, refined: FittedOrbits
) -> FittedOrbits:
    """Copy the IOD's ambiguity onto the refinement of its (single) orbit
    when the refinement recorded none."""
    if len(iod_orbits) != 1 or len(refined) == 0:
        return refined
    # The refinement's own IOD record wins; an empty one is filled.
    if refined.provenance.iod_ambiguous.null_count != len(refined):
        return refined
    source = iod_orbits.provenance
    if source.iod_ambiguous.null_count == 1:
        return refined
    return refined.set_column(
        "provenance",
        OdProvenance.from_kwargs(
            iod_ambiguous=[source.iod_ambiguous[0].as_py()] * len(refined),
            iod_alternatives=[source.iod_alternatives[0].as_py()] * len(refined),
        ),
    )


def complete_provenance(
    fitted_orbits: FittedOrbits,
    members: FittedOrbitMembers,
    *,
    fitter: object,
    propagator: object,
    models: Sequence[object],
    route: str,
) -> FittedOrbits:
    """The `run_od` record on every orbit, keeping the IOD fields already
    present."""
    n = len(fitted_orbits)
    if n == 0:
        return fitted_orbits
    from .. import __version__
    from .native_orbit_fitter import NativeOrbitFitter

    fitter_cls = type(fitter)
    settings: Optional[str] = None
    iod_backend: Optional[str] = None
    if isinstance(fitter, NativeOrbitFitter):
        try:
            iod_settings, refinement, fit_settings = fitter.native_settings()
            settings = _json(
                {"iod": iod_settings, "refinement": refinement, "fit": fit_settings}
            )
        except Exception:  # noqa: BLE001 - provenance must not fail the fit
            settings = None
        iod_backend = backend_identity(
            propagator
            if fitter.fused_propagator_matches(propagator)
            else fitter.propagator_class
        )

    existing = fitted_orbits.table.column("provenance")
    has_existing = existing.null_count < n
    previous = fitted_orbits.provenance

    orbit_ids = fitted_orbits.orbit_id.to_pylist()
    member_ids = np.asarray(members.orbit_id.to_pylist(), dtype=object)
    outlier = members.outlier.to_numpy(zero_copy_only=False)
    num_rejected: list[Optional[int]] = []
    rejected_fraction: list[Optional[float]] = []
    for orbit_id in orbit_ids:
        rows = member_ids == orbit_id
        total = int(rows.sum())
        if total == 0:
            num_rejected.append(None)
            rejected_fraction.append(None)
            continue
        rejected = int(np.asarray(outlier[rows], dtype=bool).sum())
        num_rejected.append(rejected)
        rejected_fraction.append(rejected / total)

    record = OdProvenance.from_kwargs(
        adam_core_version=[__version__] * n,
        route=[route] * n,
        fitter=[f"{fitter_cls.__module__}.{fitter_cls.__qualname__}"] * n,
        iod_backend=[iod_backend] * n,
        refinement_backend=[backend_identity(propagator)] * n,
        settings=[settings] * n,
        models=[_json([model_descriptor(model) for model in models])] * n,
        iod_ambiguous=(
            previous.iod_ambiguous if has_existing else pa.nulls(n, pa.bool_())
        ),
        iod_alternatives=(
            previous.iod_alternatives if has_existing else pa.nulls(n, pa.int64())
        ),
        num_rejected=num_rejected,
        rejected_fraction=rejected_fraction,
    )
    return fitted_orbits.set_column("provenance", record)
