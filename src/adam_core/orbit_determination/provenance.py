"""
Compact orbit-determination provenance (`OdProvenance` on `FittedOrbits`).

The IOD stage records whether its solution was ambiguous; `run_od` completes
the record on either route: adam-core version, route, fitter, IOD and
refinement providers (identity, ABI, capabilities, time scale and immutable
configuration/data hashes), source build, exact ordered fitting-input hash,
settings, complete observation-model configuration/content hashes, native IOD
and CMC diagnostics, and the outliers the final fit rejected.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import quivr as qv

from ._native_dispatch import native_backend
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, OdProvenance


def _package_version(module: str) -> Optional[str]:
    top = module.split(".")[0]
    if top == "adam_core":
        from .. import __version__

        return __version__
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


def _table_bytes(table: pa.Table | qv.Table) -> bytes:
    if isinstance(table, qv.Table):
        table = table.table
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _table_sha256(table: pa.Table | qv.Table) -> str:
    return hashlib.sha256(_table_bytes(table)).hexdigest()


def _fit_input_sha256(observations: qv.Table, members: FittedOrbitMembers) -> str:
    """Hash the ordered caller table and the exact astrometry consumed.

    ``used_astrometry`` is emitted by the same native run that fit the orbit,
    so this also covers native observation-model output without re-running a
    Python provider after the fit.
    """
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


def _numpy_sha256(value: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Canonical NumPy identity independent of byte order/view layout."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object-dtype NumPy provenance is not supported")
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    dtype_identity: Any = dtype.descr if dtype.names is not None else dtype.str
    return {
        "dtype": dtype_identity,
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _arrow_array_sha256(value: pa.Array | pa.ChunkedArray) -> str:
    return _table_sha256(pa.table({"value": value}))


def _canonical_parameter(value: Any, seen: set[int]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    identity = id(value)
    if identity in seen:
        raise ValueError("cyclic provenance configuration is not supported")
    if isinstance(value, np.ndarray):
        return {"numpy": _numpy_sha256(value)}
    if isinstance(value, (pa.Table, qv.Table)):
        return {"arrow_sha256": _table_sha256(value)}
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return {"arrow_sha256": _arrow_array_sha256(value)}
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            encoded = [
                (
                    _canonical_parameter(key, seen),
                    _canonical_parameter(item, seen),
                )
                for key, item in value.items()
            ]
            encoded.sort(key=lambda pair: _json(pair[0]))
            if all(isinstance(key, str) for key, _ in encoded):
                return {key: item for key, item in encoded}
            return {"mapping": encoded}
        if isinstance(value, Sequence):
            return [_canonical_parameter(item, seen) for item in value]
        if isinstance(value, (set, frozenset)):
            encoded = [_canonical_parameter(item, seen) for item in value]
            return sorted(encoded, key=lambda item: _json(item))
    finally:
        seen.remove(identity)
    raise TypeError(
        f"unsupported provenance field {type(value).__module__}.{type(value).__qualname__}"
    )


def model_descriptor(model: object) -> dict[str, Any]:
    """Reproducible model implementation, configuration, and content."""
    cls = type(model)
    descriptor: dict[str, Any] = {
        "type": f"{cls.__module__}.{cls.__qualname__}",
        "implementation_version": _package_version(cls.__module__),
    }
    parameters: dict[str, Any] = {}
    tables: dict[str, str] = {}
    nested: list[dict[str, Any]] = []
    for name, value in sorted(vars(model).items()):
        public_name = name.lstrip("_")
        if isinstance(value, (pa.Table, qv.Table)):
            tables[public_name] = _table_sha256(value)
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(hasattr(item, "apply") for item in value)
        ):
            nested.extend(model_descriptor(item) for item in value)
        else:
            parameters[public_name] = _canonical_parameter(value, set())
    if parameters:
        descriptor["parameters"] = parameters
    if tables:
        descriptor["tables_sha256"] = tables
    if nested:
        descriptor["models"] = nested
    descriptor["descriptor_sha256"] = hashlib.sha256(
        _json(descriptor).encode("utf-8")
    ).hexdigest()
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
    observations: Optional[qv.Table] = None,
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
    refinement_backend: Optional[str] = backend_identity(propagator)
    iod_provider: Optional[str] = None
    refinement_provider: Optional[str] = None
    if isinstance(fitter, NativeOrbitFitter):
        iod_settings, refinement, fit_settings = fitter.native_settings()
        settings = _json(
            {"iod": iod_settings, "refinement": refinement, "fit": fit_settings}
        )
        # These values were decoded from the exact retained capsules before
        # the native driver released the GIL. Do not call the Python provider
        # again here: it could return a different capsule (TOCTOU).
        native_record = fitted_orbits.provenance
        iod_values = native_record.iod_provider.to_pylist()
        refinement_values = native_record.refinement_provider.to_pylist()
        if (
            any(value is None for value in iod_values)
            or any(value is None for value in refinement_values)
            or len(set(iod_values)) != 1
            or len(set(refinement_values)) != 1
        ):
            raise RuntimeError(
                "native OD output omitted or disagreed on exact provider metadata"
            )
        iod_provider = iod_values[0]
        refinement_provider = refinement_values[0]
        iod_metadata = json.loads(iod_provider)
        refinement_metadata = json.loads(refinement_provider)
        iod_backend = f"{iod_metadata['name']} {iod_metadata['version']}"
        refinement_backend = (
            f"{refinement_metadata['name']} {refinement_metadata['version']}"
        )

    try:
        from adam_core import _rust_native

        build_version, build_commit = _rust_native.build_identity()
    except (ImportError, AttributeError):
        build_version, build_commit = __version__, "unknown"
    if build_version != __version__:
        raise RuntimeError(
            f"native adam-core version {build_version!r} does not match Python {__version__!r}"
        )
    input_fingerprint = (
        _fit_input_sha256(observations, members) if observations is not None else None
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
        adam_core_build=[build_commit] * n,
        input_fingerprint=[input_fingerprint] * n,
        route=[route] * n,
        fitter=[f"{fitter_cls.__module__}.{fitter_cls.__qualname__}"] * n,
        iod_backend=[iod_backend] * n,
        refinement_backend=[refinement_backend] * n,
        iod_provider=[iod_provider] * n,
        refinement_provider=[refinement_provider] * n,
        iod_diagnostics=(
            previous.iod_diagnostics if has_existing else pa.nulls(n, pa.large_string())
        ),
        refinement_diagnostics=(
            previous.refinement_diagnostics
            if has_existing
            else pa.nulls(n, pa.large_string())
        ),
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
