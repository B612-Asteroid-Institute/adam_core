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
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from types import CodeType
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pyarrow as pa
import quivr as qv

from ._native_dispatch import native_backend
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, OdProvenance

if TYPE_CHECKING:
    from ..propagator import Propagator


def _package_version(module: str) -> Optional[str]:
    top = module.split(".")[0]
    if top == "adam_core":
        from .. import __version__

        return str(__version__)
    for name in (top, top.replace("_", "-")):
        try:
            return str(metadata.version(name))
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
    # IPC stream bytes otherwise retain arbitrary caller chunk boundaries.
    # One canonical chunk per typed column preserves order/schema/content but
    # makes logically identical Arrow layouts hash identically.
    table = table.combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue().to_pybytes())


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


def _code_identity(code: CodeType) -> dict[str, Any]:
    constants: list[Any] = []
    for value in code.co_consts:
        if isinstance(value, CodeType):
            constants.append({"code": _code_identity(value)})
        elif value is None or isinstance(value, (bool, int, float, str)):
            constants.append(value)
        elif isinstance(value, bytes):
            constants.append({"bytes_sha256": hashlib.sha256(value).hexdigest()})
        else:
            constants.append(_canonical_parameter(value, set()))
    return {
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "constants": constants,
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
    }


def _callable_identity(value: Any, seen: set[int]) -> dict[str, Any]:
    function = value.__func__ if inspect.ismethod(value) else value
    code = getattr(function, "__code__", None)
    if not isinstance(code, CodeType):
        raise TypeError(
            f"unsupported callable provenance {type(value).__module__}.{type(value).__qualname__}"
        )
    closure = []
    for cell in function.__closure__ or ():
        closure.append(_canonical_parameter(cell.cell_contents, seen))
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "implementation_version": _package_version(function.__module__),
        "code": _code_identity(code),
        "defaults": _canonical_parameter(function.__defaults__, seen),
        "kwdefaults": _canonical_parameter(function.__kwdefaults__, seen),
        "closure": closure,
        "bound_to_model": bool(
            inspect.ismethod(value) and id(getattr(value, "__self__", None)) in seen
        ),
    }


def _canonical_parameter(value: Any, seen: set[int]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        if not value.is_file():
            raise ValueError(f"provenance path is not a readable file: {value}")
        return {
            "path": str(value),
            "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
        }
    if isinstance(value, Enum):
        cls = type(value)
        return {
            "enum": f"{cls.__module__}.{cls.__qualname__}",
            "name": value.name,
            "implementation_version": _package_version(cls.__module__),
        }
    identity = id(value)
    if identity in seen:
        raise ValueError("cyclic provenance configuration is not supported")
    if isinstance(value, np.ndarray):
        return {"numpy": _numpy_sha256(value)}
    if isinstance(value, (pa.Table, qv.Table)):
        return {"arrow_sha256": _table_sha256(value)}
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return {"arrow_sha256": _arrow_array_sha256(value)}
    if inspect.isfunction(value) or inspect.ismethod(value):
        return {"callable": _callable_identity(value, seen)}
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
        fields = getattr(value, "__dict__", None)
        if isinstance(fields, dict):
            cls = type(value)
            return {
                "object": f"{cls.__module__}.{cls.__qualname__}",
                "implementation_version": _package_version(cls.__module__),
                "fields": {
                    name: _canonical_parameter(item, seen)
                    for name, item in sorted(fields.items())
                },
            }
    finally:
        seen.remove(identity)
    raise TypeError(
        f"unsupported provenance field {type(value).__module__}.{type(value).__qualname__}"
    )


def model_descriptor(model: object) -> dict[str, Any]:
    """Reproducible model implementation, configuration, and content."""
    cls = type(model)
    apply = getattr(model, "apply", None)
    implementation = apply if callable(apply) else cls.__init__
    descriptor: dict[str, Any] = {
        "type": f"{cls.__module__}.{cls.__qualname__}",
        "implementation_version": _package_version(cls.__module__),
        "implementation_entrypoint": "apply" if callable(apply) else "__init__",
        "implementation": _callable_identity(implementation, {id(model)}),
    }
    parameters: dict[str, Any] = {}
    tables: dict[str, str] = {}
    nested: list[dict[str, Any]] = []
    for name, value in sorted(vars(model).items()):
        field_name = name
        if isinstance(value, (pa.Table, qv.Table)):
            tables[field_name] = _table_sha256(value)
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(hasattr(item, "apply") for item in value)
        ):
            nested.extend(model_descriptor(item) for item in value)
        else:
            parameters[field_name] = _canonical_parameter(value, {id(model)})
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


def _provenance_with_fields(current: OdProvenance, **updates: object) -> OdProvenance:
    """Replace named fields without erasing independently recorded stages."""
    fields = {name: current.table.column(name) for name in OdProvenance.schema.names}
    fields.update(updates)
    return OdProvenance.from_kwargs(**fields)


def with_iod_provenance(
    fitted_orbits: FittedOrbits, ambiguous_with: Optional[Sequence[int]]
) -> FittedOrbits:
    """Record IOD ambiguity without erasing native provider diagnostics."""
    n = len(fitted_orbits)
    if n == 0 or ambiguous_with is None:
        return fitted_orbits
    alternatives = len(ambiguous_with)
    return fitted_orbits.set_column(
        "provenance",
        _provenance_with_fields(
            fitted_orbits.provenance,
            iod_ambiguous=[alternatives > 0] * n,
            iod_alternatives=[alternatives] * n,
        ),
    )


def carry_iod_provenance(
    iod_orbits: FittedOrbits, refined: FittedOrbits
) -> FittedOrbits:
    """Fill absent IOD-stage fields on a refinement from its exact seed.

    Refinement-stage fields are never copied from the seed. Existing fields on
    the refinement win, which also makes this operation idempotent.
    """
    if len(iod_orbits) != 1 or len(refined) == 0:
        return refined
    source = iod_orbits.provenance
    target = refined.provenance
    updates: dict[str, object] = {}
    for name in (
        "iod_provider",
        "iod_diagnostics",
        "iod_ambiguous",
        "iod_alternatives",
    ):
        destination = target.table.column(name)
        value = source.table.column(name)[0].as_py()
        if destination.null_count == len(refined) and value is not None:
            updates[name] = [value] * len(refined)
    if not updates:
        return refined
    return refined.set_column("provenance", _provenance_with_fields(target, **updates))


@dataclass(frozen=True)
class _ProvenancePreflight:
    adam_core_version: str
    adam_core_build: str
    settings: Optional[str]
    models: str


def _preflight_provenance(
    fitter: object, models: Sequence[object]
) -> _ProvenancePreflight:
    """Validate and freeze every Python-owned provenance field before fitting."""
    from .. import __version__
    from .native_orbit_fitter import NativeOrbitFitter

    settings: Optional[str] = None
    if isinstance(fitter, NativeOrbitFitter):
        iod_settings, refinement, fit_settings = fitter.native_settings()
        settings = _json(
            {"iod": iod_settings, "refinement": refinement, "fit": fit_settings}
        )
    descriptors = _json([model_descriptor(model) for model in models])
    from adam_core import _rust_native

    build_version, build_commit = _rust_native.build_identity()
    if build_version != __version__:
        raise RuntimeError(
            f"native adam-core version {build_version!r} does not match Python {__version__!r}"
        )
    if len(build_commit) != 40 or any(
        character not in "0123456789abcdef" for character in build_commit
    ):
        raise RuntimeError(
            "native adam-core build identity must be an exact clean source commit"
        )
    return _ProvenancePreflight(__version__, build_commit, settings, descriptors)


def _uniform_provider(values: list[Optional[str]], stage: str) -> Optional[str]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values) or len(set(present)) != 1:
        raise RuntimeError(
            f"native OD output omitted or disagreed on exact {stage} provider metadata"
        )
    return present[0]


def complete_provenance(
    fitted_orbits: FittedOrbits,
    members: FittedOrbitMembers,
    *,
    fitter: object,
    propagator: object,
    models: Sequence[object],
    route: str,
    observations: Optional[qv.Table] = None,
    preflight: Optional[_ProvenancePreflight] = None,
) -> FittedOrbits:
    """The `run_od` record on every orbit, keeping the IOD fields already
    present."""
    n = len(fitted_orbits)
    if n == 0:
        return fitted_orbits
    from .native_orbit_fitter import NativeOrbitFitter

    frozen = preflight or _preflight_provenance(fitter, models)
    fitter_cls = type(fitter)
    iod_backend: Optional[str] = None
    refinement_backend: Optional[str] = backend_identity(propagator)
    iod_provider: Optional[str] = None
    refinement_provider: Optional[str] = None
    if isinstance(fitter, NativeOrbitFitter):
        native_record = fitted_orbits.provenance
        iod_provider = _uniform_provider(native_record.iod_provider.to_pylist(), "IOD")
        refinement_provider = _uniform_provider(
            native_record.refinement_provider.to_pylist(), "refinement"
        )
        if iod_provider is not None:
            iod_metadata = json.loads(iod_provider)
            iod_backend = f"{iod_metadata['name']} {iod_metadata['version']}"
        else:
            iod_backend = backend_identity(
                fitter.iod_propagator_for(cast("Propagator", propagator))
            )
        if refinement_provider is not None:
            refinement_metadata = json.loads(refinement_provider)
            refinement_backend = (
                f"{refinement_metadata['name']} {refinement_metadata['version']}"
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
        adam_core_version=[frozen.adam_core_version] * n,
        adam_core_build=[frozen.adam_core_build] * n,
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
        settings=[frozen.settings] * n,
        models=[frozen.models] * n,
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
