"""Compact, reproducible orbit-determination provenance.

Built-in adam-core observation models are identified by the exact clean native
build commit and a stable entrypoint name. Custom models must declare an
explicit :class:`ModelProvenance`; runtime source inspection and import history
are deliberately outside the identity contract.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import math
import re
import subprocess
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pyarrow as pa
import quivr as qv

from ._native_dispatch import native_backend
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, OdProvenance
from .model_provenance import ModelProvenance

if TYPE_CHECKING:
    from ..propagator import Propagator

_CONTRACT_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _module_name(value: object, *, kind: str) -> str:
    module = getattr(value, "__module__", None)
    if not isinstance(module, str) or not module:
        raise TypeError(f"{kind} has no reproducible Python module identity")
    return module


def _package_version(module: str) -> Optional[str]:
    """Installed package version for backend display only; never imports it."""
    if not isinstance(module, str) or not module:
        raise TypeError("provenance module identity must be a non-empty string")
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
    """Return the declared native backend identity or a stable Python name."""
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
    table = table.combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bytes(sink.getvalue().to_pybytes())


def _table_sha256(table: pa.Table | qv.Table) -> str:
    return hashlib.sha256(_table_bytes(table)).hexdigest()


def _fit_input_sha256(observations: qv.Table, members: FittedOrbitMembers) -> str:
    """Hash the ordered caller table and exact astrometry consumed by native OD."""
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
    """Canonical NumPy identity independent of byte order and view layout."""
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


def _named_identity(value: object, *, kind: str) -> dict[str, str]:
    """Stable public name without importing or inspecting the owning module."""
    module = _module_name(value, kind=kind)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(qualname, str) or not qualname:
        raise TypeError(f"{kind} has no reproducible qualified name")
    return {"kind": kind, "module": module, "qualname": qualname}


def _canonical_float(value: float) -> float | dict[str, str]:
    if math.isnan(value):
        return {"float": "nan"}
    if math.isinf(value):
        return {"float": "+inf" if value > 0 else "-inf"}
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return {"float": "-0"}
    return value


def _canonical_numpy_scalar(value: np.generic) -> dict[str, Any]:
    dtype = value.dtype.str
    if isinstance(value, np.datetime64):
        encoded: Any = str(value)
    elif isinstance(value, np.timedelta64):
        encoded = str(value)
    elif isinstance(value, np.complexfloating):
        encoded = {
            "real": _canonical_float(float(value.real)),
            "imag": _canonical_float(float(value.imag)),
        }
    elif isinstance(value, np.floating):
        encoded = _canonical_float(float(value))
    else:
        encoded = value.item()
    return {"dtype": dtype, "value": encoded}


def _is_namedtuple_class(value: object) -> bool:
    return (
        isinstance(value, type)
        and issubclass(value, tuple)
        and isinstance(getattr(value, "_fields", None), tuple)
        and all(isinstance(field, str) for field in value._fields)
    )


def _is_namedtuple_instance(value: object) -> bool:
    return isinstance(value, tuple) and _is_namedtuple_class(type(value))


def _canonical_parameter(value: Any, seen: set[int]) -> Any:
    """Canonical configuration independent of interpreter and import state."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, np.generic):
        return {"numpy_scalar": _canonical_numpy_scalar(value)}
    if isinstance(value, np.dtype):
        return {"numpy_dtype": value.str}
    if isinstance(value, np.ufunc):
        return {
            "numpy_ufunc": value.__name__,
            "nin": value.nin,
            "nout": value.nout,
        }
    if isinstance(value, dt.datetime):
        return {
            "datetime": value.isoformat(timespec="microseconds"),
            "fold": value.fold,
        }
    if isinstance(value, dt.date):
        return {"date": value.isoformat()}
    if isinstance(value, dt.time):
        return {"time": value.isoformat(timespec="microseconds"), "fold": value.fold}
    if isinstance(value, dt.timedelta):
        return {
            "timedelta": {
                "days": value.days,
                "seconds": value.seconds,
                "microseconds": value.microseconds,
            }
        }
    if isinstance(value, Path):
        if not value.is_file():
            raise ValueError(f"provenance path is not a readable file: {value}")
        return {
            "file_name": value.name,
            "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
        }
    if isinstance(value, type) and issubclass(value, Enum):
        return {"enum_class": _named_identity(value, kind="enum-class-v1")}
    if _is_namedtuple_class(value):
        return {
            "namedtuple_class": _named_identity(value, kind="namedtuple-class-v1"),
            "fields": list(value._fields),
        }
    if isinstance(value, type) and issubclass(value, np.generic):
        return {"numpy_type": np.dtype(value).str}
    if isinstance(value, type):
        return {"class": _named_identity(value, kind="class-v1")}
    if isinstance(value, Enum):
        return {
            "enum": _named_identity(type(value), kind="enum-class-v1"),
            "name": value.name,
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
    if isinstance(value, functools.partial):
        seen.add(identity)
        try:
            return {
                "partial": {
                    "function": _canonical_parameter(value.func, seen),
                    "args": _canonical_parameter(value.args, seen),
                    "keywords": _canonical_parameter(value.keywords or {}, seen),
                    "fields": _canonical_parameter(vars(value), seen),
                }
            }
        finally:
            seen.remove(identity)
    if isinstance(
        value,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            types.MethodType,
        ),
    ):
        function = value.__func__ if isinstance(value, types.MethodType) else value
        return {"callable": _named_identity(function, kind="callable-v1")}

    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            encoded = [
                (_canonical_parameter(key, seen), _canonical_parameter(item, seen))
                for key, item in value.items()
            ]
            encoded.sort(key=lambda pair: _json(pair[0]))
            if all(isinstance(key, str) for key, _ in encoded):
                return {key: item for key, item in encoded}
            return {"mapping": encoded}
        if _is_namedtuple_instance(value):
            return {
                "namedtuple": _named_identity(type(value), kind="namedtuple-class-v1"),
                "fields": {
                    name: _canonical_parameter(getattr(value, name), seen)
                    for name in value._fields
                },
            }
        if isinstance(value, Sequence):
            return [_canonical_parameter(item, seen) for item in value]
        if isinstance(value, (set, frozenset)):
            encoded = [_canonical_parameter(item, seen) for item in value]
            return sorted(encoded, key=_json)
        fields = getattr(value, "__dict__", None)
        if isinstance(fields, dict):
            return {
                "object": _named_identity(type(value), kind="object-class-v1"),
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


def _builtin_model_classes() -> frozenset[type[Any]]:
    from .observation_uncertainty import (
        CompositeModel,
        EFCC18DebiasModel,
        EmpiricalCovarianceModel,
        IdentityModel,
        NightBatchDeweightingModel,
        PerformanceWeightedModel,
        SigmaFloorModel,
    )
    from .veres2017 import SigmaFillModel, VeresFloorModel, VeresReplaceModel

    return frozenset(
        {
            IdentityModel,
            EmpiricalCovarianceModel,
            PerformanceWeightedModel,
            SigmaFloorModel,
            NightBatchDeweightingModel,
            EFCC18DebiasModel,
            CompositeModel,
            VeresFloorModel,
            VeresReplaceModel,
            SigmaFillModel,
        }
    )


def _is_builtin_model(model: object) -> bool:
    """Only exact shipped classes with no per-instance apply override are built-ins."""
    cls = type(model)
    try:
        fields = vars(model)
    except TypeError:
        fields = {}
    return cls in _builtin_model_classes() and "apply" not in fields


def _validated_build_identity() -> tuple[str, str]:
    from adam_core import _rust_native

    from .. import __version__

    build_version, build_commit = _rust_native.build_identity()
    if build_version != __version__:
        raise RuntimeError(
            f"native adam-core version {build_version!r} does not match Python {__version__!r}"
        )
    if not isinstance(build_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", build_commit
    ):
        raise RuntimeError(
            "native adam-core build identity must be an exact clean source commit"
        )
    return str(build_version), build_commit


def _model_fields(model: object, build_identity: tuple[str, str]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    tables: dict[str, str] = {}
    nested: list[dict[str, Any]] = []
    for name, value in sorted(vars(model).items()):
        if isinstance(value, (pa.Table, qv.Table)):
            tables[name] = _table_sha256(value)
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(hasattr(item, "apply") for item in value)
        ):
            nested.extend(
                model_descriptor(item, _build_identity=build_identity) for item in value
            )
        else:
            parameters[name] = _canonical_parameter(value, {id(model)})
    fields: dict[str, Any] = {}
    if parameters:
        fields["parameters"] = parameters
    if tables:
        fields["tables_sha256"] = tables
    if nested:
        fields["models"] = nested
    return fields


def _custom_contract(model: object) -> ModelProvenance:
    hook = getattr(model, "model_provenance", None)
    cls = type(model)
    name = f"{getattr(cls, '__module__', '<unknown>')}.{getattr(cls, '__qualname__', cls.__name__)}"
    if not callable(hook):
        raise TypeError(
            f"custom observation model {name} must implement model_provenance() "
            "and return ModelProvenance"
        )
    contract = hook()
    if not isinstance(contract, ModelProvenance):
        raise TypeError(
            f"custom observation model {name} model_provenance() must return ModelProvenance"
        )
    if not _CONTRACT_NAME.fullmatch(contract.package):
        raise TypeError(
            "model provenance package must be a stable non-empty package name"
        )
    if not isinstance(contract.version, str) or not contract.version.strip():
        raise TypeError("model provenance version must be a non-empty string")
    if not _HEX64.fullmatch(contract.implementation_fingerprint):
        raise TypeError(
            "model provenance implementation_fingerprint must be 64 lowercase hexadecimal characters"
        )
    if not isinstance(contract.configuration, Mapping):
        raise TypeError("model provenance configuration must be a mapping")
    return contract


def model_descriptor(
    model: object, *, _build_identity: tuple[str, str] | None = None
) -> dict[str, Any]:
    """Return deterministic implementation, configuration, and data identity.

    No source files, bytecode, loaded-module state, interpreter version, or
    absolute paths contribute to the descriptor.
    """
    cls = type(model)
    cls_module = _module_name(cls, kind="model class")
    type_name = f"{cls_module}.{cls.__qualname__}"
    if _is_builtin_model(model):
        build_version, build_commit = _build_identity or _validated_build_identity()
        implementation = {
            "kind": "adam-core-built-in-v1",
            "build_commit": build_commit,
            "class": type_name,
            "entrypoint": "apply",
        }
        descriptor: dict[str, Any] = {
            "type": type_name,
            "implementation_version": build_version,
            "implementation_entrypoint": "apply",
            "implementation_class": implementation,
            "implementation": implementation,
            **_model_fields(model, (build_version, build_commit)),
        }
    else:
        contract = _custom_contract(model)
        implementation = {
            "kind": "explicit-model-provenance-v1",
            "package": contract.package,
            "version": contract.version,
            "implementation_fingerprint": contract.implementation_fingerprint,
            "entrypoint": "apply",
        }
        descriptor = {
            "type": type_name,
            "implementation_version": contract.version,
            "implementation_entrypoint": "apply",
            "implementation_class": implementation,
            "implementation": implementation,
            "parameters": _canonical_parameter(contract.configuration, {id(model)}),
        }
    descriptor["descriptor_sha256"] = hashlib.sha256(
        _json(descriptor).encode("utf-8")
    ).hexdigest()
    return descriptor


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def _relevant_untracked_source() -> tuple[str, ...]:
    """Git-visible untracked source in a development checkout.

    Cargo cannot watch for arbitrary new files without recursively watching
    package directories (which creates a rebuild loop on ignored generated
    artifacts). Exact tracked files remain Cargo inputs; this runtime preflight
    closes the new-untracked-file gap for source-checkout OD runs. Release
    builds re-run on their environment change and enforce the same check in
    ``build.rs``.
    """
    repo = Path(__file__).resolve().parents[3]
    if not (repo / ".git").exists():
        return ()
    try:
        output = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo}",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "src/adam_core",
                "rust",
            ],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot verify development source provenance") from error
    relevant = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        try:
            path = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError("untracked source path is not valid UTF-8") from error
        if (
            path.startswith("src/adam_core/")
            and "/tests/" not in path
            and path.endswith(".py")
        ):
            relevant.append(path)
        elif path.startswith("rust/") and (
            "/src/" in path
            or path.endswith("/Cargo.toml")
            or path.endswith("/build.rs")
        ):
            relevant.append(path)
    return tuple(sorted(relevant))


def _preflight_provenance(
    fitter: object, models: Sequence[object]
) -> _ProvenancePreflight:
    """Validate and freeze every Python-owned provenance field before fitting."""
    from .native_orbit_fitter import NativeOrbitFitter

    # Build/source validation comes first and performs no user-model work. The
    # validated identity is then reused for every shipped model descriptor.
    build_version, build_commit = _validated_build_identity()
    untracked = _relevant_untracked_source()
    if untracked:
        raise RuntimeError(
            "native adam-core build identity cannot cover untracked source: "
            + ", ".join(untracked)
        )

    settings: Optional[str] = None
    if isinstance(fitter, NativeOrbitFitter):
        iod_settings, refinement, fit_settings = fitter.native_settings()
        settings = _json(
            {"iod": iod_settings, "refinement": refinement, "fit": fit_settings}
        )
    build_identity = (build_version, build_commit)
    descriptors = _json(
        [model_descriptor(model, _build_identity=build_identity) for model in models]
    )
    return _ProvenancePreflight(build_version, build_commit, settings, descriptors)


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
