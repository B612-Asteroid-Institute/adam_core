"""Compact, reproducible orbit-determination provenance.

Built-in adam-core observation models are identified by the exact clean native
build commit and a stable entrypoint name. Custom models must declare an
explicit :class:`ModelProvenance`; runtime source inspection and import history
are deliberately outside the identity contract.
"""

from __future__ import annotations

import ast
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
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pyarrow as pa
import quivr as qv

from ._native_dispatch import native_backend
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, OdProvenance
from .model_provenance import ModelProvenance, _BehaviorSpec

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


_STDLIB_PUBLIC_CLASS_MODULES = {
    "pathlib": frozenset(
        {
            "Path",
            "PosixPath",
            "PurePath",
            "PurePosixPath",
            "PureWindowsPath",
            "WindowsPath",
        }
    )
}


def _named_identity(value: object, *, kind: str) -> dict[str, str]:
    """Stable public name without importing or inspecting the owning module."""
    module = _module_name(value, kind=kind)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(qualname, str) or not qualname:
        raise TypeError(f"{kind} has no reproducible qualified name")
    # CPython 3.13 moved pathlib implementations into private submodules while
    # preserving their public pathlib names. Provenance records the public API,
    # not that interpreter-private relocation.
    public_name = qualname.rsplit(".", 1)[-1]
    if (
        module.startswith("pathlib.")
        and public_name in _STDLIB_PUBLIC_CLASS_MODULES["pathlib"]
    ):
        module = "pathlib"
        qualname = public_name
    return {"kind": kind, "module": module, "qualname": qualname}


def _canonical_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return "-0"
    # hex() is an exact, locale-independent representation and distinguishes
    # every finite binary64 value.
    return value.hex()


def _tag(tag: str, **fields: Any) -> dict[str, Any]:
    return {"tag": tag, **fields}


def _canonical_numpy_scalar(value: np.generic) -> dict[str, Any]:
    """Exact scalar payload using the array byte-order canonicalizer."""
    identity = _numpy_sha256(np.asarray(value))
    # Scalar and zero-dimensional array categories remain distinct because the
    # caller gives this payload the ``numpy-scalar`` tag.
    return identity


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
    """Injective tagged configuration independent of interpreter/import state.

    User mappings are always encoded as tagged entry lists, so no mapping can
    forge a scalar, sequence, set, array, object, or callable encoding.
    """
    if value is None:
        return _tag("none")
    if isinstance(value, bool):
        return _tag("bool", value=value)
    if isinstance(value, int):
        return _tag("int", value=str(value))
    if isinstance(value, str):
        return _tag("str", value=value)
    if isinstance(value, float):
        return _tag("float64", value=_canonical_float(value))
    if isinstance(value, bytes):
        return _tag("bytes", size=len(value), sha256=hashlib.sha256(value).hexdigest())
    if isinstance(value, bytearray):
        payload = bytes(value)
        return _tag(
            "bytearray", size=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
    if isinstance(value, np.generic):
        return _tag("numpy-scalar", **_canonical_numpy_scalar(value))
    if isinstance(value, np.dtype):
        return _tag("numpy-dtype", value=value.str)
    if isinstance(value, np.ufunc):
        return _tag("numpy-ufunc", name=value.__name__, nin=value.nin, nout=value.nout)
    if isinstance(value, dt.datetime):
        return _tag(
            "datetime",
            value=value.isoformat(timespec="microseconds"),
            fold=value.fold,
        )
    if isinstance(value, dt.date):
        return _tag("date", value=value.isoformat())
    if isinstance(value, dt.time):
        return _tag(
            "time", value=value.isoformat(timespec="microseconds"), fold=value.fold
        )
    if isinstance(value, dt.timedelta):
        return _tag(
            "timedelta",
            days=value.days,
            seconds=value.seconds,
            microseconds=value.microseconds,
        )
    if isinstance(value, Path):
        if not value.is_file():
            raise ValueError(f"provenance path is not a readable file: {value}")
        return _tag(
            "file",
            name=value.name,
            size=value.stat().st_size,
            sha256=hashlib.sha256(value.read_bytes()).hexdigest(),
        )
    if isinstance(value, type) and issubclass(value, Enum):
        return _tag("enum-class", identity=_named_identity(value, kind="enum-class-v1"))
    if _is_namedtuple_class(value):
        return _tag(
            "namedtuple-class",
            identity=_named_identity(value, kind="namedtuple-class-v1"),
            fields=list(value._fields),
        )
    if isinstance(value, type) and issubclass(value, np.generic):
        return _tag("numpy-type", dtype=np.dtype(value).str)
    if isinstance(value, type):
        return _tag("class", identity=_named_identity(value, kind="class-v1"))
    if isinstance(value, Enum):
        return _tag(
            "enum-member",
            identity=_named_identity(type(value), kind="enum-class-v1"),
            name=value.name,
        )

    identity = id(value)
    if identity in seen:
        raise ValueError("cyclic provenance configuration is not supported")
    if isinstance(value, np.ndarray):
        return _tag("numpy-array", **_numpy_sha256(value))
    if isinstance(value, (pa.Table, qv.Table)):
        return _tag("arrow-table", sha256=_table_sha256(value))
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        return _tag("arrow-array", sha256=_arrow_array_sha256(value))
    if isinstance(value, functools.partial):
        seen.add(identity)
        try:
            return _tag(
                "partial",
                function=_canonical_parameter(value.func, seen),
                args=_canonical_parameter(value.args, seen),
                keywords=_canonical_parameter(value.keywords or {}, seen),
                fields=_canonical_parameter(vars(value), seen),
            )
        finally:
            seen.remove(identity)
    if isinstance(value, (types.FunctionType, types.MethodType)):
        raise TypeError(
            "Python function/method provenance fields require an explicit portable "
            "identity; encode the public choice as data and cover its science in "
            "ModelProvenance.implementation_fingerprint"
        )
    if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        function = getattr(value, "__func__", value)
        return _tag(
            "builtin-callable",
            identity=_named_identity(function, kind="builtin-callable-v1"),
        )

    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            entries = [
                [_canonical_parameter(key, seen), _canonical_parameter(item, seen)]
                for key, item in value.items()
            ]
            entries.sort(key=lambda pair: _json(pair[0]))
            return _tag("mapping", entries=entries)
        if _is_namedtuple_instance(value):
            return _tag(
                "namedtuple",
                identity=_named_identity(type(value), kind="namedtuple-class-v1"),
                fields=[
                    [name, _canonical_parameter(getattr(value, name), seen)]
                    for name in value._fields
                ],
            )
        if isinstance(value, list):
            return _tag(
                "list", items=[_canonical_parameter(item, seen) for item in value]
            )
        if isinstance(value, tuple):
            return _tag(
                "tuple", items=[_canonical_parameter(item, seen) for item in value]
            )
        if isinstance(value, (set, frozenset)):
            encoded = [_canonical_parameter(item, seen) for item in value]
            return _tag(
                "frozenset" if isinstance(value, frozenset) else "set",
                items=sorted(encoded, key=_json),
            )
        fields = getattr(value, "__dict__", None)
        if isinstance(fields, dict):
            return _tag(
                "object",
                identity=_named_identity(type(value), kind="object-class-v1"),
                fields=[
                    [name, _canonical_parameter(item, seen)]
                    for name, item in sorted(fields.items())
                ],
            )
    finally:
        seen.remove(identity)
    raise TypeError(
        f"unsupported provenance field {type(value).__module__}.{type(value).__qualname__}"
    )


def _builtin_behavior_registry() -> dict[type[Any], _BehaviorSpec]:
    from .observation_uncertainty import _PROVENANCE_BEHAVIOR_REGISTRY as core
    from .veres2017 import _PROVENANCE_BEHAVIOR_REGISTRY as veres

    return {**core, **veres}


def _static_class_member(cls: type[Any], name: str) -> object | None:
    for owner in cls.__mro__:
        if name in vars(owner):
            return vars(owner)[name]
    return None


def _class_defines_member(cls: type[Any], name: str) -> bool:
    return any(name in vars(owner) for owner in cls.__mro__)


def _matches_behavior_spec(value: object, spec: _BehaviorSpec) -> bool:
    """Match an exact shipped type, its named members, and collaborators."""
    if type(value) is not spec.model_type:
        return False
    member_names = {name for name, _ in spec.class_members}
    for name, expected in spec.class_members:
        if _static_class_member(spec.model_type, name) is not expected:
            return False
    try:
        fields = vars(value)
    except TypeError:
        fields = {}
    # Registered methods and class data may not be shadowed per instance. Any
    # other callable shadow is also custom rather than silently acquiring the
    # shipped identity.
    if any(name in member_names or callable(item) for name, item in fields.items()):
        return False
    for name in spec.instance_fields:
        if name not in fields or _class_defines_member(spec.model_type, name):
            return False
    for attribute, collaborator in spec.collaborators:
        if attribute not in fields or not _matches_behavior_spec(
            fields[attribute], collaborator
        ):
            return False
    return True


def _is_builtin_model(model: object) -> bool:
    """Whether the complete registered behavior is exactly the shipped graph."""
    expected = _builtin_behavior_registry().get(type(model))
    return expected is not None and _matches_behavior_spec(model, expected)


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


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declares_class(source: str, filename: str, qualname: str) -> bool:
    """Whether a module source file declares this non-local class path."""
    try:
        body = ast.parse(source, filename=filename).body
    except (SyntaxError, ValueError):
        return False
    for part in qualname.split("."):
        declaration = next(
            (
                node
                for node in body
                if isinstance(node, ast.ClassDef) and node.name == part
            ),
            None,
        )
        if declaration is None:
            return False
        body = declaration.body
    return True


def _validate_custom_model_distribution(
    cls: type[Any], contract: ModelProvenance
) -> None:
    """Validate a custom class against immutable installed metadata.

    This reads only the claimed distribution's metadata and its directly
    listed module source. It imports no provider module and does not consult
    ``sys.modules``, import history, bytecode, or source closure.
    """
    module = _module_name(cls, kind="model class")
    qualname = getattr(cls, "__qualname__", None)
    if module == "__main__" or module.startswith("__main__."):
        raise TypeError(
            "custom observation models from __main__ are not reproducible; "
            "use a class from an installed distribution"
        )
    if (
        not isinstance(qualname, str)
        or not qualname
        or "<locals>" in qualname
        or any(not part.isidentifier() for part in qualname.split("."))
    ):
        raise TypeError(
            "custom observation model must be a non-local class declared by an "
            "installed distribution"
        )
    module_parts = module.split(".")
    if any(not part.isidentifier() for part in module_parts):
        raise TypeError(
            "custom observation model module must be an installed Python module"
        )

    try:
        distribution = metadata.distribution(contract.package)
        installed_name = distribution.metadata.get("Name")
        installed_version = distribution.version
    except Exception as error:
        raise TypeError(
            f"model provenance distribution {contract.package!r} is not installed"
        ) from error
    if not isinstance(installed_name, str) or _canonical_distribution_name(
        installed_name
    ) != _canonical_distribution_name(contract.package):
        raise TypeError(
            f"model provenance package {contract.package!r} does not match the "
            "installed distribution identity"
        )
    if installed_version != contract.version:
        raise TypeError(
            f"model provenance version {contract.version!r} does not match "
            f"installed {contract.package!r} version {installed_version!r}"
        )

    module_path = PurePosixPath(*module_parts)
    candidates = {
        module_path.with_suffix(".py").as_posix(),
        (module_path / "__init__.py").as_posix(),
    }
    files = distribution.files
    if files is None:
        files = []
    matching = []
    for file in files:
        relative = PurePosixPath(str(file).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            continue
        if relative.as_posix() in candidates:
            matching.append(file)
    if len(matching) != 1:
        raise TypeError(
            f"custom observation model module {module!r} is not Python source "
            f"owned by installed distribution {contract.package!r}"
        )
    source_file = matching[0]
    try:
        source = Path(distribution.locate_file(source_file)).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise TypeError(
            f"custom observation model module {module!r} has no readable installed "
            "Python source"
        ) from error
    if not _declares_class(source, str(source_file), qualname):
        raise TypeError(
            f"custom observation model {module}.{qualname} is dynamic or not "
            f"declared by installed distribution {contract.package!r}"
        )


def _custom_contract(model: object) -> ModelProvenance:
    from .observation_uncertainty import ObservationUncertaintyModel

    cls = type(model)
    if not isinstance(model, ObservationUncertaintyModel):
        raise TypeError(
            f"custom observation model {getattr(cls, '__module__', '<unknown>')}."
            f"{getattr(cls, '__qualname__', cls.__name__)} must subclass "
            "ObservationUncertaintyModel"
        )
    hook = getattr(model, "model_provenance", None)
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
    _validate_custom_model_distribution(cls, contract)
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


def _runtime_source_path(path: str) -> bool:
    if "target" in path.split("/"):
        return False
    if path.startswith("src/adam_core/"):
        relative = path.removeprefix("src/adam_core/")
        parts = relative.split("/")
        return (
            "tests" not in parts
            and "__pycache__" not in parts
            and relative != "_version.py"
            and not relative.startswith("_rust_native")
        )
    return path.startswith("rust/") and (
        "/src/" in path or path.endswith("/Cargo.toml") or path.endswith("/build.rs")
    )


def _git_output(repo: Path, arguments: list[str]) -> bytes:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo}", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot verify development source provenance") from error


def _decode_git_paths(output: bytes) -> set[str]:
    paths: set[str] = set()
    for raw in output.split(b"\0"):
        if not raw:
            continue
        try:
            paths.add(raw.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise RuntimeError("source path is not valid UTF-8") from error
    return paths


def _source_checkout_problems(
    build_commit: str, *, repo: Path | None = None
) -> tuple[str, ...]:
    """Return reproducibility failures for an executing Git source checkout.

    Installed wheels have no enclosing Core ``.git`` and rely solely on their
    embedded release identity. In a checkout, HEAD, tracked/staged runtime
    source, visible untracked Rust source, and *every* importable package
    Python file (including gitignored files) are verified before user work.
    """
    repo = (repo or Path(__file__).resolve().parents[3]).resolve()
    if not (repo / ".git").exists():
        return ()
    actual = _git_output(repo, ["rev-parse", "HEAD"]).decode("ascii").strip()
    problems: list[str] = []
    if actual != build_commit:
        problems.append(
            f"checkout HEAD {actual!r} does not match native build commit {build_commit!r}"
        )

    modified = _decode_git_paths(
        _git_output(
            repo,
            ["diff", "--name-only", "-z", "HEAD", "--", "src/adam_core", "rust"],
        )
    )
    problems.extend(
        f"tracked/staged source modification: {path}"
        for path in sorted(path for path in modified if _runtime_source_path(path))
    )

    tracked = _decode_git_paths(
        _git_output(repo, ["ls-files", "-z", "--", "src/adam_core", "rust"])
    )
    package = repo / "src" / "adam_core"
    if package.is_dir():
        for candidate in package.rglob("*.py"):
            relative = candidate.relative_to(repo).as_posix()
            if _runtime_source_path(relative) and relative not in tracked:
                problems.append(f"untracked package Python source: {relative}")

    rust = repo / "rust"
    if rust.is_dir():
        for candidate in rust.rglob("*"):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(repo).as_posix()
            if _runtime_source_path(relative) and relative not in tracked:
                problems.append(f"untracked Rust source: {relative}")
    return tuple(dict.fromkeys(problems))


def _relevant_untracked_source() -> tuple[str, ...]:
    """Backward-compatible detail used by older diagnostics/tests."""
    _, build_commit = _validated_build_identity()
    return tuple(
        problem.split(": ", 1)[-1]
        for problem in _source_checkout_problems(build_commit)
        if problem.startswith(("untracked package Python", "untracked Rust"))
    )


def _preflight_provenance(
    fitter: object, models: Sequence[object]
) -> _ProvenancePreflight:
    """Validate and freeze every Python-owned provenance field before fitting."""
    from .native_orbit_fitter import NativeOrbitFitter

    # Build/source validation comes first and performs no user-model work. The
    # validated identity is then reused for every shipped model descriptor.
    build_version, build_commit = _validated_build_identity()
    source_problems = _source_checkout_problems(build_commit)
    if source_problems:
        raise RuntimeError(
            "native adam-core build identity cannot cover executing source checkout: "
            + "; ".join(source_problems)
        )

    # Model identity is validated before asking even a built-in fitter or any
    # downstream provider for settings. Invalid custom provenance therefore
    # cannot trigger model, fitter, or provider work.
    build_identity = (build_version, build_commit)
    descriptors = _json(
        [model_descriptor(model, _build_identity=build_identity) for model in models]
    )
    settings: Optional[str] = None
    if isinstance(fitter, NativeOrbitFitter):
        iod_settings, refinement, fit_settings = fitter.native_settings()
        settings = _json(
            {"iod": iod_settings, "refinement": refinement, "fit": fit_settings}
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
