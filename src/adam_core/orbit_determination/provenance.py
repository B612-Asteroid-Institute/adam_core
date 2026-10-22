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

import ast
import functools
import hashlib
import inspect
import json
import subprocess
import sys
import types
import weakref
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from importlib import import_module, metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pyarrow as pa
import quivr as qv

from ._native_dispatch import native_backend
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, OdProvenance

if TYPE_CHECKING:
    from ..propagator import Propagator


def _module_name(value: object, *, kind: str) -> str:
    module = getattr(value, "__module__", None)
    if not isinstance(module, str) or not module:
        raise TypeError(f"{kind} has no reproducible Python module identity")
    return module


def _package_version(module: str) -> Optional[str]:
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


def _is_stdlib_module(module_name: str) -> bool:
    """Whether *module_name* belongs to Python's versioned standard library."""
    return module_name.split(".")[0] in sys.stdlib_module_names


_MODULE_SOURCE_CACHE: weakref.WeakKeyDictionary[types.ModuleType, str] = (
    weakref.WeakKeyDictionary()
)
_MODULE_EXECUTION_TOKENS: weakref.WeakKeyDictionary[
    types.ModuleType, dict[str, weakref.ReferenceType[object]]
] = weakref.WeakKeyDictionary()
_SOURCE_IDENTITY_CACHE: weakref.WeakKeyDictionary[object, dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)


def _source_tree_roots(module: types.ModuleType) -> list[Path]:
    top_name = module.__name__.split(".")[0]
    try:
        top = import_module(top_name)
    except Exception:
        top = module
    package_paths = getattr(top, "__path__", None)
    if package_paths is not None:
        roots = sorted(
            {Path(item).resolve() for item in package_paths if Path(item).is_dir()},
            key=str,
        )
        if roots:
            return roots
    try:
        source_file = inspect.getsourcefile(top) or inspect.getsourcefile(module)
    except TypeError:
        source_file = None
    return [] if source_file is None else [Path(source_file).resolve().parent]


def _module_text(module: types.ModuleType) -> str | None:
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    get_source = getattr(loader, "get_source", None)
    if callable(get_source):
        try:
            source = get_source(module.__name__)
        except Exception:
            source = None
        if isinstance(source, str):
            return source
    try:
        source_file = inspect.getsourcefile(module)
    except TypeError:
        source_file = None
    if source_file is not None:
        try:
            return Path(source_file).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            pass
    return None


def _local_source_record(
    module_name: str, roots: Sequence[Path]
) -> tuple[types.ModuleType, str, str] | None:
    module = sys.modules.get(module_name)
    if not isinstance(module, types.ModuleType):
        return None
    try:
        source_file = inspect.getsourcefile(module)
    except TypeError:
        return None
    if source_file is None:
        return None
    path = Path(source_file).resolve()
    root_index = next(
        (index for index, root in enumerate(roots) if path.is_relative_to(root)),
        None,
    )
    if root_index is None or path.suffix != ".py" or path.name == "_version.py":
        return None
    source = _module_text(module)
    if source is None:
        raise TypeError(
            f"provenance module {module_name!r} has no reproducible source content"
        )
    relative = path.relative_to(roots[root_index]).as_posix()
    return module, f"{root_index}/{relative}", source


def _imported_local_modules(
    module: types.ModuleType, source: str, roots: Sequence[Path]
) -> list[str]:
    """Return statically imported local modules without broad package hashing."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise TypeError(
            f"provenance module {module.__name__!r} has invalid source"
        ) from error
    package = (
        module.__name__
        if getattr(module, "__path__", None) is not None
        else module.__name__.rpartition(".")[0]
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if not package:
                continue
            relative = "." * node.level + (node.module or "")
            try:
                base = __import__(
                    "importlib.util", fromlist=["resolve_name"]
                ).resolve_name(relative, package)
            except (ImportError, ValueError):
                continue
        elif node.module:
            base = node.module
        else:
            continue
        if len(node.names) == 1 and node.names[0].name == "*":
            imported.add(base)
            continue
        found_child = False
        for alias in node.names:
            child = f"{base}.{alias.name}"
            if _local_source_record(child, roots) is not None:
                imported.add(child)
                found_child = True
        if not found_child:
            imported.add(base)
    return sorted(imported)


def _source_tree_content(module: types.ModuleType) -> str:
    """Deterministic executing-source dependency closure for local helpers.

    Only the entrypoint module and local modules it statically imports are
    included, recursively. This covers sibling/same-package delegated helpers
    without making provenance depend on tests or unrelated package files.
    """
    roots = _source_tree_roots(module)
    pending = [module.__name__]
    seen: set[str] = set()
    records: dict[str, bytes] = {}
    while pending:
        module_name = pending.pop()
        if module_name in seen:
            continue
        seen.add(module_name)
        record = _local_source_record(module_name, roots)
        if record is None:
            continue
        local_module, relative, source = record
        records[relative] = source.encode("utf-8")
        pending.extend(_imported_local_modules(local_module, source, roots))
    if not records:
        raise TypeError(
            f"provenance module {module.__name__!r} has no reproducible source tree"
        )
    digest = hashlib.sha256()
    for relative, payload in sorted(records.items()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"python-source-dependencies-v1:{digest.hexdigest()}"


def _note_executing_object(module_name: str, qualname: str, value: object) -> None:
    """Invalidate a source snapshot only when executing objects were reloaded.

    File edits alone leave the loaded class/function object unchanged and must
    not alter provenance. Re-executing/reloading a definition creates a new
    object for the same qualified name, at which point a fresh source-tree
    snapshot truthfully describes the new executing implementation.
    """
    try:
        module = import_module(module_name)
        reference = weakref.ref(value)
    except (Exception, TypeError):
        return
    tokens = _MODULE_EXECUTION_TOKENS.setdefault(module, {})
    prior = tokens.get(qualname)
    if prior is not None and prior() is not value:
        _MODULE_SOURCE_CACHE.pop(module, None)
    tokens[qualname] = reference


def _module_source(module_name: str) -> str:
    """Stable executing-package source snapshot or a fail-closed error.

    The first identity request snapshots the loaded module's deterministic
    source tree. Later on-disk edits cannot falsely claim that already-loaded
    executing code changed; reloading into a fresh module/process takes a new
    snapshot.
    """
    try:
        module = import_module(module_name)
    except Exception as error:
        raise TypeError(
            f"cannot import provenance module {module_name!r} reproducibly"
        ) from error
    cached = _MODULE_SOURCE_CACHE.get(module)
    if cached is not None:
        return cached
    source = _module_text(module)
    if not isinstance(source, str):
        raise TypeError(
            f"provenance module {module_name!r} has no reproducible source content"
        )
    # Hashing the serialized tree marker through the existing module-source
    # field preserves the stable descriptor schema while covering same-package
    # and sibling local delegated helpers.
    snapshot = source + "\n" + _source_tree_content(module)
    _MODULE_SOURCE_CACHE[module] = snapshot
    return snapshot


def _entrypoint_source(
    value: object, module_name: str, qualname: str, kind: str
) -> str:
    try:
        return inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise TypeError(
            f"{kind} {module_name}.{qualname} has no reproducible source content"
        ) from error


def _named_source_identity(
    module_name: str, qualname: str, *, kind: str, implementation_version: Optional[str]
) -> dict[str, Any]:
    return {
        "module": module_name,
        "qualname": qualname,
        "implementation_version": implementation_version,
        "implementation_kind": kind,
    }


def _source_identity(value: object, *, kind: str) -> dict[str, Any]:
    """Interpreter-independent implementation identity.

    Local and package Python code is identified by its entrypoint and a cached
    deterministic source-tree snapshot. Standard-library and language classes
    are represented by stable public names rather than version-specific stdlib
    source. CPython bytecode is deliberately excluded.
    """
    try:
        cached = _SOURCE_IDENTITY_CACHE.get(value)
    except TypeError:
        cached = None
    if cached is not None:
        return deepcopy(cached)

    module_name = _module_name(value, kind=kind)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(qualname, str) or not qualname:
        raise TypeError(f"{kind} has no reproducible qualified name")
    version = _package_version(module_name)
    _note_executing_object(module_name, qualname, value)
    if _is_stdlib_module(module_name):
        identity = _named_source_identity(
            module_name,
            qualname,
            kind="stdlib-named-v1",
            implementation_version=None,
        )
    else:
        try:
            source_file = inspect.getsourcefile(value)
        except TypeError:
            source_file = None
        if source_file is None and version is not None:
            # Extension-defined classes/functions have no Python entrypoint
            # source. Their immutable installed package tree, public name and
            # distribution version form the portable identity.
            identity = _named_source_identity(
                module_name,
                qualname,
                kind="installed-distribution-v1",
                implementation_version=version,
            )
            identity["module_source_sha256"] = hashlib.sha256(
                _module_source(module_name).encode("utf-8")
            ).hexdigest()
        else:
            if version is None:
                entrypoint_source = _entrypoint_source(
                    value, module_name, qualname, kind
                )
            else:
                try:
                    entrypoint_source = inspect.getsource(value)
                except (OSError, TypeError):
                    identity = _named_source_identity(
                        module_name,
                        qualname,
                        kind="installed-distribution-v1",
                        implementation_version=version,
                    )
                    identity["module_source_sha256"] = hashlib.sha256(
                        _module_source(module_name).encode("utf-8")
                    ).hexdigest()
                    entrypoint_source = None
            if entrypoint_source is None:
                module_source = None
            else:
                module_source = _module_source(module_name)
                identity = _portable_python_source_identity(
                    module_name,
                    qualname,
                    version,
                    entrypoint_source,
                    module_source,
                )
    if inspect.isclass(value):
        identity["mro"] = [
            _source_identity(base, kind="base class")
            for base in value.__mro__[1:]
            if base is not object
        ]
    try:
        _SOURCE_IDENTITY_CACHE[value] = deepcopy(identity)
    except TypeError:
        pass
    return identity


def _portable_python_source_identity(
    module_name: str,
    qualname: str,
    implementation_version: str | None,
    entrypoint_source: str,
    module_source: str,
) -> dict[str, Any]:
    # Keep this small source/content core stable for independent mutation
    # verification while richer MRO and package identity is added by callers.
    return {
        "module": module_name,
        "qualname": qualname,
        "implementation_version": _package_version(module_name),
        "entrypoint_source_sha256": hashlib.sha256(
            entrypoint_source.encode("utf-8")
        ).hexdigest(),
        "module_source_sha256": hashlib.sha256(
            module_source.encode("utf-8")
        ).hexdigest(),
    }


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


def _builtin_identity(value: Any) -> dict[str, Any]:
    """Stable identity for a named Python/runtime or extension builtin.

    Builtins expose no Python source. Their public module/qualified name is the
    reproducible entrypoint; an owning distribution version and module source
    hash are included when available. This deliberately excludes interpreter
    bytecode and interpreter-minor-version details.
    """
    module_name = _module_name(value, kind="builtin callable")
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(qualname, str) or not qualname:
        raise TypeError("builtin callable has no reproducible qualified name")
    identity: dict[str, Any] = {
        "module": module_name,
        "qualname": qualname,
        "implementation_version": _package_version(module_name),
        "implementation_kind": "builtin",
    }
    try:
        if _is_stdlib_module(module_name):
            raise TypeError("stdlib source is interpreter-version-specific")
        source = _module_source(module_name)
    except TypeError:
        # Language builtins and extension-only modules have no Python source;
        # their stable named API (plus package version, for extensions) is the
        # strongest portable identity available across supported CPython minors.
        identity["source_identity"] = "named-builtin-v1"
    else:
        identity["module_source_sha256"] = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()
    return identity


def _callable_identity(value: Any, seen: set[int]) -> dict[str, Any]:
    function = value.__func__ if inspect.ismethod(value) else value
    identity = _source_identity(function, kind="callable")
    closure = [
        _canonical_parameter(cell.cell_contents, seen)
        for cell in function.__closure__ or ()
    ]
    identity.update(
        {
            "defaults": _canonical_parameter(function.__defaults__, seen),
            "kwdefaults": _canonical_parameter(function.__kwdefaults__, seen),
            "closure": closure,
            # fmt: off
            "bound_to_model": bool(
                inspect.ismethod(value)
                and id(getattr(value, "__self__", None)) in seen
            ),
            # fmt: on
        }
    )
    return identity


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
    if isinstance(value, type) and issubclass(value, Enum):
        return {"enum_class": _source_identity(value, kind="enum class")}
    if isinstance(value, type):
        return {"class": _source_identity(value, kind="class")}
    if isinstance(value, Enum):
        cls = type(value)
        return {
            "enum": _source_identity(cls, kind="enum class"),
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
        identity = id(value)
        if identity in seen:
            raise ValueError("cyclic provenance configuration is not supported")
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
    if inspect.isfunction(value) or inspect.ismethod(value):
        return {"callable": _callable_identity(value, seen)}
    if inspect.isbuiltin(value):
        return {"callable": _builtin_identity(value)}
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
                "object": _source_identity(cls, kind="object class"),
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
    cls_module = _module_name(cls, kind="model class")
    apply = getattr(model, "apply", None)
    descriptor: dict[str, Any] = {
        "type": f"{cls_module}.{cls.__qualname__}",
        "implementation_version": _package_version(cls_module),
        "implementation_class": _source_identity(cls, kind="model class"),
    }
    if callable(apply):
        descriptor["implementation_entrypoint"] = "apply"
        descriptor["implementation"] = _callable_identity(apply, {id(model)})
    else:
        descriptor["implementation_entrypoint"] = "class"
        descriptor["implementation"] = _source_identity(cls, kind="model class")
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
    untracked = _relevant_untracked_source()
    if untracked:
        raise RuntimeError(
            "native adam-core build identity cannot cover untracked source: "
            + ", ".join(untracked)
        )
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
