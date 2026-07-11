"""Generate/check the complete adam-core-owned public Python surface manifest.

This deliberately inventories more than the parity benchmark registry: public
module functions/classes, adam-core-owned class constructors/methods/properties,
and public constants. Generic inherited quivr behavior is recorded separately
by the domain audits because it is not defined in adam-core source.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "adam_core"
DEFAULT_OUTPUT = ROOT / "migration" / "public_surface" / "manifest.json"

_EXCLUDED_PARTS = {"tests", "testdata", "_rust", "__pycache__"}
_PLOT_TOKENS = {"plot", "plots", "preview", "visualize", "visualization"}
_RUST_TOKENS = (
    "_rust",
    "_rust_native",
    "rust_backend",
    "_native",
    "arrow_bridge",
)
_IO_TOKENS = (
    "requests",
    "httpx",
    "urllib",
    "open(",
    ".read_text(",
    ".write_text(",
    ".to_parquet(",
    ".from_parquet(",
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["adam_core", *parts])


def _domain(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else "package"


def _static_all(tree: ast.Module) -> set[str] | None:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)) and all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in value.elts
        ):
            return {str(element.value) for element in value.elts}
    return None


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    result: list[str] = []
    for decorator in node.decorator_list:
        try:
            result.append(ast.unparse(decorator))
        except Exception:
            result.append(type(decorator).__name__)
    return result


def _function_kind(
    node: ast.FunctionDef | ast.AsyncFunctionDef, *, method: bool
) -> str:
    decorators = _decorators(node)
    if method and node.name == "__init__":
        return "constructor"
    if any(name.endswith("property") for name in decorators):
        return "property"
    if any(name.endswith("classmethod") for name in decorators):
        return "classmethod"
    if any(name.endswith("staticmethod") for name in decorators):
        return "staticmethod"
    return "method" if method else "function"


def _source_segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ""


def _signals(segment: str, *, module: str, qualname: str) -> dict[str, bool]:
    lowered = f"{module}.{qualname}".lower()
    return {
        "plotting": any(token in lowered for token in _PLOT_TOKENS),
        "rust_reference": any(token in segment for token in _RUST_TOKENS),
        "python_loop": any(f"{token} " in segment for token in ("for", "while")),
        "numpy_reference": "np." in segment or "numpy" in segment,
        "pyarrow_reference": "pyarrow" in segment
        or "pc." in segment
        or "pa." in segment,
        "quivr_reference": "quivr" in segment or "qv." in segment,
        "external_io_reference": any(token in segment for token in _IO_TOKENS),
    }


def _symbol(
    *,
    module: str,
    path: Path,
    qualname: str,
    name: str,
    kind: str,
    line: int,
    explicit_export: bool | None,
    segment: str,
) -> dict[str, Any]:
    plotting = _signals(segment, module=module, qualname=qualname)["plotting"]
    return {
        "id": f"{module}:{qualname}",
        "module": module,
        "qualname": qualname,
        "name": name,
        "kind": kind,
        "domain": _domain(module),
        "source": str(path.relative_to(ROOT)),
        "line": line,
        "explicit_export": explicit_export,
        "plotting_exemption_candidate": plotting,
        "signals": _signals(segment, module=module, qualname=qualname),
        "review_status": "unreviewed",
        "implementation_class": "unreviewed",
        "rust_entrypoint": None,
        "parity_coverage": None,
        "native_timing": None,
        "tracking_issue": "personal-cmy.37",
    }


def _iter_source_files() -> Iterable[Path]:
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.name.startswith("test_"):
            continue
        if path.name.startswith("_") and path.name != "__init__.py":
            continue
        yield path


def collect() -> dict[str, Any]:
    symbols: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for path in _iter_source_files():
        source = path.read_text()
        module = _module_name(path)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            parse_errors.append(
                {"source": str(path.relative_to(ROOT)), "error": str(exc)}
            )
            continue
        module_all = _static_all(tree)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                exported = node.name in module_all if module_all is not None else None
                symbols.append(
                    _symbol(
                        module=module,
                        path=path,
                        qualname=node.name,
                        name=node.name,
                        kind=_function_kind(node, method=False),
                        line=node.lineno,
                        explicit_export=exported,
                        segment=_source_segment(source, node),
                    )
                )
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                exported = node.name in module_all if module_all is not None else None
                symbols.append(
                    _symbol(
                        module=module,
                        path=path,
                        qualname=node.name,
                        name=node.name,
                        kind="class",
                        line=node.lineno,
                        explicit_export=exported,
                        segment=_source_segment(source, node),
                    )
                )
                for child in node.body:
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if child.name.startswith("_") and child.name != "__init__":
                        continue
                    # A property setter/deleter repeats the public property name
                    # in the AST but does not add another callable surface.
                    decorators = _decorators(child)
                    if any(
                        name.endswith((".setter", ".deleter")) for name in decorators
                    ):
                        continue
                    symbols.append(
                        _symbol(
                            module=module,
                            path=path,
                            qualname=f"{node.name}.{child.name}",
                            name=child.name,
                            kind=_function_kind(child, method=True),
                            line=child.lineno,
                            explicit_export=None,
                            segment=_source_segment(source, child),
                        )
                    )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = target.id
                    if name.startswith("_") or not name.isupper():
                        continue
                    exported = name in module_all if module_all is not None else None
                    symbols.append(
                        _symbol(
                            module=module,
                            path=path,
                            qualname=name,
                            name=name,
                            kind="constant",
                            line=node.lineno,
                            explicit_export=exported,
                            segment=_source_segment(source, node),
                        )
                    )
    symbols.sort(key=lambda item: item["id"])
    duplicate_ids = sorted(
        symbol_id
        for symbol_id, count in Counter(item["id"] for item in symbols).items()
        if count > 1
    )
    counts = Counter(item["kind"] for item in symbols)
    domains = Counter(item["domain"] for item in symbols)
    return {
        "schema_version": 1,
        "scope": {
            "source_root": "src/adam_core",
            "excludes": sorted(_EXCLUDED_PARTS),
            "rule": (
                "adam-core-owned public top-level functions/classes/constants and "
                "public class constructors/methods/properties; inherited quivr surface "
                "is audited separately"
            ),
            "plotting_policy": "plotting/display may remain Python; all other symbols require review",
        },
        "summary": {
            "symbols": len(symbols),
            "by_kind": dict(sorted(counts.items())),
            "by_domain": dict(sorted(domains.items())),
            "plotting_exemption_candidates": sum(
                bool(item["plotting_exemption_candidate"]) for item in symbols
            ),
            "unreviewed": sum(
                item["review_status"] == "unreviewed" for item in symbols
            ),
        },
        "parse_errors": parse_errors,
        "duplicate_ids": duplicate_ids,
        "symbols": symbols,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    manifest = collect()
    rendered = json.dumps(manifest, indent=2) + "\n"
    if manifest["parse_errors"] or manifest["duplicate_ids"]:
        print(
            json.dumps(
                {
                    "parse_errors": manifest["parse_errors"],
                    "duplicate_ids": manifest["duplicate_ids"],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            print(
                f"public surface manifest is stale: run {Path(__file__).relative_to(ROOT)}",
                file=sys.stderr,
            )
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(
        f"wrote {args.output} ({manifest['summary']['symbols']} symbols; "
        f"{manifest['summary']['plotting_exemption_candidates']} plotting candidates)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
