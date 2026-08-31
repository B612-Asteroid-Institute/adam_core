#!/usr/bin/env python3
"""Prepare only missing, checksum-verified files for an exact PyPI release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_release(
    project: str,
    version: str,
    *,
    registry: str = "https://pypi.org",
) -> dict[str, Any] | None:
    project_path = urllib.parse.quote(project, safe="")
    version_path = urllib.parse.quote(version, safe="")
    request = urllib.request.Request(
        f"{registry.rstrip('/')}/pypi/{project_path}/{version_path}/json",
        headers={
            "Accept": "application/json",
            "User-Agent": "adam-core-release-automation/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"PyPI request failed ({error.code}): {detail}") from error
    if not isinstance(payload, dict):
        raise TypeError("PyPI release response must be an object")
    return payload


def publication_state(
    dist: Path,
    project: str,
    version: str,
    *,
    registry: str = "https://pypi.org",
    expected_count: int | None = None,
) -> dict[str, Any]:
    wheels = sorted(dist.glob("*.whl"))
    if expected_count is not None and len(wheels) != expected_count:
        raise ValueError(f"expected {expected_count} wheels, found {len(wheels)}")
    local = {wheel.name: sha256(wheel) for wheel in wheels}
    if len(local) != len(wheels):
        raise ValueError("duplicate local wheel filenames")

    release = fetch_release(project, version, registry=registry)
    public_files = [] if release is None else release.get("urls", [])
    if not isinstance(public_files, list):
        raise TypeError("PyPI release urls must be a list")

    public: dict[str, str] = {}
    yanked: list[str] = []
    for item in public_files:
        if not isinstance(item, dict):
            raise TypeError("PyPI release file entry must be an object")
        filename = item.get("filename")
        digest = item.get("digests", {}).get("sha256")
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise TypeError(f"incomplete PyPI file identity: {item}")
        if filename in public:
            raise ValueError(f"duplicate public filename: {filename}")
        public[filename] = digest
        if item.get("yanked", False):
            yanked.append(filename)

    unexpected = sorted(set(public) - set(local))
    mismatched = sorted(
        filename
        for filename in set(public) & set(local)
        if public[filename] != local[filename]
    )
    if unexpected or mismatched or yanked:
        raise ValueError(
            "public release does not match the accepted artifact set: "
            f"unexpected={unexpected} mismatched={mismatched} yanked={sorted(yanked)}"
        )

    existing = sorted(public)
    missing = sorted(set(local) - set(public))
    return {
        "project": project,
        "version": version,
        "expected_count": len(local),
        "existing": existing,
        "missing": missing,
        "complete": not missing,
        "sha256": local,
    }


def prepare_upload_directory(dist: Path, upload_dir: Path, missing: list[str]) -> None:
    if upload_dir.exists():
        if any(upload_dir.iterdir()):
            raise ValueError(f"upload directory is not empty: {upload_dir}")
    else:
        upload_dir.mkdir(parents=True)
    for filename in missing:
        shutil.copy2(dist / filename, upload_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--registry", default="https://pypi.org")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--upload-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--attempts", type=int, default=60)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    state: dict[str, Any] | None = None
    for attempt in range(args.attempts):
        state = publication_state(
            args.dist,
            args.project,
            args.version,
            registry=args.registry,
            expected_count=args.expected_count,
        )
        if not args.require_complete or state["complete"]:
            break
        if attempt + 1 < args.attempts:
            time.sleep(args.interval)
    assert state is not None
    if args.require_complete and not state["complete"]:
        raise SystemExit(
            f"PyPI release is incomplete after retries: {state['missing']}"
        )

    if args.upload_dir is not None:
        prepare_upload_directory(args.dist, args.upload_dir, state["missing"])
    encoded = json.dumps(state, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
