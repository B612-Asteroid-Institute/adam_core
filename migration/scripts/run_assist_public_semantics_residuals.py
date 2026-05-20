"""Run the GPL assist-rs public-semantics live residual fixture.

This script resolves the installed NAIF/JPL BSP files used by the frozen Python
adam-assist fixture, runs the ignored Rust live fixture test, and records the
measured Rust-vs-Python residuals in a JSON artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "migration/artifacts/assist_public_semantics_residuals_2026-05-20.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "rust/adam_core_rs_assist/Cargo.toml"
PLANETS_PACKAGE = "naif_de440"
PLANETS_FILE = "de440.bsp"
ASTEROIDS_PACKAGE = "jpl_small_bodies_de441_n16"
ASTEROIDS_FILE = "sb441-n16.bsp"
LIVE_TEST_NAME = "live_assist_matches_public_semantics_fixture_propagation_cases"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Residual artifact path to write.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="adam_core_rs_assist Cargo.toml path.",
    )
    return parser.parse_args(argv)


def resolve_bsp(package: str, file_name: str) -> Path:
    package_root = resources.files(package)
    candidate = package_root / file_name
    if not candidate.is_file():
        raise FileNotFoundError(f"{package} does not contain {file_name}")
    return Path(str(candidate))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cargo_command(manifest_path: Path) -> list[str]:
    return [
        "cargo",
        "test",
        "--manifest-path",
        str(manifest_path),
        LIVE_TEST_NAME,
        "--",
        "--ignored",
        "--nocapture",
    ]


def run_live_fixture(
    command: Sequence[str], planets_path: Path, asteroids_path: Path, output_path: Path
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "ADAM_CORE_RS_ASSIST_PLANETS_PATH": str(planets_path),
            "ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH": str(asteroids_path),
            "ADAM_CORE_RS_ASSIST_RESIDUALS_PATH": str(output_path),
            "CARGO_NET_GIT_FETCH_WITH_CLI": "true",
        }
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def enrich_artifact(
    output_path: Path,
    command: Sequence[str],
    planets_path: Path,
    asteroids_path: Path,
) -> dict[str, object]:
    with output_path.open("r", encoding="utf-8") as file:
        artifact = json.load(file)

    artifact["generated_at_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    artifact["runner"] = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "command": list(command),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    artifact["resolved_kernel_paths"] = {
        "planets": str(planets_path),
        "asteroids": str(asteroids_path),
    }
    artifact["observed_kernel_sha256"] = {
        "planets": sha256_file(planets_path),
        "asteroids": sha256_file(asteroids_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2, sort_keys=True)
        file.write("\n")
    return artifact


def print_summary(output_path: Path, artifact: dict[str, object]) -> None:
    global_max = artifact["global_max"]
    if not isinstance(global_max, dict):
        raise TypeError("residual artifact global_max must be an object")
    print(f"Wrote {output_path}")
    print(
        "max position: "
        f"{global_max['position_abs_au']:.6e} AU "
        f"({global_max['position_abs_m']:.6e} m)"
    )
    print(
        "max velocity: "
        f"{global_max['velocity_abs_au_per_day']:.6e} AU/day "
        f"({global_max['velocity_abs_m_per_s']:.6e} m/s)"
    )
    print(f"max time: {global_max['time_abs_ns']} ns")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output.resolve()
    manifest_path = args.manifest_path.resolve()
    planets_path = resolve_bsp(PLANETS_PACKAGE, PLANETS_FILE)
    asteroids_path = resolve_bsp(ASTEROIDS_PACKAGE, ASTEROIDS_FILE)
    command = cargo_command(manifest_path)
    run_live_fixture(command, planets_path, asteroids_path, output_path)
    artifact = enrich_artifact(output_path, command, planets_path, asteroids_path)
    print_summary(output_path, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
