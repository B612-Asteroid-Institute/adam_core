"""Generate Python adam-assist public-semantics parity fixtures.

The RM-STANDALONE-007B acceptance target is Python
``adam_assist.ASSISTPropagator`` public behavior, not the current Rust
``assist-rs`` spike contract. This generator freezes small deterministic
propagation and ephemeris cases that exercise caller-facing origin/frame/time
normalization plus variant metadata preservation. Rust-backed adapters should
match these fixtures within explicitly documented tolerances before benchmark
claims are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from adam_assist import ASSISTPropagator
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.transform import transform_coordinates
from adam_core.observers.observers import Observers
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.variants import VariantEphemeris, VariantOrbits
from adam_core.time import Timestamp
from jpl_small_bodies_de441_n16 import de441_n16
from naif_de440 import de440

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "migration"
    / "artifacts"
    / "assist_public_semantics_fixture_2026-05-20.json"
)
PACKAGE_NAMES = ("adam-assist", "assist", "rebound", "adam-core")


def _package_version(package_name: str) -> str:
    return metadata.version(package_name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _kernel_metadata(
    path_value: str, *, label: str, include_sha256: bool
) -> dict[str, Any]:
    path = Path(path_value)
    data: dict[str, Any] = {
        "label": label,
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
    }
    if include_sha256:
        data["sha256"] = _sha256(path)
    return data


def _json_number(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return None
    if isinstance(value, (int, str)):
        return value
    return value


def _array_to_json(values: np.ndarray) -> list[Any]:
    array = np.asarray(values)
    if array.ndim == 1:
        return [_json_number(value) for value in array.tolist()]
    return [_array_to_json(row) for row in array]


def _arrow_to_json(values: pa.Array | pa.ChunkedArray) -> list[Any]:
    return [_json_number(value) for value in values.to_pylist()]


def _time_to_json(times: Timestamp) -> dict[str, Any]:
    return {
        "scale": times.scale,
        "days": _arrow_to_json(times.days),
        "nanos": _arrow_to_json(times.nanos),
        "mjd": _array_to_json(times.mjd().to_numpy(zero_copy_only=False)),
    }


def _origin_codes_to_json(origin: Origin) -> list[Any]:
    return _arrow_to_json(origin.code)


def _cartesian_to_json(coordinates: CartesianCoordinates) -> dict[str, Any]:
    data: dict[str, Any] = {
        "representation": "cartesian",
        "frame": coordinates.frame,
        "origin_codes": _origin_codes_to_json(coordinates.origin),
        "time": _time_to_json(coordinates.time),
        "values": _array_to_json(coordinates.values),
    }
    if not coordinates.covariance.is_all_nan():
        data["covariance"] = _array_to_json(coordinates.covariance.to_matrix())
    return data


def _spherical_to_json(coordinates: Any) -> dict[str, Any]:
    return {
        "representation": "spherical",
        "frame": coordinates.frame,
        "origin_codes": _origin_codes_to_json(coordinates.origin),
        "time": _time_to_json(coordinates.time),
        "values": _array_to_json(coordinates.values),
    }


def _orbits_to_json(orbits: Orbits | VariantOrbits) -> dict[str, Any]:
    data: dict[str, Any] = {
        "table_type": type(orbits).__name__,
        "orbit_id": _arrow_to_json(orbits.orbit_id),
        "object_id": _arrow_to_json(orbits.object_id),
        "coordinates": _cartesian_to_json(orbits.coordinates),
    }
    if isinstance(orbits, VariantOrbits):
        data.update(
            {
                "variant_id": _arrow_to_json(orbits.variant_id),
                "weights": _arrow_to_json(orbits.weights),
                "weights_cov": _arrow_to_json(orbits.weights_cov),
            }
        )
    return data


def _ephemeris_to_json(ephemeris: Ephemeris | VariantEphemeris) -> dict[str, Any]:
    data: dict[str, Any] = {
        "table_type": type(ephemeris).__name__,
        "orbit_id": _arrow_to_json(ephemeris.orbit_id),
        "object_id": _arrow_to_json(ephemeris.object_id),
        "coordinates": _spherical_to_json(ephemeris.coordinates),
        "aberrated_coordinates": _cartesian_to_json(ephemeris.aberrated_coordinates),
        "light_time": _arrow_to_json(ephemeris.light_time),
        "alpha": _arrow_to_json(ephemeris.alpha),
        "predicted_magnitude_v": _arrow_to_json(ephemeris.predicted_magnitude_v),
    }
    if isinstance(ephemeris, VariantEphemeris):
        data.update(
            {
                "variant_id": _arrow_to_json(ephemeris.variant_id),
                "weights": _arrow_to_json(ephemeris.weights),
                "weights_cov": _arrow_to_json(ephemeris.weights_cov),
            }
        )
    return data


def _base_sun_ecliptic_orbits() -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=["fixture-a", "fixture-b"],
        object_id=["fixture-a", "fixture-b"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.05, 1.35],
            y=[0.02, -0.08],
            z=[0.01, 0.03],
            vx=[-0.0005, 0.0010],
            vy=[0.0165, 0.0140],
            vz=[0.0002, -0.0001],
            time=Timestamp.from_mjd([60000.0, 60000.25], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )


def _orbits_for_public_input(
    *,
    origin_out: str,
    frame_out: str,
    time_scale: str,
) -> Orbits:
    orbits = _base_sun_ecliptic_orbits()
    coordinates = transform_coordinates(
        orbits.coordinates,
        CartesianCoordinates,
        origin_out=origin_out,
        frame_out=frame_out,
    )
    coordinates = coordinates.set_column("time", coordinates.time.rescale(time_scale))
    return orbits.set_column("coordinates", coordinates)


def _variant_orbits() -> VariantOrbits:
    return VariantOrbits.from_kwargs(
        orbit_id=["fixture-a", "fixture-a"],
        object_id=["fixture-a", "fixture-a"],
        variant_id=["v0", "v1"],
        weights=[0.25, 0.75],
        weights_cov=[0.5, 0.5],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.05, 1.051],
            y=[0.02, 0.021],
            z=[0.01, 0.011],
            vx=[-0.0005, -0.0004],
            vy=[0.0165, 0.0164],
            vz=[0.0002, 0.00021],
            time=Timestamp.from_mjd([60000.0, 60000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )


def _propagation_case(
    propagator: ASSISTPropagator,
    *,
    case_id: str,
    description: str,
    orbits: Orbits | VariantOrbits,
    times: Timestamp,
) -> dict[str, Any]:
    output = propagator.propagate_orbits(
        orbits,
        times,
        covariance=False,
        max_processes=1,
        chunk_size=1,
    )
    return {
        "case_id": case_id,
        "description": description,
        "operation": "propagate_orbits",
        "options": {"covariance": False, "max_processes": 1, "chunk_size": 1},
        "input_orbits": _orbits_to_json(orbits),
        "target_times": _time_to_json(times),
        "output_orbits": _orbits_to_json(output),
    }


def _ephemeris_case(propagator: ASSISTPropagator) -> dict[str, Any]:
    orbits = _base_sun_ecliptic_orbits()
    observers = Observers.from_codes(
        ["500", "X05", "500"],
        Timestamp.from_mjd([60001.0, 60000.5, 60001.5], scale="utc"),
    )
    ephemeris = propagator.generate_ephemeris(
        orbits,
        observers,
        covariance=False,
        max_processes=1,
        chunk_size=1,
        predict_magnitudes=False,
        predict_phase_angle=True,
    )
    return {
        "case_id": "ephemeris_mixed_observers_utc_output",
        "description": (
            "Default EphemerisMixin semantics over Python ASSISTPropagator: sorted "
            "observer epochs/codes, topocentric equatorial spherical output, UTC output "
            "times, observer-code origins, light-time, aberrated SSB/ecliptic Cartesian "
            "states, and optional phase angle without H/G magnitudes."
        ),
        "operation": "generate_ephemeris",
        "options": {
            "covariance": False,
            "max_processes": 1,
            "chunk_size": 1,
            "predict_magnitudes": False,
            "predict_phase_angle": True,
        },
        "input_orbits": _orbits_to_json(orbits),
        "observers": {
            "code": _arrow_to_json(observers.code),
            "coordinates": _cartesian_to_json(observers.coordinates),
        },
        "output_ephemeris": _ephemeris_to_json(ephemeris),
    }


def build_fixture(*, include_kernel_sha256: bool = True) -> dict[str, Any]:
    propagator = ASSISTPropagator()
    propagation_cases = [
        _propagation_case(
            propagator,
            case_id="sun_ecliptic_tdb_input_tdb_targets",
            description=(
                "Baseline heliocentric ecliptic Cartesian input. Output must preserve "
                "input SUN/ecliptic origin/frame and requested TDB target scale."
            ),
            orbits=_orbits_for_public_input(
                origin_out=OriginCodes.SUN,
                frame_out="ecliptic",
                time_scale="tdb",
            ),
            times=Timestamp.from_mjd([60002.0, 60001.0], scale="tdb"),
        ),
        _propagation_case(
            propagator,
            case_id="ssb_equatorial_tdb_input_tdb_targets",
            description=(
                "Python ASSISTPropagator accepts SSB/equatorial input and preserves "
                "that public output origin/frame after internal ASSIST normalization."
            ),
            orbits=_orbits_for_public_input(
                origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
                frame_out="equatorial",
                time_scale="tdb",
            ),
            times=Timestamp.from_mjd([60002.0, 60001.0], scale="tdb"),
        ),
        _propagation_case(
            propagator,
            case_id="sun_ecliptic_utc_input_utc_targets",
            description=(
                "Python ASSISTPropagator accepts non-TDB public times by rescaling "
                "internally to TDB and restoring requested UTC output times."
            ),
            orbits=_orbits_for_public_input(
                origin_out=OriginCodes.SUN,
                frame_out="ecliptic",
                time_scale="utc",
            ),
            times=Timestamp.from_mjd([60002.0, 60001.0], scale="utc"),
        ),
        _propagation_case(
            propagator,
            case_id="variant_metadata_tdb_targets",
            description=(
                "VariantOrbits preserve variant_id, weights, weights_cov, row order, "
                "and public output origin/frame/time semantics."
            ),
            orbits=_variant_orbits(),
            times=Timestamp.from_mjd([60000.5, 60001.0], scale="tdb"),
        ),
    ]
    return {
        "fixture_schema_version": 1,
        "fixture_id": "assist_public_semantics_fixture_2026-05-20",
        "acceptance_target": "adam_assist.ASSISTPropagator public semantics",
        "packages": {name: _package_version(name) for name in PACKAGE_NAMES},
        "kernels": [
            _kernel_metadata(
                de440, label="naif_de440", include_sha256=include_kernel_sha256
            ),
            _kernel_metadata(
                de441_n16,
                label="jpl_small_bodies_de441_n16",
                include_sha256=include_kernel_sha256,
            ),
        ],
        "semantics_contract": [
            "Inputs are transformed internally to SSB/equatorial/TDB for Python ASSIST.",
            "propagate_orbits restores the caller-facing input origin and frame.",
            "propagate_orbits restores the caller-requested output time scale.",
            "Output row order is sorted by orbit_id, variant_id when present, then time.",
            "generate_ephemeris emits topocentric equatorial spherical coordinates in UTC.",
            "generate_ephemeris coordinate origins are observer codes and aberrated states are SSB/ecliptic Cartesian.",
        ],
        "propagation_cases": propagation_cases,
        "ephemeris_cases": [_ephemeris_case(propagator)],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-kernel-sha256",
        action="store_true",
        help="Write kernel sizes but skip SHA256 hashes for quick local smoke runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    fixture = build_fixture(include_kernel_sha256=not args.skip_kernel_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
