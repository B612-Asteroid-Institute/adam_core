"""Generate exhaustive transform_coordinates branch parity fixture.

Run with the LEGACY baseline interpreter from the migration repo:

    .legacy-venv/bin/python migration/scripts/generate_transform_coordinates_branch_fixture.py

The fixture freezes legacy public ``adam_core.coordinates.transform_coordinates``
outputs and errors for a deterministic branch matrix. Unlike the randomized
parity fuzz matrix, this is intended to cover every public dispatcher branch:
identity returns, Rust-supported fused paths, explicit fallback paths,
origin/frame/representation combinations, mixed-origin and observatory-origin
translations, geodetic output, covariance/no-covariance paths, and public input
validation errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.cometary import CometaryCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.geodetics import GeodeticCoordinates
from adam_core.coordinates.keplerian import KeplerianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.coordinates.transform import transform_coordinates
from adam_core.time import Timestamp

REPO_ROOT = Path(__file__).resolve().parents[2]

REPRESENTATIONS = {
    "cartesian": CartesianCoordinates,
    "spherical": SphericalCoordinates,
    "keplerian": KeplerianCoordinates,
    "cometary": CometaryCoordinates,
    "geodetic": GeodeticCoordinates,
}

# Deterministic values intentionally avoid singularities but keep branch-specific
# units realistic enough for SPICE/origin/frame transforms.
CART_HELIO = [
    [1.20, 0.15, 0.05, -0.0010, 0.0170, 0.0002],
    [1.55, -0.35, 0.20, 0.0040, 0.0120, -0.0003],
]
CART_GEO = [
    [4.10e-5, 1.00e-6, -2.00e-6, 1.0e-7, 2.0e-7, -1.0e-7],
    [-3.90e-5, 2.50e-6, 1.50e-6, -2.0e-7, 1.0e-7, 1.5e-7],
]
SPHERICAL = [
    [1.20, 20.0, 5.0, 0.0010, 0.0020, -0.0030],
    [1.55, 310.0, -7.0, -0.0020, 0.0015, 0.0025],
]
SPHERICAL_ITRF93 = [
    [4.15e-5, 15.0, -25.0, 1.0e-7, 2.0e-5, -3.0e-5],
    [4.30e-5, 250.0, 35.0, -2.0e-7, -1.0e-5, 2.0e-5],
]
KEPLERIAN = [
    [1.30, 0.10, 5.0, 80.0, 30.0, 40.0],
    [2.10, 0.20, 10.0, 120.0, 45.0, 90.0],
]
COMETARY = [
    [0.90, 0.20, 7.0, 80.0, 40.0, 59900.0],
    [1.10, 0.40, 11.0, 90.0, 60.0, 59850.0],
]
GEODETIC = [
    [0.0, 250.0, -30.0, 0.0, 0.0, 0.0],
    [1.0e-6, 120.0, 35.0, 0.0, 0.0, 0.0],
]
TIMES = [60000.0, 60000.25]
ITRF_TIMES = [60000.0, 60100.0]


def _case(
    name: str,
    representation_in: str,
    values: list[list[float]],
    *,
    frame_in: str = "ecliptic",
    origin_in: str | list[str] = "SUN",
    representation_out: str | None = None,
    frame_out: str | None = None,
    origin_out: str | None = None,
    covariance: str = "nan",
    time_mjd: list[float] | None = None,
    expect_rust_support: bool | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "value",
        "representation_in": representation_in,
        "values": values,
        "time_mjd": time_mjd or TIMES[: len(values)],
        "frame_in": frame_in,
        "origin_in": origin_in,
        "representation_out": representation_out,
        "frame_out": frame_out,
        "origin_out": origin_out,
        "covariance": covariance,
        "expect_rust_support": expect_rust_support,
        "note": note,
    }


VALUE_CASES: list[dict[str, Any]] = [
    _case(
        "identity_cartesian_noop",
        "cartesian",
        CART_HELIO,
        expect_rust_support=None,
        note="Early return when representation, frame, and origin are unchanged.",
    ),
    _case(
        "rust_cart_ec_to_sph_eq",
        "cartesian",
        CART_HELIO,
        representation_out="spherical",
        frame_out="equatorial",
        expect_rust_support=True,
        note="Cartesian input, non-Cartesian output, constant ecliptic->equatorial frame rotation.",
    ),
    _case(
        "rust_sph_itrf93_to_geodetic",
        "spherical",
        SPHERICAL_ITRF93,
        frame_in="itrf93",
        origin_in="EARTH",
        representation_out="geodetic",
        frame_out="itrf93",
        origin_out="EARTH",
        time_mjd=ITRF_TIMES,
        expect_rust_support=True,
        note="Geodetic output branch with Earth-centered ITRF93 input.",
    ),
    _case(
        "rust_kep_ec_to_com_ec",
        "keplerian",
        KEPLERIAN,
        representation_out="cometary",
        frame_out="ecliptic",
        expect_rust_support=True,
        note="Keplerian input path with t0/mu supplied to the Rust dispatcher.",
    ),
    _case(
        "rust_com_ec_to_cart_ec",
        "cometary",
        COMETARY,
        representation_out="cartesian",
        frame_out="ecliptic",
        expect_rust_support=True,
        note="Cometary input to Cartesian output; not the pure Cartesian frame-only fallback.",
    ),
    _case(
        "rust_noncart_origin_change",
        "keplerian",
        KEPLERIAN,
        representation_out="spherical",
        frame_out="ecliptic",
        origin_out="EARTH",
        time_mjd=ITRF_TIMES,
        expect_rust_support=True,
        note="Non-Cartesian origin change uses synthetic Cartesian metadata for translation vectors.",
    ),
    _case(
        "rust_cart_origin_change",
        "cartesian",
        CART_HELIO,
        representation_out="spherical",
        frame_out="ecliptic",
        origin_out="EARTH",
        time_mjd=ITRF_TIMES,
        expect_rust_support=True,
        note="Cartesian origin change fused with representation conversion.",
    ),
    _case(
        "rust_itrf93_no_origin_change",
        "cartesian",
        CART_GEO,
        frame_in="ecliptic",
        origin_in="EARTH",
        representation_out="spherical",
        frame_out="itrf93",
        time_mjd=ITRF_TIMES,
        expect_rust_support=True,
        note="ITRF93 time-varying rotation followed by identity-frame representation conversion.",
    ),
    _case(
        "rust_itrf93_with_origin_change",
        "cartesian",
        CART_HELIO,
        frame_in="ecliptic",
        origin_in="SUN",
        representation_out="spherical",
        frame_out="itrf93",
        origin_out="EARTH",
        time_mjd=ITRF_TIMES,
        expect_rust_support=True,
        note="Origin translation plus ITRF93 time-varying rotation branch.",
    ),
    _case(
        "rust_covariance_constant_frame",
        "cartesian",
        CART_HELIO,
        representation_out="spherical",
        frame_out="equatorial",
        covariance="finite",
        expect_rust_support=True,
        note="Finite covariance path through Rust forward-mode AD.",
    ),
    _case(
        "rust_covariance_itrf93",
        "cartesian",
        CART_GEO,
        frame_in="ecliptic",
        origin_in="EARTH",
        representation_out="spherical",
        frame_out="itrf93",
        covariance="finite",
        time_mjd=ITRF_TIMES,
        expect_rust_support=True,
        note="Finite covariance with public ITRF93 dispatcher branch.",
    ),
    _case(
        "fallback_cartesian_frame_only",
        "cartesian",
        CART_HELIO,
        representation_out="cartesian",
        frame_out="equatorial",
        expect_rust_support=False,
        note="Intentional fallback: pure Cartesian 6x6 frame rotation is faster in Python/NumPy.",
    ),
    _case(
        "fallback_cartesian_origin_only",
        "cartesian",
        CART_HELIO,
        representation_out="cartesian",
        frame_out="ecliptic",
        origin_out="EARTH",
        time_mjd=ITRF_TIMES,
        expect_rust_support=False,
        note="Intentional fallback: Cartesian output with origin change adds no fusion win.",
    ),
    _case(
        "fallback_noncart_itrf93_input",
        "cometary",
        COMETARY,
        representation_out="spherical",
        frame_out="itrf93",
        time_mjd=ITRF_TIMES,
        expect_rust_support=False,
        note="Non-Cartesian input into ITRF93 frame change must route through Cartesian fallback.",
    ),
    _case(
        "mixed_origin_preserve",
        "cartesian",
        CART_HELIO,
        origin_in=["SUN", "EARTH"],
        representation_out="spherical",
        frame_out="ecliptic",
        time_mjd=ITRF_TIMES,
        expect_rust_support=True,
        note="Mixed input origins with no requested origin_out preserve per-row origins.",
    ),
    _case(
        "mixed_origin_to_single_origin",
        "cartesian",
        CART_HELIO,
        origin_in=["SUN", "EARTH"],
        representation_out="spherical",
        frame_out="ecliptic",
        origin_out="SOLAR_SYSTEM_BARYCENTER",
        time_mjd=ITRF_TIMES,
        expect_rust_support=True,
        note="Mixed input origins translated to a single target origin.",
    ),
    _case(
        "observatory_origin_to_sun",
        "cartesian",
        [[1.0e-6, 2.0e-6, -1.0e-6, 0.0, 0.0, 0.0]],
        origin_in="X05",
        representation_out="spherical",
        frame_out="ecliptic",
        origin_out="SUN",
        time_mjd=[60000.0],
        expect_rust_support=True,
        note="MPC observatory-origin translation branch in _resolve_origin_translation_vectors.",
    ),
    _case(
        "geodetic_identity_noop",
        "geodetic",
        GEODETIC,
        frame_in="itrf93",
        origin_in="EARTH",
        expect_rust_support=None,
        note="Geodetic input is allowed for identity/no-op returns.",
    ),
]

ERROR_CASES: list[dict[str, Any]] = [
    {
        "name": "error_unsupported_coordinate_type",
        "kind": "unsupported_coordinate_type",
        "note": "Initial TypeError branch for non-coordinate input.",
    },
    {
        "name": "error_invalid_frame_out",
        "kind": "invalid_frame_out",
        "note": "frame_out validation branch.",
    },
    {
        "name": "error_invalid_origin_out_type",
        "kind": "invalid_origin_out_type",
        "note": "origin_out type validation branch.",
    },
    {
        "name": "error_invalid_representation_out",
        "kind": "invalid_representation_out",
        "note": "representation_out validation branch.",
    },
    {
        "name": "error_geodetic_non_identity_transform",
        "kind": "geodetic_non_identity_transform",
        "note": "Geodetic input is not implemented for non-identity transforms and must fail loudly.",
    },
]

CASES = VALUE_CASES + ERROR_CASES


def _origin_list(origin_in: str | list[str], n: int) -> list[str]:
    if isinstance(origin_in, list):
        if len(origin_in) != n:
            raise ValueError("origin_in list length must match case row count")
        return origin_in
    return [origin_in] * n


def _finite_covariance(n: int) -> np.ndarray:
    base = np.diag([1e-8, 2e-8, 3e-8, 1e-10, 2e-10, 3e-10])
    out = np.empty((n, 6, 6), dtype=np.float64)
    for i in range(n):
        out[i] = base * (1.0 + 0.1 * i)
        out[i, 0, 1] = out[i, 1, 0] = 1e-10 * (i + 1)
        out[i, 3, 4] = out[i, 4, 3] = 1e-12 * (i + 1)
    return out


def _covariance(spec: dict[str, Any], n: int) -> CoordinateCovariances:
    if spec.get("covariance") == "finite":
        return CoordinateCovariances.from_matrix(_finite_covariance(n))
    return CoordinateCovariances.nulls(n)


def build_coordinates(spec: dict[str, Any]) -> Any:
    values = np.asarray(spec["values"], dtype=np.float64)
    n = values.shape[0]
    time = Timestamp.from_mjd(
        np.asarray(spec["time_mjd"], dtype=np.float64), scale="tdb"
    )
    origin = Origin.from_kwargs(code=_origin_list(spec["origin_in"], n))
    covariance = _covariance(spec, n)
    frame = str(spec["frame_in"])
    representation_in = str(spec["representation_in"])
    if representation_in == "cartesian":
        return CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=time,
            covariance=covariance,
            origin=origin,
            frame=frame,
        )
    if representation_in == "spherical":
        return SphericalCoordinates.from_kwargs(
            rho=values[:, 0],
            lon=values[:, 1],
            lat=values[:, 2],
            vrho=values[:, 3],
            vlon=values[:, 4],
            vlat=values[:, 5],
            time=time,
            covariance=covariance,
            origin=origin,
            frame=frame,
        )
    if representation_in == "keplerian":
        return KeplerianCoordinates.from_kwargs(
            a=values[:, 0],
            e=values[:, 1],
            i=values[:, 2],
            raan=values[:, 3],
            ap=values[:, 4],
            M=values[:, 5],
            time=time,
            covariance=covariance,
            origin=origin,
            frame=frame,
        )
    if representation_in == "cometary":
        return CometaryCoordinates.from_kwargs(
            q=values[:, 0],
            e=values[:, 1],
            i=values[:, 2],
            raan=values[:, 3],
            ap=values[:, 4],
            tp=values[:, 5],
            time=time,
            covariance=covariance,
            origin=origin,
            frame=frame,
        )
    if representation_in == "geodetic":
        return GeodeticCoordinates.from_kwargs(
            alt=values[:, 0],
            lon=values[:, 1],
            lat=values[:, 2],
            vup=values[:, 3],
            veast=values[:, 4],
            vnorth=values[:, 5],
            time=time,
            covariance=covariance,
            origin=origin,
            frame=frame,
        )
    raise ValueError(f"unsupported representation_in: {representation_in}")


def transform_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    representation_out = spec.get("representation_out")
    if representation_out is not None:
        kwargs["representation_out"] = REPRESENTATIONS[str(representation_out)]
    frame_out = spec.get("frame_out")
    if frame_out is not None:
        kwargs["frame_out"] = str(frame_out)
    origin_out = spec.get("origin_out")
    if origin_out is not None:
        kwargs["origin_out"] = OriginCodes[str(origin_out)]
    return kwargs


def flatten_coordinates(coords: Any) -> dict[str, Any]:
    return {
        "type": type(coords).__name__,
        "frame": coords.frame,
        "origin": coords.origin.code.to_pylist(),
        "days": coords.time.days.to_pylist(),
        "nanos": coords.time.nanos.to_pylist(),
        "scale": coords.time.scale,
        "values": np.asarray(coords.values, dtype=np.float64).tolist(),
        "covariance": np.asarray(
            coords.covariance.to_matrix(), dtype=np.float64
        ).tolist(),
    }


def run_case(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        kind = spec["kind"]
        if kind == "value":
            coords = build_coordinates(spec)
            return {
                "ok": True,
                "output": flatten_coordinates(
                    transform_coordinates(coords, **transform_kwargs(spec))
                ),
            }
        if kind == "unsupported_coordinate_type":
            transform_coordinates(object())
        elif kind == "invalid_frame_out":
            transform_coordinates(
                build_coordinates(VALUE_CASES[0]), frame_out="galactic"
            )
        elif kind == "invalid_origin_out_type":
            transform_coordinates(build_coordinates(VALUE_CASES[0]), origin_out="EARTH")
        elif kind == "invalid_representation_out":
            transform_coordinates(
                build_coordinates(VALUE_CASES[0]), representation_out=object
            )
        elif kind == "geodetic_non_identity_transform":
            transform_coordinates(
                build_coordinates(VALUE_CASES[-1]),
                representation_out=SphericalCoordinates,
                frame_out="itrf93",
                origin_out=OriginCodes.EARTH,
            )
        else:
            raise ValueError(f"unsupported case kind: {kind}")
    except Exception as exc:  # noqa: BLE001 - freezing public legacy behavior
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    raise AssertionError(f"case {spec['name']} did not raise")


def build_fixture() -> dict[str, Any]:
    return {
        "schema": "adam_core.transform_coordinates_branch_fixture",
        "version": 1,
        "generated_by": "migration/scripts/generate_transform_coordinates_branch_fixture.py",
        "source_contract": (
            "Legacy adam-core public transform_coordinates dispatcher, executed "
            "in the untouched legacy checkout. Values/errors are compared by "
            "the migration checkout without rerunning legacy."
        ),
        "cases": [
            {
                "name": spec["name"],
                "kind": spec["kind"],
                "note": spec.get("note", ""),
                "expect_rust_support": spec.get("expect_rust_support"),
                "input": spec,
                "legacy": run_case(spec),
            }
            for spec in CASES
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "migration"
        / "artifacts"
        / "transform_coordinates_branch_fixture_2026-07-06.json",
    )
    args = parser.parse_args()
    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1, allow_nan=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
