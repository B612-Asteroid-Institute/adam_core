"""Generate frozen whole-product fixtures for OpenSpace asset creators.

Run with the pinned untouched legacy interpreter:

    .legacy-venv/bin/python migration/scripts/generate_openspace_asset_fixture.py

The fixture covers the two complete products owned by bead
``personal-cmy.37.4.5``: orbital-Kepler's CSV+asset pair and trail-orbit's
Kepler and SPICE asset modes, with all optional rendering fields populated,
identifier sanitization, nullable object IDs, and SPICE resource snippets.
"""

import json
import tempfile
from pathlib import Path

from adam_core.coordinates import CartesianCoordinates, KeplerianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.transform import transform_coordinates
from adam_core.orbits import Orbits
from adam_core.orbits.openspace.assets import (
    create_renderable_orbital_kepler,
    create_renderable_trail_orbit,
)
from adam_core.time import Timestamp

OUT = Path("migration/artifacts/openspace_asset_fixture_2026-07-12.json")


def make_orbits() -> Orbits:
    keplerian = KeplerianCoordinates.from_kwargs(
        a=[1.2, 2.5, 5.5],
        e=[0.05, 0.1, 0.7],
        i=[5.0, 10.5, 30.0],
        raan=[10.0, 80.0, 120.5],
        ap=[20.0, 90.0, 60.75],
        M=[0.25, 45.5, 359.9],
        time=Timestamp.from_kwargs(
            days=[60000, 60123, 60250],
            nanos=[0, 123_456_789, 60_000_000_000_000],
            scale="tdb",
        ),
        origin=Origin.from_kwargs(code=["SUN"] * 3),
        frame="ecliptic",
    )
    cartesian = transform_coordinates(
        keplerian, representation_out=CartesianCoordinates
    )
    return Orbits.from_kwargs(
        orbit_id=["orbit alpha", "orbit,beta", "orbit gamma"],
        object_id=["Object Alpha", None, "Object Gamma"],
        coordinates=cartesian,
    )


def flat(orbits: Orbits) -> dict:
    c = orbits.coordinates
    return {
        "orbit_id": orbits.orbit_id.to_pylist(),
        "object_id": orbits.object_id.to_pylist(),
        "x": c.x.to_pylist(),
        "y": c.y.to_pylist(),
        "z": c.z.to_pylist(),
        "vx": c.vx.to_pylist(),
        "vy": c.vy.to_pylist(),
        "vz": c.vz.to_pylist(),
        "days": c.time.days.to_pylist(),
        "nanos": c.time.nanos.to_pylist(),
        "scale": c.time.scale,
        "frame": c.frame,
        "origin": c.origin.code.to_pylist(),
    }


def directory_contents(path: Path) -> dict[str, str]:
    return {
        item.name: item.read_text() for item in sorted(path.iterdir()) if item.is_file()
    }


def main() -> None:
    orbits = make_orbits()
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        orbital_dir = root / "orbital"
        create_renderable_orbital_kepler(
            orbits,
            str(orbital_dir),
            "Fixture Orbital",
            gui_name="Orbital GUI",
            gui_path="/ADAM/Fixture Orbital",
            color=(0.1, 0.2, 0.3),
            segment_quality=42,
            contiguous_mode=True,
            enable_max_size=False,
            enable_outline=True,
            max_size=12.5,
            outline_color=(0.4, 0.5, 0.6),
            outline_width=2.5,
            point_size_exponent=1.25,
            rendering="PointsTrails",
            render_size=8,
            start_render_idx=1,
            trail_fade=0.75,
            trail_width=0.33,
            dim_in_atmosphere=False,
            enabled=True,
            opacity=0.8,
            render_bin_mode="Opaque",
            tag=["science", "fixture"],
        )
        cases.append(
            {
                "name": "orbital_kepler_all_options",
                "kind": "orbital_kepler",
                "identifier": "Fixture Orbital",
                "kwargs": {
                    "gui_name": "Orbital GUI",
                    "gui_path": "/ADAM/Fixture Orbital",
                    "color": [0.1, 0.2, 0.3],
                    "segment_quality": 42,
                    "contiguous_mode": True,
                    "enable_max_size": False,
                    "enable_outline": True,
                    "max_size": 12.5,
                    "outline_color": [0.4, 0.5, 0.6],
                    "outline_width": 2.5,
                    "point_size_exponent": 1.25,
                    "rendering": "PointsTrails",
                    "render_size": 8,
                    "start_render_idx": 1,
                    "trail_fade": 0.75,
                    "trail_width": 0.33,
                    "dim_in_atmosphere": False,
                    "enabled": True,
                    "opacity": 0.8,
                    "render_bin_mode": "Opaque",
                    "tag": ["science", "fixture"],
                },
                "files": directory_contents(orbital_dir),
            }
        )

        trail_dir = root / "trail_kepler"
        create_renderable_trail_orbit(
            orbits,
            str(trail_dir),
            "Fixture Trail Kepler",
            trail_head=True,
            gui_name="ignored legacy argument",
            gui_path="/ADAM/Fixture Trail",
            color=(0.2, 0.4, 0.6),
            resolution=12345,
            translation_type="Kepler",
            enable_fade=True,
            line_fade_amount=0.4,
            line_length=25.0,
            line_width=1.5,
            point_size=3,
            rendering="Lines+Points",
            dim_in_atmosphere=False,
            enabled=True,
            opacity=0.55,
            period=77.25,
            render_bin_mode="Overlay",
            tag="single-tag",
        )
        cases.append(
            {
                "name": "trail_kepler_all_options",
                "kind": "trail_orbit",
                "identifier": "Fixture Trail Kepler",
                "kwargs": {
                    "trail_head": True,
                    "gui_name": "ignored legacy argument",
                    "gui_path": "/ADAM/Fixture Trail",
                    "color": [0.2, 0.4, 0.6],
                    "resolution": 12345,
                    "translation_type": "Kepler",
                    "enable_fade": True,
                    "line_fade_amount": 0.4,
                    "line_length": 25.0,
                    "line_width": 1.5,
                    "point_size": 3,
                    "rendering": "Lines+Points",
                    "dim_in_atmosphere": False,
                    "enabled": True,
                    "opacity": 0.55,
                    "period": 77.25,
                    "render_bin_mode": "Overlay",
                    "tag": "single-tag",
                },
                "files": directory_contents(trail_dir),
            }
        )

        spice_dir = root / "trail_spice"
        kernel_path = root / "kernels" / "fixture kernel.bsp"
        create_renderable_trail_orbit(
            orbits,
            str(spice_dir),
            "Fixture Trail Spice",
            trail_head=False,
            gui_path="/ADAM/Fixture Spice",
            color=(0.9, 0.8, 0.7),
            resolution=86400,
            translation_type="Spice",
            rendering="Lines",
            spice_kernel_path=str(kernel_path),
            spice_id_mappings={
                "Object Alpha": -1001,
                "orbit,beta": -1002,
                "Object Gamma": -1003,
            },
        )
        cases.append(
            {
                "name": "trail_spice_resource",
                "kind": "trail_orbit",
                "identifier": "Fixture Trail Spice",
                "kwargs": {
                    "trail_head": False,
                    "gui_path": "/ADAM/Fixture Spice",
                    "color": [0.9, 0.8, 0.7],
                    "resolution": 86400,
                    "translation_type": "Spice",
                    "rendering": "Lines",
                    # Replace the temporary root with a stable synthetic path;
                    # only the relative resource path appears in output.
                    "spice_kernel_path": "__FIXTURE_KERNEL_PATH__",
                    "spice_id_mappings": {
                        "Object Alpha": -1001,
                        "orbit,beta": -1002,
                        "Object Gamma": -1003,
                    },
                },
                "kernel_relative_to_out_dir": str(
                    Path("..") / "kernels" / "fixture kernel.bsp"
                ),
                "files": directory_contents(spice_dir),
            }
        )

    fixture = {
        "schema": "adam_core.openspace_asset_product_fixture",
        "version": 1,
        "orbits": flat(orbits),
        "cases": cases,
    }
    OUT.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
