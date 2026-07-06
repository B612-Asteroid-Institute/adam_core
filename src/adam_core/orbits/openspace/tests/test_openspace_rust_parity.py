"""OpenSpace text-rendering parity gates for the Rust renderer (bead
personal-cmy.28). Fixture generated from the untouched legacy checkout via
``.legacy-venv/bin/python migration/scripts/generate_openspace_parity_fixture.py``.
"""

import json
from pathlib import Path

import pytest

from adam_core.orbits.openspace.assets import Asset, Gui, create_initialization
from adam_core.orbits.openspace.renderable import (
    RenderableOrbitalKepler,
    RenderableOrbitalKeplerFormat,
    RenderableOrbitalKeplerRendering,
    RenderableTrailOrbit,
    RenderableTrailRendering,
    RenderableTrailTrajectory,
    RenderBinMode,
    Resource,
)
from adam_core.orbits.openspace.translation import (
    KeplerTranslation,
    SpiceTranslation,
    Transform,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[5]
    / "migration"
    / "artifacts"
    / "openspace_text_fixture_2026-07-06.json"
)


@pytest.fixture(scope="module")
def fixture():
    assert FIXTURE_PATH.exists(), (
        "OpenSpace parity fixture missing; generate with the legacy interpreter: "
        ".legacy-venv/bin/python migration/scripts/generate_openspace_parity_fixture.py"
    )
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def objects():
    gui = Gui(name="Fixture Object", path="/ADAM/Fixtures")
    resource = Resource(path="fixture orbit.csv")
    kepler_translation = KeplerTranslation(
        epoch="2026-07-06 12:34:56.789",
        semi_major_axis=149597870.7,
        eccentricity=0.123456789,
        inclination=12.5,
        argument_of_periapsis=45.25,
        ascending_node=180.0,
        mean_anomaly=270.75,
        period=31557600.0,
    )
    spice_translation = SpiceTranslation(
        target=-12345,
        observer="SOLAR SYSTEM BARYCENTER",
        fixed_date="2026 JUL 06 00:00:00",
        frame="ECLIPJ2000",
    )
    orbital_kepler = RenderableOrbitalKepler(
        color=(1.0, 0.5, 0.25),
        format=RenderableOrbitalKeplerFormat.SBDB,
        path=resource,
        segment_quality=42,
        contiguous_mode=True,
        enable_max_size=False,
        enable_outline=True,
        max_size=12.5,
        outline_color=(0.1, 0.2, 0.3),
        outline_width=2.5,
        point_size_exponent=1.25,
        rendering=RenderableOrbitalKeplerRendering.POINTS_TRAILS,
        render_size=8,
        start_render_idx=1,
        trail_fade=0.75,
        trail_width=0.33,
        dim_in_atmosphere=False,
        enabled=True,
        opacity=0.8,
        render_bin_mode=RenderBinMode.OPAQUE,
        tag=["science", "migration"],
    )
    trail_orbit = RenderableTrailOrbit(
        color=(0.2, 0.4, 0.6),
        period=365.25,
        resolution=86400,
        translation=kepler_translation,
        enable_fade=True,
        line_fade_amount=0.4,
        line_length=25.0,
        line_width=1.5,
        point_size=3,
        rendering=RenderableTrailRendering.LINES_POINTS,
        dim_in_atmosphere=True,
        enabled=False,
        opacity=0.55,
        render_bin_mode=RenderBinMode.PREDEFERREDTRANSPARENT,
        tag="single-tag",
    )
    trail_trajectory = RenderableTrailTrajectory(
        color=(0.9, 0.8, 0.7),
        start_time="2026-07-06T00:00:00.000",
        end_time="2026-07-16T00:00:00.000",
        translation=spice_translation,
        accurate_trail_positions=1,
        enable_fade=False,
        enable_sweep_chunking=2,
        line_fade_amount=0.2,
        line_length=10.0,
        line_width=0.5,
        point_size=4,
        rendering=RenderableTrailRendering.POINTS,
        sample_interval=600.0,
        show_full_trail=True,
        sweep_chunk_size=64,
        time_stamp_subsample_factor=3,
        dim_in_atmosphere=False,
        enabled=True,
        opacity=0.95,
        render_bin_mode=RenderBinMode.OVERLAY,
        tag=["trajectory", "fixture"],
    )
    asset = Asset(
        identifier="Fixture_Object",
        parent="SunEclipJ2000",
        gui=gui,
        renderable=orbital_kepler,
        transform=Transform(translation=spice_translation),
    )
    return {
        "gui": gui,
        "resource": resource,
        "kepler_translation": kepler_translation,
        "spice_translation": spice_translation,
        "orbital_kepler": orbital_kepler,
        "trail_orbit": trail_orbit,
        "trail_trajectory": trail_trajectory,
        "asset": asset,
        "transform": Transform(translation=kepler_translation),
    }


def test_openspace_lua_text_matches_legacy_fixture(fixture, objects):
    for case in fixture["cases"]:
        actual = objects[case["name"]].to_string(indent=case["indent"])
        assert actual == case["text"], case["name"]


def test_openspace_initialization_matches_legacy_fixture(fixture):
    assert (
        create_initialization(["Object00000000", "Object00000001"])
        == fixture["initialization"]
    )
