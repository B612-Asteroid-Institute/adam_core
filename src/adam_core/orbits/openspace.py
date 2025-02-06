"""
Utilities for generating and reading OpenSpace Asset files
"""

import os
import shutil
from typing import Optional, Type

from ..orbits import Orbits
from ..propagator import Propagator
from ..time import Timestamp
from .spice_kernel import orbits_to_spk


def generate_openspace_asset(
    orbits: Orbits,
    output_dir: str,
    start_time: Timestamp,
    end_time: Timestamp,
    propagator: Propagator,
    trail_color: tuple = (1.0, 1.0, 1.0),
    max_processes: Optional[int] = None,
) -> None:
    """
    Generate an OpenSpace Asset file from an Orbits object

    Parameters
    ----------
    orbits : Orbits
        The orbits to generate assets for
    output_file : str
        Path to output the OpenSpace asset file
    start_time : Timestamp
        The start time of the orbits
    end_time : Timestamp
        The end time of the orbits
    propagator : Propagator
        The propagator to use to generate the orbits
    """

    # Remove existing output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate SPK file with same name but .bsp extension
    spk_file = os.path.join(output_dir, "orbits.bsp")
    target_id_mappings = orbits_to_spk(
        orbits, spk_file, start_time, end_time, propagator, max_processes, step_days=1.0
    )

    # Write out an asset file for each orbit
    for i, (orbit_id, orbit) in enumerate(orbits.group_by_orbit_id()):
        keplerian = orbit.coordinates.to_keplerian()
        # orbital_period = float(keplerian.P[0])  # Convert to days

        # For now, hard code period to nominal orbit because we have
        # impactors which are hyperbolic
        orbital_period = 1113.0

        identifier = f"{orbit.object_id[0].as_py()}_{orbit.orbit_id[0].as_py()}"
        # Remove all parenthesis
        identifier = identifier.replace("(", "").replace(")", "").replace(" ", "_")
        print(f"Generating asset for {identifier}")
        asset_file = os.path.join(output_dir, f"{identifier}.asset")

        # Write the asset file
        with open(asset_file, "w") as f:
            # Add trail orbit
            trail = add_renderable_trail_orbit(
                identifier=f"trail_{identifier}",
                target_id=target_id_mappings[orbit_id],
                period=orbital_period,
                sample_interval=86400,  # Daily samples
                tag="ADAM",
                opacity=1.0,
                rgb=trail_color,
                line_width=2,
            )
            f.write(trail + "\n\n")

            # Add trail head
            head = add_renderable_trail_head(
                identifier=f"head_{identifier}",
                target_id=target_id_mappings[orbit_id],
                tag="ADAM",
            )
            f.write(head + "\n\n")

            relative_spk_file = os.path.relpath(spk_file, output_dir)

            # Register on initialize and remove on deinitialize
            initialization = (
                f"asset.onInitialize(function()\n"
                f"  openspace.addSceneGraphNode(Trail)\n"
                f"  openspace.addSceneGraphNode(Head)\n"
                f"end)\n"
                f"asset.onDeinitialize(function()\n"
                f"  openspace.removeSceneGraphNode(Trail)\n"
                f"  openspace.removeSceneGraphNode(Head)\n"
                f"end)\n"
                f"asset.export(\"trail_{identifier}\", Trail)\n"
                f"asset.export(\"head_{identifier}\", Head)\n"
            )

            f.write(initialization)

    # Write out a master asset file
    master_asset_file = os.path.join(output_dir, "orbits.asset")
    # For now we hardcode the path.
    with open(master_asset_file, "w") as f:
        load_kernel = (
            f"asset.onInitialize(function()\n"
            f"  openspace.spice.loadKernel(openspace.absPath(\"../data/assets/scene/solarsystem/adam/orbits.bsp\"))\n"
            f"end)\n"
            f"asset.onDeinitialize(function()\n"
            f"  openspace.spice.unloadKernel(openspace.absPath(\"../data/assets/scene/solarsystem/adam/orbits.bsp\"))\n"
            f"end)\n"
        )
        f.write(load_kernel)
        for orbit in orbits:
            identifier = f"{orbit.object_id[0].as_py()}_{orbit.orbit_id[0].as_py()}"
            # Remove all parenthesis
            identifier = identifier.replace("(", "").replace(")", "").replace(" ", "_")
            f.write(f"asset.require(\"./{identifier}.asset\")\n")


def add_renderable_trail_orbit(
    identifier: str,
    target_id: str,
    period: float,
    sample_interval: int = 3600,
    tag: str = "THOR",
    opacity: float = 1.0,
    rgb: tuple = (1.0, 1.0, 1.0),
    line_width: int = 1,
):
    return f"""local Trail = {{
    Identifier = "{identifier}",
    Parent = "SolarSystemBarycenter",
    Renderable = {{
        Type = "RenderableTrailOrbit",
        Enabled = true,
        Translation = {{
            Type = "SpiceTranslation",
            Target = "{target_id}",
            Observer = "SOLAR SYSTEM BARYCENTER",
        }},
        Color = {{{rgb[0]},{rgb[1]},{rgb[2]}}},
        Opacity = {opacity},
        Period = {period},
        EnableFade = true,
        LineLength = 0.1,
        LineFadeAmount = 0.5,
        Resolution = {sample_interval},
        LineWidth = {line_width},
        SampleInterval = {sample_interval}
    }},
    Opacity = {opacity},
    Tag = {{"{tag}"}},
    GUI = {{
        Name = "{identifier}_trail",
        Path = "/ADAM"
    }}
}}"""


def add_renderable_trail_head(
    identifier: str,
    target_id: str,
    tag: str = "ADAM",
):
    return f"""local Head = {{
    Identifier = "{identifier}",
    Parent = "SolarSystemBarycenter",
    Transform = {{
        Translation = {{
            Type = "SpiceTranslation",
            Target = "{target_id}",
            Observer = "SOLAR SYSTEM BARYCENTER",
        }},
    }},
    Tag = {{"{tag}"}},
    GUI = {{
        Name = "{identifier}",
        Path = "/ADAM"
    }}
}}"""
