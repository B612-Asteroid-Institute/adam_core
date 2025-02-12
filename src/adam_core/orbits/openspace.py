"""
Utilities for generating and reading OpenSpace Asset files
"""

import os

import numpy as np

from ..coordinates.keplerian import KeplerianCoordinates
from ..orbits import Orbits


def _safe_orbital_period(keplerian: KeplerianCoordinates) -> float:
    """
    Compute the orbital period of an orbit.
    """
    orbital_period = float(keplerian.P[0])
    # Use a default value if orbital period is infinite or NaN
    if np.isnan(orbital_period) or np.isinf(orbital_period):
        return 1000.0
    return orbital_period


def generate_openspace_asset(
    orbits: Orbits,
    spice_kernel_path: str,
    output_dir: str,
    target_id_mappings: dict,
    trail_color: tuple = (1.0, 1.0, 1.0),
) -> None:
    """
    Generate an OpenSpace Asset file from an Orbits object

    Parameters
    ----------
    orbits : Orbits
        The orbits to generate assets for
    spice_kernel_path : str
        The path to the SPICE kernel file generated for these orbits
    output_dir : str
        The directory to output the asset files
    target_id_mappings : dict
        A dictionary mapping orbit IDs to target IDs
    trail_color : tuple
        The color of the trail
    """

    os.makedirs(output_dir, exist_ok=True)

    # Write out an asset file for each orbit
    for i, (orbit_id, orbit) in enumerate(orbits.group_by_orbit_id()):
        keplerian = orbit.coordinates.to_keplerian()
        orbital_period = _safe_orbital_period(keplerian)

        identifier = f"{orbit.object_id[0].as_py()}_{orbit.orbit_id[0].as_py()}"
        # Remove all parenthesis
        identifier = identifier.replace("(", "").replace(")", "").replace(" ", "_")
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
                f'asset.export("trail_{identifier}", Trail)\n'
                f'asset.export("head_{identifier}", Head)\n'
            )

            f.write(initialization)

    # Write out a master asset file
    master_asset_file = os.path.join(output_dir, "orbits.asset")
    relative_spk_file = os.path.relpath(spice_kernel_path, output_dir)

    # For now we hardcode the path.
    with open(master_asset_file, "w") as f:
        load_kernel = (
            f"asset.onInitialize(function()\n"
            f"  local kernelResource = openspace.resource('{relative_spk_file}')\n"
            f"  openspace.spice.loadKernel(kernelResource)\n"
            f"end)\n"
            f"asset.onDeinitialize(function()\n"
            f"  openspace.spice.unloadKernel(kernelResource)\n"
            f"end)\n"
        )
        f.write(load_kernel)
        for orbit in orbits:
            identifier = f"{orbit.object_id[0].as_py()}_{orbit.orbit_id[0].as_py()}"
            # Remove all parenthesis
            identifier = identifier.replace("(", "").replace(")", "").replace(" ", "_")
            f.write(f'asset.require("./{identifier}.asset")\n')


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
