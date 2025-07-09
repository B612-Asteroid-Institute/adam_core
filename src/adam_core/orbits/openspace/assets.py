"""
Utilities for generating and reading OpenSpace Asset files
"""

import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

from ...constants import KM_P_AU, S_P_DAY
from ...coordinates.keplerian import KeplerianCoordinates
from ...coordinates.origin import OriginCodes
from ...coordinates.transform import transform_coordinates
from ...orbits import Orbits
from .lua import LuaDict
from .renderable import (
    Renderable,
    RenderableOrbitalKepler,
    RenderableOrbitalKeplerFormat,
    RenderableOrbitalKeplerRendering,
    RenderableTrailOrbit,
    RenderableTrailRendering,
    RenderBinMode,
    Resource,
)
from .translation import KeplerTranslation, SpiceTranslation, Transform


def _safe_orbital_period(period: float) -> float:
    """
    For an orbit with no period or infinite period, use a default value in days.
    """
    orbital_period = float(period)
    # Use a default value if orbital period is infinite or NaN
    if np.isnan(orbital_period) or np.isinf(orbital_period):
        return 10000.0
    return orbital_period


@dataclass(kw_only=True)
class Gui(LuaDict):
    name: str
    path: str


@dataclass(kw_only=True)
class Asset(LuaDict):
    identifier: str
    parent: str
    gui: Gui
    renderable: Optional[Renderable] = None
    transform: Optional[Transform] = None


def orbits_to_sbdb_file(orbits: Orbits, path: str) -> str:
    # Convert to Keplerian elements in heliocentric ecliptic J2000 frame
    keplerian = transform_coordinates(
        orbits.coordinates,
        representation_out=KeplerianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    # Write epochs in correct format
    epochs_tdb = orbits.coordinates.time.rescale("tdb").to_astropy()
    epochs = pa.array(
        [
            iso.split(" ")[0] + "." + f"{mjd}".split(".")[1]
            for iso, mjd in zip(epochs_tdb.iso, epochs_tdb.mjd)
        ]
    )

    table = pa.Table.from_pydict(
        {
            "full_name": orbits.orbit_id,
            "epoch_cal": epochs,
            "e": keplerian.e,
            "a": keplerian.a,
            "i": keplerian.i,
            "om": keplerian.raan,
            "w": keplerian.ap,
            "ma": keplerian.M,
            "per": keplerian.P,
        }
    )

    table.to_pandas().to_csv(path, index=False)
    return


def create_initialization(assets: List[str]) -> str:

    initialization = ["asset.onInitialize(function ()"]
    deinitialization = ["asset.onDeinitialize(function ()"]
    for asset in assets:
        initialization.append(f"  openspace.addSceneGraphNode({asset});")
        deinitialization.append(f"  openspace.removeSceneGraphNode({asset});")

    initialization.append("end)")
    deinitialization.append("end)")
    combined = "\n".join(initialization) + "\n" + "\n".join(deinitialization)
    return combined


def create_renderable_orbital_kepler(
    orbits: Orbits,
    out_dir: str,
    identifier: str,
    gui_name: Optional[str] = None,
    gui_path: Optional[str] = None,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    segment_quality: int = 10,
    contiguous_mode: Optional[bool] = None,
    enable_max_size: Optional[bool] = None,
    enable_outline: Optional[bool] = None,
    max_size: Optional[float] = None,
    outline_color: Optional[Tuple[float, float, float]] = None,
    outline_width: Optional[float] = None,
    point_size_exponent: Optional[float] = None,
    rendering: Optional[Literal["Trail", "Point", "PointsTrails"]] = None,
    render_size: Optional[int] = None,
    start_render_idx: Optional[int] = None,
    trail_fade: Optional[float] = None,
    trail_width: Optional[float] = None,
    dim_in_atmosphere: Optional[bool] = None,
    enabled: Optional[bool] = None,
    opacity: Optional[float] = None,
    render_bin_mode: Literal["Opaque", "Transparent", "Both"] = None,
    tag: Optional[Union[str, List[str]]] = None,
):
    """
    Create a renderable orbital Kepler for a given set of orbits.

    This is the best renderable to use for large numbers of orbits (e.g. > 1000).

    Example
    -------
    >>> from adam_core.orbits.query import query_sbdb
    >>> from adam_core.orbits.openspace import create_renderable_orbital_kepler
    >>> orbits = query_sbdb(["2013 RR165", "2018 BP1"])
    >>> create_renderable_orbital_kepler(orbits, "out_dir", "openspace_example")

    Parameters
    ----------
    orbits : Orbits
        The orbits to create a renderable orbitals for.
    out_dir : str
        The directory to output the asset files
    identifier : str
        The identifier for the asset
    gui_name : str, optional
        The name of the GUI for the asset.
    gui_path : str, optional
        The path of the GUI for the asset.
    color : tuple, optional
        The color of the Keplerian orbital.
    segment_quality : int, optional
        The segment quality for the Keplerian orbital.
    contiguous_mode : bool, optional
        Whether to enable contiguous mode for the Keplerian orbital.
    enable_max_size : bool, optional
        Whether to enable max size for the Keplerian orbital.
    enable_outline : bool, optional
        Whether to enable outline for the Keplerian orbital.
    max_size : float, optional
        The max size for the Keplerian orbital.
    outline_color : tuple, optional
        The color of the outline for the Keplerian orbital.
    outline_width : float, optional
        The width of the outline for the Keplerian orbital.
    point_size_exponent : float, optional
        The size exponent for the Keplerian orbital.
    rendering : str, optional
        The rendering type for the Keplerian orbital.
    render_size : int, optional
        The render size for the Keplerian orbital.
    start_render_idx : int, optional
        The start render index for the Keplerian orbital.
    trail_fade : float, optional
        The trail fade for the Keplerian orbital.
    trail_width : float, optional
        The trail width for the Keplerian orbital.
    dim_in_atmosphere : bool, optional
        Whether to render the Keplerian orbital in the atmosphere.
    enabled : bool, optional
        Whether to enable the Keplerian orbital.
    opacity : float, optional
        The opacity of the Keplerian orbital.
    render_bin_mode : str, optional
        The render bin mode for the Keplerian orbital.
    tag : str, optional
        The tag for the Keplerian orbital.
    """
    safe_identifier = identifier.replace(" ", "_")

    if rendering is not None:
        rendering = RenderableOrbitalKeplerRendering(rendering)

    if render_bin_mode is not None:
        render_bin_mode = RenderBinMode(render_bin_mode)

    if gui_name is None:
        gui_name = safe_identifier

    if gui_path is None:
        gui_path = "/ADAM"

    gui = Gui(name=gui_name, path=gui_path)

    # Create SBDB formatted file
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{safe_identifier}.csv")
    orbits_to_sbdb_file(orbits, path)

    # Initialize the renderable
    renderable = RenderableOrbitalKepler(
        color=color,
        format=RenderableOrbitalKeplerFormat.SBDB,
        path=Resource(path=os.path.basename(path)),
        segment_quality=segment_quality,
        contiguous_mode=contiguous_mode,
        enable_max_size=enable_max_size,
        enable_outline=enable_outline,
        max_size=max_size,
        outline_color=outline_color,
        outline_width=outline_width,
        point_size_exponent=point_size_exponent,
        rendering=rendering,
        render_size=render_size,
        start_render_idx=start_render_idx,
        trail_fade=trail_fade,
        trail_width=trail_width,
        dim_in_atmosphere=dim_in_atmosphere,
        enabled=enabled,
        opacity=opacity,
        render_bin_mode=render_bin_mode,
        tag=tag,
    )

    # Declare the asset
    asset = Asset(
        identifier=safe_identifier,
        parent="SunEclipJ2000",
        renderable=renderable,
        gui=gui,
    )

    with open(os.path.join(out_dir, f"{safe_identifier}.asset"), "w") as f:
        f.write("local Object = ")
        f.write(asset.to_string(indent=4))
        f.write("\n\n")
        f.write(create_initialization(["Object"]))

    return


def create_renderable_trail_orbit(
    orbits,
    out_dir: str,
    identifier: str,
    trail_head: Optional[bool] = True,
    gui_name: Optional[str] = None,
    gui_path: Optional[str] = None,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    resolution: int = 86400,
    translation_type: Literal["Kepler", "Spice"] = "Kepler",
    enable_fade: Optional[bool] = None,
    line_fade_amount: Optional[float] = None,
    line_length: Optional[float] = None,
    line_width: Optional[float] = None,
    point_size: Optional[int] = None,
    rendering: Literal["Lines", "Points", "Lines+Points"] = None,
    dim_in_atmosphere: Optional[bool] = None,
    enabled: Optional[bool] = None,
    opacity: Optional[float] = None,
    period: Optional[float] = None,
    render_bin_mode: Literal["Opaque", "Transparent", "Both"] = None,
    tag: Optional[Union[str, List[str]]] = None,
    spice_kernel_path: Optional[str] = None,
    spice_id_mappings: Optional[dict] = None,
):
    """
    Create a renderable trail orbit for a given set of orbits. These orbits can be represented by
    two different "translation" types:

    - KeplerTranslation: This is a Keplerian translation, which takes in a set of Keplerian elements and a time (default)
    - SpiceTranslation: This is a translation that is based on SPICE kernel data. SpiceKernels need to be created (see. )
    and passed to this function.

    The Keplerian translation is the default and is the best for small numbers of orbits (e.g. < 1000).
    The Spice translation is the best to use for a few dozen orbits.

    Example
    -------
    >>> from adam_core.orbits.query import query_sbdb
    >>> from adam_core.orbits.openspace import create_renderable_trail_orbit
    >>> orbits = query_sbdb(["2013 RR165", "2018 BP1"])
    >>> create_renderable_trail_orbit(orbits, "out_dir", "openspace_example", translation_type="Kepler")

    Parameters
    ----------
    orbits : Orbits
        The orbits to create a renderable trail orbit for
    out_dir : str
        The directory to output the asset files
    identifier : str
        The identifier for the asset
    trail_head : bool, optional
        Whether to include a trail head in the asset.
    gui_name : str, optional
        The name of the GUI for the asset.
    gui_path : str, optional
        The path of the GUI for the asset.
    color : tuple, optional
        The color of the trail.
    resolution : int, optional
        The resolution of the trail.
    translation_type : str, optional
        The type of translation to use.
    enable_fade : bool, optional
        Whether to enable fade for the trail.
    line_fade_amount : float, optional
        The amount of fade for the trail.
    line_length : float, optional
        The length of the trail.
    line_width : float, optional
        The width of the trail.
    point_size : int, optional
        The size of the points in the trail.
    rendering : str, optional
        The rendering type for the trail.
    dim_in_atmosphere : bool, optional
        Whether to render the trail in the atmosphere.
    enabled : bool, optional
        Whether to enable the trail.
    opacity : float, optional
        The opacity of the trail.
    period : float, optional
        The period of the trail.
    render_bin_mode : str, optional
        The render bin mode for the trail.
    tag : str, optional
        The tag for the trail.
    spice_kernel_path : str, optional
        The path to the SPICE kernel file.
    spice_id_mappings : dict, optional
        A dictionary mapping object IDs to SPICE IDs.
    """
    safe_identifier = identifier.replace(" ", "_")

    if rendering is not None:
        rendering = RenderableTrailRendering(rendering)

    if render_bin_mode is not None:
        render_bin_mode = RenderBinMode(render_bin_mode)

    if gui_path is None:
        gui_path = "/ADAM"

    keplerian = transform_coordinates(
        orbits.coordinates,
        representation_out=KeplerianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    if translation_type == "Spice":
        if spice_kernel_path is None:
            raise ValueError("Spice kernel path is required for Spice translation")
        if spice_id_mappings is None:
            raise ValueError("Spice ID mappings are required for Spice translation")

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"{safe_identifier}.asset"), "w") as f:

        assets = []
        asset_number = 0
        for i, (orbit, keplerian_i) in enumerate(zip(orbits, keplerian)):

            orbit_id = orbit.orbit_id[0].as_py()
            object_id = orbit.object_id[0].as_py()
            if object_id is None:
                object_id = orbit_id

            if trail_head:
                gui_trail = Gui(
                    name=f"{object_id} Trail", path=gui_path + f"/{object_id}"
                )
                gui_head = Gui(
                    name=f"{object_id} Head", path=gui_path + f"/{object_id}"
                )
            else:
                gui_trail = Gui(name=f"{object_id}", path=gui_path)

            if translation_type == "Kepler":
                translation = KeplerTranslation(
                    epoch=keplerian_i.time[0].rescale("tdb").to_astropy().iso[0],
                    semi_major_axis=keplerian_i.a[0].as_py() * KM_P_AU,
                    eccentricity=keplerian_i.e[0].as_py(),
                    inclination=keplerian_i.i[0].as_py(),
                    argument_of_periapsis=keplerian_i.ap[0].as_py(),
                    ascending_node=keplerian_i.raan[0].as_py(),
                    mean_anomaly=keplerian_i.M[0].as_py(),
                    period=_safe_orbital_period(keplerian_i.P[0])
                    * S_P_DAY,  # in seconds here
                )
            elif translation_type == "Spice":
                translation = SpiceTranslation(
                    target=spice_id_mappings[object_id],
                    observer="SOLAR SYSTEM BARYCENTER",
                )
            else:
                raise ValueError(f"Invalid translation type: {translation}")

            renderable = RenderableTrailOrbit(
                color=color,
                period=(
                    _safe_orbital_period(keplerian_i.P[0]) if period is None else period
                ),  # in days here (confusingly)
                resolution=resolution,
                translation=translation,
                enable_fade=enable_fade,
                line_fade_amount=line_fade_amount,
                line_length=line_length,
                line_width=line_width,
                point_size=point_size,
                rendering=rendering,
            )

            asset = Asset(
                identifier=object_id.replace(" ", "_"),
                parent="SunEclipJ2000",
                renderable=renderable,
                gui=gui_trail,
            )

            asset_id = f"Object{asset_number:08d}"
            asset_number += 1
            assets.append(asset_id)

            f.write(f"local {asset_id} = ")
            f.write(asset.to_string(indent=4))
            f.write("\n\n")

            if trail_head:
                head_asset = Asset(
                    identifier=object_id.replace(" ", "_") + "_Head",
                    parent="SunEclipJ2000",
                    transform=Transform(
                        translation=translation,
                    ),
                    gui=gui_head,
                )
                head_asset_id = f"Object{asset_number:08d}"
                asset_number += 1
                assets.append(head_asset_id)

                f.write(f"local {head_asset_id} = ")
                f.write(head_asset.to_string(indent=4))
                f.write("\n\n")

    initialization = create_initialization(assets)
    with open(os.path.join(out_dir, f"{safe_identifier}.asset"), "a") as f:
        f.write(initialization)

        if translation_type == "Spice":
            f.write("asset.onInitialize(function()\n")
            f.write(
                f"  local kernelResource = openspace.resource('{os.path.relpath(spice_kernel_path, out_dir)}')\n"
            )
            f.write("  openspace.spice.loadKernel(kernelResource)\n")
            f.write("end)\n")
            f.write("asset.onDeinitialize(function()\n")
            f.write("  openspace.spice.unloadKernel(kernelResource)\n")
            f.write("end)\n")

    return
