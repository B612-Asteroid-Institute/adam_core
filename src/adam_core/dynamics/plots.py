from typing import List, Literal, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..constants import KM_P_AU
from ..constants import Constants as c
from ..coordinates import (
    CartesianCoordinates,
    Origin,
    OriginCodes,
    SphericalCoordinates,
    transform_coordinates,
)
from ..orbits import Orbits, VariantOrbits
from ..propagator import Propagator
from ..time import Timestamp
from ..utils.plots.data import Coastlines
from ..utils.plots.logos import (
    AsteroidInstituteLogoDark,
    AsteroidInstituteLogoLight,
    get_logo_base64,
)
from ..utils.spice import get_perturber_state
from .impacts import EarthImpacts

EARTH_RADIUS_KM = c.R_EARTH_EQUATORIAL * KM_P_AU * 0.999
MOON_RADIUS_KM = 1738.1

try:
    import geopandas as gpd
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Please install adam_core[plots] to use this feature.")


def generate_impact_visualization_data(
    orbit: Orbits,
    variant_orbits: VariantOrbits,
    impacts: EarthImpacts,
    propagator: Propagator,
    time_step: float = 5,
    time_range: float = 60,
    max_processes: Optional[int] = None,
):
    """
    Generates the data for the impact visualization animation.

    Parameters
    ----------
    orbit: Orbits
        The nominal best-fit orbit to propagate.
    variant_orbits: VariantOrbits
        The variants to propagate.
    impacts: EarthImpacts
        The impacts detected within the variants.
    propagator: Propagator
        The propagator to use to propagate the orbit.
    time_step: float
        The time step to use for the propagation.
    time_range: float
        The time range to use for the propagation.
    max_processes: Optional[int]
        The maximum number of processes to use for the propagation.

    Returns
    -------
    Tuple[Orbits, Orbits]
        The propagated nominal best-fit orbit and the propagated variants.
    """
    # Create propagation times around the mean impact time
    mean_impact_time = np.mean(
        impacts.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    )
    first_propagation_time = (
        mean_impact_time - time_range / 60 / 24
    )  # round to the nearest time step
    first_propagation_time = first_propagation_time - np.mod(
        first_propagation_time, time_step / 60 / 24
    )
    last_propagation_time = mean_impact_time + time_range / 60 / 24
    last_propagation_time = last_propagation_time - np.mod(
        last_propagation_time, time_step / 60 / 24
    )

    propagation_times = Timestamp.from_mjd(
        np.arange(
            first_propagation_time,
            last_propagation_time + time_step / 60 / 24,
            time_step / 60 / 24,
        ),
        scale=impacts.coordinates.time.scale,
    )

    # Propagate the nominal best-fit orbit to the propagation times
    propagated_orbit = propagator.propagate_orbits(
        orbit, propagation_times, max_processes=max_processes
    )
    # Transform the best-fit orbit to geocentric frame
    propagated_orbit = propagated_orbit.set_column(
        "coordinates",
        transform_coordinates(
            propagated_orbit.coordinates,
            representation_out=CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.EARTH,
        ),
    )

    # Propagate the variants to the propagation times
    propagated_variants = propagator.propagate_orbits(
        Orbits.from_kwargs(
            orbit_id=variant_orbits.variant_id,
            object_id=variant_orbits.object_id,
            coordinates=variant_orbits.coordinates,
        ),
        propagation_times,
        max_processes=max_processes,
    )
    # Transform the propagated variants to geocentric frame
    propagated_variants = propagated_variants.set_column(
        "coordinates",
        transform_coordinates(
            propagated_variants.coordinates,
            representation_out=CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.EARTH,
        ),
    )

    propagated_variants = cleanse_propagated_variants(propagated_variants, impacts)

    return propagated_orbit, propagated_variants


def cleanse_propagated_variants(propagated_variants: Orbits, impacts: EarthImpacts):
    """
    Sets variants propagated after their impact time to their impact coordinates on the surface of the Earth.

    Note: Due to the nature of the impact detection code, some of the variants may already be inside the sphere of the Earth when the impact is detected.
    In these cases, the variants' distance from the geocenter is set to the radius of the Earth. The results of this function should not be used
    for high-fidelity impact predictions but instead for visualizations of the approximate impact corridor.

    Parameters
    ----------
    propagated_variants: Orbits
        The propagated variants to cleanse.
    impacts: EarthImpacts
        The impacts detected within the variants.

    Returns
    -------
    Orbits
        The propagated variants with the variants after their impact time set to their impact coordinates on the surface of the Earth.
    """
    assert propagated_variants.coordinates.frame == "ecliptic"
    assert pc.all(
        pc.equal(propagated_variants.coordinates.origin.code, "EARTH")
    ).as_py()

    for impact in impacts:

        post_impact_mask = pc.and_(
            pc.is_in(propagated_variants.orbit_id, impact.variant_id),
            pc.greater_equal(
                propagated_variants.coordinates.time.mjd(),
                impact.coordinates.time.mjd()[0],
            ),
        )

        propagated_variants_correct = propagated_variants.apply_mask(
            pc.invert(post_impact_mask)
        )
        propagated_variants_incorrect = propagated_variants.apply_mask(post_impact_mask)

        impact_coordinates = impact.impact_coordinates
        impact_coordinates = qv.concatenate(
            [impact_coordinates for _ in range(len(propagated_variants_incorrect))]
        )

        # Hack: In some cases, due to the time step of the propagation, the variants may already
        # be inside the sphere of the Earth when the impact is detected. In these cases, the variants'
        # distance from the geocenter is set to the radius of the Earth.
        impact_coordinates = impact_coordinates.set_column(
            "rho", pa.repeat(c.R_EARTH_EQUATORIAL, len(impact_coordinates))
        )

        # Override the time of the impact coordinates to the time of the propagated variants beyond the impact time, we do
        # this so we can then calculate position of these locations on the surface of the Earth as the Earth rotates.
        impact_coordinates = impact_coordinates.set_column(
            "time", propagated_variants_incorrect.coordinates.time
        )

        geocentric_impact_coordinates = transform_coordinates(
            impact_coordinates,
            representation_out=CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.EARTH,
        )
        propagated_variants_incorrect = propagated_variants_incorrect.set_column(
            "coordinates",
            geocentric_impact_coordinates,
        )

        propagated_variants = qv.concatenate(
            [propagated_variants_correct, propagated_variants_incorrect]
        )

    return propagated_variants


def create_sphere(radius, offset=None):
    """
    Create a set of points that form a sphere.

    Parameters
    ----------
    radius: float
        The radius of the sphere.
    offset: array-like, optional
        The offset of the sphere from the origin.
    """
    if offset is None:
        offset = np.array([0, 0, 0])

    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    x += offset[0]
    y += offset[1]
    z += offset[2]

    return x, y, z


def add_earth(
    time,
    coastlines: bool = True,
    origin: OriginCodes = OriginCodes.EARTH,
    frame: str = "ecliptic",
    show: bool = True,
) -> Tuple[go.Surface, List[go.Scatter3d]]:
    """
    Add the Earth to the plot.

    Parameters
    ----------
    time: Timestamp
        The time of the plot.
    coastlines: bool, optional
        Whether to add the coastlines to the plot.
    origin: Origin, optional
        The origin of the plot.
    frame: str, optional
        The frame of the plot.
    show: bool, optional
        Whether to show the Earth by default.

    Returns
    -------
    Tuple[go.Surface, List[go.Scatter3d]]
        The Earth rendered as a sphere and the traces for the coastlines.
    """
    earth_state = get_perturber_state(
        OriginCodes.EARTH,
        time,
        frame=frame,
        origin=origin,
    )
    x, y, z = create_sphere(EARTH_RADIUS_KM, offset=earth_state.r[0] * KM_P_AU)

    surface_traces = []
    if coastlines:
        world = gpd.read_file(str(Coastlines))

        for idx, row in world.iterrows():
            # Get the polygon coordinates
            if row.geometry.geom_type == "Polygon":
                coords = np.array(row.geometry.exterior.coords)
            elif row.geometry.geom_type == "MultiPolygon":
                coords = np.array(row.geometry.geoms[0].exterior.coords)
            elif row.geometry.geom_type == "LineString":
                coords = np.array(row.geometry.coords)
            elif row.geometry.geom_type == "MultiLineString":
                coords = np.array(row.geometry.geoms[0].coords)

            coords = SphericalCoordinates.from_kwargs(
                rho=pa.repeat(EARTH_RADIUS_KM / KM_P_AU, len(coords)),
                lon=coords[:, 0],
                lat=coords[:, 1],
                time=Timestamp.from_kwargs(
                    days=pa.repeat(time.days[0], len(coords)),
                    nanos=pa.repeat(time.nanos[0], len(coords)),
                    scale=time.scale,
                ),
                frame="itrf93",
                origin=Origin.from_kwargs(
                    code=pa.repeat(OriginCodes.EARTH.name, len(coords))
                ),
            )

            coords = transform_coordinates(
                coords,
                representation_out=CartesianCoordinates,
                frame_out=frame,
                origin_out=origin,
            )

            surface_traces.append(
                go.Scatter3d(
                    x=coords.x.to_numpy(zero_copy_only=False) * KM_P_AU,
                    y=coords.y.to_numpy(zero_copy_only=False) * KM_P_AU,
                    z=coords.z.to_numpy(zero_copy_only=False) * KM_P_AU,
                    mode="lines",
                    line=dict(color="white", width=2),
                    showlegend=False,
                    visible=show,
                )
            )

    earth_surface = go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.5,
        colorscale=[[0, "#015294"], [1, "#015294"]],
        showscale=False,
        name="Earth",
        showlegend=True,
        visible=show,
    )
    return earth_surface, surface_traces


def add_moon(
    time: Timestamp,
    origin: OriginCodes = OriginCodes.EARTH,
    frame: Literal["ecliptic", "equatorial", "itrf93"] = "ecliptic",
    show: bool = True,
) -> go.Surface:
    """
    Add the Moon to the plot.

    Parameters
    ----------
    time: Timestamp
        The time of the snapshot.
    origin: OriginCodes
        The origin of the plot.
    frame: Literal["ecliptic", "equatorial", "itrf93"]
        The frame of the plot.

    Returns
    -------
    go.Surface
        The Moon rendered as a sphere.
    """
    lunar_state = get_perturber_state(
        OriginCodes.MOON,
        time,
        frame=frame,
        origin=origin,
    )
    x, y, z = create_sphere(MOON_RADIUS_KM, offset=lunar_state.r[0] * KM_P_AU)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.5,
        colorscale=[[0, "#555555"], [1, "#555555"]],
        showscale=False,
        name="Moon",
        showlegend=True,
        visible=show,
    )


def plot_impact_simulation(
    propagated_best_fit_orbit: Orbits,
    propagated_variants: Orbits,
    impacts: EarthImpacts,
    grid: bool = True,
    title: str = None,
    logo: bool = True,
    show_impacting: bool = True,
    show_non_impacting: bool = True,
    show_best_fit: bool = True,
    show_earth: bool = True,
    show_moon: bool = True,
    autoplay: bool = True,
) -> go.Figure:
    """
    Plot the impact simulation.

    Parameters
    ----------
    propagated_best_fit_orbit: Orbits
        The propagated best-fit orbit.
    propagated_variants: Orbits
        The propagated variants.
    impacts: EarthImpacts
        The impacts detected within the variants.
    grid: bool, optional
        Whether to add the grid to the plot.
    title: str, optional
        The title of the plot.
    logo: bool, optional
        Whether to add the Asteroid Institute logo to the plot.

    Returns
    -------
    go.Figure
        The impact simulation plot.
    """
    # Ensure the propagated variants and best-fit orbit are sorted by time
    propagated_variants = propagated_variants.sort_by(
        ["coordinates.time.days", "coordinates.time.nanos"]
    )
    propagated_best_fit_orbit = propagated_best_fit_orbit.sort_by(
        ["coordinates.time.days", "coordinates.time.nanos"]
    )

    propagation_times = propagated_variants.coordinates.time.unique()
    propagation_times_isot = propagation_times.to_astropy().isot

    num_variants = len(propagated_variants.orbit_id.unique())

    if title is None:
        prefix = ""
    else:
        prefix = f"{title}<br>"

    # Build the individual frames for the animation
    frames = []
    for i, time in enumerate(propagation_times):

        # Select the propagated variants at the current time
        time_propagated_variants_mask = pc.and_(
            pc.equal(propagated_variants.coordinates.time.days, time.days[0]),
            pc.equal(propagated_variants.coordinates.time.nanos, time.nanos[0]),
        )
        time_propagated_variants = propagated_variants.apply_mask(
            time_propagated_variants_mask
        )

        # Create a mask for the impacting and non-impacting variants
        impactor_mask_i = pc.is_in(
            time_propagated_variants.orbit_id, impacts.variant_id
        )
        impacts_at_time = impacts.apply_mask(
            pc.less_equal(impacts.coordinates.time.mjd(), time.mjd()[0])
        )
        impact_count = len(impacts_at_time)

        non_impactor_coordinates = time_propagated_variants.apply_mask(
            pc.invert(impactor_mask_i)
        ).coordinates
        impactor_coordinates = time_propagated_variants.apply_mask(
            impactor_mask_i
        ).coordinates
        orbit_at_time = propagated_best_fit_orbit[i].coordinates

        earth_surface, surface_traces = add_earth(time, show=show_earth)

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=impactor_coordinates.x.to_numpy(zero_copy_only=False) * KM_P_AU,
                    y=impactor_coordinates.y.to_numpy(zero_copy_only=False) * KM_P_AU,
                    z=impactor_coordinates.z.to_numpy(zero_copy_only=False) * KM_P_AU,
                    mode="markers",
                    marker=dict(
                        size=1,
                        color="red",
                        opacity=1,
                        showscale=False,
                    ),
                    name="Impacting Variants",
                    visible=show_impacting,
                ),
                go.Scatter3d(
                    x=non_impactor_coordinates.x.to_numpy(zero_copy_only=False)
                    * KM_P_AU,
                    y=non_impactor_coordinates.y.to_numpy(zero_copy_only=False)
                    * KM_P_AU,
                    z=non_impactor_coordinates.z.to_numpy(zero_copy_only=False)
                    * KM_P_AU,
                    mode="markers",
                    marker=dict(size=1, color="#5685C3", opacity=0.5, showscale=False),
                    name="Non-Impacting Variants",
                    visible=show_non_impacting,
                ),
                go.Scatter3d(
                    x=orbit_at_time.x.to_numpy(zero_copy_only=False) * KM_P_AU,
                    y=orbit_at_time.y.to_numpy(zero_copy_only=False) * KM_P_AU,
                    z=orbit_at_time.z.to_numpy(zero_copy_only=False) * KM_P_AU,
                    mode="markers",
                    marker=dict(size=4, color="#F07620", opacity=1, showscale=False),
                    name="Best-Fit Orbit",
                    visible=show_best_fit,
                ),
                earth_surface,
                add_moon(time, show=show_moon),
                *surface_traces,
            ],
            name=str(i),
            layout=dict(
                title=dict(
                    text=f"{prefix}Time: {propagation_times_isot[i]}<br>Impacts: {impact_count} of {num_variants} Variants<br>Impact Probability: {impact_count/num_variants * 100:.3f}%",
                    x=0.01,
                    y=0.97,
                    font=dict(size=14, color="white"),
                )
            ),
        )

        frames.append(frame)

    # Plot the figure
    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)

    if grid:
        config = dict(
            showgrid=True,
            zeroline=True,
            visible=True,
            showticklabels=True,
            gridcolor="rgba(128,128,128,0.1)",
            showbackground=False,
            backgroundcolor="rgb(0,0,0)",
            tickfont=dict(color="white"),
        )
    else:
        config = dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            showticklabels=False,
            gridcolor="rgba(128,128,128,0.0)",
            showbackground=False,
            backgroundcolor="rgb(0,0,0)",
            tickfont=dict(color="white"),
        )

    if logo:
        images = [
            dict(
                source=get_logo_base64(AsteroidInstituteLogoDark),
                xref="paper",
                yref="paper",
                x=1.02,
                y=-0.15,
                sizex=0.20,
                sizey=0.20,
                xanchor="left",
                yanchor="bottom",
                layer="above",
            )
        ]
    else:
        images = []

    fig.update_layout(
        scene=dict(
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode="data",
            camera=dict(
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1),
            ),
            bgcolor="rgb(0,0,0)",
            xaxis=dict(title=dict(text="x [km]", font=dict(color="white")), **config),
            yaxis=dict(title=dict(text="y [km]", font=dict(color="white")), **config),
            zaxis=dict(title=dict(text="z [km]", font=dict(color="white")), **config),
        ),
        autosize=True,
        margin=dict(l=7, r=7, t=10, b=7, pad=0),
        paper_bgcolor="rgb(0,0,0)",
        plot_bgcolor="rgb(0,0,0)",
        font=dict(color="white"),
        images=images,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.04,
                y=-0.03,
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="⏸",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                currentvalue=dict(prefix="Time: "),
                pad=dict(t=50),
                len=0.85,
                x=0.12,
                y=0.05,
                font=dict(color="white", size=10),
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            dict(
                                frame=dict(duration=50, redraw=True),
                                mode="immediate",
                            ),
                        ],
                        label=propagation_times_isot[i],
                        method="animate",
                    )
                    for i in range(len(frames))
                ],
            )
        ],
    )

    return fig


def plot_risk_corridor(
    impacts: EarthImpacts, title: Optional[str] = None, logo: bool = True
) -> go.Figure:
    """
    Plot the risk corridor with toggleable globe/map views.
    Points colored by time with a linear scale and animated sequence.

    Parameters
    ----------
    impacts : Impact data containing coordinates
    title : str, optional
        Plot title
    logo : bool, optional
        Whether to add the Asteroid Institute logo to the plot.

    Returns
    -------
    go.Figure
        The risk corridor plot.
    """

    # Sort all data by time
    times = impacts.impact_coordinates.time.to_astropy()
    time_order = np.argsort(times.mjd)
    lon = impacts.impact_coordinates.lon.to_numpy(zero_copy_only=False)[time_order]
    lat = impacts.impact_coordinates.lat.to_numpy(zero_copy_only=False)[time_order]
    times = times[time_order]

    # Convert times to minutes since first impact
    time_nums = (times.mjd - times.mjd.min()) * 24 * 60
    first_impact_time = times[0].iso
    time_step = 2

    color_bar_config = dict(
        tickmode="array",
        tickangle=0,
        orientation="h",
        x=0.5,
        y=-0.3,
        xanchor="center",
        yanchor="bottom",
        thickness=25,
    )

    # Calculate center
    center_lon = lon[0]
    center_lat = lat[0]

    plot_config = dict(
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=1,
        ),
    )

    # Create frames for animation
    frames = []
    for i in range(len(lon)):

        ticktext = [
            f"T+{i:d} min"
            for i in np.arange(0, time_nums[-1] + time_step, time_step)
            .astype(int)
            .tolist()
        ]
        frame = go.Frame(
            data=[
                go.Scattermap(
                    lon=lon[: i + 1],
                    lat=lat[: i + 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=time_nums[: i + 1],
                        colorscale="Viridis",
                        opacity=0.8,
                        showscale=True,
                        colorbar=dict(
                            title=dict(
                                text=f"Minutes After First Detected Possible Impact {first_impact_time}",
                                side="top",
                                font=dict(size=12, color="black"),
                            ),
                            ticktext=ticktext,
                            tickvals=np.arange(0, time_nums[-1] + time_step, time_step),
                            **color_bar_config,
                        ),
                    ),
                    name="Variant Impact Locations",
                    hovertext=[
                        f"Time: {t.iso}<br>Lon: {lo:.2f}°<br>Lat: {la:.2f}°<br>+{mins:.1f} min"
                        for t, lo, la, mins in zip(
                            times[: i + 1],
                            lon[: i + 1],
                            lat[: i + 1],
                            time_nums[: i + 1],
                        )
                    ],
                )
            ],
            name=str(i),
        )
        frames.append(frame)

    # Create the figure with initial state
    fig = go.Figure(
        data=[
            go.Scattermap(
                lon=[np.nan],
                lat=[np.nan],
                mode="markers",
                marker=dict(
                    size=8,
                    color=[],
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text=f"Minutes After First Detected Possible Impact {first_impact_time}",
                            side="top",
                            font=dict(size=12, color="black"),
                        ),
                        ticktext=[
                            f"T+{i:d} min"
                            for i in np.arange(0, time_nums[-1] + time_step, time_step)
                            .astype(int)
                            .tolist()
                        ],
                        tickvals=[0, time_nums[-1]],
                        **color_bar_config,
                    ),
                ),
                name="Variant Impact Locations",
                hovertext=[
                    f"Time: {times[0].iso}<br>Lon: {lon[0]:.2f}°<br>Lat: {lat[0]:.2f}°<br>+{time_nums[0]:.1f} min"
                ],
            )
        ],
        frames=frames,
    )

    if title is None:
        title = "Risk Corridor"

    if logo:
        images = [
            dict(
                source=get_logo_base64(AsteroidInstituteLogoLight),
                xref="paper",
                yref="paper",
                x=1.03,
                y=-0.30,
                sizex=0.19,
                sizey=0.19,
                xanchor="left",
                yanchor="bottom",
                layer="above",
            )
        ]
    else:
        images = []

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
            y=0.98,
            font=dict(size=14, color="black"),
        ),
        autosize=True,
        margin=dict(l=7, r=7, t=30, b=7, pad=0),
        **plot_config,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.03,
                y=-0.03,
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="⏸",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        images=images,
        sliders=[
            dict(
                currentvalue=dict(prefix="Variant: "),
                pad=dict(t=50),
                len=0.85,
                x=0.10,
                y=0.07,
                font=dict(color="black", size=10),
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            dict(
                                frame=dict(duration=50, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                        label=str(i),
                        method="animate",
                    )
                    for i in range(len(frames))
                ],
            )
        ],
    )

    return fig


def plot_risk_corridor_globe(
    impacts: EarthImpacts, title: Optional[str] = None
) -> go.Figure:
    """
    Plot the risk corridor with toggleable globe/map views.
    Points colored by time with a linear scale and animated sequence.

    Parameters
    ----------
    impacts : Impact data containing coordinates
    title : str, optional
        Plot title
    """
    # Sort all data by time
    times = impacts.impact_coordinates.time.to_astropy()
    time_order = np.argsort(times.mjd)
    lon = impacts.impact_coordinates.lon.to_numpy(zero_copy_only=False)[time_order]
    lat = impacts.impact_coordinates.lat.to_numpy(zero_copy_only=False)[time_order]
    times = times[time_order]

    # Convert times to minutes since first impact
    time_nums = (times.mjd - times.mjd.min()) * 24 * 60
    first_impact_time = times[0].iso
    time_step = 2

    color_bar_config = dict(
        tickmode="array",
        tickangle=0,
        orientation="h",
        x=0.5,
        y=-0.1,
        xanchor="center",
        yanchor="bottom",
        thickness=25,
    )

    layout_config = dict(
        autosize=True,
        showlegend=True,
        paper_bgcolor="white",
        margin=dict(l=7, r=7, t=10, b=7, pad=0),
    )

    geo_config = dict(
        geo=dict(
            projection_type="orthographic",
            showland=True,
            showcountries=True,
            showocean=True,
            countrywidth=0.5,
            landcolor="rgb(243, 243, 243)",
            oceancolor="rgb(204, 229, 255)",
            showcoastlines=True,
            coastlinewidth=1,
            showframe=False,
            bgcolor="rgba(0,0,0,0)",
            resolution=110,
            showsubunits=True,
            subunitcolor="rgb(255, 255, 255)",
            showrivers=True,
            rivercolor="rgb(204, 229, 255)",
            riverwidth=1,
            projection=dict(scale=1.0),  # Adjust scale to show full globe
        )
    )

    # Create frames for animation
    frames = []
    for i in range(len(lon)):

        ticktext = [
            f"T+{i:d} min"
            for i in np.arange(0, time_nums[-1] + time_step, time_step)
            .astype(int)
            .tolist()
        ]
        frame = go.Frame(
            data=[
                go.Scattergeo(
                    lon=lon[: i + 1],
                    lat=lat[: i + 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=time_nums[: i + 1],
                        colorscale="Viridis",
                        opacity=0.8,
                        showscale=True,
                        colorbar=dict(
                            title=dict(
                                text=f"Minutes After First Detected Possible Impact {first_impact_time}",
                                side="top",
                                font=dict(size=12, color="black"),
                            ),
                            ticktext=ticktext,
                            tickvals=np.arange(0, time_nums[-1] + time_step, time_step),
                            **color_bar_config,
                        ),
                    ),
                    name="Variant Impact Locations",
                    hovertext=[
                        f"Time: {t.iso}<br>Lon: {lo:.2f}°<br>Lat: {la:.2f}°<br>+{mins:.1f} min"
                        for t, lo, la, mins in zip(
                            times[: i + 1],
                            lon[: i + 1],
                            lat[: i + 1],
                            time_nums[: i + 1],
                        )
                    ],
                )
            ],
            name=str(i),
            layout=layout_config,
        )
        frames.append(frame)

    # Create the figure with initial state
    fig = go.Figure(
        data=[
            go.Scattergeo(
                lon=[lon[0]],
                lat=[lat[0]],
                mode="markers",
                marker=dict(
                    size=8,
                    color=[time_nums[0]],
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text=f"Minutes After First Detected Possible Impact {first_impact_time}",
                            side="top",
                            font=dict(size=12, color="black"),
                        ),
                        ticktext=[
                            f"T+{i:d} min"
                            for i in np.arange(0, time_nums[-1] + time_step, time_step)
                            .astype(int)
                            .tolist()
                        ],
                        tickvals=np.arange(0, time_nums[-1] + time_step, time_step),
                        **color_bar_config,
                    ),
                ),
                name="Variant Impact Locations",
                hovertext=f"Time: {times[0].iso}<br>Lon: {lon[0]:.2f}°<br>Lat: {lat[0]:.2f}°<br>+{time_nums[0]:.1f} min",
            )
        ],
        frames=frames,
        layout=layout_config,
    )

    if title is None:
        title = "Risk Corridor"

    # Update layout
    fig.update_layout(
        title=dict(
            text=title, xanchor="left", yanchor="top", font=dict(size=16, color="black")
        ),
        **geo_config,
        **layout_config,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=-0.1,
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="⏸",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        # Add slider for manual control
        sliders=[
            {
                "currentvalue": {"prefix": "Variant: "},
                "pad": {"t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": -0.05,
                "steps": [
                    dict(
                        args=[
                            [str(i)],
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        label=str(i),
                        method="animate",
                    )
                    for i in range(len(frames))
                ],
            }
        ],
    )

    return fig
