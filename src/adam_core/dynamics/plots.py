import logging
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
from .impacts import CollisionEvent

EARTH_RADIUS_KM = c.R_EARTH_EQUATORIAL * KM_P_AU * 0.999
MOON_RADIUS_KM = 1738.1

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Please install adam_core[plots] to use this feature.")


def prepare_propagated_variants(
    propagated_variants: Orbits, impacts: CollisionEvent
) -> dict[Literal["Non-Impacting", "EARTH", "MOON"], Orbits]:
    """
    Sets variants propagated after their impact time to their impact coordinates on the surface of the colliding body. If the colliding body is the Earth,
    the variants are set to the impact coordinates on the surface of the Earth. If the colliding body is the Moon, the variants are set to the impact coordinates
    in the lunarcentric frame (but not fixed to the surface of the Moon).

    Note: Due to the nature of the impact detection code, some of the variants may already be inside the sphere of the Earth when the impact is detected.
    In these cases, the variants' distance from the geocenter is set to the radius of the Earth. The results of this function should not be used
    for high-fidelity impact predictions but instead for visualizations of the approximate impact corridor.

    Parameters
    ----------
    propagated_variants: Orbits
        The propagated variants to cleanse.
    impacts: CollisionEvent
        The impacts detected within the variants.

    Returns
    -------
    dict[Literal["Non-Impacting", "EARTH", "MOON"], Orbits]
        A dictionary containing the prepared variants, with keys:
          - "Non-Impacting": Variants that don't impact any body
          - "EARTH": Variants that impact Earth (if any)
          - "MOON": Variants that impact the Moon (if any)
        Only bodies that appear in the impacts will be included as keys.
    """
    assert propagated_variants.coordinates.frame == "ecliptic"

    colliding_bodies = impacts.collision_object.code.unique().to_pylist()
    prepared_variants = {}

    # Remove the non-impacting variants
    impacting_variants = propagated_variants.apply_mask(
        pc.is_in(propagated_variants.orbit_id, impacts.variant_id)
    )
    non_impacting_variants = propagated_variants.apply_mask(
        pc.invert(pc.is_in(propagated_variants.orbit_id, impacts.variant_id))
    )
    prepared_variants["Non-Impacting"] = non_impacting_variants

    for colliding_body in colliding_bodies:

        if colliding_body == "EARTH":
            radius = EARTH_RADIUS_KM
        elif colliding_body == "MOON":
            radius = MOON_RADIUS_KM
        else:
            raise ValueError(
                f"CollisionEvent visualizations are currently supported for the Earth and Moon. {colliding_body} is not supported."
            )

        impacts_on_colliding_body = impacts.apply_mask(
            pc.equal(impacts.collision_object.code, colliding_body)
        )
        impacting_variants_body = impacting_variants.apply_mask(
            pc.is_in(impacting_variants.orbit_id, impacts_on_colliding_body.variant_id)
        )

        for impact in impacts_on_colliding_body:

            post_impact_mask = pc.and_(
                pc.is_in(impacting_variants_body.orbit_id, impact.variant_id),
                pc.greater_equal(
                    impacting_variants_body.coordinates.time.mjd(),
                    impact.coordinates.time.rescale(
                        impacting_variants_body.coordinates.time.scale
                    ).mjd()[0],
                ),
            )

            impacting_variants_body_correct = impacting_variants_body.apply_mask(
                pc.invert(post_impact_mask)
            )
            impacting_variants_body_incorrect = impacting_variants_body.apply_mask(
                post_impact_mask
            )

            if len(impacting_variants_body_incorrect) > 0:
                collision_coordinates = impact.collision_coordinates
                collision_coordinates = qv.concatenate(
                    [
                        collision_coordinates
                        for _ in range(len(impacting_variants_body_incorrect))
                    ]
                )

                # Hack: In some cases, due to the time step of the propagation, the variants may already
                # be inside the sphere of the colliding body when the impact is detected. In these cases, the variants'
                # distance from the bodycenter is set its radius.
                collision_coordinates = collision_coordinates.set_column(
                    "rho", pa.repeat(radius / KM_P_AU, len(collision_coordinates))
                )

                # Override the time of the impact coordinates to the time of the propagated variants beyond the impact time, we do
                # this so we can then calculate position of these locations on the surface of the Earth as the Earth rotates.
                collision_coordinates = collision_coordinates.set_column(
                    "time", impacting_variants_body_incorrect.coordinates.time
                )

                geocentric_coordinates = transform_coordinates(
                    collision_coordinates,
                    representation_out=CartesianCoordinates,
                    frame_out="ecliptic",
                    origin_out=OriginCodes.EARTH,
                )
                impacting_variants_body_incorrect = (
                    impacting_variants_body_incorrect.set_column(
                        "coordinates",
                        geocentric_coordinates,
                    )
                )

            impacting_variants_body = qv.concatenate(
                [impacting_variants_body_correct, impacting_variants_body_incorrect]
            )

        prepared_variants[colliding_body] = impacting_variants_body

    return prepared_variants


def generate_impact_visualization_data(
    orbit: Orbits,
    variant_orbits: VariantOrbits,
    impacts: CollisionEvent,
    propagator: Propagator,
    time_step: float = 5,
    time_range: float = 60,
    max_processes: Optional[int] = None,
) -> Tuple[Timestamp, Orbits, dict[str, Orbits]]:
    """
    Generates the data for the impact visualization animation. The user should be careful
    to only send in collision events that correspond to an impact with a planetary body or moon.
    Non-impacting collisions such as close approaches have not been tested for this function.

    CollisionEvents visualizations are currently supported for the Earth and Moon.

    Parameters
    ----------
    orbit: Orbits
        The nominal best-fit orbit to propagate.
    variant_orbits: VariantOrbits
        The variants to propagate.
    impacts: CollisionEvent
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
    Tuple[Timestamp, Orbits, dict[str, Orbits]]
        The propagation times, the propagated nominal best-fit orbit and the propagated variants.
    """
    if pc.any(
        pc.invert(
            pc.or_(
                pc.equal(impacts.collision_object.code, "EARTH"),
                pc.equal(impacts.collision_object.code, "MOON"),
            )
        )
    ).as_py():
        raise ValueError(
            "CollisionEvents visualizations are currently supported for the Earth and Moon."
        )

    # Calculate the range of impact times
    impact_times = impacts.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    first_impact_time = np.min(impact_times)
    last_impact_time = np.max(impact_times)

    # Create propagation times around the range of impact times
    mjds = np.arange(
        first_impact_time - time_range / 60 / 24,
        last_impact_time + time_range / 60 / 24 + time_step / 60 / 24,
        time_step / 60 / 24,
    )
    mjds = mjds - np.mod(mjds, time_step / 60 / 24)
    propagation_times = Timestamp.from_mjd(mjds, scale=impacts.coordinates.time.scale)

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

    propagated_variants = prepare_propagated_variants(propagated_variants, impacts)

    for k, v in propagated_variants.items():
        propagated_variants[k] = v.sort_by(
            ["coordinates.time.days", "coordinates.time.nanos"]
        )

    return propagation_times, propagated_orbit, propagated_variants


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
    x, y, z = create_sphere(EARTH_RADIUS_KM * 0.999, offset=earth_state.r[0] * KM_P_AU)

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
                    legendgroup="Earth",
                )
            )

    earth_surface = go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=1,
        colorscale=[[0, "#015294"], [1, "#015294"]],
        showscale=False,
        name="Earth",
        legendgroup="Earth",
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
        opacity=1.0,
        colorscale=[[0, "#A9A9A9"], [1, "#A9A9A9"]],
        showscale=False,
        name="Moon",
        showlegend=True,
        visible=show,
    )


def plot_impact_simulation(
    propagation_times: Timestamp,
    propagated_best_fit_orbit: Orbits,
    propagated_variants: dict[str, Orbits],
    impacts: CollisionEvent,
    grid: bool = True,
    title: str = None,
    logo: bool = True,
    show_impacting: bool = True,
    show_non_impacting: bool = True,
    show_best_fit: bool = True,
    show_earth: bool = True,
    show_moon: bool = True,
    sample_impactors: Optional[float] = None,
    sample_non_impactors: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    Plot the impact simulation.

    Parameters
    ----------
    propagated_best_fit_orbit: Orbits
        The propagated best-fit orbit.
    propagated_variants: Orbits
        The propagated variants.
    impacts: CollisionEvent
        The impacts detected within the variants.
    grid: bool, optional
        Whether to add the grid to the plot.
    title: str, optional
        The title of the plot.
    logo: bool, optional
        Whether to add the Asteroid Institute logo to the plot.
    show_impacting: bool, optional
        Whether to show the impacting variants.
    show_non_impacting: bool, optional
        Whether to show the non-impacting variants.
    show_best_fit: bool, optional
        Whether to show the best-fit orbit.
    show_earth: bool, optional
        Whether to show the Earth.
    show_moon: bool, optional
        Whether to show the Moon.
    sample_impactors: Optional[float], optional
        Randomly sample the impactors for plotting. Should be between 0 and 1.
    sample_non_impactors: Optional[float], optional
        Randomly sample the non-impactors for plotting. Should be between 0 and 1.
    height: int, optional
        The height of the plot.
    width: int, optional
        The width of the plot.

    Returns
    -------
    go.Figure
        The impact simulation plot.
    """
    propagation_times_isot = propagation_times.to_astropy().isot

    num_variants = 0
    impact_count = {}
    sampled_variants = {}
    for k, v in propagated_variants.items():
        num_variants += len(v.orbit_id.unique())

        if k == "Non-Impacting":
            if sample_non_impactors is not None:
                orbit_ids = v.orbit_id.unique()
                numpy_orbit_ids = orbit_ids.to_numpy(
                    zero_copy_only=False
                )  # Convert to NumPy

                if len(numpy_orbit_ids) == 0:
                    orbit_ids_sample = numpy_orbit_ids  # Already an empty numpy array with correct dtype
                    logger.info("No non-impacting variants available to sample.")
                else:
                    sample_size = np.ceil(
                        len(numpy_orbit_ids) * sample_non_impactors
                    ).astype(int)
                    sample_size = min(
                        sample_size, len(numpy_orbit_ids)
                    )  # Ensure sample_size <= population
                    orbit_ids_sample = np.random.choice(
                        numpy_orbit_ids,
                        sample_size,
                        replace=False,
                    )
                    logger.info(
                        f"Sampled {len(orbit_ids_sample)} non-impacting variants out of {len(numpy_orbit_ids)}"
                    )

                # Create Arrow array with explicit type
                arrow_orbit_ids_sample = pa.array(
                    orbit_ids_sample, type=v.orbit_id.type
                )
                sampled_variants[k] = v.__class__.from_pyarrow(
                    v.apply_mask(
                        pc.is_in(v.orbit_id, arrow_orbit_ids_sample)
                    ).table.combine_chunks()
                )
            else:
                sampled_variants[k] = v

        if k != "Non-Impacting":
            impact_count[k] = 0

            if sample_impactors is not None:
                orbit_ids = v.orbit_id.unique()
                numpy_orbit_ids = orbit_ids.to_numpy(
                    zero_copy_only=False
                )  # Convert to NumPy

                if len(numpy_orbit_ids) == 0:
                    orbit_ids_sample = numpy_orbit_ids  # Empty array with correct dtype
                    logger.info(f"No impacting variants for '{k}' available to sample.")
                else:
                    sample_size = np.ceil(
                        len(numpy_orbit_ids) * sample_impactors
                    ).astype(int)
                    sample_size = min(
                        sample_size, len(numpy_orbit_ids)
                    )  # Ensure sample_size <= population
                    orbit_ids_sample = np.random.choice(
                        numpy_orbit_ids,
                        sample_size,
                        replace=False,
                    )
                    logger.info(
                        f"Sampled {len(orbit_ids_sample)} impacting variants for '{k}' out of {len(numpy_orbit_ids)}"
                    )

                # Create Arrow array with explicit type
                arrow_orbit_ids_sample = pa.array(
                    orbit_ids_sample, type=v.orbit_id.type
                )
                sampled_variants[k] = v.__class__.from_pyarrow(
                    v.apply_mask(
                        pc.is_in(v.orbit_id, arrow_orbit_ids_sample)
                    ).table.combine_chunks()
                )
            else:
                sampled_variants[k] = v

    all_potential_impactor_ids_from_impacts_table = set(
        impacts.variant_id.unique().to_pylist()
    )

    if title is None:
        prefix = ""
    else:
        prefix = f"{title}<br>"

    # Build the individual frames for the animation
    frames = []
    for i, time in enumerate(propagation_times):

        # 1. Get all impacts up to the current time
        all_impacts_up_to_current_time = impacts.apply_mask(
            pc.less_equal(impacts.coordinates.time.mjd(), time.mjd()[0])
        )

        # 2. Get the set of unique variant IDs that have impacted *anything* up to current time
        current_frame_total_unique_impacted_ids_set = set(
            all_impacts_up_to_current_time.variant_id.unique().to_pylist()
        )

        # 3. Update impact_count for each body (for title text)
        for (
            body_key
        ) in (
            impact_count.keys()
        ):  # These are "EARTH", "MOON", etc. as initialized earlier
            impacts_on_this_body_up_to_current_time = (
                all_impacts_up_to_current_time.apply_mask(
                    pc.equal(
                        all_impacts_up_to_current_time.collision_object.code, body_key
                    )
                )
            )
            impact_count[body_key] = len(
                impacts_on_this_body_up_to_current_time.variant_id.unique()
            )

        # Create the data for the frame
        data = []
        for k, v in sampled_variants.items():

            if k == "Non-Impacting" and not show_non_impacting:
                continue

            if k == "Impacting" and not show_impacting:
                continue

            v_at_time = v.apply_mask(
                pc.and_(
                    pc.equal(v.coordinates.time.days, time.days[0]),
                    pc.and_(
                        pc.less_equal(
                            v.coordinates.time.nanos, time.nanos[0].as_py() + 100000
                        ),
                        pc.greater_equal(
                            v.coordinates.time.nanos, time.nanos[0].as_py() - 100000
                        ),
                    ),
                )
            )

            if k == "Non-Impacting":
                color = "#5685C3"
                size = 1
                name = k
            else:
                color = "red"
                size = 2
                name = f"{k.lower().capitalize()} Impacting"

            x = v_at_time.coordinates.x.to_numpy(zero_copy_only=False) * KM_P_AU
            y = v_at_time.coordinates.y.to_numpy(zero_copy_only=False) * KM_P_AU
            z = v_at_time.coordinates.z.to_numpy(zero_copy_only=False) * KM_P_AU

            data.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=1,
                        showscale=False,
                    ),
                    name=name,
                    visible=True,
                    showlegend=True,
                )
            )

        # Add best-fit orbit
        if show_best_fit:
            orbit_at_time = propagated_best_fit_orbit[i].coordinates
            data.append(
                go.Scatter3d(
                    x=orbit_at_time.x.to_numpy(zero_copy_only=False) * KM_P_AU,
                    y=orbit_at_time.y.to_numpy(zero_copy_only=False) * KM_P_AU,
                    z=orbit_at_time.z.to_numpy(zero_copy_only=False) * KM_P_AU,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color="#F07620",
                        opacity=1,
                        showscale=False,
                    ),
                    name="Best-Fit Orbit",
                    visible=True,
                    showlegend=True,
                )
            )

        earth_surface, surface_traces = add_earth(time, show=show_earth)
        data.append(earth_surface)
        data.extend(surface_traces)
        data.append(add_moon(time, show=show_moon))

        text = f"{prefix}Time: {propagation_times_isot[i]}"
        for k in impact_count:
            text += f"<br>{k.lower().capitalize()} Impacts: {impact_count[k]} of {num_variants} Variants<br>{k.lower().capitalize()} Impact Probability: {impact_count[k]/num_variants * 100:.3f}%"

        frame = go.Frame(
            data=data,
            name=str(i),
            layout=dict(
                title=dict(
                    text=text,
                    x=0.01,
                    y=0.97,
                    font=dict(size=14, color="white"),
                ),
            ),
        )

        frames.append(frame)

        # If all impacting variants (from the original impacts table) have impacted by this frame, stop.
        if all_potential_impactor_ids_from_impacts_table:
            if all_potential_impactor_ids_from_impacts_table.issubset(
                current_frame_total_unique_impacted_ids_set
            ):
                break

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
                x=0.96,
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
        height=height,
        width=width,
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
        sliders=[
            dict(
                currentvalue=dict(prefix="Time: "),
                pad=dict(t=50),
                len=0.90 if not logo else 0.80,
                x=0.10,
                y=0.07,
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
        uirevision="constant",
    )

    return fig


def plot_risk_corridor(
    impacts: CollisionEvent,
    title: Optional[str] = None,
    logo: bool = True,
    height: Optional[int] = None,
    width: Optional[int] = None,
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
    height : int, optional
        The height of the plot.
    width : int, optional
        The width of the plot.

    Returns
    -------
    go.Figure
        The risk corridor plot.
    """
    # Filter to only include impacts on the Earth
    impacts = impacts.apply_mask(pc.equal(impacts.collision_object.code, "EARTH"))
    if len(impacts) == 0:
        raise ValueError(
            "No Earth impacts found. Other collision objects are not supported yet."
        )

    # Transform impact coordinates to ITRF
    impacts = impacts.set_column(
        "collision_coordinates",
        transform_coordinates(
            impacts.collision_coordinates,
            representation_out=SphericalCoordinates,
            frame_out="itrf93",
            origin_out=OriginCodes.EARTH,
        ),
    )

    # Sort all data by time
    times = impacts.collision_coordinates.time.to_astropy()
    time_order = np.argsort(times.mjd)
    lon = impacts.collision_coordinates.lon.to_numpy(zero_copy_only=False)[time_order]
    lat = impacts.collision_coordinates.lat.to_numpy(zero_copy_only=False)[time_order]
    times = times[time_order]

    # Convert times to minutes since first impact
    time_nums = (times.mjd - times.mjd.min()) * 24 * 60
    first_impact_time = times[0].iso

    # Calculate center
    center_lon = lon[0]
    center_lat = lat[0]

    plot_config = dict(
        height=height,
        width=width,
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=1,
        ),
    )
    color_bar_config = dict(
        tickmode="array",
        tickangle=0,
        orientation="h",
        x=0.05,
        y=-0.3,
        xanchor="left",
        yanchor="bottom",
        thickness=25,
        len=0.90 if not logo else 0.75,
    )

    # Create frames for animation
    frames = []
    for i in range(len(lon)):
        # Calculate time_step to have approximately 10 ticks
        time_step = int(np.ceil(time_nums[-1] / 10))
        # Round to nearest 5 or 10 for cleaner intervals
        if time_step > 10:
            time_step = int(np.ceil(time_step / 10) * 10)
        elif time_step > 5:
            time_step = int(np.ceil(time_step / 5) * 5)

        # Determine effective step for np.arange to avoid step=0.
        # If time_step is 0 (which implies time_nums[-1] == 0), use 1.
        # This ensures np.arange(0, 0 + 1, 1) yields [0] for the T+0 min case.
        effective_arange_step = time_step if time_step > 0 else 1

        ticktext = [
            f"T+{i:d} min"
            for i in np.arange(
                0, time_nums[-1] + effective_arange_step, effective_arange_step
            )
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
                            tickvals=np.arange(
                                0,
                                time_nums[-1] + effective_arange_step,
                                effective_arange_step,
                            ).tolist(),
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
                            for i in np.arange(
                                0,
                                time_nums[-1] + effective_arange_step,
                                effective_arange_step,
                            )
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
                x=0.81,
                y=-0.26,
                sizex=0.18,
                sizey=0.18,
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
                len=0.90 if not logo else 0.75,
                x=0.05,
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
