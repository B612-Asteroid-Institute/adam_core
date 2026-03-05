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
    GeodeticCoordinates,
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
) -> dict[str, Orbits]:
    """
    Sets variants propagated after their impact time to their impact coordinates on
    the surface of the colliding body for stopping collision events. Non-stopping
    close-approach events are preserved as propagated.

    Note: Due to the nature of the collision detection code, some variants may already
    be inside the sphere of the colliding body when a stopping event is detected.
    In these cases, the variants' distance from the body center is set to the body
    radius. Results are intended for visualization, not high-fidelity impact prediction.

    Parameters
    ----------
    propagated_variants: Orbits
        The propagated variants to cleanse.
    impacts: CollisionEvent
        The impacts detected within the variants.

    Returns
    -------
    dict[str, Orbits]
        A dictionary containing the prepared variants, with keys:
          - "Non-Impacting": Variants with no collision events
          - "{BODY} Impacting": Variants with stopping collisions for that body
          - "{BODY} Close-Approaching": Variants with non-stopping collisions for that body
    """
    assert propagated_variants.coordinates.frame == "ecliptic"

    if len(impacts) == 0:
        return {"Non-Impacting": propagated_variants}

    colliding_bodies = impacts.collision_object.code.unique().to_pylist()
    prepared_variants = {}

    impact_events = impacts.apply_mask(pc.equal(impacts.stopping_condition, True))
    close_approach_events = impacts.apply_mask(
        pc.equal(impacts.stopping_condition, False)
    )

    impact_variant_ids = set(impact_events.variant_id.unique().to_pylist())
    close_variant_ids = set(close_approach_events.variant_id.unique().to_pylist())
    all_event_variant_ids = impact_variant_ids.union(close_variant_ids)

    all_event_variant_ids_arrow = pa.array(
        list(all_event_variant_ids), type=propagated_variants.orbit_id.type
    )

    non_impacting_variants = propagated_variants.apply_mask(
        pc.invert(pc.is_in(propagated_variants.orbit_id, all_event_variant_ids_arrow))
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

        impacts_on_colliding_body = impact_events.apply_mask(
            pc.equal(impact_events.collision_object.code, colliding_body)
        )

        if len(impacts_on_colliding_body) > 0:
            impacting_variants_body = propagated_variants.apply_mask(
                pc.is_in(
                    propagated_variants.orbit_id, impacts_on_colliding_body.variant_id
                )
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

                    # The surface-clamp applies to stopping collisions only.
                    collision_coordinates = collision_coordinates.set_column(
                        "rho", pa.repeat(radius / KM_P_AU, len(collision_coordinates))
                    )

                    # Override impact coordinate times with propagated timestamps so
                    # Earth-fixed points rotate correctly in geocentric ecliptic view.
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

            prepared_variants[f"{colliding_body} Impacting"] = impacting_variants_body

        close_approaches_on_colliding_body = close_approach_events.apply_mask(
            pc.equal(close_approach_events.collision_object.code, colliding_body)
        )
        close_approach_variant_ids = set(
            close_approaches_on_colliding_body.variant_id.unique().to_pylist()
        )
        impacting_variant_ids_for_body = set(
            impacts_on_colliding_body.variant_id.unique().to_pylist()
        )
        close_approach_variant_ids.difference_update(impacting_variant_ids_for_body)

        if close_approach_variant_ids:
            close_approach_variant_ids_arrow = pa.array(
                list(close_approach_variant_ids), type=propagated_variants.orbit_id.type
            )
            prepared_variants[f"{colliding_body} Close-Approaching"] = (
                propagated_variants.apply_mask(
                    pc.is_in(
                        propagated_variants.orbit_id, close_approach_variant_ids_arrow
                    )
                )
            )

    return prepared_variants


def _closest_event_time_window(
    impacts: CollisionEvent,
    focus_body: Optional[Literal["EARTH", "MOON"]] = None,
    window_percentiles: Tuple[float, float] = (10.0, 90.0),
    window_padding: float = 10.0,
) -> tuple[float, float]:
    """
    Compute a focused visualization window centered on closest events.
    """
    if len(impacts) == 0:
        raise ValueError("No collision events available for visualization.")

    body_codes = impacts.collision_object.code.to_numpy(zero_copy_only=False)
    available_bodies = sorted(set(body_codes.tolist()))

    if focus_body is None:
        selected_body = "MOON" if "MOON" in available_bodies else available_bodies[0]
    else:
        if focus_body not in available_bodies:
            raise ValueError(
                f"Requested focus_body '{focus_body}' not found in collision events: {available_bodies}"
            )
        selected_body = focus_body

    stopping_condition = impacts.stopping_condition.to_numpy(zero_copy_only=False)
    rho = impacts.collision_coordinates.rho.to_numpy(zero_copy_only=False)
    times_mjd = impacts.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    variant_ids = impacts.variant_id.to_numpy(zero_copy_only=False)

    body_mask = body_codes == selected_body
    close_approach_mask = np.logical_and(body_mask, np.logical_not(stopping_condition))
    candidate_mask = close_approach_mask

    # If no non-stopping events are present for this body, fall back to all events.
    if not np.any(candidate_mask):
        candidate_mask = body_mask

    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) == 0:
        raise ValueError(
            f"No collision events available for focus body '{selected_body}'."
        )

    # Keep one event per variant: the closest event for that variant.
    closest_indices_by_variant: dict[str, int] = {}
    closest_rho_by_variant: dict[str, float] = {}

    for idx in candidate_indices:
        variant_id = variant_ids[idx]
        event_rho = rho[idx]
        previous_rho = closest_rho_by_variant.get(variant_id)
        if previous_rho is None or event_rho < previous_rho:
            closest_rho_by_variant[variant_id] = event_rho
            closest_indices_by_variant[variant_id] = idx

    closest_indices = np.array(list(closest_indices_by_variant.values()), dtype=int)
    closest_times_mjd = times_mjd[closest_indices]

    percentile_low, percentile_high = window_percentiles
    if not (0.0 <= percentile_low <= percentile_high <= 100.0):
        raise ValueError(
            "window_percentiles must satisfy 0 <= low <= high <= 100."
        )

    start_mjd = np.percentile(closest_times_mjd, percentile_low)
    end_mjd = np.percentile(closest_times_mjd, percentile_high)
    padding_days = window_padding / 60 / 24
    start_mjd -= padding_days
    end_mjd += padding_days

    logger.info(
        "Closest-approach window for %s: %.6f to %.6f MJD (%d variants).",
        selected_body,
        start_mjd,
        end_mjd,
        len(closest_indices_by_variant),
    )

    return start_mjd, end_mjd


def generate_impact_visualization_data(
    orbit: Orbits,
    variant_orbits: VariantOrbits,
    impacts: CollisionEvent,
    propagator: Propagator,
    time_step: float = 5,
    time_range: float = 60,
    window_mode: Literal["event_range", "closest_approach"] = "event_range",
    focus_body: Optional[Literal["EARTH", "MOON"]] = None,
    window_percentiles: Tuple[float, float] = (10.0, 90.0),
    window_padding: float = 10.0,
    target_frames: Optional[int] = 180,
    min_time_step: float = 1.0,
    max_processes: Optional[int] = None,
) -> Tuple[Timestamp, Orbits, dict[str, Orbits]]:
    """
    Generates the data for collision-event visualization animation (impacts and/or
    close-approaches) for supported planetary bodies.

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
    window_mode: Literal["event_range", "closest_approach"]
        Time window selection mode. "event_range" uses full first/last event range
        with time_range padding. "closest_approach" focuses on closest events.
    focus_body: Optional[Literal["EARTH", "MOON"]]
        Body used by closest_approach window mode. Defaults to MOON if present.
    window_percentiles: Tuple[float, float]
        Percentiles used to bracket closest-event times in closest_approach mode.
    window_padding: float
        Padding (minutes) added to both sides of closest_approach window.
    target_frames: Optional[int]
        Target frame count for closest_approach mode; time_step may be increased
        to keep frame count near this value.
    min_time_step: float
        Minimum time step (minutes) allowed for closest_approach auto-scaling.
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

    if len(impacts) == 0:
        raise ValueError(
            "No collision events found. Provide impacts/close-approaches to visualize."
        )

    effective_time_step = time_step
    if window_mode == "event_range":
        event_times = impacts.coordinates.time.mjd().to_numpy(zero_copy_only=False)
        first_impact_time = np.min(event_times)
        last_impact_time = np.max(event_times)
        start_mjd = first_impact_time - time_range / 60 / 24
        end_mjd = last_impact_time + time_range / 60 / 24
    elif window_mode == "closest_approach":
        start_mjd, end_mjd = _closest_event_time_window(
            impacts,
            focus_body=focus_body,
            window_percentiles=window_percentiles,
            window_padding=window_padding,
        )

        if target_frames is not None and target_frames > 1:
            window_minutes = (end_mjd - start_mjd) * 24 * 60
            auto_time_step = max(min_time_step, window_minutes / (target_frames - 1))
            effective_time_step = max(time_step, auto_time_step)
    else:
        raise ValueError(
            f"Unknown window_mode '{window_mode}'. Expected 'event_range' or 'closest_approach'."
        )

    if end_mjd <= start_mjd:
        end_mjd = start_mjd + max(effective_time_step, min_time_step) / 60 / 24

    # Create propagation times around the range of impact times
    mjds = np.arange(
        start_mjd,
        end_mjd + effective_time_step / 60 / 24,
        effective_time_step / 60 / 24,
    )
    mjds = mjds - np.mod(mjds, effective_time_step / 60 / 24)
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


def _sample_variant_group(
    variants: Orbits, sample_fraction: Optional[float], log_label: str
) -> Orbits:
    """
    Sample orbit_id groups in an Orbits table by fraction.
    """
    if sample_fraction is None:
        return variants

    orbit_ids = variants.orbit_id.unique()
    numpy_orbit_ids = orbit_ids.to_numpy(zero_copy_only=False)

    if len(numpy_orbit_ids) == 0:
        logger.info(f"No variants available to sample for '{log_label}'.")
        orbit_ids_sample = numpy_orbit_ids
    else:
        sample_size = np.ceil(len(numpy_orbit_ids) * sample_fraction).astype(int)
        sample_size = min(sample_size, len(numpy_orbit_ids))
        orbit_ids_sample = np.random.choice(
            numpy_orbit_ids,
            sample_size,
            replace=False,
        )
        logger.info(
            f"Sampled {len(orbit_ids_sample)} variants for '{log_label}' out of {len(numpy_orbit_ids)}"
        )

    arrow_orbit_ids_sample = pa.array(orbit_ids_sample, type=variants.orbit_id.type)
    return variants.__class__.from_pyarrow(
        variants.apply_mask(
            pc.is_in(variants.orbit_id, arrow_orbit_ids_sample)
        ).table.combine_chunks()
    )


def _group_plot_config(group_name: str) -> tuple[str, str, str, int]:
    """
    Return group_type, body_name, marker_color, marker_size for a group key.
    """
    if group_name == "Non-Impacting":
        return "non-impacting", "All", "#5685C3", 1

    if group_name.endswith(" Close-Approaching"):
        body_name = group_name.removesuffix(" Close-Approaching").capitalize()
        return "close-approach", body_name, "#F2B134", 2

    if group_name.endswith(" Impacting"):
        body_name = group_name.removesuffix(" Impacting").capitalize()
        return "impacting", body_name, "red", 2

    # Backward compatibility for older keys like "EARTH"/"MOON".
    body_name = group_name.capitalize()
    return "impacting", body_name, "red", 2


def plot_impact_simulation(
    propagation_times: Timestamp,
    propagated_best_fit_orbit: Orbits,
    propagated_variants: dict[str, Orbits],
    impacts: CollisionEvent,
    grid: bool = True,
    title: str = None,
    logo: bool = True,
    show_impacting: bool = True,
    show_close_approaching: bool = True,
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
    show_close_approaching: bool, optional
        Whether to show variants with non-stopping close-approach events.
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

    all_variant_ids = set()
    for variants_group in propagated_variants.values():
        all_variant_ids.update(variants_group.orbit_id.unique().to_pylist())
    num_variants = max(len(all_variant_ids), 1)

    collision_bodies = sorted(set(impacts.collision_object.code.unique().to_pylist()))
    sampled_variants = {}
    for k, v in propagated_variants.items():
        group_type, _, _, _ = _group_plot_config(k)
        if group_type == "non-impacting":
            sampled_variants[k] = _sample_variant_group(v, sample_non_impactors, k)
        else:
            sampled_variants[k] = _sample_variant_group(v, sample_impactors, k)

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

        body_event_counts = {}
        for body_key in collision_bodies:
            body_events_up_to_current_time = all_impacts_up_to_current_time.apply_mask(
                pc.equal(all_impacts_up_to_current_time.collision_object.code, body_key)
            )

            impacting_ids = set(
                body_events_up_to_current_time.apply_mask(
                    pc.equal(body_events_up_to_current_time.stopping_condition, True)
                )
                .variant_id.unique()
                .to_pylist()
            )
            close_approach_ids = set(
                body_events_up_to_current_time.apply_mask(
                    pc.equal(body_events_up_to_current_time.stopping_condition, False)
                )
                .variant_id.unique()
                .to_pylist()
            )
            close_approach_ids.difference_update(impacting_ids)

            body_event_counts[body_key] = {
                "impacting": len(impacting_ids),
                "close-approach": len(close_approach_ids),
            }

        # Create the data for the frame
        data = []
        for k, v in sampled_variants.items():

            group_type, body_name, color, size = _group_plot_config(k)

            if group_type == "non-impacting" and not show_non_impacting:
                continue

            if group_type == "impacting" and not show_impacting:
                continue

            if group_type == "close-approach" and not show_close_approaching:
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

            if group_type == "non-impacting":
                name = "Non-Impacting"
            elif group_type == "close-approach":
                name = f"{body_name} Close-Approaching"
            else:
                name = f"{body_name} Impacting"

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
        for body_name, counts in body_event_counts.items():
            impact_probability = counts["impacting"] / num_variants * 100
            text += (
                f"<br>{body_name.capitalize()} Impacts: {counts['impacting']} of {num_variants} Variants"
                f"<br>{body_name.capitalize()} Impact Probability: {impact_probability:.3f}%"
            )
            if counts["close-approach"] > 0:
                close_probability = counts["close-approach"] / num_variants * 100
                text += (
                    f"<br>{body_name.capitalize()} Close Approaches: {counts['close-approach']} of {num_variants} Variants"
                    f"<br>{body_name.capitalize()} Close-Approach Probability: {close_probability:.3f}%"
                )

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
                y=0.0,
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
    # Filter to only include Earth collision events (impacts or close approaches)
    impacts = impacts.apply_mask(pc.equal(impacts.collision_object.code, "EARTH"))
    if len(impacts) == 0:
        raise ValueError(
            "No Earth collision events found. Other collision objects are not supported yet."
        )

    # Transform impact coordinates to ITRF93 Geodetic Coordinates
    geodetic_impacts = transform_coordinates(
        impacts.collision_coordinates,
        representation_out=GeodeticCoordinates,
        frame_out="itrf93",
        origin_out=OriginCodes.EARTH,
    )

    # Sort all data by time
    times = geodetic_impacts.time.to_astropy()
    time_order = np.argsort(times.mjd)
    lon = geodetic_impacts.lon.to_numpy(zero_copy_only=False)[time_order]
    lat = geodetic_impacts.lat.to_numpy(zero_copy_only=False)[time_order]
    times = times[time_order]

    # Convert times to minutes since first impact
    time_nums = (times.mjd - times.mjd.min()) * 24 * 60

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
        orientation="v",  # Vertical orientation for right side
        x=1.02,
        y=0.5,  # Centered vertically
        xanchor="left",
        yanchor="middle",
        thickness=25,
        len=0.8,
    )

    # Create frames for animation
    frames = []
    for i in range(len(lon)):
        # Create ticks for min, middle (when applicable), and max
        current_max_time = time_nums[i]

        if i == 0:
            # First frame: only show T+0 min
            tick_values = [0]
            tick_labels = ["T+0 min"]
        elif i == 1:
            # Second frame: show T+0 min and max
            tick_values = [0, current_max_time]
            tick_labels = ["T+0 min", f"T+{current_max_time:.0f} min"]
        else:
            # Third frame and beyond: show T+0 min, middle, and max
            middle_time = current_max_time / 2
            tick_values = [0, middle_time, current_max_time]
            tick_labels = [
                "T+0 min",
                f"T+{middle_time:.0f} min",
                f"T+{current_max_time:.0f} min",
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
                                side="right",
                                font=dict(size=12, color="black"),
                            ),
                            ticktext=tick_labels,
                            tickvals=tick_values,
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
                            side="right",
                            font=dict(size=12, color="black"),
                        ),
                        ticktext=["T+0 min", f"T+{time_nums[-1]:.0f} min"],
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
                y=-0.20,  # Moved logo up slightly
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
        margin=dict(
            l=7, r=120, t=30, b=30, pad=0
        ),  # Increased right margin for vertical colorbar
        **plot_config,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.03,
                y=-0.12,  # Positioned between slider and colorbar
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
