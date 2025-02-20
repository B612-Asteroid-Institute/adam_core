from typing import Optional

import numpy as np

from ..coordinates import OriginCodes
from ..orbits import Orbits
from ..propagator import Propagator
from ..time import Timestamp
from ..utils.plots.logos import AsteroidInstituteLogoDark, get_logo_base64
from ..utils.spice import get_perturber_state

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Please install adam_core[plots] to use this feature.")


def plot_orbit(
    orbit: Orbits,
    propagator: Propagator,
    start_time: Optional[Timestamp] = None,
    logo: bool = True,
) -> go.Figure:
    """
    Plot an orbit.

    Parameters
    ----------
    orbit: Orbits
        The orbit to plot.
    propagator: Propagator
        The propagator to use to propagate the orbit.
    start_time: Optional[Timestamp]
        The start time to use for the propagation.
    logo: bool, optional
        Whether to add the Asteroid Institute logo to the plot.
    """
    # Get the period so we know how long to propagate in cartesian space
    keplerian = orbit.coordinates.to_keplerian()

    period_index = int(np.argmax(keplerian.P))
    period = keplerian.P[period_index]

    if start_time is None:
        start_time = orbit.coordinates.time[period_index]

    # Add the period in days to the orbit.coordinates.time
    propagation_end_date = (
        start_time.add_days(np.ceil(period).astype(int)).mjd()[0].as_py()
    )

    sample_dates_mjd = np.arange(
        start_time.mjd()[0].as_py(), propagation_end_date + 5, 5
    )
    sample_dates = Timestamp.from_mjd(sample_dates_mjd, scale=start_time.scale)

    # Propagate the orbit in cartesian space
    propagated_orbits = propagator.propagate_orbits(orbit, sample_dates)

    max_r = np.max(np.abs(propagated_orbits.coordinates.r))

    # Give a default max_r if the orbits are smaller than the inner planets
    max_r = np.max((max_r, 4))

    # Render the planets conditionally based on max r of the propagated orbits
    conditional_planet_distances = (
        ("Jupiter", 5, OriginCodes.JUPITER_BARYCENTER),
        ("Saturn", 10, OriginCodes.SATURN_BARYCENTER),
        ("Uranus", 15, OriginCodes.URANUS_BARYCENTER),
        ("Neptune", 20, OriginCodes.NEPTUNE_BARYCENTER),
    )

    planet_states = {
        "Earth": get_perturber_state(OriginCodes.EARTH, sample_dates),
        "Mars": get_perturber_state(OriginCodes.MARS_BARYCENTER, sample_dates),
        "Venus": get_perturber_state(OriginCodes.VENUS, sample_dates),
        "Mercury": get_perturber_state(OriginCodes.MERCURY, sample_dates),
    }

    for planet, distance, origin_code in conditional_planet_distances:
        if distance < max_r:
            planet_states[planet] = get_perturber_state(origin_code, sample_dates)

    traces = []

    # Create figure with black background and appropriate styling
    fig = go.Figure(
        layout=dict(
            paper_bgcolor="black",
            plot_bgcolor="black",
            scene=dict(
                bgcolor="black",
            ),
        )
    )

    # Define a color scheme for planets with transparency
    planet_colors = {
        "Mercury": "rgba(160, 82, 45, 0.6)",  # Brown with alpha
        "Venus": "rgba(218, 165, 32, 0.6)",  # Goldenrod with alpha
        "Earth": "rgba(65, 105, 225, 0.6)",  # Royal Blue with alpha
        "Mars": "rgba(205, 92, 92, 0.6)",  # Indian Red with alpha
        "Jupiter": "rgba(222, 184, 135, 0.6)",  # Burlywood with alpha
        "Saturn": "rgba(244, 164, 96, 0.6)",  # Sandy Brown with alpha
        "Uranus": "rgba(135, 206, 235, 0.6)",  # Sky Blue with alpha
        "Neptune": "rgba(30, 144, 255, 0.6)",  # Dodger Blue with alpha
    }

    for planet, state in planet_states.items():
        traces.append(
            go.Scatter3d(
                x=state.x,
                y=state.y,
                z=state.z,
                mode="lines",
                name=planet,
                line=dict(width=1, color=planet_colors[planet]),
            )
        )

    # Use bright green for all non-planet orbits
    orbit_color = "#00FF00"  # Bright green

    for orbit_id in orbit.orbit_id.unique():
        propagated_orbit = propagated_orbits.select("orbit_id", orbit_id)
        cartesian = propagated_orbit.coordinates
        isot_time = propagated_orbit.coordinates.time.to_astropy().isot
        traces.append(
            go.Scatter3d(
                x=cartesian.x,
                y=cartesian.y,
                z=cartesian.z,
                mode="lines",
                name=f"{propagated_orbit.orbit_id[0].as_py()} {propagated_orbit.object_id[0].as_py()}",
                hovertext=isot_time,
                line=dict(width=2, color=orbit_color),
            )
        )

    # For now we are always heliocentric, so plot the sun as a sphere
    traces.append(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            name="Sun",
            marker=dict(size=1, color="yellow"),
        )
    )

    for trace in traces:
        fig.add_trace(trace)

    max_r_padded = max_r * 1.1

    # Add logo if requested
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

    fig.update(
        layout=dict(
            scene=dict(
                xaxis=dict(
                    range=[-max_r_padded, max_r_padded],
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    showbackground=False,
                    color="white",
                ),
                yaxis=dict(
                    range=[-max_r_padded, max_r_padded],
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    showbackground=False,
                    color="white",
                ),
                zaxis=dict(
                    range=[-max_r_padded, max_r_padded],
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    showbackground=False,
                    color="white",
                ),
                xaxis_title="x [au]",
                yaxis_title="y [au]",
                zaxis_title="z [au]",
            ),
            font=dict(color="white"),  # Make all text white
            images=images,  # Add the logo images
        )
    )

    return fig
