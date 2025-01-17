from typing import Optional

import numpy as np

from ..coordinates import OriginCodes
from ..orbits import Orbits
from ..propagator import Propagator
from ..time import Timestamp
from ..utils.spice import get_perturber_state

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Please install adam_core[plots] to use this feature.")


def plot_orbit(orbit: Orbits, propagator: Propagator, start_time: Optional[Timestamp] = None) -> None:
    """
    Plot an orbit.
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
    sample_dates = Timestamp.from_mjd(
        sample_dates_mjd, scale=start_time.scale
    )

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

    fig = go.Figure(layout=dict(width=800, height=800))
    
    for planet, state in planet_states.items():
        traces.append(
            go.Scatter3d(x=state.x, y=state.y, z=state.z, mode="lines", name=planet)
        )

    for orbit_id in orbit.orbit_id.unique():
        propagated_orbit = propagated_orbits.select("orbit_id", orbit_id)
        cartesian = propagated_orbit.coordinates
        isot_time = propagated_orbit.coordinates.time.to_astropy().isot
        # Make sure to add teh timestamp as mjd to the hover data
        traces.append(
            go.Scatter3d(
                x=cartesian.x,
                y=cartesian.y,
                z=cartesian.z,
                mode="lines",
                name=f"{propagated_orbit.orbit_id[0].as_py()} {propagated_orbit.object_id[0].as_py()}",
                hovertext=isot_time,
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
            marker=dict(size=10, color="yellow"),
        )
    )

    for trace in traces:
        fig.add_trace(trace)


    max_r_padded = max_r * 1.1
    fig.update(
        layout=dict(
            scene=dict(
                xaxis=dict(range=[-max_r_padded, max_r_padded]),
                yaxis=dict(range=[-max_r_padded, max_r_padded]),
                zaxis=dict(range=[-max_r_padded, max_r_padded]),
                xaxis_title="x [au]",
                yaxis_title="y [au]",
                zaxis_title="z [au]",
            )
        ),
    )
    with open("debug.json", "w") as f:
        f.write(str(fig.to_plotly_json()))

    fig.show()
