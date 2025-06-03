import logging
import multiprocessing as mp
import warnings
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.express.colors as pcolors
import plotly.graph_objects as go
import quivr as qv
import ray
from astropy.time import Time
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import griddata

from adam_core.constants import KM_P_AU, S_P_DAY
from adam_core.coordinates import CartesianCoordinates, transform_coordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.dynamics.lambert import calculate_c3, solve_lambert
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils import get_perturber_state
from adam_core.utils.iter import _iterate_chunk_indices

logger = logging.getLogger(__name__)


class LambertOutput(qv.Table):
    departure_state = CartesianCoordinates.as_column()
    arrival_state = CartesianCoordinates.as_column()
    vx_1 = qv.Float64Column()
    vy_1 = qv.Float64Column()
    vz_1 = qv.Float64Column()
    vx_2 = qv.Float64Column()
    vy_2 = qv.Float64Column()
    vz_2 = qv.Float64Column()
    origin = Origin.as_column()

    def c3(self) -> npt.NDArray[np.float64]:
        """
        Return the C3 in au^2/d^2.
        """
        return calculate_c3(
            np.array(self.table.select(["vx_1", "vy_1", "vz_1"])),
            self.departure_state.v,
        )

    def vinf(self) -> npt.NDArray[np.float64]:
        """
        Return the v infinity in au/d.
        """
        return np.linalg.norm(
            np.array(self.table.select(["vx_2", "vy_2", "vz_2"]))
            - self.departure_state.v,
            axis=1,
        )

    def time_of_flight(self) -> npt.NDArray[np.float64]:
        """
        Return the time of flight in days.
        """
        return self.arrival_state.time.mjd().to_numpy(
            zero_copy_only=False
        ) - self.departure_state.time.mjd().to_numpy(zero_copy_only=False)


def lambert_worker(
    departure_coordinates: CartesianCoordinates,
    arrival_coordinates: CartesianCoordinates,
    propagation_origin: OriginCodes,
    prograde: bool = True,
    max_iter: int = 35,
    tol: float = 1e-10,
) -> LambertOutput:
    r1 = departure_coordinates.r
    r2 = arrival_coordinates.r
    tof = arrival_coordinates.time.mjd().to_numpy(
        zero_copy_only=False
    ) - departure_coordinates.time.mjd().to_numpy(zero_copy_only=False)

    origins = Origin.from_OriginCodes(propagation_origin, size=len(r1))
    mu = origins.mu()[0]
    v1, v2 = solve_lambert(r1, r2, tof, mu, prograde, max_iter, tol)

    return LambertOutput.from_kwargs(
        departure_state=departure_coordinates,
        arrival_state=arrival_coordinates,
        vx_1=v1[:, 0],
        vy_1=v1[:, 1],
        vz_1=v1[:, 2],
        vx_2=v2[:, 0],
        vy_2=v2[:, 1],
        vz_2=v2[:, 2],
        origin=origins,
    )


lambert_worker_remote = ray.remote(lambert_worker)


def prepare_and_propagate_orbits(
    body: Union[Orbits, OriginCodes],
    earliest_launch_time: Timestamp,
    maximum_arrival_time: Timestamp,
    propagation_origin: OriginCodes = OriginCodes.SUN,
    step_size: float = 1.0,
    propagator_class: Optional[type[Propagator]] = None,
    max_processes: Optional[int] = 1,
) -> CartesianCoordinates:
    """
    Prepare and propagate orbits for a single body.
    """
    # if body is an Orbit, ensure its origin is the propagation_origin
    if isinstance(body, Orbits):
        body = body.set_column(
            "coordinates",
            transform_coordinates(
                body.coordinates,
                representation_out=CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=propagation_origin,
            ),
        )

    # create empty CartesianCoordinates
    coordinates = CartesianCoordinates.empty(
        frame="ecliptic",
    )

    times = Timestamp.from_mjd(
        np.arange(
            earliest_launch_time.rescale("tdb").mjd()[0].as_py(),
            maximum_arrival_time.rescale("tdb").mjd()[0].as_py(),
            step_size,
        ),
        scale="tdb",
    )

    # get coordinates for the body at specified times
    if isinstance(body, Orbits):
        propagator = propagator_class()
        coordinates = propagator.propagate_orbits(
            body, times, max_processes=max_processes
        ).coordinates
    else:
        coordinates = get_perturber_state(
            body, times, frame="ecliptic", origin=propagation_origin
        )

    return coordinates


def generate_porkchop_data(
    departure_coordinates: CartesianCoordinates,
    arrival_coordinates: CartesianCoordinates,
    propagation_origin: OriginCodes = OriginCodes.SUN,
    prograde: bool = True,
    max_iter: int = 35,
    tol: float = 1e-10,
    max_processes: Optional[int] = 1,
) -> LambertOutput:
    """
    Generate data for a porkchop plot by solving Lambert's problem for a grid of
    departure and arrival times.

    Parameters
    ----------
    departure_body : Union[Orbits, OriginCodes]
        The departure body.
    arrival_body : Union[Orbits, OriginCodes]
        The arrival body.
    earliest_launch_time : Timestamp
        The earliest launch time.
    maximum_arrival_time : Timestamp
        The maximum arrival time.
    propagation_origin : OriginCodes, optional
        The origin of the propagation.
    step_size : float, optional
        The step size for the porkchop plot.
    prograde : bool, optional
        If True, assume prograde motion. If False, assume retrograde motion.
    max_iter : int, optional
        The maximum number of iterations for Lambert's solver.
    tol : float, optional
        The numerical tolerance for Lambert's solver.
    propagator_class : Optional[type[Propagator]], optional
        The propagator class to use.
    max_processes : Optional[int], optional
        The maximum number of processes to use.

    Returns
    -------
    porkchop_data : LambertOutput
        The porkchop data.
    """

    # First let's make sure departure and arrival coordinates are time-ordered
    departure_coordinates = departure_coordinates.sort_by(["time.days", "time.nanos"])
    arrival_coordinates = arrival_coordinates.sort_by(["time.days", "time.nanos"])

    x, y = np.meshgrid(
        np.arange(len(departure_coordinates)), np.arange(len(arrival_coordinates))
    )

    # Filter to ensure departure time is before arrival time
    # and create a mask for valid time combinations
    valid_indices = y > x  # arrival index must be greater than departure index

    # Apply the mask to flatten only valid combinations
    x = x[valid_indices].flatten()
    y = y[valid_indices].flatten()

    stacked_departure_coordinates = departure_coordinates.take(x)
    stacked_arrival_coordinates = arrival_coordinates.take(y)

    if max_processes is None:
        max_processes = mp.cpu_count()

    use_ray = initialize_use_ray(max_processes)

    lambert_results = LambertOutput.empty()
    if use_ray:
        futures = []
        for start, end in _iterate_chunk_indices(
            stacked_departure_coordinates, chunk_size=100
        ):
            futures.append(
                lambert_worker_remote.remote(
                    stacked_departure_coordinates[start:end],
                    stacked_arrival_coordinates[start:end],
                    propagation_origin,
                    prograde,
                    max_iter,
                    tol,
                )
            )

            if len(futures) > 1.5 * max_processes:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                lambert_results = qv.concatenate([lambert_results, result])

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            result = ray.get(finished[0])
            lambert_results = qv.concatenate([lambert_results, result])

    else:
        lambert_results = lambert_worker(
            stacked_departure_coordinates,
            stacked_arrival_coordinates,
            propagation_origin,
            prograde,
            max_iter,
            tol,
        )

    return lambert_results

def plot_porkchop_plotly(
    porkchop_data: LambertOutput,
    width: int = 900,
    height: int = 700,
    c3_min: Optional[float] = None,
    c3_max: Optional[float] = None,
    c3_step: Optional[float] = None,
    vinf_min: Optional[float] = None,
    vinf_max: Optional[float] = None,
    vinf_step: Optional[float] = None,
    tof_min: Optional[float] = None,
    tof_max: Optional[float] = None,
    tof_step: Optional[float] = None,
    metric_colorscale: str = "Viridis",
    tof_line_color: str = "red",
    xlim_mjd: Optional[Tuple[float, float]] = None,
    ylim_mjd: Optional[Tuple[float, float]] = None,
    title: str = "Porkchop Plot",
    show_optimal: bool = True,
    optimal_hover: bool = True,
    trim_to_valid: bool = True,
    date_buffer_days: float = 3.0,
    bundle_raw_data: bool = True,
):
    """
    Plot the porkchop plot from Lambert trajectory data using Plotly.

    Parameters
    ----------
    porkchop_data : LambertOutput
        The output from generate_porkchop_data.
    width : int, optional
        Figure width in pixels.
    height : int, optional
        Figure height in pixels.
    c3_min : float, optional
        Minimum C3 value (km^2/s^2) for contour levels.
    c3_max : float, optional
        Maximum C3 value (km^2/s^2) for contour levels.
    c3_step : float, optional
        Step size for C3 contour levels.
    vinf_min : float, optional
        Minimum Vinf value (km/s) for hover display.
    vinf_max : float, optional
        Maximum Vinf value (km/s) for hover display.
    vinf_step : float, optional
        Step size for Vinf hover display.
    tof_min : float, optional
        Minimum ToF value (days) for contour levels.
    tof_max : float, optional
        Maximum ToF value (days) for contour levels.
    tof_step : float, optional
        Step size for ToF contour levels.
    metric_colorscale : str, optional
        Plotly colorscale name for the C3 filled contours.
    tof_line_color : str, optional
        Color for the ToF contour lines.
    xlim_mjd : Tuple[float, float], optional
        x-axis limits (min_mjd, max_mjd).
    ylim_mjd : Tuple[float, float], optional
        y-axis limits (min_mjd, max_mjd).
    title : str, optional
        Plot title.
    show_optimal : bool, optional
        If True, marks the optimal point on the plot.
    optimal_hover : bool, optional
        If True, enables hover information for the optimal point.
    trim_to_valid : bool, optional
        If True, trims the plot to only include valid data.
    date_buffer_days : float, optional
        Number of days to add as buffer around the min and max dates (default: 3).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object.
    """
    # --- Extract basic raw data ---
    c3_values_au_d2 = porkchop_data.c3()  # C3 in (AU/day)^2
    vinf_values_au_day = np.sqrt(
        np.maximum(0, c3_values_au_d2)
    )  # Departure Vinf in AU/day

    time_of_flight_days = porkchop_data.time_of_flight()
    departure_times = porkchop_data.departure_state.time
    arrival_times = porkchop_data.arrival_state.time

    # Convert to metric units
    km_s_per_au_day = KM_P_AU / S_P_DAY
    c3_values_km2_s2 = c3_values_au_d2 * (km_s_per_au_day**2)
    vinf_values_km_s = vinf_values_au_day * km_s_per_au_day
    # Define default C3 range if not provided
    if c3_min is None:
        c3_min = np.nanpercentile(c3_values_km2_s2, 5)
    if c3_max is None:
        c3_max = np.nanpercentile(c3_values_km2_s2, 95)

    assert c3_max > c3_min, "C3 max must be greater than C3 min"

    if c3_step is None:
        c3_step = (c3_max - c3_min) / 10  # 10 levels by default
    assert c3_step < (c3_max - c3_min), "C3 step must be less than the C3 range"

    # Extract raw MJD values for all points
    departure_times_mjd = departure_times.mjd().to_numpy(zero_copy_only=False)
    arrival_times_mjd = arrival_times.mjd().to_numpy(zero_copy_only=False)

    # --- Identify valid data and filter time ranges ---
    # Create a mask for valid C3 values (not NaN and not over max)
    valid_c3_mask = ~np.isnan(c3_values_km2_s2) & (c3_values_km2_s2 <= c3_max)

    # Extract all unique times from the raw data
    all_departure_mjd = np.sort(np.unique(departure_times_mjd))
    all_arrival_mjd = np.sort(np.unique(arrival_times_mjd))

    if trim_to_valid and np.any(valid_c3_mask):

        # Extract departure and arrival times only for valid data points
        valid_departure_times_mjd = departure_times_mjd[valid_c3_mask]
        valid_arrival_times_mjd = arrival_times_mjd[valid_c3_mask]

        # Get unique departure and arrival times directly from valid data
        unique_departure_mjd = np.sort(np.unique(valid_departure_times_mjd))
        unique_arrival_mjd = np.sort(np.unique(valid_arrival_times_mjd))

        # Add buffer around min/max dates if requested
        if date_buffer_days is not None and date_buffer_days > 0:
            # Get min/max of valid times
            min_dep_mjd, max_dep_mjd = np.min(unique_departure_mjd), np.max(
                unique_departure_mjd
            )
            min_arr_mjd, max_arr_mjd = np.min(unique_arrival_mjd), np.max(
                unique_arrival_mjd
            )

            # Apply buffer, but don't go beyond bounds of all available dates
            min_dep_with_buffer = max(
                min_dep_mjd - date_buffer_days, np.min(all_departure_mjd)
            )
            max_dep_with_buffer = min(
                max_dep_mjd + date_buffer_days, np.max(all_departure_mjd)
            )
            min_arr_with_buffer = max(
                min_arr_mjd - date_buffer_days, np.min(all_arrival_mjd)
            )
            max_arr_with_buffer = min(
                max_arr_mjd + date_buffer_days, np.max(all_arrival_mjd)
            )

            # Include additional dates within buffer
            unique_departure_mjd = all_departure_mjd[
                (all_departure_mjd >= min_dep_with_buffer)
                & (all_departure_mjd <= max_dep_with_buffer)
            ]
            unique_arrival_mjd = all_arrival_mjd[
                (all_arrival_mjd >= min_arr_with_buffer)
                & (all_arrival_mjd <= max_arr_with_buffer)
            ]
    else:
        # If not trimming or no valid data, use all unique times
        unique_departure_mjd = all_departure_mjd
        unique_arrival_mjd = all_arrival_mjd
    # Check if we have enough unique times to create a grid
    if len(unique_departure_mjd) < 2 or len(unique_arrival_mjd) < 2:
        warnings.warn(
            "Porkchop plotting: Not enough unique times for grid. Returning empty figure."
        )
        fig_empty = go.Figure()
        fig_empty.update_layout(
            title=title + " (Insufficient data for grid)",
            xaxis_title="Departure Date",
            yaxis_title="Arrival Date",
            width=width,
            height=height,
            autosize=False,
            xaxis=dict(type="date"),
            yaxis=dict(type="date"),
        )
        return fig_empty

    # Convert to datetime objects for plotting axes
    unique_departure_dates_dt = [
        Time(mjd, format="mjd").datetime for mjd in unique_departure_mjd
    ]
    unique_arrival_dates_dt = [
        Time(mjd, format="mjd").datetime for mjd in unique_arrival_mjd
    ]

    # --- Unit Conversions and Grid Setup ---
    # Create the grid for interpolation with unique times derived from valid data
    grid_departure_mjd, grid_arrival_mjd = np.meshgrid(
        unique_departure_mjd, unique_arrival_mjd
    )
    points = np.vstack((departure_times_mjd, arrival_times_mjd)).T

    grid_c3_km2_s2 = griddata(
        points,
        c3_values_km2_s2,
        (grid_departure_mjd, grid_arrival_mjd),
        method="cubic",
        fill_value=np.nan,
    )
    grid_vinf_km_s = griddata(
        points,
        vinf_values_km_s,
        (grid_departure_mjd, grid_arrival_mjd),
        method="cubic",
        fill_value=np.nan,
    )

    original_tof_grid_days = grid_arrival_mjd - grid_departure_mjd

    # For Vinf (derive from C3 if not specified)
    if vinf_min is None:
        vinf_min = np.sqrt(c3_min)
    if vinf_max is None:
        vinf_max = np.sqrt(c3_max)
    if vinf_step is None:
        vinf_step = (vinf_max - vinf_min) / 10  # 10 levels by default
    # For ToF
    if tof_min is None:
        tof_min = np.nanmin(original_tof_grid_days[original_tof_grid_days > 0])
    if tof_max is None:
        tof_max = np.nanmax(original_tof_grid_days)
    if tof_step is None:
        tof_step = max(5, (tof_max - tof_min) / 10)  # 10 levels, minimum step of 5 days
        tof_step = round(tof_step / 5) * 5  # Round to multiple of 5

    # Validate ToF range - handle cases where ToF values are invalid
    if not (np.isfinite(tof_min) and np.isfinite(tof_max)) or tof_max <= tof_min:
        logger.warning(
            f"Invalid ToF range: tof_min={tof_min}, tof_max={tof_max}. "
            "Using default range."
        )
        tof_min = 30.0  # Default minimum ToF
        tof_max = 365.0  # Default maximum ToF

    # Validate all step sizes are positive
    assert c3_step > 0, f"c3_step must be positive, got {c3_step}"
    assert vinf_step > 0, f"vinf_step must be positive, got {vinf_step}"
    assert tof_step > 0, f"tof_step must be positive, got {tof_step}"

    # --- Replace NaN values and over-max values with sentinel ---
    # Use a sentinel value that's 2x the maximum
    sentinel_value = c3_max * 1.01

    # Replace NaN and over-max values with the sentinel value
    grid_c3_for_plot = np.copy(grid_c3_km2_s2)
    mask_nan = np.isnan(grid_c3_for_plot)

    # Apply sentinel value to all invalid areas
    grid_c3_for_plot[mask_nan] = sentinel_value

    # --- Trim plot to valid data region if requested ---
    # Note: We've already trimmed before creating the grid,
    # so this section is no longer needed

    # Set up the date limits for the plot
    # Convert the min/max MJD values to datetime objects for Plotly
    xlim_dt = [
        Time(np.min(grid_departure_mjd), format="mjd").datetime,
        Time(np.max(grid_departure_mjd), format="mjd").datetime,
    ]
    ylim_dt = [
        Time(np.min(grid_arrival_mjd), format="mjd").datetime,
        Time(np.max(grid_arrival_mjd), format="mjd").datetime,
    ]

    # If explicit limits were provided, use those instead
    if xlim_mjd:
        xlim_dt = [
            Time(xlim_mjd[0], format="mjd").datetime,
            Time(xlim_mjd[1], format="mjd").datetime,
        ]
    if ylim_mjd:
        ylim_dt = [
            Time(ylim_mjd[0], format="mjd").datetime,
            Time(ylim_mjd[1], format="mjd").datetime,
        ]

    # --- Create custom colorscale ---
    # Define function to create custom colorscale with white for sentinel
    def create_colorscale_with_sentinel(base_colorscale, vmin, vmax, sentinel):
        # Get standard colorscale
        standard_colors = pcolors.sample_colorscale(
            base_colorscale, np.linspace(0, 1, 11)
        )

        # Reverse the colors
        standard_colors = standard_colors[::-1]

        # Convert to normalized range (0-1)
        norm_max = 0.9  # Reserve top 10% for sentinel

        # Create standard part of colorscale
        colorscale = [
            (i * norm_max / 10, color) for i, color in enumerate(standard_colors)
        ]

        # Add sentinel value as white
        colorscale.append((1.0, "rgba(255,255,255,1)"))  # White with transparency

        return colorscale

    # Create custom colorscale for C3
    c3_colorscale = create_colorscale_with_sentinel(
        metric_colorscale, c3_min, c3_max, sentinel_value
    )

    # --- Prepare customdata for hover template ---
    # For date strings, we need to convert the grid MJD values to timestamps and then to ISO strings
    # First, create Timestamp objects from the grid MJD values (after trimming)
    flat_dep_mjd = grid_departure_mjd.flatten()
    flat_arr_mjd = grid_arrival_mjd.flatten()

    # Create Timestamp objects from the flattened MJD values
    flat_dep_times = Timestamp.from_mjd(flat_dep_mjd, scale="tdb")
    flat_arr_times = Timestamp.from_mjd(flat_arr_mjd, scale="tdb")

    # Get ISO 8601 strings
    flat_dep_iso = flat_dep_times.to_iso8601().to_numpy(zero_copy_only=False)
    flat_arr_iso = flat_arr_times.to_iso8601().to_numpy(zero_copy_only=False)

    # Reshape back to grid shape
    grid_dep_iso = flat_dep_iso.reshape(grid_departure_mjd.shape)
    grid_arr_iso = flat_arr_iso.reshape(grid_arrival_mjd.shape)

    customdata = None

    if bundle_raw_data:
        # Create a 3D array to hold the numeric data for hover:
        # [C3, Vinf, ToF, Departure X, Departure Y, Departure Z, Arrival X, Arrival Y, Arrival Z]
        customdata = np.full(
            (grid_c3_km2_s2.shape[0], grid_c3_km2_s2.shape[1], 9),
            np.nan,
            dtype=np.float64,
        )

        # Populate with gridded C3, Vinf, ToF
        customdata[:, :, 0] = grid_c3_km2_s2
        customdata[:, :, 1] = grid_vinf_km_s
        customdata[:, :, 2] = original_tof_grid_days

        # Get raw departure Cartesian coordinates converted to KM
        raw_departure_x = (
            porkchop_data.departure_state.x.to_numpy(zero_copy_only=False) * KM_P_AU
        )
        raw_departure_y = (
            porkchop_data.departure_state.y.to_numpy(zero_copy_only=False) * KM_P_AU
        )
        raw_departure_z = (
            porkchop_data.departure_state.z.to_numpy(zero_copy_only=False) * KM_P_AU
        )

        # Get raw arrival Cartesian coordinates converted to KM
        raw_arrival_x = (
            porkchop_data.arrival_state.x.to_numpy(zero_copy_only=False) * KM_P_AU
        )
        raw_arrival_y = (
            porkchop_data.arrival_state.y.to_numpy(zero_copy_only=False) * KM_P_AU
        )
        raw_arrival_z = (
            porkchop_data.arrival_state.z.to_numpy(zero_copy_only=False) * KM_P_AU
        )

        # Interpolate Cartesian coordinates onto the grid
        grid_departure_x = griddata(
            points,
            raw_departure_x,
            (grid_departure_mjd, grid_arrival_mjd),
            method="cubic",
            fill_value=np.nan,
        )
        grid_departure_y = griddata(
            points,
            raw_departure_y,
            (grid_departure_mjd, grid_arrival_mjd),
            method="cubic",
            fill_value=np.nan,
        )

        grid_departure_z = griddata(
            points,
            raw_departure_z,
            (grid_departure_mjd, grid_arrival_mjd),
            method="cubic",
            fill_value=np.nan,
        )

        grid_arrival_x = griddata(
            points,
            raw_arrival_x,
            (grid_departure_mjd, grid_arrival_mjd),
            method="cubic",
            fill_value=np.nan,
        )
        grid_arrival_y = griddata(
            points,
            raw_arrival_y,
            (grid_departure_mjd, grid_arrival_mjd),
            method="cubic",
            fill_value=np.nan,
        )
        grid_arrival_z = griddata(
            points,
            raw_arrival_z,
            (grid_departure_mjd, grid_arrival_mjd),
            method="cubic",
            fill_value=np.nan,
        )

        # Populate customdata with gridded arrival Cartesian coordinates
        customdata[:, :, 3] = grid_departure_x
        customdata[:, :, 4] = grid_departure_y
        customdata[:, :, 5] = grid_departure_z
        customdata[:, :, 6] = grid_arrival_x
        customdata[:, :, 7] = grid_arrival_y
        customdata[:, :, 8] = grid_arrival_z

    # --- Create C3 Contour Trace with hover template ---
    plotly_traces = []
    plotly_traces.append(
        go.Contour(
            x=unique_departure_dates_dt,
            y=unique_arrival_dates_dt,
            z=grid_c3_for_plot,
            zauto=False,  # Don't auto-scale z values
            zmin=c3_min,  # Min value for colorscale
            zmax=c3_max * 1.1,  # Max value for data display (not including sentinel)
            colorscale=c3_colorscale,
            customdata=customdata,
            text=np.stack(
                [grid_dep_iso, grid_arr_iso], axis=-1
            ),  # Store date strings as text
            hovertemplate=(
                "<b>Departure:</b> %{text[0]}<br>"  # Use text array for dates
                + "<b>Arrival:</b> %{text[1]}<br>"  # Use text array for dates
                + "<b>C3:</b> %{customdata[0]:.1f} km²/s²<br>"
                + "<b>Vinf:</b> %{customdata[1]:.1f} km/s<br>"
                + "<b>ToF:</b> %{customdata[2]:.1f} days<br>"
            ),
            contours=dict(
                coloring="fill",
                showlabels=True,
                labelfont=dict(size=10, color="black"),
                start=c3_min,
                end=c3_max,
                size=c3_step,
                # Format C3 values to 1 decimal place
                labelformat=".1f",
            ),
            line=dict(width=0.5, smoothing=1.3),
            name="C3 (km²/s²) / Vinf (km/s)",
            colorbar=dict(
                title="<b>C3</b> (km²/s²) / <b>Vinf</b> (km/s)",
                # Generate ticks based on the actual step size
                tickvals=[
                    level
                    for level in np.arange(c3_min, c3_max + 0.5 * c3_step, c3_step)
                ],
                ticktext=[
                    f"{c3:.1f} / {np.sqrt(c3):.1f}"
                    for c3 in np.arange(c3_min, c3_max + 0.5 * c3_step, c3_step)
                ],
            ),
            connectgaps=True,
            visible=True,
        )
    )

    # --- ToF Contours ---
    plotly_traces.append(
        go.Contour(
            x=unique_departure_dates_dt,
            y=unique_arrival_dates_dt,
            z=original_tof_grid_days,  # Original ToF grid with NaNs
            colorscale=[[0, tof_line_color], [1, tof_line_color]],
            contours=dict(
                coloring="lines",
                showlabels=True,
                labelfont=dict(size=10, color=tof_line_color),
                start=tof_min,
                end=tof_max,
                size=tof_step,
            ),
            line=dict(color=tof_line_color, width=1, dash="dash"),
            name="ToF (days)",
            showscale=False,
            hoverinfo="skip",
            connectgaps=False,  # Don't connect across NaN gaps
            visible=True,
        )
    )

    # --- Optimal point (showing both C3 and Vinf info) ---
    if show_optimal and np.any(~np.isnan(c3_values_km2_s2)):
        min_c3_idx = np.nanargmin(c3_values_km2_s2)
        best_c3_val = c3_values_km2_s2[min_c3_idx]
        best_vinf_val = vinf_values_km_s[min_c3_idx]
        best_tof = time_of_flight_days[min_c3_idx]
        # Get the timestamp objects directly from original data
        # Convert numpy index to Python integer for proper indexing into quivr/pyarrow tables
        best_dep_time = departure_times[int(min_c3_idx)]
        best_arr_time = arrival_times[int(min_c3_idx)]

        # Get ISO strings for hover text
        best_dep_iso = best_dep_time.to_iso8601().to_numpy(zero_copy_only=False)[0]
        best_arr_iso = best_arr_time.to_iso8601().to_numpy(zero_copy_only=False)[0]

        # For scatter point positioning, get datetime objects
        best_dep_dt = best_dep_time.to_astropy()[0].datetime
        best_arr_dt = best_arr_time.to_astropy()[0].datetime

        # Check if the optimal point falls within our current plot range
        optimal_in_range = (
            xlim_dt[0] <= best_dep_dt <= xlim_dt[1]
            and ylim_dt[0] <= best_arr_dt <= ylim_dt[1]
        )

        if optimal_in_range:
            # Create customdata for optimal point with numeric values
            optimal_customdata = np.array([[best_c3_val, best_vinf_val, best_tof]])

            # Create text array for date strings
            optimal_text = np.array([[best_dep_iso, best_arr_iso]])

            # Configure hover behavior
            if optimal_hover:
                hover_info = dict(
                    customdata=optimal_customdata,
                    text=optimal_text,
                    hovertemplate=(
                        "<b>Optimal C3</b><br>"
                        + "<b>Departure:</b> %{text[0]}<br>"
                        + "<b>Arrival:</b> %{text[1]}<br>"
                        + "<b>C3:</b> %{customdata[0]:.1f} km²/s²<br>"
                        + "<b>Vinf:</b> %{customdata[1]:.1f} km/s<br>"
                        + "<b>ToF:</b> %{customdata[2]:.1f} days<extra></extra>"
                    ),
                    hoverlabel=dict(bgcolor="red"),
                )
            else:
                hover_info = dict(
                    hoverinfo="skip",  # Completely disable hover
                )
            plotly_traces.append(
                go.Scatter(
                    x=[best_dep_dt],
                    y=[best_arr_dt],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        color="red",
                        size=12,
                        line=dict(color="black", width=1),
                    ),
                    showlegend=False,
                    name="Optimal C3",
                    visible=True,
                    **hover_info,
                )
            )
        else:
            logger.warning(
                f"Optimal point ({best_dep_iso}, {best_arr_iso}) falls outside the current plot range and will not be displayed"
            )

    # --- Figure Creation and Layout Update ---
    fig = go.Figure(data=plotly_traces)

    fig.update_layout(
        title_text=title,
        xaxis_title="Departure Date",
        yaxis_title="Arrival Date",
        xaxis=dict(tickformat="%Y-%m-%d", tickangle=-45, range=xlim_dt),
        yaxis=dict(tickformat="%Y-%m-%d", range=ylim_dt),
        width=width,
        height=height,
        autosize=False,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
