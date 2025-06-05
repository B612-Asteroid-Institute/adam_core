import logging
import multiprocessing as mp
import warnings
from typing import Any, List, Optional, Tuple, Union

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
from adam_core.coordinates.spherical import SphericalCoordinates
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
    
    def c3_departure(self) -> npt.NDArray[np.float64]:
        """
        Return the C3 in au^2/d^2.
        """
        return calculate_c3(
            np.array(self.table.select(["vx_1", "vy_1", "vz_1"])),
            self.departure_state.v,
        )

    def c3_arrival(self) -> npt.NDArray[np.float64]:
        """
        Return the C3 in au^2/d^2.
        """
        return calculate_c3(
            np.array(self.table.select(["vx_2", "vy_2", "vz_2"])),
            self.arrival_state.v,
        )
    
    def vinf_departure(self) -> npt.NDArray[np.float64]:
        """
        Return the v infinity in au/d.
        """
        return np.linalg.norm(
            np.array(self.table.select(["vx_1", "vy_1", "vz_1"]))
            - self.departure_state.v,
            axis=1,
        )


    def vinf_arrival(self) -> npt.NDArray[np.float64]:
        """
        Return the v infinity in au/d.
        """
        return np.linalg.norm(
            np.array(self.table.select(["vx_2", "vy_2", "vz_2"]))
            - self.arrival_state.v,
            axis=1,
        )

    def time_of_flight(self) -> npt.NDArray[np.float64]:
        """
        Return the time of flight in days.
        """
        return self.arrival_state.time.mjd().to_numpy(
            zero_copy_only=False
        ) - self.departure_state.time.mjd().to_numpy(zero_copy_only=False)


def departure_spherical_coordinates(
    departure_origin: OriginCodes,
    times: Timestamp,
    frame: str,
    vx: npt.NDArray[np.float64],
    vy: npt.NDArray[np.float64],
    vz: npt.NDArray[np.float64],
) -> SphericalCoordinates:
    """
    Return the spherical coordinates of the departure vector.

    Parameters
    ----------
    departure_origin : OriginCodes
        The origin of the departure and also the frame of the departure vectors.
    times : Timestamp
        The times of the departure vectors.
    frame : str
        The frame of the departure vectors.
    vx : npt.NDArray[np.float64]
        The x-component of the departure vectors.
    vy : npt.NDArray[np.float64]
        The y-component of the departure vectors.
    vz : npt.NDArray[np.float64]
        The z-component of the departure vectors.

    Returns
    -------
    SphericalCoordinates
        The spherical coordinates of the departure unit vectors.
        Can be used to express ra / dec of the departure direction.
    """
    assert len(vx) == len(vy) == len(vz) == len(times), "All arrays must have the same length"
    assert len(vx) > 0, "At least one departure vector is required"
    
    # Create unit direction vectors from the velocity vectors
    # Normalize the velocity vectors to get direction only
    velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
    direction_x = vx / velocity_magnitude
    direction_y = vy / velocity_magnitude  
    direction_z = vz / velocity_magnitude
    
    # Create CartesianCoordinates with the direction as position (on unit sphere)
    # and zero velocity since we only care about the direction
    direction_coords = CartesianCoordinates.from_kwargs(
        time=times,
        x=direction_x,  # Unit vector pointing in velocity direction
        y=direction_y,  
        z=direction_z,
        vx=np.zeros_like(vx),  # No velocity needed for direction
        vy=np.zeros_like(vy), 
        vz=np.zeros_like(vz),
        # From our departing origin.
        origin=Origin.from_OriginCodes(departure_origin, size=len(vx)),
        frame=frame,
    )
    
    # Transform direction to equatorial frame for proper RA/Dec coordinates
    # These are inertial celestial coordinates, suitable for any departure origin
    spherical = transform_coordinates(
        direction_coords,
        SphericalCoordinates,
        frame_out="equatorial",
        origin_out=departure_origin
    )
    return spherical



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
    start_time: Timestamp,
    end_time: Timestamp,
    propagation_origin: OriginCodes = OriginCodes.SUN,
    step_size: float = 1.0,
    propagator_class: Optional[type[Propagator]] = None,
    max_processes: Optional[int] = 1,
) -> CartesianCoordinates:
    """
    Prepare and propagate orbits for a single body over a specified time range.
    
    Parameters
    ----------
    body : Union[Orbits, OriginCodes]
        The body to propagate (either an Orbits object or an OriginCode for a major body).
    start_time : Timestamp
        The start time for propagation.
    end_time : Timestamp
        The end time for propagation.
    propagation_origin : OriginCodes, optional
        The origin of the propagation (default: SUN).
    step_size : float, optional
        The step size in days (default: 1.0).
    propagator_class : Optional[type[Propagator]], optional
        The propagator class to use for orbit propagation.
    max_processes : Optional[int], optional
        The maximum number of processes to use.
        
    Returns
    -------
    CartesianCoordinates
        The propagated coordinates over the specified time range.
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

    times = Timestamp.from_mjd(
        np.arange(
            start_time.rescale("tdb").mjd()[0].as_py(),
            end_time.rescale("tdb").mjd()[0].as_py(),
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
    departure_coordinates : CartesianCoordinates
        The departure coordinates.
    arrival_coordinates : CartesianCoordinates
        The arrival coordinates.
    propagation_origin : OriginCodes
        The origin of the propagation.
    prograde : bool, optional
        If True, assume prograde motion. If False, assume retrograde motion.
    max_iter : int, optional
        The maximum number of iterations for Lambert's solver.
    tol : float, optional
        The numerical tolerance for Lambert's solver.
    max_processes : Optional[int], optional
        The maximum number of processes to use.
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

    # Get the actual times for comparison
    dep_times_mjd = departure_coordinates.time.mjd().to_numpy(zero_copy_only=False)
    arr_times_mjd = arrival_coordinates.time.mjd().to_numpy(zero_copy_only=False)

    # Create meshgrids of indices and times
    dep_indices, arr_indices = np.meshgrid(
        np.arange(len(departure_coordinates)), np.arange(len(arrival_coordinates))
    )
    dep_time_grid, arr_time_grid = np.meshgrid(dep_times_mjd, arr_times_mjd)

    # Filter to ensure departure time is before arrival time
    # Use actual time comparison instead of index comparison
    valid_indices = arr_time_grid > dep_time_grid

    # Apply the mask to flatten only valid combinations
    dep_indices_flat = dep_indices[valid_indices].flatten()
    arr_indices_flat = arr_indices[valid_indices].flatten()

    stacked_departure_coordinates = departure_coordinates.take(dep_indices_flat)
    stacked_arrival_coordinates = arrival_coordinates.take(arr_indices_flat)

    # If no valid combinations exist, return empty results
    if len(stacked_departure_coordinates) == 0:
        return LambertOutput.empty()

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
    c3_base_colorscale: str = "Reds",
    vinf_base_colorscale: str = "Blues",
    tof_line_color: str = "black",
    xlim_mjd: Optional[Tuple[float, float]] = None,
    ylim_mjd: Optional[Tuple[float, float]] = None,
    title: str = "Porkchop Plot",
    show_optimal: bool = True,
    optimal_hover: bool = True,
    trim_to_valid: bool = True,
    date_buffer_days: float = 3.0,
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
    c3_departure_au_d2 = porkchop_data.c3_departure()  # C3 departure in (AU/day)^2
    vinf_arrival_au_day = porkchop_data.vinf_arrival()  # V∞ arrival in AU/day

    time_of_flight_days = porkchop_data.time_of_flight()
    departure_times = porkchop_data.departure_state.time
    arrival_times = porkchop_data.arrival_state.time

    # Convert to metric units
    km_s_per_au_day = KM_P_AU / S_P_DAY
    c3_departure_km2_s2 = c3_departure_au_d2 * (km_s_per_au_day**2)
    vinf_arrival_km_s = vinf_arrival_au_day * km_s_per_au_day
    # Define default C3 range if not provided
    if c3_min is None:
        c3_min = np.nanpercentile(c3_departure_km2_s2, 5)
    if c3_max is None:
        c3_max = np.nanpercentile(c3_departure_km2_s2, 95)

    assert c3_max > c3_min, "C3 max must be greater than C3 min"

    if c3_step is None:
        c3_step = (c3_max - c3_min) / 10  # 10 levels by default
    assert c3_step < (c3_max - c3_min), "C3 step must be less than the C3 range"

    # Define default V∞ range if not provided
    if vinf_min is None:
        vinf_min = np.nanpercentile(vinf_arrival_km_s, 5)
    if vinf_max is None:
        vinf_max = np.nanpercentile(vinf_arrival_km_s, 95)
    if vinf_step is None:
        vinf_step = (vinf_max - vinf_min) / 10  # 10 levels by default

    # Extract raw MJD values for all points
    departure_times_mjd = departure_times.mjd().to_numpy(zero_copy_only=False)
    arrival_times_mjd = arrival_times.mjd().to_numpy(zero_copy_only=False)

    # --- Identify valid data and filter time ranges ---
    # Create a mask for valid C3 values (not NaN and not over max)
    valid_c3_mask = ~np.isnan(c3_departure_km2_s2) & (c3_departure_km2_s2 <= c3_max)

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

    # Interpolate both C3 departure and V∞ arrival onto the grid
    grid_c3_departure_km2_s2 = griddata(
        points,
        c3_departure_km2_s2,
        (grid_departure_mjd, grid_arrival_mjd),
        method="cubic",
        fill_value=np.nan,
    )
    grid_vinf_arrival_km_s = griddata(
        points,
        vinf_arrival_km_s,
        (grid_departure_mjd, grid_arrival_mjd),
        method="cubic",
        fill_value=np.nan,
    )

    original_tof_grid_days = grid_arrival_mjd - grid_departure_mjd
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
    # Use a sentinel value that's 2x the maximum for C3
    c3_sentinel_value = c3_max * 1.01
    vinf_sentinel_value = vinf_max * 1.01

    # Replace NaN and over-max values with the sentinel value for C3
    grid_c3_for_plot = np.copy(grid_c3_departure_km2_s2)
    mask_c3_nan = np.isnan(grid_c3_for_plot)
    grid_c3_for_plot[mask_c3_nan] = c3_sentinel_value

    # Replace NaN and over-max values with the sentinel value for V∞
    grid_vinf_for_plot = np.copy(grid_vinf_arrival_km_s)
    mask_vinf_nan = np.isnan(grid_vinf_for_plot)
    grid_vinf_for_plot[mask_vinf_nan] = vinf_sentinel_value

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
        # Get standard colorscale using brighter range to avoid muddy overlaps
        # Use range 0.2 to 0.8 instead of 0 to 1 to skip very dark and very light colors
        standard_colors = pcolors.sample_colorscale(
            base_colorscale, np.linspace(0.2, 0.8, 11)
        )

        # Reverse the colors (brightest for lowest values)
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

    # Create custom colorscales for both C3 and V∞
    c3_colorscale = create_colorscale_with_sentinel(
        c3_base_colorscale, c3_min, c3_max, c3_sentinel_value
    )
    vinf_colorscale = create_colorscale_with_sentinel(
        vinf_base_colorscale, vinf_min, vinf_max, vinf_sentinel_value
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

    # Create a 3D array to hold the numeric data for hover:
    # [C3_departure, Vinf_arrival, ToF]
    customdata = np.full(
        (grid_c3_departure_km2_s2.shape[0], grid_c3_departure_km2_s2.shape[1], 3),
        np.nan,
        dtype=np.float64,
    )

    # Populate with gridded C3 departure, V∞ arrival, ToF
    customdata[:, :, 0] = grid_c3_departure_km2_s2
    customdata[:, :, 1] = grid_vinf_arrival_km_s
    customdata[:, :, 2] = original_tof_grid_days

    # --- Create Dual Contour Traces ---
    plotly_traces = []
    
    # C3 Departure Contour Trace (warm colorscale)
    plotly_traces.append(
        go.Contour(
            x=unique_departure_dates_dt,
            y=unique_arrival_dates_dt,
            z=grid_c3_for_plot,
            zauto=False,
            zmin=c3_min,
            zmax=c3_max * 1.1,
            colorscale=c3_colorscale,
            opacity=0.3,  # More transparency for better layering
            customdata=customdata,
            text=np.stack([grid_dep_iso, grid_arr_iso], axis=-1),
            hovertemplate=(
                "<b>C3 Departure</b><br>"
                + "<b>Departure:</b> %{text[0]}<br>"
                + "<b>Arrival:</b> %{text[1]}<br>"
                + "<b>C3:</b> %{customdata[0]:.1f} km²/s²<br>"
                + "<b>V∞ Arrival:</b> %{customdata[1]:.1f} km/s<br>"
                + "<b>ToF:</b> %{customdata[2]:.1f} days<extra></extra>"
            ),
            contours=dict(
                coloring="fill",
                showlabels=True,
                labelfont=dict(size=10, color="darkred"),
                start=c3_min,
                end=c3_max,
                size=c3_step,
                labelformat=".1f",
            ),
            line=dict(width=0.5, smoothing=1.3),
            name="C3 Departure",
            showscale=False,  # Remove colorbar from main trace
            connectgaps=True,
            visible=True,
            showlegend=True,
        )
    )
    
    # V∞ Arrival Contour Trace (cool colorscale)
    plotly_traces.append(
        go.Contour(
            x=unique_departure_dates_dt,
            y=unique_arrival_dates_dt,
            z=grid_vinf_for_plot,
            zauto=False,
            zmin=vinf_min,
            zmax=vinf_max * 1.1,
            colorscale=vinf_colorscale,
            opacity=0.3,  # More transparency for better layering
            customdata=customdata,
            text=np.stack([grid_dep_iso, grid_arr_iso], axis=-1),
            hovertemplate=(
                "<b>V∞ Arrival</b><br>"
                + "<b>Departure:</b> %{text[0]}<br>"
                + "<b>Arrival:</b> %{text[1]}<br>"
                + "<b>C3:</b> %{customdata[0]:.1f} km²/s²<br>"
                + "<b>V∞ Arrival:</b> %{customdata[1]:.1f} km/s<br>"
                + "<b>ToF:</b> %{customdata[2]:.1f} days<extra></extra>"
            ),
            contours=dict(
                coloring="fill",
                showlabels=True,
                labelfont=dict(size=10, color="darkblue"),
                start=vinf_min,
                end=vinf_max,
                size=vinf_step,
                labelformat=".1f",
            ),
            line=dict(width=0.5, smoothing=1.3),
            name="V∞ Arrival",
            showscale=False,  # Remove colorbar from main trace
            connectgaps=False,  # Faster rendering by not connecting across gaps
            visible=True,
            showlegend=True,
        )
    )
    
    # --- Persistent Colorbars (invisible traces that hold colorbars) ---
    # C3 Departure Colorbar (always visible)
    plotly_traces.append(
        go.Scatter(
            x=[None],  # No actual data points
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=c3_colorscale,
                cmin=c3_min,
                cmax=c3_max,
                color=[c3_min],  # Dummy value for colorscale
                colorbar=dict(
                    title="<b>C3 Departure</b><br>(km²/s²)",
                    x=1.05,  # Position to the right with more space
                    len=0.75,  # Shorter to fit both colorbars
                    tickvals=[
                        level
                        for level in np.arange(c3_min, c3_max + 0.5 * c3_step, c3_step)
                    ],
                    ticktext=[
                        f"{c3:.1f}"
                        for c3 in np.arange(c3_min, c3_max + 0.5 * c3_step, c3_step)
                    ],
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
            visible=True,  # Always visible
        )
    )
    
    # V∞ Arrival Colorbar (always visible)
    plotly_traces.append(
        go.Scatter(
            x=[None],  # No actual data points
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=vinf_colorscale,
                cmin=vinf_min,
                cmax=vinf_max,
                color=[vinf_min],  # Dummy value for colorscale
                colorbar=dict(
                    title="<b>V∞ Arrival</b><br>(km/s)",
                    x=1.15,  # Position further to the right with more space
                    len=0.75,  # Shorter to fit both colorbars
                    tickvals=[
                        level
                        for level in np.arange(vinf_min, vinf_max + 0.5 * vinf_step, vinf_step)
                    ],
                    ticktext=[
                        f"{vinf:.1f}"
                        for vinf in np.arange(vinf_min, vinf_max + 0.5 * vinf_step, vinf_step)
                    ],
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
            visible=True,  # Always visible
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
            line=dict(color=tof_line_color, width=1, dash="longdash"),
            name="ToF (days)",
            showscale=False,
            hoverinfo="skip",
            connectgaps=False,  # Don't connect across NaN gaps
            visible=True,
        )
    )

    # --- Optimal Points (separate for C3 and V∞) ---
    if show_optimal:
        # Optimal C3 Departure Point
        if np.any(~np.isnan(c3_departure_km2_s2)):
            min_c3_idx = np.nanargmin(c3_departure_km2_s2)
            best_c3_val = c3_departure_km2_s2[min_c3_idx]
            best_c3_vinf_val = vinf_arrival_km_s[min_c3_idx]
            best_c3_tof = time_of_flight_days[min_c3_idx]
            
            # Get the timestamp objects directly from original data
            best_c3_dep_time = departure_times[int(min_c3_idx)]
            best_c3_arr_time = arrival_times[int(min_c3_idx)]

            # Get ISO strings for hover text
            best_c3_dep_iso = best_c3_dep_time.to_iso8601().to_numpy(zero_copy_only=False)[0]
            best_c3_arr_iso = best_c3_arr_time.to_iso8601().to_numpy(zero_copy_only=False)[0]

            # For scatter point positioning, get datetime objects
            best_c3_dep_dt = best_c3_dep_time.to_astropy()[0].datetime
            best_c3_arr_dt = best_c3_arr_time.to_astropy()[0].datetime

            # Check if the optimal C3 point falls within our current plot range
            c3_optimal_in_range = (
                xlim_dt[0] <= best_c3_dep_dt <= xlim_dt[1]
                and ylim_dt[0] <= best_c3_arr_dt <= ylim_dt[1]
            )

            if c3_optimal_in_range:
                # Create customdata for optimal C3 point
                c3_optimal_customdata = np.array([[best_c3_val, best_c3_vinf_val, best_c3_tof]])
                c3_optimal_text = np.array([[best_c3_dep_iso, best_c3_arr_iso]])

                # Configure hover behavior for C3 optimal
                if optimal_hover:
                    c3_hover_info = dict(
                        customdata=c3_optimal_customdata,
                        text=c3_optimal_text,
                        hovertemplate=(
                            "<b>Optimal C3 Departure</b><br>"
                            + "<b>Departure:</b> %{text[0]}<br>"
                            + "<b>Arrival:</b> %{text[1]}<br>"
                            + "<b>C3:</b> %{customdata[0]:.1f} km²/s²<br>"
                            + "<b>V∞ Arrival:</b> %{customdata[1]:.1f} km/s<br>"
                            + "<b>ToF:</b> %{customdata[2]:.1f} days<extra></extra>"
                        ),
                        hoverlabel=dict(bgcolor="darkred"),
                    )
                else:
                    c3_hover_info = dict(hoverinfo="skip")
                    
                plotly_traces.append(
                    go.Scatter(
                        x=[best_c3_dep_dt],
                        y=[best_c3_arr_dt],
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            color="darkred",
                            size=10,
                            line=dict(color="white", width=2),
                        ),
                        showlegend=True,
                        name="Optimal C3",
                        visible=True,
                        **c3_hover_info,
                    )
                )
            else:
                logger.warning(
                    f"Optimal C3 point ({best_c3_dep_iso}, {best_c3_arr_iso}) falls outside the current plot range"
                )
        
        # Optimal V∞ Arrival Point
        if np.any(~np.isnan(vinf_arrival_km_s)):
            min_vinf_idx = np.nanargmin(vinf_arrival_km_s)
            best_vinf_c3_val = c3_departure_km2_s2[min_vinf_idx]
            best_vinf_val = vinf_arrival_km_s[min_vinf_idx]
            best_vinf_tof = time_of_flight_days[min_vinf_idx]
            
            # Get the timestamp objects directly from original data
            best_vinf_dep_time = departure_times[int(min_vinf_idx)]
            best_vinf_arr_time = arrival_times[int(min_vinf_idx)]

            # Get ISO strings for hover text
            best_vinf_dep_iso = best_vinf_dep_time.to_iso8601().to_numpy(zero_copy_only=False)[0]
            best_vinf_arr_iso = best_vinf_arr_time.to_iso8601().to_numpy(zero_copy_only=False)[0]

            # For scatter point positioning, get datetime objects
            best_vinf_dep_dt = best_vinf_dep_time.to_astropy()[0].datetime
            best_vinf_arr_dt = best_vinf_arr_time.to_astropy()[0].datetime

            # Check if the optimal V∞ point falls within our current plot range
            vinf_optimal_in_range = (
                xlim_dt[0] <= best_vinf_dep_dt <= xlim_dt[1]
                and ylim_dt[0] <= best_vinf_arr_dt <= ylim_dt[1]
            )

            if vinf_optimal_in_range:
                # Create customdata for optimal V∞ point
                vinf_optimal_customdata = np.array([[best_vinf_c3_val, best_vinf_val, best_vinf_tof]])
                vinf_optimal_text = np.array([[best_vinf_dep_iso, best_vinf_arr_iso]])

                # Configure hover behavior for V∞ optimal
                if optimal_hover:
                    vinf_hover_info = dict(
                        customdata=vinf_optimal_customdata,
                        text=vinf_optimal_text,
                        hovertemplate=(
                            "<b>Optimal V∞ Arrival</b><br>"
                            + "<b>Departure:</b> %{text[0]}<br>"
                            + "<b>Arrival:</b> %{text[1]}<br>"
                            + "<b>C3:</b> %{customdata[0]:.1f} km²/s²<br>"
                            + "<b>V∞ Arrival:</b> %{customdata[1]:.1f} km/s<br>"
                            + "<b>ToF:</b> %{customdata[2]:.1f} days<extra></extra>"
                        ),
                        hoverlabel=dict(bgcolor="darkblue"),
                    )
                else:
                    vinf_hover_info = dict(hoverinfo="skip")
                    
                plotly_traces.append(
                    go.Scatter(
                        x=[best_vinf_dep_dt],
                        y=[best_vinf_arr_dt],
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            color="darkblue",
                            size=10,
                            line=dict(color="white", width=2),
                        ),
                        showlegend=True,
                        name="Optimal V∞",
                        visible=True,
                        **vinf_hover_info,
                    )
                )
            else:
                logger.warning(
                    f"Optimal V∞ point ({best_vinf_dep_iso}, {best_vinf_arr_iso}) falls outside the current plot range"
                )

    # --- Figure Creation and Layout Update ---
    fig = go.Figure(data=plotly_traces)

    fig.update_layout(
        title_text=title,
        xaxis_title="Departure Date",
        yaxis_title="Arrival Date",
        xaxis=dict(tickformat="%Y-%m-%d", tickangle=-45, range=xlim_dt),
        yaxis=dict(tickformat="%Y-%m-%d", range=ylim_dt),
        width=width + 200,  # Extra width for colorbars
        height=height,
        autosize=False,
        hovermode="closest",
        margin=dict(r=200),  # Right margin for colorbars
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="left", 
            x=0,
            itemsizing="constant",  # Keep legend items same size when hidden
            font=dict(size=12),  # Larger legend text
            bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
            bordercolor="Black",
            borderwidth=1
        ),
    )

    return fig
