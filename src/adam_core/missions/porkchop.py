import multiprocessing as mp
import time
import warnings
from typing import List, Optional, Tuple, Union

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

from adam_core.coordinates import CartesianCoordinates, transform_coordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.dynamics.lambert import calculate_c3, solve_lambert
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils import get_perturber_state
from adam_core.utils.iter import _iterate_chunk_indices


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


def generate_porkchop_data(
    departure_body: Union[Orbits, OriginCodes],
    arrival_body: Union[Orbits, OriginCodes],
    earliest_launch_time: Timestamp,
    maximum_arrival_time: Timestamp,
    propagation_origin: OriginCodes = OriginCodes.SUN,
    step_size: float = 0.1,
    prograde: bool = True,
    max_iter: int = 35,
    tol: float = 1e-10,
    propagator_class: Optional[type[Propagator]] = None,
    max_processes: Optional[int] = 1,
) -> LambertOutput:
    """
    Generate data for a porkchop plot by solving Lambert's problem for a grid of
    departure and arrival times.

    Parameters
    ----------
    r1_func : callable
        Function that returns the position vector of the departure body at a given time.
    r2_func : callable
        Function that returns the position vector of the arrival body at a given time.
    departure_times : `~numpy.ndarray` (N)
        Array of departure times in days (e.g., MJD).
    arrival_times : `~numpy.ndarray` (M)
        Array of arrival times in days (e.g., MJD).
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    prograde : bool, optional
        If True, assume prograde motion. If False, assume retrograde motion.
    max_iter : int, optional
        Maximum number of iterations for Lambert's solver.
    tol : float, optional
        Numerical tolerance for Lambert's solver.

    Returns
    -------
    delta_v_departure : `~numpy.ndarray` (N, M)
        Delta-v required at departure for each departure-arrival time combination.
    delta_v_arrival : `~numpy.ndarray` (N, M)
        Delta-v required at arrival for each departure-arrival time combination.
    total_delta_v : `~numpy.ndarray` (N, M)
        Total delta-v (departure + arrival) for each departure-arrival time combination.
    """

    start_total = time.time()
    print(f"Starting porkchop data generation...")

    # if departure_body is an Orbit, ensure its origin is the propagation_origin
    start_prep = time.time()
    if isinstance(departure_body, Orbits):
        departure_body = departure_body.set_column(
            "coordinates",
            transform_coordinates(
                departure_body.coordinates,
                representation_out=CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=propagation_origin,
            ),
        )

    # if arrival_body is an Orbit, ensure its origin is the propagation_origin
    if isinstance(arrival_body, Orbits):
        arrival_body = arrival_body.set_column(
            "coordinates",
            transform_coordinates(
                arrival_body.coordinates,
                representation_out=CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=propagation_origin,
            ),
        )
    prep_time = time.time() - start_prep
    print(f"Initial coordinate transformation took {prep_time:.3f} seconds")

    # pre-generate the state vectors for the departure and arrival bodies and the specified time grid
    start_time_gen = time.time()
    # create empty CartesianCoordinates
    departure_coordinates = CartesianCoordinates.empty(
        frame="ecliptic",
    )
    arrival_coordinates = CartesianCoordinates.empty(
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
    time_gen_time = time.time() - start_time_gen
    print(f"Time grid generation took {time_gen_time:.3f} seconds")

    # get r1 (departure body) for times
    start_dep_prop = time.time()
    if isinstance(departure_body, Orbits):
        propagator = propagator_class()
        departure_coordinates = propagator.propagate_orbits(
            departure_body, times, max_processes=max_processes
        ).coordinates
    else:
        departure_coordinates = get_perturber_state(
            departure_body, times, frame="ecliptic", origin=propagation_origin
        )
    dep_prop_time = time.time() - start_dep_prop
    print(f"Departure body propagation took {dep_prop_time:.3f} seconds")

    # get r2 (arrival body) for times
    start_arr_prop = time.time()
    if isinstance(arrival_body, Orbits):
        propagator = propagator_class()
        arrival_coordinates = propagator.propagate_orbits(
            arrival_body, times, max_processes=max_processes
        ).coordinates
    else:
        arrival_coordinates = get_perturber_state(
            arrival_body, times, frame="ecliptic", origin=propagation_origin
        )
    arr_prop_time = time.time() - start_arr_prop
    print(f"Arrival body propagation took {arr_prop_time:.3f} seconds")

    start_mesh = time.time()
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
    mesh_time = time.time() - start_mesh
    print(f"Mesh creation and coordinate stacking took {mesh_time:.3f} seconds")

    if max_processes is None:
        max_processes = mp.cpu_count()

    start_ray = time.time()
    use_ray = initialize_use_ray(max_processes)
    ray_init_time = time.time() - start_ray
    print(f"Ray initialization took {ray_init_time:.3f} seconds")

    start_lambert = time.time()
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
    lambert_time = time.time() - start_lambert
    print(f"Lambert problem solution took {lambert_time:.3f} seconds")

    total_time = time.time() - start_total
    print(f"\nTotal porkchop data generation took {total_time:.3f} seconds")
    print(
        f"  - Coordinate preparation: {prep_time:.3f}s ({prep_time/total_time*100:.1f}%)"
    )
    print(
        f"  - Time grid generation: {time_gen_time:.3f}s ({time_gen_time/total_time*100:.1f}%)"
    )
    print(
        f"  - Departure propagation: {dep_prop_time:.3f}s ({dep_prop_time/total_time*100:.1f}%)"
    )
    print(
        f"  - Arrival propagation: {arr_prop_time:.3f}s ({arr_prop_time/total_time*100:.1f}%)"
    )
    print(f"  - Mesh and stacking: {mesh_time:.3f}s ({mesh_time/total_time*100:.1f}%)")
    print(
        f"  - Ray initialization: {ray_init_time:.3f}s ({ray_init_time/total_time*100:.1f}%)"
    )
    print(
        f"  - Lambert solution: {lambert_time:.3f}s ({lambert_time/total_time*100:.1f}%)"
    )

    return lambert_results


def plot_porkchop(
    porkchop_data: LambertOutput,
    figsize: Tuple[int, int] = (12, 10),
    c3_levels: Optional[list] = None,
    cmap: str = "Wistia",
    tof_levels: Optional[list] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Porkchop Plot",
    show_optimal: bool = True,
    log_scale: bool = True,
):
    """
    Plot the porkchop plot from Lambert trajectory data.

    Parameters
    ----------
    porkchop_data : LambertOutput
        The output from generate_porkchop_data containing Lambert trajectory solutions.
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches. Default is (12, 10).
    c3_levels : list, optional
        List of C3 values for contour levels. If None, default levels will be used.
    cmap : str, optional
        Matplotlib colormap name for the filled contours. Default is 'Wistia'.
    tof_levels : list, optional
        List of time of flight values for contour levels. If None, dynamic levels will be used.
    xlim : Tuple[float, float], optional
        x-axis limits (min, max) in MJD. If None, will use data range.
    ylim : Tuple[float, float], optional
        y-axis limits (min, max) in MJD. If None, will use data range.
    title : str, optional
        Plot title. Default is 'Porkchop Plot'.
    show_optimal : bool, optional
        If True, marks the optimal (minimum C3) point on the plot. Default is True.
    log_scale : bool, optional
        If True, uses logarithmic scale for C3 contours. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    from astropy.time import Time

    # Extract data from LambertOutput
    c3_values = porkchop_data.c3()
    time_of_flight = porkchop_data.time_of_flight()
    departure_times = porkchop_data.departure_state.time.mjd().to_numpy(
        zero_copy_only=False
    )
    arrival_times = porkchop_data.arrival_state.time.mjd().to_numpy(
        zero_copy_only=False
    )

    # Convert C3 from (AU/day)^2 to km^2/s^2
    c3_values_km2_s2 = c3_values * (0.02004**2 * 86400**2)

    # Create figure and prepare grid
    fig, ax = plt.subplots(figsize=figsize)
    unique_departure_times = np.sort(np.unique(departure_times))
    unique_arrival_times = np.sort(np.unique(arrival_times))
    grid_departure, grid_arrival = np.meshgrid(
        unique_departure_times, unique_arrival_times
    )

    # Interpolate C3 values onto the regular grid
    grid_c3 = griddata(
        (departure_times, arrival_times),
        c3_values_km2_s2,
        (grid_departure, grid_arrival),
        method="cubic",
        fill_value=np.nan,
    )

    # Define default contour levels for C3 if not provided
    if c3_levels is None:
        c3_levels = [8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]

    # Create contour plot
    contour = ax.contour(
        grid_departure,
        grid_arrival,
        grid_c3,
        levels=c3_levels,
        colors="black",
        linewidths=1,
    )
    ax.clabel(contour, inline=True, fontsize=10, fmt="%d")

    # Create filled contour plot with appropriate color scale
    norm = (
        LogNorm(vmin=min(c3_levels), vmax=max(c3_levels))
        if log_scale
        else Normalize(vmin=min(c3_levels), vmax=max(c3_levels))
    )
    contourf = ax.contourf(
        grid_departure, grid_arrival, grid_c3, levels=c3_levels, cmap=cmap, norm=norm
    )

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label("C3 (km²/s²)", fontsize=12)

    # Mark the optimal point if requested
    if show_optimal:
        min_idx = np.nanargmin(c3_values_km2_s2)
        best_departure = departure_times[min_idx]
        best_arrival = arrival_times[min_idx]
        best_tof = time_of_flight[min_idx]
        ax.scatter(
            best_departure,
            best_arrival,
            color="red",
            s=100,
            marker="*",
            label=f"Optimal: C3={c3_values_km2_s2[min_idx]:.1f} km²/s², ToF={best_tof:.1f} days",
        )

    # Add time of flight contours
    tof_grid = grid_arrival - grid_departure

    if tof_levels is None:
        # Get optimal TOF and round to a nice number
        min_idx = np.nanargmin(c3_values_km2_s2)
        optimal_tof = time_of_flight[min_idx]
        optimal_tof_rounded = (
            round(optimal_tof / 5) * 5
            if optimal_tof < 100
            else round(optimal_tof / 10) * 10
        )

        # Set TOF range and create levels centered around optimal
        min_tof = max(0, np.nanmin(tof_grid))
        max_tof = np.nanmax(tof_grid)
        num_levels = 9  # Odd number to have optimal in the middle
        step_size = max(5, round((max_tof - min_tof) / num_levels / 5) * 5)

        # Create levels below and above optimal
        half_levels = (num_levels - 1) // 2
        lower_bound = max(min_tof, optimal_tof_rounded - half_levels * step_size)
        upper_bound = min(max_tof, optimal_tof_rounded + (half_levels + 1) * step_size)

        lower_levels = np.arange(lower_bound, optimal_tof_rounded, step_size)
        upper_levels = np.arange(optimal_tof_rounded, upper_bound, step_size)
        tof_levels = np.concatenate([lower_levels, upper_levels])
        tof_levels = tof_levels[tof_levels > 0]

    # Draw TOF contours
    tof_contour = ax.contour(
        grid_departure,
        grid_arrival,
        tof_grid,
        levels=tof_levels,
        colors="red",
        linestyles="dashed",
        linewidths=1,
    )

    # Improve TOF contour labels visibility
    ax.clabel(
        tof_contour,
        inline=True,
        fontsize=10,
        fmt="%d days",
        inline_spacing=10,  # More space around labels
        use_clabeltext=True,  # Enable manual positioning if needed
        colors="red",  # Match contour color
    )

    # Set axis limits based on valid C3 values if not explicitly provided
    if xlim is None or ylim is None:
        max_c3 = max(c3_levels)
        valid_mask = (~np.isnan(grid_c3)) & (grid_c3 <= max_c3)

        if np.any(valid_mask):
            valid_cols_mask = np.any(valid_mask, axis=0)
            valid_rows_mask = np.any(valid_mask, axis=1)

            if np.any(valid_cols_mask) and np.any(valid_rows_mask):
                valid_col_indices = np.where(valid_cols_mask)[0]
                valid_row_indices = np.where(valid_rows_mask)[0]

                col_padding = max(2, int(0.05 * len(valid_col_indices)))
                row_padding = max(2, int(0.05 * len(valid_row_indices)))

                max_col = min(
                    len(unique_departure_times) - 1, valid_col_indices[-1] + col_padding
                )
                min_row = max(0, valid_row_indices[0] - row_padding)

                if xlim is None:
                    xlim = (min(departure_times), unique_departure_times[max_col])
                if ylim is None:
                    ylim = (unique_arrival_times[min_row], max(arrival_times))

    # Apply the limits
    if xlim is None:
        xlim = (min(departure_times), max(departure_times))
    if ylim is None:
        ylim = (min(arrival_times), max(arrival_times))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add labels and title
    ax.set_xlabel("Departure Date", fontsize=12)
    ax.set_ylabel("Arrival Date", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Get current ticks
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Filter out any dates that might cause ERFA warnings (typically far future/past dates)
    # Only keep ticks within our data range plus a small margin
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    margin = 10  # 10 day margin

    xticks = xticks[(xticks >= x_min - margin) & (xticks <= x_max + margin)]
    yticks = yticks[(yticks >= y_min - margin) & (yticks <= y_max + margin)]

    # Explicitly set the ticks before setting labels
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Suppress ERFA warnings when converting dates
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        xtick_labels = Time(xticks, format="mjd").iso
        ytick_labels = Time(yticks, format="mjd").iso

    # Simplify date format to just show YYYY-MM-DD
    xtick_labels = [label.split(" ")[0] for label in xtick_labels]
    ytick_labels = [label.split(" ")[0] for label in ytick_labels]

    # Set the tick labels
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_yticklabels(ytick_labels)

    if show_optimal:
        ax.legend(loc="upper left")

    plt.tight_layout()
    return fig, ax


def _generate_custom_log_colorscale(base_colorscale_name: str, levels: List[float]):
    """Generates a custom Plotly colorscale for logarithmic perception.
    Maps original data levels to colors sampled logarithmically from a base colorscale.
    """
    if not levels:
        return base_colorscale_name

    # 1. Use unique, sorted, finite levels
    original_levels_np = np.array(
        sorted(list(set(l for l in levels if l is not None and np.isfinite(l))))
    )
    if len(original_levels_np) < 2:
        # Not enough points to define a meaningful scale, fallback or use a single color if one level exists
        if len(original_levels_np) == 1:
            color = pcolors.sample_colorscale(base_colorscale_name, [0.5])[0]
            return [[0.0, color], [1.0, color]]
        return base_colorscale_name

    min_orig, max_orig = original_levels_np[0], original_levels_np[-1]

    # 2. Normalize original levels to [0, 1] for output colorscale positions
    if np.isclose(min_orig, max_orig):
        color = pcolors.sample_colorscale(base_colorscale_name, [0.5])[0]
        return [[0.0, color], [1.0, color]]
    norm_orig_positions = (original_levels_np - min_orig) / (max_orig - min_orig)

    # 3. Determine colorscale sampling points (logarithmically mapped from original_levels_np)
    # These points (0-1) are where we pick colors from the base_colorscale_name.
    positive_mask = original_levels_np > 1e-9  # Epsilon for "positive"
    positive_levels_for_log_range = original_levels_np[positive_mask]

    sampling_points = norm_orig_positions.copy()  # Default to linear sampling

    if len(positive_levels_for_log_range) >= 2:
        log_vals = np.log10(positive_levels_for_log_range)
        min_log, max_log = (
            log_vals[0],
            log_vals[-1],
        )  # positive_levels_for_log_range is sorted

        if not np.isclose(min_log, max_log):  # Meaningful log scale range exists
            # For levels that are positive, map them to their log-normalized position
            # For non-positive levels, their sampling_points will remain their linear norm_orig_positions,
            # effectively taking colors from the linear start of the sampled base colorscale.
            # Or, map them to the color of the smallest positive level.
            # Let's try the latter: non-positives get color of min_positive_level.

            # Calculate log-normalized positions for all original_levels_np
            # Values <= smallest positive level get sampling point 0 (color of smallest positive level)
            # Values >= largest positive level get sampling point 1 (color of largest positive level)
            # Values in between are log-interpolated.

            # Clamp original levels to the range of positive levels for log transformation input
            clamped_for_log = np.clip(
                original_levels_np,
                positive_levels_for_log_range[0],
                positive_levels_for_log_range[-1],
            )
            log_of_clamped = np.log10(clamped_for_log)  # All values are now positive

            current_sampling_points = (log_of_clamped - min_log) / (max_log - min_log)
            current_sampling_points = np.clip(current_sampling_points, 0, 1)
            sampling_points = (
                current_sampling_points  # Apply if log scaling was successful
            )

    # 4. Sample the colors from the base colorscale using the determined sampling_points
    colors = pcolors.sample_colorscale(base_colorscale_name, sampling_points)

    # 5. Construct the Plotly colorscale [[norm_pos, color_str], ...]
    custom_scale = []
    for i in range(len(original_levels_np)):
        custom_scale.append([norm_orig_positions[i], colors[i]])

    # Ensure the scale is well-formed (handles cases where norm_orig_positions might not be unique due to float precision)
    # And ensures 0.0 and 1.0 points if they are not exactly hit by norm_orig_positions from few levels.
    # Given original_levels_np is sorted unique and norm_orig_positions are derived, this is simpler.
    # If custom_scale has only one entry due to levels being too close, duplicate it for a valid scale.
    if len(custom_scale) == 1:
        return [[0.0, custom_scale[0][1]], [1.0, custom_scale[0][1]]]

    return custom_scale


def plot_porkchop_plotly(
    porkchop_data: LambertOutput,
    width: int = 900,
    height: int = 700,
    c3_levels: Optional[List[float]] = None,
    c3_colorscale: str = "Viridis",
    log_scale_c3: bool = True,
    tof_levels: Optional[List[float]] = None,
    tof_line_color: str = "red",
    xlim_mjd: Optional[Tuple[float, float]] = None,
    ylim_mjd: Optional[Tuple[float, float]] = None,
    title: str = "Porkchop Plot",
    show_optimal: bool = True,
):
    """
    Plot the porkchop plot from Lambert trajectory data using Plotly.

    Parameters
    ----------
    porkchop_data : LambertOutput
        The output from generate_porkchop_data containing Lambert trajectory solutions.
    width : int, optional
        Figure width in pixels. Default is 900.
    height : int, optional
        Figure height in pixels. Default is 700.
    c3_levels : list, optional
        List of C3 values (km^2/s^2) for contour levels. If None, default levels will be used.
    c3_colorscale : str, optional
        Plotly colorscale name for the C3 filled contours. Default is 'Viridis'.
    log_scale_c3 : bool, optional
        If True, uses logarithmic scale for C3 contours and colorbar. Default is True.
    tof_levels : list, optional
        List of time of flight values (days) for contour levels. If None, dynamic levels will be used.
    tof_line_color : str, optional
        Color for the ToF contour lines. Default is 'red'.
    xlim_mjd : Tuple[float, float], optional
        x-axis limits (min_mjd, max_mjd). If None, will use data range.
    ylim_mjd : Tuple[float, float], optional
        y-axis limits (min_mjd, max_mjd). If None, will use data range.
    title : str, optional
        Plot title. Default is 'Porkchop Plot'.
    show_optimal : bool, optional
        If True, marks the optimal (minimum C3) point on the plot. Default is True.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object.
    """
    # Extract data from LambertOutput
    c3_values_au_d2 = porkchop_data.c3()
    time_of_flight_days = porkchop_data.time_of_flight()
    departure_times_mjd = porkchop_data.departure_state.time.mjd().to_numpy(
        zero_copy_only=False
    )
    arrival_times_mjd = porkchop_data.arrival_state.time.mjd().to_numpy(
        zero_copy_only=False
    )

    fig = go.Figure()

    # Preliminary data checks
    if len(departure_times_mjd) == 0 or len(arrival_times_mjd) == 0:
        warnings.warn(
            "Porkchop plotting: No departure or arrival times available. Returning empty figure."
        )
        # Update layout for an empty plot to at least show axes and title
        fig.update_layout(
            title=title + " (No data)",
            xaxis_title="Departure Date",
            yaxis_title="Arrival Date",
            width=width,
            height=height,
            autosize=False,
            xaxis=dict(type="date"),
            yaxis=dict(type="date"),
        )
        return fig

    # Define unique_departure_times_mjd and unique_arrival_times_mjd here
    unique_departure_times_mjd = np.sort(np.unique(departure_times_mjd))
    unique_arrival_times_mjd = np.sort(np.unique(arrival_times_mjd))

    # Now check the length of the unique times
    if len(unique_departure_times_mjd) < 2 or len(unique_arrival_times_mjd) < 2:
        warnings.warn(
            "Porkchop plotting: Not enough unique departure/arrival times to create a grid. Returning empty figure."
        )
        fig.update_layout(
            title=title + " (Insufficient data for grid)",
            xaxis_title="Departure Date",
            yaxis_title="Arrival Date",
            width=width,
            height=height,
            autosize=False,
            xaxis=dict(type="date"),
            yaxis=dict(type="date"),
        )
        return fig

    # Convert C3 from (AU/day)^2 to km^2/s^2
    # Factor = (km/AU / s/day)^2 = (1731.4568 km/s per AU/day)^2
    km_s_per_au_day = 1731.4568368
    c3_values_km2_s2 = c3_values_au_d2 * (km_s_per_au_day**2)

    # Convert MJD times to datetime objects for Plotly axes
    unique_departure_dates_dt = [
        Time(mjd, format="mjd").datetime for mjd in unique_departure_times_mjd
    ]
    unique_arrival_dates_dt = [
        Time(mjd, format="mjd").datetime for mjd in unique_arrival_times_mjd
    ]

    grid_departure_mjd, grid_arrival_mjd = np.meshgrid(
        unique_departure_times_mjd, unique_arrival_times_mjd
    )

    # Interpolate C3 values onto the regular grid
    # griddata expects points as (N, D) array, so stack departure and arrival times
    points = np.vstack((departure_times_mjd, arrival_times_mjd)).T
    grid_c3_km2_s2 = griddata(
        points,
        c3_values_km2_s2,
        (grid_departure_mjd, grid_arrival_mjd),
        method="cubic",
        fill_value=np.nan,
    )

    # Original TOF grid for hovertext
    original_tof_grid_days = grid_arrival_mjd - grid_departure_mjd

    # Prepare custom hovertext for C3 contours (Vectorized)
    # 1. Convert MJD grids to date strings
    dep_time_obj = Time(grid_departure_mjd.ravel(), format="mjd")
    dep_date_str_flat = dep_time_obj.to_value("iso", subfmt="date")
    dep_date_str_grid = dep_date_str_flat.reshape(grid_departure_mjd.shape)

    arr_time_obj = Time(grid_arrival_mjd.ravel(), format="mjd")
    arr_date_str_flat = arr_time_obj.to_value("iso", subfmt="date")
    arr_date_str_grid = arr_date_str_flat.reshape(grid_arrival_mjd.shape)

    # 2. Format C3 and ToF grids to strings
    # C3
    c3_flat = grid_c3_km2_s2.ravel()
    c3_str_flat = np.full(c3_flat.shape, "N/A", dtype=object)  # Initialize with N/A
    valid_c3_mask_flat = ~np.isnan(c3_flat)
    # Apply formatting only to valid (non-NaN) C3 values
    c3_str_flat[valid_c3_mask_flat] = [
        f"{val:.1f} km²/s²" for val in c3_flat[valid_c3_mask_flat]
    ]
    c3_str_grid = c3_str_flat.reshape(grid_c3_km2_s2.shape)

    # ToF
    tof_flat = original_tof_grid_days.ravel()
    tof_str_flat = np.full(tof_flat.shape, "N/A", dtype=object)
    valid_tof_mask_flat = ~np.isnan(tof_flat)
    tof_str_flat[valid_tof_mask_flat] = [
        f"{val:.1f} days" for val in tof_flat[valid_tof_mask_flat]
    ]
    tof_str_grid = tof_str_flat.reshape(original_tof_grid_days.shape)

    # 3. Concatenate into final hovertext strings
    hover_texts_np = (
        np.char.add(np.char.add("<b>Departure:</b> ", dep_date_str_grid), "<br>")
        + np.char.add(np.char.add("<b>Arrival:</b> ", arr_date_str_grid), "<br>")
        + np.char.add(np.char.add("<b>C3:</b> ", c3_str_grid), "<br>")
        + np.char.add(np.char.add("<b>ToF:</b> ", tof_str_grid), "")
    )
    c3_hovertext = hover_texts_np.tolist()  # Convert to list of lists for Plotly

    # Check if interpolated C3 data is all NaN (MOVED HERE)
    if np.all(np.isnan(grid_c3_km2_s2)):
        warnings.warn(
            "Porkchop plotting: Interpolated C3 data is all NaN. Contour plot may be empty or missing."
        )
        # We can still try to plot the optimal point if raw data exists

    # Initialize a list to hold all plotly traces
    plotly_traces = []

    # Define default contour levels for C3 if not provided
    current_c3_levels = c3_levels
    if current_c3_levels is None:
        current_c3_levels = [
            8.0,
            10.0,
            12.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
            50.0,
            75.0,
            100.0,
        ]

    # Data for C3 plotting is ALWAYS original data now
    c3_data_for_plot = grid_c3_km2_s2
    # Levels for C3 plotting are ALWAYS original levels
    c3_levels_for_plot = sorted([l for l in current_c3_levels if l is not None])

    # Determine colorscale and colorbar for C3 based on log_scale_c3
    final_c3_colorscale_for_fill = c3_colorscale  # Default to the name

    final_c3_colorbar = dict(
        title="C3 (km²/s²)"
        # tickvals and ticktext removed for automatic Plotly ticks
    )

    if log_scale_c3:
        if c3_levels_for_plot and len(c3_levels_for_plot) >= 2:
            final_c3_colorscale_for_fill = _generate_custom_log_colorscale(
                c3_colorscale, c3_levels_for_plot
            )
        # colorbar setup is already fine, shows original levels

    # Replace NaNs in c3_data_for_plot (original scale)
    if np.any(np.isfinite(c3_data_for_plot)):
        max_c3_val = np.nanmax(c3_data_for_plot[np.isfinite(c3_data_for_plot)])
        # Choose a replacement value far from the actual data range.
        # If max_c3_val is large, 1.5*max_c3_val is good.
        # If levels are e.g. 8-100, max_c3_val could be 1000 due to interpolation overshoot.
        # A very large fixed number or relative to overall range might be safer if max_c3_val is small.
        nan_replacement_c3 = max_c3_val * 1.5
        if max_c3_val <= 0:  # handles all zero or negative case
            nan_replacement_c3 = (
                1000  # A large arbitrary positive if data is not positive
            )
        if (
            not np.isfinite(nan_replacement_c3) and c3_levels_for_plot
        ):  # Fallback if calc fails
            nan_replacement_c3 = (
                (max(c3_levels_for_plot) * 2) if c3_levels_for_plot else 1000
            )
        elif not np.isfinite(nan_replacement_c3):
            nan_replacement_c3 = 1000

        c3_data_for_plot = np.nan_to_num(
            c3_data_for_plot,
            nan=nan_replacement_c3,
            posinf=nan_replacement_c3,
            neginf=nan_replacement_c3,
        )

    # C3 Contours
    # 1. C3 Fill Trace (always added)
    plotly_traces.append(
        go.Contour(
            x=unique_departure_dates_dt,
            y=unique_arrival_dates_dt,
            z=c3_data_for_plot,  # ORIGINAL C3 data
            colorscale=final_c3_colorscale_for_fill,  # Custom log or string name
            hovertext=c3_hovertext,  # Assign custom hovertext
            hoverinfo="text",  # Use only custom hovertext
            contours=dict(
                coloring="fill",
                showlabels=True,
                labelfont=dict(size=10, color="black"),
            ),
            line=dict(width=0.5),
            name="C3",
            colorbar=final_c3_colorbar,
            zmin=(
                min(c3_levels_for_plot) if c3_levels_for_plot else None
            ),  # Still use levels for zmin/zmax if available
            zmax=max(c3_levels_for_plot) if c3_levels_for_plot else None,
            autocontour=False,  # Fill trace now uses autocontour
            connectgaps=True,
        )
    )

    # Time of Flight (ToF) Contours
    tof_grid_days = grid_arrival_mjd - grid_departure_mjd

    # Replace NaNs in tof_grid_days
    if np.any(np.isfinite(tof_grid_days)):
        max_tof_val = np.nanmax(tof_grid_days[np.isfinite(tof_grid_days)])
        nan_replacement_tof = (
            max_tof_val * 1.5 if max_tof_val > 0 else 1.5
        )  # Handle all-zero or negative max
        if np.isnan(nan_replacement_tof) or np.isinf(nan_replacement_tof):
            pass  # Don't replace if replacement value is not sensible
        else:
            tof_grid_days = np.nan_to_num(
                tof_grid_days,
                nan=nan_replacement_tof,
                posinf=nan_replacement_tof,
                neginf=nan_replacement_tof,
            )

    current_tof_levels = tof_levels
    if (
        current_tof_levels is None
        and np.any(~np.isnan(c3_values_km2_s2))
        and np.any(~np.isnan(time_of_flight_days))
    ):
        min_c3_idx = np.nanargmin(c3_values_km2_s2)
        optimal_tof = time_of_flight_days[min_c3_idx]

        optimal_tof_rounded = (
            round(optimal_tof / 5) * 5
            if optimal_tof < 100
            else round(optimal_tof / 10) * 10
        )

        # Use nanmin/nanmax on the original tof_grid_days before NaN replacement for level calculation
        original_tof_grid_for_levels = grid_arrival_mjd - grid_departure_mjd
        min_tof_on_grid = (
            np.nanmin(original_tof_grid_for_levels[original_tof_grid_for_levels > 0])
            if np.any(original_tof_grid_for_levels > 0)
            else 0
        )
        max_tof_on_grid = (
            np.nanmax(original_tof_grid_for_levels)
            if np.any(original_tof_grid_for_levels > 0)
            else optimal_tof_rounded + 50
        )

        num_levels = 9
        step_size_tof = (
            max(5.0, round((max_tof_on_grid - min_tof_on_grid) / num_levels / 5) * 5)
            if max_tof_on_grid > min_tof_on_grid
            else 5.0
        )

        half_levels = (num_levels - 1) // 2
        lower_bound = max(
            min_tof_on_grid, optimal_tof_rounded - half_levels * step_size_tof
        )
        upper_bound = min(
            max_tof_on_grid + step_size_tof,
            optimal_tof_rounded + (half_levels + 1) * step_size_tof,
        )

        lower_levels = np.arange(lower_bound, optimal_tof_rounded, step_size_tof)
        upper_levels = np.arange(optimal_tof_rounded, upper_bound, step_size_tof)
        current_tof_levels = np.concatenate([lower_levels, upper_levels])
        current_tof_levels = np.unique(
            current_tof_levels[current_tof_levels > 0]
        ).tolist()
        if not current_tof_levels and optimal_tof_rounded > 0:
            current_tof_levels = [optimal_tof_rounded]

    if current_tof_levels:
        plotly_traces.append(
            go.Contour(
                x=unique_departure_dates_dt,
                y=unique_arrival_dates_dt,
                z=tof_grid_days,  # Use the NaN-replaced version for plotting
                colorscale=[  # force black color
                    [0, "rgb(0,0,0)"],
                    [1, "rgb(0,0,0)"],
                ],
                contours=dict(
                    coloring="lines",
                    showlabels=True,
                    labelfont=dict(size=10, color="red"),
                    start=min(current_tof_levels) if current_tof_levels else None,
                    end=max(current_tof_levels) if current_tof_levels else None,
                ),
                line=dict(color="black", width=0.5, dash="dash"),
                name="Time of Flight (days)",
                showscale=False,
                autocontour=False if current_tof_levels else True,
                hoverinfo="skip",  # Disable hover for ToF lines
                connectgaps=True,
            )
        )

    # Optimal Point
    if show_optimal and np.any(~np.isnan(c3_values_km2_s2)):
        min_c3_idx = np.nanargmin(c3_values_km2_s2)
        best_departure_mjd = departure_times_mjd[min_c3_idx]
        best_arrival_mjd = arrival_times_mjd[min_c3_idx]
        best_tof_days = time_of_flight_days[min_c3_idx]
        best_c3_km2_s2 = c3_values_km2_s2[min_c3_idx]

        best_departure_dt = Time(best_departure_mjd, format="mjd").datetime
        best_arrival_dt = Time(best_arrival_mjd, format="mjd").datetime

        hover_text = (
            f"<b>Optimal Transfer</b><br>"
            f"Departure: {Time(best_departure_mjd, format='mjd').iso.split(' ')[0]}<br>"
            f"Arrival: {Time(best_arrival_mjd, format='mjd').iso.split(' ')[0]}<br>"
            f"C3: {best_c3_km2_s2:.1f} km²/s²<br>"
            f"ToF: {best_tof_days:.1f} days"
        )

        plotly_traces.append(
            go.Scatter(
                x=[best_departure_dt],
                y=[best_arrival_dt],
                mode="markers",
                marker=dict(
                    symbol="star",
                    color="red",
                    size=10,
                    line=dict(color="black", width=1),
                ),
                name=f"Optimal: C3={best_c3_km2_s2:.1f}, ToF={best_tof_days:.1f}d",
                hoverinfo="text",
                text=[hover_text],
            )
        )

    # Determine axis ranges
    final_xlim_dt = None
    final_ylim_dt = None

    if xlim_mjd:
        final_xlim_dt = [
            Time(xlim_mjd[0], format="mjd").datetime,
            Time(xlim_mjd[1], format="mjd").datetime,
        ]
    if ylim_mjd:
        final_ylim_dt = [
            Time(ylim_mjd[0], format="mjd").datetime,
            Time(ylim_mjd[1], format="mjd").datetime,
        ]

    if (final_xlim_dt is None or final_ylim_dt is None) and current_c3_levels:
        data_for_ranging = c3_data_for_plot  # This is already NaN-replaced, which is fine for range finding
        levels_for_ranging = (
            c3_levels_for_plot
            if c3_levels_for_plot
            else [np.nanmin(data_for_ranging), np.nanmax(data_for_ranging)]
        )
        if not levels_for_ranging or all(
            np.isnan(level) for level in levels_for_ranging
        ):
            pass
        else:
            max_c3_for_ranging = np.nanmax(levels_for_ranging)

            # Use a mask that considers finite values for ranging, not the replaced NaNs.
            # Original c3_data_for_plot before NaN replacement might be better here if available, but
            # using the original grid_c3_km2_s2 (or its log version if log_scale_c3) is safer.
            # Since c3_data_for_plot is now always original, this logic simplifies.
            # The autoranging should be based on where the contours are actually drawn.
            # The levels_for_ranging should be the original c3_levels_for_plot.
            max_c3_for_ranging = (
                np.nanmax(c3_levels_for_plot)
                if c3_levels_for_plot
                else np.nanmax(c3_data_for_plot[np.isfinite(c3_data_for_plot)])
            )
            if not np.isfinite(max_c3_for_ranging):
                pass  # Can't range if no valid max level or data
            else:
                # valid_mask uses original c3_data_for_plot against original max_c3_for_ranging
                valid_mask = (~np.isnan(grid_c3_km2_s2)) & (
                    grid_c3_km2_s2 <= max_c3_for_ranging
                )  # grid_c3_km2_s2 is pre-NaN replacement

                if np.any(valid_mask):
                    valid_cols_mask = np.any(valid_mask, axis=0)
                    valid_rows_mask = np.any(valid_mask, axis=1)

                    if np.any(valid_cols_mask) and np.any(valid_rows_mask):
                        valid_col_indices = np.where(valid_cols_mask)[0]
                        valid_row_indices = np.where(valid_rows_mask)[0]

                        col_padding = max(2, int(0.05 * len(valid_col_indices)))
                        row_padding = max(2, int(0.05 * len(valid_row_indices)))

                        if final_xlim_dt is None:
                            min_x_mjd = np.min(departure_times_mjd)
                            max_x_idx = min(
                                len(unique_departure_times_mjd) - 1,
                                valid_col_indices[-1] + col_padding,
                            )
                            max_x_mjd = unique_departure_times_mjd[max_x_idx]
                            final_xlim_dt = [
                                Time(min_x_mjd, format="mjd").datetime,
                                Time(max_x_mjd, format="mjd").datetime,
                            ]

                        if final_ylim_dt is None:
                            min_y_idx = max(0, valid_row_indices[0] - row_padding)
                            min_y_mjd = unique_arrival_times_mjd[min_y_idx]
                            max_y_mjd = np.max(arrival_times_mjd)
                            final_ylim_dt = [
                                Time(min_y_mjd, format="mjd").datetime,
                                Time(max_y_mjd, format="mjd").datetime,
                            ]

    # Create the figure with all traces
    fig = go.Figure(data=plotly_traces)

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Departure Date",
        yaxis_title="Arrival Date",
        xaxis=dict(
            tickformat="%Y-%m-%d",
            tickangle=-45,
            range=final_xlim_dt,
        ),
        yaxis=dict(
            tickformat="%Y-%m-%d",
            range=final_ylim_dt,
        ),
        width=width,
        height=height,
        autosize=False,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
