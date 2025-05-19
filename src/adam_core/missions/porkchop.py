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
from scipy import ndimage
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
        return np.linalg.norm(self.vx_2 - self.departure_state.v, axis=1)

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
    step_size: float = 1.0,
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

    print(f"len(c3_values_km2_s2): {len(c3_values_km2_s2)}")
    print(f"len(grid_c3): {len(grid_c3)}")

    # Define default contour levels for C3 if not provided
    if c3_levels is None:
        c3_levels = np.linspace(8.0, 100.0, 11)  # 11 levels from 8 to 100

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

    if len(custom_scale) == 1:
        return [[0.0, custom_scale[0][1]], [1.0, custom_scale[0][1]]]

    return custom_scale


def _generate_hovertext_grid(
    grid_departure_mjd: npt.NDArray[np.float64],
    grid_arrival_mjd: npt.NDArray[np.float64],
    original_tof_grid_days: npt.NDArray[np.float64],
    grid_c3_km2_s2: npt.NDArray[np.float64],
    grid_vinf_km_s: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.str_]]:
    # --- Hovertext Preparation ---
    dep_time_obj = Time(grid_departure_mjd.ravel(), format="mjd")
    dep_date_str_flat = dep_time_obj.to_value("iso", subfmt="date")
    dep_date_str_grid = dep_date_str_flat.reshape(grid_departure_mjd.shape)

    arr_time_obj = Time(grid_arrival_mjd.ravel(), format="mjd")
    arr_date_str_flat = arr_time_obj.to_value("iso", subfmt="date")
    arr_date_str_grid = arr_date_str_flat.reshape(grid_arrival_mjd.shape)

    tof_flat_grid = original_tof_grid_days.ravel()
    tof_str_flat_grid = np.full(tof_flat_grid.shape, "N/A", dtype=object)
    valid_tof_mask_flat_grid = ~np.isnan(tof_flat_grid)
    tof_str_flat_grid[valid_tof_mask_flat_grid] = [
        f"{val:.1f} days" for val in tof_flat_grid[valid_tof_mask_flat_grid]
    ]
    tof_str_grid_final = tof_str_flat_grid.reshape(original_tof_grid_days.shape)

    base_hover_text_np = np.char.add(
        np.char.add("<b>Departure:</b> ", dep_date_str_grid), "<br>"
    ) + np.char.add(np.char.add("<b>Arrival:</b> ", arr_date_str_grid), "<br>")

    c3_metric_flat_grid = grid_c3_km2_s2.ravel()
    c3_metric_str_flat_grid = np.full(c3_metric_flat_grid.shape, "N/A", dtype=object)
    valid_c3_metric_mask_flat_grid = ~np.isnan(c3_metric_flat_grid)
    c3_metric_str_flat_grid[valid_c3_metric_mask_flat_grid] = [
        f"{val:.1f} km²/s²"
        for val in c3_metric_flat_grid[valid_c3_metric_mask_flat_grid]
    ]
    c3_metric_str_grid_final = c3_metric_str_flat_grid.reshape(grid_c3_km2_s2.shape)
    c3_hovertext_grid = (
        base_hover_text_np
        + np.char.add(np.char.add("<b>C3:</b> ", c3_metric_str_grid_final), "<br>")
        + np.char.add(np.char.add("<b>ToF:</b> ", tof_str_grid_final), "")
    ).tolist()

    vinf_metric_flat_grid = grid_vinf_km_s.ravel()
    vinf_metric_str_flat_grid = np.full(
        vinf_metric_flat_grid.shape, "N/A", dtype=object
    )
    valid_vinf_metric_mask_flat_grid = ~np.isnan(vinf_metric_flat_grid)
    vinf_metric_str_flat_grid[valid_vinf_metric_mask_flat_grid] = [
        f"{val:.1f} km/s"
        for val in vinf_metric_flat_grid[valid_vinf_metric_mask_flat_grid]
    ]
    vinf_metric_str_grid_final = vinf_metric_str_flat_grid.reshape(grid_vinf_km_s.shape)
    vinf_hovertext_grid = (
        base_hover_text_np
        + np.char.add(np.char.add("<b>Vinf:</b> ", vinf_metric_str_grid_final), "<br>")
        + np.char.add(np.char.add("<b>ToF:</b> ", tof_str_grid_final), "")
    ).tolist()

    return vinf_hovertext_grid, c3_hovertext_grid

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
    tof_line_color: str = "black",
    xlim_mjd: Optional[Tuple[float, float]] = None,
    ylim_mjd: Optional[Tuple[float, float]] = None,
    title: str = "Porkchop Plot",
    show_optimal: bool = True,
):
    """
    Plot the porkchop plot from Lambert trajectory data using Plotly,
    with a button to toggle between C3 and Vinf (departure) views.

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
        Minimum Vinf value (km/s) for contour levels.
    vinf_max : float, optional
        Maximum Vinf value (km/s) for contour levels.
    vinf_step : float, optional
        Step size for Vinf contour levels.
    tof_min : float, optional
        Minimum ToF value (days) for contour levels.
    tof_max : float, optional
        Maximum ToF value (days) for contour levels.
    tof_step : float, optional
        Step size for ToF contour levels.
    metric_colorscale : str, optional
        Plotly colorscale name for the C3/Vinf filled contours.
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

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object.
    """
    # --- Data Extraction and Basic Setup ---
    c3_values_au_d2 = porkchop_data.c3()  # C3 in (AU/day)^2
    vinf_values_au_day = np.sqrt(
        np.maximum(0, c3_values_au_d2)
    )  # Departure Vinf in AU/day

    time_of_flight_days = porkchop_data.time_of_flight()
    departure_times = porkchop_data.departure_state.time
    departure_times_mjd = departure_times.mjd().to_numpy(
        zero_copy_only=False
    )
    arrival_times = porkchop_data.arrival_state.time
    arrival_times_mjd = arrival_times.mjd().to_numpy(
        zero_copy_only=False
    )

    unique_departure_times = departure_times.unique()
    unique_arrival_times = arrival_times.unique()
    unique_departure_times_mjd = np.sort(np.unique(departure_times_mjd))
    unique_arrival_times_mjd = np.sort(np.unique(arrival_times_mjd))
    unique_departure_dates_dt = [dt.datetime for dt in unique_departure_times.to_astropy()]
    unique_arrival_dates_dt = [dt.datetime for dt in unique_arrival_times.to_astropy()]


    if len(unique_departure_times) < 2 or len(unique_arrival_times) < 2:
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


    # --- Unit Conversions and Grid Setup ---
    km_s_per_au_day = KM_P_AU / S_P_DAY
    c3_values_km2_s2 = c3_values_au_d2 * (km_s_per_au_day**2)
    vinf_values_km_s = vinf_values_au_day * km_s_per_au_day

    grid_departure_mjd, grid_arrival_mjd = np.meshgrid(
        unique_departure_times_mjd, unique_arrival_times_mjd
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

    vinf_hovertext_grid, c3_hovertext_grid = _generate_hovertext_grid(
        grid_departure_mjd,
        grid_arrival_mjd,
        original_tof_grid_days,
        grid_c3_km2_s2,
        grid_vinf_km_s,
    )

    # Determine contour level parameters
    # For C3
    if c3_min is None:
        c3_min = max(8.0, np.nanpercentile(grid_c3_km2_s2, 5))  # 5th percentile with minimum floor
    if c3_max is None:
        c3_max = min(100.0, np.nanpercentile(grid_c3_km2_s2, 95))  # 95th percentile with maximum ceiling
    if c3_step is None:
        c3_step = (c3_max - c3_min) / 10  # 10 levels by default
    
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

    # --- Replace NaN values and over-max values with sentinel ---
    # Use a sentinel value that's 2x the maximum
    sentinel_value = c3_max * 2

    # Replace NaN and over-max values with the sentinel value
    grid_c3_for_plot = np.copy(grid_c3_km2_s2)
    mask_nan = np.isnan(grid_c3_for_plot)
    mask_over = grid_c3_for_plot > c3_max
    
    # Apply sentinel value to all invalid areas
    grid_c3_for_plot[mask_nan | mask_over] = sentinel_value

    # Apply the same approach for Vinf
    sentinel_value_vinf = vinf_max * 2
    grid_vinf_for_plot = np.copy(grid_vinf_km_s)
    mask_nan_vinf = np.isnan(grid_vinf_for_plot)
    mask_over_vinf = grid_vinf_for_plot > vinf_max
    
    # Apply sentinel value to all invalid areas
    grid_vinf_for_plot[mask_nan_vinf | mask_over_vinf] = sentinel_value_vinf

    # --- Update the hover text for sentinel values ---
    # Create masks for sentinel values
    c3_sentinel_mask = (grid_c3_for_plot == sentinel_value)
    vinf_sentinel_mask = (grid_vinf_for_plot == sentinel_value_vinf)

    # Convert hovertext to numpy arrays for easier manipulation
    c3_hovertext_grid_np = np.array(c3_hovertext_grid)
    vinf_hovertext_grid_np = np.array(vinf_hovertext_grid)

    # Update hover text for sentinel values
    for i in range(c3_sentinel_mask.shape[0]):
        for j in range(c3_sentinel_mask.shape[1]):
            if c3_sentinel_mask[i, j]:
                parts = c3_hovertext_grid_np[i, j].split("<br>")
                parts[2] = "<b>C3:</b> N/A"
                parts[3] = "<b>ToF:</b> N/A"
                c3_hovertext_grid_np[i, j] = "<br>".join(parts)

    for i in range(vinf_sentinel_mask.shape[0]):
        for j in range(vinf_sentinel_mask.shape[1]):
            if vinf_sentinel_mask[i, j]:
                parts = vinf_hovertext_grid_np[i, j].split("<br>")
                parts[2] = "<b>Vinf:</b> N/A" 
                parts[3] = "<b>ToF:</b> N/A"
                vinf_hovertext_grid_np[i, j] = "<br>".join(parts)

    # Create hoverinfo arrays (all set to "text")
    c3_hoverinfo = np.full(c3_sentinel_mask.shape, "text", dtype=object)
    vinf_hoverinfo = np.full(vinf_sentinel_mask.shape, "text", dtype=object)

    # Convert numpy arrays back to lists for Plotly
    c3_hovertext_grid_updated = c3_hovertext_grid_np.tolist()
    vinf_hovertext_grid_updated = vinf_hovertext_grid_np.tolist()

    # --- Create custom colorscales ---
    # Define function to create custom colorscale with white for sentinel
    def create_colorscale_with_sentinel(base_colorscale, vmin, vmax, sentinel):
        # Get standard colorscale
        standard_colors = pcolors.sample_colorscale(base_colorscale, np.linspace(0, 1, 11))
        
        # Convert to normalized range (0-1)
        norm_max = 0.9  # Reserve top 10% for sentinel
        
        # Create standard part of colorscale
        colorscale = [(i * norm_max / 10, color) for i, color in enumerate(standard_colors)]
        
        # Add sentinel value as white
        colorscale.append((1.0, 'rgba(255,255,255,1)'))  # White with transparency
        
        return colorscale

    # Create custom colorscales
    c3_colorscale = create_colorscale_with_sentinel(
        metric_colorscale, c3_min, c3_max, sentinel_value
    )
    vinf_colorscale = create_colorscale_with_sentinel(
        metric_colorscale, vinf_min, vinf_max, sentinel_value_vinf
    )

    # --- Update C3 Contour Trace ---
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
            hovertext=c3_hovertext_grid_updated,
            hoverinfo=c3_hoverinfo.tolist(),
            contours=dict(
                coloring="fill", 
                showlabels=True, 
                labelfont=dict(size=10, color="black"),
                start=c3_min,
                end=c3_max,
                size=c3_step
            ),
            line=dict(width=0.5, smoothing=1.3),
            name="C3",
            colorbar=dict(
                title="C3 (km²/s²)",
                # Cap the colorbar range to exclude sentinel
                tickvals=np.linspace(c3_min, c3_max, 6),
                ticktext=[f"{val:.1f}" for val in np.linspace(c3_min, c3_max, 6)]
            ),
            connectgaps=True,
            visible=True,
        )
    )

    # --- Update Vinf Contour Trace ---
    plotly_traces.append(
        go.Contour(
            x=unique_departure_dates_dt,
            y=unique_arrival_dates_dt,
            z=grid_vinf_for_plot,
            zauto=False,
            zmin=vinf_min,
            zmax=vinf_max * 1.1,
            colorscale=vinf_colorscale,
            hovertext=vinf_hovertext_grid_updated,
            hoverinfo=vinf_hoverinfo.tolist(),
            contours=dict(
                coloring="fill", 
                showlabels=True, 
                labelfont=dict(size=10, color="black"),
                start=vinf_min,
                end=vinf_max,
                size=vinf_step
            ),
            line=dict(width=0.5, smoothing=1.3),
            name="Vinf",
            colorbar=dict(
                title="Vinf (km/s)",
                tickvals=np.linspace(vinf_min, vinf_max, 6),
                ticktext=[f"{val:.1f}" for val in np.linspace(vinf_min, vinf_max, 6)]
            ),
            connectgaps=True,
            visible=False,
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
                size=tof_step
            ),
            line=dict(color=tof_line_color, width=1, dash="dash"),
            name="ToF (days)",
            showscale=False,
            hoverinfo="skip",
            connectgaps=False,  # Don't connect across NaN gaps
            visible=True,
        )
    )

    # --- Optimal C3 point (visible with C3 plot) ---
    if show_optimal and np.any(~np.isnan(c3_values_km2_s2)):
        min_c3_idx = np.nanargmin(c3_values_km2_s2)
        best_dep_mjd_c3 = departure_times_mjd[min_c3_idx]
        best_arr_mjd_c3 = arrival_times_mjd[min_c3_idx]
        best_tof_c3 = time_of_flight_days[min_c3_idx]
        best_c3_val = c3_values_km2_s2[min_c3_idx]
        
        ht_c3 = f"<b>Optimal C3</b><br>Dep: {Time(best_dep_mjd_c3, format='mjd').iso[:10]}<br>Arr: {Time(best_arr_mjd_c3, format='mjd').iso[:10]}<br>C3: {best_c3_val:.1f} km²/s²<br>ToF: {best_tof_c3:.1f} d"
        
        plotly_traces.append(
            go.Scatter(
                x=[Time(best_dep_mjd_c3, format="mjd").datetime],
                y=[Time(best_arr_mjd_c3, format="mjd").datetime],
                mode="markers",
                marker=dict(
                    symbol="star",
                    color="red",
                    size=12,
                    line=dict(color="black", width=1),
                ),
                showlegend=False,
                name=f"Opt C3: {best_c3_val:.1f}",
                hoverinfo="text",
                text=[ht_c3],
                visible=True,
                hoverlabel=dict(bgcolor="red"),
                hovertemplate=ht_c3,
            )
        )
    else:
        # Add placeholder to maintain trace index positions
        plotly_traces.append(go.Scatter(visible=False))

    # --- Optimal Vinf point (visible with Vinf plot) ---
    if show_optimal and np.any(~np.isnan(vinf_values_km_s)):
        min_vinf_idx = np.nanargmin(vinf_values_km_s)
        best_dep_mjd_vinf = departure_times_mjd[min_vinf_idx]
        best_arr_mjd_vinf = arrival_times_mjd[min_vinf_idx]
        best_tof_vinf = time_of_flight_days[min_vinf_idx]
        best_vinf_val = vinf_values_km_s[min_vinf_idx]
        
        ht_vinf = f"<b>Optimal Vinf</b><br>Dep: {Time(best_dep_mjd_vinf, format='mjd').iso[:10]}<br>Arr: {Time(best_arr_mjd_vinf, format='mjd').iso[:10]}<br>Vinf: {best_vinf_val:.1f} km/s<br>ToF: {best_tof_vinf:.1f} d"
        
        plotly_traces.append(
            go.Scatter(
                x=[Time(best_dep_mjd_vinf, format="mjd").datetime],
                y=[Time(best_arr_mjd_vinf, format="mjd").datetime],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    color="blue",
                    size=12,
                    line=dict(color="black", width=1),
                ),
                showlegend=False,
                name=f"Opt Vinf: {best_vinf_val:.1f}",
                hoverinfo="text",
                text=[ht_vinf],
                visible=False,
                hoverlabel=dict(bgcolor="blue"),
                hovertemplate=ht_vinf,
            )
        )
    else:
        # Add placeholder to maintain trace index positions
        plotly_traces.append(go.Scatter(visible=False))

    # --- Axis Range Determination ---
    final_xlim_dt = (
        [
            Time(xlim_mjd[0], format="mjd").datetime,
            Time(xlim_mjd[1], format="mjd").datetime,
        ]
        if xlim_mjd
        else (
            [min(unique_departure_dates_dt), max(unique_departure_dates_dt)]
            if unique_departure_dates_dt
            else None
        )
    )
    final_ylim_dt = (
        [
            Time(ylim_mjd[0], format="mjd").datetime,
            Time(ylim_mjd[1], format="mjd").datetime,
        ]
        if ylim_mjd
        else (
            [min(unique_arrival_dates_dt), max(unique_arrival_dates_dt)]
            if unique_arrival_dates_dt
            else None
        )
    )

    # --- Figure Creation and Layout Update ---
    fig = go.Figure(data=plotly_traces)

    # Correct trace ordering for visibility arrays:
    # 0: C3 contour
    # 1: Vinf contour
    # 2: ToF contours
    # 3: Optimal C3 point
    # 4: Optimal Vinf point

    # Visibility for C3 view - Showing C3 contour, ToF contours, and C3 optimal point
    c3_view_visibility = [True, False, True, True, False]  # Default visibility pattern

    # Visibility for Vinf view - Showing Vinf contour, ToF contours, and Vinf optimal point
    vinf_view_visibility = [False, True, True, False, True]  # Default visibility pattern

    fig.update_layout(
        xaxis_title="Departure Date",
        yaxis_title="Arrival Date",
        xaxis=dict(tickformat="%Y-%m-%d", tickangle=-45, range=final_xlim_dt),
        yaxis=dict(tickformat="%Y-%m-%d", range=final_ylim_dt),
        width=width,
        height=height,
        autosize=False,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1,
                xanchor="right",
                y=1.15,
                yanchor="top",
                showactive=True,
                buttons=[
                    dict(
                        label="Display C3",
                        method="restyle",  # Use "restyle" for visibility changes
                        args=[
                            {"visible": c3_view_visibility[: len(fig.data)]},
                            list(range(len(fig.data))),
                        ],
                    ),
                    # args2 for title update is not directly supported with "restyle" for multiple traces easily.
                    # Title update will be handled by a separate call or by initial setting.
                    dict(
                        label="Display Vinf",
                        method="restyle",
                        args=[
                            {"visible": vinf_view_visibility[: len(fig.data)]},
                            list(range(len(fig.data))),
                        ],
                    ),
                ],
            )
        ],
    )
    # Set initial title - this will be static unless we add more complex callbacks or change approach
    # For simplicity, the title will remain the base title, or users can set it when calling.
    # If dynamic title update with buttons is critical, it requires a different approach (e.g. custom JS or Dash)
    fig.update_layout(title_text=title)  # Set base title

    return fig
