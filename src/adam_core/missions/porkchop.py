import multiprocessing as mp
import time
import warnings
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import quivr as qv
import ray
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
