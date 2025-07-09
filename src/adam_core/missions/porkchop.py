import logging
import multiprocessing as mp
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import quivr as qv
import ray
from astropy.time import Time

from adam_core.coordinates import CartesianCoordinates, transform_coordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.coordinates.units import au_per_day_to_km_per_s
from adam_core.dynamics.lambert import calculate_c3, solve_lambert
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils import get_perturber_state
from adam_core.utils.iter import _iterate_chunk_indices
from adam_core.utils.plots.logos import AsteroidInstituteLogoLight, get_logo_base64

logger = logging.getLogger(__name__)


def generate_saturated_colorscale(
    base_color: str, n_levels: int = 8, max_alpha: float = 0.8, min_alpha: float = 0.1
) -> List[List]:
    """
    Generate a colorscale from light to dark based on a base color with full saturation
    and variable transparency that increases with color intensity.

    Parameters
    ----------
    base_color : str
        Base color name (e.g., 'red', 'blue') or hex code (e.g., '#FF0000')
    n_levels : int, optional
        Number of levels in the colorscale (default: 8)
    max_alpha : float, optional
        Maximum alpha (opacity) for darkest colors (default: 0.8)
    min_alpha : float, optional
        Minimum alpha (opacity) for lightest colors (default: 0.1)

    Returns
    -------
    List[List]
        Plotly colorscale format with RGBA: [[position, color], ...]
    """
    # Color mapping for common base colors to RGB with full saturation
    color_map = {
        "red": (255, 0, 0),  # Pure red, full saturation
        "blue": (0, 0, 255),  # Pure blue, full saturation
        "green": (0, 255, 0),  # Pure green, full saturation
        "orange": (255, 165, 0),  # Pure orange, full saturation
        "purple": (128, 0, 128),  # Pure purple, full saturation
        "yellow": (255, 255, 0),  # Pure yellow, full saturation
        "cyan": (0, 255, 255),  # Pure cyan, full saturation
        "magenta": (255, 0, 255),  # Pure magenta, full saturation
    }

    # Parse base color
    if base_color.startswith("#"):
        # Hex color
        hex_color = base_color.lstrip("#")
        base_rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    elif base_color.lower() in color_map:
        base_rgb = color_map[base_color.lower()]
    else:
        # Default to pure red if unknown
        base_rgb = (255, 0, 0)

    colorscale = []
    for i in range(n_levels):
        position = i / (n_levels - 1)

        # Create lightness variation while maintaining full saturation
        # Lighter colors: mix with white but preserve hue
        # Darker colors: reduce brightness but keep saturation
        if position == 0:
            # Lightest: mix with white for pastel effect
            lightness_factor = 0.9  # Very light
            r = int(base_rgb[0] * lightness_factor + 255 * (1 - lightness_factor))
            g = int(base_rgb[1] * lightness_factor + 255 * (1 - lightness_factor))
            b = int(base_rgb[2] * lightness_factor + 255 * (1 - lightness_factor))
        else:
            # Use power curve for smooth transition
            intensity = np.power(position, 0.8)

            # Maintain saturation by scaling from full saturation down
            r = int(base_rgb[0] * (0.3 + 0.7 * intensity))
            g = int(base_rgb[1] * (0.3 + 0.7 * intensity))
            b = int(base_rgb[2] * (0.3 + 0.7 * intensity))

        # Ensure values are within valid RGB range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        # Calculate alpha based on position (lighter = more transparent)
        alpha = min_alpha + (max_alpha - min_alpha) * position

        colorscale.append([position, f"rgba({r}, {g}, {b}, {alpha:.2f})"])

    return colorscale


def generate_perceptual_colorscale(
    base_color: str,
    n_levels: int = 8,
    min_lightness: float = 0.3,
    max_lightness: float = 0.9,
    max_alpha: float = 0.8,
    min_alpha: float = 0.1,
) -> List[List]:
    """
    Generate a perceptually uniform colorscale with full saturation and variable transparency
    that works better for overlaying contours.

    Parameters
    ----------
    base_color : str
        Base color name (e.g., 'red', 'blue') or hex code (e.g., '#FF0000')
    n_levels : int, optional
        Number of levels in the colorscale (default: 8)
    min_lightness : float, optional
        Minimum lightness value (0-1, default: 0.3 for good contrast)
    max_lightness : float, optional
        Maximum lightness value (0-1, default: 0.9 for visibility with transparency)
    max_alpha : float, optional
        Maximum alpha (opacity) for darkest colors (default: 0.8)
    min_alpha : float, optional
        Minimum alpha (opacity) for lightest colors (default: 0.1)

    Returns
    -------
    List[List]
        Plotly colorscale format with RGBA: [[position, color], ...]
    """
    # Full saturation color mapping for maximum color purity
    color_map = {
        "red": (255, 0, 0),  # Pure red
        "blue": (0, 0, 255),  # Pure blue
        "green": (0, 255, 0),  # Pure green
        "orange": (255, 165, 0),  # Pure orange
        "purple": (128, 0, 128),  # Pure purple
        "yellow": (255, 255, 0),  # Pure yellow
        "cyan": (0, 255, 255),  # Pure cyan
        "magenta": (255, 0, 255),  # Pure magenta
    }

    # Parse base color
    if base_color.startswith("#"):
        hex_color = base_color.lstrip("#")
        base_rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    elif base_color.lower() in color_map:
        base_rgb = color_map[base_color.lower()]
    else:
        base_rgb = (255, 0, 0)  # Default pure red

    # Convert base color to normalized RGB for calculations
    base_r, base_g, base_b = [x / 255.0 for x in base_rgb]

    colorscale = []
    for i in range(n_levels):
        position = i / (n_levels - 1)

        # Create perceptually uniform lightness steps
        lightness = max_lightness - (position * (max_lightness - min_lightness))

        # Use full saturation throughout, varying only lightness

        if position == 0:
            # Lightest: mix with white while maintaining hue
            white_mix = 1 - lightness
            r = base_r * lightness + white_mix
            g = base_g * lightness + white_mix
            b = base_b * lightness + white_mix
        else:
            # Scale the base color by lightness while maintaining saturation
            # Use HSV-like scaling to preserve hue and saturation
            max_component = max(base_r, base_g, base_b)

            if max_component > 0:
                # Scale all components proportionally to achieve desired lightness
                scale_factor = lightness / max_component
                r = base_r * scale_factor
                g = base_g * scale_factor
                b = base_b * scale_factor
            else:
                r = g = b = lightness

        # Convert back to 0-255 range and ensure validity
        r_int = max(0, min(255, int(r * 255)))
        g_int = max(0, min(255, int(g * 255)))
        b_int = max(0, min(255, int(b * 255)))

        # Calculate alpha based on position (lighter = more transparent)
        alpha = min_alpha + (max_alpha - min_alpha) * position

        colorscale.append([position, f"rgba({r_int}, {g_int}, {b_int}, {alpha:.2f})"])

    return colorscale


# class LambertOutput(qv.Table):
class LambertSolutions(qv.Table):
    # departure_state = CartesianCoordinates.as_column()
    # arrival_state = CartesianCoordinates.as_column()
    departure_body_id = qv.LargeStringColumn()
    departure_time = Timestamp.as_column()
    departure_body_x = qv.Float64Column()
    departure_body_y = qv.Float64Column()
    departure_body_z = qv.Float64Column()
    departure_body_vx = qv.Float64Column()
    departure_body_vy = qv.Float64Column()
    departure_body_vz = qv.Float64Column()
    arrival_body_id = qv.LargeStringColumn()
    arrival_time = Timestamp.as_column()
    arrival_body_x = qv.Float64Column()
    arrival_body_y = qv.Float64Column()
    arrival_body_z = qv.Float64Column()
    arrival_body_vx = qv.Float64Column()
    arrival_body_vy = qv.Float64Column()
    arrival_body_vz = qv.Float64Column()
    solution_departure_vx = qv.Float64Column()
    solution_departure_vy = qv.Float64Column()
    solution_departure_vz = qv.Float64Column()
    solution_arrival_vx = qv.Float64Column()
    solution_arrival_vy = qv.Float64Column()
    solution_arrival_vz = qv.Float64Column()
    frame = qv.StringAttribute(default="unspecified")
    origin = Origin.as_column()

    def departure_body_orbit(self) -> Orbits:
        """
        Return the departure body orbit.
        """
        return Orbits.from_kwargs(
            orbit_id=self.departure_body_id,
            coordinates=CartesianCoordinates.from_kwargs(
                time=self.departure_time,
                x=self.departure_body_x,
                y=self.departure_body_y,
                z=self.departure_body_z,
                vx=self.departure_body_vx,
                vy=self.departure_body_vy,
                vz=self.departure_body_vz,
                origin=self.origin,
                frame=self.frame,
            ),
        )

    def arrival_body_orbit(self) -> Orbits:
        """
        Return the arrival body orbit.
        """
        return Orbits.from_kwargs(
            orbit_id=self.arrival_body_id,
            coordinates=CartesianCoordinates.from_kwargs(
                time=self.arrival_time,
                x=self.arrival_body_x,
                y=self.arrival_body_y,
                z=self.arrival_body_z,
                vx=self.arrival_body_vx,
                vy=self.arrival_body_vy,
                vz=self.arrival_body_vz,
                origin=self.origin,
                frame=self.frame,
            ),
        )

    def solution_departure_orbit(self) -> Orbits:
        """
        Return the solution departure orbit.
        """
        solution_departure_orbit_id = [
            f"solution_departure_orbit_{i}"
            for i in range(len(self.solution_departure_vx))
        ]
        return Orbits.from_kwargs(
            orbit_id=solution_departure_orbit_id,
            coordinates=CartesianCoordinates.from_kwargs(
                time=self.departure_time,
                x=self.departure_body_x,
                y=self.departure_body_y,
                z=self.departure_body_z,
                vx=self.solution_departure_vx,
                vy=self.solution_departure_vy,
                vz=self.solution_departure_vz,
                origin=self.origin,
                frame=self.frame,
            ),
        )

    def solution_arrival_orbit(self) -> Orbits:
        """
        Return the solution arrival orbit.
        """
        solution_arrival_orbit_id = [
            f"solution_arrival_orbit_{i}" for i in range(len(self.solution_arrival_vx))
        ]
        return Orbits.from_kwargs(
            orbit_id=solution_arrival_orbit_id,
            coordinates=CartesianCoordinates.from_kwargs(
                time=self.arrival_time,
                x=self.arrival_body_x,
                y=self.arrival_body_y,
                z=self.arrival_body_z,
                vx=self.solution_arrival_vx,
                vy=self.solution_arrival_vy,
                vz=self.solution_arrival_vz,
                origin=self.origin,
                frame=self.frame,
            ),
        )

    def c3_departure(self) -> npt.NDArray[np.float64]:
        """
        Return the C3 in au^2/d^2.
        """
        return calculate_c3(
            np.array(
                self.table.select(
                    [
                        "solution_departure_vx",
                        "solution_departure_vy",
                        "solution_departure_vz",
                    ]
                )
            ),
            np.array(
                self.table.select(
                    ["departure_body_vx", "departure_body_vy", "departure_body_vz"]
                )
            ),
        )

    def c3_arrival(self) -> npt.NDArray[np.float64]:
        """
        Return the C3 in au^2/d^2.
        """
        return calculate_c3(
            np.array(
                self.table.select(
                    [
                        "solution_arrival_vx",
                        "solution_arrival_vy",
                        "solution_arrival_vz",
                    ]
                )
            ),
            np.array(
                self.table.select(
                    ["arrival_body_vx", "arrival_body_vy", "arrival_body_vz"]
                )
            ),
        )

    def vinf_departure(self) -> npt.NDArray[np.float64]:
        """
        Return the v infinity in au/d.
        """
        return np.linalg.norm(
            np.array(
                self.table.select(
                    [
                        "solution_departure_vx",
                        "solution_departure_vy",
                        "solution_departure_vz",
                    ]
                )
            )
            - np.array(
                self.table.select(
                    ["departure_body_vx", "departure_body_vy", "departure_body_vz"]
                )
            ),
            axis=1,
        )

    def vinf_arrival(self) -> npt.NDArray[np.float64]:
        """
        Return the v infinity in au/d.
        """
        return np.linalg.norm(
            np.array(
                self.table.select(
                    [
                        "solution_arrival_vx",
                        "solution_arrival_vy",
                        "solution_arrival_vz",
                    ]
                )
            )
            - np.array(
                self.table.select(
                    ["arrival_body_vx", "arrival_body_vy", "arrival_body_vz"]
                )
            ),
            axis=1,
        )

    def time_of_flight(self) -> npt.NDArray[np.float64]:
        """
        Return the time of flight in days.
        """
        return self.arrival_time.mjd().to_numpy(
            zero_copy_only=False
        ) - self.departure_time.mjd().to_numpy(zero_copy_only=False)


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
    assert (
        len(vx) == len(vy) == len(vz) == len(times)
    ), "All arrays must have the same length"
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
        origin_out=departure_origin,
    )
    return spherical


def lambert_worker(
    departure_orbits: Orbits,
    arrival_orbits: Orbits,
    propagation_origin: OriginCodes,
    prograde: bool = True,
    max_iter: int = 35,
    tol: float = 1e-10,
) -> LambertSolutions:
    # Extract coordinates from orbits
    departure_coordinates = departure_orbits.coordinates
    arrival_coordinates = arrival_orbits.coordinates

    r1 = departure_coordinates.r
    r2 = arrival_coordinates.r
    tof = arrival_coordinates.time.mjd().to_numpy(
        zero_copy_only=False
    ) - departure_coordinates.time.mjd().to_numpy(zero_copy_only=False)

    origins = Origin.from_OriginCodes(propagation_origin, size=len(r1))
    mu = origins.mu()[0]
    v1, v2 = solve_lambert(r1, r2, tof, mu, prograde, max_iter, tol)

    # Use actual orbit IDs from the Orbits objects
    departure_body_ids = departure_orbits.orbit_id.to_pylist()
    arrival_body_ids = arrival_orbits.orbit_id.to_pylist()

    return LambertSolutions.from_kwargs(
        departure_body_id=departure_body_ids,
        departure_time=departure_coordinates.time,
        departure_body_x=departure_coordinates.x,
        departure_body_y=departure_coordinates.y,
        departure_body_z=departure_coordinates.z,
        departure_body_vx=departure_coordinates.vx,
        departure_body_vy=departure_coordinates.vy,
        departure_body_vz=departure_coordinates.vz,
        arrival_body_id=arrival_body_ids,
        arrival_time=arrival_coordinates.time,
        arrival_body_x=arrival_coordinates.x,
        arrival_body_y=arrival_coordinates.y,
        arrival_body_z=arrival_coordinates.z,
        arrival_body_vx=arrival_coordinates.vx,
        arrival_body_vy=arrival_coordinates.vy,
        arrival_body_vz=arrival_coordinates.vz,
        solution_departure_vx=v1[:, 0],
        solution_departure_vy=v1[:, 1],
        solution_departure_vz=v1[:, 2],
        solution_arrival_vx=v2[:, 0],
        solution_arrival_vy=v2[:, 1],
        solution_arrival_vz=v2[:, 2],
        frame=departure_coordinates.frame,
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
) -> Orbits:
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
    Orbits
        The propagated orbits over the specified time range.
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

    # get orbits for the body at specified times
    if isinstance(body, Orbits):
        propagator = propagator_class()
        orbits = propagator.propagate_orbits(body, times, max_processes=max_processes)
    else:
        # For major bodies, create an Orbits object with the body's origin code as the orbit_id
        coordinates = get_perturber_state(
            body, times, frame="ecliptic", origin=propagation_origin
        )
        # Create orbit IDs based on the body name and time index
        orbit_ids = np.repeat(body.name, len(coordinates))
        orbits = Orbits.from_kwargs(
            orbit_id=orbit_ids,
            coordinates=coordinates,
        )

    return orbits


def generate_porkchop_data(
    departure_orbits: Orbits,
    arrival_orbits: Orbits,
    propagation_origin: OriginCodes = OriginCodes.SUN,
    prograde: bool = True,
    max_iter: int = 35,
    tol: float = 1e-10,
    max_processes: Optional[int] = 1,
) -> LambertSolutions:
    """
    Generate data for a porkchop plot by solving Lambert's problem for a grid of
    departure and arrival times.

    Parameters
    ----------
    departure_orbits : Orbits
        The departure orbits.
    arrival_orbits : Orbits
        The arrival orbits.
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

    assert (
        departure_orbits.coordinates.frame == arrival_orbits.coordinates.frame
    ), "Departure and arrival frames must be the same"
    assert len(departure_orbits.coordinates.origin.code.unique()) == 1
    assert len(arrival_orbits.coordinates.origin.code.unique()) == 1

    assert (
        departure_orbits.coordinates.origin.code[0]
        == arrival_orbits.coordinates.origin.code[0]
    ), "Departure and arrival origins must be the same"

    # First let's make sure departure and arrival orbits are time-ordered
    departure_orbits = departure_orbits.sort_by(
        ["coordinates.time.days", "coordinates.time.nanos"]
    )
    arrival_orbits = arrival_orbits.sort_by(
        ["coordinates.time.days", "coordinates.time.nanos"]
    )

    # Get the actual times for comparison
    dep_times_mjd = departure_orbits.coordinates.time.mjd().to_numpy(
        zero_copy_only=False
    )
    arr_times_mjd = arrival_orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False)

    # Create meshgrids of indices and times
    dep_indices, arr_indices = np.meshgrid(
        np.arange(len(departure_orbits)), np.arange(len(arrival_orbits))
    )
    dep_time_grid, arr_time_grid = np.meshgrid(dep_times_mjd, arr_times_mjd)

    # Filter to ensure departure time is before arrival time
    # Use actual time comparison instead of index comparison
    valid_indices = arr_time_grid > dep_time_grid

    # Apply the mask to flatten only valid combinations
    dep_indices_flat = dep_indices[valid_indices].flatten()
    arr_indices_flat = arr_indices[valid_indices].flatten()

    stacked_departure_orbits = departure_orbits.take(dep_indices_flat)
    stacked_arrival_orbits = arrival_orbits.take(arr_indices_flat)

    # If no valid combinations exist, return empty results
    if len(stacked_departure_orbits) == 0:
        return LambertSolutions.empty()

    if max_processes is None:
        max_processes = mp.cpu_count()

    use_ray = initialize_use_ray(max_processes)

    lambert_results = LambertSolutions.empty()
    if use_ray:
        futures = []
        for start, end in _iterate_chunk_indices(
            stacked_departure_orbits, chunk_size=100
        ):
            futures.append(
                lambert_worker_remote.remote(
                    stacked_departure_orbits[start:end],
                    stacked_arrival_orbits[start:end],
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
            stacked_departure_orbits,
            stacked_arrival_orbits,
            propagation_origin,
            prograde,
            max_iter,
            tol,
        )

    return lambert_results


def plot_porkchop_plotly(
    porkchop_data: LambertSolutions,
    width: int = 900,
    height: int = 700,
    c3_departure_min: Optional[float] = None,
    c3_departure_max: Optional[float] = None,
    vinf_arrival_min: Optional[float] = None,
    vinf_arrival_max: Optional[float] = None,
    tof_min: Optional[float] = None,
    tof_max: Optional[float] = None,
    c3_base_colorscale: str = "Reds",
    vinf_base_colorscale: str = "Blues",
    tof_line_color: str = "black",
    xlim_mjd: Optional[Tuple[float, float]] = None,
    ylim_mjd: Optional[Tuple[float, float]] = None,
    title: str = "Porkchop Plot",
    show_optimal: bool = True,
    show_hover: bool = False,
    logo: bool = True,
):
    """
    Plot the porkchop plot from Lambert trajectory data using Plotly.

    Parameters
    ----------
    porkchop_data : LambertOutput
        The porkchop data.
    width : int, optional
        The width of the plot.
    height : int, optional
        The height of the plot.
    c3_departure_min : float, optional
        The minimum C3 departure value.
    c3_departure_max : float, optional
        The maximum C3 departure value.
    vinf_arrival_min : float, optional
        The minimum V∞ arrival value.
    vinf_arrival_max : float, optional
        The maximum V∞ arrival value.
    tof_min : float, optional
        The minimum time of flight value.
    tof_max : float, optional
        The maximum time of flight value.
    c3_base_colorscale : str, optional
        The base colorscale for C3.
    vinf_base_colorscale : str, optional
        The base colorscale for V∞.
    tof_line_color : str, optional
        The color of the time of flight line.
    xlim_mjd : tuple, optional
        The x-axis limits in MJD.
    ylim_mjd : tuple, optional
        The y-axis limits in MJD.
    title : str, optional
        The title of the plot.
    show_optimal : bool, optional
        Whether to show the optimal V∞ point.
    show_hover : bool, optional
        Whether to show the hover information.
    logo : bool, optional
        Whether to show the logo.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object.
    """
    # --- Extract basic raw data ---
    c3_departure_au_d2 = porkchop_data.c3_departure()  # C3 departure in (AU/day)^2
    vinf_arrival_au_day = porkchop_data.vinf_arrival()  # V∞ arrival in AU/day
    time_of_flight_days = porkchop_data.time_of_flight()
    departure_times = porkchop_data.departure_time
    arrival_times = porkchop_data.arrival_time

    # Convert to metric units using unit conversion functions
    c3_departure_km2_s2 = c3_departure_au_d2 * (au_per_day_to_km_per_s(1.0) ** 2)
    vinf_arrival_km_s = au_per_day_to_km_per_s(vinf_arrival_au_day)
    # Define default C3 range if not provided
    if c3_departure_min is None:
        c3_departure_min = 0
    if c3_departure_max is None:
        c3_departure_max = np.max(c3_departure_km2_s2)

    assert c3_departure_max > c3_departure_min, "C3 max must be greater than C3 min"

    c3_step = (c3_departure_max - c3_departure_min) / 10  # 10 levels by default
    assert c3_step < (
        c3_departure_max - c3_departure_min
    ), "C3 step must be less than the C3 range"

    # Define default V∞ range if not provided
    if vinf_arrival_min is None:
        vinf_arrival_min = 0
    if vinf_arrival_max is None:
        vinf_arrival_max = np.max(vinf_arrival_km_s)
    vinf_step = (vinf_arrival_max - vinf_arrival_min) / 10  # 10 levels by default

    if tof_min is None:
        tof_min = 0
    if tof_max is None:
        tof_max = np.max(time_of_flight_days)

    tof_step = max(5, (tof_max - tof_min) / 10)  # 10 levels, minimum step of 5 days
    tof_step = round(tof_step / 5) * 5  # Round to multiple of 5

    # Validate all step sizes are positive
    assert c3_step > 0, f"c3_step must be positive, got {c3_step}"
    assert vinf_step > 0, f"vinf_step must be positive, got {vinf_step}"
    assert tof_step > 0, f"tof_step must be positive, got {tof_step}"

    # Extract raw MJD values for all points
    departure_times_mjd = departure_times.mjd().to_numpy(zero_copy_only=False)
    arrival_times_mjd = arrival_times.mjd().to_numpy(zero_copy_only=False)

    # --- Apply all filtering to the actual data in one place ---
    # We want to keep all solutions that are not NaN and are within the specified ranges of c3, vinf and tof
    data_mask = (
        ~np.isnan(c3_departure_km2_s2)
        & ~np.isnan(vinf_arrival_km_s)  # Also filter out V∞ NaN values
        & (c3_departure_km2_s2 <= c3_departure_max)
        & (c3_departure_km2_s2 >= c3_departure_min)
        & (vinf_arrival_km_s >= vinf_arrival_min)
        & (vinf_arrival_km_s <= vinf_arrival_max)
        & (time_of_flight_days >= tof_min)
        & (time_of_flight_days <= tof_max)
    )

    # Filter all our data arrays using the combined mask
    filtered_departure_mjd = departure_times_mjd[data_mask]
    filtered_arrival_mjd = arrival_times_mjd[data_mask]
    filtered_c3_km2_s2 = c3_departure_km2_s2[data_mask]
    filtered_vinf_km_s = vinf_arrival_km_s[data_mask]
    filtered_tof_days = time_of_flight_days[data_mask]

    # Recalculate min/max and step sizes based on filtered data
    if len(filtered_c3_km2_s2) > 0:
        c3_departure_min_filtered = np.min(filtered_c3_km2_s2)
        c3_departure_max_filtered = np.max(filtered_c3_km2_s2)
        c3_step_filtered = (c3_departure_max_filtered - c3_departure_min_filtered) / 10
        if c3_step_filtered <= 0:
            c3_step_filtered = 1.0  # Fallback for constant data
    else:
        c3_departure_min_filtered = c3_departure_min
        c3_departure_max_filtered = c3_departure_max
        c3_step_filtered = c3_step

    if len(filtered_vinf_km_s) > 0:
        vinf_arrival_min_filtered = np.min(filtered_vinf_km_s)
        vinf_arrival_max_filtered = np.max(filtered_vinf_km_s)
        vinf_step_filtered = (
            vinf_arrival_max_filtered - vinf_arrival_min_filtered
        ) / 10
        if vinf_step_filtered <= 0:
            vinf_step_filtered = 1.0  # Fallback for constant data
    else:
        vinf_arrival_min_filtered = vinf_arrival_min
        vinf_arrival_max_filtered = vinf_arrival_max
        vinf_step_filtered = vinf_step

    if len(filtered_tof_days) > 0:
        tof_min_filtered = np.min(filtered_tof_days)
        tof_max_filtered = np.max(filtered_tof_days)
        tof_step_filtered = max(5, (tof_max_filtered - tof_min_filtered) / 10)
        tof_step_filtered = round(tof_step_filtered / 5) * 5  # Round to multiple of 5
        if tof_step_filtered <= 0:
            tof_step_filtered = 5  # Fallback minimum step
    else:
        tof_min_filtered = tof_min
        tof_max_filtered = tof_max
        tof_step_filtered = tof_step

    # Get unique times from the filtered data - this guarantees all data points have corresponding unique times
    unique_departure_mjd, dep_indices = np.unique(
        filtered_departure_mjd, return_inverse=True
    )
    unique_arrival_mjd, arr_indices = np.unique(
        filtered_arrival_mjd, return_inverse=True
    )
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
    # Create the grid including date combinations that do not have valid Lambert solutions
    grid_departure_mjd, grid_arrival_mjd = np.meshgrid(
        unique_departure_mjd, unique_arrival_mjd
    )

    # Initialize grid arrays with NaN and fill using the filtered data
    # Since we used return_inverse=True, dep_indices and arr_indices are guaranteed to be valid
    grid_c3_departure_km2_s2 = np.full(
        (len(unique_arrival_mjd), len(unique_departure_mjd)), np.nan, dtype=np.float64
    )
    grid_vinf_arrival_km_s = np.full(
        (len(unique_arrival_mjd), len(unique_departure_mjd)), np.nan, dtype=np.float64
    )

    # Fill the grid directly - no validity masking needed since we pre-filtered the data
    grid_c3_departure_km2_s2[arr_indices, dep_indices] = filtered_c3_km2_s2
    grid_vinf_arrival_km_s[arr_indices, dep_indices] = filtered_vinf_km_s
    grid_tof_days = grid_arrival_mjd - grid_departure_mjd

    # --- Use original grids with NaN values for native Plotly handling ---
    grid_c3_for_plot = grid_c3_departure_km2_s2
    grid_vinf_for_plot = grid_vinf_arrival_km_s

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

    # --- Generate custom colorscales with better saturation at minimum values ---
    # Map common Plotly colorscale names to base colors
    colorscale_to_color = {
        "Reds": "red",
        "Blues": "blue",
        "Greens": "green",
        "Oranges": "orange",
        "Purples": "purple",
    }

    # Generate C3 colorscale with full saturation and built-in transparency
    if c3_base_colorscale in colorscale_to_color:
        # Using saturated colorscale with transparency built into the colorscale
        c3_colorscale = generate_saturated_colorscale(
            colorscale_to_color[c3_base_colorscale],
            n_levels=8,
            max_alpha=0.7,  # Maximum opacity for darkest colors
            min_alpha=0.15,  # Minimum opacity for lightest colors
        )
    else:
        c3_colorscale = c3_base_colorscale

    # Generate V∞ colorscale with full saturation and built-in transparency
    if vinf_base_colorscale in colorscale_to_color:
        # Using saturated colorscale with transparency built into the colorscale
        vinf_colorscale = generate_saturated_colorscale(
            colorscale_to_color[vinf_base_colorscale],
            n_levels=8,
            max_alpha=0.7,  # Maximum opacity for darkest colors
            min_alpha=0.15,  # Minimum opacity for lightest colors
        )
    else:
        vinf_colorscale = vinf_base_colorscale

    # --- Create hover information grids if requested ---
    hover_info = "none"
    custom_data = None
    hover_template = None

    if show_hover:
        # Create date strings for hover display
        grid_departure_date_strings = np.array(
            [
                [
                    Time(mjd, format="mjd").strftime("%Y-%m-%d")
                    for mjd in unique_departure_mjd
                ]
                for _ in unique_arrival_mjd
            ]
        )
        grid_arrival_date_strings = np.array(
            [
                [
                    Time(mjd, format="mjd").strftime("%Y-%m-%d")
                    for _ in unique_departure_mjd
                ]
                for mjd in unique_arrival_mjd
            ]
        )

        # Stack all the data we want in hover info
        # Shape: (n_arrival, n_departure, 5) for [c3, vinf, tof, dep_date, arr_date]
        custom_data = np.stack(
            [
                grid_c3_departure_km2_s2,  # C3 in km²/s²
                grid_vinf_arrival_km_s,  # V∞ in km/s
                grid_tof_days,  # ToF in days
                grid_departure_date_strings,  # Departure date strings
                grid_arrival_date_strings,  # Arrival date strings
            ],
            axis=-1,
        )

        hover_info = "text"
        hover_template = (
            "Departure: %{customdata[3]}<br>"
            "Arrival: %{customdata[4]}<br>"
            "Time of Flight: %{customdata[2]:.1f} days<br>"
            "C3 Departure: %{customdata[0]:.2f} km²/s²<br>"
            "V∞ Arrival: %{customdata[1]:.2f} km/s<br>"
            "<extra></extra>"
        )

    # --- Create Dual Contour Traces ---
    plotly_traces = []

    # C3 Departure Contour Trace (warm colorscale with built-in transparency)
    plotly_traces.append(
        go.Contour(
            x=unique_departure_dates_dt,
            y=unique_arrival_dates_dt,
            z=grid_c3_for_plot,
            zauto=False,
            zmin=c3_departure_min_filtered,
            zmax=c3_departure_max_filtered,
            colorscale=c3_colorscale,
            opacity=1.0,  # Use full opacity since transparency is built into colorscale
            hoverinfo=hover_info,
            hovertemplate=hover_template,
            customdata=custom_data,
            contours=dict(
                coloring="fill",
                showlabels=True,
                labelfont=dict(size=10, color="darkred"),
                start=c3_departure_min_filtered,
                end=c3_departure_max_filtered,
                size=c3_step_filtered,
                labelformat=".1f",
            ),
            ncontours=10,  # Ensure exactly 10 contour levels
            line=dict(width=1.0, smoothing=1.3),
            name="C3 Departure",
            showscale=False,  # Remove colorbar from main trace
            connectgaps=False,  # Don't connect across gaps to match V∞ behavior
            visible=True,
            showlegend=True,
        )
    )

    # V∞ Arrival Contour Trace (cool colorscale with built-in transparency)
    plotly_traces.append(
        go.Contour(
            x=unique_departure_dates_dt,
            y=unique_arrival_dates_dt,
            z=grid_vinf_for_plot,
            zauto=False,
            zmin=vinf_arrival_min_filtered,
            zmax=vinf_arrival_max_filtered,
            colorscale=vinf_colorscale,
            opacity=1.0,  # Use full opacity since transparency is built into colorscale
            hoverinfo=hover_info,
            hovertemplate=hover_template,
            customdata=custom_data,
            contours=dict(
                coloring="fill",
                showlabels=True,
                labelfont=dict(size=10, color="darkblue"),
                start=vinf_arrival_min_filtered,
                end=vinf_arrival_max_filtered,
                size=vinf_step_filtered,
                labelformat=".1f",
            ),
            ncontours=10,  # Ensure exactly 10 contour levels
            line=dict(width=1.0, smoothing=1.3),
            name="V∞ Arrival",
            showscale=False,  # Remove colorbar from main trace
            connectgaps=False,  # Faster rendering by not connecting across gaps
            visible="legendonly",
            showlegend=True,
        )
    )
    # --- ToF Contours ---
    plotly_traces.append(
        go.Contour(
            x=unique_departure_dates_dt,
            y=unique_arrival_dates_dt,
            z=grid_tof_days,  # Original ToF grid with NaNs
            colorscale=[[0, tof_line_color], [1, tof_line_color]],
            contours=dict(
                coloring="lines",
                showlabels=True,
                labelfont=dict(size=10, color=tof_line_color),
                start=tof_min_filtered,
                end=tof_max_filtered,
                size=tof_step_filtered,
            ),
            line=dict(color=tof_line_color, width=1, dash="longdash"),
            name="ToF (days)",
            showscale=False,
            hoverinfo="skip",  # Skip hover for ToF contours
            connectgaps=False,  # Don't connect across NaN gaps
            visible=True,
        )
    )

    # --- Optimal Points (separate for C3 and V∞) ---
    if show_optimal:
        # Optimal C3 Departure Point from filtered data
        if len(filtered_c3_km2_s2) > 0:
            min_c3_filtered_idx = np.nanargmin(filtered_c3_km2_s2)

            # Get the corresponding departure and arrival times from filtered data
            best_c3_dep_mjd = filtered_departure_mjd[min_c3_filtered_idx]
            best_c3_arr_mjd = filtered_arrival_mjd[min_c3_filtered_idx]

            # Convert to datetime objects for plotting
            best_c3_dep_dt = Time(best_c3_dep_mjd, format="mjd").datetime
            best_c3_arr_dt = Time(best_c3_arr_mjd, format="mjd").datetime

            # Check if the optimal C3 point falls within our current plot range
            c3_optimal_in_range = (
                xlim_dt[0] <= best_c3_dep_dt <= xlim_dt[1]
                and ylim_dt[0] <= best_c3_arr_dt <= ylim_dt[1]
            )

            if c3_optimal_in_range:
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
                        hoverinfo="skip",  # Skip hover for optimal points
                    )
                )

        # Optimal V∞ Arrival Point from filtered data
        if len(filtered_vinf_km_s) > 0:
            min_vinf_filtered_idx = np.nanargmin(filtered_vinf_km_s)

            # Get the corresponding departure and arrival times from filtered data
            best_vinf_dep_mjd = filtered_departure_mjd[min_vinf_filtered_idx]
            best_vinf_arr_mjd = filtered_arrival_mjd[min_vinf_filtered_idx]

            # Convert to datetime objects for plotting
            best_vinf_dep_dt = Time(best_vinf_dep_mjd, format="mjd").datetime
            best_vinf_arr_dt = Time(best_vinf_arr_mjd, format="mjd").datetime

            # Check if the optimal V∞ point falls within our current plot range
            vinf_optimal_in_range = (
                xlim_dt[0] <= best_vinf_dep_dt <= xlim_dt[1]
                and ylim_dt[0] <= best_vinf_arr_dt <= ylim_dt[1]
            )

            if vinf_optimal_in_range:
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
                        hoverinfo="skip",  # Skip hover for optimal points
                    )
                )

    # --- Figure Creation and Layout Update ---
    fig = go.Figure(data=plotly_traces)

    if logo:
        images = [
            dict(
                source=get_logo_base64(AsteroidInstituteLogoLight),
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.02,
                sizex=0.12,
                sizey=0.12,
                xanchor="right",
                yanchor="bottom",
            )
        ]
    else:
        images = []

    fig.update_layout(
        title_text=title,
        xaxis_title="Departure Date",
        yaxis_title="Arrival Date",
        xaxis=dict(
            tickformat="%Y-%m-%d",
            tickangle=-45,
            range=xlim_dt,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
        ),
        yaxis=dict(
            tickformat="%Y-%m-%d",
            range=ylim_dt,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
        ),
        plot_bgcolor="white",
        width=width,
        height=height,
        autosize=False,
        hovermode="closest",
        images=images,
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
            borderwidth=1,
        ),
    )

    return fig
