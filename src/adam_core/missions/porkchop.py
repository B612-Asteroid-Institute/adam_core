from typing import Optional, Tuple, Union

import numpy as np
import quivr as qv
import ray

from adam_core.coordinates import CartesianCoordinates, transform_coordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.dynamics.lambert import solve_lambert
from adam_core.orbits import Orbit
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils import get_perturber_state


class LambertOutput(qv.Table):
    departure_time = Timestamp.as_column()
    arrival_time = Timestamp.as_column()
    vx_1 = qv.Float64Column()
    vy_1 = qv.Float64Column()
    vz_1 = qv.Float64Column()
    vx_2 = qv.Float64Column()
    vy_2 = qv.Float64Column()
    vz_2 = qv.Float64Column()
    origin = Origin.as_column()

    def c3(self) -> float:
        pass


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
    tof = arrival_coordinates.timestamp.mjd().to_numpy(
        zero_copy_only=False
    ) - departure_coordinates.timestamp.mjd().to_numpy(zero_copy_only=False)

    origins = Origin.from_OriginCodes(propagation_origin, size=len(r1))
    mu = origins.mu()[0]
    v1, v2 = solve_lambert(r1, r2, tof, mu, prograde, max_iter, tol)

    return LambertOutput.from_kwargs(
        departure_time=departure_coordinates.timestamp,
        arrival_time=arrival_coordinates.timestamp,
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
    departure_body: Union[Orbit, OriginCodes],
    arrival_body: Union[Orbit, OriginCodes],
    earliest_launch_time: Timestamp,
    maximum_arrival_time: Timestamp,
    propagation_origin: OriginCodes = OriginCodes.SUN,
    step_size: float = 0.1,
    prograde: bool = True,
    max_iter: int = 35,
    tol: float = 1e-10,
    propagator_class: Optional[type[Propagator]] = None,
    max_processes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # if departure_body is an Orbit, ensure its origin is the propagation_origin
    if isinstance(departure_body, Orbit):
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
    if isinstance(arrival_body, Orbit):
        arrival_body = arrival_body.set_column(
            "coordinates",
            transform_coordinates(
                arrival_body.coordinates,
                representation_out=CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=propagation_origin,
            ),
        )

    # pre-generate the state vectors for the departure and arrival bodies and the specified time grid

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

    # get r1 (departure body) for times
    if isinstance(departure_body, Orbit):
        propagator = propagator_class()
        departure_coordinates = propagator.propagate(
            departure_body, times, max_processes=max_processes
        ).coordinates
    else:
        departure_coordinates = get_perturber_state(
            departure_body, times, frame="ecliptic", origin=propagation_origin
        )

    # get r2 (arrival body) for times
    if isinstance(arrival_body, Orbit):
        propagator = propagator_class()
        arrival_coordinates = propagator.propagate(
            arrival_body, times, max_processes=max_processes
        ).coordinates
    else:
        arrival_coordinates = get_perturber_state(
            arrival_body, times, frame="ecliptic", origin=propagation_origin
        )

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

    use_ray = initialize_use_ray(max_processes)

    if use_ray:
        futures = []
        for i in range(len(stacked_departure_coordinates)):
            # TODO: actually queue up the work