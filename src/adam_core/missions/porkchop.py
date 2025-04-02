import multiprocessing as mp
import time
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import quivr as qv
import ray

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
        departure_coordinates = propagator.propagate(
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
        arrival_coordinates = propagator.propagate(
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
    print(f"  - Coordinate preparation: {prep_time:.3f}s ({prep_time/total_time*100:.1f}%)")
    print(f"  - Time grid generation: {time_gen_time:.3f}s ({time_gen_time/total_time*100:.1f}%)")
    print(f"  - Departure propagation: {dep_prop_time:.3f}s ({dep_prop_time/total_time*100:.1f}%)")
    print(f"  - Arrival propagation: {arr_prop_time:.3f}s ({arr_prop_time/total_time*100:.1f}%)")
    print(f"  - Mesh and stacking: {mesh_time:.3f}s ({mesh_time/total_time*100:.1f}%)")
    print(f"  - Ray initialization: {ray_init_time:.3f}s ({ray_init_time/total_time*100:.1f}%)")
    print(f"  - Lambert solution: {lambert_time:.3f}s ({lambert_time/total_time*100:.1f}%)")

    return lambert_results
