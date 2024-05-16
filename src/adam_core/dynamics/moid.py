"""
Implementation of SGD-MOID

References
----------
[1] Hedo, J. M. et al. (2019). Minimum orbital intersection distance: an asymptotic approach.
    Astronomy & Astrophysics, 633, A22. https://doi.org/10.1051/0004-6361/201936502

Algorithm 1:
From an arbitrary point P_0 on the primary ellipse,
project the point onto the plane of the other ellipse.

Then find the minimum distance from the projected point,
using a minimization function.

Algorithm 2:

1. Divide up the primary ellipse into a series of points
2. For each point, run algorithm 1
3. After finding the point with the lowest distance,
search for the overall minimum.
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from ..coordinates import CartesianCoordinates, KeplerianCoordinates
from ..dynamics.propagation import _propagate_2body
from ..orbits import Orbits
from ..time import Timestamp


def project_point_on_plane(
    P0: npt.NDArray, plane_coordinates: CartesianCoordinates
) -> npt.NDArray:
    """
    Take a point P0 (from the secondary ellipse) and project it onto the plane of the primary ellipse.
    """
    assert len(plane_coordinates) == 1

    n_hat = plane_coordinates.h[0] / plane_coordinates.h_mag[0]
    projected_point = P0 - np.dot(P0, n_hat) * n_hat
    return projected_point


def coplanar_distance_to_ellipse(
    P: npt.NDArray, keplerian_coordinates: KeplerianCoordinates, u: float
) -> float:
    """
    Calculate the distance from point P on the plane to the ellipse.
    """
    a, e, i, ap, raan, M = keplerian_coordinates.values[0]
    # Calculate the eccentric anomaly
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(u / 2))
    # Calculate the true anomaly
    v = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(u / 2))
    # Calculate the radius
    r = a * (1 - e * np.cos(E))
    # Calculate the position vector
    r_vec = np.array([r * np.cos(v), r * np.sin(v), 0])
    # Calculate the distance from the point to the ellipse
    distance = np.linalg.norm(P - r_vec)
    return distance


def minimize_distance_from_coplanar_point_to_ellipse(
    P: npt.NDArray, keplerian_coordinates: KeplerianCoordinates
):
    """
    Find the minimum distance to the ellipse from planar point P

    We only need the magnitude of the distance here to calculate distance from P0 to closest point on ellipse
    """
    # Minimize the distance function with respect to an angle u
    result = minimize_scalar(
        lambda u: coplanar_distance_to_ellipse(P, keplerian_coordinates, u),
        bounds=(0, 2 * np.pi),
        method="bounded",
        tol=1e-12,
    )
    return result.fun


def distance_from_point_to_ellipse(
    P0: npt.NDArray, P: npt.NDArray, d_parallel: float
) -> float:
    """
    Takes the parallel and perpendicular distances from the point to the ellipse and calculates the total distance
    """
    d_perp = np.linalg.norm(P - P0)
    distance = np.sqrt(d_perp**2 + d_parallel**2)
    return distance


def calculate_distance_from_point_to_ellipse(P0: npt.NDArray, primary_ellipse: Orbits):
    """
    Calculate the distance from point P0 (from first ellipse) to the closest point on the ellipse
    """
    # Project the point onto the plane of the other ellipse
    P = project_point_on_plane(P0, primary_ellipse.coordinates)
    # Find the minimum distance from the projected point to the ellipse
    d_parallel = minimize_distance_from_coplanar_point_to_ellipse(
        P, primary_ellipse.coordinates.to_keplerian()
    )
    # Calculate the distance from P0 to the closest point on the ellipse
    distance = distance_from_point_to_ellipse(P0, P, d_parallel)
    return distance


def calculate_moid_for_dt(
    primary_ellipse: Orbits, secondary_ellipse: Orbits, dt: float
):
    # Propagate the primary ellipse by dt
    t_0 = primary_ellipse.coordinates.time.mjd()[0].as_py()
    primary_ellipse_propagated = _propagate_2body(
        primary_ellipse.coordinates.values[0],
        t_0,
        t_0 + dt,
        primary_ellipse.coordinates.origin.mu()[0],
    )
    P0 = primary_ellipse_propagated[:3]
    distance_to_secondary_ellipse = calculate_distance_from_point_to_ellipse(
        P0, secondary_ellipse
    )
    return distance_to_secondary_ellipse


# Now for Algorithm 1, where we discretize the primary ellipse and run Algorithm 1 for each point
def calculate_moid(
    primary_ellipse: Orbits, secondary_ellipse: Orbits
) -> tuple[float, Timestamp]:
    """
    Calculate the Minimum Orbit Intersection Distance (MOID) between two orbits.
    """
    keplerian = primary_ellipse.coordinates.to_keplerian()
    period = keplerian.P[0]
    e = keplerian.e[0].as_py()
    if e < 1:
        bounds = (0, period)
    else:
        bounds = (0, 10000)

    result = minimize_scalar(
        lambda dt: calculate_moid_for_dt(primary_ellipse, secondary_ellipse, dt),
        bounds=bounds,
        method="bounded",
        tol=1e-12,
    )
    moid = result.fun
    moid_time = Timestamp.from_mjd(
        [primary_ellipse.coordinates.time.mjd()[0].as_py() + result.x], scale="tdb"
    )
    if result.status != 0:
        raise ValueError("MOID calculation did not converge.")

    return moid, moid_time
