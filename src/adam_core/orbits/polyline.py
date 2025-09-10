"""
Orbit polyline sampling and segment representation for geometric overlap queries.

This module provides functionality to sample orbital ellipses as polylines with
adaptive resolution based on chord length constraints, compute segment AABBs
with guard band padding, and represent the results in quivr tables.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.keplerian import KeplerianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.transform import transform_coordinates
from ..orbits.orbits import Orbits
from ..time import Timestamp

__all__ = [
    "OrbitsPlaneParams",
    "OrbitPolylineSegments",
    "sample_ellipse_adaptive",
    "compute_segment_aabbs",
]

logger = logging.getLogger(__name__)


class OrbitsPlaneParams(qv.Table):
    """
    Orbital plane parameters and basis vectors for each orbit.

    This table stores the fundamental geometric parameters needed to
    reconstruct the orbital ellipse and perform coordinate transformations
    between the orbital plane and the SSB frame.
    """

    #: Unique identifier for the orbit
    orbit_id = qv.LargeStringColumn()

    #: Epoch time for the orbital elements
    t0 = Timestamp.as_column()

    #: Orbital plane basis vector p (towards perihelion) - x component
    p_x = qv.Float64Column()
    #: Orbital plane basis vector p (towards perihelion) - y component
    p_y = qv.Float64Column()
    #: Orbital plane basis vector p (towards perihelion) - z component
    p_z = qv.Float64Column()

    #: Orbital plane basis vector q (in-plane, perpendicular to p) - x component
    q_x = qv.Float64Column()
    #: Orbital plane basis vector q (in-plane, perpendicular to p) - y component
    q_y = qv.Float64Column()
    #: Orbital plane basis vector q (in-plane, perpendicular to p) - z component
    q_z = qv.Float64Column()

    #: Orbital plane normal vector n - x component
    n_x = qv.Float64Column()
    #: Orbital plane normal vector n - y component
    n_y = qv.Float64Column()
    #: Orbital plane normal vector n - z component
    n_z = qv.Float64Column()

    #: Ellipse center vector (from focus to center) - x component in AU
    r0_x = qv.Float64Column()
    #: Ellipse center vector (from focus to center) - y component in AU
    r0_y = qv.Float64Column()
    #: Ellipse center vector (from focus to center) - z component in AU
    r0_z = qv.Float64Column()

    #: Semi-major axis in AU
    a = qv.Float64Column()
    #: Eccentricity
    e = qv.Float64Column()
    #: Mean anomaly at epoch in radians
    M0 = qv.Float64Column()

    #: Coordinate frame
    frame = qv.StringAttribute(default="ecliptic")
    #: Origin of coordinate system
    origin = Origin.as_column()


class OrbitPolylineSegments(qv.Table):
    """
    Polyline segments representing orbital ellipses with AABBs for BVH construction.

    Each row represents a line segment connecting two consecutive points on an
    orbital ellipse, with precomputed axis-aligned bounding boxes for efficient
    geometric queries.
    """

    #: Unique identifier for the orbit
    orbit_id = qv.LargeStringColumn()

    #: Segment identifier within the orbit (0-based)
    seg_id = qv.Int32Column()

    #: Segment start point - x component in AU (SSB frame)
    x0 = qv.Float64Column()
    #: Segment start point - y component in AU (SSB frame)
    y0 = qv.Float64Column()
    #: Segment start point - z component in AU (SSB frame)
    z0 = qv.Float64Column()

    #: Segment end point - x component in AU (SSB frame)
    x1 = qv.Float64Column()
    #: Segment end point - y component in AU (SSB frame)
    y1 = qv.Float64Column()
    #: Segment end point - z component in AU (SSB frame)
    z1 = qv.Float64Column()

    #: AABB minimum bound - x component in AU
    aabb_min_x = qv.Float64Column()
    #: AABB minimum bound - y component in AU
    aabb_min_y = qv.Float64Column()
    #: AABB minimum bound - z component in AU
    aabb_min_z = qv.Float64Column()

    #: AABB maximum bound - x component in AU
    aabb_max_x = qv.Float64Column()
    #: AABB maximum bound - y component in AU
    aabb_max_y = qv.Float64Column()
    #: AABB maximum bound - z component in AU
    aabb_max_z = qv.Float64Column()

    #: Heliocentric distance at segment midpoint in AU
    r_mid_au = qv.Float64Column()

    #: Orbital plane normal vector - x component (duplicated for convenience)
    n_x = qv.Float64Column()
    #: Orbital plane normal vector - y component (duplicated for convenience)
    n_y = qv.Float64Column()
    #: Orbital plane normal vector - z component (duplicated for convenience)
    n_z = qv.Float64Column()


def _compute_orbital_basis(
    kep_coords: KeplerianCoordinates,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute orthonormal orbital plane basis vectors from Keplerian elements.

    Parameters
    ----------
    kep_coords : KeplerianCoordinates
        Keplerian orbital elements

    Returns
    -------
    p_hat : np.ndarray (N, 3)
        Unit vector towards perihelion in SSB frame
    q_hat : np.ndarray (N, 3)
        Unit vector perpendicular to p in orbital plane
    n_hat : np.ndarray (N, 3)
        Unit vector normal to orbital plane (angular momentum direction)
    """
    # Extract angles in radians
    i = np.radians(kep_coords.i.to_numpy())  # inclination
    raan = np.radians(kep_coords.raan.to_numpy())  # right ascension of ascending node
    ap_raw = kep_coords.ap.to_numpy()  # argument of periapsis

    # Handle NaN argument of periapsis (occurs for circular orbits)
    ap = np.where(np.isnan(ap_raw), 0.0, np.radians(ap_raw))

    # Compute rotation matrices
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_ap = np.cos(ap)
    sin_ap = np.sin(ap)

    # Orbital plane basis in SSB ecliptic frame
    # p_hat points towards perihelion
    p_hat = np.column_stack(
        [
            cos_raan * cos_ap - sin_raan * sin_ap * cos_i,
            sin_raan * cos_ap + cos_raan * sin_ap * cos_i,
            sin_ap * sin_i,
        ]
    )

    # q_hat is perpendicular to p_hat in the orbital plane
    q_hat = np.column_stack(
        [
            -cos_raan * sin_ap - sin_raan * cos_ap * cos_i,
            -sin_raan * sin_ap + cos_raan * cos_ap * cos_i,
            cos_ap * sin_i,
        ]
    )

    # n_hat is the normal to the orbital plane (angular momentum direction)
    n_hat = np.column_stack([sin_raan * sin_i, -cos_raan * sin_i, cos_i])

    # Ensure orthonormality within tolerance
    tolerance = 1e-12
    for i in range(len(kep_coords)):
        # Check unit lengths
        p_norm = np.linalg.norm(p_hat[i])
        q_norm = np.linalg.norm(q_hat[i])
        n_norm = np.linalg.norm(n_hat[i])

        if (
            abs(p_norm - 1.0) > tolerance
            or abs(q_norm - 1.0) > tolerance
            or abs(n_norm - 1.0) > tolerance
        ):
            logger.warning(
                f"Basis vectors not unit length for orbit {i}: |p|={p_norm:.2e}, |q|={q_norm:.2e}, |n|={n_norm:.2e}"
            )
            # Renormalize
            p_hat[i] /= p_norm
            q_hat[i] /= q_norm
            n_hat[i] /= n_norm

        # Check orthogonality
        pq_dot = np.dot(p_hat[i], q_hat[i])
        pn_dot = np.dot(p_hat[i], n_hat[i])
        qn_dot = np.dot(q_hat[i], n_hat[i])

        if (
            abs(pq_dot) > tolerance
            or abs(pn_dot) > tolerance
            or abs(qn_dot) > tolerance
        ):
            logger.warning(
                f"Basis vectors not orthogonal for orbit {i}: p·q={pq_dot:.2e}, p·n={pn_dot:.2e}, q·n={qn_dot:.2e}"
            )
            # Re-orthogonalize using Gram-Schmidt
            q_hat[i] = q_hat[i] - np.dot(q_hat[i], p_hat[i]) * p_hat[i]
            q_hat[i] /= np.linalg.norm(q_hat[i])
            n_hat[i] = np.cross(p_hat[i], q_hat[i])

    return p_hat, q_hat, n_hat


def _sample_ellipse_points(
    a: npt.NDArray[np.float64],
    e: npt.NDArray[np.float64],
    p_hat: npt.NDArray[np.float64],
    q_hat: npt.NDArray[np.float64],
    true_anomalies: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute 3D positions on orbital ellipses for given true anomalies.

    Parameters
    ----------
    a : np.ndarray (N,)
        Semi-major axes in AU
    e : np.ndarray (N,)
        Eccentricities
    p_hat : np.ndarray (N, 3)
        Unit vectors towards perihelion
    q_hat : np.ndarray (N, 3)
        Unit vectors perpendicular to p in orbital plane
    true_anomalies : np.ndarray (N, M)
        True anomalies in radians for each orbit

    Returns
    -------
    positions : np.ndarray (N, M, 3)
        3D positions in SSB frame
    """
    # Semi-latus rectum
    p = a * (1 - e**2)

    # Expand dimensions for broadcasting
    a_exp = a[:, np.newaxis]
    e_exp = e[:, np.newaxis]
    p_exp = p[:, np.newaxis]
    p_hat_exp = p_hat[:, np.newaxis, :]
    q_hat_exp = q_hat[:, np.newaxis, :]

    # Radial distance
    cos_f = np.cos(true_anomalies)
    sin_f = np.sin(true_anomalies)
    r = p_exp / (1 + e_exp * cos_f)

    # Position in orbital plane
    x_orb = r * cos_f
    y_orb = r * sin_f

    # Transform to SSB frame
    positions = x_orb[..., np.newaxis] * p_hat_exp + y_orb[..., np.newaxis] * q_hat_exp

    return positions


def sample_ellipse_adaptive(
    orbits: Orbits,
    max_chord_arcmin: float = 0.3,
    max_segments_per_orbit: int = 65536,
) -> Tuple[OrbitsPlaneParams, OrbitPolylineSegments]:
    """
    Sample orbital ellipses adaptively based on chord length constraints.

    This function converts orbits to SSB ecliptic frame, computes orbital plane
    basis vectors, and samples each ellipse with adaptive resolution to ensure
    no segment's chord length exceeds the specified angular constraint.

    Parameters
    ----------
    orbits : Orbits
        Input orbits to sample
    max_chord_arcmin : float, default=0.3
        Maximum allowed chord length in arcminutes as seen from 1 AU
    max_segments_per_orbit : int, default=8192
        Maximum number of segments per orbit to prevent explosion

    Returns
    -------
    plane_params : OrbitsPlaneParams
        Orbital plane parameters and basis vectors
    segments : OrbitPolylineSegments
        Polyline segments with endpoints but no AABBs (use compute_segment_aabbs)
    """
    if len(orbits) == 0:
        return (OrbitsPlaneParams.empty(), OrbitPolylineSegments.empty())

    # Ensure orbits are in SSB ecliptic frame
    coords = orbits.coordinates
    origin_codes = coords.origin.code.to_pylist()
    expected_codes = [OriginCodes.SUN.name] * len(coords.origin)
    if coords.frame != "ecliptic" or origin_codes != expected_codes:
        logger.info("Transforming orbits to SSB ecliptic frame")
        coords = transform_coordinates(
            coords,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        )

    # Convert to Keplerian elements
    kep_coords = coords.to_keplerian()

    # Compute orbital plane basis vectors
    p_hat, q_hat, n_hat = _compute_orbital_basis(kep_coords)

    # Extract orbital elements (angles in radians)
    a = kep_coords.a.to_numpy()
    e = kep_coords.e.to_numpy()
    M0 = np.radians(kep_coords.M.to_numpy())
    orbit_ids = orbits.orbit_id.to_pylist()
    times = coords.time

    # Ellipse center vectors (from focus to geometric center)
    r0 = -a[:, np.newaxis] * e[:, np.newaxis] * p_hat

    # Convert max chord constraint to radians
    theta_max = max_chord_arcmin * np.pi / (180 * 60)

    # Collect all segments across orbits
    all_orbit_ids = []
    all_seg_ids = []
    all_x0, all_y0, all_z0 = [], [], []
    all_x1, all_y1, all_z1 = [], [], []
    all_r_mid = []
    all_n_x, all_n_y, all_n_z = [], [], []

    for i, orbit_id in enumerate(orbit_ids):
        # Start with uniform sampling
        n_initial = 1024
        f_samples = np.linspace(0, 2 * np.pi, n_initial + 1)[
            :-1
        ]  # Exclude 2π (same as 0)

        # Iteratively refine segments that violate chord constraint
        segments_to_check = [
            (f_samples[j], f_samples[(j + 1) % len(f_samples)])
            for j in range(len(f_samples))
        ]
        final_f_values = set(f_samples)

        iteration = 0
        max_iterations = 10

        while (
            segments_to_check
            and len(final_f_values) < max_segments_per_orbit
            and iteration < max_iterations
        ):
            new_segments = []

            for f0, f1 in segments_to_check:
                # Compute positions at segment endpoints
                pos0 = _sample_ellipse_points(
                    np.array([a[i]]),
                    np.array([e[i]]),
                    p_hat[i : i + 1],
                    q_hat[i : i + 1],
                    np.array([[f0]]),
                )[0, 0]

                pos1 = _sample_ellipse_points(
                    np.array([a[i]]),
                    np.array([e[i]]),
                    p_hat[i : i + 1],
                    q_hat[i : i + 1],
                    np.array([[f1]]),
                )[0, 0]

                # Chord length and midpoint distance
                chord_length = np.linalg.norm(pos1 - pos0)
                r_mid = np.linalg.norm((pos0 + pos1) / 2)

                # Maximum allowed chord length at this distance
                max_chord_au = theta_max * max(r_mid, 1.0)

                if chord_length > max_chord_au:
                    # Split this segment
                    f_mid = (f0 + f1) / 2
                    if f1 < f0:  # Handle wrap-around
                        f_mid = (f0 + f1 + 2 * np.pi) / 2
                        if f_mid > 2 * np.pi:
                            f_mid -= 2 * np.pi

                    final_f_values.add(f_mid)
                    new_segments.append((f0, f_mid))
                    new_segments.append((f_mid, f1))

            segments_to_check = new_segments
            iteration += 1

        if len(final_f_values) >= max_segments_per_orbit:
            logger.warning(
                f"Orbit {orbit_id} hit segment limit {max_segments_per_orbit}"
            )

        # Sort final anomaly values
        f_final = sorted(final_f_values)

        # Compute final positions
        positions = _sample_ellipse_points(
            np.array([a[i]]),
            np.array([e[i]]),
            p_hat[i : i + 1],
            q_hat[i : i + 1],
            np.array([f_final]),
        )[
            0
        ]  # Shape: (n_points, 3)

        # Create segments
        for j in range(len(f_final)):
            j_next = (j + 1) % len(f_final)

            pos0 = positions[j]
            pos1 = positions[j_next]
            r_mid = np.linalg.norm((pos0 + pos1) / 2)

            all_orbit_ids.append(orbit_id)
            all_seg_ids.append(j)
            all_x0.append(pos0[0])
            all_y0.append(pos0[1])
            all_z0.append(pos0[2])
            all_x1.append(pos1[0])
            all_y1.append(pos1[1])
            all_z1.append(pos1[2])
            all_r_mid.append(r_mid)
            all_n_x.append(n_hat[i, 0])
            all_n_y.append(n_hat[i, 1])
            all_n_z.append(n_hat[i, 2])

    # Create plane parameters table
    plane_params = OrbitsPlaneParams.from_kwargs(
        orbit_id=orbit_ids,
        t0=times,
        p_x=p_hat[:, 0],
        p_y=p_hat[:, 1],
        p_z=p_hat[:, 2],
        q_x=q_hat[:, 0],
        q_y=q_hat[:, 1],
        q_z=q_hat[:, 2],
        n_x=n_hat[:, 0],
        n_y=n_hat[:, 1],
        n_z=n_hat[:, 2],
        r0_x=r0[:, 0],
        r0_y=r0[:, 1],
        r0_z=r0[:, 2],
        a=a,
        e=e,
        M0=M0,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * len(orbits)),
    )

    # Create segments table (AABBs will be filled by compute_segment_aabbs)
    segments = OrbitPolylineSegments.from_kwargs(
        orbit_id=all_orbit_ids,
        seg_id=all_seg_ids,
        x0=all_x0,
        y0=all_y0,
        z0=all_z0,
        x1=all_x1,
        y1=all_y1,
        z1=all_z1,
        aabb_min_x=[np.nan] * len(all_orbit_ids),  # To be filled
        aabb_min_y=[np.nan] * len(all_orbit_ids),
        aabb_min_z=[np.nan] * len(all_orbit_ids),
        aabb_max_x=[np.nan] * len(all_orbit_ids),
        aabb_max_y=[np.nan] * len(all_orbit_ids),
        aabb_max_z=[np.nan] * len(all_orbit_ids),
        r_mid_au=all_r_mid,
        n_x=all_n_x,
        n_y=all_n_y,
        n_z=all_n_z,
    )

    logger.info(f"Sampled {len(orbits)} orbits into {len(segments)} segments")

    return plane_params, segments


def compute_segment_aabbs(
    segments: OrbitPolylineSegments,
    guard_arcmin: float = 1.0,
    epsilon_n_au: float = 1e-6,
) -> OrbitPolylineSegments:
    """
    Compute axis-aligned bounding boxes for orbit segments with guard band padding.

    Parameters
    ----------
    segments : OrbitPolylineSegments
        Input segments with endpoints but potentially missing AABBs
    guard_arcmin : float, default=1.0
        Guard band in arcminutes for in-plane padding
    epsilon_n_au : float, default=1e-6
        Small padding along orbital plane normal in AU

    Returns
    -------
    segments_with_aabbs : OrbitPolylineSegments
        Segments with filled AABB columns
    """
    if len(segments) == 0:
        return segments

    # Convert guard band to radians
    theta_guard = guard_arcmin * np.pi / (180 * 60)

    # Extract segment endpoints
    x0 = segments.x0.to_numpy()
    y0 = segments.y0.to_numpy()
    z0 = segments.z0.to_numpy()
    x1 = segments.x1.to_numpy()
    y1 = segments.y1.to_numpy()
    z1 = segments.z1.to_numpy()

    # Segment midpoint distances
    r_mid = segments.r_mid_au.to_numpy()

    # Orbital plane normals
    n_x = segments.n_x.to_numpy()
    n_y = segments.n_y.to_numpy()
    n_z = segments.n_z.to_numpy()

    # Compute unpadded AABBs
    min_x = np.minimum(x0, x1)
    max_x = np.maximum(x0, x1)
    min_y = np.minimum(y0, y1)
    max_y = np.maximum(y0, y1)
    min_z = np.minimum(z0, z1)
    max_z = np.maximum(z0, z1)

    # In-plane padding based on guard band and distance (conservative: at least 1 AU)
    pad_in_plane = theta_guard * np.maximum(r_mid, 1.0)

    # Apply in-plane padding (conservative - pad all directions)
    min_x -= pad_in_plane
    max_x += pad_in_plane
    min_y -= pad_in_plane
    max_y += pad_in_plane
    min_z -= pad_in_plane
    max_z += pad_in_plane

    # Additional small padding along orbital plane normal
    min_x -= epsilon_n_au * np.abs(n_x)
    max_x += epsilon_n_au * np.abs(n_x)
    min_y -= epsilon_n_au * np.abs(n_y)
    max_y += epsilon_n_au * np.abs(n_y)
    min_z -= epsilon_n_au * np.abs(n_z)
    max_z += epsilon_n_au * np.abs(n_z)

    # Create new segments table with filled AABBs
    segments_with_aabbs = OrbitPolylineSegments.from_kwargs(
        orbit_id=segments.orbit_id.to_pylist(),
        seg_id=segments.seg_id.to_pylist(),
        x0=segments.x0.to_pylist(),
        y0=segments.y0.to_pylist(),
        z0=segments.z0.to_pylist(),
        x1=segments.x1.to_pylist(),
        y1=segments.y1.to_pylist(),
        z1=segments.z1.to_pylist(),
        aabb_min_x=min_x.tolist(),
        aabb_min_y=min_y.tolist(),
        aabb_min_z=min_z.tolist(),
        aabb_max_x=max_x.tolist(),
        aabb_max_y=max_y.tolist(),
        aabb_max_z=max_z.tolist(),
        r_mid_au=segments.r_mid_au.to_pylist(),
        n_x=segments.n_x.to_pylist(),
        n_y=segments.n_y.to_pylist(),
        n_z=segments.n_z.to_pylist(),
    )

    return segments_with_aabbs
