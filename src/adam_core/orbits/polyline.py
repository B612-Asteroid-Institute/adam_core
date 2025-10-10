"""
Orbit polyline sampling and segment representation for geometric overlap queries.

This module provides functionality to sample orbital ellipses as polylines with
adaptive resolution based on chord length constraints, compute segment AABBs
with guard band padding, and represent the results in quivr tables.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import quivr as qv
import ray

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.keplerian import KeplerianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.transform import transform_coordinates
from ..orbits.orbits import Orbits
from ..ray_cluster import initialize_use_ray
from ..time import Timestamp
from ..utils.iter import _iterate_chunk_indices

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

    # Provenance: sampling parameters
    sample_max_chord_arcmin = qv.FloatAttribute(default=0.0)
    sample_max_segments_per_orbit = qv.IntAttribute(default=0)


class OrbitPolylineSegments(qv.Table):
    """
    Polyline segments representing orbital ellipses used for BVH construction.

    Each row represents a line segment connecting two consecutive points on an
    orbital ellipse. AABBs are no longer stored on the segments; they are
    computed in-memory during BVH construction and recorded on BVH nodes.
    """

    #: Unique identifier for the orbit
    orbit_id = qv.LargeStringColumn()

    #: Segment identifier within the orbit (0-based)
    seg_id = qv.Int32Column()

    #: Segment start point - x component in AU (SSB frame) [float32]
    x0 = qv.Float32Column()
    #: Segment start point - y component in AU (SSB frame) [float32]
    y0 = qv.Float32Column()
    #: Segment start point - z component in AU (SSB frame) [float32]
    z0 = qv.Float32Column()

    #: Segment end point - x component in AU (SSB frame) [float32]
    x1 = qv.Float32Column()
    #: Segment end point - y component in AU (SSB frame) [float32]
    y1 = qv.Float32Column()
    #: Segment end point - z component in AU (SSB frame) [float32]
    z1 = qv.Float32Column()

    #: Heliocentric distance at segment midpoint in AU [float32]
    r_mid_au = qv.Float32Column()

    #: Orbital plane normal vector - x component [float32]
    n_x = qv.Float32Column()
    #: Orbital plane normal vector - y component [float32]
    n_y = qv.Float32Column()
    #: Orbital plane normal vector - z component [float32]
    n_z = qv.Float32Column()

    # Provenance: sampling parameters (AABB provenance is stored on BVH nodes)
    sample_max_chord_arcmin = qv.FloatAttribute(default=0.0)
    sample_max_segments_per_orbit = qv.IntAttribute(default=0)


@partial(jax.jit, static_argnames=("use_sagitta_guard",))
def _compute_aabbs_kernel(
    x0: jnp.ndarray,
    y0: jnp.ndarray,
    z0: jnp.ndarray,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    z1: jnp.ndarray,
    r_mid: jnp.ndarray,
    n_x: jnp.ndarray,
    n_y: jnp.ndarray,
    n_z: jnp.ndarray,
    theta_guard: float,
    theta_c: float,
    epsilon_n_au: float,
    *,
    use_sagitta_guard: bool,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    # Type constants to input dtype to avoid float64 promotion
    dtype = x0.dtype
    one = jnp.array(1.0, dtype=dtype)
    eight = jnp.array(8.0, dtype=dtype)

    # Unpadded AABBs
    min_x = jnp.minimum(x0, x1)
    max_x = jnp.maximum(x0, x1)
    min_y = jnp.minimum(y0, y1)
    max_y = jnp.maximum(y0, y1)
    min_z = jnp.minimum(z0, z1)
    max_z = jnp.maximum(z0, z1)

    # In-plane padding
    r_eff = jnp.maximum(r_mid, one)
    pad_guard = jnp.array(theta_guard, dtype=dtype) * r_eff
    theta_c_typed = jnp.array(theta_c, dtype=dtype)
    pad_sagitta = (theta_c_typed * theta_c_typed / eight) * r_eff
    if use_sagitta_guard:
        pad_in_plane = jnp.maximum(pad_guard, pad_sagitta)
    else:
        pad_in_plane = pad_guard

    # Apply in-plane padding
    min_x = min_x - pad_in_plane
    max_x = max_x + pad_in_plane
    min_y = min_y - pad_in_plane
    max_y = max_y + pad_in_plane
    min_z = min_z - pad_in_plane
    max_z = max_z + pad_in_plane

    # Padding along orbital plane normal
    epsilon_typed = jnp.array(epsilon_n_au, dtype=dtype)
    abs_nx = jnp.abs(n_x)
    abs_ny = jnp.abs(n_y)
    abs_nz = jnp.abs(n_z)
    min_x = min_x - epsilon_typed * abs_nx
    max_x = max_x + epsilon_typed * abs_nx
    min_y = min_y - epsilon_typed * abs_ny
    max_y = max_y + epsilon_typed * abs_ny
    min_z = min_z - epsilon_typed * abs_nz
    max_z = max_z + epsilon_typed * abs_nz

    return min_x, min_y, min_z, max_x, max_y, max_z


def _compute_aabbs_chunk(
    segments: OrbitPolylineSegments,
    guard_arcmin: float,
    epsilon_n_au: float,
    *,
    padding_method: "PaddingMethod",
    window_len: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(segments) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    theta_guard = guard_arcmin * np.pi / (180.0 * 60.0)
    sample_chord_arcmin = float(getattr(segments, "sample_max_chord_arcmin", 0.0))
    theta_c = sample_chord_arcmin * np.pi / (180.0 * 60.0)
    use_sagitta_guard = padding_method == "sagitta_guard"

    # Use NumPy inputs; pad to fixed window_len for stable JIT shape if requested
    x0_np = segments.x0.to_numpy()
    y0_np = segments.y0.to_numpy()
    z0_np = segments.z0.to_numpy()
    x1_np = segments.x1.to_numpy()
    y1_np = segments.y1.to_numpy()
    z1_np = segments.z1.to_numpy()
    r_mid_np = segments.r_mid_au.to_numpy()
    n_x_np = segments.n_x.to_numpy()
    n_y_np = segments.n_y.to_numpy()
    n_z_np = segments.n_z.to_numpy()

    orig_len = len(x0_np)
    if window_len is not None and orig_len < window_len:
        pad = window_len - orig_len

        def pad1(a: np.ndarray) -> np.ndarray:
            return np.pad(a, (0, pad))

        def pad3(a: np.ndarray) -> np.ndarray:
            return np.pad(a, (0, pad))

        x0 = pad1(x0_np)
        y0 = pad1(y0_np)
        z0 = pad1(z0_np)
        x1 = pad1(x1_np)
        y1 = pad1(y1_np)
        z1 = pad1(z1_np)
        r_mid = pad1(r_mid_np)
        n_x = pad1(n_x_np)
        n_y = pad1(n_y_np)
        n_z = pad1(n_z_np)
    else:
        x0 = x0_np
        y0 = y0_np
        z0 = z0_np
        x1 = x1_np
        y1 = y1_np
        z1 = z1_np
        r_mid = r_mid_np
        n_x = n_x_np
        n_y = n_y_np
        n_z = n_z_np

    out = _compute_aabbs_kernel(
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        r_mid,
        n_x,
        n_y,
        n_z,
        float(theta_guard),
        float(theta_c),
        float(epsilon_n_au),
        use_sagitta_guard=use_sagitta_guard,
    )
    # Convert to NumPy
    outs = tuple(np.asarray(arr) for arr in out)
    if window_len is not None and orig_len < window_len:
        outs = tuple(o[:orig_len] for o in outs)
    return outs  # type: ignore[return-value]


@ray.remote
def _compute_aabbs_worker_remote(
    start: int,
    segments: OrbitPolylineSegments,
    guard_arcmin: float,
    epsilon_n_au: float,
    *,
    padding_method: "PaddingMethod",
    window_len: int | None = None,
) -> Tuple[
    int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    res = _compute_aabbs_chunk(
        segments,
        guard_arcmin=guard_arcmin,
        epsilon_n_au=epsilon_n_au,
        padding_method=padding_method,
        window_len=window_len,
    )
    return start, res


@partial(jax.jit, static_argnums=(4,))
def _sample_positions_jax(
    a: jnp.ndarray,
    e: jnp.ndarray,
    p_hat: jnp.ndarray,
    q_hat: jnp.ndarray,
    n_points: int,
) -> jnp.ndarray:
    """
    JAX-jitted sampler for orbit positions at uniformly spaced true anomalies.

    Parameters
    ----------
    a : (N,) semi-major axes
    e : (N,) eccentricities
    p_hat : (N,3) unit vectors towards perihelion
    q_hat : (N,3) unit vectors perpendicular to p in the orbital plane
    n_points : int, number of true anomaly samples on [0, 2pi)
    """
    p = a * (1.0 - e * e)
    f = jnp.linspace(0.0, 2.0 * jnp.pi, int(n_points) + 1, dtype=a.dtype)[:-1]
    cos_f = jnp.cos(f)
    sin_f = jnp.sin(f)

    r = p[:, None] / (1.0 + e[:, None] * cos_f[None, :])
    x_orb = r * cos_f[None, :]
    y_orb = r * sin_f[None, :]

    positions = (
        x_orb[..., None] * p_hat[:, None, :] + y_orb[..., None] * q_hat[:, None, :]
    )
    return positions  # (N, n_points, 3)


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
    max_chord_arcmin: float,
    max_segments_per_orbit: int,
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
    max_chord_arcmin : float
        Maximum allowed chord length in arcminutes as seen from 1 AU
    max_segments_per_orbit : int
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

    # Prepare JAX sampling grid (always use JAX path)
    # Grid resolution scales with allowed segments; cap high but allow tight chord limits
    grid_points = int(min(max_segments_per_orbit * 2, 131072))
    grid_points = max(grid_points, 64)

    # JAX will implicitly convert NumPy to device arrays
    positions_jax = _sample_positions_jax(
        a,
        e,
        p_hat,
        q_hat,
        grid_points,
    )
    # Bring back to NumPy for downstream Quivr operations
    positions = np.asarray(positions_jax)

    # Initial selection: coarse uniform subset
    n_initial = min(1024, grid_points)
    stride = max(grid_points // n_initial, 1)
    initial_idx = np.arange(0, grid_points, stride, dtype=np.int32)

    selected = np.zeros((len(orbits), grid_points), dtype=bool)
    selected[:, initial_idx] = True

    # Fixed number of refinement iterations to limit work; validated by tests
    max_iterations = 6
    for _ in range(max_iterations):
        any_change = False
        for i, orbit_id in enumerate(orbit_ids):
            idx = np.flatnonzero(selected[i])
            if len(idx) < 3:
                continue
            to_add = []
            for j in range(len(idx)):
                i0 = idx[j]
                i1 = idx[(j + 1) % len(idx)]
                # steps forward on circular grid
                step = (i1 - i0) % grid_points
                if step == 0:
                    step = grid_points
                # Endpoints
                p0 = positions[i, i0]
                p1 = positions[i, i1]
                chord = float(np.linalg.norm(p1 - p0))
                r_mid = float(np.linalg.norm((p0 + p1) * 0.5))
                max_chord_au = float(theta_max * max(r_mid, 1.0))
                if chord > max_chord_au and step > 1:
                    mid = (i0 + step // 2) % grid_points
                    if not selected[i, mid]:
                        to_add.append(mid)
            if to_add:
                selected[i, np.array(to_add, dtype=np.int32)] = True
                any_change = True
        if not any_change:
            break

    # Collect all segments across orbits
    all_orbit_ids = []
    all_seg_ids = []
    all_x0, all_y0, all_z0 = [], [], []
    all_x1, all_y1, all_z1 = [], [], []
    all_r_mid = []
    all_n_x, all_n_y, all_n_z = [], [], []

    for i, orbit_id in enumerate(orbit_ids):
        idx = np.flatnonzero(selected[i])
        if len(idx) == 0:
            continue
        idx_sorted = np.sort(idx)
        P = len(idx_sorted)
        # Enforce strict cap on segments per orbit by uniformly thinning if needed
        if P > int(max_segments_per_orbit):
            take = int(max_segments_per_orbit)
            # Evenly spaced selection over the circular ring
            sel = (np.linspace(0, P, num=take, endpoint=False)).astype(np.int64)
            idx_sorted = idx_sorted[sel]
            P = take
        for j in range(P):
            i0 = idx_sorted[j]
            i1 = idx_sorted[(j + 1) % P]
            p0 = positions[i, i0]
            p1 = positions[i, i1]
            r_mid = float(np.linalg.norm((p0 + p1) * 0.5))

            all_orbit_ids.append(orbit_id)
            all_seg_ids.append(j)
            all_x0.append(np.float32(p0[0]))
            all_y0.append(np.float32(p0[1]))
            all_z0.append(np.float32(p0[2]))
            all_x1.append(np.float32(p1[0]))
            all_y1.append(np.float32(p1[1]))
            all_z1.append(np.float32(p1[2]))
            all_r_mid.append(np.float32(r_mid))
            all_n_x.append(np.float32(n_hat[i, 0]))
            all_n_y.append(np.float32(n_hat[i, 1]))
            all_n_z.append(np.float32(n_hat[i, 2]))

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
        sample_max_chord_arcmin=float(max_chord_arcmin),
        sample_max_segments_per_orbit=int(max_segments_per_orbit),
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
        r_mid_au=all_r_mid,
        n_x=all_n_x,
        n_y=all_n_y,
        n_z=all_n_z,
        sample_max_chord_arcmin=float(max_chord_arcmin),
        sample_max_segments_per_orbit=int(max_segments_per_orbit),
    )

    logger.info(f"Sampled {len(orbits)} orbits into {len(segments)} segments")

    return plane_params, segments


PaddingMethod = Literal["baseline", "sagitta_guard"]


def compute_segment_aabbs(
    segments: OrbitPolylineSegments,
    guard_arcmin: float,
    epsilon_n_au: float,
    *,
    padding_method: PaddingMethod = "baseline",
    max_processes: int | None = 1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute axis-aligned bounding boxes for orbit segments with guard band padding.

    Uses a JAX JIT kernel for per-element math. When max_processes > 1, shards the
    segments and computes AABBs in parallel with Ray, using backpressure to limit
    outstanding futures.
    """
    if len(segments) == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    if padding_method not in ("baseline", "sagitta_guard"):
        raise ValueError(f"Unknown padding_method: {padding_method}")

    # Fixed window length for stable shapes
    WINDOW_LEN = 8192

    # Single-process path: iterate in fixed windows to keep JAX shapes stable
    if max_processes is None or max_processes <= 1:
        n = len(segments)
        out_min_x = np.empty(n, dtype=np.float32)
        out_min_y = np.empty(n, dtype=np.float32)
        out_min_z = np.empty(n, dtype=np.float32)
        out_max_x = np.empty(n, dtype=np.float32)
        out_max_y = np.empty(n, dtype=np.float32)
        out_max_z = np.empty(n, dtype=np.float32)

        for s in range(0, n, WINDOW_LEN):
            e = min(s + WINDOW_LEN, n)
            rmin_x, rmin_y, rmin_z, rmax_x, rmax_y, rmax_z = _compute_aabbs_chunk(
                segments[s:e],
                guard_arcmin=guard_arcmin,
                epsilon_n_au=epsilon_n_au,
                padding_method=padding_method,
                window_len=WINDOW_LEN,
            )
            out_min_x[s:e] = rmin_x
            out_min_y[s:e] = rmin_y
            out_min_z[s:e] = rmin_z
            out_max_x[s:e] = rmax_x
            out_max_y[s:e] = rmax_y
            out_max_z[s:e] = rmax_z

        return out_min_x, out_min_y, out_min_z, out_max_x, out_max_y, out_max_z

    # Parallel path with Ray
    initialize_use_ray(num_cpus=max_processes)

    n = len(segments)
    # Use fixed-size shards for shape stability
    chunk_size = WINDOW_LEN

    # Pre-allocate outputs
    out_min_x = np.empty(n, dtype=np.float32)
    out_min_y = np.empty(n, dtype=np.float32)
    out_min_z = np.empty(n, dtype=np.float32)
    out_max_x = np.empty(n, dtype=np.float32)
    out_max_y = np.empty(n, dtype=np.float32)
    out_max_z = np.empty(n, dtype=np.float32)

    futures: list[ray.ObjectRef] = []
    max_active = max(1, int(1.5 * int(max_processes)))
    for start, end in _iterate_chunk_indices(segments, chunk_size):
        fut = _compute_aabbs_worker_remote.remote(
            start,
            segments[start:end],
            guard_arcmin,
            epsilon_n_au,
            padding_method=padding_method,
            window_len=WINDOW_LEN,
        )
        futures.append(fut)
        if len(futures) >= max_active:
            finished, futures = ray.wait(futures, num_returns=1)
            s, res = ray.get(finished[0])
            e = s + len(res[0])
            (
                out_min_x[s:e],
                out_min_y[s:e],
                out_min_z[s:e],
                out_max_x[s:e],
                out_max_y[s:e],
                out_max_z[s:e],
            ) = res

    while futures:
        finished, futures = ray.wait(futures, num_returns=1)
        s, res = ray.get(finished[0])
        e = s + len(res[0])
        (
            out_min_x[s:e],
            out_min_y[s:e],
            out_min_z[s:e],
            out_max_x[s:e],
            out_max_y[s:e],
            out_max_z[s:e],
        ) = res

    return out_min_x, out_min_y, out_min_z, out_max_x, out_max_y, out_max_z
