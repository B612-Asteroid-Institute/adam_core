"""
Hardening tests for geometric overlap: negative controls, angular checks, determinism.
"""

import numpy as np
import pytest
from adam_assist import ASSISTPropagator

from adam_core.geometry import build_bvh, ephemeris_to_rays, query_bvh
from adam_core.observers.observers import Observers
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.orbits.query.sbdb import query_sbdb
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp


def _unit_vector(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _rotate_los_by_arcmin(u: np.ndarray, angle_arcmin: float) -> np.ndarray:
    """Rotate each LOS vector by a fixed small angle around a perpendicular axis.

    u: (N,3) unit vectors
    returns: (N,3) rotated unit vectors
    """
    angle_rad = np.deg2rad(angle_arcmin / 60.0)
    # Choose a fallback axis per vector that is not collinear with u
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    # If |u·x| < 0.9 use x, else use y
    dotx = (u * x_axis).sum(axis=-1)
    use_x = np.abs(dotx) < 0.9
    fallback = np.where(use_x[:, None], x_axis[None, :], y_axis[None, :])
    a = _unit_vector(np.cross(u, fallback))
    # Rodrigues' rotation formula (batch)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    cross_term = np.cross(a, u)
    proj = a * (a * u).sum(axis=-1, keepdims=True)
    u_rot = u * cos + cross_term * sin + proj * (1.0 - cos)
    return _unit_vector(u_rot)


def _build_basic_ephemerides(sbdb_id: str, n_variants: int = 10, n_epochs: int = 21):
    nominal = query_sbdb([sbdb_id])[0:1]
    variants = VariantOrbits.create(
        nominal, method="monte-carlo", num_samples=n_variants, seed=42
    )
    epoch_start = nominal.coordinates.time.mjd().to_numpy()[0]
    times = Timestamp.from_mjd(
        np.linspace(epoch_start, epoch_start + 90, n_epochs), scale="tdb"
    )
    stations = (
        ["X05"] * (n_epochs // 3)
        + ["T08"] * (n_epochs // 3)
        + ["I41"] * (n_epochs // 3)
    )
    observers = Observers.from_codes(times=times, codes=stations)
    prop = ASSISTPropagator()
    ephem = prop.generate_ephemeris(variants, observers, max_processes=1)
    return nominal, variants, observers, ephem


def _build_index(nominal, chord_arcmin: float):
    _, segments = sample_ellipse_adaptive(nominal, max_chord_arcmin=chord_arcmin)
    segs_aabb = compute_segment_aabbs(segments)
    bvh = build_bvh(segs_aabb)
    return segments, segs_aabb, bvh


class TestHardening:
    @pytest.mark.parametrize("guard_arcmin", [0.1, 0.05, 0.02, 0.01])
    def test_negative_control_offset(self, guard_arcmin):
        sbdb_id = "1998 SG172"
        nominal, variants, observers, ephem = _build_basic_ephemerides(sbdb_id)
        # Index with coarse chord (fast; proven sufficient)
        segments, segs_aabb, bvh = _build_index(nominal, chord_arcmin=60.0)

        # Build baseline rays
        det_ids = [
            f"{ephem.orbit_id[i].as_py()}:{ephem.coordinates.origin.code[i].as_py()}:{i}"
            for i in range(len(ephem))
        ]
        rays = ephemeris_to_rays(ephem, observers=observers, det_id=det_ids)

        # Offset LOS by +5 arcmin
        u = np.column_stack(
            [rays.u_x.to_numpy(), rays.u_y.to_numpy(), rays.u_z.to_numpy()]
        )
        u_rot = _rotate_los_by_arcmin(u, angle_arcmin=5.0)
        rays_offset = (
            rays.set_column("u_x", u_rot[:, 0])
            .set_column("u_y", u_rot[:, 1])
            .set_column("u_z", u_rot[:, 2])
        )

        # Query with tiny guard
        hits = query_bvh(bvh, segs_aabb, rays_offset, guard_arcmin=guard_arcmin)
        assert (
            len(hits) == 0
        ), f"Expected 0 hits at guard={guard_arcmin}′ after +5′ offset, got {len(hits)}"

    @pytest.mark.parametrize("guard_arcmin", [0.1, 0.05])
    def test_angular_separation_spotcheck(self, guard_arcmin):
        sbdb_id = "1998 SG172"
        nominal, variants, observers, ephem = _build_basic_ephemerides(sbdb_id)
        segments, segs_aabb, bvh = _build_index(nominal, chord_arcmin=60.0)

        det_ids = [
            f"{ephem.orbit_id[i].as_py()}:{ephem.coordinates.origin.code[i].as_py()}:{i}"
            for i in range(len(ephem))
        ]
        rays = ephemeris_to_rays(ephem, observers=observers, det_id=det_ids)
        hits = query_bvh(bvh, segs_aabb, rays, guard_arcmin=guard_arcmin)
        assert len(hits) > 0

        # Sample subset
        rng = np.random.default_rng(42)
        idx = rng.choice(len(hits), size=min(256, len(hits)), replace=False)

        # Build mapping from det_id -> ray index
        det_id_list = [
            f"{ephem.orbit_id[i].as_py()}:{ephem.coordinates.origin.code[i].as_py()}:{i}"
            for i in range(len(ephem))
        ]
        det_to_idx = {d: i for i, d in enumerate(det_id_list)}

        # Arrays for rays and segments
        u_arr = np.column_stack(
            [rays.u_x.to_numpy(), rays.u_y.to_numpy(), rays.u_z.to_numpy()]
        )
        O_arr = np.column_stack(
            [
                rays.observer.x.to_numpy(),
                rays.observer.y.to_numpy(),
                rays.observer.z.to_numpy(),
            ]
        )
        P0_arr = np.column_stack(
            [segments.x0.to_numpy(), segments.y0.to_numpy(), segments.z0.to_numpy()]
        )
        P1_arr = np.column_stack(
            [segments.x1.to_numpy(), segments.y1.to_numpy(), segments.z1.to_numpy()]
        )

        # Gather sampled pairs
        hit_det_ids = hits.det_id.to_pylist()
        hit_seg_ids = hits.seg_id.to_numpy()
        ray_idx = np.array([det_to_idx[d] for d in hit_det_ids])[idx]
        seg_idx = hit_seg_ids[idx]

        O = O_arr[ray_idx]
        U = _unit_vector(u_arr[ray_idx])
        P0 = P0_arr[seg_idx]
        V = P1_arr[seg_idx] - P0

        # Closest point between ray (O + t U, t>=0) and segment (P0 + s V, s in [0,1])
        a = (U * U).sum(axis=1)
        b = (U * V).sum(axis=1)
        c = (V * V).sum(axis=1)
        W0 = O - P0
        d = -(U * W0).sum(axis=1)
        e = (V * W0).sum(axis=1)
        denom = a * c - b * b
        # Unconstrained
        s = (b * d + a * e) / np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        t = (b * e + c * d) / np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        # Clamp s to [0,1], recompute t accordingly, clamp t to >=0
        s = np.clip(s, 0.0, 1.0)
        # Recompute t from derivative condition with s fixed: t = -U·(W0 + sV)/a
        t = -(U * (W0 + s[:, None] * V)).sum(axis=1) / np.where(a < 1e-15, 1e-15, a)
        t = np.maximum(t, 0.0)

        S = P0 + s[:, None] * V
        R = O + t[:, None] * U
        # Use direction from observer to closest segment point
        dir_to_seg = _unit_vector(S - O)
        dots = np.clip((dir_to_seg * U).sum(axis=1), -1.0, 1.0)
        ang = np.arccos(dots)
        guard_rad = np.deg2rad(guard_arcmin / 60.0)
        assert np.all(
            ang < guard_rad + 1e-8
        ), "Some hits exceed guard angular separation"
