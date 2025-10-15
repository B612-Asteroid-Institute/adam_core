from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from adam_core.geometry.anomaly_labeling import _compute_candidates_kernel


def _keplerian_to_cartesian_inertial(a, e, i, raan, ap, E):
    # Perifocal coordinates
    b = a * np.sqrt(1.0 - e * e)
    x_p = a * (np.cos(E) - e)
    y_p = b * np.sin(E)
    # Rotation matrix from perifocal to inertial (Q)
    cos_O = np.cos(raan)
    sin_O = np.sin(raan)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_w = np.cos(ap)
    sin_w = np.sin(ap)

    Q11 = cos_O * cos_w - sin_O * sin_w * cos_i
    Q12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    Q13 = sin_O * sin_i
    Q21 = sin_O * cos_w + cos_O * sin_w * cos_i
    Q22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    Q23 = -cos_O * sin_i
    Q31 = sin_w * sin_i
    Q32 = cos_w * sin_i
    Q33 = cos_i

    x = Q11 * x_p + Q12 * y_p
    y = Q21 * x_p + Q22 * y_p
    z = Q31 * x_p + Q32 * y_p
    return np.array([x, y, z])


def test_compute_candidates_kernel_exact_on_ellipse():
    # Single-hit exact geometry: observer at origin, ray points to true ellipse point
    a = 2.0
    e = 0.4
    i = np.deg2rad(15.0)
    raan = np.deg2rad(40.0)
    ap = np.deg2rad(25.0)
    E_true = 1.2

    r_xyz = _keplerian_to_cartesian_inertial(a, e, i, raan, ap, E_true)
    # Plane normal
    n = np.array([np.sin(i) * np.sin(raan), -np.sin(i) * np.cos(raan), np.cos(i)])
    plane_normal = jnp.asarray(n)
    # Choose observer 1 AU along +normal, direction pointing to the ellipse point
    ro_np = n / np.linalg.norm(n)
    rd_np = r_xyz - ro_np
    rd_np = rd_np / np.linalg.norm(rd_np)
    ro = jnp.asarray(ro_np)
    rd = jnp.asarray(rd_np)

    # Batch with one sample
    ray_origins = jnp.asarray(ro).reshape(1, 3)
    ray_directions = jnp.asarray(rd).reshape(1, 3)
    plane_normals = jnp.asarray(plane_normal).reshape(1, 3)
    arr_a = jnp.asarray(np.array([a]))
    arr_e = jnp.asarray(np.array([e]))
    arr_i = jnp.asarray(np.array([i]))
    arr_raan = jnp.asarray(np.array([raan]))
    arr_ap = jnp.asarray(np.array([ap]))

    max_k = 1
    snap_thr = float("inf")
    dedupe_tol = 1e-6

    plane_d, snap_d, nu_vals, valid_mask, n_valid = _compute_candidates_kernel(
        ray_origins,
        ray_directions,
        plane_normals,
        arr_a,
        arr_e,
        arr_i,
        arr_raan,
        arr_ap,
        max_k,
        snap_thr,
        dedupe_tol,
    )

    # Expect zero snap error and one valid candidate
    npt.assert_allclose(np.asarray(snap_d)[0, 0], 0.0, atol=1e-6)
    assert int(np.asarray(n_valid)[0]) >= 1
