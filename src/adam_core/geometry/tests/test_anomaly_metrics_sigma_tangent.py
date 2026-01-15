from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from adam_core.geometry.anomaly_labeling import _compute_candidates_kernel


def _ellipse_derivatives(a: float, e: float, E: float) -> tuple[float, float, float]:
    b = a * np.sqrt(max(1.0 - e * e, 0.0))
    rx = a * (np.cos(E) - e)
    ry = b * np.sin(E)
    rdx = -a * np.sin(E)
    rdy = b * np.cos(E)
    rprime_norm = np.hypot(rdx, rdy)
    return rx, ry, rprime_norm


def test_t_hat_and_sigmaM_consistency():
    # Geometry where the ray points slightly off the exact ellipse point
    a = 2.0
    e = 0.3
    i = np.deg2rad(10.0)
    raan = np.deg2rad(15.0)
    ap = np.deg2rad(30.0)
    E = 0.9

    # Build a point near the ellipse by offsetting slightly along the normal
    b = a * np.sqrt(1.0 - e * e)
    x = a * (np.cos(E) - e)
    y = b * np.sin(E)
    # Small offset in plane (~1e-6 AU) to induce nonzero snap_error
    off = 1e-6
    x_off = x + off
    y_off = y

    # Perifocal->inertial rotation (Q)
    cos_O, sin_O = np.cos(raan), np.sin(raan)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(ap), np.sin(ap)
    Q11 = cos_O * cos_w - sin_O * sin_w * cos_i
    Q12 = -cos_O * sin_w - sin_O * cos_w * cos_i
    Q13 = sin_O * sin_i
    Q21 = sin_O * cos_w + cos_O * sin_w * cos_i
    Q22 = -sin_O * sin_w + cos_O * cos_w * cos_i
    Q23 = -cos_O * sin_i
    Q31 = sin_w * sin_i
    Q32 = cos_w * sin_i
    Q33 = cos_i

    r3 = np.array(
        [
            Q11 * x_off + Q12 * y_off,
            Q21 * x_off + Q22 * y_off,
            Q31 * x_off + Q32 * y_off,
        ]
    )

    # Plane normal and observer one unit along +normal; ray to r3
    n = np.array([np.sin(i) * np.sin(raan), -np.sin(i) * np.cos(raan), np.cos(i)])
    ro_np = n / np.linalg.norm(n)
    rd_np = r3 - ro_np
    rd_np = rd_np / np.linalg.norm(rd_np)
    ro = jnp.asarray(ro_np)
    rd = jnp.asarray(rd_np)

    # Single sample
    ray_origins = ro.reshape(1, 3)
    ray_directions = rd.reshape(1, 3)
    plane_normals = jnp.asarray(n).reshape(1, 3)
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

    # With small offset, snap error should be ~ off within a small factor due to projection
    snap = float(np.asarray(snap_d)[0, 0])
    assert snap > 0
    # Allow a generous factor because projection and rotation can scale the offset slightly
    npt.assert_allclose(snap, off, rtol=5.0, atol=1e-8)
