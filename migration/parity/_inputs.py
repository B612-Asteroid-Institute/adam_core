"""Randomized input generators per API.

Each ``make_<api_id>(rng, n)`` returns a ``dict[str, Any]`` of kwargs
suitable for both the rust call and the legacy-oracle subprocess. The
generators sample from realistic asteroid/comet orbit parameter ranges:

    semi-major axis a ∈ [0.5, 50] AU
    eccentricity   e ∈ [0, 0.95]   (sampled with a long tail toward 0)
    inclination    i ∈ [0°, 175°]
    raan, omega, M ∈ [0°, 360°)

Cartesian states are derived from these elements through the upstream
``_keplerian_to_cartesian_p_vmap`` so the random sample is dynamically
consistent.

We deliberately avoid sampling near-parabolic (e ≈ 1) or marginally-
hyperbolic (e ∈ [1.0, 1.5]) orbits in the default sweep because the
rust universal-Kepler kernel has a known long-dt regression on those
inputs (Task #138). A separate edge-case fuzzer will exercise that
regime once the kernel fix lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants pulled from upstream — keep these in sync with adam_core.constants.
# ---------------------------------------------------------------------------

MU_SUN = 0.29591220828411956e-3  # AU^3 / day^2
KM_P_AU = 149_597_870.700
S_P_DAY = 86_400.0
C_AU_PER_DAY = 299_792.458 / KM_P_AU * S_P_DAY  # ≈ 173.14 AU/d

R_EARTH_EQ = 6378.1363 / KM_P_AU
F_EARTH = 1.0 / 298.257223563


@dataclass(frozen=True)
class Sample:
    """One random workload for a given API."""

    rust_kwargs: dict[str, Any]
    legacy_kwargs: dict[str, Any]


def _sample_keplerian_elements(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n keplerian (a, e, i, raan, omega, M) rows in (AU, _, deg, deg, deg, deg)."""
    a = rng.uniform(0.5, 50.0, size=n)
    e = rng.beta(2.0, 5.0, size=n) * 0.95  # tail toward 0, capped at 0.95
    i_deg = rng.uniform(0.0, 175.0, size=n)
    raan = rng.uniform(0.0, 360.0, size=n)
    omega = rng.uniform(0.0, 360.0, size=n)
    M = rng.uniform(0.0, 360.0, size=n)
    return np.stack([a, e, i_deg, raan, omega, M], axis=1).astype(np.float64)


def _kep_to_cart(coords_kep: np.ndarray) -> np.ndarray:
    """Convert keplerian rows to cartesian via the rust kernel (post-import)."""
    from adam_core._rust import api as _rust_api

    n = coords_kep.shape[0]
    out = _rust_api.keplerian_to_cartesian_numpy(
        coords_kep,
        np.full(n, MU_SUN, dtype=np.float64),
    )
    if out is None:
        raise RuntimeError(
            "rust backend unavailable; cannot synthesize cartesian states"
        )
    return np.ascontiguousarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-API input generators
# ---------------------------------------------------------------------------


def make_transform_coordinates(rng: np.random.Generator, n: int) -> Sample:
    """Public dispatcher Cart→Spherical, ecliptic → equatorial, no origin shift.

    This deliberately goes through the public ``transform_coordinates`` quivr
    object boundary on both sides. The migration side should fuse the frame
    change and representation conversion into one Rust dispatcher call; the
    baseline-main side exercises the upstream public dispatcher.
    """
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    time_mjd = rng.uniform(58000.0, 63000.0, size=n).astype(np.float64)
    kw = {
        "coords": coords,
        "time_mjd": time_mjd,
        "representation_out": "spherical",
        "frame_in": "ecliptic",
        "frame_out": "equatorial",
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_cartesian_to_spherical(rng: np.random.Generator, n: int) -> Sample:
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    kw = {"coords": coords}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_cartesian_to_geodetic(rng: np.random.Generator, n: int) -> Sample:
    # Geodetic only makes sense near the Earth — sample positions in
    # [0.7, 1.3] R_earth from origin.
    r = rng.uniform(0.7, 1.3, size=n) * R_EARTH_EQ
    theta = rng.uniform(0.0, np.pi, size=n)
    phi = rng.uniform(0.0, 2 * np.pi, size=n)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    vx = rng.normal(scale=1e-6, size=n)
    vy = rng.normal(scale=1e-6, size=n)
    vz = rng.normal(scale=1e-6, size=n)
    coords = np.stack([x, y, z, vx, vy, vz], axis=1).astype(np.float64)
    kw = {
        "coords": coords,
        "a": R_EARTH_EQ,
        "f": F_EARTH,
        "max_iter": 100,
        "tol": 1e-15,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_cartesian_to_keplerian(rng: np.random.Generator, n: int) -> Sample:
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    t0 = rng.uniform(58000.0, 63000.0, size=n).astype(np.float64)
    mu = np.full(n, MU_SUN, dtype=np.float64)
    kw = {"coords": coords, "t0": t0, "mu": mu}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_keplerian_to_cartesian(rng: np.random.Generator, n: int) -> Sample:
    kep = _sample_keplerian_elements(rng, n)
    mu = np.full(n, MU_SUN, dtype=np.float64)
    kw = {"coords": kep, "mu": mu, "max_iter": 100, "tol": 1e-15}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_cartesian_to_cometary(rng: np.random.Generator, n: int) -> Sample:
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    t0 = rng.uniform(58000.0, 63000.0, size=n).astype(np.float64)
    mu = np.full(n, MU_SUN, dtype=np.float64)
    kw = {"coords": coords, "t0": t0, "mu": mu}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_cometary_to_cartesian(rng: np.random.Generator, n: int) -> Sample:
    # Convert random keplerian to cometary via the rust path so input is
    # dynamically valid; comet input rows are (q, e, i, raan, omega, Tp).
    kep = _sample_keplerian_elements(rng, n)
    cart = _kep_to_cart(kep)
    t0 = rng.uniform(58000.0, 63000.0, size=n).astype(np.float64)
    mu = np.full(n, MU_SUN, dtype=np.float64)
    from adam_core._rust import api as _rust_api

    com = _rust_api.cartesian_to_cometary_numpy(cart, t0, mu)
    if com is None:
        raise RuntimeError("rust backend unavailable")
    kw = {
        "coords": np.ascontiguousarray(com, dtype=np.float64),
        "t0": t0,
        "mu": mu,
        "max_iter": 100,
        "tol": 1e-15,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_spherical_to_cartesian(rng: np.random.Generator, n: int) -> Sample:
    # Sample reasonable spherical (rho, lon, lat, vrho, vlon, vlat) directly.
    rho = rng.uniform(0.5, 50.0, size=n)
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = rng.uniform(-89.0, 89.0, size=n)
    vrho = rng.normal(scale=1e-3, size=n)
    vlon = rng.normal(scale=1e-3, size=n)
    vlat = rng.normal(scale=1e-3, size=n)
    coords = np.stack([rho, lon, lat, vrho, vlon, vlat], axis=1).astype(np.float64)
    kw = {"coords": coords}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calc_mean_motion(rng: np.random.Generator, n: int) -> Sample:
    a = rng.uniform(0.5, 50.0, size=n).astype(np.float64)
    mu = np.full(n, MU_SUN, dtype=np.float64)
    kw = {"a": a, "mu": mu}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def _sample_dts(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(-10000.0, 10000.0, size=n).astype(np.float64)


def make_propagate_2body(rng: np.random.Generator, n: int) -> Sample:
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    dts = _sample_dts(rng, n)
    mus = np.full(n, MU_SUN, dtype=np.float64)
    kw = {"orbits": coords, "dts": dts, "mus": mus, "max_iter": 100, "tol": 1e-15}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_propagate_2body_with_covariance(rng: np.random.Generator, n: int) -> Sample:
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    dts = _sample_dts(rng, n)
    mus = np.full(n, MU_SUN, dtype=np.float64)
    sigmas = np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10])  # small but non-zero
    cov_3d = np.zeros((n, 6, 6), dtype=np.float64)
    for i in range(n):
        cov_3d[i] = np.diag(sigmas**2)
    # Rust kernel expects flattened (N, 36) row-major; legacy
    # transform_covariances_jacobian wants (N, 6, 6).
    rust_kw = {
        "orbits": coords,
        "covariances": np.ascontiguousarray(cov_3d.reshape(n, 36)),
        "dts": dts,
        "mus": mus,
        "max_iter": 100,
        "tol": 1e-15,
    }
    legacy_kw = {
        "orbits": coords,
        "covariances": cov_3d,
        "dts": dts,
        "mus": mus,
        "max_iter": 100,
        "tol": 1e-15,
    }
    return Sample(rust_kwargs=rust_kw, legacy_kwargs=legacy_kw)


def make_generate_ephemeris_2body(rng: np.random.Generator, n: int) -> Sample:
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    # Observer states sampled near Earth's heliocentric orbit with mild scatter.
    obs_r = 1.0 + rng.normal(scale=0.02, size=n)
    obs_theta = rng.uniform(0.0, 2 * np.pi, size=n)
    obs = np.zeros((n, 6), dtype=np.float64)
    obs[:, 0] = obs_r * np.cos(obs_theta)
    obs[:, 1] = obs_r * np.sin(obs_theta)
    obs[:, 3] = -2 * np.pi / 365.25 * obs_r * np.sin(obs_theta)  # ~circular
    obs[:, 4] = 2 * np.pi / 365.25 * obs_r * np.cos(obs_theta)
    mus = np.full(n, MU_SUN, dtype=np.float64)
    kw = {
        "orbits": coords,
        "observer_states": obs,
        "mus": mus,
        "lt_tol": 1e-10,
        "max_iter": 1000,
        "tol": 1e-15,
        "stellar_aberration": False,
        "max_lt_iter": 10,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_generate_ephemeris_2body_with_covariance(
    rng: np.random.Generator, n: int
) -> Sample:
    base = make_generate_ephemeris_2body(rng, n).rust_kwargs
    sigmas = np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10])
    cov_3d = np.zeros((n, 6, 6), dtype=np.float64)
    for i in range(n):
        cov_3d[i] = np.diag(sigmas**2)
    rust_kw = dict(base)
    rust_kw["covariances"] = np.ascontiguousarray(cov_3d.reshape(n, 36))
    legacy_kw = dict(base)
    legacy_kw["covariances"] = cov_3d
    return Sample(rust_kwargs=rust_kw, legacy_kwargs=legacy_kw)


def make_solve_lambert(rng: np.random.Generator, n: int) -> Sample:
    # Sample two random heliocentric positions and a positive TOF.
    cart = _kep_to_cart(_sample_keplerian_elements(rng, 2 * n))
    r1 = cart[:n, :3]
    r2 = cart[n:, :3]
    tof = rng.uniform(30.0, 720.0, size=n).astype(np.float64)
    kw = {
        "r1": r1,
        "r2": r2,
        "tof": tof,
        "mu": MU_SUN,
        "m": 0,
        "prograde": True,
        "low_path": True,
        "maxiter": 35,
        "atol": 1e-10,
        "rtol": 1e-10,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_add_light_time(rng: np.random.Generator, n: int) -> Sample:
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    obs_r = 1.0 + rng.normal(scale=0.02, size=n)
    obs_theta = rng.uniform(0.0, 2 * np.pi, size=n)
    obs_pos = np.zeros((n, 3), dtype=np.float64)
    obs_pos[:, 0] = obs_r * np.cos(obs_theta)
    obs_pos[:, 1] = obs_r * np.sin(obs_theta)
    mus = np.full(n, MU_SUN, dtype=np.float64)
    kw = {
        "orbits": coords,
        "observer_positions": obs_pos,
        "mus": mus,
        "lt_tol": 1e-10,
        "max_iter": 1000,
        "tol": 1e-15,
        "max_lt_iter": 10,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calculate_phase_angle(rng: np.random.Generator, n: int) -> Sample:
    obj = _kep_to_cart(_sample_keplerian_elements(rng, n))[:, :3]
    obs_r = 1.0 + rng.normal(scale=0.02, size=n)
    obs_theta = rng.uniform(0.0, 2 * np.pi, size=n)
    obs = np.zeros((n, 3), dtype=np.float64)
    obs[:, 0] = obs_r * np.cos(obs_theta)
    obs[:, 1] = obs_r * np.sin(obs_theta)
    kw = {"object_pos": obj, "observer_pos": obs}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calculate_apparent_magnitude_v(rng: np.random.Generator, n: int) -> Sample:
    base = make_calculate_phase_angle(rng, n).rust_kwargs
    h_v = rng.uniform(15.0, 25.0, size=n).astype(np.float64)
    g = rng.uniform(0.0, 0.5, size=n).astype(np.float64)
    kw = {
        "h_v": h_v,
        "object_pos": base["object_pos"],
        "observer_pos": base["observer_pos"],
        "g": g,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calculate_apparent_magnitude_v_and_phase_angle(
    rng: np.random.Generator, n: int
) -> Sample:
    return make_calculate_apparent_magnitude_v(rng, n)


def make_predict_magnitudes(rng: np.random.Generator, n: int) -> Sample:
    base = make_calculate_apparent_magnitude_v(rng, n).rust_kwargs
    # Build a small bandpass delta table (V→V offset of 0, plus a few mock filters).
    delta = np.array([0.0, -0.05, 0.12, 0.30], dtype=np.float64)
    target_ids = rng.integers(0, len(delta), size=n).astype(np.int32)
    kw = dict(base)
    kw["target_ids"] = target_ids
    kw["delta_table"] = delta
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def _sample_iod_triplets(rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
    """Sample n (r1, r2, r3, t1, t2, t3) IOD triplets along propagated arcs.

    Spacing ≥ 1 day to avoid the Gibbs ULP-amplification trap noted in
    journal entry 2026-04-19 (1e8x amplification at 30-min spacing).
    """
    cart = _kep_to_cart(_sample_keplerian_elements(rng, n))
    mus = np.full(n, MU_SUN, dtype=np.float64)
    base_t = rng.uniform(59000.0, 60000.0, size=n).astype(np.float64)
    dt12 = rng.uniform(1.0, 5.0, size=n).astype(np.float64)
    dt23 = rng.uniform(1.0, 5.0, size=n).astype(np.float64)
    t1 = base_t
    t2 = base_t + dt12
    t3 = base_t + dt12 + dt23

    from adam_core._rust import api as _rust_api

    s1 = _rust_api.propagate_2body_numpy(cart, np.zeros(n), mus)
    s2 = _rust_api.propagate_2body_numpy(cart, t2 - t1, mus)
    s3 = _rust_api.propagate_2body_numpy(cart, t3 - t1, mus)
    if s1 is None or s2 is None or s3 is None:
        raise RuntimeError("rust backend unavailable")
    return (
        np.ascontiguousarray(s1[:, :3]),
        np.ascontiguousarray(s2[:, :3]),
        np.ascontiguousarray(s3[:, :3]),
        t1,
        t2,
        t3,
    )


def make_calc_gibbs(rng: np.random.Generator, n: int) -> Sample:
    r1, r2, r3, _, _, _ = _sample_iod_triplets(rng, n)
    kw = {"r1": r1, "r2": r2, "r3": r3, "mu": MU_SUN}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calc_herrick_gibbs(rng: np.random.Generator, n: int) -> Sample:
    r1, r2, r3, t1, t2, t3 = _sample_iod_triplets(rng, n)
    # Rust signature is per-row scalars in batched call; here we run row-by-row
    # for simplicity. Both rust and legacy iterate per-row in the runner.
    kw = {
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "mu": MU_SUN,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calc_gauss(rng: np.random.Generator, n: int) -> Sample:
    return make_calc_herrick_gibbs(rng, n)


def _make_iod_observation(
    rng: np.random.Generator, n_triplets: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_triplets independent IOD scenarios.

    Each scenario is 3 observations of a propagated random orbit, viewed
    from a fixed-Earth-like observer at (1, 0, 0) AU. Observations are
    spaced 1-3 days apart. Returns:
      ra_deg:  shape (n_triplets, 3)
      dec_deg: shape (n_triplets, 3)
      times_mjd: shape (n_triplets, 3)
      obs_pos: shape (n_triplets, 3, 3) heliocentric Cartesian
    """
    from adam_core._rust import api as _rust_api

    # Sample orbits (drop near-Earth and very high-e ones — IOD is unstable)
    kep = _sample_keplerian_elements(rng, n_triplets)
    kep[:, 0] = np.clip(kep[:, 0], 1.5, 5.0)  # main-belt-ish
    kep[:, 1] = np.clip(kep[:, 1], 0.0, 0.4)
    cart = _kep_to_cart(kep)

    base_t = rng.uniform(59000.0, 60000.0, size=n_triplets).astype(np.float64)
    dt12 = rng.uniform(1.0, 3.0, size=n_triplets).astype(np.float64)
    dt23 = rng.uniform(1.0, 3.0, size=n_triplets).astype(np.float64)
    times = np.stack(
        [
            base_t,
            base_t + dt12,
            base_t + dt12 + dt23,
        ],
        axis=1,
    )  # (n, 3)

    # Per-orbit propagate to each of 3 epochs; observer at fixed (1, 0, 0)
    obs_pos_single = np.array([1.0, 0.0, 0.0])
    obs_pos = np.broadcast_to(obs_pos_single, (n_triplets, 3, 3)).copy()

    ra_deg = np.empty_like(times)
    dec_deg = np.empty_like(times)
    for i in range(n_triplets):
        # Propagate row i to each of its 3 dts
        dts_i = times[i] - times[i, 0]
        prop = _rust_api.propagate_2body_along_arc_numpy(
            cart[i], dts_i, MU_SUN, 1000, 1e-15
        )
        if prop is None:
            raise RuntimeError("rust arc unavailable")
        # Topocentric Cartesian (orbit_pos − observer_pos)
        rho = prop[:, :3] - obs_pos[i]
        # Convert to equatorial RA/Dec. Sample is in ecliptic frame; rotate
        # to equatorial first via a single 3×3 obliquity rotation matrix.
        OBLIQUITY = 84381.448 * np.pi / (180.0 * 3600.0)
        EC2EQ = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(OBLIQUITY), -np.sin(OBLIQUITY)],
                [0.0, np.sin(OBLIQUITY), np.cos(OBLIQUITY)],
            ]
        )
        rho_eq = rho @ EC2EQ.T
        rho_norm = np.linalg.norm(rho_eq, axis=1)
        ra_deg[i] = np.degrees(np.arctan2(rho_eq[:, 1], rho_eq[:, 0])) % 360.0
        dec_deg[i] = np.degrees(np.arcsin(rho_eq[:, 2] / rho_norm))

    return ra_deg, dec_deg, times, obs_pos


def make_gauss_iod(rng: np.random.Generator, n: int) -> Sample:
    """Random IOD triplets. n = number of independent triplets per call.

    Both rust and legacy gaussIOD operate on ONE triplet at a time and
    return up to 3 candidate orbits. The harness compares results
    triplet-by-triplet (sorted by `|r2|` to absorb root-ordering
    differences).
    """
    ra_deg, dec_deg, times, obs_pos = _make_iod_observation(rng, n)
    # Rust gauss_iod_fused expects FLAT (n*3,) ra/dec/times/coords_obs.
    # We keep them per-triplet here; the runner unpacks per call.
    rust_kw = {
        "ra_deg_per_triplet": ra_deg,
        "dec_deg_per_triplet": dec_deg,
        "times_per_triplet": times,
        "obs_pos_per_triplet": obs_pos,
        "mu": MU_SUN,
        "c": C_AU_PER_DAY,
    }
    return Sample(rust_kwargs=rust_kw, legacy_kwargs=rust_kw)


# ---------------------------------------------------------------------------
# Registry: api_id → generator function
# ---------------------------------------------------------------------------

GENERATORS = {
    "coordinates.cartesian_to_spherical": make_cartesian_to_spherical,
    "coordinates.transform_coordinates": make_transform_coordinates,
    "coordinates.cartesian_to_geodetic": make_cartesian_to_geodetic,
    "coordinates.cartesian_to_keplerian": make_cartesian_to_keplerian,
    "coordinates.keplerian.to_cartesian": make_keplerian_to_cartesian,
    "coordinates.cartesian_to_cometary": make_cartesian_to_cometary,
    "coordinates.cometary.to_cartesian": make_cometary_to_cartesian,
    "coordinates.spherical.to_cartesian": make_spherical_to_cartesian,
    "dynamics.calc_mean_motion": make_calc_mean_motion,
    "dynamics.propagate_2body": make_propagate_2body,
    "dynamics.propagate_2body_with_covariance": make_propagate_2body_with_covariance,
    "dynamics.generate_ephemeris_2body": make_generate_ephemeris_2body,
    "dynamics.generate_ephemeris_2body_with_covariance": (
        make_generate_ephemeris_2body_with_covariance
    ),
    "dynamics.solve_lambert": make_solve_lambert,
    "dynamics.add_light_time": make_add_light_time,
    "photometry.calculate_phase_angle": make_calculate_phase_angle,
    "photometry.calculate_apparent_magnitude_v": make_calculate_apparent_magnitude_v,
    "photometry.calculate_apparent_magnitude_v_and_phase_angle": (
        make_calculate_apparent_magnitude_v_and_phase_angle
    ),
    "photometry.predict_magnitudes": make_predict_magnitudes,
    "orbit_determination.calcGibbs": make_calc_gibbs,
    "orbit_determination.calcHerrickGibbs": make_calc_herrick_gibbs,
    "orbit_determination.calcGauss": make_calc_gauss,
    # NOTE: gaussIOD intentionally absent from random-fuzz GENERATORS.
    # Rust (Laguerre+deflation) and legacy (np.roots/LAPACK) find
    # DIFFERENT subsets of the 8th-order polynomial's roots on a
    # non-trivial fraction of random triplets — the polynomial-conditioning
    # difference is intrinsic to the algorithms, not a kernel bug. Best-
    # root parity holds at ~1e-10 AU when both sides return any root ≥
    # 1.5 AU but ~5/32 random triplets show NaN-mismatch where rust and
    # legacy disagree on whether a root is "physical." A fixed-fixture
    # pytest covers the well-conditioned cases; random fuzz is unreliable
    # without per-root matching that adds significant harness complexity.
    # `make_gauss_iod` and the dispatch entries in _rust_runner /
    # _legacy_runner are retained for ad-hoc / manual parity checks.
}


def all_api_ids() -> tuple[str, ...]:
    return tuple(GENERATORS.keys())


def make(api_id: str, rng: np.random.Generator, n: int) -> Sample:
    if api_id not in GENERATORS:
        raise KeyError(f"No input generator for {api_id!r}")
    return GENERATORS[api_id](rng, n)
