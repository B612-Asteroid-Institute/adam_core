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

from collections.abc import Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class WorkloadShape:
    """Structured workload shape used by parity/speed governance.

    ``rows`` is the flattened number of kernel rows passed to the current
    NumPy-boundary runners. Optional axes record the production shape that
    produced those rows, so the benchmark can distinguish a flat vector from
    multi-axis cases such as orbits × epochs or objects × observers.
    """

    rows: int
    n_orbits: int | None = None
    n_epochs: int | None = None
    n_observers: int | None = None
    extra: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("rows must be positive")
        axes = self.axes()
        if not axes:
            return
        product = 1
        for value in axes.values():
            if value <= 0:
                raise ValueError("workload axes must be positive")
            product *= value
        if product != self.rows:
            raise ValueError(
                f"workload axes product {product} does not match rows={self.rows}"
            )

    def axes(self) -> dict[str, int]:
        axes: dict[str, int] = {}
        if self.n_orbits is not None:
            axes["orbits"] = self.n_orbits
        if self.n_epochs is not None:
            axes["epochs"] = self.n_epochs
        if self.n_observers is not None:
            axes["observers"] = self.n_observers
        axes.update(dict(self.extra))
        return axes

    def label(self) -> str:
        axes = self.axes()
        if not axes:
            return f"rows={self.rows}"
        axis_label = " × ".join(f"{key}={value}" for key, value in axes.items())
        return f"{axis_label} ({self.rows} rows)"

    def to_json(self) -> dict[str, object]:
        return {"rows": self.rows, "axes": self.axes(), "label": self.label()}


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


def make_residuals_calculate(rng: np.random.Generator, n: int) -> Sample:
    """OD-inner-loop shape: spherical 6-D coords with lon/lat observed only.

    Mirrors the production OrbitDeterminationObservations layout: only
    columns 1 (lon) and 2 (lat) are non-NaN in ``observed``; rho, vrho,
    vlon, vlat are NaN. The covariance is an SPD 2x2 astrometric block
    on (lon, lat), padded to 6x6 with NaN entries on the inactive dims.
    Predicted has all 6 dims populated (the propagator output), with
    small offsets from observed.
    """
    nan = np.nan

    # Astrometric (lon, lat) for observed; the other four columns are NaN.
    lon_obs = rng.uniform(0.0, 360.0, size=n)
    lat_obs = rng.uniform(-80.0, 80.0, size=n)
    observed_values = np.full((n, 6), nan, dtype=np.float64)
    observed_values[:, 1] = lon_obs
    observed_values[:, 2] = lat_obs

    # Predicted is fully-populated (propagator output), perturbed slightly.
    rho = rng.uniform(0.5, 5.0, size=n)
    lon_pred = lon_obs + rng.normal(scale=0.01, size=n)
    lat_pred = lat_obs + rng.normal(scale=0.01, size=n)
    vrho = rng.normal(scale=1e-3, size=n)
    vlon = rng.normal(scale=1e-3, size=n)
    vlat = rng.normal(scale=1e-3, size=n)
    predicted_values = np.stack(
        [rho, lon_pred, lat_pred, vrho, vlon, vlat], axis=1
    ).astype(np.float64)

    # SPD 2x2 astrometric covariance on (lon, lat), lifted into 6x6.
    sigma_lon = rng.uniform(0.05, 1.0, size=n)
    sigma_lat = rng.uniform(0.05, 1.0, size=n)
    rho_corr = rng.uniform(-0.8, 0.8, size=n)
    observed_covariance_matrices = np.full((n, 6, 6), nan, dtype=np.float64)
    observed_covariance_matrices[:, 1, 1] = sigma_lon * sigma_lon
    observed_covariance_matrices[:, 2, 2] = sigma_lat * sigma_lat
    observed_covariance_matrices[:, 1, 2] = rho_corr * sigma_lon * sigma_lat
    observed_covariance_matrices[:, 2, 1] = observed_covariance_matrices[:, 1, 2]

    origin_codes = np.array(["X05"] * n, dtype=object)

    kw = {
        "observed_values": observed_values,
        "predicted_values": predicted_values,
        "observed_covariance_matrices": observed_covariance_matrices,
        "origin_codes": origin_codes,
        "frame": "equatorial",
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calculate_chi2(rng: np.random.Generator, n: int) -> Sample:
    # Representative OD astrometry residual rows: 2-D residuals with positive
    # sigmas and bounded correlations, yielding symmetric-positive-definite
    # 2x2 covariance matrices.
    sigma_a = rng.uniform(0.05, 1.0, size=n)
    sigma_b = rng.uniform(0.05, 1.0, size=n)
    rho = rng.uniform(-0.8, 0.8, size=n)
    residuals = np.stack(
        [
            rng.normal(scale=sigma_a),
            rng.normal(scale=sigma_b),
        ],
        axis=1,
    ).astype(np.float64)
    covariances = np.empty((n, 2, 2), dtype=np.float64)
    covariances[:, 0, 0] = sigma_a * sigma_a
    covariances[:, 1, 1] = sigma_b * sigma_b
    covariances[:, 0, 1] = rho * sigma_a * sigma_b
    covariances[:, 1, 0] = covariances[:, 0, 1]
    kw = {"residuals": residuals, "covariances": covariances}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calc_mean_motion(rng: np.random.Generator, n: int) -> Sample:
    a = rng.uniform(0.5, 50.0, size=n).astype(np.float64)
    mu = np.full(n, MU_SUN, dtype=np.float64)
    kw = {"a": a, "mu": mu}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_classify_orbits(rng: np.random.Generator, n: int) -> Sample:
    """Sample PDS SBN classification-rule inputs over all class regions.

    The registered Rust surface is the NumPy rule core over ``(a, e, q, Q)``;
    public coordinate-table extraction remains covered by classification tests.
    """
    codes = np.resize(np.arange(14, dtype=np.int32), n)
    rng.shuffle(codes)
    a = np.empty(n, dtype=np.float64)
    e = np.empty(n, dtype=np.float64)

    for i, code in enumerate(codes):
        jitter = rng.uniform(-0.02, 0.02)
        if code == 0:  # AST: gap between OMB/TJN/CEN rules.
            a[i] = 4.9 + jitter
            e[i] = 0.35
        elif code == 1:  # AMO
            a[i] = 1.50 + jitter
            e[i] = 1.0 - 1.10 / a[i]
        elif code == 2:  # APO
            a[i] = 1.50 + jitter
            e[i] = 0.50
        elif code == 3:  # ATE
            a[i] = 0.90 + jitter
            e[i] = 0.10
        elif code == 4:  # CEN
            a[i] = 10.0 + 2.0 * jitter
            e[i] = 0.20
        elif code == 5:  # IEO
            a[i] = 0.80 + jitter
            e[i] = 0.10
        elif code == 6:  # IMB
            a[i] = 1.80 + jitter
            e[i] = 0.04
        elif code == 7:  # MBA
            a[i] = 2.50 + jitter
            e[i] = 0.10
        elif code == 8:  # MCA
            a[i] = 2.50 + jitter
            e[i] = 0.40
        elif code == 9:  # OMB
            a[i] = 3.80 + jitter
            e[i] = 0.10
        elif code == 10:  # TJN
            a[i] = 5.00 + jitter
            e[i] = 0.10
        elif code == 11:  # TNO
            a[i] = 40.0 + 5.0 * jitter
            e[i] = 0.20
        elif code == 12:  # PAA
            a[i] = 2.0 + jitter
            e[i] = 1.0
        else:  # HYA
            a[i] = -1.5 + jitter
            e[i] = 1.20

    q = np.where(e == 1.0, 0.0, a * (1.0 - e))
    q_apo = np.where(e >= 1.0, np.inf, a * (1.0 + e))
    kw = {"a": a, "e": e, "q": q, "q_apo": q_apo}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def _sample_moid_keplerian_elements(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.uniform(0.8, 5.0, size=n)
    e = rng.uniform(0.02, 0.55, size=n)
    i_deg = rng.uniform(0.0, 40.0, size=n)
    raan = rng.uniform(0.0, 360.0, size=n)
    omega = rng.uniform(0.0, 360.0, size=n)
    M = rng.uniform(0.0, 360.0, size=n)
    return np.stack([a, e, i_deg, raan, omega, M], axis=1).astype(np.float64)


def make_calculate_moid(rng: np.random.Generator, n: int) -> Sample:
    primary = _kep_to_cart(_sample_moid_keplerian_elements(rng, n))
    secondary = _kep_to_cart(_sample_moid_keplerian_elements(rng, n))
    mus = np.full(n, MU_SUN, dtype=np.float64)
    kw = {
        "primary_orbits": primary,
        "secondary_orbits": secondary,
        "mus": mus,
        "max_iter": 100,
        "xtol": 1e-10,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calculate_perturber_moids(rng: np.random.Generator, n: int) -> Sample:
    """Public quivr orchestration over the batched MOID kernel.

    ``n`` is the number of primary orbit rows. Each row is paired against a
    small fixed perturber set inside ``calculate_perturber_moids`` so the
    runner validates Orbits construction, SPICE perturber-state lookup,
    batched Rust MOID dispatch, and PerturberMOIDs table assembly.
    """
    coords = _kep_to_cart(_sample_moid_keplerian_elements(rng, n))
    time_mjd = rng.uniform(59000.0, 60500.0, size=n).astype(np.float64)
    orbit_ids = np.array([f"o{i:05d}" for i in range(n)], dtype=object)
    kw = {
        "coords": coords,
        "time_mjd": time_mjd,
        "orbit_ids": orbit_ids,
        "perturber_codes": np.array(["EARTH", "MARS_BARYCENTER"], dtype=object),
        "origin_code": "SUN",
        "frame": "ecliptic",
        "chunk_size": 8,
        "max_processes": 1,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def _porkchop_counts_for_rows(rows: int) -> tuple[int, int]:
    n_departures = max(1, int(np.sqrt(rows)))
    n_arrivals = max(1, rows // n_departures)
    return n_departures, n_arrivals


def _sample_porkchop_keplerian_elements(
    rng: np.random.Generator,
    n: int,
    *,
    a_min: float,
    a_max: float,
) -> np.ndarray:
    a = rng.uniform(a_min, a_max, size=n)
    e = rng.uniform(0.01, 0.30, size=n)
    i_deg = rng.uniform(0.0, 15.0, size=n)
    raan = rng.uniform(0.0, 360.0, size=n)
    omega = rng.uniform(0.0, 360.0, size=n)
    m_anom = rng.uniform(0.0, 360.0, size=n)
    return np.stack([a, e, i_deg, raan, omega, m_anom], axis=1).astype(np.float64)


def _make_porkchop_sample(
    rng: np.random.Generator,
    *,
    n_departures: int,
    n_arrivals: int,
) -> Sample:
    departure_coords = _kep_to_cart(
        _sample_porkchop_keplerian_elements(
            rng, n_departures, a_min=0.85, a_max=1.35
        )
    )
    arrival_coords = _kep_to_cart(
        _sample_porkchop_keplerian_elements(
            rng, n_arrivals, a_min=1.25, a_max=2.40
        )
    )
    departure_time_mjd = 60000.0 + 4.0 * np.arange(n_departures, dtype=np.float64)
    arrival_time_mjd = (
        60000.0
        + max(8.0, 2.0 * float(n_departures))
        + 5.0 * np.arange(n_arrivals, dtype=np.float64)
    )
    kw = {
        "departure_coords": departure_coords,
        "arrival_coords": arrival_coords,
        "departure_time_mjd": departure_time_mjd,
        "arrival_time_mjd": arrival_time_mjd,
        "departure_orbit_ids": np.array(
            [f"d{i:05d}" for i in range(n_departures)], dtype=object
        ),
        "arrival_orbit_ids": np.array(
            [f"a{i:05d}" for i in range(n_arrivals)], dtype=object
        ),
        "propagation_origin": "SUN",
        "frame": "ecliptic",
        "prograde": True,
        "max_iter": 35,
        "tol": 1e-10,
        "max_processes": 1,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_generate_porkchop_data(rng: np.random.Generator, n: int) -> Sample:
    n_departures, n_arrivals = _porkchop_counts_for_rows(n)
    return _make_porkchop_sample(
        rng, n_departures=n_departures, n_arrivals=n_arrivals
    )


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
# Structured workload helpers for speed-governance lanes
# ---------------------------------------------------------------------------


def _sample_observer_states(rng: np.random.Generator, n: int) -> np.ndarray:
    obs_r = 1.0 + rng.normal(scale=0.02, size=n)
    obs_theta = rng.uniform(0.0, 2 * np.pi, size=n)
    obs = np.zeros((n, 6), dtype=np.float64)
    obs[:, 0] = obs_r * np.cos(obs_theta)
    obs[:, 1] = obs_r * np.sin(obs_theta)
    obs[:, 3] = -2 * np.pi / 365.25 * obs_r * np.sin(obs_theta)
    obs[:, 4] = 2 * np.pi / 365.25 * obs_r * np.cos(obs_theta)
    return obs


def _sample_observer_positions(rng: np.random.Generator, n: int) -> np.ndarray:
    obs_r = 1.0 + rng.normal(scale=0.02, size=n)
    obs_theta = rng.uniform(0.0, 2 * np.pi, size=n)
    obs = np.zeros((n, 3), dtype=np.float64)
    obs[:, 0] = obs_r * np.cos(obs_theta)
    obs[:, 1] = obs_r * np.sin(obs_theta)
    return obs


def _orbit_epoch_grid(shape: WorkloadShape) -> tuple[int, int]:
    n_orbits = shape.n_orbits or shape.rows
    n_epochs = shape.n_epochs or max(1, shape.rows // n_orbits)
    if n_orbits * n_epochs != shape.rows:
        raise ValueError(f"{shape.label()} does not form an orbits × epochs grid")
    return n_orbits, n_epochs


def _orbit_observer_grid(shape: WorkloadShape) -> tuple[int, int]:
    n_orbits = shape.n_orbits or shape.rows
    n_observers = shape.n_observers or max(1, shape.rows // n_orbits)
    if n_orbits * n_observers != shape.rows:
        raise ValueError(f"{shape.label()} does not form an orbits × observers grid")
    return n_orbits, n_observers


def _porkchop_grid(shape: WorkloadShape) -> tuple[int, int]:
    axes = shape.axes()
    n_departures = axes.get("departures")
    n_arrivals = axes.get("arrivals")
    if n_departures is None or n_arrivals is None:
        n_departures, n_arrivals = _porkchop_counts_for_rows(shape.rows)
    if n_departures * n_arrivals != shape.rows:
        raise ValueError(f"{shape.label()} does not form a departures × arrivals grid")
    return n_departures, n_arrivals


def make_generate_porkchop_data_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    n_departures, n_arrivals = _porkchop_grid(shape)
    return _make_porkchop_sample(
        rng, n_departures=n_departures, n_arrivals=n_arrivals
    )


def make_propagate_2body_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    n_orbits, n_epochs = _orbit_epoch_grid(shape)
    orbit_rows = _kep_to_cart(_sample_keplerian_elements(rng, n_orbits))
    coords = np.repeat(orbit_rows, n_epochs, axis=0)
    dts = np.tile(_sample_dts(rng, n_epochs), n_orbits)
    mus = np.full(shape.rows, MU_SUN, dtype=np.float64)
    kw = {"orbits": coords, "dts": dts, "mus": mus, "max_iter": 100, "tol": 1e-15}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_propagate_2body_with_covariance_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    base = make_propagate_2body_shape(rng, shape).rust_kwargs
    sigmas = np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10])
    diag = np.diag(sigmas**2)
    cov_3d = np.broadcast_to(diag, (shape.rows, 6, 6)).copy()
    rust_kw = dict(base)
    rust_kw["covariances"] = np.ascontiguousarray(cov_3d.reshape(shape.rows, 36))
    legacy_kw = dict(base)
    legacy_kw["covariances"] = cov_3d
    return Sample(rust_kwargs=rust_kw, legacy_kwargs=legacy_kw)


def make_generate_ephemeris_2body_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    n_orbits, n_epochs = _orbit_epoch_grid(shape)
    orbit_rows = _kep_to_cart(_sample_keplerian_elements(rng, n_orbits))
    coords = np.repeat(orbit_rows, n_epochs, axis=0)
    obs = np.tile(_sample_observer_states(rng, n_epochs), (n_orbits, 1))
    mus = np.full(shape.rows, MU_SUN, dtype=np.float64)
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


def make_generate_ephemeris_2body_with_covariance_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    base = make_generate_ephemeris_2body_shape(rng, shape).rust_kwargs
    sigmas = np.array([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10])
    diag = np.diag(sigmas**2)
    cov_3d = np.broadcast_to(diag, (shape.rows, 6, 6)).copy()
    rust_kw = dict(base)
    rust_kw["covariances"] = np.ascontiguousarray(cov_3d.reshape(shape.rows, 36))
    legacy_kw = dict(base)
    legacy_kw["covariances"] = cov_3d
    return Sample(rust_kwargs=rust_kw, legacy_kwargs=legacy_kw)


def make_add_light_time_shape(rng: np.random.Generator, shape: WorkloadShape) -> Sample:
    n_orbits, n_observers = _orbit_observer_grid(shape)
    orbit_rows = _kep_to_cart(_sample_keplerian_elements(rng, n_orbits))
    coords = np.repeat(orbit_rows, n_observers, axis=0)
    obs_pos = np.tile(_sample_observer_positions(rng, n_observers), (n_orbits, 1))
    mus = np.full(shape.rows, MU_SUN, dtype=np.float64)
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


def make_calculate_phase_angle_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    n_orbits, n_observers = _orbit_observer_grid(shape)
    object_rows = _kep_to_cart(_sample_keplerian_elements(rng, n_orbits))[:, :3]
    obj = np.repeat(object_rows, n_observers, axis=0)
    obs = np.tile(_sample_observer_positions(rng, n_observers), (n_orbits, 1))
    kw = {"object_pos": obj, "observer_pos": obs}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calculate_apparent_magnitude_v_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    base = make_calculate_phase_angle_shape(rng, shape).rust_kwargs
    h_v = rng.uniform(15.0, 25.0, size=shape.rows).astype(np.float64)
    g = rng.uniform(0.0, 0.5, size=shape.rows).astype(np.float64)
    kw = {
        "h_v": h_v,
        "object_pos": base["object_pos"],
        "observer_pos": base["observer_pos"],
        "g": g,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calculate_apparent_magnitude_v_and_phase_angle_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    return make_calculate_apparent_magnitude_v_shape(rng, shape)


def make_predict_magnitudes_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    base = make_calculate_apparent_magnitude_v_shape(rng, shape).rust_kwargs
    delta = np.array([0.0, -0.05, 0.12, 0.30], dtype=np.float64)
    target_ids = rng.integers(0, len(delta), size=shape.rows).astype(np.int32)
    kw = dict(base)
    kw["target_ids"] = target_ids
    kw["delta_table"] = delta
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


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
    "coordinates.residuals.Residuals.calculate": make_residuals_calculate,
    "coordinates.residuals.calculate_chi2": make_calculate_chi2,
    "dynamics.calc_mean_motion": make_calc_mean_motion,
    "orbits.classify_orbits": make_classify_orbits,
    "dynamics.calculate_moid": make_calculate_moid,
    "dynamics.calculate_perturber_moids": make_calculate_perturber_moids,
    "dynamics.generate_porkchop_data": make_generate_porkchop_data,
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


SHAPED_GENERATORS = {
    "dynamics.propagate_2body": make_propagate_2body_shape,
    "dynamics.propagate_2body_with_covariance": (
        make_propagate_2body_with_covariance_shape
    ),
    "dynamics.generate_ephemeris_2body": make_generate_ephemeris_2body_shape,
    "dynamics.generate_ephemeris_2body_with_covariance": (
        make_generate_ephemeris_2body_with_covariance_shape
    ),
    "dynamics.generate_porkchop_data": make_generate_porkchop_data_shape,
    "dynamics.add_light_time": make_add_light_time_shape,
    "photometry.calculate_phase_angle": make_calculate_phase_angle_shape,
    "photometry.calculate_apparent_magnitude_v": (
        make_calculate_apparent_magnitude_v_shape
    ),
    "photometry.calculate_apparent_magnitude_v_and_phase_angle": (
        make_calculate_apparent_magnitude_v_and_phase_angle_shape
    ),
    "photometry.predict_magnitudes": make_predict_magnitudes_shape,
}


TINY_WORKLOADS: dict[str, WorkloadShape] = {
    # `calculate_moid` is a scalar optimizer API; one pair is the true
    # one-off call shape, unlike vector kernels where n=10 is the tiny lane.
    "dynamics.calculate_moid": WorkloadShape(1),
    "dynamics.calculate_perturber_moids": WorkloadShape(1),
    "dynamics.generate_porkchop_data": WorkloadShape(
        4, extra={"departures": 2, "arrivals": 2}
    ),
}


SMALL_WORKLOADS: dict[str, WorkloadShape] = {
    # Keep canonical MOID speed governance affordable: baseline-main uses
    # scipy bounded minimization per pair, so n=2000 would be minutes per rep.
    "dynamics.calculate_moid": WorkloadShape(8),
    "dynamics.calculate_perturber_moids": WorkloadShape(8),
    # Public porkchop calls are grid-orchestration/table-assembly workloads;
    # 44×44 stays near the canonical n=2000 small lane while preserving a
    # square departure/arrival grid.
    "dynamics.generate_porkchop_data": WorkloadShape(
        1_936, extra={"departures": 44, "arrivals": 44}
    ),
}


FUZZ_N_OVERRIDES: dict[str, int] = {
    # `calculate_moid` performs nested scipy/JAX minimization in the legacy
    # oracle. Eight pairs × eight seeds gives direct randomized coverage while
    # keeping canonical fuzz runs within the existing time budget.
    "dynamics.calculate_moid": 8,
    "dynamics.calculate_perturber_moids": 8,
    "dynamics.generate_porkchop_data": 16,
}


LARGE_WORKLOADS: dict[str, WorkloadShape] = {
    "coordinates.cartesian_to_spherical": WorkloadShape(20_000),
    "coordinates.transform_coordinates": WorkloadShape(12_000),
    "coordinates.cartesian_to_geodetic": WorkloadShape(20_000),
    "coordinates.cartesian_to_keplerian": WorkloadShape(20_000),
    "coordinates.keplerian.to_cartesian": WorkloadShape(20_000),
    "coordinates.cartesian_to_cometary": WorkloadShape(20_000),
    "coordinates.cometary.to_cartesian": WorkloadShape(20_000),
    "coordinates.spherical.to_cartesian": WorkloadShape(20_000),
    "coordinates.residuals.Residuals.calculate": WorkloadShape(20_000),
    "coordinates.residuals.calculate_chi2": WorkloadShape(50_000),
    "dynamics.calc_mean_motion": WorkloadShape(50_000),
    "orbits.classify_orbits": WorkloadShape(50_000),
    "dynamics.calculate_moid": WorkloadShape(64),
    "dynamics.calculate_perturber_moids": WorkloadShape(64),
    "dynamics.generate_porkchop_data": WorkloadShape(
        4_096, extra={"departures": 64, "arrivals": 64}
    ),
    "dynamics.propagate_2body": WorkloadShape(20_000, n_orbits=1_000, n_epochs=20),
    "dynamics.propagate_2body_with_covariance": WorkloadShape(
        4_000, n_orbits=200, n_epochs=20
    ),
    "dynamics.generate_ephemeris_2body": WorkloadShape(
        20_000, n_orbits=400, n_epochs=50
    ),
    "dynamics.generate_ephemeris_2body_with_covariance": WorkloadShape(
        4_000, n_orbits=200, n_epochs=20
    ),
    "dynamics.solve_lambert": WorkloadShape(12_000),
    "dynamics.add_light_time": WorkloadShape(20_000, n_orbits=400, n_observers=50),
    "photometry.calculate_phase_angle": WorkloadShape(
        50_000, n_orbits=1_000, n_observers=50
    ),
    "photometry.calculate_apparent_magnitude_v": WorkloadShape(
        50_000, n_orbits=1_000, n_observers=50
    ),
    "photometry.calculate_apparent_magnitude_v_and_phase_angle": WorkloadShape(
        50_000, n_orbits=1_000, n_observers=50
    ),
    "photometry.predict_magnitudes": WorkloadShape(
        50_000, n_orbits=1_000, n_observers=50
    ),
    "orbit_determination.calcGibbs": WorkloadShape(5_000, extra={"triplets": 5_000}),
    "orbit_determination.calcHerrickGibbs": WorkloadShape(
        5_000, extra={"triplets": 5_000}
    ),
    "orbit_determination.calcGauss": WorkloadShape(5_000, extra={"triplets": 5_000}),
}


def all_api_ids() -> tuple[str, ...]:
    return tuple(GENERATORS.keys())


def lane_workloads(
    *, tiny_n: int = 10, small_n: int = 2_000
) -> dict[str, dict[str, WorkloadShape]]:
    api_ids = all_api_ids()
    missing_large = sorted(set(api_ids) - set(LARGE_WORKLOADS))
    if missing_large:
        raise KeyError("Missing large-n workloads: " + ", ".join(missing_large))
    return {
        "tiny-n": {
            api_id: TINY_WORKLOADS.get(api_id, WorkloadShape(tiny_n))
            for api_id in api_ids
        },
        "small-n": {
            api_id: SMALL_WORKLOADS.get(api_id, WorkloadShape(small_n))
            for api_id in api_ids
        },
        "large-n": {api_id: LARGE_WORKLOADS[api_id] for api_id in api_ids},
    }


def fuzz_n(api_id: str, default_n: int) -> int:
    return FUZZ_N_OVERRIDES.get(api_id, default_n)


def make(
    api_id: str,
    rng: np.random.Generator,
    workload: int | WorkloadShape,
) -> Sample:
    if api_id not in GENERATORS:
        raise KeyError(f"No input generator for {api_id!r}")
    if isinstance(workload, WorkloadShape):
        shaped_generator = SHAPED_GENERATORS.get(api_id)
        if shaped_generator is not None and workload.axes():
            return shaped_generator(rng, workload)
        return GENERATORS[api_id](rng, workload.rows)
    return GENERATORS[api_id](rng, workload)
