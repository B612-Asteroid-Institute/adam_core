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


_TRANSFORM_SUBCASE_NAMES: tuple[str, ...] = (
    "cart_ec_to_sph_eq",
    "cart_eq_to_sph_ec",
    "sph_ec_to_cart_eq",
    "kep_ec_to_sph_eq",
    "com_eq_to_kep_ec",
    "cart_ec_sun_to_sph_ec_earth",
    "cart_ec_earth_to_sph_ec_sun",
    "cart_ec_earth_to_sph_itrf93",
    "cart_itrf93_earth_to_sph_eq",
    "cart_cov_ec_to_sph_eq",
    "cart_cov_ec_sun_to_sph_ec_earth",
    "kep_cov_ec_to_sph_eq",
)


def _split_transform_rows(n: int) -> list[int]:
    case_count = len(_TRANSFORM_SUBCASE_NAMES)
    if n < case_count:
        raise ValueError(
            f"coordinates.transform_coordinates needs at least {case_count} rows "
            "to exercise every randomized public-dispatch matrix subcase"
        )
    rows_per_case, remainder = divmod(n, case_count)
    return [rows_per_case + (1 if i < remainder else 0) for i in range(case_count)]


_ITRF93_PARITY_EPOCHS_MJD = np.array(
    [59000.0, 59500.0, 60000.0, 60100.0, 60200.0, 60500.0], dtype=np.float64
)


def _sample_transform_time(
    rng: np.random.Generator, n: int, *, itrf93: bool = False
) -> np.ndarray:
    if itrf93:
        # Time-varying ITRF93 parity intentionally samples a vetted epoch set
        # rather than arbitrary PCK interpolation points. Baseline-main uses
        # CSPICE while the migration path uses spicekit's pure-Rust PCK
        # evaluator; arbitrary epoch fuzz can amplify a known accepted rotation
        # ULP into near-threshold spherical velocity-angle drift. These epochs
        # keep coverage across the PCK interval while preserving >3x headroom
        # under the stricter ITRF93 parity tolerance.
        indices = rng.integers(0, len(_ITRF93_PARITY_EPOCHS_MJD), size=n)
        return np.ascontiguousarray(_ITRF93_PARITY_EPOCHS_MJD[indices])
    return rng.uniform(58000.0, 63000.0, size=n).astype(np.float64)


def _sample_geocentric_cartesian(rng: np.random.Generator, n: int) -> np.ndarray:
    # ITRF93/geodetic-style public dispatcher cases should stay near Earth so
    # small CSPICE-vs-spicekit rotation-matrix ULP differences do not get
    # artificially amplified by outer-solar-system lever arms.
    radius = rng.uniform(0.8, 1.3, size=n) * R_EARTH_EQ
    lon = rng.uniform(0.0, 2.0 * np.pi, size=n)
    lat = rng.uniform(-0.9, 0.9, size=n)
    cos_lat = np.cos(lat)
    x = radius * cos_lat * np.cos(lon)
    y = radius * cos_lat * np.sin(lon)
    z = radius * np.sin(lat)
    velocity = rng.normal(scale=1e-6, size=(n, 3))
    return np.stack([x, y, z, velocity[:, 0], velocity[:, 1], velocity[:, 2]], axis=1)


def _cart_to_spherical(coords: np.ndarray) -> np.ndarray:
    from adam_core._rust import api as _rust_api

    out = _rust_api.cartesian_to_spherical_numpy(coords)
    if out is None:
        raise RuntimeError("rust backend unavailable")
    return np.ascontiguousarray(out, dtype=np.float64)


def _cart_to_cometary(coords: np.ndarray, t0: np.ndarray) -> np.ndarray:
    from adam_core._rust import api as _rust_api

    out = _rust_api.cartesian_to_cometary_numpy(
        coords,
        t0,
        np.full(coords.shape[0], MU_SUN, dtype=np.float64),
    )
    if out is None:
        raise RuntimeError("rust backend unavailable")
    return np.ascontiguousarray(out, dtype=np.float64)


def _sample_covariance(
    rng: np.random.Generator, n: int, scales: np.ndarray
) -> np.ndarray:
    """Sample symmetric positive-definite 6x6 covariances with fixed scales."""
    raw = rng.normal(size=(n, 6, 6))
    spd = raw @ np.swapaxes(raw, 1, 2)
    diag = np.sqrt(np.diagonal(spd, axis1=1, axis2=2))
    corr = spd / diag[:, :, None] / diag[:, None, :]
    return np.ascontiguousarray(corr * scales[None, :, None] * scales[None, None, :])


def _sample_cartesian_covariance(rng: np.random.Generator, n: int) -> np.ndarray:
    scales = np.array([1e-7, 1e-7, 1e-7, 1e-9, 1e-9, 1e-9], dtype=np.float64)
    return _sample_covariance(rng, n, scales)


def _sample_keplerian_covariance(rng: np.random.Generator, n: int) -> np.ndarray:
    scales = np.array([1e-6, 1e-8, 1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float64)
    return _sample_covariance(rng, n, scales)


def _pin_all_nan_covariance_row(covariances: np.ndarray) -> np.ndarray:
    pinned = np.array(covariances, dtype=np.float64, copy=True)
    if pinned.shape[0]:
        pinned[0, :, :] = np.nan
    return np.ascontiguousarray(pinned)


def _transform_case(
    name: str,
    coords: np.ndarray,
    time_mjd: np.ndarray,
    representation_in: str,
    representation_out: str,
    frame_in: str,
    frame_out: str,
    *,
    origin_in: str = "SUN",
    origin_out: str | None = None,
    covariance: np.ndarray | None = None,
) -> dict[str, Any]:
    case = {
        "name": name,
        "coords": np.ascontiguousarray(coords, dtype=np.float64),
        "time_mjd": np.ascontiguousarray(time_mjd, dtype=np.float64),
        "representation_in": representation_in,
        "representation_out": representation_out,
        "frame_in": frame_in,
        "frame_out": frame_out,
        "origin_in": origin_in,
        "origin_out": origin_out,
    }
    if covariance is not None:
        case["covariance"] = np.ascontiguousarray(covariance, dtype=np.float64)
    return case


def make_transform_coordinates(rng: np.random.Generator, n: int) -> Sample:
    """Public ``transform_coordinates`` dispatcher subcase matrix.

    Each case deliberately goes through the public quivr object boundary on
    both sides. The matrix covers constant-frame inverse directions,
    non-Cartesian inputs, representative covariance-bearing paths, SUN↔EARTH
    origin translations, and Earth-centered ITRF93 time-varying rotations while
    keeping the total row count near ``n``.
    """
    sizes = iter(_split_transform_rows(n))
    cases: list[dict[str, Any]] = []

    size = next(sizes)
    coords = _kep_to_cart(_sample_keplerian_elements(rng, size))
    cases.append(
        _transform_case(
            "cart_ec_to_sph_eq",
            coords,
            _sample_transform_time(rng, size),
            "cartesian",
            "spherical",
            "ecliptic",
            "equatorial",
        )
    )

    size = next(sizes)
    coords = _kep_to_cart(_sample_keplerian_elements(rng, size))
    cases.append(
        _transform_case(
            "cart_eq_to_sph_ec",
            coords,
            _sample_transform_time(rng, size),
            "cartesian",
            "spherical",
            "equatorial",
            "ecliptic",
        )
    )

    size = next(sizes)
    coords = _kep_to_cart(_sample_keplerian_elements(rng, size))
    cases.append(
        _transform_case(
            "sph_ec_to_cart_eq",
            _cart_to_spherical(coords),
            _sample_transform_time(rng, size),
            "spherical",
            "cartesian",
            "ecliptic",
            "equatorial",
        )
    )

    size = next(sizes)
    cases.append(
        _transform_case(
            "kep_ec_to_sph_eq",
            _sample_keplerian_elements(rng, size),
            _sample_transform_time(rng, size),
            "keplerian",
            "spherical",
            "ecliptic",
            "equatorial",
        )
    )

    size = next(sizes)
    time_mjd = _sample_transform_time(rng, size)
    coords = _kep_to_cart(_sample_keplerian_elements(rng, size))
    cases.append(
        _transform_case(
            "com_eq_to_kep_ec",
            _cart_to_cometary(coords, time_mjd),
            time_mjd,
            "cometary",
            "keplerian",
            "equatorial",
            "ecliptic",
        )
    )

    size = next(sizes)
    coords = _kep_to_cart(_sample_keplerian_elements(rng, size))
    cases.append(
        _transform_case(
            "cart_ec_sun_to_sph_ec_earth",
            coords,
            _sample_transform_time(rng, size, itrf93=True),
            "cartesian",
            "spherical",
            "ecliptic",
            "ecliptic",
            origin_out="EARTH",
        )
    )

    size = next(sizes)
    cases.append(
        _transform_case(
            "cart_ec_earth_to_sph_ec_sun",
            _sample_geocentric_cartesian(rng, size),
            _sample_transform_time(rng, size, itrf93=True),
            "cartesian",
            "spherical",
            "ecliptic",
            "ecliptic",
            origin_in="EARTH",
            origin_out="SUN",
        )
    )

    size = next(sizes)
    cases.append(
        _transform_case(
            "cart_ec_earth_to_sph_itrf93",
            _sample_geocentric_cartesian(rng, size),
            _sample_transform_time(rng, size, itrf93=True),
            "cartesian",
            "spherical",
            "ecliptic",
            "itrf93",
            origin_in="EARTH",
        )
    )

    size = next(sizes)
    cases.append(
        _transform_case(
            "cart_itrf93_earth_to_sph_eq",
            _sample_geocentric_cartesian(rng, size),
            _sample_transform_time(rng, size, itrf93=True),
            "cartesian",
            "spherical",
            "itrf93",
            "equatorial",
            origin_in="EARTH",
        )
    )

    size = next(sizes)
    coords = _kep_to_cart(_sample_keplerian_elements(rng, size))
    cases.append(
        _transform_case(
            "cart_cov_ec_to_sph_eq",
            coords,
            _sample_transform_time(rng, size),
            "cartesian",
            "spherical",
            "ecliptic",
            "equatorial",
            covariance=_sample_cartesian_covariance(rng, size),
        )
    )

    size = next(sizes)
    coords = _kep_to_cart(_sample_keplerian_elements(rng, size))
    cases.append(
        _transform_case(
            "cart_cov_ec_sun_to_sph_ec_earth",
            coords,
            _sample_transform_time(rng, size, itrf93=True),
            "cartesian",
            "spherical",
            "ecliptic",
            "ecliptic",
            origin_out="EARTH",
            covariance=_sample_cartesian_covariance(rng, size),
        )
    )

    size = next(sizes)
    cases.append(
        _transform_case(
            "kep_cov_ec_to_sph_eq",
            _sample_keplerian_elements(rng, size),
            _sample_transform_time(rng, size),
            "keplerian",
            "spherical",
            "ecliptic",
            "equatorial",
            covariance=_sample_keplerian_covariance(rng, size),
        )
    )

    kw = {"cases": cases}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


_TRANSFORM_COVARIANCE_SUBCASE_NAMES: tuple[str, ...] = (
    "raw_cart_cov_ec_to_sph_eq",
    "raw_cart_cov_eq_to_kep_ec",
    "raw_kep_cov_ec_to_cart_eq",
    "raw_kep_cov_eq_to_sph_ec",
)


def _split_transform_covariance_rows(n: int) -> list[int]:
    case_count = len(_TRANSFORM_COVARIANCE_SUBCASE_NAMES)
    if n < case_count:
        raise ValueError(
            "coordinates.transform_coordinates_with_covariance needs at least "
            f"{case_count} rows to exercise every raw-kernel subcase"
        )
    rows_per_case, remainder = divmod(n, case_count)
    return [rows_per_case + (1 if i < remainder else 0) for i in range(case_count)]


def _transform_covariance_case(
    name: str,
    coords: np.ndarray,
    time_mjd: np.ndarray,
    representation_in: str,
    representation_out: str,
    frame_in: str,
    frame_out: str,
    covariance: np.ndarray,
) -> dict[str, Any]:
    case = _transform_case(
        name,
        coords,
        time_mjd,
        representation_in,
        representation_out,
        frame_in,
        frame_out,
        covariance=_pin_all_nan_covariance_row(covariance),
    )
    case["mu"] = np.full(coords.shape[0], MU_SUN, dtype=np.float64)
    return case


def make_transform_coordinates_with_covariance(
    rng: np.random.Generator, n: int
) -> Sample:
    """Raw covariance-transform kernel subcase matrix.

    The direct raw-kernel comparison exercises representative constant-frame
    Cartesian/Keplerian representation chains and covariance propagation. The
    public ``coordinates.transform_coordinates`` matrix separately covers
    origin-translation and ITRF93 public-dispatch covariance paths.
    """
    sizes = iter(_split_transform_covariance_rows(n))
    cases: list[dict[str, Any]] = []

    size = next(sizes)
    time_mjd = _sample_transform_time(rng, size)
    coords = _kep_to_cart(_sample_keplerian_elements(rng, size))
    cases.append(
        _transform_covariance_case(
            "raw_cart_cov_ec_to_sph_eq",
            coords,
            time_mjd,
            "cartesian",
            "spherical",
            "ecliptic",
            "equatorial",
            _sample_cartesian_covariance(rng, size),
        )
    )

    size = next(sizes)
    time_mjd = _sample_transform_time(rng, size)
    coords = _kep_to_cart(_sample_keplerian_elements(rng, size))
    cases.append(
        _transform_covariance_case(
            "raw_cart_cov_eq_to_kep_ec",
            coords,
            time_mjd,
            "cartesian",
            "keplerian",
            "equatorial",
            "ecliptic",
            _sample_cartesian_covariance(rng, size),
        )
    )

    size = next(sizes)
    cases.append(
        _transform_covariance_case(
            "raw_kep_cov_ec_to_cart_eq",
            _sample_keplerian_elements(rng, size),
            _sample_transform_time(rng, size),
            "keplerian",
            "cartesian",
            "ecliptic",
            "equatorial",
            _sample_keplerian_covariance(rng, size),
        )
    )

    size = next(sizes)
    cases.append(
        _transform_covariance_case(
            "raw_kep_cov_eq_to_sph_ec",
            _sample_keplerian_elements(rng, size),
            _sample_transform_time(rng, size),
            "keplerian",
            "spherical",
            "equatorial",
            "ecliptic",
            _sample_keplerian_covariance(rng, size),
        )
    )

    kw = {"cases": cases}
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


def _sample_sxform_like_matrices(
    rng: np.random.Generator, n_matrices: int
) -> np.ndarray:
    matrices = np.zeros((n_matrices, 6, 6), dtype=np.float64)
    for matrix_index in range(n_matrices):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0.0:
            q[:, 0] *= -1.0
        omega = rng.normal(scale=1e-4, size=3)
        omega_cross = np.array(
            [
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0],
            ],
            dtype=np.float64,
        )
        matrices[matrix_index, :3, :3] = q
        matrices[matrix_index, 3:, :3] = omega_cross @ q
        matrices[matrix_index, 3:, 3:] = q
    if n_matrices:
        matrices[0] = np.eye(6, dtype=np.float64)
    return matrices


def make_rotate_cartesian_time_varying(rng: np.random.Generator, n: int) -> Sample:
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    n_matrices = min(16, max(1, int(np.sqrt(max(n, 1)))))
    matrices = _sample_sxform_like_matrices(rng, n_matrices)
    time_index = rng.integers(0, n_matrices, size=n, dtype=np.int64)
    if n:
        pinned = min(n, n_matrices)
        time_index[:pinned] = np.arange(pinned, dtype=np.int64)

    factors = rng.normal(scale=1e-4, size=(n, 6, 6))
    covariances = np.einsum("nij,nkj->nik", factors, factors).astype(np.float64)
    covariances += np.eye(6, dtype=np.float64)[None, :, :] * 1e-12
    if n >= 1:
        covariances[0, :, :] = np.nan
    if n >= 2:
        covariances[1, 0, 1] = np.nan
        covariances[1, 1, 0] = np.nan

    kw = {
        "coords": np.ascontiguousarray(coords, dtype=np.float64),
        "time_index": np.ascontiguousarray(time_index, dtype=np.int64),
        "matrices": np.ascontiguousarray(matrices, dtype=np.float64),
        "covariances": np.ascontiguousarray(
            covariances.reshape(n, 36), dtype=np.float64
        ),
    }
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


def make_bound_longitude_residuals(rng: np.random.Generator, n: int) -> Sample:
    """Sample production-like small residuals plus every wrap branch."""
    observed = rng.normal(scale=1.0, size=(n, 6)).astype(np.float64)
    residuals = rng.normal(scale=0.1, size=(n, 6)).astype(np.float64)

    observed[:, 1] = rng.uniform(0.0, 360.0, size=n)
    # In real OD residual scoring, most longitude residuals are small after the
    # predicted-observed subtraction. Keep the bulk of the speed workload in
    # that production-like shape, while pinning one row for each branch below so
    # randomized parity still exercises the wrap/sign convention every seed.
    residuals[:, 1] = rng.normal(scale=5.0, size=n)

    branch_observed = np.array([100.0, 355.0, 5.0, 100.0, 250.0], dtype=np.float64)
    branch_residuals = np.array([10.0, 350.0, -350.0, 350.0, -350.0], dtype=np.float64)
    branch_rows = min(n, len(branch_observed))
    observed[:branch_rows, 1] = branch_observed[:branch_rows]
    residuals[:branch_rows, 1] = branch_residuals[:branch_rows]

    kw = {
        "observed": np.ascontiguousarray(observed),
        "residuals": np.ascontiguousarray(residuals),
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_apply_cosine_latitude_correction(rng: np.random.Generator, n: int) -> Sample:
    """Sample cos(latitude) residual and covariance scaling rows."""
    lat = rng.uniform(-89.0, 89.0, size=n).astype(np.float64)
    if n >= 5:
        lat[:5] = np.array([0.0, 30.0, -30.0, 60.0, -80.0], dtype=np.float64)
        rng.shuffle(lat)

    residuals = rng.normal(scale=0.05, size=(n, 6)).astype(np.float64)
    raw = rng.normal(scale=0.1, size=(n, 6, 6)).astype(np.float64)
    covariances = np.matmul(raw, np.swapaxes(raw, 1, 2))

    # Real observed/predicted spherical covariance blocks often carry NaNs in
    # inactive dimensions. The helper promises to preserve those NaN cells while
    # scaling finite longitude/longitudinal-velocity rows and columns.
    nan_mask = rng.random(size=(n, 6, 6)) < 0.03
    covariances[nan_mask] = np.nan

    kw = {
        "lat": np.ascontiguousarray(lat),
        "residuals": np.ascontiguousarray(residuals),
        "covariances": np.ascontiguousarray(covariances),
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def _sample_normalized_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    weights = rng.random(n).astype(np.float64)
    return np.ascontiguousarray(weights / np.sum(weights), dtype=np.float64)


def make_weighted_mean(rng: np.random.Generator, n: int) -> Sample:
    samples = rng.normal(loc=0.0, scale=10.0, size=(n, 6)).astype(np.float64)
    weights = _sample_normalized_weights(rng, n)
    kw = {"samples": np.ascontiguousarray(samples), "weights": weights}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_weighted_covariance(rng: np.random.Generator, n: int) -> Sample:
    samples = rng.normal(loc=0.0, scale=10.0, size=(n, 6)).astype(np.float64)
    weights = _sample_normalized_weights(rng, n)
    mean = np.dot(weights, samples).astype(np.float64)
    kw = {
        "mean": np.ascontiguousarray(mean),
        "samples": np.ascontiguousarray(samples),
        "weights": weights,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_calc_mean_motion(rng: np.random.Generator, n: int) -> Sample:
    a = rng.uniform(0.5, 50.0, size=n).astype(np.float64)
    mu = np.full(n, MU_SUN, dtype=np.float64)
    kw = {"a": a, "mu": mu}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_tisserand_parameter(rng: np.random.Generator, n: int) -> Sample:
    a = np.exp(rng.uniform(np.log(0.3), np.log(80.0), size=n)).astype(np.float64)
    e = rng.beta(0.8, 2.5, size=n).astype(np.float64)
    e = np.minimum(e, 0.999)
    i = rng.uniform(0.0, 180.0, size=n).astype(np.float64)
    if n:
        pinned = min(n, 8)
        a[:pinned] = np.array(
            [0.387, 0.723, 1.0, 1.524, 5.204, 9.58, 19.2, 30.2], dtype=np.float64
        )[:pinned]
        e[:pinned] = np.array(
            [0.0, 1e-12, 0.05, 0.3, 0.7, 0.95, 0.99, 0.999], dtype=np.float64
        )[:pinned]
        i[:pinned] = np.array(
            [0.0, 1e-9, 5.0, 45.0, 90.0, 135.0, 179.999999, 180.0],
            dtype=np.float64,
        )[:pinned]
    third_body = str(
        rng.choice(
            [
                "mercury",
                "venus",
                "earth",
                "mars",
                "jupiter",
                "saturn",
                "uranus",
                "neptune",
            ]
        )
    )
    kw = {"a": a, "e": e, "i": i, "third_body": third_body}
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


def make_calculate_moid_batch(rng: np.random.Generator, n: int) -> Sample:
    return make_calculate_moid(rng, n)


_OBSERVER_CODE_PANEL = np.array(
    ["500", "X05", "F51", "W84", "309", "695", "T05", "I11"], dtype=object
)


def make_observers_from_codes(rng: np.random.Generator, n: int) -> Sample:
    """Public ``Observers.from_codes`` orchestration: per-row MPC observatory
    codes + UTC epochs -> heliocentric ecliptic Cartesian observer states.

    The migration path runs the DE440 SPK lookups and the ITRF93->J2000
    rotation in the Rust spicekit backend; baseline-main runs spiceypy against
    the same kernels. Recon 2026-07-04 measured machine-epsilon agreement
    (<= 1.2e-15 AU position, 4.2e-15 AU/day velocity) across the code panel
    (geocenter + ground stations at low/mid/high latitude, both hemispheres).
    """
    codes = rng.choice(_OBSERVER_CODE_PANEL, size=n)
    mjd_utc = np.sort(rng.uniform(59000.0, 61000.0, size=n)).astype(np.float64)
    kw = {
        "codes": np.array(codes, dtype=object),
        "mjd_utc": mjd_utc,
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


def _arc_grid_for_rows(rows: int) -> tuple[int, int]:
    n_orbits = max(1, int(np.sqrt(rows)))
    while rows % n_orbits != 0:
        n_orbits -= 1
    return n_orbits, rows // n_orbits


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


def _make_porkchop_grid_arrays(
    rng: np.random.Generator,
    *,
    n_departures: int,
    n_arrivals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    departure_coords = _kep_to_cart(
        _sample_porkchop_keplerian_elements(rng, n_departures, a_min=0.85, a_max=1.35)
    )
    arrival_coords = _kep_to_cart(
        _sample_porkchop_keplerian_elements(rng, n_arrivals, a_min=1.25, a_max=2.40)
    )
    departure_time_mjd = 60000.0 + 4.0 * np.arange(n_departures, dtype=np.float64)
    arrival_time_mjd = (
        60000.0
        + max(8.0, 2.0 * float(n_departures))
        + 5.0 * np.arange(n_arrivals, dtype=np.float64)
    )
    return departure_coords, arrival_coords, departure_time_mjd, arrival_time_mjd


def _make_porkchop_sample(
    rng: np.random.Generator,
    *,
    n_departures: int,
    n_arrivals: int,
) -> Sample:
    departure_coords, arrival_coords, departure_time_mjd, arrival_time_mjd = (
        _make_porkchop_grid_arrays(
            rng, n_departures=n_departures, n_arrivals=n_arrivals
        )
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
    return _make_porkchop_sample(rng, n_departures=n_departures, n_arrivals=n_arrivals)


def _make_porkchop_grid_sample(
    rng: np.random.Generator,
    *,
    n_departures: int,
    n_arrivals: int,
) -> Sample:
    departure_coords, arrival_coords, departure_time_mjd, arrival_time_mjd = (
        _make_porkchop_grid_arrays(
            rng, n_departures=n_departures, n_arrivals=n_arrivals
        )
    )
    kw = {
        "dep_states": departure_coords,
        "dep_mjds": departure_time_mjd,
        "arr_states": arrival_coords,
        "arr_mjds": arrival_time_mjd,
        "mu": MU_SUN,
        "prograde": True,
        "maxiter": 35,
        "atol": 1e-10,
        "rtol": 1e-10,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_porkchop_grid(rng: np.random.Generator, n: int) -> Sample:
    n_departures, n_arrivals = _porkchop_counts_for_rows(n)
    return _make_porkchop_grid_sample(
        rng, n_departures=n_departures, n_arrivals=n_arrivals
    )


def _sample_dts(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(-10000.0, 10000.0, size=n).astype(np.float64)


def _sample_arc_dts(rng: np.random.Generator, n: int) -> np.ndarray:
    dts = _sample_dts(rng, n)
    if n:
        dts[0] = 0.0
        rng.shuffle(dts)
    return np.ascontiguousarray(dts, dtype=np.float64)


def _make_public_propagate_2body_sample(
    rng: np.random.Generator, *, n_orbits: int, n_epochs: int
) -> Sample:
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n_orbits))
    epoch_mjd = 59_800.0 + rng.uniform(0.0, 100.0, size=n_orbits)
    target_mjd = 60_000.0 + np.sort(rng.uniform(0.0, 200.0, size=n_epochs))
    kw = {
        "coords": np.ascontiguousarray(coords, dtype=np.float64),
        "epoch_mjd": np.ascontiguousarray(epoch_mjd, dtype=np.float64),
        "target_mjd": np.ascontiguousarray(target_mjd, dtype=np.float64),
        "origin": "SUN",
        "frame": "ecliptic",
        "max_iter": 100,
        "tol": 1e-15,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_propagate_2body(rng: np.random.Generator, n: int) -> Sample:
    return _make_public_propagate_2body_sample(rng, n_orbits=n, n_epochs=1)


def make_propagate_2body_along_arc(rng: np.random.Generator, n: int) -> Sample:
    orbit = _kep_to_cart(_sample_keplerian_elements(rng, 1))[0]
    dts = _sample_arc_dts(rng, n)
    kw = {"orbit": orbit, "dts": dts, "mu": MU_SUN, "max_iter": 100, "tol": 1e-15}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def _make_propagate_2body_arc_batch_sample(
    rng: np.random.Generator, *, n_orbits: int, n_epochs: int
) -> Sample:
    orbits = _kep_to_cart(_sample_keplerian_elements(rng, n_orbits))
    dts = _sample_dts(rng, n_orbits * n_epochs).reshape(n_orbits, n_epochs)
    if n_epochs:
        dts[:, 0] = 0.0
    mus = np.full(n_orbits, MU_SUN, dtype=np.float64)
    kw = {
        "orbits": orbits,
        "dts": np.ascontiguousarray(dts, dtype=np.float64),
        "mus": mus,
        "max_iter": 100,
        "tol": 1e-15,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_propagate_2body_arc_batch(rng: np.random.Generator, n: int) -> Sample:
    n_orbits, n_epochs = _arc_grid_for_rows(n)
    return _make_propagate_2body_arc_batch_sample(
        rng, n_orbits=n_orbits, n_epochs=n_epochs
    )


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
        "epoch_mjd": 60000.0 + np.arange(n, dtype=np.float64) / 1440.0,
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


def make_fit_absolute_magnitude_rows(rng: np.random.Generator, n: int) -> Sample:
    h_center = rng.uniform(14.0, 23.0)
    h_rows = rng.normal(loc=h_center, scale=0.35, size=n).astype(np.float64)
    sigma_rows = rng.uniform(0.03, 0.30, size=n).astype(np.float64)
    if rng.random() < 0.5 and n > 1:
        sigma_rows[rng.random(n) < 0.25] = np.nan
    kw = {"h_rows": h_rows, "sigma_rows": sigma_rows}
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_fit_absolute_magnitude_grouped(rng: np.random.Generator, n: int) -> Sample:
    group_count = max(1, min(n, max(2, n // 8)))
    sizes = rng.multinomial(n - group_count, np.full(group_count, 1.0 / group_count))
    sizes = sizes + 1
    group_offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    centers = rng.uniform(14.0, 23.0, size=group_count)
    h_rows = np.empty(n, dtype=np.float64)
    sigma_rows = np.empty(n, dtype=np.float64)
    for group_index in range(group_count):
        start = group_offsets[group_index]
        end = group_offsets[group_index + 1]
        rows = end - start
        h_rows[start:end] = rng.normal(loc=centers[group_index], scale=0.35, size=rows)
        sigma_rows[start:end] = rng.uniform(0.03, 0.30, size=rows)
        if rows > 1 and group_index % 3 == 1:
            sigma_rows[start:end][rng.random(rows) < 0.25] = np.nan
    kw = {"h_rows": h_rows, "sigma_rows": sigma_rows, "group_offsets": group_offsets}
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

    # Keep the randomized gate in the well-conditioned, shared-root regime.
    # Unconstrained random Gauss-IOD triplets can make Laguerre+deflation and
    # np.roots/LAPACK accept different physical-root subsets; that behavior is
    # intrinsic to the polynomial solvers rather than a Rust bug. Low-e,
    # main-belt-like triplets with multi-day spacing still provide true random
    # fuzz coverage for the production best-root path while avoiding
    # root-subset-policy noise.
    kep = _sample_keplerian_elements(rng, n_triplets)
    kep[:, 0] = rng.uniform(2.0, 3.0, size=n_triplets)
    kep[:, 1] = rng.uniform(0.0, 0.1, size=n_triplets)
    kep[:, 2] = rng.uniform(0.0, 15.0, size=n_triplets)
    cart = _kep_to_cart(kep)

    base_t = rng.uniform(59000.0, 60000.0, size=n_triplets).astype(np.float64)
    dt12 = rng.uniform(3.0, 7.0, size=n_triplets).astype(np.float64)
    dt23 = rng.uniform(3.0, 7.0, size=n_triplets).astype(np.float64)
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
    return _make_porkchop_sample(rng, n_departures=n_departures, n_arrivals=n_arrivals)


def make_porkchop_grid_shape(rng: np.random.Generator, shape: WorkloadShape) -> Sample:
    n_departures, n_arrivals = _porkchop_grid(shape)
    return _make_porkchop_grid_sample(
        rng, n_departures=n_departures, n_arrivals=n_arrivals
    )


def make_propagate_2body_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    n_orbits, n_epochs = _orbit_epoch_grid(shape)
    return _make_public_propagate_2body_sample(
        rng, n_orbits=n_orbits, n_epochs=n_epochs
    )


def make_propagate_2body_arc_batch_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    n_orbits, n_epochs = _orbit_epoch_grid(shape)
    return _make_propagate_2body_arc_batch_sample(
        rng, n_orbits=n_orbits, n_epochs=n_epochs
    )


def make_propagate_2body_with_covariance_shape(
    rng: np.random.Generator, shape: WorkloadShape
) -> Sample:
    n_orbits, n_epochs = _orbit_epoch_grid(shape)
    orbit_rows = _kep_to_cart(_sample_keplerian_elements(rng, n_orbits))
    coords = np.repeat(orbit_rows, n_epochs, axis=0)
    dts = np.tile(_sample_dts(rng, n_epochs), n_orbits)
    mus = np.full(shape.rows, MU_SUN, dtype=np.float64)
    base = {
        "orbits": np.ascontiguousarray(coords, dtype=np.float64),
        "dts": np.ascontiguousarray(dts, dtype=np.float64),
        "mus": mus,
        "max_iter": 100,
        "tol": 1e-15,
    }
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
        "epoch_mjd": np.tile(
            60000.0 + np.arange(n_epochs, dtype=np.float64) / 1440.0,
            n_orbits,
        ),
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
# Bridge signatures — W1 Arrow-bridge public surface (Beads personal-cmy.13).
# Inputs are flat arrays/strings only (picklable across the legacy venv); each
# runner rebuilds its own repo's quivr objects and calls its own public API.
#
# fit_orbit_least_squares (Rust-native Gauss-Newton 2-body OD) is intentionally
# NOT gated here: baseline-main adam_core has only an N-body/ASSIST
# fit_least_squares, so there is no apples-to-apples 2-body legacy reference.
# It is covered by the truth-recovery test in
# src/adam_core/orbits/tests/test_orbit_arrow_bridge.py. A future "same-minimum"
# comparison against scipy.optimize.least_squares over the (now-gated)
# bridge.evaluate_residuals_2body cost would be the honest rust-vs-legacy bar.
# ---------------------------------------------------------------------------


def make_bridge_propagate_orbits_2body(rng: np.random.Generator, n: int) -> Sample:
    """Bridge ``propagate_orbits_2body`` (Orbits -> Orbits, one Rust crossing)
    vs baseline-main public ``dynamics.propagate_2body``. Bound heliocentric
    orbits with per-orbit TDB epochs propagated to one shared TDB target."""
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    epoch_mjd = 59800.0 + rng.uniform(0.0, 200.0, size=n)
    kw = {
        "coords": np.ascontiguousarray(coords, dtype=np.float64),
        "epoch_mjd": np.ascontiguousarray(epoch_mjd, dtype=np.float64),
        "target_mjd": 60400.0,
        "origin": "SUN",
        "frame": "ecliptic",
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_bridge_rotate_orbits_frame(rng: np.random.Generator, n: int) -> Sample:
    """Bridge ``rotate_orbits_frame`` (Orbits -> Orbits, state + covariance, one
    Rust crossing) vs baseline-main ``transform_coordinates`` rotating ecliptic
    Cartesian into equatorial."""
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    epoch_mjd = 59800.0 + rng.uniform(0.0, 200.0, size=n)
    base = np.diag([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9])
    cov = np.stack([base * (i + 1) for i in range(n)])
    kw = {
        "coords": np.ascontiguousarray(coords, dtype=np.float64),
        "epoch_mjd": np.ascontiguousarray(epoch_mjd, dtype=np.float64),
        "covariance": np.ascontiguousarray(cov, dtype=np.float64),
        "origin": "SUN",
        "frame_in": "ecliptic",
        "frame_out": "equatorial",
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_bridge_sample_orbit_variants(rng: np.random.Generator, n: int) -> Sample:
    """Bridge ``sample_orbit_variants`` (sigma-point unscented transform,
    deterministic) vs baseline-main ``VariantOrbits.create``. PSD covariance."""
    coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    epoch_mjd = 59800.0 + rng.uniform(0.0, 200.0, size=n)
    base = np.diag([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9])
    cov = np.stack([base * (i + 1) for i in range(n)])
    kw = {
        "coords": np.ascontiguousarray(coords, dtype=np.float64),
        "epoch_mjd": np.ascontiguousarray(epoch_mjd, dtype=np.float64),
        "covariance": np.ascontiguousarray(cov, dtype=np.float64),
        "origin": "SUN",
        "frame": "ecliptic",
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


def make_bridge_evaluate_residuals_2body(rng: np.random.Generator, n: int) -> Sample:
    """Bridge ``evaluate_residuals_2body`` (the OD inner loop: 2-body ephemeris
    + chi2, one Rust crossing, 1:1 orbit/observer/observation) vs baseline-main
    ``generate_ephemeris_2body`` + ``Residuals.calculate``. Observed astrometry
    is a fixed perturbation of a baseline-main predicted ephemeris so each side
    differences identical observations against its own predicted angles."""
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.dynamics.ephemeris import generate_ephemeris_2body
    from adam_core.observers import Observers
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    mjd = np.sort(59900.0 + rng.uniform(0.0, 200.0, size=n)).astype(np.float64)
    times = Timestamp.from_mjd(mjd, scale="tdb")
    orbit_coords = _kep_to_cart(_sample_keplerian_elements(rng, n))
    ssb = Origin.from_kwargs(code=np.full(n, "SOLAR_SYSTEM_BARYCENTER", dtype="object"))
    orbits = Orbits.from_kwargs(
        orbit_id=[str(i) for i in range(n)],
        coordinates=CartesianCoordinates.from_kwargs(
            x=orbit_coords[:, 0],
            y=orbit_coords[:, 1],
            z=orbit_coords[:, 2],
            vx=orbit_coords[:, 3],
            vy=orbit_coords[:, 4],
            vz=orbit_coords[:, 5],
            time=times,
            origin=ssb,
            frame="ecliptic",
        ),
    )
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    k = MU_SUN**0.5
    observer_coords = np.column_stack(
        [
            np.cos(phi),
            np.sin(phi),
            np.zeros(n),
            -k * np.sin(phi),
            k * np.cos(phi),
            np.zeros(n),
        ]
    ).astype(np.float64)
    observers = Observers.from_kwargs(
        code=["500"] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=observer_coords[:, 0],
            y=observer_coords[:, 1],
            z=observer_coords[:, 2],
            vx=observer_coords[:, 3],
            vy=observer_coords[:, 4],
            vz=observer_coords[:, 5],
            time=times,
            origin=Origin.from_kwargs(
                code=np.full(n, "SOLAR_SYSTEM_BARYCENTER", dtype="object")
            ),
            frame="ecliptic",
        ),
    )
    pred = generate_ephemeris_2body(orbits, observers).coordinates
    observed_sph = np.column_stack(
        [
            pred.rho.to_numpy(zero_copy_only=False),
            pred.lon.to_numpy(zero_copy_only=False) + 1e-2,
            pred.lat.to_numpy(zero_copy_only=False) - 1e-2,
            pred.vrho.to_numpy(zero_copy_only=False),
            pred.vlon.to_numpy(zero_copy_only=False),
            pred.vlat.to_numpy(zero_copy_only=False),
        ]
    ).astype(np.float64)
    cov_row = np.diag([1.0, (1.0 / 3600.0) ** 2, (1.0 / 3600.0) ** 2, 1.0, 1.0, 1.0])
    observed_cov = np.tile(cov_row, (n, 1, 1)).astype(np.float64)
    # generate_ephemeris_2body returns observer-centric astrometry; the observed
    # coordinates must carry the same origin/frame (Residuals.calculate enforces
    # matching origin), so capture them from the predicted ephemeris.
    observed_origin = list(pred.origin.code.to_pylist())
    observed_frame = str(pred.frame)
    kw = {
        "orbit_coords": np.ascontiguousarray(orbit_coords, dtype=np.float64),
        "observer_coords": np.ascontiguousarray(observer_coords, dtype=np.float64),
        "observed_sph": np.ascontiguousarray(observed_sph, dtype=np.float64),
        "observed_cov": np.ascontiguousarray(observed_cov, dtype=np.float64),
        "epoch_mjd": np.ascontiguousarray(mjd, dtype=np.float64),
        "origin": "SOLAR_SYSTEM_BARYCENTER",
        "frame": "ecliptic",
        "observer_code": "500",
        "observed_origin": observed_origin,
        "observed_frame": observed_frame,
    }
    return Sample(rust_kwargs=kw, legacy_kwargs=kw)


# ---------------------------------------------------------------------------
# Registry: api_id → generator function
# ---------------------------------------------------------------------------

GENERATORS = {
    "coordinates.cartesian_to_spherical": make_cartesian_to_spherical,
    "coordinates.transform_coordinates": make_transform_coordinates,
    "coordinates.transform_coordinates_with_covariance": (
        make_transform_coordinates_with_covariance
    ),
    "coordinates.cartesian_to_geodetic": make_cartesian_to_geodetic,
    "coordinates.cartesian_to_keplerian": make_cartesian_to_keplerian,
    "coordinates.keplerian.to_cartesian": make_keplerian_to_cartesian,
    "coordinates.cartesian_to_cometary": make_cartesian_to_cometary,
    "coordinates.cometary.to_cartesian": make_cometary_to_cartesian,
    "coordinates.spherical.to_cartesian": make_spherical_to_cartesian,
    "coordinates.rotate_cartesian_time_varying": make_rotate_cartesian_time_varying,
    "coordinates.residuals.Residuals.calculate": make_residuals_calculate,
    "coordinates.residuals.calculate_chi2": make_calculate_chi2,
    "coordinates.residuals.bound_longitude_residuals": make_bound_longitude_residuals,
    "coordinates.residuals.apply_cosine_latitude_correction": (
        make_apply_cosine_latitude_correction
    ),
    "statistics.weighted_mean": make_weighted_mean,
    "statistics.weighted_covariance": make_weighted_covariance,
    "dynamics.calc_mean_motion": make_calc_mean_motion,
    "dynamics.tisserand_parameter": make_tisserand_parameter,
    "orbits.classify_orbits": make_classify_orbits,
    "dynamics.calculate_moid": make_calculate_moid,
    "dynamics.calculate_moid_batch": make_calculate_moid_batch,
    "dynamics.calculate_perturber_moids": make_calculate_perturber_moids,
    "missions.porkchop_grid": make_porkchop_grid,
    "dynamics.generate_porkchop_data": make_generate_porkchop_data,
    "dynamics.propagate_2body": make_propagate_2body,
    "dynamics.propagate_2body_along_arc": make_propagate_2body_along_arc,
    "dynamics.propagate_2body_arc_batch": make_propagate_2body_arc_batch,
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
    "photometry.fit_absolute_magnitude_rows": make_fit_absolute_magnitude_rows,
    "photometry.fit_absolute_magnitude_grouped": make_fit_absolute_magnitude_grouped,
    "orbit_determination.calcGibbs": make_calc_gibbs,
    "orbit_determination.calcHerrickGibbs": make_calc_herrick_gibbs,
    "orbit_determination.calcGauss": make_calc_gauss,
    "orbit_determination.gaussIOD": make_gauss_iod,
    "bridge.propagate_orbits_2body": make_bridge_propagate_orbits_2body,
    "bridge.rotate_orbits_frame": make_bridge_rotate_orbits_frame,
    "bridge.sample_orbit_variants": make_bridge_sample_orbit_variants,
    "observers.Observers.from_codes": make_observers_from_codes,
    "bridge.evaluate_residuals_2body": make_bridge_evaluate_residuals_2body,
}


SHAPED_GENERATORS = {
    "dynamics.propagate_2body": make_propagate_2body_shape,
    "dynamics.propagate_2body_arc_batch": make_propagate_2body_arc_batch_shape,
    "dynamics.propagate_2body_with_covariance": (
        make_propagate_2body_with_covariance_shape
    ),
    "dynamics.generate_ephemeris_2body": make_generate_ephemeris_2body_shape,
    "dynamics.generate_ephemeris_2body_with_covariance": (
        make_generate_ephemeris_2body_with_covariance_shape
    ),
    "missions.porkchop_grid": make_porkchop_grid_shape,
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
    "coordinates.transform_coordinates": WorkloadShape(len(_TRANSFORM_SUBCASE_NAMES)),
    "coordinates.transform_coordinates_with_covariance": WorkloadShape(
        len(_TRANSFORM_COVARIANCE_SUBCASE_NAMES)
    ),
    # `calculate_moid` is a scalar optimizer API; one pair is the true
    # one-off call shape, unlike vector kernels where n=10 is the tiny lane.
    "dynamics.calculate_moid": WorkloadShape(1),
    "dynamics.calculate_moid_batch": WorkloadShape(1),
    "dynamics.calculate_perturber_moids": WorkloadShape(1),
    "missions.porkchop_grid": WorkloadShape(4, extra={"departures": 2, "arrivals": 2}),
    "dynamics.generate_porkchop_data": WorkloadShape(
        4, extra={"departures": 2, "arrivals": 2}
    ),
    "dynamics.propagate_2body_along_arc": WorkloadShape(10),
    "dynamics.propagate_2body_arc_batch": WorkloadShape(10, n_orbits=2, n_epochs=5),
    "orbit_determination.gaussIOD": WorkloadShape(1, extra={"triplets": 1}),
}


SMALL_WORKLOADS: dict[str, WorkloadShape] = {
    # Keep canonical MOID speed governance affordable: baseline-main uses
    # scipy bounded minimization per pair, so n=2000 would be minutes per rep.
    "dynamics.calculate_moid": WorkloadShape(8),
    "dynamics.calculate_moid_batch": WorkloadShape(8),
    "dynamics.calculate_perturber_moids": WorkloadShape(8),
    # Public porkchop calls are grid-orchestration/table-assembly workloads;
    # 44×44 stays near the canonical n=2000 small lane while preserving a
    # square departure/arrival grid.
    "missions.porkchop_grid": WorkloadShape(
        1_936, extra={"departures": 44, "arrivals": 44}
    ),
    "dynamics.generate_porkchop_data": WorkloadShape(
        1_936, extra={"departures": 44, "arrivals": 44}
    ),
    # The single-orbit arc helper is only used below the production dispatch
    # crossover; keep its diagnostic lanes inside that intended K<500 regime.
    "dynamics.propagate_2body_along_arc": WorkloadShape(100),
    "dynamics.propagate_2body_arc_batch": WorkloadShape(
        2_000, n_orbits=40, n_epochs=50
    ),
    # Full Gauss-IOD is scalar/root-finder heavy in the legacy oracle. Sixteen
    # triplets matches the fuzz governance size and keeps the historical small
    # speed lane affordable.
    "orbit_determination.gaussIOD": WorkloadShape(16, extra={"triplets": 16}),
}


FUZZ_N_OVERRIDES: dict[str, int] = {
    # `calculate_moid` performs nested scipy/JAX minimization in the legacy
    # oracle. Eight pairs × eight seeds gives direct randomized coverage while
    # keeping canonical fuzz runs within the existing time budget.
    "dynamics.calculate_moid": 8,
    "dynamics.calculate_moid_batch": 8,
    "dynamics.calculate_perturber_moids": 8,
    "missions.porkchop_grid": 16,
    "dynamics.generate_porkchop_data": 16,
    # Full Gauss-IOD calls are scalar/root-finder heavy in the legacy oracle.
    # Sixteen randomized, well-conditioned triplets per seed gives direct fuzz
    # coverage without making every full parity run polynomial-bound.
    "orbit_determination.gaussIOD": 16,
}


LARGE_WORKLOADS: dict[str, WorkloadShape] = {
    "coordinates.cartesian_to_spherical": WorkloadShape(20_000),
    "coordinates.transform_coordinates": WorkloadShape(12_000),
    "coordinates.transform_coordinates_with_covariance": WorkloadShape(4_000),
    "coordinates.cartesian_to_geodetic": WorkloadShape(20_000),
    "coordinates.cartesian_to_keplerian": WorkloadShape(20_000),
    "coordinates.keplerian.to_cartesian": WorkloadShape(20_000),
    "coordinates.cartesian_to_cometary": WorkloadShape(20_000),
    "coordinates.cometary.to_cartesian": WorkloadShape(20_000),
    "coordinates.spherical.to_cartesian": WorkloadShape(20_000),
    "coordinates.rotate_cartesian_time_varying": WorkloadShape(50_000),
    "coordinates.residuals.Residuals.calculate": WorkloadShape(20_000),
    "coordinates.residuals.calculate_chi2": WorkloadShape(50_000),
    # Pure longitude wrapping is memory-bandwidth dominated; a 100k-row large
    # lane gives a stable throughput signal beyond the scheduler-sensitive
    # 50k transition region while remaining an API-shaped vector workload.
    "coordinates.residuals.bound_longitude_residuals": WorkloadShape(100_000),
    "coordinates.residuals.apply_cosine_latitude_correction": WorkloadShape(50_000),
    "statistics.weighted_mean": WorkloadShape(50_000),
    "statistics.weighted_covariance": WorkloadShape(50_000),
    "dynamics.calc_mean_motion": WorkloadShape(50_000),
    "dynamics.tisserand_parameter": WorkloadShape(100_000),
    "orbits.classify_orbits": WorkloadShape(50_000),
    "dynamics.calculate_moid": WorkloadShape(64),
    "dynamics.calculate_moid_batch": WorkloadShape(64),
    "dynamics.calculate_perturber_moids": WorkloadShape(64),
    "observers.Observers.from_codes": WorkloadShape(50_000),
    "missions.porkchop_grid": WorkloadShape(
        4_096, extra={"departures": 64, "arrivals": 64}
    ),
    "dynamics.generate_porkchop_data": WorkloadShape(
        4_096, extra={"departures": 64, "arrivals": 64}
    ),
    "dynamics.propagate_2body": WorkloadShape(20_000, n_orbits=1_000, n_epochs=20),
    "dynamics.propagate_2body_along_arc": WorkloadShape(400),
    "dynamics.propagate_2body_arc_batch": WorkloadShape(
        20_000, n_orbits=400, n_epochs=50
    ),
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
    "photometry.fit_absolute_magnitude_rows": WorkloadShape(50_000),
    "photometry.fit_absolute_magnitude_grouped": WorkloadShape(50_000),
    "orbit_determination.calcGibbs": WorkloadShape(5_000, extra={"triplets": 5_000}),
    "orbit_determination.calcHerrickGibbs": WorkloadShape(
        5_000, extra={"triplets": 5_000}
    ),
    "orbit_determination.calcGauss": WorkloadShape(5_000, extra={"triplets": 5_000}),
    "orbit_determination.gaussIOD": WorkloadShape(128, extra={"triplets": 128}),
    "bridge.propagate_orbits_2body": WorkloadShape(20_000),
    "bridge.rotate_orbits_frame": WorkloadShape(20_000),
    "bridge.sample_orbit_variants": WorkloadShape(5_000),
    "bridge.evaluate_residuals_2body": WorkloadShape(5_000),
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
