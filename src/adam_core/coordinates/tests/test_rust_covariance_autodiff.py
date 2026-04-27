"""
Wide-coverage correctness tests for the Rust forward-mode AD covariance
transform. Each test computes the propagated covariance via the new Rust
kernel (through the ``to_cartesian`` / ``from_cartesian`` classmethods) and
compares to the JAX reference obtained by monkey-patching
``rust_covariance_transform`` to return ``None`` (forcing the legacy path).

The tests exercise:
  - All 7 rep conversions that previously hit JAX in the fallthrough path.
  - Random grids plus hand-picked edge cases (high-e, retrograde, polar,
    near-parabolic, tiny-inclination, nan-masked covariances).
  - The Cart->Keplerian->Cart round trip cov recovery.
"""

from __future__ import annotations

import numpy as np
import pytest

from adam_core.coordinates import covariances as cov_module
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.cometary import CometaryCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.geodetics import GeodeticCoordinates
from adam_core.coordinates.keplerian import KeplerianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.time import Timestamp
from adam_core.utils.helpers.orbits import make_real_orbits

# Covariance comparison tolerances. Rust forward-mode AD and JAX forward-mode
# AD both propagate float64 through the same analytical chain-rule, but tiny
# ordering / libm differences accumulate. 1e-10 rel + 1e-20 abs is tight
# enough to catch any real semantic divergence while accepting last-bit jitter.
_COV_RTOL = 1e-10
_COV_ATOL = 1e-20


@pytest.fixture
def disable_rust_cov(monkeypatch):
    """Force the JAX fallback path by making the Rust helper return None."""

    def _return_none(*args, **kwargs):
        return None

    monkeypatch.setattr(
        cov_module, "rust_covariance_transform", _return_none, raising=True
    )


def _random_cov(rng: np.random.Generator, n: int, scale: float = 1e-4) -> np.ndarray:
    cov = np.zeros((n, 6, 6))
    for i in range(n):
        a = rng.normal(0, scale, size=(6, 6))
        cov[i] = a @ a.T
    return cov


def _assert_cov_close(rust: np.ndarray, jax_ref: np.ndarray, label: str) -> None:
    nan_rust = np.isnan(rust)
    nan_ref = np.isnan(jax_ref)
    assert np.array_equal(nan_rust, nan_ref), f"{label}: NaN masks diverge"
    valid = ~nan_rust
    if not valid.any():
        return
    np.testing.assert_allclose(
        rust[valid],
        jax_ref[valid],
        rtol=_COV_RTOL,
        atol=_COV_ATOL,
        err_msg=f"{label}: cov mismatch",
    )


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_kepler(
    rng: np.random.Generator,
    n: int,
    *,
    a_range=(0.8, 5.0),
    e_range=(0.0, 0.3),
    i_range=(0.0, 30.0),
    cov_scale: float = 1e-4,
    seeded_times: bool = False,
) -> KeplerianCoordinates:
    a = rng.uniform(*a_range, size=n)
    e = rng.uniform(*e_range, size=n)
    i = rng.uniform(*i_range, size=n)
    raan = rng.uniform(0.0, 360.0, size=n)
    ap = rng.uniform(0.0, 360.0, size=n)
    m = rng.uniform(0.0, 360.0, size=n)
    cov = _random_cov(rng, n, scale=cov_scale)
    mjds = np.full(n, 60000.0) if not seeded_times else 60000.0 + rng.uniform(0, 365, n)
    return KeplerianCoordinates.from_kwargs(
        a=a,
        e=e,
        i=i,
        raan=raan,
        ap=ap,
        M=m,
        time=Timestamp.from_mjd(mjds, scale="tdb"),
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=["SUN"] * n),
        frame="ecliptic",
    )


def _make_cart_from_kepler(kep: KeplerianCoordinates) -> CartesianCoordinates:
    return kep.to_cartesian()


def _make_cometary(
    rng: np.random.Generator,
    n: int,
    *,
    q_range=(0.3, 2.5),
    e_range=(0.0, 0.4),
    i_range=(0.0, 30.0),
    cov_scale: float = 1e-4,
) -> CometaryCoordinates:
    q = rng.uniform(*q_range, size=n)
    e = rng.uniform(*e_range, size=n)
    i = rng.uniform(*i_range, size=n)
    raan = rng.uniform(0.0, 360.0, size=n)
    ap = rng.uniform(0.0, 360.0, size=n)
    mjds = np.full(n, 60000.0)
    # tp within a few days of t0 to keep mean anomaly moderate
    tp = mjds + rng.uniform(-10.0, 10.0, size=n)
    cov = _random_cov(rng, n, scale=cov_scale)
    return CometaryCoordinates.from_kwargs(
        q=q,
        e=e,
        i=i,
        raan=raan,
        ap=ap,
        tp=tp,
        time=Timestamp.from_mjd(mjds, scale="tdb"),
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=["SUN"] * n),
        frame="ecliptic",
    )


def _make_spherical(
    rng: np.random.Generator, n: int, cov_scale: float = 1e-4
) -> SphericalCoordinates:
    rho = rng.uniform(0.5, 5.0, size=n)
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = rng.uniform(-85.0, 85.0, size=n)
    vrho = rng.normal(0.0, 0.01, size=n)
    vlon = rng.normal(0.0, 0.5, size=n)
    vlat = rng.normal(0.0, 0.5, size=n)
    mjds = np.full(n, 60000.0)
    cov = _random_cov(rng, n, scale=cov_scale)
    return SphericalCoordinates.from_kwargs(
        rho=rho,
        lon=lon,
        lat=lat,
        vrho=vrho,
        vlon=vlon,
        vlat=vlat,
        time=Timestamp.from_mjd(mjds, scale="tdb"),
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=["SUN"] * n),
        frame="ecliptic",
    )


def _make_cart_itrf93(
    rng: np.random.Generator, n: int, cov_scale: float = 1e-6
) -> CartesianCoordinates:
    # Points outside Earth center, meters scale is not required since the
    # geodetic WGS84 a is consistent with the inputs here. Use AU-scale for
    # consistency with the rest of the test matrix.
    pts = rng.normal(0.0, 1.0, size=(n, 6))
    pts[:, 0:3] += np.array([1.0, 0.0, 0.0])  # push away from the origin
    cov = _random_cov(rng, n, scale=cov_scale)
    mjds = np.full(n, 60000.0)
    return CartesianCoordinates.from_kwargs(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        vx=pts[:, 3],
        vy=pts[:, 4],
        vz=pts[:, 5],
        time=Timestamp.from_mjd(mjds, scale="tdb"),
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=["EARTH"] * n),
        frame="itrf93",
    )


# ---------------------------------------------------------------------------
# Tests: each target conversion is exercised with both a broad random sweep
# and hand-picked edge cases. ``disable_rust_cov`` toggles the JAX reference.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 11, 42, 1337])
def test_kepler_to_cartesian_matches_jax(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    kep = _make_kepler(rng, 24)

    cart_rust = kep.to_cartesian()
    cart_rust_cov = cart_rust.covariance.to_matrix()

    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        cart_jax = kep.to_cartesian()
    cart_jax_cov = cart_jax.covariance.to_matrix()

    np.testing.assert_allclose(
        cart_rust.values, cart_jax.values, rtol=1e-12, atol=1e-12
    )
    _assert_cov_close(cart_rust_cov, cart_jax_cov, "kepler->cartesian")


@pytest.mark.parametrize("seed", [0, 11, 42, 1337])
def test_cartesian_to_kepler_matches_jax(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    kep = _make_kepler(rng, 24)
    cart = _make_cart_from_kepler(kep)

    kep_rust = KeplerianCoordinates.from_cartesian(cart)
    kep_rust_cov = kep_rust.covariance.to_matrix()

    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        kep_jax = KeplerianCoordinates.from_cartesian(cart)
    kep_jax_cov = kep_jax.covariance.to_matrix()

    np.testing.assert_allclose(
        kep_rust.values, kep_jax.values, rtol=1e-10, atol=1e-10
    )
    _assert_cov_close(kep_rust_cov, kep_jax_cov, "cartesian->kepler")


@pytest.mark.parametrize("seed", [0, 7, 101, 2025])
def test_cometary_to_cartesian_matches_jax(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    com = _make_cometary(rng, 20)

    cart_rust = com.to_cartesian()
    cart_rust_cov = cart_rust.covariance.to_matrix()

    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        cart_jax = com.to_cartesian()
    cart_jax_cov = cart_jax.covariance.to_matrix()

    np.testing.assert_allclose(
        cart_rust.values, cart_jax.values, rtol=1e-12, atol=1e-12
    )
    _assert_cov_close(cart_rust_cov, cart_jax_cov, "cometary->cartesian")


@pytest.mark.parametrize("seed", [0, 7, 101, 2025])
def test_cartesian_to_cometary_matches_jax(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    com = _make_cometary(rng, 20)
    cart = com.to_cartesian()

    com_rust = CometaryCoordinates.from_cartesian(cart)
    com_rust_cov = com_rust.covariance.to_matrix()

    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        com_jax = CometaryCoordinates.from_cartesian(cart)
    com_jax_cov = com_jax.covariance.to_matrix()

    np.testing.assert_allclose(
        com_rust.values, com_jax.values, rtol=1e-10, atol=1e-10
    )
    _assert_cov_close(com_rust_cov, com_jax_cov, "cartesian->cometary")


@pytest.mark.parametrize("seed", [0, 7, 101, 2025])
def test_spherical_to_cartesian_matches_jax(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    sph = _make_spherical(rng, 16)

    cart_rust = sph.to_cartesian()
    cart_rust_cov = cart_rust.covariance.to_matrix()

    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        cart_jax = sph.to_cartesian()
    cart_jax_cov = cart_jax.covariance.to_matrix()

    np.testing.assert_allclose(
        cart_rust.values, cart_jax.values, rtol=1e-12, atol=1e-12
    )
    _assert_cov_close(cart_rust_cov, cart_jax_cov, "spherical->cartesian")


@pytest.mark.parametrize("seed", [0, 7, 101, 2025])
def test_cartesian_to_spherical_matches_jax(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    sph = _make_spherical(rng, 16)
    cart = sph.to_cartesian()

    sph_rust = SphericalCoordinates.from_cartesian(cart)
    sph_rust_cov = sph_rust.covariance.to_matrix()

    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        sph_jax = SphericalCoordinates.from_cartesian(cart)
    sph_jax_cov = sph_jax.covariance.to_matrix()

    np.testing.assert_allclose(
        sph_rust.values, sph_jax.values, rtol=1e-10, atol=1e-10
    )
    _assert_cov_close(sph_rust_cov, sph_jax_cov, "cartesian->spherical")


@pytest.mark.parametrize("seed", [0, 7, 101, 2025])
def test_cartesian_to_geodetic_matches_jax(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    cart = _make_cart_itrf93(rng, 16)

    geo_rust = GeodeticCoordinates.from_cartesian(cart)
    geo_rust_cov = geo_rust.covariance.to_matrix()

    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        geo_jax = GeodeticCoordinates.from_cartesian(cart)
    geo_jax_cov = geo_jax.covariance.to_matrix()

    np.testing.assert_allclose(
        geo_rust.values, geo_jax.values, rtol=1e-10, atol=1e-10
    )
    _assert_cov_close(geo_rust_cov, geo_jax_cov, "cartesian->geodetic")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_kepler_edges_all_paths(monkeypatch):
    """Retrograde, polar, high-e, near-zero-e elements."""
    n = 6
    a = np.array([1.0, 1.5, 0.9, 2.0, 3.0, 5.0])
    e = np.array([0.0, 0.5, 0.8, 0.05, 0.01, 0.2])
    i = np.array([0.0, 45.0, 90.0, 120.0, 170.0, 179.9])
    raan = np.array([0.0, 45.0, 90.0, 135.0, 270.0, 359.9])
    ap = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 300.0])
    m = np.array([0.0, 30.0, 90.0, 180.0, 240.0, 350.0])
    cov = np.tile(np.eye(6) * 1e-6, (n, 1, 1))
    mjds = np.full(n, 60000.0)
    kep = KeplerianCoordinates.from_kwargs(
        a=a, e=e, i=i, raan=raan, ap=ap, M=m,
        time=Timestamp.from_mjd(mjds, scale="tdb"),
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=["SUN"] * n),
        frame="ecliptic",
    )
    cart_rust = kep.to_cartesian()
    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        cart_jax = kep.to_cartesian()
    _assert_cov_close(
        cart_rust.covariance.to_matrix(),
        cart_jax.covariance.to_matrix(),
        "kepler->cart edges",
    )


def test_partial_nan_covariance_propagates_nan(monkeypatch):
    """A row whose covariance has any NaN must produce an all-NaN output
    covariance for that row, and the rust and jax paths must agree."""
    rng = np.random.default_rng(99)
    kep = _make_kepler(rng, 5)
    cov = kep.covariance.to_matrix()
    cov[1, 2, 2] = np.nan  # taint row 1
    cov[3, :, :] = np.nan  # row 3 all-NaN
    kep_nan = KeplerianCoordinates.from_kwargs(
        a=kep.values[:, 0], e=kep.values[:, 1], i=kep.values[:, 2],
        raan=kep.values[:, 3], ap=kep.values[:, 4], M=kep.values[:, 5],
        time=kep.time,
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=kep.origin,
        frame=kep.frame,
    )
    cart_rust = kep_nan.to_cartesian()
    cart_rust_cov = cart_rust.covariance.to_matrix()

    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        cart_jax = kep_nan.to_cartesian()
    cart_jax_cov = cart_jax.covariance.to_matrix()

    # Rows 1 and 3 must be all-NaN.
    assert np.isnan(cart_rust_cov[1]).all()
    assert np.isnan(cart_rust_cov[3]).all()
    # And the rust/jax nan masks must align everywhere.
    assert np.array_equal(np.isnan(cart_rust_cov), np.isnan(cart_jax_cov))
    # Non-NaN rows must also numerically match.
    _assert_cov_close(cart_rust_cov, cart_jax_cov, "kepler->cart partial nan")


def test_roundtrip_preserves_covariance_on_real_orbits():
    """Cart -> Kep -> Cart on real orbit fixtures must preserve covariance."""
    orbits = make_real_orbits()[:12]
    cart0 = orbits.coordinates
    kep = KeplerianCoordinates.from_cartesian(cart0)
    cart_back = kep.to_cartesian()
    cov0 = cart0.covariance.to_matrix()
    cov_back = cart_back.covariance.to_matrix()
    # Round trip should preserve cov to within double precision * condition.
    # The JAX implementation historically achieves ~1e-14 rel; we allow 1e-10.
    _assert_cov_close(cov_back, cov0, "roundtrip cart->kep->cart")


def test_roundtrip_preserves_covariance_cometary():
    orbits = make_real_orbits()[:8]
    cart0 = orbits.coordinates
    com = CometaryCoordinates.from_cartesian(cart0)
    cart_back = com.to_cartesian()
    cov0 = cart0.covariance.to_matrix()
    cov_back = cart_back.covariance.to_matrix()
    _assert_cov_close(cov_back, cov0, "roundtrip cart->com->cart")


def test_roundtrip_preserves_covariance_spherical():
    orbits = make_real_orbits()[:8]
    cart0 = orbits.coordinates
    sph = SphericalCoordinates.from_cartesian(cart0)
    cart_back = sph.to_cartesian()
    cov0 = cart0.covariance.to_matrix()
    cov_back = cart_back.covariance.to_matrix()
    _assert_cov_close(cov_back, cov0, "roundtrip cart->sph->cart")


def test_real_orbits_cart_to_keplerian_matches_jax(monkeypatch):
    orbits = make_real_orbits()[:16]
    cart = orbits.coordinates
    kep_rust = KeplerianCoordinates.from_cartesian(cart)

    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        kep_jax = KeplerianCoordinates.from_cartesian(cart)

    np.testing.assert_allclose(
        kep_rust.values, kep_jax.values, rtol=1e-10, atol=1e-10
    )
    _assert_cov_close(
        kep_rust.covariance.to_matrix(),
        kep_jax.covariance.to_matrix(),
        "real orbits cart->kep",
    )


def test_real_orbits_cart_to_cometary_matches_jax(monkeypatch):
    orbits = make_real_orbits()[:16]
    cart = orbits.coordinates
    com_rust = CometaryCoordinates.from_cartesian(cart)
    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        com_jax = CometaryCoordinates.from_cartesian(cart)
    np.testing.assert_allclose(
        com_rust.values, com_jax.values, rtol=1e-10, atol=1e-10
    )
    _assert_cov_close(
        com_rust.covariance.to_matrix(),
        com_jax.covariance.to_matrix(),
        "real orbits cart->com",
    )


def test_real_orbits_cart_to_spherical_matches_jax(monkeypatch):
    orbits = make_real_orbits()[:16]
    cart = orbits.coordinates
    sph_rust = SphericalCoordinates.from_cartesian(cart)
    with monkeypatch.context() as mp:
        mp.setattr(
            cov_module, "rust_covariance_transform", lambda *a, **k: None, raising=True
        )
        sph_jax = SphericalCoordinates.from_cartesian(cart)
    np.testing.assert_allclose(
        sph_rust.values, sph_jax.values, rtol=1e-10, atol=1e-10
    )
    _assert_cov_close(
        sph_rust.covariance.to_matrix(),
        sph_jax.covariance.to_matrix(),
        "real orbits cart->sph",
    )
