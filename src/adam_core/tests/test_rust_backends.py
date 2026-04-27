import numpy as np
import numpy.testing as npt
import pytest

from adam_core._rust import api as rust_api
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.cometary import CometaryCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.geodetics import GeodeticCoordinates
from adam_core.coordinates.keplerian import KeplerianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.coordinates.transform import (
    _rust_transform_supports,
    _try_transform_coordinates_rust,
    cartesian_to_geodetic,
    cartesian_to_keplerian,
    cartesian_to_spherical,
    keplerian_to_cartesian,
    spherical_to_cartesian,
    transform_coordinates,
)
from adam_core._rust import calc_mean_motion_numpy as _calc_mean_motion_rust
from adam_core.time import Timestamp


class _FakeRustNative:
    @staticmethod
    def cartesian_to_spherical_numpy(coords: np.ndarray) -> np.ndarray:
        out = np.zeros_like(coords)
        out[:, 0] = np.linalg.norm(coords[:, :3], axis=1)
        return out

    @staticmethod
    def spherical_to_cartesian_numpy(coords: np.ndarray) -> np.ndarray:
        out = np.zeros_like(coords)
        rho = coords[:, 0]
        lon = np.radians(coords[:, 1])
        lat = np.radians(coords[:, 2])
        vrho = coords[:, 3]
        vlon = np.radians(coords[:, 4])
        vlat = np.radians(coords[:, 5])
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)

        out[:, 0] = rho * cos_lat * cos_lon
        out[:, 1] = rho * cos_lat * sin_lon
        out[:, 2] = rho * sin_lat
        out[:, 3] = (
            cos_lat * cos_lon * vrho
            - rho * cos_lat * sin_lon * vlon
            - rho * sin_lat * cos_lon * vlat
        )
        out[:, 4] = (
            cos_lat * sin_lon * vrho
            + rho * cos_lat * cos_lon * vlon
            - rho * sin_lat * sin_lon * vlat
        )
        out[:, 5] = sin_lat * vrho + rho * cos_lat * vlat
        return out

    @staticmethod
    def calc_mean_motion_numpy(a: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.sqrt(mu / np.abs(a) ** 3)

    @staticmethod
    def cartesian_to_geodetic_numpy(
        coords: np.ndarray,
        a: float,
        f: float,
        max_iter: int,
        tol: float,
    ) -> np.ndarray:
        out = np.zeros_like(coords)
        out[:, 0] = coords[:, 0] + a + f + max_iter + tol
        return out

    @staticmethod
    def cartesian_to_keplerian_numpy(
        coords: np.ndarray, t0: np.ndarray, mu: np.ndarray
    ) -> np.ndarray:
        out = np.zeros((coords.shape[0], 13), dtype=np.float64)
        out[:, 0] = coords[:, 0] + t0 + mu
        return out

    @staticmethod
    def keplerian_to_cartesian_numpy(
        coords: np.ndarray,
        mu: np.ndarray,
        max_iter: int,
        tol: float,
    ) -> np.ndarray:
        out = np.zeros((coords.shape[0], 6), dtype=np.float64)
        out[:, 0] = coords[:, 0] + mu + max_iter + tol
        return out

    @staticmethod
    def transform_coordinates_numpy(
        coords: np.ndarray,
        representation_in: str,
        representation_out: str,
        t0: np.ndarray | None = None,
        mu: np.ndarray | None = None,
        a: float | None = None,
        f: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-15,
        frame_in: str | None = None,
        frame_out: str | None = None,
        translation_vectors: np.ndarray | None = None,
    ) -> np.ndarray:
        out = np.zeros((coords.shape[0], 6), dtype=np.float64)
        if representation_in == "spherical" and representation_out == "geodetic":
            out[:, 0] = 42.0
        return out


def test_cartesian_to_spherical_prefers_rust_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())
    monkeypatch.setattr(rust_api, "RUST_BACKEND_AVAILABLE", True)

    coords = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    got = cartesian_to_spherical(coords)

    npt.assert_allclose(
        got, np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), rtol=0.0, atol=1e-15
    )


def test_calc_mean_motion_rust_wrapper_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())
    monkeypatch.setattr(rust_api, "RUST_BACKEND_AVAILABLE", True)

    a = np.array([2.0], dtype=np.float64)
    mu = np.array([8.0], dtype=np.float64)

    got = np.asarray(rust_api.calc_mean_motion_numpy(a, mu), dtype=np.float64)
    npt.assert_allclose(got, np.array([1.0]), rtol=0.0, atol=1e-15)


def test_spherical_to_cartesian_prefers_rust_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())
    monkeypatch.setattr(rust_api, "RUST_BACKEND_AVAILABLE", True)

    coords = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    got = spherical_to_cartesian(coords)

    npt.assert_allclose(
        got, np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), rtol=0.0, atol=1e-15
    )


def test_cartesian_to_geodetic_prefers_rust_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())
    monkeypatch.setattr(rust_api, "RUST_BACKEND_AVAILABLE", True)

    coords = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    got = cartesian_to_geodetic(coords, a=1.0, f=0.1, max_iter=3, tol=0.01)

    npt.assert_allclose(
        got,
        np.array([[5.11, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        rtol=0.0,
        atol=1e-15,
    )


def test_cartesian_to_keplerian_prefers_rust_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())
    monkeypatch.setattr(rust_api, "RUST_BACKEND_AVAILABLE", True)

    coords = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    t0 = np.array([10.0], dtype=np.float64)
    mu = np.array([2.0], dtype=np.float64)
    got = cartesian_to_keplerian(coords, t0, mu)

    npt.assert_allclose(
        got[0, 0],
        13.0,
        rtol=0.0,
        atol=1e-15,
    )


def test_keplerian_to_cartesian_prefers_rust_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())
    monkeypatch.setattr(rust_api, "RUST_BACKEND_AVAILABLE", True)

    coords = np.array([[1.0, 0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    mu = np.array([2.0], dtype=np.float64)
    got = keplerian_to_cartesian(coords, mu, max_iter=3, tol=0.01)

    npt.assert_allclose(
        got[:, 0],
        np.array([6.01], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )


def test_transform_coordinates_uses_single_rust_crossing_for_supported_path(
    monkeypatch,
):
    class _StrictRustNative(_FakeRustNative):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def transform_coordinates_numpy(
            self,
            coords: np.ndarray,
            representation_in: str,
            representation_out: str,
            t0: np.ndarray | None = None,
            mu: np.ndarray | None = None,
            a: float | None = None,
            f: float | None = None,
            max_iter: int = 100,
            tol: float = 1e-15,
            frame_in: str | None = None,
            frame_out: str | None = None,
            translation_vectors: np.ndarray | None = None,
        ) -> np.ndarray:
            self.calls.append((representation_in, representation_out))
            out = np.zeros((coords.shape[0], 6), dtype=np.float64)
            out[:, 0] = 12.34
            return out

        def spherical_to_cartesian_numpy(self, coords: np.ndarray) -> np.ndarray:  # type: ignore[override]
            raise AssertionError(
                "unexpected ping-pong via spherical_to_cartesian_numpy"
            )

        def cartesian_to_geodetic_numpy(  # type: ignore[override]
            self, coords: np.ndarray, a: float, f: float, max_iter: int, tol: float
        ) -> np.ndarray:
            raise AssertionError("unexpected ping-pong via cartesian_to_geodetic_numpy")

    strict_native = _StrictRustNative()
    monkeypatch.setattr(rust_api, "_native", strict_native)
    monkeypatch.setattr(rust_api, "RUST_BACKEND_AVAILABLE", True)

    coords = SphericalCoordinates.from_kwargs(
        rho=[1.0],
        lon=[0.0],
        lat=[0.0],
        vrho=[0.0],
        vlon=[0.0],
        vlat=[0.0],
        time=Timestamp.from_mjd(np.array([60000.0]), scale="tdb"),
        covariance=CoordinateCovariances.nulls(1),
        origin=Origin.from_OriginCodes(OriginCodes.EARTH, size=1),
        frame="itrf93",
    )

    got = transform_coordinates(
        coords,
        representation_out=GeodeticCoordinates,
        frame_out="itrf93",
        origin_out=OriginCodes.EARTH,
    )

    assert strict_native.calls == [("spherical", "geodetic")]
    npt.assert_allclose(
        got.values[:, 0],
        np.array([12.34], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )


def test_transform_coordinates_uses_single_rust_crossing_for_frame_change(monkeypatch):
    class _StrictRustNative(_FakeRustNative):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str | None, str | None]] = []

        def transform_coordinates_numpy(
            self,
            coords: np.ndarray,
            representation_in: str,
            representation_out: str,
            t0: np.ndarray | None = None,
            mu: np.ndarray | None = None,
            a: float | None = None,
            f: float | None = None,
            max_iter: int = 100,
            tol: float = 1e-15,
            frame_in: str | None = None,
            frame_out: str | None = None,
            translation_vectors: np.ndarray | None = None,
        ) -> np.ndarray:
            self.calls.append(
                (representation_in, representation_out, frame_in, frame_out)
            )
            out = np.zeros((coords.shape[0], 6), dtype=np.float64)
            out[:, 0] = 99.0
            return out

        def spherical_to_cartesian_numpy(self, coords: np.ndarray) -> np.ndarray:  # type: ignore[override]
            raise AssertionError(
                "unexpected ping-pong via spherical_to_cartesian_numpy"
            )

        def cartesian_to_spherical_numpy(self, coords: np.ndarray) -> np.ndarray:  # type: ignore[override]
            raise AssertionError(
                "unexpected ping-pong via cartesian_to_spherical_numpy"
            )

    strict_native = _StrictRustNative()
    monkeypatch.setattr(rust_api, "_native", strict_native)
    monkeypatch.setattr(rust_api, "RUST_BACKEND_AVAILABLE", True)

    coords = SphericalCoordinates.from_kwargs(
        rho=[1.0],
        lon=[0.0],
        lat=[0.0],
        vrho=[0.0],
        vlon=[0.0],
        vlat=[0.0],
        time=Timestamp.from_mjd(np.array([60000.0]), scale="tdb"),
        covariance=CoordinateCovariances.nulls(1),
        origin=Origin.from_OriginCodes(OriginCodes.SUN, size=1),
        frame="equatorial",
    )

    got = transform_coordinates(
        coords,
        representation_out=SphericalCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    assert strict_native.calls == [("spherical", "spherical", "equatorial", "ecliptic")]
    assert got.frame == "ecliptic"
    npt.assert_allclose(
        got.values[:, 0],
        np.array([99.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )


def test_transform_coordinates_uses_single_rust_crossing_for_keplerian_input(
    monkeypatch,
):
    class _StrictRustNative(_FakeRustNative):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, bool]] = []

        def transform_coordinates_numpy(
            self,
            coords: np.ndarray,
            representation_in: str,
            representation_out: str,
            t0: np.ndarray | None = None,
            mu: np.ndarray | None = None,
            a: float | None = None,
            f: float | None = None,
            max_iter: int = 100,
            tol: float = 1e-15,
            frame_in: str | None = None,
            frame_out: str | None = None,
            translation_vectors: np.ndarray | None = None,
        ) -> np.ndarray:
            self.calls.append((representation_in, representation_out, mu is not None))
            out = np.zeros((coords.shape[0], 6), dtype=np.float64)
            out[:, 0] = 7.0
            return out

        def keplerian_to_cartesian_numpy(  # type: ignore[override]
            self, coords: np.ndarray, mu: np.ndarray, max_iter: int, tol: float
        ) -> np.ndarray:
            raise AssertionError(
                "unexpected ping-pong via keplerian_to_cartesian_numpy"
            )

    strict_native = _StrictRustNative()
    monkeypatch.setattr(rust_api, "_native", strict_native)
    monkeypatch.setattr(rust_api, "RUST_BACKEND_AVAILABLE", True)

    coords = KeplerianCoordinates.from_kwargs(
        a=[1.0],
        e=[0.1],
        i=[1.0],
        raan=[2.0],
        ap=[3.0],
        M=[4.0],
        time=Timestamp.from_mjd(np.array([60000.0]), scale="tdb"),
        covariance=CoordinateCovariances.nulls(1),
        origin=Origin.from_OriginCodes(OriginCodes.SUN, size=1),
        frame="ecliptic",
    )

    got = transform_coordinates(
        coords,
        representation_out=SphericalCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    assert strict_native.calls == [("keplerian", "spherical", True)]
    npt.assert_allclose(
        got.values[:, 0],
        np.array([7.0], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )


def _make_cartesian(
    *,
    frame: str = "ecliptic",
    origin_code: OriginCodes = OriginCodes.SUN,
    with_covariance: bool = False,
) -> CartesianCoordinates:
    covariance = (
        CoordinateCovariances.from_sigmas(np.full((1, 6), 0.1, dtype=np.float64))
        if with_covariance
        else CoordinateCovariances.nulls(1)
    )
    return CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=Timestamp.from_mjd(np.array([60000.0]), scale="tdb"),
        covariance=covariance,
        origin=Origin.from_OriginCodes(origin_code, size=1),
        frame=frame,
    )


def _make_cometary() -> CometaryCoordinates:
    return CometaryCoordinates.from_kwargs(
        q=[1.0],
        e=[0.1],
        i=[0.0],
        raan=[0.0],
        ap=[0.0],
        tp=[60000.0],
        time=Timestamp.from_mjd(np.array([60000.0]), scale="tdb"),
        covariance=CoordinateCovariances.nulls(1),
        origin=Origin.from_OriginCodes(OriginCodes.SUN, size=1),
        frame="ecliptic",
    )


@pytest.mark.parametrize(
    "case_id,coords_factory,representation_out,frame_out,origin_out",
    [
        (
            "geodetic-non-earth-origin",
            lambda: _make_cartesian(frame="itrf93"),
            GeodeticCoordinates,
            "itrf93",
            OriginCodes.SUN,
        ),
        (
            "cartesian-to-cartesian-frame-change",
            _make_cartesian,
            CartesianCoordinates,
            "equatorial",
            OriginCodes.SUN,
        ),
    ],
)
def test_rust_transform_unsupported_paths_return_none(
    case_id: str,
    coords_factory,
    representation_out,
    frame_out,
    origin_out,
):
    """Every unsupported path must fail the predicate AND return None from
    the try-function, so the dispatcher always falls back to legacy."""
    coords = coords_factory()
    assert not _rust_transform_supports(
        coords, representation_out, frame_out, origin_out
    ), f"{case_id}: predicate unexpectedly reported Rust supports this path"
    assert (
        _try_transform_coordinates_rust(
            coords, representation_out, frame_out, origin_out
        )
        is None
    ), f"{case_id}: try-function returned a value for an unsupported path"


def test_rust_transform_supports_covers_known_good_path():
    coords = _make_cartesian()
    assert _rust_transform_supports(
        coords, SphericalCoordinates, "ecliptic", OriginCodes.SUN
    )


def test_rust_transform_supports_covers_origin_change():
    """Origin changes now route through the Rust dispatcher: translate via
    SPICE, then dispatch frame+representation change so the covariance AD
    stays on the Rust path."""
    coords = _make_cartesian()
    assert _rust_transform_supports(
        coords, SphericalCoordinates, "ecliptic", OriginCodes.EARTH
    )
    # But mixed-origin arrays (no single target) are still unsupported.
    mixed_origin_out = np.array(["EARTH", "SUN"])
    coords_two = CartesianCoordinates.from_kwargs(
        x=[1.0, 2.0], y=[0.0, 0.0], z=[0.0, 0.0],
        vx=[0.0, 0.0], vy=[0.0, 0.0], vz=[0.0, 0.0],
        time=Timestamp.from_mjd(np.array([60000.0, 60000.0]), scale="tdb"),
        covariance=CoordinateCovariances.nulls(2),
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        frame="ecliptic",
    )
    assert not _rust_transform_supports(
        coords_two, SphericalCoordinates, "ecliptic", mixed_origin_out
    )


def test_rust_transform_supports_covers_itrf93_frame_change():
    """Rust routes ITRF93 frame changes via time-varying rotation + identity-
    frame representation conversion, avoiding the JAX from_cartesian path."""
    assert _rust_transform_supports(
        _make_cartesian(frame="itrf93"),
        SphericalCoordinates,
        "ecliptic",
        OriginCodes.SUN,
    )
    assert _rust_transform_supports(
        _make_cartesian(frame="ecliptic"),
        KeplerianCoordinates,
        "itrf93",
        OriginCodes.SUN,
    )
    # Non-Cartesian input into an ITRF93 frame change stays unsupported.
    assert not _rust_transform_supports(
        _make_cometary(),
        CartesianCoordinates,
        "itrf93",
        OriginCodes.SUN,
    )


def test_rust_transform_supports_covers_cometary_roundtrip():
    """Rust now covers Cometary in either direction."""
    assert _rust_transform_supports(
        _make_cometary(), SphericalCoordinates, "ecliptic", OriginCodes.SUN
    )
    assert _rust_transform_supports(
        _make_cartesian(), CometaryCoordinates, "ecliptic", OriginCodes.SUN
    )


def test_rust_transform_supports_covariance_present():
    """Rust now propagates covariances via forward-mode AD."""
    coords = _make_cartesian(with_covariance=True)
    assert _rust_transform_supports(
        coords, SphericalCoordinates, "ecliptic", OriginCodes.SUN
    )
    result = _try_transform_coordinates_rust(
        coords, SphericalCoordinates, "ecliptic", OriginCodes.SUN
    )
    assert result is not None
    assert not result.covariance.is_all_nan()
