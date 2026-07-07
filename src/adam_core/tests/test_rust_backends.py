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
    _try_transform_coordinates_rust,
    cartesian_to_geodetic,
    cartesian_to_keplerian,
    cartesian_to_spherical,
    keplerian_to_cartesian,
    spherical_to_cartesian,
    transform_coordinates,
)
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


def test_rust_api_mandatory_native_contract() -> None:
    assert rust_api.RUST_BACKEND_AVAILABLE is True
    assert rust_api.SPICEKIT_AVAILABLE is True
    missing = [
        name
        for name in rust_api._REQUIRED_NATIVE_SYMBOLS
        if not hasattr(rust_api._native, name)
    ]
    assert missing == []


def test_standalone_data_model_schema_metadata_matches_python_adapter_contract() -> (
    None
):
    coordinate_fields, coordinate_metadata = (
        rust_api.cartesian_coordinate_schema_metadata()
    )
    assert coordinate_fields[:9] == [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "time_days",
        "time_nanos",
        "origin_code",
    ]
    assert coordinate_fields[9] == "covariance_00"
    assert coordinate_fields[-1] == "covariance_55"
    assert len(coordinate_fields) == 45
    assert (
        coordinate_metadata["adam_core_schema"] == "CoordinateBatch.cartesian.flat.v1"
    )
    assert coordinate_metadata["adam_core_representation"] == "cartesian"

    orbit_fields, orbit_metadata = rust_api.orbit_schema_metadata()
    assert orbit_fields[:11] == ["orbit_id", "object_id", *coordinate_fields[:9]]
    assert orbit_fields[-1] == "covariance_55"
    assert len(orbit_fields) == 47
    assert orbit_metadata["adam_core_schema"] == "OrbitBatch.cartesian.flat.v1"


def test_cartesian_to_spherical_prefers_rust_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())

    coords = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    got = cartesian_to_spherical(coords)

    npt.assert_allclose(
        got, np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), rtol=0.0, atol=1e-15
    )


def test_calc_mean_motion_rust_wrapper_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())

    a = np.array([2.0], dtype=np.float64)
    mu = np.array([8.0], dtype=np.float64)

    got = np.asarray(rust_api.calc_mean_motion_numpy(a, mu), dtype=np.float64)
    npt.assert_allclose(got, np.array([1.0]), rtol=0.0, atol=1e-15)


def test_spherical_to_cartesian_prefers_rust_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())

    coords = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    got = spherical_to_cartesian(coords)

    npt.assert_allclose(
        got, np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), rtol=0.0, atol=1e-15
    )


def test_cartesian_to_geodetic_prefers_rust_when_available(monkeypatch):
    monkeypatch.setattr(rust_api, "_native", _FakeRustNative())

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

    coords = np.array([[1.0, 0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    mu = np.array([2.0], dtype=np.float64)
    got = keplerian_to_cartesian(coords, mu, max_iter=3, tol=0.01)

    npt.assert_allclose(
        got[:, 0],
        np.array([6.01], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )


def test_transform_coordinates_with_covariance_requires_rep_parameters() -> None:
    coords = np.array([[1.0, 0.0, 0.0, 0.0, 0.01, 0.0]], dtype=np.float64)
    covariances = np.eye(6, dtype=np.float64).reshape(1, 36) * 1e-12
    t0 = np.array([60000.0], dtype=np.float64)
    mu = np.array([0.00029591220828411956], dtype=np.float64)

    with pytest.raises(ValueError, match="mu is required"):
        rust_api.transform_coordinates_with_covariance_numpy(
            coords,
            covariances,
            "cartesian",
            "keplerian",
            t0=t0,
            frame_in="ecliptic",
            frame_out="ecliptic",
        )

    with pytest.raises(ValueError, match="t0 is required"):
        rust_api.transform_coordinates_with_covariance_numpy(
            coords,
            covariances,
            "cartesian",
            "cometary",
            mu=mu,
            frame_in="ecliptic",
            frame_out="ecliptic",
        )

    with pytest.raises(ValueError, match="a is required"):
        rust_api.transform_coordinates_with_covariance_numpy(
            coords,
            covariances,
            "cartesian",
            "geodetic",
            frame_in="itrf93",
            frame_out="itrf93",
        )

    with pytest.raises(ValueError, match="f is required"):
        rust_api.transform_coordinates_with_covariance_numpy(
            coords,
            covariances,
            "cartesian",
            "geodetic",
            a=1.0,
            frame_in="itrf93",
            frame_out="itrf93",
        )

    with pytest.raises(ValueError, match="geodetic input is not supported"):
        rust_api.transform_coordinates_with_covariance_numpy(
            coords,
            covariances,
            "geodetic",
            "cartesian",
            a=1.0,
            f=0.0,
            frame_in="itrf93",
            frame_out="itrf93",
        )


def test_transform_coordinates_uses_single_rust_crossing_for_supported_path(
    monkeypatch,
):
    class _StrictRustNative(_FakeRustNative):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def transform_coordinates_native(
            self,
            coords: np.ndarray,
            representation_in: str,
            representation_out: str,
            frame_in: str,
            frame_out: str,
            origin_codes: list[str],
            target_origin: str | None,
            time_scale: str,
            time_days: np.ndarray,
            time_nanos: np.ndarray,
            covariances: np.ndarray | None = None,
            t0: np.ndarray | None = None,
            mu: np.ndarray | None = None,
            a: float | None = None,
            f: float | None = None,
            max_iter: int = 100,
            tol: float = 1e-15,
        ) -> tuple[np.ndarray, np.ndarray | None]:
            self.calls.append((representation_in, representation_out))
            out = np.zeros((coords.shape[0], 6), dtype=np.float64)
            out[:, 0] = 12.34
            return out, None

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

        def transform_coordinates_native(
            self,
            coords: np.ndarray,
            representation_in: str,
            representation_out: str,
            frame_in: str,
            frame_out: str,
            origin_codes: list[str],
            target_origin: str | None,
            time_scale: str,
            time_days: np.ndarray,
            time_nanos: np.ndarray,
            covariances: np.ndarray | None = None,
            t0: np.ndarray | None = None,
            mu: np.ndarray | None = None,
            a: float | None = None,
            f: float | None = None,
            max_iter: int = 100,
            tol: float = 1e-15,
        ) -> tuple[np.ndarray, np.ndarray | None]:
            self.calls.append(
                (representation_in, representation_out, frame_in, frame_out)
            )
            out = np.zeros((coords.shape[0], 6), dtype=np.float64)
            out[:, 0] = 99.0
            return out, None

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

        def transform_coordinates_native(
            self,
            coords: np.ndarray,
            representation_in: str,
            representation_out: str,
            frame_in: str,
            frame_out: str,
            origin_codes: list[str],
            target_origin: str | None,
            time_scale: str,
            time_days: np.ndarray,
            time_nanos: np.ndarray,
            covariances: np.ndarray | None = None,
            t0: np.ndarray | None = None,
            mu: np.ndarray | None = None,
            a: float | None = None,
            f: float | None = None,
            max_iter: int = 100,
            tol: float = 1e-15,
        ) -> tuple[np.ndarray, np.ndarray | None]:
            self.calls.append((representation_in, representation_out, mu is not None))
            out = np.zeros((coords.shape[0], 6), dtype=np.float64)
            out[:, 0] = 7.0
            return out, None

        def keplerian_to_cartesian_numpy(  # type: ignore[override]
            self, coords: np.ndarray, mu: np.ndarray, max_iter: int, tol: float
        ) -> np.ndarray:
            raise AssertionError(
                "unexpected ping-pong via keplerian_to_cartesian_numpy"
            )

    strict_native = _StrictRustNative()
    monkeypatch.setattr(rust_api, "_native", strict_native)

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
            "noncart-cometary-into-itrf93-cartesian",
            _make_cometary,
            CartesianCoordinates,
            "itrf93",
            OriginCodes.SUN,
        ),
        (
            "noncart-cometary-into-itrf93-spherical",
            _make_cometary,
            SphericalCoordinates,
            "itrf93",
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
    """Cases the native single-crossing path does not cover (non-Cartesian
    input into an ITRF93 frame change) must return None from the try-function
    so the public transform_coordinates uses the thin Python fallthrough."""
    coords = coords_factory()
    assert (
        _try_transform_coordinates_rust(
            coords, representation_out, frame_out, origin_out
        )
        is None
    ), f"{case_id}: try-function returned a value for a natively-uncovered path"


def test_rust_transform_covariance_present_uses_native_single_crossing():
    """The native path propagates covariance via forward-mode AD in Rust, so a
    covariance-carrying transform returns a non-null covariance without leaving
    Rust."""
    coords = _make_cartesian(with_covariance=True)
    result = _try_transform_coordinates_rust(
        coords, SphericalCoordinates, "ecliptic", OriginCodes.SUN
    )
    assert result is not None
    assert not result.covariance.is_all_nan()


def test_rust_transform_covers_cometary_natively():
    """Native coverage now includes Cometary in either direction (previously a
    legacy-only branch)."""
    assert (
        _try_transform_coordinates_rust(
            _make_cometary(), SphericalCoordinates, "ecliptic", OriginCodes.SUN
        )
        is not None
    )
    assert (
        _try_transform_coordinates_rust(
            _make_cartesian(), CometaryCoordinates, "ecliptic", OriginCodes.SUN
        )
        is not None
    )
