"""
Unit tests for FindOrbBackend.

All calls to ``adam_fo.fo`` are mocked so these tests run without a Find_Orb
binary installed.  Integration tests that require the actual binary are gated
behind ``@pytest.mark.skipif(not FIND_ORB_AVAILABLE, ...)``.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observations.ades import ADESObservations
from adam_core.observers import Observers
from adam_core.orbit_determination.backends import FIND_ORB_AVAILABLE, FindOrbBackend
from adam_core.orbit_determination.backends.find_orb import _observations_to_ades
from adam_core.orbit_determination.config import BackendConfig, WeightingPolicy
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from adam_core.orbits import Orbits
from adam_core.time import Timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_observations(n: int = 6) -> OrbitDeterminationObservations:
    times = Timestamp.from_mjd(np.linspace(59000, 59030, n), scale="utc")
    sigmas = np.full((n, 6), np.nan)
    sigmas[:, 1] = 1.0 / 3600.0
    sigmas[:, 2] = 1.0 / 3600.0
    coords = SphericalCoordinates.from_kwargs(
        lon=np.full(n, 180.0),
        lat=np.zeros(n),
        time=times,
        origin=Origin.from_kwargs(code=pa.repeat("X05", n)),
        frame="equatorial",
        covariance=CoordinateCovariances.from_sigmas(sigmas),
    )
    observers = Observers.from_codes(codes=pa.repeat("X05", n), times=times)
    return OrbitDeterminationObservations.from_kwargs(
        id=[f"obs{i:03d}" for i in range(n)],
        coordinates=coords,
        observers=observers,
    )


def _make_raw_orbit(orbit_id: str = "obs000") -> Orbits:
    """Return a minimal Orbits object as adam_fo.fo would produce."""
    from adam_core.coordinates import CartesianCoordinates
    from adam_core.coordinates.covariances import CoordinateCovariances

    cov = CoordinateCovariances.from_matrix(np.eye(6).reshape(1, 6, 6) * 1e-10)
    coords = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.01],
        vz=[0.0],
        time=Timestamp.from_mjd([59000.0], scale="tt"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
        covariance=cov,
    )
    return Orbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[orbit_id],
        coordinates=coords,
    )


# ---------------------------------------------------------------------------
# _observations_to_ades
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FIND_ORB_AVAILABLE, reason="adam_fo not installed")
class TestObservationsToAdes:
    def test_returns_string_and_table(self):
        obs = _make_observations(n=3)
        ades_str, ades_table = _observations_to_ades(obs)
        assert isinstance(ades_str, str)
        assert len(ades_str) > 0
        assert isinstance(ades_table, ADESObservations)
        assert len(ades_table) == 3

    def test_trk_sub_truncated_to_8_chars(self):
        obs = _make_observations(n=3)
        _, ades_table = _observations_to_ades(obs)
        for trk in ades_table.trkSub.to_pylist():
            assert len(trk) <= 8

    def test_sigma_converted_to_arcsec(self):
        """1-degree sigma in SphericalCoordinates → 3600 arcsec in ADES."""
        n = 3
        times = Timestamp.from_mjd(np.linspace(59000, 59010, n), scale="utc")
        sigmas = np.zeros((n, 6))
        sigmas[:, 1] = 1.0  # 1 degree RA
        sigmas[:, 2] = 1.0  # 1 degree Dec
        coords = SphericalCoordinates.from_kwargs(
            lon=np.zeros(n),
            lat=np.zeros(n),  # cos(0)=1 so RAcosDec = RA
            time=times,
            origin=Origin.from_kwargs(code=pa.repeat("X05", n)),
            frame="equatorial",
            covariance=CoordinateCovariances.from_sigmas(sigmas),
        )
        observers = Observers.from_codes(codes=pa.repeat("X05", n), times=times)
        obs = OrbitDeterminationObservations.from_kwargs(
            id=[f"obs{i}" for i in range(n)],
            coordinates=coords,
            observers=observers,
        )
        _, ades = _observations_to_ades(obs)
        assert abs(ades.rmsRACosDec[0].as_py() - 3600.0) < 1e-6
        assert abs(ades.rmsDec[0].as_py() - 3600.0) < 1e-6

    def test_empty_raises(self):
        empty = OrbitDeterminationObservations.empty()
        with pytest.raises(ValueError, match="empty"):
            _observations_to_ades(empty)


# ---------------------------------------------------------------------------
# FindOrbBackend.fit — mocked
# ---------------------------------------------------------------------------


class TestFindOrbBackendMocked:
    """Tests that mock adam_fo.fo — run without a binary."""

    def _patch_fo(self, orbit_raw, rejected_raw=None, error=None):
        if rejected_raw is None:
            rejected_raw = ADESObservations.empty()
        return patch(
            "adam_core.orbit_determination.backends.find_orb._fo",
            return_value=(orbit_raw, rejected_raw, error),
        )

    @pytest.mark.skipif(not FIND_ORB_AVAILABLE, reason="adam_fo not installed")
    def test_successful_fit_returns_fitted_orbits(self):
        obs = _make_observations(n=6)
        orbit_raw = _make_raw_orbit("obs000")

        with self._patch_fo(orbit_raw):
            backend = FindOrbBackend()
            fitted, members = backend.fit(obs, BackendConfig())

        assert isinstance(fitted, FittedOrbits)
        assert isinstance(members, FittedOrbitMembers)
        assert len(fitted) == 1
        assert len(members) == 6

    @pytest.mark.skipif(not FIND_ORB_AVAILABLE, reason="adam_fo not installed")
    def test_provenance_set(self):
        obs = _make_observations(n=6)
        orbit_raw = _make_raw_orbit("obs000")

        with self._patch_fo(orbit_raw):
            backend = FindOrbBackend()
            fitted, _ = backend.fit(obs, BackendConfig())

        assert fitted.backend[0].as_py() == "find_orb"
        # backend_version may be None if adam-fo metadata is absent — just
        # check the column exists.
        assert "backend_version" in fitted.schema.names

    @pytest.mark.skipif(not FIND_ORB_AVAILABLE, reason="adam_fo not installed")
    def test_fo_error_raises_runtime_error(self):
        obs = _make_observations(n=6)

        with self._patch_fo(Orbits.empty(), error="Find_Orb failed"):
            backend = FindOrbBackend()
            with pytest.raises(RuntimeError, match="Find_Orb failed"):
                backend.fit(obs, BackendConfig())

    @pytest.mark.skipif(not FIND_ORB_AVAILABLE, reason="adam_fo not installed")
    def test_delegate_mode_chi2_nan(self):
        obs = _make_observations(n=6)
        orbit_raw = _make_raw_orbit("obs000")

        with self._patch_fo(orbit_raw):
            backend = FindOrbBackend()
            fitted, _ = backend.fit(obs, BackendConfig())

        import math
        assert math.isnan(fitted.chi2[0].as_py())
        assert math.isnan(fitted.reduced_chi2[0].as_py())

    @pytest.mark.skipif(not FIND_ORB_AVAILABLE, reason="adam_fo not installed")
    def test_delegate_mode_residuals_none(self):
        obs = _make_observations(n=6)
        orbit_raw = _make_raw_orbit("obs000")

        with self._patch_fo(orbit_raw):
            backend = FindOrbBackend()
            _, members = backend.fit(obs, BackendConfig())

        # In DELEGATE mode residuals column should be all-null
        assert members.residuals is None or all(
            v is None for v in members.residuals.chi2.to_pylist()
        )

    @pytest.mark.skipif(not FIND_ORB_AVAILABLE, reason="adam_fo not installed")
    def test_outlier_flags_set_from_rejected(self):
        obs = _make_observations(n=6)
        orbit_raw = _make_raw_orbit("obs000")

        # Mark the second observation as rejected by Find_Orb
        t = obs.coordinates.time
        rejected = ADESObservations.from_kwargs(
            trkSub=["obs000"],
            obsTime=Timestamp.from_mjd([t.mjd()[1].as_py()], scale="utc"),
            ra=[180.0],
            dec=[0.0],
            stn=["X05"],
            mode=["NA"],
            astCat=["NA"],
        )

        with self._patch_fo(orbit_raw, rejected_raw=rejected):
            backend = FindOrbBackend()
            _, members = backend.fit(obs, BackendConfig())

        outliers = members.outlier.to_pylist()
        assert outliers[1] is True
        assert sum(outliers) == 1

    def test_unavailable_raises_import_error(self):
        """Finding the backend available flag False should raise ImportError."""
        backend = FindOrbBackend()
        backend.AVAILABLE = False  # type: ignore[assignment]
        with pytest.raises(ImportError):
            backend.fit(_make_observations(), BackendConfig())

    @pytest.mark.skipif(not FIND_ORB_AVAILABLE, reason="adam_fo not installed")
    def test_adam_weighting_requires_propagator(self):
        obs = _make_observations(n=6)
        orbit_raw = _make_raw_orbit("obs000")
        cfg = BackendConfig(weighting_policy=WeightingPolicy.ADAM)

        with self._patch_fo(orbit_raw):
            backend = FindOrbBackend()
            with pytest.raises(ValueError, match="propagator"):
                backend.fit(obs, cfg)


# ---------------------------------------------------------------------------
# Integration test — requires find_orb binary
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FIND_ORB_AVAILABLE, reason="find_orb not installed")
class TestFindOrbBackendIntegration:
    """
    Integration tests that call the real Find_Orb binary.

    These tests use fixtures stored in tests/data/od/.  They are skipped
    automatically when ``adam_fo`` is not installed.
    """

    def test_placeholder(self):
        """
        Placeholder — replace with real integration test once fixture data
        under tests/data/od/ is available.

        TODO(od-module): add a real observation fixture (5–20 obs for a known
        object) and assert:
          - len(fitted) >= 1
          - fitted.backend[0] == "find_orb"
          - chi2 / reduced_chi2 within expected bounds (DELEGATE: NaN; ADAM: finite)
          - residuals within expected bounds in ADAM mode
        """
        pytest.skip("Integration fixtures not yet populated under tests/data/od/")
