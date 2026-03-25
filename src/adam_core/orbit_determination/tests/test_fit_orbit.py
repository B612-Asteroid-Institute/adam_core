"""
Unit tests for fit_orbit().

All backend calls are mocked so these tests exercise input validation and
dispatch logic independently of any external binary or propagator.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.coordinates.residuals import Residuals
from adam_core.observers import Observers
from adam_core.orbit_determination.backends import FIND_ORB_AVAILABLE
from adam_core.orbit_determination.config import BackendConfig, WeightingPolicy
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbit_determination.fit_orbit import fit_orbit
from adam_core.orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from adam_core.time import Timestamp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_observations(n: int = 6, missing_cov: bool = False) -> OrbitDeterminationObservations:
    """Build a minimal but valid OrbitDeterminationObservations for testing."""
    times = Timestamp.from_mjd(np.linspace(59000, 59030, n), scale="utc")
    sigmas = np.full((n, 6), np.nan)
    if not missing_cov:
        sigmas[:, 1] = 1.0 / 3600.0  # 1 arcsec in degrees
        sigmas[:, 2] = 1.0 / 3600.0
    coords = SphericalCoordinates.from_kwargs(
        lon=np.full(n, 180.0),
        lat=np.zeros(n),
        time=times,
        origin=Origin.from_kwargs(code=pa.repeat("X05", n)),
        frame="equatorial",
        covariance=CoordinateCovariances.from_sigmas(sigmas),
    )
    observers = Observers.from_codes(
        codes=pa.repeat("X05", n),
        times=times,
    )
    return OrbitDeterminationObservations.from_kwargs(
        id=[f"obs{i:03d}" for i in range(n)],
        coordinates=coords,
        observers=observers,
    )


def _make_fitted_orbits() -> FittedOrbits:
    from adam_core.coordinates import CartesianCoordinates
    from adam_core.coordinates.origin import Origin

    coords = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.01],
        vz=[0.0],
        time=Timestamp.from_mjd([59000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )
    return FittedOrbits.from_kwargs(
        orbit_id=["test_orbit_01"],
        object_id=pa.repeat(None, 1),
        coordinates=coords,
        arc_length=[30.0],
        num_obs=[6],
        chi2=[float("nan")],
        reduced_chi2=[float("nan")],
        success=[True],
        status_code=[0],
        backend=["find_orb"],
        backend_version=["1.0.0"],
    )


def _make_fitted_members(orbit_id: str = "test_orbit_01", n: int = 6) -> FittedOrbitMembers:
    return FittedOrbitMembers.from_kwargs(
        orbit_id=pa.repeat(orbit_id, n),
        obs_id=[f"obs{i:03d}" for i in range(n)],
        residuals=None,
        solution=pa.repeat(True, n),
        outlier=pa.repeat(False, n),
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestFitOrbitValidation:
    def test_empty_observations_raises(self):
        empty = OrbitDeterminationObservations.empty()
        with pytest.raises(ValueError, match="0 observations"):
            fit_orbit(empty, backend="find_orb")

    def test_all_missing_covariance_raises(self):
        obs = _make_observations(n=6, missing_cov=True)
        with pytest.raises(ValueError, match="missing RA/Dec covariance"):
            fit_orbit(obs, backend="find_orb")

    def test_missing_observer_states_raises(self):
        """If all observer x-coords are NaN the call should fail."""
        obs = _make_observations(n=6)
        # Manually zero out observer coordinates by building a fake observers table
        from adam_core.coordinates import CartesianCoordinates

        bad_coords = CartesianCoordinates.from_kwargs(
            x=pa.repeat(float("nan"), 6),
            y=pa.repeat(float("nan"), 6),
            z=pa.repeat(float("nan"), 6),
            vx=pa.repeat(float("nan"), 6),
            vy=pa.repeat(float("nan"), 6),
            vz=pa.repeat(float("nan"), 6),
            time=obs.observers.coordinates.time,
            origin=obs.observers.coordinates.origin,
            frame="equatorial",
        )
        bad_observers = Observers.from_kwargs(
            code=obs.observers.code,
            coordinates=bad_coords,
        )
        bad_obs = OrbitDeterminationObservations.from_kwargs(
            id=obs.id,
            coordinates=obs.coordinates,
            observers=bad_observers,
        )
        with pytest.raises(ValueError, match="observer states"):
            fit_orbit(bad_obs, backend="find_orb")

    def test_unknown_backend_raises(self):
        obs = _make_observations()
        with pytest.raises(ValueError, match="Unknown backend"):
            fit_orbit(obs, backend="not_a_backend")  # type: ignore[arg-type]

    def test_unavailable_backend_raises_import_error(self):
        obs = _make_observations()
        # Temporarily pretend orbfit is unavailable
        with patch(
            "adam_core.orbit_determination.fit_orbit._BACKEND_AVAILABLE",
            {"adam": True, "find_orb": False, "orbfit": False, "layup": False},
        ):
            with pytest.raises(ImportError, match="pip install"):
                fit_orbit(obs, backend="find_orb")


# ---------------------------------------------------------------------------
# Dispatch — backend is mocked, output normalisation is tested
# ---------------------------------------------------------------------------


class TestFitOrbitDispatch:
    def test_dispatches_to_find_orb_backend(self):
        obs = _make_observations()
        mock_fitted = _make_fitted_orbits()
        mock_members = _make_fitted_members()

        with patch(
            "adam_core.orbit_determination.fit_orbit.FindOrbBackend"
        ) as MockCls:
            instance = MagicMock()
            instance.fit.return_value = (mock_fitted, mock_members)
            MockCls.return_value = instance

            with patch(
                "adam_core.orbit_determination.fit_orbit._BACKEND_AVAILABLE",
                {"adam": True, "find_orb": True, "orbfit": False, "layup": False},
            ):
                result_orbits, result_members = fit_orbit(obs, backend="find_orb")

            instance.fit.assert_called_once()
            assert isinstance(result_orbits, FittedOrbits)
            assert isinstance(result_members, FittedOrbitMembers)

    def test_dispatches_to_adam_backend(self):
        obs = _make_observations()
        mock_fitted = _make_fitted_orbits()
        mock_members = _make_fitted_members()

        with patch(
            "adam_core.orbit_determination.fit_orbit.AdamBackend"
        ) as MockCls:
            instance = MagicMock()
            instance.fit.return_value = (mock_fitted, mock_members)
            MockCls.return_value = instance

            result_orbits, result_members = fit_orbit(obs, backend="adam")

            instance.fit.assert_called_once()
            assert len(result_orbits) == 1

    def test_default_config_is_used_when_none(self):
        obs = _make_observations()
        mock_fitted = _make_fitted_orbits()
        mock_members = _make_fitted_members()

        with patch(
            "adam_core.orbit_determination.fit_orbit.AdamBackend"
        ) as MockCls:
            instance = MagicMock()
            instance.fit.return_value = (mock_fitted, mock_members)
            MockCls.return_value = instance

            fit_orbit(obs, backend="adam", config=None)

            call_args = instance.fit.call_args
            passed_config = call_args[0][1]
            assert isinstance(passed_config, BackendConfig)
            assert passed_config.weighting_policy is WeightingPolicy.DELEGATE

    def test_custom_config_is_forwarded(self):
        obs = _make_observations()
        mock_fitted = _make_fitted_orbits()
        mock_members = _make_fitted_members()
        cfg = BackendConfig(
            weighting_policy=WeightingPolicy.ADAM,
            backend_kwargs={"propagator": object},
        )

        with patch(
            "adam_core.orbit_determination.fit_orbit.AdamBackend"
        ) as MockCls:
            instance = MagicMock()
            instance.fit.return_value = (mock_fitted, mock_members)
            MockCls.return_value = instance

            fit_orbit(obs, backend="adam", config=cfg)

            call_args = instance.fit.call_args
            passed_config = call_args[0][1]
            assert passed_config.weighting_policy is WeightingPolicy.ADAM

    def test_output_shapes(self):
        obs = _make_observations(n=10)
        mock_fitted = _make_fitted_orbits()
        mock_members = _make_fitted_members(n=10)

        with patch(
            "adam_core.orbit_determination.fit_orbit.AdamBackend"
        ) as MockCls:
            instance = MagicMock()
            instance.fit.return_value = (mock_fitted, mock_members)
            MockCls.return_value = instance

            fitted, members = fit_orbit(obs, backend="adam")

        assert len(fitted) == 1
        assert len(members) == 10

    def test_provenance_columns_present(self):
        obs = _make_observations()
        mock_fitted = _make_fitted_orbits()
        mock_members = _make_fitted_members()

        with patch(
            "adam_core.orbit_determination.fit_orbit.AdamBackend"
        ) as MockCls:
            instance = MagicMock()
            instance.fit.return_value = (mock_fitted, mock_members)
            MockCls.return_value = instance

            fitted, _ = fit_orbit(obs, backend="adam")

        # Provenance columns must exist (may be null if backend didn't set them)
        assert "backend" in fitted.schema.names
        assert "backend_version" in fitted.schema.names
