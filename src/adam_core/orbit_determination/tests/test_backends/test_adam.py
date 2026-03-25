"""
Unit tests for AdamBackend.

The propagator and underlying IOD/DC functions are mocked so these tests
run without adam-assist or any other propagator installed.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbit_determination.backends.adam import AdamBackend
from adam_core.orbit_determination.config import BackendConfig, WeightingPolicy
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from adam_core.time import Timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_observations(n: int = 8) -> OrbitDeterminationObservations:
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


def _make_iod_result(n_orbits: int = 2, n_obs: int = 8) -> tuple:
    coords = CartesianCoordinates.from_kwargs(
        x=[1.0] * n_orbits,
        y=[0.0] * n_orbits,
        z=[0.0] * n_orbits,
        vx=[0.0] * n_orbits,
        vy=[0.01] * n_orbits,
        vz=[0.0] * n_orbits,
        time=Timestamp.from_mjd([59000.0] * n_orbits, scale="tdb"),
        origin=Origin.from_kwargs(code=pa.repeat("SUN", n_orbits)),
        frame="ecliptic",
    )
    iod_orbits = FittedOrbits.from_kwargs(
        orbit_id=[f"iod_{i}" for i in range(n_orbits)],
        object_id=pa.repeat(None, n_orbits),
        coordinates=coords,
        arc_length=pa.repeat(30.0, n_orbits),
        num_obs=pa.repeat(n_obs, n_orbits),
        chi2=pa.repeat(5.0, n_orbits),
        reduced_chi2=pa.repeat(0.5, n_orbits),
        success=pa.repeat(True, n_orbits),
        status_code=pa.repeat(0, n_orbits),
    )
    iod_members = FittedOrbitMembers.from_kwargs(
        orbit_id=pa.repeat("iod_0", n_obs),
        obs_id=[f"obs{i:03d}" for i in range(n_obs)],
        residuals=None,
        solution=pa.repeat(True, n_obs),
        outlier=pa.repeat(False, n_obs),
    )
    return iod_orbits, iod_members


def _make_dc_result(n_obs: int = 8) -> tuple:
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
    dc_orbits = FittedOrbits.from_kwargs(
        orbit_id=["dc_orbit_0"],
        object_id=pa.repeat(None, 1),
        coordinates=coords,
        arc_length=[30.0],
        num_obs=[n_obs],
        chi2=[4.8],
        reduced_chi2=[0.48],
        iterations=[10],
        success=[True],
        status_code=[0],
    )
    dc_members = FittedOrbitMembers.from_kwargs(
        orbit_id=pa.repeat("dc_orbit_0", n_obs),
        obs_id=[f"obs{i:03d}" for i in range(n_obs)],
        residuals=None,
        solution=pa.repeat(True, n_obs),
        outlier=pa.repeat(False, n_obs),
    )
    return dc_orbits, dc_members


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdamBackend:
    def test_always_available(self):
        assert AdamBackend.AVAILABLE is True

    def test_backend_name(self):
        assert AdamBackend.BACKEND_NAME == "adam"

    def test_missing_propagator_raises_value_error(self):
        obs = _make_observations()
        backend = AdamBackend()
        cfg = BackendConfig()  # no propagator in backend_kwargs
        with pytest.raises(ValueError, match="propagator"):
            backend.fit(obs, cfg)

    def test_successful_fit_shape(self):
        obs = _make_observations(n=8)
        iod_result = _make_iod_result(n_orbits=1, n_obs=8)
        dc_result = _make_dc_result(n_obs=8)
        mock_propagator = MagicMock()
        cfg = BackendConfig(backend_kwargs={"propagator": mock_propagator})

        with patch("adam_core.orbit_determination.backends.adam.iod", return_value=iod_result), \
             patch("adam_core.orbit_determination.backends.adam.differential_correction", return_value=dc_result):
            backend = AdamBackend()
            fitted, members = backend.fit(obs, cfg)

        assert isinstance(fitted, FittedOrbits)
        assert isinstance(members, FittedOrbitMembers)
        assert len(fitted) == 1
        assert len(members) == 8

    def test_provenance_set_after_fit(self):
        obs = _make_observations(n=8)
        iod_result = _make_iod_result(n_orbits=1, n_obs=8)
        dc_result = _make_dc_result(n_obs=8)
        mock_propagator = MagicMock()
        cfg = BackendConfig(backend_kwargs={"propagator": mock_propagator})

        with patch("adam_core.orbit_determination.backends.adam.iod", return_value=iod_result), \
             patch("adam_core.orbit_determination.backends.adam.differential_correction", return_value=dc_result):
            backend = AdamBackend()
            fitted, _ = backend.fit(obs, cfg)

        assert fitted.backend[0].as_py() == "adam"
        assert fitted.backend_version[0].as_py() is not None

    def test_iod_no_candidates_raises_value_error(self):
        obs = _make_observations(n=8)
        empty_iod = (FittedOrbits.empty(), FittedOrbitMembers.empty())
        mock_propagator = MagicMock()
        cfg = BackendConfig(backend_kwargs={"propagator": mock_propagator})

        with patch("adam_core.orbit_determination.backends.adam.iod", return_value=empty_iod):
            backend = AdamBackend()
            with pytest.raises(ValueError, match="IOD found no valid orbit"):
                backend.fit(obs, cfg)

    def test_dc_no_orbits_raises_runtime_error(self):
        obs = _make_observations(n=8)
        iod_result = _make_iod_result(n_orbits=1, n_obs=8)
        empty_dc = (FittedOrbits.empty(), FittedOrbitMembers.empty())
        mock_propagator = MagicMock()
        cfg = BackendConfig(backend_kwargs={"propagator": mock_propagator})

        with patch("adam_core.orbit_determination.backends.adam.iod", return_value=iod_result), \
             patch("adam_core.orbit_determination.backends.adam.differential_correction", return_value=empty_dc):
            backend = AdamBackend()
            with pytest.raises(RuntimeError, match="differential correction produced no orbits"):
                backend.fit(obs, cfg)

    def test_backend_kwargs_forwarded_to_iod(self):
        obs = _make_observations(n=8)
        iod_result = _make_iod_result(n_orbits=1, n_obs=8)
        dc_result = _make_dc_result(n_obs=8)
        mock_propagator = MagicMock()
        cfg = BackendConfig(backend_kwargs={
            "propagator": mock_propagator,
            "min_obs": 4,
            "rchi2_threshold": 500.0,
        })

        with patch("adam_core.orbit_determination.backends.adam.iod", return_value=iod_result) as mock_iod, \
             patch("adam_core.orbit_determination.backends.adam.differential_correction", return_value=dc_result):
            backend = AdamBackend()
            backend.fit(obs, cfg)

        call_kwargs = mock_iod.call_args[1]
        assert call_kwargs["min_obs"] == 4
        assert call_kwargs["rchi2_threshold"] == 500.0

    def test_weighting_policy_has_no_effect(self):
        """AdamBackend always computes residuals; WeightingPolicy is ignored."""
        obs = _make_observations(n=8)
        iod_result = _make_iod_result(n_orbits=1, n_obs=8)
        dc_result = _make_dc_result(n_obs=8)
        mock_propagator = MagicMock()

        for policy in (WeightingPolicy.DELEGATE, WeightingPolicy.ADAM):
            cfg = BackendConfig(
                weighting_policy=policy,
                backend_kwargs={"propagator": mock_propagator},
            )
            with patch("adam_core.orbit_determination.backends.adam.iod", return_value=iod_result), \
                 patch("adam_core.orbit_determination.backends.adam.differential_correction", return_value=dc_result):
                backend = AdamBackend()
                fitted, _ = backend.fit(obs, cfg)
            assert fitted.backend[0].as_py() == "adam"
