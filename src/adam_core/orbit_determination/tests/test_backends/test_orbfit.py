"""
Unit and integration tests for OrbFitBackend.

The backend is a stub; the only substantive test is that it raises
``NotImplementedError`` when ``.fit()`` is called.
"""

import numpy as np
import pyarrow as pa
import pytest

from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbit_determination.backends import ORBFIT_AVAILABLE, OrbFitBackend
from adam_core.orbit_determination.config import BackendConfig
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.time import Timestamp


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


class TestOrbFitBackend:
    def test_backend_name(self):
        assert OrbFitBackend.BACKEND_NAME == "orbfit"

    def test_unavailable_raises_import_error(self):
        backend = OrbFitBackend()
        backend.AVAILABLE = False  # type: ignore[assignment]
        with pytest.raises(ImportError):
            backend.fit(_make_observations(), BackendConfig())

    @pytest.mark.skipif(not ORBFIT_AVAILABLE, reason="orbfit not installed")
    def test_fit_raises_not_implemented(self):
        """Even when orbfit is installed the stub raises NotImplementedError."""
        backend = OrbFitBackend()
        with pytest.raises(NotImplementedError):
            backend.fit(_make_observations(), BackendConfig())


@pytest.mark.skipif(not ORBFIT_AVAILABLE, reason="orbfit not installed")
class TestOrbFitBackendIntegration:
    def test_placeholder(self):
        """
        TODO(od-module): implement OrbFit backend and replace this placeholder
        with real integration tests.
        """
        pytest.skip("OrbFit backend not yet implemented.")
