"""
Tests for Kepler clock gating functionality.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.geometry.clock_gating import (
    ClockGateConfig,
    ClockGateResults,
    apply_clock_gating,
    compute_orbital_positions_at_times,
)
from adam_core.geometry.rays import ObservationRays
from adam_core.observers.observers import Observers
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import sample_ellipse_adaptive
from adam_core.time import Timestamp


class TestClockGating:
    """Test clock gating functionality."""

    def test_empty_inputs(self):
        """Test clock gating with empty inputs."""
        config = ClockGateConfig.from_kwargs(
            max_angular_sep_arcsec=[60.0],
            max_radial_sep_au=[0.1],
            max_extrapolation_days=[30.0],
        )

        empty_obs = ObservationRays.empty()
        empty_params = sample_ellipse_adaptive(
            Orbits.empty(), max_chord_arcmin=1.0, max_segments_per_orbit=1024
        )[0]

        result = apply_clock_gating(empty_obs, empty_params, config)
        assert len(result) == 0

    def test_circular_orbit_clock_gating(self):
        """Test clock gating with a circular orbit."""
        # Create a circular orbit at 1 AU
        times = Timestamp.from_mjd([59000.0], scale="tdb")

        # Circular orbit at 1 AU with mean anomaly = 0 at epoch
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],  # At periapsis
            vx=[0.0],
            vy=[0.017202],
            vz=[0.0],  # Circular velocity
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
            frame="ecliptic",
        )

        orbit = Orbits.from_kwargs(
            orbit_id=["test_orbit"],
            coordinates=coords,
        )

        plane_params, _ = sample_ellipse_adaptive(
            orbit, max_chord_arcmin=1.0, max_segments_per_orbit=1024
        )

        # Create observation rays near the predicted orbital positions
        # Observe at epoch + 0.25 periods (90 degrees mean anomaly)
        # For 1 AU orbit: period ≈ 365.25 days, so 0.25*period ≈ 91.3 days
        obs_time = times.mjd().to_numpy()[0] + 91.3
        obs_times = Timestamp.from_mjd([obs_time], scale="tdb")

        # Compute predicted position direction using clock kernels
        a = plane_params.a.to_numpy()[0]
        e = plane_params.e.to_numpy()[0]
        M0 = plane_params.M0.to_numpy()[0]
        k = 0.01720209895
        n = k / (a * np.sqrt(a))
        elements = jnp.array([a, e, M0, n])
        p_hat = jnp.array(
            [
                plane_params.p_x.to_numpy()[0],
                plane_params.p_y.to_numpy()[0],
                plane_params.p_z.to_numpy()[0],
            ]
        )
        q_hat = jnp.array(
            [
                plane_params.q_x.to_numpy()[0],
                plane_params.q_y.to_numpy()[0],
                plane_params.q_z.to_numpy()[0],
            ]
        )
        bases = jnp.array([p_hat, q_hat])
        epoch_mjd = jnp.array(plane_params.t0.mjd().to_numpy()[0])
        obs_times_j = jnp.array([obs_time])
        positions, _, _ = compute_orbital_positions_at_times(
            elements, bases, epoch_mjd, obs_times_j
        )
        pred = np.array(positions[0])
        u = pred / np.linalg.norm(pred)

        # Create observation ray along predicted direction from SSB
        obs_rays = ObservationRays.from_kwargs(
            det_id=["det_001"],
            orbit_id=["test_orbit"],
            observer=Observers.from_kwargs(
                code=["500"],  # SSB
                coordinates=CartesianCoordinates.from_kwargs(
                    x=[0.0],
                    y=[0.0],
                    z=[0.0],
                    vx=[0.0],
                    vy=[0.0],
                    vz=[0.0],
                    time=obs_times,
                    origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
                    frame="ecliptic",
                ),
            ),
            u_x=[float(u[0])],
            u_y=[float(u[1])],
            u_z=[float(u[2])],
        )

        # Apply clock gating with reasonable tolerances
        config = ClockGateConfig.from_kwargs(
            max_angular_sep_arcsec=[3600.0],  # 1 degree
            max_radial_sep_au=[2.0],  # permissive due to unknown range along ray
            max_extrapolation_days=[100.0],  # 100 days
        )

        result = apply_clock_gating(obs_rays, plane_params, config)

        # Should have one result that passes
        assert len(result) == 1
        assert result.passed[0].as_py() == True
        assert result.det_id[0].as_py() == "det_001"
        assert result.orbit_id[0].as_py() == "test_orbit"

        # Check that angular separation is reasonable (should be small)
        angular_sep = result.angular_sep_arcsec[0].as_py()
        assert angular_sep < 3600.0  # Less than 1 degree

        # Check predicted anomalies: increment relative to M0 should be ~π/2
        pred_M = result.predicted_mean_anomaly[0].as_py()
        pred_f = result.predicted_true_anomaly[0].as_py()

        M0 = plane_params.M0.to_numpy()[0]
        k = 0.01720209895
        a = plane_params.a.to_numpy()[0]
        n = k / (a * np.sqrt(a))
        dt = obs_time - plane_params.t0.mjd().to_numpy()[0]
        expected_delta = n * dt

        def wrap_angle(x):
            return (x + np.pi) % (2 * np.pi) - np.pi

        assert abs(wrap_angle((pred_M - M0) - expected_delta)) < 0.5
        # For circular orbits, f ≈ M modulo 2π
        assert abs(wrap_angle(pred_f - pred_M)) < 1e-4

    def test_clock_gating_filtering(self):
        """Test that clock gating properly filters out bad matches."""
        # Create orbit at epoch
        times = Timestamp.from_mjd([59000.0], scale="tdb")

        coords = CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.017202],
            vz=[0.0],
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
            frame="ecliptic",
        )

        orbit = Orbits.from_kwargs(
            orbit_id=["test_orbit"],
            coordinates=coords,
        )

        plane_params, _ = sample_ellipse_adaptive(
            orbit, max_chord_arcmin=1.0, max_segments_per_orbit=1024
        )

        # Create observation rays: one good (predicted), one bad (opposite)
        obs_times = Timestamp.from_mjd([59000.0 + 91.3, 59000.0 + 91.3], scale="tdb")

        # Good observation: near predicted position
        # Bad observation: far from predicted position
        # Reuse elements/bases
        a = plane_params.a.to_numpy()[0]
        e = plane_params.e.to_numpy()[0]
        M0 = plane_params.M0.to_numpy()[0]
        k = 0.01720209895
        n = k / (a * np.sqrt(a))
        elements = jnp.array([a, e, M0, n])
        p_hat = jnp.array(
            [
                plane_params.p_x.to_numpy()[0],
                plane_params.p_y.to_numpy()[0],
                plane_params.p_z.to_numpy()[0],
            ]
        )
        q_hat = jnp.array(
            [
                plane_params.q_x.to_numpy()[0],
                plane_params.q_y.to_numpy()[0],
                plane_params.q_z.to_numpy()[0],
            ]
        )
        bases = jnp.array([p_hat, q_hat])
        epoch_mjd = jnp.array(plane_params.t0.mjd().to_numpy()[0])
        obs_times_j = jnp.array(
            [obs_times.mjd().to_numpy()[0], obs_times.mjd().to_numpy()[1]]
        )
        positions, _, _ = compute_orbital_positions_at_times(
            elements, bases, epoch_mjd, obs_times_j
        )
        pred = np.array(positions[0])
        u_good = pred / np.linalg.norm(pred)
        u_bad = -u_good

        obs_rays = ObservationRays.from_kwargs(
            det_id=["good_det", "bad_det"],
            orbit_id=["test_orbit", "test_orbit"],
            observer=Observers.from_kwargs(
                code=["500", "500"],
                coordinates=CartesianCoordinates.from_kwargs(
                    x=[0.0, 0.0],
                    y=[0.0, 0.0],
                    z=[0.0, 0.0],
                    vx=[0.0, 0.0],
                    vy=[0.0, 0.0],
                    vz=[0.0, 0.0],
                    time=obs_times,
                    origin=Origin.from_kwargs(
                        code=[OriginCodes.SUN.name, OriginCodes.SUN.name]
                    ),
                    frame="ecliptic",
                ),
            ),
            u_x=[float(u_good[0]), float(u_bad[0])],
            u_y=[float(u_good[1]), float(u_bad[1])],
            u_z=[float(u_good[2]), float(u_bad[2])],
        )

        # Apply strict clock gating
        config = ClockGateConfig.from_kwargs(
            max_angular_sep_arcsec=[1800.0],  # 0.5 degrees
            max_radial_sep_au=[2.0],  # permissive range tolerance
            max_extrapolation_days=[100.0],
        )

        result = apply_clock_gating(obs_rays, plane_params, config)

        # Should have 2 results, but only one should pass
        assert len(result) == 2
        passed_results = result.apply_mask(result.passed)
        assert len(passed_results) == 1
        assert passed_results.det_id[0].as_py() == "good_det"

    def test_time_extrapolation_limits(self):
        """Test that clock gating respects time extrapolation limits."""
        times = Timestamp.from_mjd([59000.0], scale="tdb")

        coords = CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.017202],
            vz=[0.0],
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
            frame="ecliptic",
        )

        orbit = Orbits.from_kwargs(
            orbit_id=["test_orbit"],
            coordinates=coords,
        )

        plane_params, _ = sample_ellipse_adaptive(
            orbit, max_chord_arcmin=1.0, max_segments_per_orbit=1024
        )

        # Create observation ray far in the future (beyond extrapolation limit)
        obs_time = times.mjd().to_numpy()[0] + 1000.0  # 1000 days later
        obs_times = Timestamp.from_mjd([obs_time], scale="tdb")

        obs_rays = ObservationRays.from_kwargs(
            det_id=["future_det"],
            orbit_id=["test_orbit"],
            observer=Observers.from_kwargs(
                code=["500"],
                coordinates=CartesianCoordinates.from_kwargs(
                    x=[0.0],
                    y=[0.0],
                    z=[0.0],
                    vx=[0.0],
                    vy=[0.0],
                    vz=[0.0],
                    time=obs_times,
                    origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
                    frame="ecliptic",
                ),
            ),
            u_x=[1.0],
            u_y=[0.0],
            u_z=[0.0],
        )

        # Apply clock gating with short extrapolation limit
        config = ClockGateConfig.from_kwargs(
            max_angular_sep_arcsec=[36000.0],  # Very permissive
            max_radial_sep_au=[1.0],  # Very permissive
            max_extrapolation_days=[100.0],  # But short time limit
        )

        result = apply_clock_gating(obs_rays, plane_params, config)

        # Should fail due to time extrapolation limit
        assert len(result) == 1
        assert result.passed[0].as_py() == False
        assert result.extrapolation_days[0].as_py() == 1000.0

    def test_compute_orbital_positions_at_times(self):
        """Test the low-level orbital position computation."""
        # Simple circular orbit parameters
        a, e, M0 = 1.0, 0.0, 0.0  # 1 AU circular orbit at periapsis
        k = 0.01720209895  # Gaussian constant
        n = k / (a * np.sqrt(a))  # Mean motion

        elements = jnp.array([a, e, M0, n])

        # Identity basis for simplicity (orbit in xy plane)
        p_hat = jnp.array([1.0, 0.0, 0.0])  # x direction
        q_hat = jnp.array([0.0, 1.0, 0.0])  # y direction
        bases = jnp.array([p_hat, q_hat])

        epoch_mjd = jnp.array(59000.0)

        # Test at epoch and quarter period later
        quarter_period_days = 365.25 / 4  # ~91.3 days
        obs_times = jnp.array([59000.0, 59000.0 + quarter_period_days])

        positions, mean_anomalies, true_anomalies = compute_orbital_positions_at_times(
            elements, bases, epoch_mjd, obs_times
        )

        # At epoch: should be at (1, 0, 0) with M=f=0
        np.testing.assert_allclose(positions[0], [1.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(mean_anomalies[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(true_anomalies[0], 0.0, atol=1e-10)

        # At quarter period: should be at (0, 1, 0) with M=f=π/2
        np.testing.assert_allclose(positions[1], [0.0, 1.0, 0.0], atol=5e-5)
        np.testing.assert_allclose(mean_anomalies[1], np.pi / 2, atol=5e-5)
        np.testing.assert_allclose(true_anomalies[1], np.pi / 2, atol=5e-5)

    def test_eccentric_orbit(self):
        """Test clock gating with an eccentric orbit."""
        times = Timestamp.from_mjd([59000.0], scale="tdb")

        # Eccentric orbit (e=0.5) at periapsis
        coords = CartesianCoordinates.from_kwargs(
            x=[0.5],
            y=[0.0],
            z=[0.0],  # At periapsis for a=1, e=0.5
            vx=[0.0],
            vy=[0.024248],
            vz=[0.0],  # Velocity at periapsis
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
            frame="ecliptic",
        )

        orbit = Orbits.from_kwargs(
            orbit_id=["eccentric_orbit"],
            coordinates=coords,
        )

        plane_params, _ = sample_ellipse_adaptive(
            orbit, max_chord_arcmin=1.0, max_segments_per_orbit=1024
        )

        # Create observation at half period (should be at apoapsis)
        half_period = 365.25 / 2
        obs_time = times.mjd().to_numpy()[0] + half_period
        obs_times = Timestamp.from_mjd([obs_time], scale="tdb")

        # Build observation ray along predicted direction at obs_time
        a = plane_params.a.to_numpy()[0]
        e = plane_params.e.to_numpy()[0]
        M0 = plane_params.M0.to_numpy()[0]
        k = 0.01720209895
        n = k / (a * np.sqrt(a))
        elements = jnp.array([a, e, M0, n])
        p_hat = jnp.array(
            [
                plane_params.p_x.to_numpy()[0],
                plane_params.p_y.to_numpy()[0],
                plane_params.p_z.to_numpy()[0],
            ]
        )
        q_hat = jnp.array(
            [
                plane_params.q_x.to_numpy()[0],
                plane_params.q_y.to_numpy()[0],
                plane_params.q_z.to_numpy()[0],
            ]
        )
        bases = jnp.array([p_hat, q_hat])
        epoch_mjd = jnp.array(plane_params.t0.mjd().to_numpy()[0])
        obs_times_j = jnp.array([obs_time])
        positions, _, _ = compute_orbital_positions_at_times(
            elements, bases, epoch_mjd, obs_times_j
        )
        pred = np.array(positions[0])
        u = pred / np.linalg.norm(pred)

        obs_rays = ObservationRays.from_kwargs(
            det_id=["apoapsis_det"],
            orbit_id=["eccentric_orbit"],
            observer=Observers.from_kwargs(
                code=["500"],
                coordinates=CartesianCoordinates.from_kwargs(
                    x=[0.0],
                    y=[0.0],
                    z=[0.0],
                    vx=[0.0],
                    vy=[0.0],
                    vz=[0.0],
                    time=obs_times,
                    origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
                    frame="ecliptic",
                ),
            ),
            u_x=[float(u[0])],
            u_y=[float(u[1])],
            u_z=[float(u[2])],
        )

        config = ClockGateConfig.from_kwargs(
            max_angular_sep_arcsec=[3600.0],  # 1 degree
            max_radial_sep_au=[0.2],  # 0.2 AU tolerance
            max_extrapolation_days=[200.0],  # 200 days
        )

        result = apply_clock_gating(obs_rays, plane_params, config)

        # Should pass the filter
        assert len(result) == 1
        assert result.passed[0].as_py() is True


class TestClockGateConfig:
    """Test ClockGateConfig table."""

    def test_config_creation(self):
        """Test creating configuration table."""
        config = ClockGateConfig.from_kwargs(
            max_angular_sep_arcsec=[3600.0],
            max_radial_sep_au=[0.1],
            max_extrapolation_days=[30.0],
        )

        assert len(config) == 1
        assert config.max_angular_sep_arcsec[0].as_py() == 3600.0
        assert config.max_radial_sep_au[0].as_py() == 0.1
        assert config.max_extrapolation_days[0].as_py() == 30.0

    def test_empty_config(self):
        """Test empty configuration table."""
        empty = ClockGateConfig.empty()
        assert len(empty) == 0


class TestClockGateResults:
    """Test ClockGateResults table."""

    def test_results_creation(self):
        """Test creating results table."""
        results = ClockGateResults.from_kwargs(
            det_id=["det_001", "det_002"],
            orbit_id=["orbit_A", "orbit_B"],
            passed=[True, False],
            angular_sep_arcsec=[100.0, 5000.0],
            radial_sep_au=[0.01, 0.5],
            extrapolation_days=[10.0, 50.0],
            predicted_mean_anomaly=[1.5, 3.0],
            predicted_true_anomaly=[1.6, 2.9],
        )

        assert len(results) == 2
        assert results.det_id[0].as_py() == "det_001"
        assert results.passed[0].as_py() == True
        assert results.passed[1].as_py() == False

    def test_empty_results(self):
        """Test empty results table."""
        empty = ClockGateResults.empty()
        assert len(empty) == 0
