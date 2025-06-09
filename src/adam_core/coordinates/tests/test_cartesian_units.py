"""
Tests for the unit conversion methods on CartesianCoordinates class.
"""

import numpy as np
import pytest

from adam_core.constants import KM_P_AU, S_P_DAY
from adam_core.coordinates import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.time import Timestamp


class TestCartesianCoordinatesUnitMethods:
    """Test the km conversion methods on CartesianCoordinates."""

    @pytest.fixture
    def sample_coordinates(self):
        """Create sample CartesianCoordinates for testing."""
        # Create coordinates at 1 AU from Sun with 1 AU/day velocity
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0],
            y=[0.0, -0.5],
            z=[0.0, 1.0],
            vx=[0.0, 0.01],
            vy=[0.01, 0.0],
            vz=[0.0, -0.005],
            time=Timestamp.from_mjd([60000.0, 60001.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="equatorial",
        )
        return coords

    @pytest.fixture
    def sample_coordinates_with_covariance(self):
        """Create sample CartesianCoordinates with covariance for testing."""
        # Create a simple diagonal covariance matrix
        N = 2
        cov_matrices = np.zeros((N, 6, 6))

        # Set diagonal elements: 0.1 AU position uncertainty, 0.001 AU/day velocity uncertainty
        for i in range(N):
            for j in range(3):
                cov_matrices[i, j, j] = 0.01  # 0.1² AU²
                cov_matrices[i, j + 3, j + 3] = 0.000001  # 0.001² (AU/day)²

        covariance = CoordinateCovariances.from_matrix(cov_matrices)

        coords = CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0],
            y=[0.0, -0.5],
            z=[0.0, 1.0],
            vx=[0.0, 0.01],
            vy=[0.01, 0.0],
            vz=[0.0, -0.005],
            time=Timestamp.from_mjd([60000.0, 60001.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="equatorial",
            covariance=covariance,
        )
        return coords

    def test_values_km_property(self, sample_coordinates):
        """Test the values_km property."""
        values_km = sample_coordinates.values_km
        values_au = sample_coordinates.values

        # Check shape is preserved
        assert values_km.shape == values_au.shape

        # Check position conversion (first 3 columns)
        expected_positions_km = values_au[:, :3] * KM_P_AU
        np.testing.assert_allclose(values_km[:, :3], expected_positions_km)

        # Check velocity conversion (last 3 columns)
        expected_velocities_km_s = values_au[:, 3:] * KM_P_AU / S_P_DAY
        np.testing.assert_allclose(values_km[:, 3:], expected_velocities_km_s)

    def test_r_km_property(self, sample_coordinates):
        """Test the r_km property."""
        r_km = sample_coordinates.r_km
        r_au = sample_coordinates.r

        # Check conversion
        expected_r_km = r_au * KM_P_AU
        np.testing.assert_allclose(r_km, expected_r_km)

        # Check specific values
        assert np.isclose(r_km[0, 0], KM_P_AU)  # 1 AU -> KM_P_AU km
        assert np.isclose(r_km[1, 0], 2 * KM_P_AU)  # 2 AU -> 2*KM_P_AU km

    def test_v_km_s_property(self, sample_coordinates):
        """Test the v_km_s property."""
        v_km_s = sample_coordinates.v_km_s
        v_au_day = sample_coordinates.v

        # Check conversion
        expected_v_km_s = v_au_day * KM_P_AU / S_P_DAY
        np.testing.assert_allclose(v_km_s, expected_v_km_s)

        # Check specific values
        expected_01_au_day_to_km_s = 0.01 * KM_P_AU / S_P_DAY
        assert np.isclose(v_km_s[0, 1], expected_01_au_day_to_km_s)

    def test_covariance_km_method_with_covariance(
        self, sample_coordinates_with_covariance
    ):
        """Test the covariance_km method with actual covariance data."""
        cov_km = sample_coordinates_with_covariance.covariance_km()
        cov_au = sample_coordinates_with_covariance.covariance.to_matrix()

        # Check shape is preserved
        assert cov_km.shape == cov_au.shape

        # Check specific diagonal elements
        # Position variance: 0.01 AU² -> 0.01 * KM_P_AU² km²
        expected_pos_var_km = 0.01 * KM_P_AU**2
        for i in range(2):  # Two coordinate sets
            for j in range(3):  # x, y, z
                assert np.isclose(cov_km[i, j, j], expected_pos_var_km)

        # Velocity variance: 0.000001 (AU/day)² -> 0.000001 * (KM_P_AU/S_P_DAY)² (km/s)²
        expected_vel_var_km_s = 0.000001 * (KM_P_AU / S_P_DAY) ** 2
        for i in range(2):
            for j in range(3, 6):  # vx, vy, vz
                assert np.isclose(cov_km[i, j, j], expected_vel_var_km_s)

    def test_covariance_km_method_with_nan_covariance(self, sample_coordinates):
        """Test the covariance_km method when covariance is all NaN."""
        cov_km = sample_coordinates.covariance_km()
        cov_au = sample_coordinates.covariance.to_matrix()

        # Should return the same NaN matrix since no conversion is applied
        np.testing.assert_array_equal(cov_km, cov_au)

    def test_values_km_consistency_with_individual_properties(self, sample_coordinates):
        """Test that values_km is consistent with r_km and v_km_s properties."""
        values_km = sample_coordinates.values_km
        r_km = sample_coordinates.r_km
        v_km_s = sample_coordinates.v_km_s

        # Check that values_km[:, :3] matches r_km
        np.testing.assert_allclose(values_km[:, :3], r_km)

        # Check that values_km[:, 3:] matches v_km_s
        np.testing.assert_allclose(values_km[:, 3:], v_km_s)

    def test_unit_conversion_preserves_array_structure(self, sample_coordinates):
        """Test that unit conversion preserves the array structure."""
        original_shape = sample_coordinates.values.shape

        # All km properties should preserve the original shape
        assert sample_coordinates.values_km.shape == original_shape
        assert sample_coordinates.r_km.shape == (original_shape[0], 3)
        assert sample_coordinates.v_km_s.shape == (original_shape[0], 3)

    def test_unit_conversion_with_zero_values(self):
        """Test unit conversion with coordinates containing zeros."""
        coords = CartesianCoordinates.from_kwargs(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=Timestamp.from_mjd([60000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="equatorial",
        )

        # All km values should be zero
        np.testing.assert_allclose(coords.values_km, np.zeros((1, 6)))
        np.testing.assert_allclose(coords.r_km, np.zeros((1, 3)))
        np.testing.assert_allclose(coords.v_km_s, np.zeros((1, 3)))

    def test_unit_conversion_with_negative_values(self):
        """Test unit conversion with negative coordinate values."""
        coords = CartesianCoordinates.from_kwargs(
            x=[-1.0],
            y=[-0.5],
            z=[0.0],
            vx=[-0.01],
            vy=[0.0],
            vz=[0.005],
            time=Timestamp.from_mjd([60000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="equatorial",
        )

        # Check that negative values are preserved
        assert coords.values_km[0, 0] == -KM_P_AU  # -1 AU -> -KM_P_AU km
        assert coords.values_km[0, 1] == -0.5 * KM_P_AU  # -0.5 AU -> -0.5*KM_P_AU km
        assert (
            coords.values_km[0, 3] == -0.01 * KM_P_AU / S_P_DAY
        )  # -0.01 AU/day -> negative km/s


class TestCartesianCoordinatesIntegration:
    """Integration tests for CartesianCoordinates unit conversion with other functionality."""

    def test_round_trip_conversion_conceptual(self):
        """
        Test that we can conceptually do a round-trip conversion.

        Note: We don't have km->AU methods on CartesianCoordinates (by design),
        so this tests the conceptual round-trip using the utility functions.
        """
        from adam_core.coordinates.units import convert_cartesian_values_km_to_au

        # Create original coordinates
        original_coords = CartesianCoordinates.from_kwargs(
            x=[1.5, -2.0],
            y=[0.8, 0.0],
            z=[-0.3, 1.2],
            vx=[0.005, -0.01],
            vy=[-0.002, 0.008],
            vz=[0.001, 0.0],
            time=Timestamp.from_mjd([60000.0, 60001.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="equatorial",
        )

        # Convert to km using the new methods
        values_km = original_coords.values_km

        # Convert back to AU using utility functions
        values_au_round_trip = convert_cartesian_values_km_to_au(values_km)

        # Should match original values
        np.testing.assert_allclose(
            values_au_round_trip, original_coords.values, rtol=1e-15
        )

    def test_covariance_round_trip_conversion(self):
        """Test round-trip conversion of covariance matrices."""
        from adam_core.coordinates.units import convert_cartesian_covariance_km_to_au

        # Create coordinates with covariance
        N = 2
        cov_matrices = np.random.rand(N, 6, 6) * 0.01

        # Make symmetric and positive definite
        for i in range(N):
            cov_matrices[i] = (cov_matrices[i] + cov_matrices[i].T) / 2
            cov_matrices[i] += (
                np.eye(6) * 0.001
            )  # Add small diagonal term for positive definiteness

        covariance = CoordinateCovariances.from_matrix(cov_matrices)

        coords = CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0],
            y=[0.0, -0.5],
            z=[0.0, 1.0],
            vx=[0.0, 0.01],
            vy=[0.01, 0.0],
            vz=[0.0, -0.005],
            time=Timestamp.from_mjd([60000.0, 60001.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="equatorial",
            covariance=covariance,
        )

        # Convert to km using the new method
        cov_km = coords.covariance_km()

        # Convert back to AU using utility function
        cov_au_round_trip = convert_cartesian_covariance_km_to_au(cov_km)

        # Should match original covariance (within numerical precision)
        np.testing.assert_allclose(cov_au_round_trip, cov_matrices, rtol=1e-13)
