"""
Tests for unit conversion utilities.
"""

import numpy as np

from adam_core.constants import KM_P_AU, S_P_DAY
from adam_core.coordinates.units import (
    au_per_day_to_km_per_s,
    au_to_km,
    convert_cartesian_covariance_au_to_km,
    convert_cartesian_covariance_km_to_au,
    convert_cartesian_values_au_to_km,
    convert_cartesian_values_km_to_au,
    km_per_s_to_au_per_day,
    km_to_au,
)


class TestBasicUnitConversions:
    """Test basic scalar and array unit conversion functions."""

    def test_au_to_km_scalar(self):
        """Test AU to km conversion with scalar input."""
        result = au_to_km(1.0)
        expected = KM_P_AU
        assert np.isclose(result, expected)

    def test_au_to_km_array(self):
        """Test AU to km conversion with array input."""
        input_au = np.array([1.0, 2.0, 0.5])
        result = au_to_km(input_au)
        expected = input_au * KM_P_AU
        np.testing.assert_allclose(result, expected)

    def test_km_to_au_scalar(self):
        """Test km to AU conversion with scalar input."""
        result = km_to_au(KM_P_AU)
        expected = 1.0
        assert np.isclose(result, expected)

    def test_km_to_au_array(self):
        """Test km to AU conversion with array input."""
        input_km = np.array([KM_P_AU, 2 * KM_P_AU, 0.5 * KM_P_AU])
        result = km_to_au(input_km)
        expected = np.array([1.0, 2.0, 0.5])
        np.testing.assert_allclose(result, expected)

    def test_au_per_day_to_km_per_s_scalar(self):
        """Test AU/day to km/s conversion with scalar input."""
        result = au_per_day_to_km_per_s(1.0)
        expected = KM_P_AU / S_P_DAY
        assert np.isclose(result, expected)

    def test_au_per_day_to_km_per_s_array(self):
        """Test AU/day to km/s conversion with array input."""
        input_au_day = np.array([1.0, 2.0, 0.5])
        result = au_per_day_to_km_per_s(input_au_day)
        expected = input_au_day * KM_P_AU / S_P_DAY
        np.testing.assert_allclose(result, expected)

    def test_km_per_s_to_au_per_day_scalar(self):
        """Test km/s to AU/day conversion with scalar input."""
        km_s_value = KM_P_AU / S_P_DAY
        result = km_per_s_to_au_per_day(km_s_value)
        expected = 1.0
        assert np.isclose(result, expected)

    def test_km_per_s_to_au_per_day_array(self):
        """Test km/s to AU/day conversion with array input."""
        input_km_s = np.array([KM_P_AU / S_P_DAY, 2 * KM_P_AU / S_P_DAY])
        result = km_per_s_to_au_per_day(input_km_s)
        expected = np.array([1.0, 2.0])
        np.testing.assert_allclose(result, expected)

    def test_round_trip_position_conversion(self):
        """Test that AU->km->AU gives back original values."""
        original_au = np.array([1.0, 2.5, -0.8, 10.0])
        km_values = au_to_km(original_au)
        round_trip_au = km_to_au(km_values)
        np.testing.assert_allclose(round_trip_au, original_au, rtol=1e-15)

    def test_round_trip_velocity_conversion(self):
        """Test that AU/day->km/s->AU/day gives back original values."""
        original_au_day = np.array([0.1, -0.5, 2.0, 0.01])
        km_s_values = au_per_day_to_km_per_s(original_au_day)
        round_trip_au_day = km_per_s_to_au_per_day(km_s_values)
        np.testing.assert_allclose(round_trip_au_day, original_au_day, rtol=1e-15)


class TestCartesianValuesConversion:
    """Test conversion of full 6D Cartesian coordinate arrays."""

    def test_convert_cartesian_values_au_to_km(self):
        """Test conversion of Cartesian values from AU to km."""
        # Create test data: [x, y, z, vx, vy, vz] in AU and AU/day
        values_au = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.01, 0.0],  # Simple case
                [-2.5, 1.5, -0.5, 0.005, -0.002, 0.001],  # More complex case
            ]
        )

        result = convert_cartesian_values_au_to_km(values_au)

        # Check positions (first 3 columns)
        expected_positions = values_au[:, :3] * KM_P_AU
        np.testing.assert_allclose(result[:, :3], expected_positions)

        # Check velocities (last 3 columns)
        expected_velocities = values_au[:, 3:] * KM_P_AU / S_P_DAY
        np.testing.assert_allclose(result[:, 3:], expected_velocities)

    def test_convert_cartesian_values_km_to_au(self):
        """Test conversion of Cartesian values from km to AU."""
        # Create test data: [x, y, z, vx, vy, vz] in km and km/s
        values_km = np.array(
            [
                [KM_P_AU, 0.0, 0.0, 0.0, KM_P_AU / S_P_DAY, 0.0],  # 1 AU, 1 AU/day
                [
                    2 * KM_P_AU,
                    -0.5 * KM_P_AU,
                    KM_P_AU,
                    0.1 * KM_P_AU / S_P_DAY,
                    0.0,
                    -0.05 * KM_P_AU / S_P_DAY,
                ],
            ]
        )

        result = convert_cartesian_values_km_to_au(values_km)

        # Check positions (first 3 columns)
        expected_positions = values_km[:, :3] / KM_P_AU
        np.testing.assert_allclose(result[:, :3], expected_positions)

        # Check velocities (last 3 columns)
        expected_velocities = values_km[:, 3:] / KM_P_AU * S_P_DAY
        np.testing.assert_allclose(result[:, 3:], expected_velocities)

    def test_round_trip_cartesian_values_conversion(self):
        """Test that AU->km->AU conversion preserves original values."""
        original_au = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.01, 0.0],
                [-2.5, 1.5, -0.5, 0.005, -0.002, 0.001],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Test zeros
            ]
        )

        km_values = convert_cartesian_values_au_to_km(original_au)
        round_trip_au = convert_cartesian_values_km_to_au(km_values)

        np.testing.assert_allclose(round_trip_au, original_au, rtol=1e-15)


class TestCovarianceConversion:
    """Test covariance matrix conversion functions."""

    def test_diagonal_covariance_au_to_km(self):
        """Test conversion of diagonal covariance matrix from AU to km."""
        # Create a simple diagonal covariance matrix in AU units
        N = 2
        cov_au = np.zeros((N, 6, 6))

        # Set diagonal elements: position uncertainties in AU, velocity in AU/day
        pos_sigma_au = [0.1, 0.2, 0.05]  # AU
        vel_sigma_au_day = [0.001, 0.002, 0.0005]  # AU/day

        for i in range(N):
            for j in range(3):
                cov_au[i, j, j] = pos_sigma_au[j] ** 2  # AU²
                cov_au[i, j + 3, j + 3] = vel_sigma_au_day[j] ** 2  # (AU/day)²

        result = convert_cartesian_covariance_au_to_km(cov_au)

        # Check position variances (first 3x3 block)
        for i in range(N):
            for j in range(3):
                expected_pos_var = (pos_sigma_au[j] * KM_P_AU) ** 2
                assert np.isclose(result[i, j, j], expected_pos_var)

                expected_vel_var = (vel_sigma_au_day[j] * KM_P_AU / S_P_DAY) ** 2
                assert np.isclose(result[i, j + 3, j + 3], expected_vel_var)

    def test_diagonal_covariance_km_to_au(self):
        """Test conversion of diagonal covariance matrix from km to AU."""
        # Create a simple diagonal covariance matrix in km units
        N = 1
        cov_km = np.zeros((N, 6, 6))

        # Set diagonal elements: position uncertainties in km, velocity in km/s
        pos_sigma_km = [100.0, 200.0, 50.0]  # km
        vel_sigma_km_s = [0.1, 0.2, 0.05]  # km/s

        for j in range(3):
            cov_km[0, j, j] = pos_sigma_km[j] ** 2  # km²
            cov_km[0, j + 3, j + 3] = vel_sigma_km_s[j] ** 2  # (km/s)²

        result = convert_cartesian_covariance_km_to_au(cov_km)

        # Check position variances
        for j in range(3):
            expected_pos_var = (pos_sigma_km[j] / KM_P_AU) ** 2
            assert np.isclose(result[0, j, j], expected_pos_var)

            expected_vel_var = (vel_sigma_km_s[j] / KM_P_AU * S_P_DAY) ** 2
            assert np.isclose(result[0, j + 3, j + 3], expected_vel_var)

    def test_full_covariance_conversion(self):
        """Test conversion of full covariance matrix with off-diagonal terms."""
        # Create a covariance matrix with correlations between position and velocity
        N = 1
        cov_au = np.zeros((N, 6, 6))

        # Set some diagonal terms
        cov_au[0, 0, 0] = 0.01  # x variance (AU²)
        cov_au[0, 1, 1] = 0.02  # y variance (AU²)
        cov_au[0, 3, 3] = 0.0001  # vx variance ((AU/day)²)
        cov_au[0, 4, 4] = 0.0002  # vy variance ((AU/day)²)

        # Set some off-diagonal terms (correlations)
        cov_au[0, 0, 1] = cov_au[0, 1, 0] = 0.005  # x-y correlation (AU²)
        cov_au[0, 0, 3] = cov_au[0, 3, 0] = 0.001  # x-vx correlation (AU·AU/day)
        cov_au[0, 1, 4] = cov_au[0, 4, 1] = 0.002  # y-vy correlation (AU·AU/day)
        cov_au[0, 3, 4] = cov_au[0, 4, 3] = 0.00005  # vx-vy correlation ((AU/day)²)

        result_km = convert_cartesian_covariance_au_to_km(cov_au)

        # Test diagonal elements
        assert np.isclose(result_km[0, 0, 0], 0.01 * KM_P_AU**2)  # x variance
        assert np.isclose(result_km[0, 1, 1], 0.02 * KM_P_AU**2)  # y variance
        assert np.isclose(
            result_km[0, 3, 3], 0.0001 * (KM_P_AU / S_P_DAY) ** 2
        )  # vx variance
        assert np.isclose(
            result_km[0, 4, 4], 0.0002 * (KM_P_AU / S_P_DAY) ** 2
        )  # vy variance

        # Test off-diagonal elements
        assert np.isclose(result_km[0, 0, 1], 0.005 * KM_P_AU**2)  # x-y correlation
        assert np.isclose(result_km[0, 1, 0], 0.005 * KM_P_AU**2)  # symmetry
        assert np.isclose(
            result_km[0, 0, 3], 0.001 * KM_P_AU * KM_P_AU / S_P_DAY
        )  # x-vx correlation
        assert np.isclose(
            result_km[0, 3, 0], 0.001 * KM_P_AU * KM_P_AU / S_P_DAY
        )  # symmetry

    def test_round_trip_covariance_conversion(self):
        """Test that AU->km->AU covariance conversion preserves original values."""
        # Create a random symmetric positive definite covariance matrix
        N = 3
        np.random.seed(42)  # For reproducibility

        # Generate random covariance matrices
        cov_au_original = np.zeros((N, 6, 6))
        for i in range(N):
            # Create a random positive definite matrix
            A = np.random.randn(6, 6)
            cov_au_original[i] = A @ A.T * 0.01  # Scale to reasonable values

        # Convert AU -> km -> AU
        cov_km = convert_cartesian_covariance_au_to_km(cov_au_original)
        cov_au_round_trip = convert_cartesian_covariance_km_to_au(cov_km)

        np.testing.assert_allclose(cov_au_round_trip, cov_au_original, rtol=1e-13)

    def test_covariance_conversion_preserves_symmetry(self):
        """Test that covariance conversion preserves matrix symmetry."""
        # Create a symmetric covariance matrix
        N = 2
        cov_au = np.random.randn(N, 6, 6) * 0.01

        # Make it symmetric
        for i in range(N):
            cov_au[i] = (cov_au[i] + cov_au[i].T) / 2

        result_km = convert_cartesian_covariance_au_to_km(cov_au)

        # Check that result is still symmetric
        for i in range(N):
            np.testing.assert_allclose(result_km[i], result_km[i].T, rtol=1e-15)

    def test_covariance_conversion_known_values(self):
        """Test covariance conversion with known manual calculation."""
        # Test a simple case where we can manually verify the calculation
        N = 1
        cov_au = np.zeros((N, 6, 6))

        # Set one position-position term and one velocity-velocity term
        cov_au[0, 0, 0] = 1.0  # 1 AU²
        cov_au[0, 3, 3] = 1.0  # 1 (AU/day)²

        result = convert_cartesian_covariance_au_to_km(cov_au)

        # Manual calculation:
        # Position variance: 1 AU² × (KM_P_AU)² = KM_P_AU² km²
        # Velocity variance: 1 (AU/day)² × (KM_P_AU/S_P_DAY)² = (KM_P_AU/S_P_DAY)² (km/s)²

        expected_pos_var = KM_P_AU**2
        expected_vel_var = (KM_P_AU / S_P_DAY) ** 2

        assert np.isclose(result[0, 0, 0], expected_pos_var)
        assert np.isclose(result[0, 3, 3], expected_vel_var)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_values(self):
        """Test that zero values are handled correctly."""
        assert au_to_km(0.0) == 0.0
        assert km_to_au(0.0) == 0.0
        assert au_per_day_to_km_per_s(0.0) == 0.0
        assert km_per_s_to_au_per_day(0.0) == 0.0

    def test_negative_values(self):
        """Test that negative values are handled correctly."""
        assert au_to_km(-1.0) == -KM_P_AU
        assert km_to_au(-KM_P_AU) == -1.0

    def test_very_small_values(self):
        """Test conversion of very small values."""
        tiny_au = 1e-10
        result_km = au_to_km(tiny_au)
        round_trip = km_to_au(result_km)
        assert np.isclose(round_trip, tiny_au, rtol=1e-15)

    def test_very_large_values(self):
        """Test conversion of very large values."""
        large_au = 1e6
        result_km = au_to_km(large_au)
        round_trip = km_to_au(result_km)
        assert np.isclose(round_trip, large_au, rtol=1e-15)
