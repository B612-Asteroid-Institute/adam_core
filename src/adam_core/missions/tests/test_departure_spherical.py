import numpy as np
import pytest

from adam_core.coordinates import CartesianCoordinates, SphericalCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.transform import transform_coordinates
from adam_core.time import Timestamp

from ..porkchop import departure_spherical_coordinates


class TestDepartureSphericalCoordinates:
    """Test suite for departure_spherical_coordinates function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple velocity vectors."""
        # Test with a single velocity vector pointing in +X direction
        times = Timestamp.from_mjd([60000.0], scale="tdb")
        vx = np.array([1.0])
        vy = np.array([0.0])
        vz = np.array([0.0])

        result = departure_spherical_coordinates(
            departure_origin=OriginCodes.EARTH,
            times=times,
            frame="ecliptic",
            vx=vx,
            vy=vy,
            vz=vz,
        )

        # Verify result is SphericalCoordinates
        assert isinstance(result, SphericalCoordinates)
        assert len(result) == 1
        assert result.origin.as_OriginCodes() == OriginCodes.EARTH
        assert result.frame == "equatorial"  # Corrected implementation behavior

    def test_normalization(self):
        """Test that velocity vectors are properly normalized."""
        times = Timestamp.from_mjd([60000.0], scale="tdb")

        # Test with different magnitude vectors - should give same direction
        vx1 = np.array([1.0])
        vy1 = np.array([0.0])
        vz1 = np.array([0.0])

        vx2 = np.array([5.0])  # 5x larger magnitude
        vy2 = np.array([0.0])
        vz2 = np.array([0.0])

        result1 = departure_spherical_coordinates(
            OriginCodes.EARTH, times, "ecliptic", vx1, vy1, vz1
        )
        result2 = departure_spherical_coordinates(
            OriginCodes.EARTH, times, "ecliptic", vx2, vy2, vz2
        )

        # Should have same spherical angles despite different magnitudes
        np.testing.assert_allclose(result1.lon, result2.lon, rtol=1e-12)
        np.testing.assert_allclose(result1.lat, result2.lat, rtol=1e-12)

    def test_multiple_vectors(self):
        """Test with multiple velocity vectors."""
        times = Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb")
        vx = np.array([1.0, 0.0, 0.0])
        vy = np.array([0.0, 1.0, 0.0])
        vz = np.array([0.0, 0.0, 1.0])

        result = departure_spherical_coordinates(
            OriginCodes.EARTH, times, "ecliptic", vx, vy, vz
        )

        assert len(result) == 3
        assert len(result.time) == 3

    def test_error_conditions(self):
        """Test various error conditions."""
        times = Timestamp.from_mjd([60000.0], scale="tdb")
        vx = np.array([1.0])
        vy = np.array([0.0])
        vz = np.array([0.0])

        # Test with Mars origin - should now work!
        result_mars = departure_spherical_coordinates(
            OriginCodes.MARS, times, "ecliptic", vx, vy, vz
        )
        assert isinstance(result_mars, SphericalCoordinates)
        assert result_mars.origin.as_OriginCodes() == OriginCodes.MARS
        assert result_mars.frame == "equatorial"

        # Test mismatched array lengths
        times_wrong = Timestamp.from_mjd([60000.0, 60001.0], scale="tdb")
        with pytest.raises(
            AssertionError, match="All arrays must have the same length"
        ):
            departure_spherical_coordinates(
                OriginCodes.EARTH, times_wrong, "ecliptic", vx, vy, vz
            )

        # Test empty arrays
        empty_times = Timestamp.from_mjd([], scale="tdb")
        empty_vx = np.array([])
        empty_vy = np.array([])
        empty_vz = np.array([])
        with pytest.raises(
            AssertionError, match="At least one departure vector is required"
        ):
            departure_spherical_coordinates(
                OriginCodes.EARTH, empty_times, "ecliptic", empty_vx, empty_vy, empty_vz
            )

    def test_coordinate_directions_simple_cases(self):
        """Test coordinate directions for simple cardinal directions."""
        times = Timestamp.from_mjd([60000.0], scale="tdb")

        # Test +X direction (should be longitude=0, latitude=0 in ecliptic)
        result_x = departure_spherical_coordinates(
            OriginCodes.EARTH,
            times,
            "ecliptic",
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
        )

        # Test +Y direction (should be longitude=90, latitude=0 in ecliptic)
        result_y = departure_spherical_coordinates(
            OriginCodes.EARTH,
            times,
            "ecliptic",
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
        )

        # Test +Z direction (should be latitude=90 in ecliptic)
        result_z = departure_spherical_coordinates(
            OriginCodes.EARTH,
            times,
            "ecliptic",
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
        )

        # Note: Since current implementation transforms to ITRF93,
        # we can't easily predict exact values without knowing Earth's orientation
        # But we can verify they're different and make sense
        assert not np.allclose(result_x.lon, result_y.lon)
        assert not np.allclose(result_x.lat, result_z.lat)

    def test_time_dependence(self):
        """Test that results properly depend on time due to Earth rotation."""
        # Same velocity vector at different times should give different ITRF93 coordinates
        times1 = Timestamp.from_mjd([60000.0], scale="tdb")
        times2 = Timestamp.from_mjd([60000.5], scale="tdb")  # 12 hours later

        vx = np.array([1.0])
        vy = np.array([0.0])
        vz = np.array([0.0])

        result1 = departure_spherical_coordinates(
            OriginCodes.EARTH, times1, "ecliptic", vx, vy, vz
        )
        result2 = departure_spherical_coordinates(
            OriginCodes.EARTH, times2, "ecliptic", vx, vy, vz
        )

        # In equatorial frame, same velocity vector should give same RA/Dec
        # regardless of time (since it's inertial, not Earth-fixed)
        # Note: There might be very small differences due to numerical precision
        np.testing.assert_allclose(result1.lon, result2.lon, rtol=1e-10)
        np.testing.assert_allclose(result1.lat, result2.lat, rtol=1e-10)

    def test_different_input_frames(self):
        """Test behavior with different input frames."""
        times = Timestamp.from_mjd([60000.0], scale="tdb")
        # Use a vector with a Z component to make the frame difference more obvious
        vx = np.array([1.0])
        vy = np.array([0.0])
        vz = np.array([1.0])  # Changed from 0.0 to 1.0

        # Test with different input frames
        result_ecliptic = departure_spherical_coordinates(
            OriginCodes.EARTH, times, "ecliptic", vx, vy, vz
        )
        result_equatorial = departure_spherical_coordinates(
            OriginCodes.EARTH, times, "equatorial", vx, vy, vz
        )

        # Results should be different because the same velocity vector
        # in different frames represents different physical directions
        # Note: We need to use arrays with .to_numpy() for comparison
        lon_ecliptic = result_ecliptic.lon.to_numpy(zero_copy_only=False)
        lon_equatorial = result_equatorial.lon.to_numpy(zero_copy_only=False)
        lat_ecliptic = result_ecliptic.lat.to_numpy(zero_copy_only=False)
        lat_equatorial = result_equatorial.lat.to_numpy(zero_copy_only=False)

        # At least one of longitude or latitude should be different
        assert not np.allclose(
            lon_ecliptic, lon_equatorial, rtol=1e-6
        ) or not np.allclose(lat_ecliptic, lat_equatorial, rtol=1e-6)

    def test_multiple_origins_support(self):
        """Test that the function now supports multiple departure origins."""
        times = Timestamp.from_mjd([60000.0], scale="tdb")
        vx = np.array([1.0])
        vy = np.array([0.0])
        vz = np.array([0.0])

        # Test various departure origins
        origins_to_test = [
            OriginCodes.EARTH,
            OriginCodes.MARS,
            OriginCodes.JUPITER_BARYCENTER,
            OriginCodes.SUN,
        ]

        for origin in origins_to_test:
            result = departure_spherical_coordinates(
                origin, times, "ecliptic", vx, vy, vz
            )
            assert isinstance(result, SphericalCoordinates)
            assert result.origin.as_OriginCodes() == origin
            assert result.frame == "equatorial"  # All should be in equatorial frame

            # Verify RA/Dec are in valid ranges
            ra = result.lon.to_numpy(zero_copy_only=False)[0]
            dec = result.lat.to_numpy(zero_copy_only=False)[0]
            assert 0 <= ra < 360
            assert -90 <= dec <= 90

    def test_proposed_fix_behavior(self):
        """Test what the corrected behavior should look like."""
        times = Timestamp.from_mjd([60000.0], scale="tdb")
        vx = np.array([1.0])
        vy = np.array([0.0])
        vz = np.array([0.0])

        # Manual implementation of what the function should do:
        # 1. Create direction vectors in input frame
        velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        direction_x = vx / velocity_magnitude
        direction_y = vy / velocity_magnitude
        direction_z = vz / velocity_magnitude

        # 2. Create CartesianCoordinates in input frame
        direction_coords = CartesianCoordinates.from_kwargs(
            time=times,
            x=direction_x,
            y=direction_y,
            z=direction_z,
            vx=np.zeros_like(vx),
            vy=np.zeros_like(vy),
            vz=np.zeros_like(vz),
            origin=Origin.from_OriginCodes(OriginCodes.EARTH, size=len(vx)),
            frame="ecliptic",  # Input frame
        )

        # 3. Transform to equatorial frame (proper for RA/Dec), not ITRF93
        spherical_correct = transform_coordinates(
            direction_coords,
            SphericalCoordinates,
            frame_out="equatorial",  # This is what we want for RA/Dec
            origin_out=OriginCodes.EARTH,
        )

        # This would give us proper RA/Dec coordinates
        assert spherical_correct.frame == "equatorial"

        # The longitude would be RA and latitude would be Dec
        ra_degrees = spherical_correct.lon.to_numpy(zero_copy_only=False)[0]
        dec_degrees = spherical_correct.lat.to_numpy(zero_copy_only=False)[0]

        # Verify these are in valid RA/Dec ranges
        assert 0 <= ra_degrees < 360
        assert -90 <= dec_degrees <= 90

    def test_zero_velocity_handling(self):
        """Test handling of zero velocity vectors."""
        times = Timestamp.from_mjd([60000.0], scale="tdb")
        vx = np.array([0.0])
        vy = np.array([0.0])
        vz = np.array([0.0])

        # Should handle zero velocity gracefully (though result may be undefined)
        with pytest.warns(RuntimeWarning):  # Division by zero warning expected
            departure_spherical_coordinates(
                OriginCodes.EARTH, times, "ecliptic", vx, vy, vz
            )

    def test_interplanetary_mission_example(self):
        """Test a realistic interplanetary mission scenario."""
        times = Timestamp.from_mjd([60000.0], scale="tdb")

        # Example: departure velocity pointing towards a specific RA/Dec
        # Let's say we want to depart towards RA=120°, Dec=30°
        target_ra = 120.0  # degrees
        target_dec = 30.0  # degrees

        # Convert target RA/Dec to unit vector in equatorial frame
        ra_rad = np.radians(target_ra)
        dec_rad = np.radians(target_dec)
        target_x = np.cos(dec_rad) * np.cos(ra_rad)
        target_y = np.cos(dec_rad) * np.sin(ra_rad)
        target_z = np.sin(dec_rad)

        # Test departures from different origins towards same celestial direction
        for origin in [OriginCodes.EARTH, OriginCodes.MARS]:
            # Convert to input frame (ecliptic) - this is a simplification
            # In reality you'd need proper coordinate transformation
            vx = np.array(
                [target_x * 10.0]
            )  # Scale doesn't matter due to normalization
            vy = np.array([target_y * 10.0])
            vz = np.array([target_z * 10.0])

            result = departure_spherical_coordinates(
                origin, times, "equatorial", vx, vy, vz
            )

            # Should get approximately the target RA/Dec regardless of departure origin
            result_ra = result.lon.to_numpy(zero_copy_only=False)[0]
            result_dec = result.lat.to_numpy(zero_copy_only=False)[0]

            # Allow some tolerance for coordinate transformation approximations
            np.testing.assert_allclose(result_ra, target_ra, atol=0.1)
            np.testing.assert_allclose(result_dec, target_dec, atol=0.1)
