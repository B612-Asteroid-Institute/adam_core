import numpy as np
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observers.observers import Observers
from ...time import Timestamp
from ..simple_magnitude import (
    InstrumentFilters,
    StandardFilters,
    calculate_apparent_magnitude,
    convert_magnitude,
    find_conversion_path,
)


class TestSimpleMagnitude:
    """Test suite for the simple_magnitude module."""

    @pytest.fixture
    def object_coords(self):
        """Create a sample object coordinates."""
        return CartesianCoordinates.from_kwargs(
            x=[1.5],  # 1.5 AU from Sun in x direction
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )

    @pytest.fixture
    def earth_observer(self):
        """Create a sample Earth-based observer."""
        return Observers.from_kwargs(
            code=["500"],  # Geocenter
            coordinates=CartesianCoordinates.from_kwargs(
                x=[1.0],  # 1 AU from Sun (Earth)
                y=[0.0],
                z=[0.0],
                vx=[0.0],
                vy=[0.0],
                vz=[0.0],
                time=Timestamp.from_mjd([60000], scale="tdb"),
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"]),
            ),
        )

    def test_apparent_mag_distance_dependence(self, earth_observer):
        """Test that apparent magnitude follows inverse square law with distance."""
        H = 15.0
        
        # Create objects at different heliocentric distances
        # Use Y-axis to avoid geometric issues with observer-sun-object alignment
        distances = [1.0, 2.0, 3.0]  # AU
        mags = []
        
        for r in distances:
            coords = CartesianCoordinates.from_kwargs(
                x=[0.0], y=[r], z=[0.0],  # Use Y-axis instead of X-axis
                vx=[0.0], vy=[0.0], vz=[0.0],
                time=Timestamp.from_mjd([60000], scale="tdb"),
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"]),
            )
            mag = calculate_apparent_magnitude(H, coords, earth_observer)
            mags.append(mag)
        
        # Magnitude should increase (get fainter) with distance
        assert mags[1] > mags[0]  # 2 AU fainter than 1 AU
        assert mags[2] > mags[1]  # 3 AU fainter than 2 AU
        
        # Check approximate inverse square law behavior
        # For small phase angles, magnitude difference should be ~5*log10(r2/r1)
        mag_diff_2_1 = mags[1] - mags[0]
        expected_diff_2_1 = 5 * np.log10(2.0/1.0)  # ~1.5 mag
        assert abs(mag_diff_2_1 - expected_diff_2_1) < 0.5  # Allow for phase effects

    def test_apparent_mag_observer_distance_dependence(self, object_coords):
        """Test that apparent magnitude depends on observer distance."""
        H = 15.0
        
        # Create observers at different distances from object
        observer_distances = [0.5, 1.0, 1.5]  # AU from object
        mags = []
        
        for d in observer_distances:
            # Object at (1.5, 0, 0), observer at (1.5-d, 0, 0)
            observer = Observers.from_kwargs(
                code=["500"],
                coordinates=CartesianCoordinates.from_kwargs(
                    x=[1.5 - d], y=[0.0], z=[0.0],
                    vx=[0.0], vy=[0.0], vz=[0.0],
                    time=Timestamp.from_mjd([60000], scale="tdb"),
                    frame="ecliptic",
                    origin=Origin.from_kwargs(code=["SUN"]),
                ),
            )
            mag = calculate_apparent_magnitude(H, object_coords, observer)
            mags.append(mag)
        
        # Magnitude should increase with observer distance
        assert mags[1] > mags[0]  # 1.0 AU fainter than 0.5 AU
        assert mags[2] > mags[1]  # 1.5 AU fainter than 1.0 AU

    def test_apparent_mag_phase_function_effect(self, earth_observer):
        """Test that phase function affects magnitude as expected."""
        H = 15.0
        
        # Create configurations with different phase angles
        # Object at (2, 0, 0), observer at (1, 0, 0) -> phase = 0
        # Object at (0, 2, 0), observer at (1, 0, 0) -> phase > 0
        
        # Opposition (phase ~ 0)
        coords_opposition = CartesianCoordinates.from_kwargs(
            x=[2.0], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )
        
        # Higher phase angle
        coords_phase = CartesianCoordinates.from_kwargs(
            x=[0.0], y=[2.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )
        
        mag_opposition = calculate_apparent_magnitude(H, coords_opposition, earth_observer)
        mag_phase = calculate_apparent_magnitude(H, coords_phase, earth_observer)
        
        # Object should be brighter at opposition than at higher phase
        assert mag_opposition < mag_phase

    def test_apparent_mag_array_consistency(self, earth_observer):
        """Test that array inputs give consistent results."""
        H_array = np.array([10.0, 15.0, 20.0])
        
        # Create multiple identical objects
        multi_coords = CartesianCoordinates.from_kwargs(
            x=[1.5, 1.5, 1.5], y=[0.0, 0.0, 0.0], z=[0.0, 0.0, 0.0],
            vx=[0.0, 0.0, 0.0], vy=[0.0, 0.0, 0.0], vz=[0.0, 0.0, 0.0],
            time=Timestamp.from_mjd([60000, 60000, 60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        )
        
        multi_observer = Observers.from_kwargs(
            code=["500", "500", "500"],
            coordinates=CartesianCoordinates.from_kwargs(
                x=[1.0, 1.0, 1.0], y=[0.0, 0.0, 0.0], z=[0.0, 0.0, 0.0],
                vx=[0.0, 0.0, 0.0], vy=[0.0, 0.0, 0.0], vz=[0.0, 0.0, 0.0],
                time=Timestamp.from_mjd([60000, 60000, 60000], scale="tdb"),
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
            ),
        )
        
        # Calculate with array
        array_result = calculate_apparent_magnitude(H_array, multi_coords, multi_observer)
        
        # Calculate individually
        individual_results = []
        for i in range(3):
            single_coords = CartesianCoordinates.from_kwargs(
                x=[1.5], y=[0.0], z=[0.0],
                vx=[0.0], vy=[0.0], vz=[0.0],
                time=Timestamp.from_mjd([60000], scale="tdb"),
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"]),
            )
            single_observer = Observers.from_kwargs(
                code=["500"],
                coordinates=CartesianCoordinates.from_kwargs(
                    x=[1.0], y=[0.0], z=[0.0],
                    vx=[0.0], vy=[0.0], vz=[0.0],
                    time=Timestamp.from_mjd([60000], scale="tdb"),
                    frame="ecliptic",
                    origin=Origin.from_kwargs(code=["SUN"]),
                ),
            )
            mag = calculate_apparent_magnitude(H_array[i], single_coords, single_observer)
            # Extract scalar value from array result
            individual_results.append(mag.item() if hasattr(mag, 'item') else float(mag))
        
        # Results should be identical
        assert np.allclose(array_result, individual_results, rtol=1e-10)

    def test_convert_magnitude_invertibility(self):
        """Test that magnitude conversions are invertible."""
        test_magnitude = 15.0
        
        # Test some common conversions (only verified ones)
        filter_pairs = [
            ("V", "g"),
            ("LSST_g", "g"),
            ("DECam_r", "r"),
        ]
        
        for filter1, filter2 in filter_pairs:
            # Convert forward and back
            converted = convert_magnitude(test_magnitude, filter1, filter2)
            back_converted = convert_magnitude(converted, filter2, filter1)
            
            # Note: Empirical transformations may not be perfect mathematical inverses
            # Allow for typical photometric transformation uncertainties
            tolerance = 1e-3 
            
            assert abs(back_converted - test_magnitude) < tolerance, \
                f"Invertibility failed for {filter1} <-> {filter2}: {back_converted} vs {test_magnitude}"
            
            # Check that the conversion maintains reasonable magnitude range
            assert abs(converted - test_magnitude) < 5.0, \
                f"Conversion {filter1} -> {filter2} unreasonable: {converted}"

    def test_convert_magnitude_array_consistency(self):
        """Test that array conversions are consistent with scalar conversions."""
        test_mags = np.array([10.0, 15.0, 20.0])
        
        # Test V to g conversion
        array_result = convert_magnitude(test_mags, "V", "g")
        
        # Compare with individual conversions
        individual_results = [convert_magnitude(mag, "V", "g") for mag in test_mags]
        
        assert np.allclose(array_result, individual_results, rtol=1e-10)

    def test_convert_magnitude_identity(self):
        """Test that same-filter conversions are identity."""
        test_magnitude = 15.0
        test_array = np.array([10.0, 15.0, 20.0])
        
        # Test various filters (only those with verified conversions)
        filters = ["V", "g", "LSST_g", "DECam_r"]
        
        for filt in filters:
            # Scalar test
            result = convert_magnitude(test_magnitude, filt, filt)
            assert result == test_magnitude
            
            # Array test
            result_array = convert_magnitude(test_array, filt, filt)
            assert np.array_equal(result_array, test_array)


    def test_convert_magnitude_path_finding(self):
        """Test that conversion path finding works correctly."""
        # Test direct path
        path = find_conversion_path("V", "g")
        assert path == ["V", "g"] or len(path) == 2
        
        # Test identity path
        path = find_conversion_path("V", "V")
        assert path == ["V"]
        
        # Test that some path exists between verified filters
        path = find_conversion_path("LSST_g", "DECam_g")
        assert len(path) >= 2  # At least source and target
        assert path[0] == "LSST_g"
        assert path[-1] == "DECam_g"

    def test_convert_magnitude_invalid_filters(self):
        """Test proper error handling for invalid filters."""
        test_magnitude = 15.0
        
        # Test invalid source filter
        with pytest.raises(ValueError, match="No conversion path"):
            convert_magnitude(test_magnitude, "NonExistentFilter", "V")
        
        # Test invalid target filter
        with pytest.raises(ValueError, match="No conversion path"):
            convert_magnitude(test_magnitude, "V", "NonExistentFilter")
        
        # Test conversion between filters with no available path
        with pytest.raises(ValueError, match="No conversion path"):
            convert_magnitude(test_magnitude, "ZTF_g", "DECam_u")

    def test_convert_magnitude_physical_reasonableness(self):
        """Test that conversions produce physically reasonable results."""
        test_magnitude = 15.0
        
        # Test some known filter relationships (only verified ones)
        # Most conversions should be within a few magnitudes
        filter_pairs = [
            ("V", "g"),
            ("LSST_g", "g"),
        ]
        
        for filter1, filter2 in filter_pairs:
            converted = convert_magnitude(test_magnitude, filter1, filter2)
            
            # Conversion should be reasonable (within ~5 magnitudes)
            assert abs(converted - test_magnitude) < 5.0, \
                f"Unreasonable conversion from {filter1} to {filter2}: {converted}"
            
            # Result should be finite
            assert np.isfinite(converted), \
                f"Non-finite result for {filter1} to {filter2}"
