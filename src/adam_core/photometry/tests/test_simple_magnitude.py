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
    get_filter_properties,
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

    def test_apparent_mag_basic(self, object_coords, earth_observer):
        """Test basic calculation of apparent magnitude."""
        # For an object at 1.5 AU from Sun and 0.5 AU from Earth
        # with H=15, the apparent magnitude should be around 15.99
        H = 15.0
        apparent_mag = calculate_apparent_magnitude(
            H, object_coords, earth_observer, G=0.15, filter_name="V"
        )

        # Expected: H + 5*log10(r*delta) - 2.5*log10(phase_function)
        # r = 1.5 AU, delta = 0.5 AU, phase_angle = 0 (aligned)
        # phase_function ≈ 1 for phase_angle = 0
        # Expected ≈ 15 + 5*log10(1.5*0.5) - 2.5*log10(1)
        # Expected ≈ 15 + 5*log10(0.75) - 0
        # Expected ≈ 15 - 0.65 ≈ 14.35
        assert 14.3 <= apparent_mag <= 14.4

    def test_apparent_mag_array(self, object_coords, earth_observer):
        """Test calculation of apparent magnitude with array input."""
        # Test with array of H values
        H_array = np.array([15.0, 16.0, 17.0])

        # Duplicate the coordinates and observer for each H value
        multi_coords = CartesianCoordinates.from_kwargs(
            x=[1.5, 1.5, 1.5],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
            vx=[0.0, 0.0, 0.0],
            vy=[0.0, 0.0, 0.0],
            vz=[0.0, 0.0, 0.0],
            time=Timestamp.from_mjd([60000, 60000, 60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        )

        multi_observer = Observers.from_kwargs(
            code=["500", "500", "500"],
            coordinates=CartesianCoordinates.from_kwargs(
                x=[1.0, 1.0, 1.0],
                y=[0.0, 0.0, 0.0],
                z=[0.0, 0.0, 0.0],
                vx=[0.0, 0.0, 0.0],
                vy=[0.0, 0.0, 0.0],
                vz=[0.0, 0.0, 0.0],
                time=Timestamp.from_mjd([60000, 60000, 60000], scale="tdb"),
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
            ),
        )

        apparent_mags = calculate_apparent_magnitude(
            H_array, multi_coords, multi_observer, G=0.15, filter_name="V"
        )

        # Each result should be offset by the same amount as the H values
        assert len(apparent_mags) == 3
        assert np.isclose(apparent_mags[1] - apparent_mags[0], 1.0, atol=0.01)
        assert np.isclose(apparent_mags[2] - apparent_mags[1], 1.0, atol=0.01)

    def test_apparent_mag_filters(self, object_coords, earth_observer):
        """Test calculation with different filters."""
        H = 15.0

        # Calculate for different filters
        v_mag = calculate_apparent_magnitude(
            H, object_coords, earth_observer, filter_name="V"
        )
        g_mag = calculate_apparent_magnitude(
            H, object_coords, earth_observer, filter_name="g"
        )
        r_mag = calculate_apparent_magnitude(
            H, object_coords, earth_observer, filter_name="r"
        )

        # Verify filter conversions are applied
        # V to g conversion: g = 1.0210*V - 0.0852
        expected_g = 1.0210 * v_mag - 0.0852
        assert np.isclose(g_mag, expected_g, atol=0.01)

        # V to r conversion: r = 0.9613*V + 0.2087
        expected_r = 0.9613 * v_mag + 0.2087
        assert np.isclose(r_mag, expected_r, atol=0.01)

    def test_convert_magnitude_direct(self):
        """Test direct filter conversions."""
        # Test V to g conversion
        v_mag = 15.0
        g_mag = convert_magnitude(v_mag, "V", "g")
        expected_g = 1.0210 * v_mag - 0.0852
        assert np.isclose(g_mag, expected_g, atol=0.001)

        # Test g to V conversion
        g_mag = 16.0
        v_mag = convert_magnitude(g_mag, "g", "V")
        expected_v = 0.9137 * g_mag + 0.2083
        assert np.isclose(v_mag, expected_v, atol=0.001)

        # Test array input
        v_mags = np.array([15.0, 16.0, 17.0])
        g_mags = convert_magnitude(v_mags, "V", "g")
        expected_g = 1.0210 * v_mags - 0.0852
        assert np.allclose(g_mags, expected_g, atol=0.001)

    def test_convert_magnitude_indirect(self):
        """Test indirect filter conversions through V."""
        # Test g to r conversion (should go through V)
        g_mag = 16.0
        r_mag = convert_magnitude(g_mag, "g", "r")

        # Expected: g -> V -> r
        # V = 0.9137*g + 0.2083
        # r = 0.9613*V + 0.2087
        expected_v = 0.9137 * g_mag + 0.2083
        expected_r = 0.9613 * expected_v + 0.2087
        assert np.isclose(r_mag, expected_r, atol=0.001)

        # Test LSST_g to DECam_r conversion
        lsst_g = 17.0
        decam_r = convert_magnitude(lsst_g, "LSST_g", "DECam_r")

        # Now that we have direct conversions, we can test more precisely
        # LSST_g to DECam_r is a direct conversion: 0.9963*x + 0.0005
        expected_decam_r = 0.9963 * lsst_g + 0.0005
        assert np.isclose(decam_r, expected_decam_r, atol=0.001)

    def test_convert_magnitude_same_filter(self):
        """Test conversion to the same filter."""
        v_mag = 15.0
        result = convert_magnitude(v_mag, "V", "V")
        assert result == v_mag

        # Test with array
        v_mags = np.array([15.0, 16.0, 17.0])
        results = convert_magnitude(v_mags, "V", "V")
        assert np.array_equal(results, v_mags)

    def test_convert_magnitude_invalid(self):
        """Test conversion with invalid filter names."""
        with pytest.raises(ValueError):
            convert_magnitude(15.0, "V", "NonExistentFilter")

        with pytest.raises(ValueError):
            convert_magnitude(15.0, "NonExistentFilter", "V")

    def test_get_filter_properties(self):
        """Test getting filter properties."""
        # Test standard filter
        v_props = get_filter_properties("V")
        assert len(v_props) == 3
        assert v_props[0] == 547.7  # wavelength
        assert v_props[1] == 85.0  # width
        assert v_props[2] == 25.03  # zeropoint

        # Test instrument filter
        lsst_g_props = get_filter_properties("LSST_g")
        assert len(lsst_g_props) == 3
        assert lsst_g_props[0] == 482.5  # wavelength
        assert lsst_g_props[1] == 128.0  # width
        assert lsst_g_props[2] == 25.17  # zeropoint

        # Test invalid filter
        with pytest.raises(ValueError):
            get_filter_properties("NonExistentFilter")

    def test_filter_enums(self):
        """Test the filter enum classes."""
        # Test StandardFilters
        assert StandardFilters.V.value[0] == 547.7
        assert StandardFilters.g.value[1] == 137.9

        # Test InstrumentFilters
        assert InstrumentFilters.LSST_g.value[0] == 482.5
        assert InstrumentFilters.DECam_r.value[2] == 24.85
