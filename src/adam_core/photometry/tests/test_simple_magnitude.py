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
    calculate_apparent_magnitude_jax,
    convert_magnitude,
)


def _as_scalar(x) -> float:
    """
    Coerce a possibly-0d or 1-element numpy result into a Python float.

    `calculate_apparent_magnitude` returns numpy arrays even for a single object.
    """
    arr = np.asarray(x)
    if arr.shape == ():
        return float(arr)
    if arr.size != 1:
        raise ValueError(f"Expected scalar/length-1 result, got shape={arr.shape}")
    return float(arr.reshape(-1)[0])


class TestSimpleMagnitude:
    """Test suite for the simple_magnitude module."""

    @pytest.fixture
    def earth_observer(self):
        """Create a sample Earth-based observer at (1,0,0) AU."""
        return Observers.from_kwargs(
            code=["500"],
            coordinates=CartesianCoordinates.from_kwargs(
                x=[1.0], y=[0.0], z=[0.0],
                vx=[0.0], vy=[0.0], vz=[0.0],
                time=Timestamp.from_mjd([60000], scale="tdb"),
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"]),
            ),
        )

    def test_apparent_mag_geometry_affects_brightness(self, earth_observer):
        """
        Sanity check that magnitude responds correctly to geometry:
        farther objects are fainter, opposition is brighter than quadrature.
        """
        H = 15.0

        # Object at 1 AU vs 2 AU (both on Y-axis to have non-zero phase)
        coords_near = CartesianCoordinates.from_kwargs(
            x=[0.0], y=[1.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )
        coords_far = CartesianCoordinates.from_kwargs(
            x=[0.0], y=[2.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )
        # Opposition: object at (2,0,0), observer at (1,0,0) -> phase=0
        coords_opposition = CartesianCoordinates.from_kwargs(
            x=[2.0], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )

        mag_near = _as_scalar(calculate_apparent_magnitude(H, coords_near, earth_observer))
        mag_far = _as_scalar(calculate_apparent_magnitude(H, coords_far, earth_observer))
        mag_opp = _as_scalar(calculate_apparent_magnitude(H, coords_opposition, earth_observer))

        # Farther is fainter
        assert mag_far > mag_near
        # Opposition (phase=0) is brighter than quadrature (coords_far has ~90° phase)
        assert mag_opp < mag_far

    def test_apparent_mag_array_consistency(self, earth_observer):
        """Test that array inputs give consistent results with scalar calls."""
        H_array = np.array([10.0, 15.0, 20.0])

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
            individual_results.append(_as_scalar(mag))

        assert np.allclose(array_result, individual_results, rtol=1e-10)

    def test_apparent_mag_opposition_matches_closed_form(self):
        """
        At opposition (phase angle = 0), the H-G phase function is exactly 1,
        so m = H + 5*log10(r*delta).
        """
        H = 15.0
        time = Timestamp.from_mjd([60000], scale="tdb")

        # Observer at 1 AU on +x, object at 2 AU on +x: r=2, delta=1, phase=0
        observer = Observers.from_kwargs(
            code=["500"],
            coordinates=CartesianCoordinates.from_kwargs(
                x=[1.0], y=[0.0], z=[0.0],
                vx=[0.0], vy=[0.0], vz=[0.0],
                time=time,
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"]),
            ),
        )
        obj = CartesianCoordinates.from_kwargs(
            x=[2.0], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )

        expected = H + 5.0 * np.log10(2.0 * 1.0)
        # Phase function = 1 regardless of G at phase=0
        assert _as_scalar(calculate_apparent_magnitude(H, obj, observer, G=0.0)) == pytest.approx(expected, abs=1e-10)
        assert _as_scalar(calculate_apparent_magnitude(H, obj, observer, G=1.0)) == pytest.approx(expected, abs=1e-10)

    def test_apparent_mag_output_filter_matches_convert_magnitude(self):
        """
        calculate_apparent_magnitude with output_filter should match
        convert_magnitude(m_v, "V", target) for both string and Enum inputs.
        """
        H = 15.0
        time = Timestamp.from_mjd([60000], scale="tdb")

        observer = Observers.from_kwargs(
            code=["500"],
            coordinates=CartesianCoordinates.from_kwargs(
                x=[1.0], y=[0.0], z=[0.0],
                vx=[0.0], vy=[0.0], vz=[0.0],
                time=time,
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"]),
            ),
        )
        obj = CartesianCoordinates.from_kwargs(
            x=[2.0], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )

        m_v = _as_scalar(calculate_apparent_magnitude(H, obj, observer, output_filter="V"))

        # Test a few representative filters (string and enum)
        for name, enum in [("g", StandardFilters.g), ("LSST_r", InstrumentFilters.LSST_r)]:
            expected = convert_magnitude(m_v, "V", name)
            assert _as_scalar(calculate_apparent_magnitude(H, obj, observer, output_filter=name)) == pytest.approx(expected, abs=1e-10)
            assert _as_scalar(calculate_apparent_magnitude(H, obj, observer, output_filter=enum)) == pytest.approx(expected, abs=1e-10)

    def test_apparent_mag_validates_array_lengths(self, earth_observer):
        """Mismatched array lengths should raise ValueError."""
        H = np.array([15.0, 16.0])
        G = np.array([0.15, 0.15, 0.15])

        coords_len1 = CartesianCoordinates.from_kwargs(
            x=[1.5], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )

        with pytest.raises(ValueError, match="object_coords length"):
            calculate_apparent_magnitude(H, coords_len1, earth_observer)

        coords_len2 = CartesianCoordinates.from_kwargs(
            x=[1.5, 1.5], y=[0.0, 0.0], z=[0.0, 0.0],
            vx=[0.0, 0.0], vy=[0.0, 0.0], vz=[0.0, 0.0],
            time=Timestamp.from_mjd([60000, 60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        )
        observer_len2 = Observers.from_kwargs(
            code=["500", "500"],
            coordinates=CartesianCoordinates.from_kwargs(
                x=[1.0, 1.0], y=[0.0, 0.0], z=[0.0, 0.0],
                vx=[0.0, 0.0], vy=[0.0, 0.0], vz=[0.0, 0.0],
                time=Timestamp.from_mjd([60000, 60000], scale="tdb"),
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            ),
        )

        with pytest.raises(ValueError, match="G array length"):
            calculate_apparent_magnitude(H, coords_len2, observer_len2, G=G)

    def test_convert_magnitude_invertibility(self):
        """Round-trip conversions should recover the original magnitude."""
        test_magnitude = 15.0

        for filter1, filter2 in [("V", "g"), ("LSST_g", "g"), ("DECam_r", "r")]:
            converted = convert_magnitude(test_magnitude, filter1, filter2)
            back = convert_magnitude(converted, filter2, filter1)
            assert back == pytest.approx(test_magnitude, abs=1e-3)

    def test_convert_magnitude_identity(self):
        """Same-filter conversion is identity."""
        assert convert_magnitude(15.0, "V", "V") == 15.0

    def test_convert_magnitude_invalid_filters(self):
        """Unknown filters or unreachable paths should raise ValueError."""
        with pytest.raises(ValueError, match="No conversion path"):
            convert_magnitude(15.0, "NonExistentFilter", "V")

        with pytest.raises(ValueError, match="No conversion path"):
            convert_magnitude(15.0, "V", "NonExistentFilter")

    def test_apparent_magnitude_jax_matches_numpy(self):
        """JAX implementation should numerically match the NumPy implementation."""
        rng = np.random.default_rng(42)
        n = 64
        time = Timestamp.from_mjd(np.full(n, 60000), scale="tdb")

        # Keep geometry well-conditioned (avoid r ~ 0 or delta ~ 0)
        obj_x = rng.uniform(1.2, 3.0, size=n)
        obj_y = rng.uniform(0.1, 2.0, size=n)
        obj_z = rng.uniform(-0.5, 0.5, size=n)

        observer = Observers.from_kwargs(
            code=["500"] * n,
            coordinates=CartesianCoordinates.from_kwargs(
                x=np.full(n, 1.0),
                y=np.zeros(n),
                z=np.zeros(n),
                vx=np.zeros(n),
                vy=np.zeros(n),
                vz=np.zeros(n),
                time=time,
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"] * n),
            ),
        )
        obj = CartesianCoordinates.from_kwargs(
            x=obj_x,
            y=obj_y,
            z=obj_z,
            vx=np.zeros(n),
            vy=np.zeros(n),
            vz=np.zeros(n),
            time=time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * n),
        )

        H = rng.uniform(10.0, 25.0, size=n)
        G = rng.uniform(0.0, 1.0, size=n)

        mags_np = calculate_apparent_magnitude(H, obj, observer, G=G, output_filter="V")
        mags_jax = calculate_apparent_magnitude_jax(H, obj, observer, G=G, output_filter="V")

        # JAX may run in float32 depending on configuration; keep tolerance realistic.
        assert np.allclose(np.asarray(mags_jax), mags_np, rtol=1e-6, atol=1e-8)

    def test_apparent_magnitude_jax_output_filter(self, earth_observer):
        """JAX implementation should support output_filter conversion."""
        H = 15.0
        obj = CartesianCoordinates.from_kwargs(
            x=[2.0], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )

        m_g_np = _as_scalar(calculate_apparent_magnitude(H, obj, earth_observer, output_filter="g"))
        m_g_jax = _as_scalar(calculate_apparent_magnitude_jax(H, obj, earth_observer, output_filter="g"))
        assert m_g_jax == pytest.approx(m_g_np, abs=1e-6)
