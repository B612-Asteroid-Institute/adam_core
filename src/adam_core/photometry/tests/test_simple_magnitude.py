import numpy as np
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observers.observers import Observers
from ...time import Timestamp
from ..simple_magnitude import (
    InstrumentFilters,
    StandardFilters,
    calculate_apparent_magnitude_v,
    calculate_apparent_magnitude_v_auto,
    calculate_apparent_magnitude_v_jax,
    encode_filters,
    convert_magnitude,
    convert_magnitude_auto,
    convert_magnitude_jax_codes,
    convert_magnitude_jax,
)


def _as_scalar(x) -> float:
    """
    Coerce a possibly-0d or 1-element numpy result into a Python float.

    `calculate_apparent_magnitude_v` returns numpy arrays even for a single object.
    """
    arr = np.asarray(x)
    if arr.shape == ():
        return float(arr)
    if arr.size != 1:
        raise ValueError(f"Expected scalar/length-1 result, got shape={arr.shape}")
    return float(arr.reshape(-1)[0])


def _convert_scalar(mag: float, source_filter, target_filter) -> float:
    """Call array-only convert_magnitude for a single value."""
    out = convert_magnitude(
        np.asarray([mag], dtype=float),
        np.asarray([source_filter], dtype=object),
        np.asarray([target_filter], dtype=object),
    )
    return _as_scalar(out)


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

        mag_near = _as_scalar(calculate_apparent_magnitude_v(H, coords_near, earth_observer))
        mag_far = _as_scalar(calculate_apparent_magnitude_v(H, coords_far, earth_observer))
        mag_opp = _as_scalar(calculate_apparent_magnitude_v(H, coords_opposition, earth_observer))

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

        array_result = calculate_apparent_magnitude_v(H_array, multi_coords, multi_observer)

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
            mag = calculate_apparent_magnitude_v(H_array[i], single_coords, single_observer)
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
        assert _as_scalar(calculate_apparent_magnitude_v(H, obj, observer, G=0.0)) == pytest.approx(expected, abs=1e-10)
        assert _as_scalar(calculate_apparent_magnitude_v(H, obj, observer, G=1.0)) == pytest.approx(expected, abs=1e-10)

    def test_apparent_mag_convert_magnitude_matches_explicit_conversion(self):
        """
        With V-only magnitude functions, conversion should be done explicitly:

            convert_magnitude(m_v, "V", target)
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

        m_v = _as_scalar(calculate_apparent_magnitude_v(H, obj, observer))

        # Test a few representative filters (string and enum)
        for name, enum in [("g", StandardFilters.g), ("LSST_r", InstrumentFilters.LSST_r)]:
            expected_str = _convert_scalar(m_v, "V", name)
            expected_enum = _convert_scalar(m_v, "V", enum)
            assert expected_str == pytest.approx(expected_enum, abs=1e-10)

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
            calculate_apparent_magnitude_v(H, coords_len1, earth_observer)

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
            calculate_apparent_magnitude_v(H, coords_len2, observer_len2, G=G)

    def test_convert_magnitude_invertibility(self):
        """Round-trip conversions should recover the original magnitude."""
        test_magnitude = 15.0

        for filter1, filter2 in [("V", "g"), ("LSST_g", "g"), ("DECam_r", "r")]:
            converted = _convert_scalar(test_magnitude, filter1, filter2)
            back = _convert_scalar(converted, filter2, filter1)
            assert back == pytest.approx(test_magnitude, abs=1e-3)

    def test_convert_magnitude_identity(self):
        """Same-filter conversion is identity."""
        assert _convert_scalar(15.0, "V", "V") == 15.0

    def test_convert_magnitude_mixed_filters_vectorized(self):
        """Vector inputs can provide per-element source/target filters."""
        mags = np.array([15.0, 16.0, 17.0], dtype=float)
        src = np.array(["V", "V", "V"], dtype=object)
        tgt = np.array(["g", "r", "V"], dtype=object)

        out = convert_magnitude(mags, src, tgt)
        expected = np.array(
            [
                _convert_scalar(mags[0], "V", "g"),
                _convert_scalar(mags[1], "V", "r"),
                _convert_scalar(mags[2], "V", "V"),
            ],
            dtype=float,
        )
        assert np.allclose(out, expected, rtol=0.0, atol=1e-12)

    def test_convert_magnitude_length_one_arrays(self):
        """Length-1 arrays are the scalar use-case for the array-only API."""
        mags = np.array([15.0], dtype=float)
        src = np.array(["V"], dtype=object)
        tgt = np.array(["g"], dtype=object)

        out1 = convert_magnitude(mags, src, tgt)
        # Compare against the same conversion done in a longer vector call.
        out2 = convert_magnitude(
            np.array([15.0, 99.0], dtype=float),
            np.array(["V", "V"], dtype=object),
            np.array(["g", "g"], dtype=object),
        )
        assert _as_scalar(out1) == pytest.approx(float(out2[0]), abs=1e-12)

    def test_convert_magnitude_accepts_enum_arrays(self):
        """Filters may be provided as Enums inside object ndarrays."""
        mags = np.array([15.0, 16.0], dtype=float)
        src = np.array([StandardFilters.V, StandardFilters.V], dtype=object)
        tgt = np.array([StandardFilters.g, InstrumentFilters.LSST_r], dtype=object)

        out = convert_magnitude(mags, src, tgt)
        expected = np.array(
            [
                _convert_scalar(15.0, "V", "g"),
                _convert_scalar(16.0, "V", "LSST_r"),
            ],
            dtype=float,
        )
        assert np.allclose(out, expected, rtol=0.0, atol=1e-12)

    def test_encode_filters_accepts_strings_and_enums(self):
        filters = np.array(["V", StandardFilters.g, InstrumentFilters.LSST_r], dtype=object)
        codes = encode_filters(filters)
        assert codes.shape == (3,)
        assert codes.dtype == np.int32

    def test_convert_magnitude_validates_1d_inputs(self):
        mags2d = np.array([[15.0, 16.0]], dtype=float)
        src = np.array(["V", "V"], dtype=object)
        tgt = np.array(["g", "g"], dtype=object)
        with pytest.raises(ValueError, match="magnitude must be a 1D"):
            convert_magnitude(mags2d, src, tgt)
        with pytest.raises(ValueError, match="source_filter must be a 1D"):
            convert_magnitude(np.array([15.0, 16.0], dtype=float), src.reshape(1, 2), tgt)
        with pytest.raises(ValueError, match="target_filter must be a 1D"):
            convert_magnitude(np.array([15.0, 16.0], dtype=float), src, tgt.reshape(1, 2))

    def test_convert_magnitude_jax_matches_numpy_for_mixed_filters(self):
        mags = np.array([15.0, 16.0, 17.0], dtype=float)
        src = np.array(["V", "V", "V"], dtype=object)
        tgt = np.array(["g", "r", "V"], dtype=object)

        out_np = convert_magnitude(mags, src, tgt)
        out_jax = np.asarray(convert_magnitude_jax(mags, src, tgt))
        assert np.allclose(out_jax, out_np, rtol=1e-7, atol=1e-12)

    def test_convert_magnitude_jax_codes_matches_numpy(self):
        mags = np.array([15.0, 16.0, 17.0], dtype=float)
        src = np.array(["V", "V", "V"], dtype=object)
        tgt = np.array(["g", "r", "V"], dtype=object)
        src_codes = encode_filters(src)
        tgt_codes = encode_filters(tgt)

        out_np = convert_magnitude(mags, src, tgt)
        out_jax = np.asarray(convert_magnitude_jax_codes(mags, src_codes, tgt_codes))
        assert np.allclose(out_jax, out_np, rtol=1e-7, atol=1e-12)

    def test_convert_magnitude_jax_length_mismatch_raises(self):
        mags = np.array([15.0, 16.0], dtype=float)
        src = np.array(["V"], dtype=object)
        tgt = np.array(["g", "g"], dtype=object)
        with pytest.raises(ValueError, match="source_filter length"):
            convert_magnitude_jax(mags, src, tgt)

    def test_convert_magnitude_auto_backend_explicit(self):
        mags = np.array([15.0, 16.0, 17.0], dtype=float)
        src = np.array(["V", "V", "V"], dtype=object)
        tgt = np.array(["g", "r", "V"], dtype=object)

        out_np = convert_magnitude(mags, src, tgt)
        out_auto_np = convert_magnitude_auto(mags, src, tgt, use_jax=False)
        out_auto_jax = convert_magnitude_auto(mags, src, tgt, use_jax=True)

        assert np.allclose(out_auto_np, out_np, rtol=0.0, atol=1e-12)
        assert np.allclose(out_auto_jax, out_np, rtol=1e-7, atol=1e-12)

    def test_convert_magnitude_auto_selects_backend_by_n(self, monkeypatch):
        # Import the module so we can monkeypatch the actual function objects used by the wrapper.
        from .. import simple_magnitude as sm

        def fake_np(m, s, t):
            return np.full(int(m.shape[0]), 1.0, dtype=float)

        class _FakeJaxArr:
            def __init__(self, n):
                self._n = n

            def __array__(self, dtype=None):
                return np.full(self._n, 2.0, dtype=float if dtype is None else dtype)

        def fake_jax(m, s, t):
            return _FakeJaxArr(int(m.shape[0]))

        monkeypatch.setattr(sm, "convert_magnitude", fake_np)
        monkeypatch.setattr(sm, "convert_magnitude_jax", fake_jax)

        mags_small = np.zeros(3, dtype=float)
        mags_big = np.zeros(5, dtype=float)
        src_small = np.array(["V"] * 3, dtype=object)
        tgt_small = np.array(["V"] * 3, dtype=object)
        src_big = np.array(["V"] * 5, dtype=object)
        tgt_big = np.array(["V"] * 5, dtype=object)

        out_small = sm.convert_magnitude_auto(
            mags_small, src_small, tgt_small, use_jax=None, jax_threshold=4
        )
        out_big = sm.convert_magnitude_auto(
            mags_big, src_big, tgt_big, use_jax=None, jax_threshold=4
        )

        assert np.allclose(out_small, 1.0)
        assert np.allclose(out_big, 2.0)

    def test_convert_magnitude_mixed_filters_broadcast_source(self):
        """If you want broadcast, build the full arrays explicitly."""
        mags = np.array([15.0, 16.0, 17.0], dtype=float)
        src = np.array(["V", "V", "V"], dtype=object)
        tgt = np.array([StandardFilters.g, StandardFilters.r, StandardFilters.g], dtype=object)

        out = convert_magnitude(mags, src, tgt)
        expected = np.array(
            [
                _convert_scalar(mags[0], "V", "g"),
                _convert_scalar(mags[1], "V", "r"),
                _convert_scalar(mags[2], "V", "g"),
            ],
            dtype=float,
        )
        assert np.allclose(out, expected, rtol=0.0, atol=1e-12)

    def test_convert_magnitude_mixed_filters_length_mismatch_raises(self):
        mags = np.array([15.0, 16.0], dtype=float)
        with pytest.raises(ValueError, match="length.*must match magnitude length"):
            convert_magnitude(
                mags,
                np.asarray(["V"], dtype=object),
                np.asarray(["g", "r"], dtype=object),
            )

    def test_convert_magnitude_invalid_filters(self):
        """Unknown filters or unreachable paths should raise ValueError."""
        with pytest.raises(ValueError, match="No conversion path"):
            _convert_scalar(15.0, "NonExistentFilter", "V")

        with pytest.raises(ValueError, match="No conversion path"):
            _convert_scalar(15.0, "V", "NonExistentFilter")

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

        mags_np = calculate_apparent_magnitude_v(H, obj, observer, G=G)
        mags_jax = calculate_apparent_magnitude_v_jax(H, obj, observer, G=G)

        # JAX may run in float32 depending on configuration; keep tolerance realistic.
        assert np.allclose(np.asarray(mags_jax), mags_np, rtol=1e-6, atol=1e-8)

    def test_apparent_magnitude_auto_backend_explicit(self, earth_observer):
        rng = np.random.default_rng(0)
        n = 16
        time = Timestamp.from_mjd(np.full(n, 60000), scale="tdb")

        obj = CartesianCoordinates.from_kwargs(
            x=rng.uniform(1.2, 3.0, size=n),
            y=rng.uniform(0.1, 2.0, size=n),
            z=rng.uniform(-0.5, 0.5, size=n),
            vx=np.zeros(n),
            vy=np.zeros(n),
            vz=np.zeros(n),
            time=time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * n),
        )
        H = rng.uniform(10.0, 25.0, size=n)
        G = rng.uniform(0.0, 1.0, size=n)

        mags_np = calculate_apparent_magnitude_v(H, obj, earth_observer, G=G)
        mags_auto_np = calculate_apparent_magnitude_v_auto(
            H, obj, earth_observer, G=G, use_jax=False
        )
        mags_auto_jax = calculate_apparent_magnitude_v_auto(
            H, obj, earth_observer, G=G, use_jax=True
        )

        assert np.allclose(np.asarray(mags_auto_np), mags_np, rtol=1e-10, atol=1e-12)
        assert np.allclose(np.asarray(mags_auto_jax), mags_np, rtol=1e-6, atol=1e-8)

    def test_apparent_magnitude_auto_selects_backend_by_n(self, monkeypatch, earth_observer):
        from .. import simple_magnitude as sm

        def fake_np(H_v, object_coords, observer, G=0.15):
            return np.full(len(object_coords), 1.0, dtype=float)

        def fake_jax(H_v, object_coords, observer, G=0.15):
            return np.full(len(object_coords), 2.0, dtype=float)

        monkeypatch.setattr(sm, "calculate_apparent_magnitude_v", fake_np)
        monkeypatch.setattr(sm, "calculate_apparent_magnitude_v_jax", fake_jax)

        # Reuse a valid 1-length coord/observer from the fixture by slicing
        obj1 = CartesianCoordinates.from_kwargs(
            x=[2.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.0],
            vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )
        # Make a 5-length object set by repeating; observer fixture is length-1 so rebuild.
        obj5 = CartesianCoordinates.from_kwargs(
            x=[2.0] * 5,
            y=[0.0] * 5,
            z=[0.0] * 5,
            vx=[0.0] * 5,
            vy=[0.0] * 5,
            vz=[0.0] * 5,
            time=Timestamp.from_mjd([60000] * 5, scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * 5),
        )
        obs5 = Observers.from_kwargs(
            code=["500"] * 5,
            coordinates=CartesianCoordinates.from_kwargs(
                x=[1.0] * 5,
                y=[0.0] * 5,
                z=[0.0] * 5,
                vx=[0.0] * 5,
                vy=[0.0] * 5,
                vz=[0.0] * 5,
                time=Timestamp.from_mjd([60000] * 5, scale="tdb"),
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"] * 5),
            ),
        )

        out_small = sm.calculate_apparent_magnitude_v_auto(
            15.0, obj1, earth_observer, use_jax=None, jax_threshold=4
        )
        out_big = sm.calculate_apparent_magnitude_v_auto(
            15.0, obj5, obs5, use_jax=None, jax_threshold=4
        )

        assert np.allclose(out_small, 1.0)
        assert np.allclose(out_big, 2.0)

    def test_apparent_magnitude_v_jax_matches_explicit_conversion(self, earth_observer):
        """JAX V-band magnitude + explicit conversion should match NumPy explicit conversion."""
        H = 15.0
        obj = CartesianCoordinates.from_kwargs(
            x=[2.0], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.0], vz=[0.0],
            time=Timestamp.from_mjd([60000], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        )

        m_v_np = _as_scalar(calculate_apparent_magnitude_v(H, obj, earth_observer))
        m_v_jax = _as_scalar(calculate_apparent_magnitude_v_jax(H, obj, earth_observer))
        m_g_np = _convert_scalar(m_v_np, "V", "g")
        m_g_jax = _convert_scalar(m_v_jax, "V", "g")
        assert m_g_jax == pytest.approx(m_g_np, abs=1e-6)
