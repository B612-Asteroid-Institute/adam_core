import numpy as np
import pyarrow.compute as pc
import pytest

from ...coordinates import CartesianCoordinates, Origin, SphericalCoordinates
from ...orbits.physical_parameters import PhysicalParameters
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from ..variants import VariantEphemeris, VariantOrbits


def test_VariantOrbits():

    # Get a sample of real orbits
    orbits = make_real_orbits(10)

    # Create a variant orbits object (expands the covariance matrix)
    # around the mean state
    variant_orbits = VariantOrbits.create(orbits)

    # For these 10 orbits this will select sigma-points so lets
    # check that the number of sigma-points is correct
    assert len(variant_orbits) == len(orbits) * 13

    # Now lets collapse the sigma-points back and see if we can reconstruct
    # the input covairance matrix
    collapsed_orbits = variant_orbits.collapse(orbits)

    # Check that the covariance matrices are close
    np.testing.assert_allclose(
        collapsed_orbits.coordinates.covariance.to_matrix(),
        orbits.coordinates.covariance.to_matrix(),
        rtol=0,
        atol=1e-14,
    )

    # Check that the orbit ids are the same
    np.testing.assert_equal(
        collapsed_orbits.orbit_id.to_numpy(zero_copy_only=False),
        orbits.orbit_id.to_numpy(zero_copy_only=False),
    )


def test_VariantOrbits_collapse_by_object_id():
    """Test that VariantOrbits.collapse_by_object_id correctly collapses variants into mean orbits."""

    # Create variant orbits with multiple objects, each having multiple variants
    variant_orbits = VariantOrbits.from_kwargs(
        orbit_id=["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"],
        object_id=["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"],
        variant_id=["0", "1", "2", "0", "1", "2"],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.0, 15.0, 15.0, 17.5, 17.5, 17.5],
            G=[0.15, 0.15, 0.15, 0.25, 0.25, 0.25],
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            y=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            z=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            vx=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            vy=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            vz=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            time=Timestamp.from_mjd([60000] * 6),
            origin=Origin.from_kwargs(code=["SUN"] * 6),
            frame="ecliptic",
        ),
    )

    # Collapse the variants
    collapsed = variant_orbits.collapse_by_object_id()

    # Check basic properties
    assert len(collapsed) == 2  # Should have one orbit per object
    assert set(collapsed.object_id.to_pylist()) == {"obj1", "obj2"}

    # Check that means are computed correctly for each object
    obj1 = collapsed.select("object_id", "obj1")
    obj2 = collapsed.select("object_id", "obj2")

    # Check means for obj1
    np.testing.assert_allclose(
        obj1.coordinates.values[0],
        np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),  # Expected mean for obj1
        rtol=1e-14,
    )

    # Check means for obj2
    np.testing.assert_allclose(
        obj2.coordinates.values[0],
        np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2]),  # Expected mean for obj2
        rtol=1e-14,
    )

    # Physical parameters should be preserved per object_id
    assert obj1.physical_parameters.H_v[0].as_py() == 15.0
    assert obj1.physical_parameters.G[0].as_py() == 0.15
    assert obj2.physical_parameters.H_v[0].as_py() == 17.5
    assert obj2.physical_parameters.G[0].as_py() == 0.25

    # Check that covariance matrices are computed correctly
    # For obj1, the variance should be approximately 0.00667 for each component
    obj1_cov = obj1.coordinates.covariance.to_matrix()[0]
    # The variance is sum((x - mean)^2) / n where n=3
    # For positions: (1.1 - 1.0)^2 + (0.9 - 1.0)^2 + (1.0 - 1.0)^2 = 0.02
    # So variance = 0.02/3 ≈ 0.00667
    expected_variance_obj1_pos = 0.02 / 3  # Population variance
    expected_variance_obj1_vel = 0.0002 / 3  # Population variance
    np.testing.assert_allclose(
        np.diag(obj1_cov),
        [expected_variance_obj1_pos] * 3 + [expected_variance_obj1_vel] * 3,
        rtol=1e-6,
    )

    # For obj2, the variance should also be approximately 0.00667
    obj2_cov = obj2.coordinates.covariance.to_matrix()[0]
    expected_variance_obj2_pos = (
        0.02 / 3
    )  # (2.1 - 2.0)^2 + (1.9 - 2.0)^2 + (2.0 - 2.0)^2 = 0.02/3
    expected_variance_obj2_vel = (
        0.0002 / 3
    )  # (0.21 - 0.2)^2 + (0.19 - 0.2)^2 + (0.2 - 0.2)^2 = 0.0002/3
    np.testing.assert_allclose(
        np.diag(obj2_cov),
        [expected_variance_obj2_pos] * 3 + [expected_variance_obj2_vel] * 3,
        rtol=1e-6,
    )

    # Test that time and origin are preserved
    assert all(t == 60000 for t in collapsed.coordinates.time.mjd().to_pylist())
    assert all(o == "SUN" for o in collapsed.coordinates.origin.code.to_pylist())
    assert collapsed.coordinates.frame == "ecliptic"

    # Test error cases
    # Test that variants with different times raise an error
    variant_orbits_diff_times = VariantOrbits.from_kwargs(
        orbit_id=["obj1", "obj1"],
        object_id=["obj1", "obj1"],
        variant_id=["0", "1"],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.0, 15.0], G=[0.15, 0.15]
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.1],
            y=[1.0, 1.1],
            z=[1.0, 1.1],
            vx=[0.1, 0.11],
            vy=[0.1, 0.11],
            vz=[0.1, 0.11],
            time=Timestamp.from_mjd([60000, 60001]),  # Different times
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )
    with pytest.raises(AssertionError):
        variant_orbits_diff_times.collapse_by_object_id()

    # Test that variants with different origins raise an error
    variant_orbits_diff_origins = VariantOrbits.from_kwargs(
        orbit_id=["obj1", "obj1"],
        object_id=["obj1", "obj1"],
        variant_id=["0", "1"],
        physical_parameters=PhysicalParameters.from_kwargs(
            H_v=[15.0, 15.0], G=[0.15, 0.15]
        ),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 1.1],
            y=[1.0, 1.1],
            z=[1.0, 1.1],
            vx=[0.1, 0.11],
            vy=[0.1, 0.11],
            vz=[0.1, 0.11],
            time=Timestamp.from_mjd([60000, 60000]),
            origin=Origin.from_kwargs(code=["SUN", "EARTH"]),  # Different origins
            frame="ecliptic",
        ),
    )
    with pytest.raises(AssertionError):
        variant_orbits_diff_origins.collapse_by_object_id()


def test_VariantEphemeris_collapse_by_object_id_single_epoch():
    """Test that VariantEphemeris.collapse_by_object_id collapses by object_id for a single epoch."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"],
        object_id=["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"],
        variant_id=["0", "1", "2", "0", "1", "2"],
        predicted_magnitude_v=[20.0, 21.0, 19.0, 18.0, 18.5, 17.5],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            lon=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            lat=[1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            vrho=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            vlon=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            vlat=[0.1, 0.11, 0.09, 0.2, 0.21, 0.19],
            time=Timestamp.from_mjd([60000] * 6),
            origin=Origin.from_kwargs(code=["500"] * 6),
            frame="equatorial",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()

    assert len(collapsed) == 2
    assert set(collapsed.object_id.to_pylist()) == {"obj1", "obj2"}

    obj1 = collapsed.select("object_id", "obj1")
    obj2 = collapsed.select("object_id", "obj2")

    np.testing.assert_allclose(
        obj1.coordinates.values[0],
        np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        obj2.coordinates.values[0],
        np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2]),
        rtol=1e-14,
    )

    # Mean magnitudes should be propagated.
    assert obj1.predicted_magnitude_v[0].as_py() == pytest.approx(20.0)
    assert obj2.predicted_magnitude_v[0].as_py() == pytest.approx(18.0)

    # Check covariance diagonals (population variance with n=3).
    obj1_cov = obj1.coordinates.covariance.to_matrix()[0]
    expected_variance_pos = 0.02 / 3
    expected_variance_vel = 0.0002 / 3
    np.testing.assert_allclose(
        np.diag(obj1_cov),
        [expected_variance_pos] * 3 + [expected_variance_vel] * 3,
        rtol=1e-6,
    )

    obj2_cov = obj2.coordinates.covariance.to_matrix()[0]
    np.testing.assert_allclose(
        np.diag(obj2_cov),
        [expected_variance_pos] * 3 + [expected_variance_vel] * 3,
        rtol=1e-6,
    )

    # Time, origin, and frame should be preserved.
    assert all(t == 60000 for t in collapsed.coordinates.time.mjd().to_pylist())
    assert all(o == "500" for o in collapsed.coordinates.origin.code.to_pylist())
    assert collapsed.coordinates.frame == "equatorial"


def test_VariantEphemeris_collapse_by_object_id_groups_by_time_and_origin():
    """Collapse should be performed per (object_id, time, origin_code) group."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1"] * 6,
        object_id=["obj1"] * 6,
        variant_id=["0", "1", "0", "1", "0", "1"],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.2, 10.0, 10.2, 100.0, 100.2],
            lon=[1.0, 1.2, 10.0, 10.2, 100.0, 100.2],
            lat=[1.0, 1.2, 10.0, 10.2, 100.0, 100.2],
            vrho=[0.1, 0.12, 1.0, 1.02, 10.0, 10.02],
            vlon=[0.1, 0.12, 1.0, 1.02, 10.0, 10.02],
            vlat=[0.1, 0.12, 1.0, 1.02, 10.0, 10.02],
            time=Timestamp.from_mjd([60000, 60000, 60001, 60001, 60000, 60000]),
            origin=Origin.from_kwargs(code=["500", "500", "500", "500", "X05", "X05"]),
            frame="equatorial",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()

    # Keys: (60000, "500"), (60001, "500"), (60000, "X05")
    assert len(collapsed) == 3
    assert set(collapsed.coordinates.origin.code.to_pylist()) == {"500", "X05"}
    assert set(collapsed.coordinates.time.mjd().to_pylist()) == {60000, 60001}

    # Validate one group: mjd=60000, origin=500 -> mean of [1.0, 1.2] => 1.1 etc.
    group = (
        collapsed.select("coordinates.origin.code", "500")
        .select("coordinates.time.days", 60000)
        .select("coordinates.time.nanos", 0)
    )
    assert len(group) == 1
    np.testing.assert_allclose(
        group.coordinates.values[0],
        np.array([1.1, 1.1, 1.1, 0.11, 0.11, 0.11]),
        rtol=0,
        atol=1e-12,
    )


def test_VariantEphemeris_collapse_by_object_id_wraps_longitude():
    """Longitude is circular in degrees; mean should respect wrap-around near 0/360."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1", "obj1"],
        object_id=["obj1", "obj1", "obj1"],
        variant_id=["0", "1", "2"],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.0, 1.0],
            lon=[359.0, 1.0, 0.0],
            lat=[0.0, 0.0, 0.0],
            vrho=[0.0, 0.0, 0.0],
            vlon=[0.0, 0.0, 0.0],
            vlat=[0.0, 0.0, 0.0],
            time=Timestamp.from_mjd([60000] * 3),
            origin=Origin.from_kwargs(code=["500"] * 3),
            frame="equatorial",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()
    assert len(collapsed) == 1
    lon = collapsed.coordinates.values[0][1]
    assert lon == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= lon < 360.0


def test_VariantEphemeris_collapse_by_object_id_partial_aberrated_raises():
    """Aberrated coordinates are ignored/dropped and regenerated after collapse."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1"],
        object_id=["obj1", "obj1"],
        variant_id=["0", "1"],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.1],
            lon=[1.0, 1.1],
            lat=[1.0, 1.1],
            vrho=[0.1, 0.11],
            vlon=[0.1, 0.11],
            vlat=[0.1, 0.11],
            time=Timestamp.from_mjd([60000] * 2),
            origin=Origin.from_kwargs(code=["500"] * 2),
            frame="equatorial",
        ),
        aberrated_coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, None],
            y=[1.0, None],
            z=[1.0, None],
            vx=[0.1, None],
            vy=[0.1, None],
            vz=[0.1, None],
            time=Timestamp.from_mjd([60000] * 2),
            origin=Origin.from_kwargs(code=["SUN"] * 2),
            frame="ecliptic",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()
    assert len(collapsed) == 1
    assert not pc.all(pc.is_null(collapsed.aberrated_coordinates.x)).as_py()
    assert not pc.all(pc.is_null(collapsed.light_time)).as_py()


def test_VariantEphemeris_collapse_by_object_id_aberrated_times_can_vary():
    """Collapsed aberrated emission times should be consistent with light_time."""
    variant_ephemeris = VariantEphemeris.from_kwargs(
        orbit_id=["obj1", "obj1", "obj1"],
        object_id=["obj1", "obj1", "obj1"],
        variant_id=["0", "1", "2"],
        coordinates=SphericalCoordinates.from_kwargs(
            rho=[1.0, 1.1, 0.9],
            lon=[1.0, 1.1, 0.9],
            lat=[1.0, 1.1, 0.9],
            vrho=[0.1, 0.11, 0.09],
            vlon=[0.1, 0.11, 0.09],
            vlat=[0.1, 0.11, 0.09],
            time=Timestamp.from_mjd([60000.0] * 3, scale="utc"),
            origin=Origin.from_kwargs(code=["500"] * 3),
            frame="equatorial",
        ),
    )

    collapsed = variant_ephemeris.collapse_by_object_id()
    assert len(collapsed) == 1

    coords_tdb = collapsed.coordinates.time.rescale("tdb")
    aberr_tdb = collapsed.aberrated_coordinates.time.rescale("tdb")
    delta_days, delta_nanos = coords_tdb.difference(aberr_tdb)
    fractional_days = pc.divide(delta_nanos, 86400 * 1e9)
    delta = pc.add(delta_days, fractional_days).to_numpy(zero_copy_only=False)
    lt = collapsed.light_time.to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(delta, lt, atol=1e-6)
