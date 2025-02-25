import numpy as np
import pytest

from ...coordinates import CartesianCoordinates, Origin
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from ..variants import VariantOrbits


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
