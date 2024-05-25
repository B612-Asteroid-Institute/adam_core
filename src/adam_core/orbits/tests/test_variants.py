import numpy as np

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
