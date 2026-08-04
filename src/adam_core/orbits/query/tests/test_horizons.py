import numpy as np

from adam_core.orbits.query import query_horizons
from adam_core.time import Timestamp


def test_query_horizons_chunking():
    """Test that query_horizons correctly handles batches of times larger than the time limit (50)."""

    # Create 100 timestamps (more than the 50 time limit)
    base_time = "2024-01-01T00:00:00Z"
    times = Timestamp.from_iso8601([base_time for _ in range(100)], scale="utc")
    # Add different hours to each timestamp to make them unique
    times = times.add_fractional_days(np.arange(100) / 24)

    # Test object - using Bennu as it's a well-known object
    object_ids = ["101955"]  # Bennu's JPL small-body ID

    # Test all coordinate types
    coordinate_types = ["cartesian", "keplerian", "cometary"]

    for coord_type in coordinate_types:
        # Query for all times at once (should trigger internal batching)
        orbits_batched = query_horizons(
            object_ids,
            times,
            coordinate_type=coord_type,
            location="@sun",
            id_type="smallbody",
        )

        # Check that passing None for id_type works
        orbits_batched_none = query_horizons(
            object_ids,
            times,
            coordinate_type=coord_type,
            location="@sun",
        )

        # Check that the results are the same
        np.testing.assert_allclose(
            orbits_batched.coordinates.r, orbits_batched_none.coordinates.r, rtol=1e-15
        )

        # Query in two explicit batches to compare
        orbits_first_half = query_horizons(
            object_ids,
            times[:50],
            coordinate_type=coord_type,
            location="@sun",
            id_type="smallbody",
        )
        orbits_second_half = query_horizons(
            object_ids,
            times[50:],
            coordinate_type=coord_type,
            location="@sun",
            id_type="smallbody",
        )

        # Verify we got all times
        assert (
            len(orbits_batched) == 100
        ), f"Expected 100 orbits for {coord_type}, got {len(orbits_batched)}"

        # Verify the results are identical whether queried in batches or all at once
        # Compare positions
        np.testing.assert_allclose(
            orbits_batched.coordinates.r[:50],
            orbits_first_half.coordinates.r,
            rtol=1e-15,
        )
        np.testing.assert_allclose(
            orbits_batched.coordinates.r[50:],
            orbits_second_half.coordinates.r,
            rtol=1e-15,
        )

        # Compare velocities
        np.testing.assert_allclose(
            orbits_batched.coordinates.v[:50],
            orbits_first_half.coordinates.v,
            rtol=1e-15,
        )
        np.testing.assert_allclose(
            orbits_batched.coordinates.v[50:],
            orbits_second_half.coordinates.v,
            rtol=1e-15,
        )
