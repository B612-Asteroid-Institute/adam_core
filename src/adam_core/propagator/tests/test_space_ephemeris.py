"""
Tests for generating ephemeris using custom SPICE kernels with JWST as an observer.
This file demonstrates how to register custom kernels and create observers from them.
"""

from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest
from adam_assist import ASSISTPropagator

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observers.observers import Observers
from ...orbits.orbits import Orbits
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from ...utils.spice import register_spice_kernel, unregister_spice_kernel

# JWST SPICE kernel paths
JWST_KERNEL_DIR = (
    Path(__file__).parent.parent.parent / "utils" / "tests" / "data" / "spice"
)
JWST_KERNEL_PATH = JWST_KERNEL_DIR / "jwst_horizons_20200101_20240101_v01.bsp"


@pytest.fixture
def jwst_kernel():
    """Fixture to handle JWST kernel registration and cleanup."""
    if not JWST_KERNEL_PATH.exists():
        pytest.skip(
            "JWST SPICE kernel not found. Please download it to tests/data/spice/jwst_horizons_20200101_20240101_v01.bsp"
        )

    register_spice_kernel(str(JWST_KERNEL_PATH))
    yield
    unregister_spice_kernel(str(JWST_KERNEL_PATH))


@pytest.mark.parametrize("max_processes", [1, 4])
def test_generate_ephemeris_with_custom_kernel(jwst_kernel, max_processes):
    """Test generating ephemeris using the JWST kernel with ASSISTPropagator."""
    # Create observation times within the JWST kernel range (2020-01-01 to 2024-01-01)
    obs_times = Timestamp.from_iso8601(
        [
            "2022-06-15T00:00:00Z",
            "2022-06-15T06:00:00Z",
            "2022-06-15T12:00:00Z",
            "2022-06-15T18:00:00Z",
        ]
    )

    # Create JWST observers using the registered kernel
    jwst_observers = Observers.from_code("JWST", obs_times)

    # Create some test orbits - using the helper function
    orbits = make_real_orbits(5)

    # Initialize the ASSISTPropagator
    propagator = ASSISTPropagator()

    # Generate ephemeris using the custom kernel observer
    ephemeris = propagator.generate_ephemeris(
        orbits, jwst_observers, max_processes=max_processes
    )

    # Verify basic properties of the generated ephemeris
    assert len(ephemeris) == len(orbits) * len(jwst_observers)

    # Check that observer is correctly set to JWST
    assert np.all(
        [code.as_py() == "JWST" for code in ephemeris.coordinates.origin.code]
    )

    # Check that the ephemeris time scale is UTC (as expected for output)
    assert ephemeris.coordinates.time.scale == "utc"

    # Verify light time calculations are reasonable
    # Light time should be positive and less than ~1 hour (0.04 days) for Solar System objects
    assert np.all(ephemeris.light_time.to_numpy(zero_copy_only=False) > 0)
    assert np.all(ephemeris.light_time.to_numpy(zero_copy_only=False) < 0.04)

    # Verify that coordinates.time - aberrated_coordinates.time equals light_time
    time_difference_days, time_difference_nanos = ephemeris.coordinates.time.rescale(
        "tdb"
    ).difference(ephemeris.aberrated_coordinates.time)
    fractional_days = pc.divide(time_difference_nanos, 86400 * 1e9)
    time_difference = pc.add(time_difference_days, fractional_days)
    np.testing.assert_allclose(
        time_difference.to_numpy(zero_copy_only=False),
        ephemeris.light_time.to_numpy(zero_copy_only=False),
        atol=1e-6,
    )


@pytest.mark.parametrize("max_processes", [1, 4])
def test_mixed_observer_ephemeris(jwst_kernel, max_processes):
    """Test generating ephemeris with both JWST and ground-based observers."""

    # Create combined observer list with both JWST and a ground station
    all_observers = Observers.from_codes(
        ["JWST", "500", "JWST", "500"],  # JWST and geocenter (500)
        Timestamp.from_iso8601(
            [
                "2022-06-15T00:00:00Z",
                "2022-06-15T00:00:00Z",
                "2022-06-15T12:00:00Z",
                "2022-06-15T12:00:00Z",
            ]
        ),
    )

    # Create specific test orbits with known properties
    test_orbits = Orbits.from_kwargs(
        orbit_id=["near_earth", "distant"],
        object_id=["near_earth", "distant"],
        coordinates=CartesianCoordinates.from_kwargs(
            # First orbit close to Earth, second one further away
            x=[1.0, 3.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[1.0, 0.6],  # Different orbital velocities
            vz=[0.0, 0.0],
            # Need to pick a time close to JWST spice kernel observer times so we don't
            # launch out of the solar system
            time=Timestamp.from_iso8601(
                ["2022-06-14T00:00:00Z", "2022-06-14T00:00:00Z"]
            ),  # Same start time for both orbits
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )

    # Generate ephemeris with mixed observer types
    propagator = ASSISTPropagator()
    ephemeris = propagator.generate_ephemeris(
        test_orbits, all_observers, max_processes=max_processes
    )

    # Verify we got ephemeris for all combinations
    assert len(ephemeris) == len(test_orbits) * len(all_observers)

    # Check that we have both types of observers in the results
    observer_codes = [code.as_py() for code in ephemeris.coordinates.origin.code]
    assert "JWST" in observer_codes
    assert "500" in observer_codes

    # Separate ephemeris by observer type
    jwst_ephemeris = ephemeris.select("coordinates.origin.code", "JWST")
    geo_ephemeris = ephemeris.select("coordinates.origin.code", "500")

    # Both should have the same number of rows
    assert len(jwst_ephemeris) == len(geo_ephemeris)

    # For the near-Earth object, there should be a noticeable difference in position
    # as observed from JWST vs. Earth for the same timestamp
    near_earth_jwst = jwst_ephemeris.select("orbit_id", "near_earth")
    near_earth_geo = geo_ephemeris.select("orbit_id", "near_earth")

    # Extract spherical coordinates to compare the angular differences
    jwst_ra = near_earth_jwst.coordinates.lon.to_numpy(zero_copy_only=False)
    geo_ra = near_earth_geo.coordinates.lon.to_numpy(zero_copy_only=False)
    jwst_dec = near_earth_jwst.coordinates.lat.to_numpy(zero_copy_only=False)
    geo_dec = near_earth_geo.coordinates.lat.to_numpy(zero_copy_only=False)

    # There should be measurable parallax between Earth and JWST for near objects
    # (at least a few arcseconds)
    ra_diff = np.abs(jwst_ra - geo_ra)
    dec_diff = np.abs(jwst_dec - geo_dec)

    # Check that there's at least some difference in the observed positions
    assert np.any(ra_diff > 1e-3) or np.any(dec_diff > 1e-3)

    # Verify linkage to observers works
    linkage = ephemeris.link_to_observers(all_observers)
    assert len(linkage.all_unique_values) == len(all_observers)
