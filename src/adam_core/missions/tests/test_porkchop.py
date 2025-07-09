import numpy as np
import pytest

from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.units import au_per_day_to_km_per_s
from adam_core.time import Timestamp

from ..porkchop import (
    generate_porkchop_data,
    plot_porkchop_plotly,
    prepare_and_propagate_orbits,
)


def test_generate_porkchop_data_origins():
    # Test with different origins
    departure_start = Timestamp.from_mjd([60000], scale="tdb")
    departure_end = Timestamp.from_mjd([60050], scale="tdb")
    arrival_start = Timestamp.from_mjd([60050], scale="tdb")
    arrival_end = Timestamp.from_mjd([60100], scale="tdb")

    # Get departure orbits for Earth
    departure_orbits = prepare_and_propagate_orbits(
        body=OriginCodes.EARTH,
        start_time=departure_start,
        end_time=departure_end,
        propagation_origin=OriginCodes.SUN,
        step_size=5.0,  # Larger step size for faster test
    )

    # Get arrival orbits for Mars
    arrival_orbits = prepare_and_propagate_orbits(
        body=OriginCodes.MARS_BARYCENTER,
        start_time=arrival_start,
        end_time=arrival_end,
        propagation_origin=OriginCodes.SUN,
        step_size=5.0,  # Larger step size for faster test
    )

    results_sun = generate_porkchop_data(
        departure_orbits=departure_orbits,
        arrival_orbits=arrival_orbits,
    )

    # Verify that results are generated
    assert len(results_sun) > 0

    # Check that origins match what we specified
    assert results_sun.origin.as_OriginCodes() == OriginCodes.SUN

    # Check that time of flight is valid (positive)
    assert np.all(results_sun.time_of_flight() > 0)
    # Check that C3 values are computed
    assert np.all(~np.isnan(results_sun.c3_departure()))


def test_generate_real_porkchop_plot(tmp_path):
    """
    Generate a realistic porkchop plot for an Earth-Mars transfer window.
    This test will generate and display a plot for visual inspection.

    The plot files are saved to a temporary directory that is not deleted
    after the test runs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture that provides a temporary directory unique to the test.
    """
    # Use 2022-2024 Mars transfer window dates
    # These are approximate optimal transfer dates for illustration
    departure_start = Timestamp.from_iso8601(["2022-01-01T00:00:00"], scale="tdb")
    departure_end = Timestamp.from_iso8601(["2022-12-31T00:00:00"], scale="tdb")
    arrival_start = Timestamp.from_iso8601(["2022-06-01T00:00:00"], scale="tdb")
    arrival_end = Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="tdb")

    # Get departure orbits for Earth
    departure_orbits = prepare_and_propagate_orbits(
        body=OriginCodes.EARTH,
        start_time=departure_start,
        end_time=departure_end,
        propagation_origin=OriginCodes.SUN,
        step_size=1.0,  # 1-day intervals for good resolution
    )

    # Get arrival orbits for Mars
    arrival_orbits = prepare_and_propagate_orbits(
        body=OriginCodes.MARS_BARYCENTER,
        start_time=arrival_start,
        end_time=arrival_end,
        propagation_origin=OriginCodes.SUN,
        step_size=1.0,  # 1-day intervals for good resolution
    )
    # Generate porkchop data with reasonable resolution
    results = generate_porkchop_data(
        departure_orbits=departure_orbits,
        arrival_orbits=arrival_orbits,
    )

    # Verify data was generated successfully
    assert len(results) > 0
    assert np.all(~np.isnan(results.c3_departure()))

    # Generate the plot using plotly for interactive visualization
    fig = plot_porkchop_plotly(
        results,
        title="Earth to Mars Transfer (2022-2024)",
        c3_departure_min=0.0,
        c3_departure_max=100.0,
        vinf_arrival_min=0.0,
        vinf_arrival_max=100.0,
        tof_min=0.0,
        tof_max=1000.0,
        show_hover=True,
    )

    # Save to pytest's temporary directory
    output_dir = tmp_path

    # Save as interactive HTML
    html_path = output_dir / "earth_mars_porkchop.html"
    fig.write_html(str(html_path))

    # Try to save as static image if kaleido is installed
    try:
        png_path = output_dir / "earth_mars_porkchop.png"
        fig.write_image(str(png_path))
        print(f"\nPorkchop plot PNG saved to: {png_path}")
    except Exception as e:
        print(f"\nCould not save PNG image: {e}")
        print(
            "To save PNG images, install the kaleido package with: pip install -U kaleido"
        )

    # Print the path to the HTML file
    print(f"\nOpen plot with: open {html_path}")

    # return fig  # Return the figure for further analysis if needed


def test_porkchop_different_time_grids():
    """
    Test that porkchop functions work correctly when departure and arrival
    orbits have different time grids.
    """
    from adam_core.coordinates import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    # Create departure coordinates with one time grid
    departure_times = Timestamp.from_mjd(
        [60000, 60005, 60010, 60015, 60020], scale="tdb"
    )
    departure_coords = CartesianCoordinates.from_kwargs(
        time=departure_times,
        x=[1.0, 1.1, 1.2, 1.3, 1.4],  # Earth-like orbit variations
        y=[0.0, 0.1, 0.2, 0.3, 0.4],
        z=[0.0, 0.01, 0.02, 0.03, 0.04],
        vx=[0.0, 0.01, 0.02, 0.03, 0.04],
        vy=[
            0.017,
            0.016,
            0.015,
            0.014,
            0.013,
        ],  # Earth orbital velocity ~17 km/s -> 0.017 AU/day
        vz=[0.0, 0.001, 0.002, 0.003, 0.004],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(departure_times)),
    )

    # Create arrival coordinates with a DIFFERENT time grid (shifted and different spacing)
    arrival_times = Timestamp.from_mjd(
        [60030, 60040, 60050, 60060], scale="tdb"
    )  # Different times!
    arrival_coords = CartesianCoordinates.from_kwargs(
        time=arrival_times,
        x=[1.5, 1.6, 1.7, 1.8],  # Mars-like orbit
        y=[0.5, 0.6, 0.7, 0.8],
        z=[0.05, 0.06, 0.07, 0.08],
        vx=[0.05, 0.06, 0.07, 0.08],
        vy=[0.010, 0.009, 0.008, 0.007],  # Mars orbital velocity
        vz=[0.005, 0.006, 0.007, 0.008],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(arrival_times)),
    )

    # Convert coordinates to orbits
    departure_orbits = Orbits.from_kwargs(
        orbit_id=[f"departure_{i}" for i in range(len(departure_times))],
        coordinates=departure_coords,
    )

    arrival_orbits = Orbits.from_kwargs(
        orbit_id=[f"arrival_{i}" for i in range(len(arrival_times))],
        coordinates=arrival_coords,
    )

    # Generate porkchop data - this should work with different time grids
    results = generate_porkchop_data(
        departure_orbits=departure_orbits,
        arrival_orbits=arrival_orbits,
        propagation_origin=OriginCodes.SUN,
    )

    # Verify that results are generated
    assert len(results) > 0, "Should generate some results with different time grids"

    # Verify all combinations have positive time of flight
    tof = results.time_of_flight()
    assert np.all(tof > 0), f"All time of flight should be positive, got: {tof}"

    # Verify C3 values are finite
    c3_values = results.c3_departure()
    assert np.all(np.isfinite(c3_values)), "All C3 values should be finite"

    # Verify that departure times are before arrival times
    dep_mjd = results.departure_time.mjd().to_numpy(zero_copy_only=False)
    arr_mjd = results.arrival_time.mjd().to_numpy(zero_copy_only=False)
    assert np.all(
        dep_mjd < arr_mjd
    ), "All departure times should be before arrival times"

    # Verify we get the expected number of combinations
    # With departure times [60000, 60005, 60010, 60015, 60020] and
    # arrival times [60030, 60040, 60050, 60060], we should get all 5*4=20 combinations
    # since all departure times are before all arrival times
    expected_combinations = len(departure_times) * len(arrival_times)
    assert (
        len(results) == expected_combinations
    ), f"Expected {expected_combinations} combinations, got {len(results)}"

    # Test plotting with the different time grids
    fig = plot_porkchop_plotly(
        results,
        title="Test: Different Time Grids",
        show_optimal=True,
    )

    # Verify figure was created successfully
    assert fig is not None, "Plot should be created successfully"


def test_porkchop_overlapping_time_grids():
    """
    Test case where departure and arrival time grids overlap but have different
    ranges and step sizes.
    """
    from adam_core.coordinates import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    # Departure times: every 5 days from day 60000 to 60020
    departure_times = Timestamp.from_mjd(
        [60000, 60005, 60010, 60015, 60020], scale="tdb"
    )
    departure_coords = CartesianCoordinates.from_kwargs(
        time=departure_times,
        x=[1.0, 1.1, 1.2, 1.3, 1.4],
        y=[0.0, 0.1, 0.2, 0.3, 0.4],
        z=[0.0, 0.01, 0.02, 0.03, 0.04],
        vx=[0.0, 0.01, 0.02, 0.03, 0.04],
        vy=[0.017, 0.016, 0.015, 0.014, 0.013],
        vz=[0.0, 0.001, 0.002, 0.003, 0.004],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(departure_times)),
    )

    # Arrival times: every 3 days from day 60010 to 60025 (overlapping!)
    arrival_times = Timestamp.from_mjd(
        [60010, 60013, 60016, 60019, 60022, 60025], scale="tdb"
    )
    arrival_coords = CartesianCoordinates.from_kwargs(
        time=arrival_times,
        x=[1.5, 1.55, 1.6, 1.65, 1.7, 1.75],
        y=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
        z=[0.05, 0.055, 0.06, 0.065, 0.07, 0.075],
        vx=[0.05, 0.055, 0.06, 0.065, 0.07, 0.075],
        vy=[0.010, 0.0095, 0.009, 0.0085, 0.008, 0.0075],
        vz=[0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(arrival_times)),
    )

    # Convert coordinates to orbits
    departure_orbits = Orbits.from_kwargs(
        orbit_id=[f"departure_{i}" for i in range(len(departure_times))],
        coordinates=departure_coords,
    )

    arrival_orbits = Orbits.from_kwargs(
        orbit_id=[f"arrival_{i}" for i in range(len(arrival_times))],
        coordinates=arrival_coords,
    )

    # Generate porkchop data
    results = generate_porkchop_data(
        departure_orbits=departure_orbits,
        arrival_orbits=arrival_orbits,
        propagation_origin=OriginCodes.SUN,
    )

    # With the old index-based filtering, this would incorrectly filter combinations
    # With the new time-based filtering, this should work correctly

    # Verify that results are generated
    assert len(results) > 0, "Should generate results with overlapping time grids"

    # Verify all combinations have positive time of flight
    tof = results.time_of_flight()
    assert np.all(tof > 0), "All time of flight should be positive"

    # Verify that departure times are before arrival times
    dep_mjd = results.departure_time.mjd().to_numpy(zero_copy_only=False)
    arr_mjd = results.arrival_time.mjd().to_numpy(zero_copy_only=False)
    assert np.all(
        dep_mjd < arr_mjd
    ), "All departure times should be before arrival times"

    # Count valid combinations manually to verify correctness
    dep_mjd_all = departure_times.mjd().to_numpy(zero_copy_only=False)
    arr_mjd_all = arrival_times.mjd().to_numpy(zero_copy_only=False)

    valid_count = 0
    for dep_time in dep_mjd_all:
        for arr_time in arr_mjd_all:
            if dep_time < arr_time:
                valid_count += 1

    assert (
        len(results) == valid_count
    ), f"Expected {valid_count} valid combinations, got {len(results)}"


def test_porkchop_problematic_case_that_old_version_would_fail():
    """
    Test a specific case that would have failed with the old index-based filtering
    but works correctly with the new time-based filtering.
    """
    from adam_core.coordinates import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    # Create a case where index-based filtering would give wrong results:
    # Departure times are LATER than arrival times
    departure_times = Timestamp.from_mjd(
        [60020, 60021, 60022], scale="tdb"
    )  # Later times
    departure_coords = CartesianCoordinates.from_kwargs(
        time=departure_times,
        x=[1.0, 1.1, 1.2],
        y=[0.0, 0.1, 0.2],
        z=[0.0, 0.01, 0.02],
        vx=[0.0, 0.01, 0.02],
        vy=[0.017, 0.016, 0.015],
        vz=[0.0, 0.001, 0.002],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(departure_times)),
    )

    # Arrival times are EARLIER than departure times
    arrival_times = Timestamp.from_mjd(
        [60000, 60001, 60002], scale="tdb"
    )  # Earlier times
    arrival_coords = CartesianCoordinates.from_kwargs(
        time=arrival_times,
        x=[1.5, 1.6, 1.7],
        y=[0.5, 0.6, 0.7],
        z=[0.05, 0.06, 0.07],
        vx=[0.05, 0.06, 0.07],
        vy=[0.010, 0.009, 0.008],
        vz=[0.005, 0.006, 0.007],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(arrival_times)),
    )

    # Convert coordinates to orbits
    departure_orbits = Orbits.from_kwargs(
        orbit_id=[f"departure_{i}" for i in range(len(departure_times))],
        coordinates=departure_coords,
    )

    arrival_orbits = Orbits.from_kwargs(
        orbit_id=[f"arrival_{i}" for i in range(len(arrival_times))],
        coordinates=arrival_coords,
    )

    # With the old index-based filtering, this would have incorrectly included
    # combinations where index(arrival) > index(departure), even though
    # time(arrival) < time(departure), which is physically invalid

    # With our new time-based filtering, this should return NO valid combinations
    # because all departure times are after all arrival times
    results = generate_porkchop_data(
        departure_orbits=departure_orbits,
        arrival_orbits=arrival_orbits,
        propagation_origin=OriginCodes.SUN,
    )

    # Should have zero valid combinations since departure is always after arrival
    assert (
        len(results) == 0
    ), f"Expected 0 valid combinations (departure after arrival), got {len(results)}"


def test_index_out_of_bounds_regression():
    """
    Regression test for index out of bounds error that occurred when data points
    fell outside the filtered unique time arrays.

    This test creates a scenario where:
    1. Some data points have invalid C3 values (filtered out)
    2. The invalid data points have times outside the valid time range
    3. Old implementation would try to use np.searchsorted with out-of-bounds results
    """
    from adam_core.coordinates import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    # Create data where the first few points have very early times with valid data
    # and the last few points have very late times with invalid data
    departure_times = Timestamp.from_mjd(
        [
            60000,
            60001,
            60002,  # Early times with valid data
            60100,
            60101,
            60102,
        ],  # Much later times with invalid data
        scale="tdb",
    )
    departure_coords = CartesianCoordinates.from_kwargs(
        time=departure_times,
        x=[1.0, 1.01, 1.02, 1.5, 1.51, 1.52],
        y=[0.0, 0.01, 0.02, 0.5, 0.51, 0.52],
        z=[0.0, 0.001, 0.002, 0.05, 0.051, 0.052],
        vx=[0.0, 0.001, 0.002, 0.05, 0.051, 0.052],
        vy=[0.017, 0.0169, 0.0168, 0.010, 0.0099, 0.0098],
        vz=[0.0, 0.0001, 0.0002, 0.005, 0.0051, 0.0052],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(departure_times)),
    )

    # Create arrival times that span a different range
    arrival_times = Timestamp.from_mjd(
        [
            60010,
            60011,
            60012,  # Early-ish times
            60110,
            60111,
            60112,
        ],  # Much later times
        scale="tdb",
    )
    arrival_coords = CartesianCoordinates.from_kwargs(
        time=arrival_times,
        x=[1.2, 1.21, 1.22, 1.7, 1.71, 1.72],
        y=[0.2, 0.21, 0.22, 0.7, 0.71, 0.72],
        z=[0.02, 0.021, 0.022, 0.07, 0.071, 0.072],
        vx=[0.02, 0.021, 0.022, 0.07, 0.071, 0.072],
        vy=[0.015, 0.0149, 0.0148, 0.008, 0.0079, 0.0078],
        vz=[0.002, 0.0021, 0.0022, 0.007, 0.0071, 0.0072],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(arrival_times)),
    )

    # Convert coordinates to orbits
    departure_orbits = Orbits.from_kwargs(
        orbit_id=[f"departure_{i}" for i in range(len(departure_times))],
        coordinates=departure_coords,
    )

    arrival_orbits = Orbits.from_kwargs(
        orbit_id=[f"arrival_{i}" for i in range(len(arrival_times))],
        coordinates=arrival_coords,
    )

    # Generate porkchop data - this will create Lambert solutions for all valid combinations
    results = generate_porkchop_data(
        departure_orbits=departure_orbits,
        arrival_orbits=arrival_orbits,
        propagation_origin=OriginCodes.SUN,
    )

    # Manually modify the results to simulate some very high C3 values
    # This simulates what would happen with difficult Lambert solutions
    c3_values_au_d2 = results.c3_departure()
    c3_values_km2_s2 = c3_values_au_d2 * (au_per_day_to_km_per_s(1.0) ** 2)

    # Create a plotting scenario that would trigger the old bug:
    # 1. Set c3_max to filter out some data but ensure it's valid
    # 2. The filtered unique arrays won't cover all data points
    c3_min_auto = np.nanpercentile(c3_values_km2_s2, 5)
    c3_max_auto = np.nanpercentile(c3_values_km2_s2, 95)
    c3_max_for_test = (
        c3_min_auto + (c3_max_auto - c3_min_auto) * 0.6
    )  # 60% of the range

    # This should work without errors in the new implementation
    # but would have failed with "index X is out of bounds" in the old implementation
    fig = plot_porkchop_plotly(
        results,
        title="Regression Test - Index Out of Bounds",
        c3_departure_min=c3_min_auto,
        c3_departure_max=c3_max_for_test,  # This will filter out high C3 data
        show_optimal=True,
    )

    # Verify the plot was created successfully
    assert fig is not None, "Plot should be created without index errors"

    # Verify we have some data in the plot (not everything was filtered out)
    assert len(fig.data) > 0, "Plot should contain some data traces"

    # If we get here, the regression test passed - no index out of bounds error occurred


def test_extreme_filtering_edge_case():
    """
    Test an extreme case where filtering removes almost all data except a tiny window.
    This tests the edge case handling when very few data points remain after filtering.
    """
    from adam_core.coordinates import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    # Create a large time span with data
    departure_times = Timestamp.from_mjd(
        np.arange(60000, 60100, 5), scale="tdb"
    )  # 20 times
    departure_coords = CartesianCoordinates.from_kwargs(
        time=departure_times,
        x=np.linspace(0.95, 1.05, len(departure_times)),
        y=np.linspace(-0.05, 0.05, len(departure_times)),
        z=np.linspace(-0.01, 0.01, len(departure_times)),
        vx=np.linspace(-0.01, 0.01, len(departure_times)),
        vy=np.linspace(0.015, 0.019, len(departure_times)),
        vz=np.linspace(-0.001, 0.001, len(departure_times)),
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(departure_times)),
    )

    arrival_times = Timestamp.from_mjd(
        np.arange(60050, 60150, 5), scale="tdb"
    )  # 20 times
    arrival_coords = CartesianCoordinates.from_kwargs(
        time=arrival_times,
        x=np.linspace(1.45, 1.55, len(arrival_times)),
        y=np.linspace(0.45, 0.55, len(arrival_times)),
        z=np.linspace(0.045, 0.055, len(arrival_times)),
        vx=np.linspace(0.045, 0.055, len(arrival_times)),
        vy=np.linspace(0.008, 0.012, len(arrival_times)),
        vz=np.linspace(0.004, 0.008, len(arrival_times)),
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(arrival_times)),
    )

    # Convert coordinates to orbits
    departure_orbits = Orbits.from_kwargs(
        orbit_id=[f"departure_{i}" for i in range(len(departure_times))],
        coordinates=departure_coords,
    )

    arrival_orbits = Orbits.from_kwargs(
        orbit_id=[f"arrival_{i}" for i in range(len(arrival_times))],
        coordinates=arrival_coords,
    )

    # Generate porkchop data
    results = generate_porkchop_data(
        departure_orbits=departure_orbits,
        arrival_orbits=arrival_orbits,
        propagation_origin=OriginCodes.SUN,
    )

    # Set very restrictive C3 limits to filter out most data
    c3_values_au_d2 = results.c3_departure()
    c3_values_km2_s2 = c3_values_au_d2 * (au_per_day_to_km_per_s(1.0) ** 2)
    c3_min_auto = np.nanpercentile(c3_values_km2_s2, 5)
    c3_max_auto = np.nanpercentile(c3_values_km2_s2, 95)
    very_low_c3_max = (
        c3_min_auto + (c3_max_auto - c3_min_auto) * 0.2
    )  # Keep only bottom 20%

    # This extreme filtering should still work without errors
    fig = plot_porkchop_plotly(
        results,
        title="Extreme Filtering Test",
        c3_departure_min=c3_min_auto,
        c3_departure_max=very_low_c3_max,  # Very restrictive filtering
        show_optimal=False,  # Disable optimal points to avoid issues with very few points
    )

    # Should create a plot even with extreme filtering
    assert fig is not None, "Plot should be created even with extreme filtering"


def test_generate_porkchop_data_mismatched_inputs():
    """
    Test that generate_porkchop_data fails with assertion errors when
    given orbits with mismatched frames or mixed/mismatched origins.
    """
    from adam_core.coordinates import CartesianCoordinates, Origin
    from adam_core.orbits import Orbits

    # Create base coordinates for testing
    departure_times = Timestamp.from_mjd([60000, 60001], scale="tdb")
    arrival_times = Timestamp.from_mjd([60050, 60051], scale="tdb")

    # Consistent departure coordinates (ecliptic, SUN origin)
    departure_coords = CartesianCoordinates.from_kwargs(
        time=departure_times,
        x=[1.0, 1.1],
        y=[0.0, 0.1],
        z=[0.0, 0.01],
        vx=[0.0, 0.01],
        vy=[0.017, 0.016],
        vz=[0.0, 0.001],
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
    )

    departure_orbits = Orbits.from_kwargs(
        orbit_id=["dep_0", "dep_1"],
        coordinates=departure_coords,
    )

    # --- Test mismatched frames ---
    arrival_coords_diff_frame = CartesianCoordinates.from_kwargs(
        time=arrival_times,
        x=[1.5, 1.6],
        y=[0.5, 0.6],
        z=[0.05, 0.06],
        vx=[0.05, 0.06],
        vy=[0.010, 0.009],
        vz=[0.005, 0.006],
        frame="equatorial",  # Different frame
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
    )

    arrival_orbits_diff_frame = Orbits.from_kwargs(
        orbit_id=["arr_0", "arr_1"],
        coordinates=arrival_coords_diff_frame,
    )

    with pytest.raises(
        AssertionError, match="Departure and arrival frames must be the same"
    ):
        generate_porkchop_data(
            departure_orbits=departure_orbits,
            arrival_orbits=arrival_orbits_diff_frame,
        )

    # Consistent arrival coordinates for further tests
    arrival_coords = CartesianCoordinates.from_kwargs(
        time=arrival_times,
        x=[1.5, 1.6],
        y=[0.5, 0.6],
        z=[0.05, 0.06],
        vx=[0.05, 0.06],
        vy=[0.010, 0.009],
        vz=[0.005, 0.006],
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
    )

    arrival_orbits = Orbits.from_kwargs(
        orbit_id=["arr_0", "arr_1"],
        coordinates=arrival_coords,
    )

    # --- Test mixed departure origins ---
    departure_coords_mixed_origin = departure_coords.set_column(
        "origin", Origin.from_kwargs(code=["SUN", "EARTH"])
    )
    departure_orbits_mixed_origin = Orbits.from_kwargs(
        orbit_id=["dep_0", "dep_1"],
        coordinates=departure_coords_mixed_origin,
    )
    with pytest.raises(AssertionError):
        generate_porkchop_data(
            departure_orbits=departure_orbits_mixed_origin,
            arrival_orbits=arrival_orbits,
        )

    # --- Test mixed arrival origins ---
    arrival_coords_mixed_origin = arrival_coords.set_column(
        "origin", Origin.from_kwargs(code=["SUN", "MARS_BARYCENTER"])
    )
    arrival_orbits_mixed_origin = Orbits.from_kwargs(
        orbit_id=["arr_0", "arr_1"],
        coordinates=arrival_coords_mixed_origin,
    )
    with pytest.raises(AssertionError):
        generate_porkchop_data(
            departure_orbits=departure_orbits,
            arrival_orbits=arrival_orbits_mixed_origin,
        )

    # --- Test mismatched (but consistent) origins ---
    arrival_coords_different_origin = arrival_coords.set_column(
        "origin", Origin.from_kwargs(code=["MARS_BARYCENTER", "MARS_BARYCENTER"])
    )
    arrival_orbits_different_origin = Orbits.from_kwargs(
        orbit_id=["arr_0", "arr_1"],
        coordinates=arrival_coords_different_origin,
    )
    with pytest.raises(
        AssertionError, match="Departure and arrival origins must be the same"
    ):
        generate_porkchop_data(
            departure_orbits=departure_orbits,
            arrival_orbits=arrival_orbits_different_origin,
        )


def test_lambert_solution_orbits():
    """
    Test the LambertSolutions.solution_departure_orbit and .solution_arrival_orbit methods.
    This test verifies that the methods correctly construct Orbits objects
    from Lambert solution data.
    """
    from adam_core.coordinates import CartesianCoordinates
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    # Create simple test coordinates for Lambert solution
    departure_times = Timestamp.from_mjd([60000, 60005, 60010], scale="tdb")
    departure_coords = CartesianCoordinates.from_kwargs(
        time=departure_times,
        x=[1.0, 1.1, 1.2],  # Earth-like orbit
        y=[0.0, 0.1, 0.2],
        z=[0.0, 0.01, 0.02],
        vx=[0.0, 0.01, 0.02],
        vy=[0.017, 0.016, 0.015],  # Earth orbital velocity
        vz=[0.0, 0.001, 0.002],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(departure_times)),
    )

    arrival_times = Timestamp.from_mjd([60050, 60055, 60060], scale="tdb")
    arrival_coords = CartesianCoordinates.from_kwargs(
        time=arrival_times,
        x=[1.5, 1.6, 1.7],  # Mars-like orbit
        y=[0.5, 0.6, 0.7],
        z=[0.05, 0.06, 0.07],
        vx=[0.05, 0.06, 0.07],
        vy=[0.010, 0.009, 0.008],  # Mars orbital velocity
        vz=[0.005, 0.006, 0.007],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(arrival_times)),
    )

    # Convert coordinates to orbits
    departure_orbits = Orbits.from_kwargs(
        orbit_id=[f"departure_{i}" for i in range(len(departure_times))],
        coordinates=departure_coords,
    )

    arrival_orbits = Orbits.from_kwargs(
        orbit_id=[f"arrival_{i}" for i in range(len(arrival_times))],
        coordinates=arrival_coords,
    )

    # Generate Lambert solutions
    lambert_results = generate_porkchop_data(
        departure_orbits=departure_orbits,
        arrival_orbits=arrival_orbits,
        propagation_origin=OriginCodes.SUN,
    )

    # Verify we have some results to test with
    assert len(lambert_results) > 0, "Should have Lambert solutions for testing"

    # Test departure orbit construction
    departure_orbits = lambert_results.solution_departure_orbit()

    # Verify the basic structure of the returned Orbits object
    assert isinstance(departure_orbits, Orbits), "Should return an Orbits object"
    assert len(departure_orbits) == len(
        lambert_results
    ), "Should have same number of orbits as Lambert results"

    # Verify orbit IDs follow the expected pattern
    expected_departure_ids = [
        f"solution_departure_orbit_{i}" for i in range(len(lambert_results))
    ]
    actual_departure_ids = departure_orbits.orbit_id.to_pylist()
    assert (
        actual_departure_ids == expected_departure_ids
    ), f"Expected orbit IDs {expected_departure_ids}, got {actual_departure_ids}"

    # Verify the coordinates match the departure state positions and Lambert velocities
    departure_coords_from_orbit = departure_orbits.coordinates

    # Check positions match departure state
    np.testing.assert_array_almost_equal(
        departure_coords_from_orbit.x.to_numpy(zero_copy_only=False),
        lambert_results.departure_body_x.to_numpy(zero_copy_only=False),
        decimal=10,
        err_msg="Departure orbit positions should match departure state positions",
    )

    # Check velocities match Lambert solution departure velocities (solution_departure_vx, vy, vz)
    np.testing.assert_array_almost_equal(
        departure_coords_from_orbit.vx.to_numpy(zero_copy_only=False),
        lambert_results.solution_departure_vx.to_numpy(zero_copy_only=False),
        decimal=10,
        err_msg="Departure orbit velocities should match Lambert solution v1",
    )

    # Check time and frame match
    assert departure_coords_from_orbit.time.equals(
        lambert_results.departure_time
    ), "Times should match"
    assert (
        departure_coords_from_orbit.frame == lambert_results.frame
    ), "Frames should match"
    assert departure_coords_from_orbit.origin.code.equals(
        lambert_results.origin.code
    ), "Origins should match"

    # Test arrival orbit construction
    arrival_orbits = lambert_results.solution_arrival_orbit()

    # Verify the basic structure
    assert isinstance(arrival_orbits, Orbits), "Should return an Orbits object"
    assert len(arrival_orbits) == len(
        lambert_results
    ), "Should have same number of orbits as Lambert results"

    # Verify orbit IDs follow the expected pattern
    expected_arrival_ids = [
        f"solution_arrival_orbit_{i}" for i in range(len(lambert_results))
    ]
    actual_arrival_ids = arrival_orbits.orbit_id.to_pylist()
    assert (
        actual_arrival_ids == expected_arrival_ids
    ), f"Expected orbit IDs {expected_arrival_ids}, got {actual_arrival_ids}"

    # Verify the coordinates match the arrival state positions and Lambert velocities
    arrival_coords_from_orbit = arrival_orbits.coordinates

    # Check positions match arrival state
    np.testing.assert_array_almost_equal(
        arrival_coords_from_orbit.x.to_numpy(zero_copy_only=False),
        lambert_results.arrival_body_x.to_numpy(zero_copy_only=False),
        decimal=10,
        err_msg="Arrival orbit positions should match arrival state positions",
    )

    # Check velocities match Lambert solution arrival velocities (solution_arrival_vx, vy, vz)
    np.testing.assert_array_almost_equal(
        arrival_coords_from_orbit.vx.to_numpy(zero_copy_only=False),
        lambert_results.solution_arrival_vx.to_numpy(zero_copy_only=False),
        decimal=10,
        err_msg="Arrival orbit velocities should match Lambert solution v2",
    )

    # Check time and frame match
    assert arrival_coords_from_orbit.time.equals(
        lambert_results.arrival_time
    ), "Times should match"
    assert (
        arrival_coords_from_orbit.frame == lambert_results.frame
    ), "Frames should match"
    assert arrival_coords_from_orbit.origin.code.equals(
        lambert_results.origin.code
    ), "Origins should match"


def test_lambert_solution_orbits_keplerian_consistency():
    """
    Test that orbits generated from departure and arrival states represent
    the same transfer trajectory by comparing their Keplerian elements.

    This test verifies that the Lambert solution creates a consistent transfer
    orbit regardless of whether we construct it from the departure or arrival state.
    """
    from adam_core.coordinates import (
        CartesianCoordinates,
        KeplerianCoordinates,
        transform_coordinates,
    )
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    # Create test coordinates with larger separations for more realistic Lambert solutions
    departure_times = Timestamp.from_mjd([60000, 60010], scale="tdb")
    departure_coords = CartesianCoordinates.from_kwargs(
        time=departure_times,
        x=[1.0, 1.05],  # Earth-like orbit variations
        y=[0.0, 0.1],
        z=[0.0, 0.01],
        vx=[0.0, 0.01],
        vy=[0.017, 0.016],  # Earth orbital velocity ~17 km/s
        vz=[0.0, 0.001],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(departure_times)),
    )

    # Arrival coordinates with sufficient time separation for realistic transfers
    arrival_times = Timestamp.from_mjd(
        [60200, 60250], scale="tdb"
    )  # ~200-250 days later
    arrival_coords = CartesianCoordinates.from_kwargs(
        time=arrival_times,
        x=[1.52, 1.55],  # Mars-like orbit variations
        y=[0.5, 0.6],
        z=[0.02, 0.03],
        vx=[0.02, 0.03],
        vy=[0.012, 0.011],  # Mars orbital velocity ~12 km/s
        vz=[0.001, 0.002],
        frame="ecliptic",
        origin=Origin.from_OriginCodes(OriginCodes.SUN, len(arrival_times)),
    )

    # Convert coordinates to orbits
    departure_orbits = Orbits.from_kwargs(
        orbit_id=[f"departure_{i}" for i in range(len(departure_times))],
        coordinates=departure_coords,
    )

    arrival_orbits = Orbits.from_kwargs(
        orbit_id=[f"arrival_{i}" for i in range(len(arrival_times))],
        coordinates=arrival_coords,
    )

    # Generate Lambert solutions
    lambert_results = generate_porkchop_data(
        departure_orbits=departure_orbits,
        arrival_orbits=arrival_orbits,
        propagation_origin=OriginCodes.SUN,
    )

    # Verify we have solutions to test
    assert (
        len(lambert_results) > 0
    ), "Should have Lambert solutions for Keplerian comparison"

    # Generate orbits from both departure and arrival states
    departure_orbits = lambert_results.solution_departure_orbit()
    arrival_orbits = lambert_results.solution_arrival_orbit()

    # Convert both to Keplerian coordinates
    departure_keplerian = transform_coordinates(
        departure_orbits.coordinates,
        KeplerianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    arrival_keplerian = transform_coordinates(
        arrival_orbits.coordinates,
        KeplerianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    # Compare orbital elements that should be identical to machine precision for the same transfer orbit
    # Semi-major axis should be identical (tolerance: ~1.5 mm for 1 AU orbit)
    departure_a = departure_keplerian.a.to_numpy(zero_copy_only=False)
    arrival_a = arrival_keplerian.a.to_numpy(zero_copy_only=False)

    np.testing.assert_allclose(
        departure_a,
        arrival_a,
        rtol=1e-14,
        atol=1e-16,
        err_msg="Semi-major axis should be identical for departure and arrival orbits",
    )

    # Eccentricity should be identical (dimensionless, expect machine precision)
    departure_e = departure_keplerian.e.to_numpy(zero_copy_only=False)
    arrival_e = arrival_keplerian.e.to_numpy(zero_copy_only=False)

    np.testing.assert_allclose(
        departure_e,
        arrival_e,
        rtol=1e-14,
        atol=1e-16,
        err_msg="Eccentricity should be identical for departure and arrival orbits",
    )

    # Inclination should be identical (tolerance: ~1e-12 degrees ≈ 3 microarcseconds)
    departure_i = departure_keplerian.i.to_numpy(zero_copy_only=False)
    arrival_i = arrival_keplerian.i.to_numpy(zero_copy_only=False)

    np.testing.assert_allclose(
        departure_i,
        arrival_i,
        rtol=1e-14,
        atol=1e-16,
        err_msg="Inclination should be identical for departure and arrival orbits",
    )

    # Longitude of ascending node should be nearly identical (normalize for 0-360 wraparound)
    departure_raan = departure_keplerian.raan.to_numpy(zero_copy_only=False)
    arrival_raan = arrival_keplerian.raan.to_numpy(zero_copy_only=False)

    # Convert to radians, normalize to [0, 2π], then back to degrees for comparison
    departure_raan_rad = np.deg2rad(departure_raan)
    arrival_raan_rad = np.deg2rad(arrival_raan)
    departure_raan_norm = np.mod(departure_raan_rad, 2 * np.pi)
    arrival_raan_norm = np.mod(arrival_raan_rad, 2 * np.pi)

    np.testing.assert_allclose(
        departure_raan_norm,
        arrival_raan_norm,
        rtol=1e-14,
        atol=1e-16,
        err_msg="RAAN should be identical for departure and arrival orbits",
    )

    # Argument of periapsis should be identical (normalize for 0-360 wraparound)
    departure_ap = departure_keplerian.ap.to_numpy(zero_copy_only=False)
    arrival_ap = arrival_keplerian.ap.to_numpy(zero_copy_only=False)

    # Convert to radians, normalize to [0, 2π] for machine precision comparison
    departure_ap_rad = np.deg2rad(departure_ap)
    arrival_ap_rad = np.deg2rad(arrival_ap)
    departure_ap_norm = np.mod(departure_ap_rad, 2 * np.pi)
    arrival_ap_norm = np.mod(arrival_ap_rad, 2 * np.pi)

    np.testing.assert_allclose(
        departure_ap_norm,
        arrival_ap_norm,
        rtol=1e-14,
        atol=1e-16,
        err_msg="Argument of periapsis should be identical for departure and arrival orbits",
    )

    # Mean anomaly will be different due to different times, but let's just verify
    # that both states represent the same orbital trajectory
    departure_keplerian.M.to_numpy(zero_copy_only=False)
    arrival_keplerian.M.to_numpy(zero_copy_only=False)

    # We don't need to check mean anomaly equality since they're at different times
    # along the same orbit. The fact that a, e, i, raan, and ap are consistent
    # is sufficient to verify the orbital trajectory is the same.

    print(
        f"Successfully verified Keplerian consistency for {len(lambert_results)} Lambert solutions"
    )
    print(
        f"  Semi-major axis range: {np.min(departure_a):.6f} - {np.max(departure_a):.6f} AU"
    )
    print(
        f"  Eccentricity range: {np.min(departure_e):.6f} - {np.max(departure_e):.6f}"
    )
    print(
        f"  Inclination range: {np.min(departure_i)*180/np.pi:.3f} - {np.max(departure_i)*180/np.pi:.3f} degrees"
    )
