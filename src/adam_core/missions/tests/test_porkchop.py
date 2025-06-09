import numpy as np

from adam_core.coordinates.origin import OriginCodes
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

    # Get departure coordinates for Earth
    departure_coordinates = prepare_and_propagate_orbits(
        body=OriginCodes.EARTH,
        start_time=departure_start,
        end_time=departure_end,
        propagation_origin=OriginCodes.SUN,
        step_size=5.0,  # Larger step size for faster test
    )

    # Get arrival coordinates for Mars
    arrival_coordinates = prepare_and_propagate_orbits(
        body=OriginCodes.MARS_BARYCENTER,
        start_time=arrival_start,
        end_time=arrival_end,
        propagation_origin=OriginCodes.SUN,
        step_size=5.0,  # Larger step size for faster test
    )

    results_sun = generate_porkchop_data(
        departure_coordinates=departure_coordinates,
        arrival_coordinates=arrival_coordinates,
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

    # Get departure coordinates for Earth
    departure_coordinates = prepare_and_propagate_orbits(
        body=OriginCodes.EARTH,
        start_time=departure_start,
        end_time=departure_end,
        propagation_origin=OriginCodes.SUN,
        step_size=1.0,  # 1-day intervals for good resolution
    )

    # Get arrival coordinates for Mars
    arrival_coordinates = prepare_and_propagate_orbits(
        body=OriginCodes.MARS_BARYCENTER,
        start_time=arrival_start,
        end_time=arrival_end,
        propagation_origin=OriginCodes.SUN,
        step_size=1.0,  # 1-day intervals for good resolution
    )
    # Generate porkchop data with reasonable resolution
    results = generate_porkchop_data(
        departure_coordinates=departure_coordinates,
        arrival_coordinates=arrival_coordinates,
    )

    # Verify data was generated successfully
    assert len(results) > 0
    assert np.all(~np.isnan(results.c3_departure()))

    # Generate the plot using plotly for interactive visualization
    fig = plot_porkchop_plotly(
        results,
        title="Earth to Mars Transfer (2022-2024)",
        optimal_hover=False,
        trim_to_valid=True,
        date_buffer_days=3.0,  # Add 3 days of buffer around valid data
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
    coordinates have different time grids.
    """
    from adam_core.coordinates import CartesianCoordinates
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
    )

    # Generate porkchop data - this should work with different time grids
    results = generate_porkchop_data(
        departure_coordinates=departure_coords,
        arrival_coordinates=arrival_coords,
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
    dep_mjd = results.departure_state.time.mjd().to_numpy(zero_copy_only=False)
    arr_mjd = results.arrival_state.time.mjd().to_numpy(zero_copy_only=False)
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
        trim_to_valid=False,  # Don't trim so we can see the full grid
    )

    # Verify figure was created successfully
    assert fig is not None, "Plot should be created successfully"


def test_porkchop_overlapping_time_grids():
    """
    Test case where departure and arrival time grids overlap but have different
    ranges and step sizes.
    """
    from adam_core.coordinates import CartesianCoordinates
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
    )

    # Generate porkchop data
    results = generate_porkchop_data(
        departure_coordinates=departure_coords,
        arrival_coordinates=arrival_coords,
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
    dep_mjd = results.departure_state.time.mjd().to_numpy(zero_copy_only=False)
    arr_mjd = results.arrival_state.time.mjd().to_numpy(zero_copy_only=False)
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
    )

    # With the old index-based filtering, this would have incorrectly included
    # combinations where index(arrival) > index(departure), even though
    # time(arrival) < time(departure), which is physically invalid

    # With our new time-based filtering, this should return NO valid combinations
    # because all departure times are after all arrival times
    results = generate_porkchop_data(
        departure_coordinates=departure_coords,
        arrival_coordinates=arrival_coords,
        propagation_origin=OriginCodes.SUN,
    )

    # Should have zero valid combinations since departure is always after arrival
    assert (
        len(results) == 0
    ), f"Expected 0 valid combinations (departure after arrival), got {len(results)}"
