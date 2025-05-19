import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from adam_core.coordinates.origin import OriginCodes
from adam_core.time import Timestamp

from ..porkchop import generate_porkchop_data, plot_porkchop, plot_porkchop_plotly


def test_generate_porkchop_data_origins():
    # Test with different origins
    earliest_launch = Timestamp.from_mjd([60000], scale="tdb")
    maximum_arrival = Timestamp.from_mjd([60100], scale="tdb")
    
    # Test with Sun as origin
    results_sun = generate_porkchop_data(
        departure_body=OriginCodes.EARTH,
        arrival_body=OriginCodes.MARS_BARYCENTER,
        earliest_launch_time=earliest_launch,
        maximum_arrival_time=maximum_arrival,
        propagation_origin=OriginCodes.SUN,
        step_size=5.0,  # Larger step size for faster test
    )
    
    
    # Verify that results are generated
    assert len(results_sun) > 0
    
    # Check that origins match what we specified
    assert results_sun.origin.as_OriginCodes() == OriginCodes.SUN

    # Check that time of flight is valid (positive)
    assert np.all(results_sun.time_of_flight() > 0)
    
    # Check that C3 values are computed
    assert np.all(~np.isnan(results_sun.c3()))


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
    earliest_launch = Timestamp.from_iso8601(["2022-01-01T00:00:00"], scale="tdb")
    maximum_arrival = Timestamp.from_iso8601(["2024-01-01T00:00:00"], scale="tdb")
    
    # Generate porkchop data with reasonable resolution
    results = generate_porkchop_data(
        departure_body=OriginCodes.EARTH,
        arrival_body=OriginCodes.MARS_BARYCENTER,
        earliest_launch_time=earliest_launch,
        maximum_arrival_time=maximum_arrival,
        propagation_origin=OriginCodes.SUN,
        step_size=1.0,  # 1-day intervals for good resolution
    )
    
    # Verify data was generated successfully
    assert len(results) > 0
    assert np.all(~np.isnan(results.c3()))
    
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
        print("To save PNG images, install the kaleido package with: pip install -U kaleido")
    
    # Print the path to the HTML file
    print(f"\nPorkchop plot HTML saved to: {html_path}")
    
    # return fig  # Return the figure for further analysis if needed


