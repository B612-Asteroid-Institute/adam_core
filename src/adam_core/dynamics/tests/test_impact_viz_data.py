from pathlib import Path

import numpy as np
import pytest

# Check if optional dependencies are available
try:
    from adam_assist import ASSISTPropagator

    HAS_ASSIST = True
except ImportError:
    HAS_ASSIST = False

try:
    import geopandas as gpd  # noqa: F401
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from adam_core.dynamics.impacts import CollisionEvent
from adam_core.dynamics.plots import (
    generate_impact_visualization_data,
    plot_impact_simulation,
    plot_risk_corridor,
)
from adam_core.orbits import Orbits, VariantOrbits
from adam_core.time import Timestamp

# Define test data directory - adjust as necessary
TEST_DATA_DIR = Path(__file__).parent / "data"


def load_test_data():
    """
    Load test data from parquet files.

    Returns
    -------
    tuple
        Orbit, VariantOrbits, and CollisionEvent objects loaded from parquet files.
    """
    # Check if test data files exist
    orbit_file = TEST_DATA_DIR / "orbits.parquet"
    variants_file = TEST_DATA_DIR / "variants.parquet"
    impacts_file = TEST_DATA_DIR / "collisions.parquet"

    # Check if all required test files exist
    files_exist = all(f.exists() for f in [orbit_file, variants_file, impacts_file])
    if not files_exist:
        pytest.skip("Test data files not found. Will be populated later by the user.")

    # Load the data from parquet files
    orbit = Orbits.from_parquet(orbit_file)
    variant_orbits = VariantOrbits.from_parquet(variants_file)
    impacts = CollisionEvent.from_parquet(impacts_file)

    return orbit, variant_orbits, impacts


@pytest.mark.skipif(not HAS_ASSIST, reason="ASSISTPropagator not available")
@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
@pytest.mark.parametrize(
    "time_step,time_range,max_processes",
    [
        (5, 60, None),  # Default values
        (10, 30, 1),  # Smaller range, single process
    ],
)
def test_generate_impact_visualization_data(time_step, time_range, max_processes):
    """
    Test the generate_impact_visualization_data function using data from parquet files.

    Parameters
    ----------
    time_step : float
        Time step for propagation in minutes.
    time_range : float
        Time range for propagation in minutes.
    max_processes : int or None
        Maximum number of processes to use for propagation.
    """
    # Load test data
    orbit, variant_orbits, impacts = load_test_data()

    # Create propagator
    propagator = ASSISTPropagator()

    # Call the function
    propagation_times, propagated_best_fit_orbit, propagated_variants = (
        generate_impact_visualization_data(
            orbit,
            variant_orbits,
            impacts,
            propagator,
            time_step=time_step,
            time_range=time_range,
            max_processes=max_processes,
        )
    )

    # Check types of returned objects
    assert isinstance(propagation_times, Timestamp)
    assert isinstance(propagated_best_fit_orbit, Orbits)
    assert isinstance(propagated_variants, dict)

    # Check that propagation times have reasonable length
    # based on time_step and time_range
    impact_times = impacts.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    time_span_days = (np.max(impact_times) - np.min(impact_times)) + (
        time_range / 60 / 24 * 2
    )
    expected_num_times = int(time_span_days * 24 * 60 / time_step) + 1
    # Allow for some flexibility in the number of time steps due to rounding
    assert len(propagation_times) >= expected_num_times * 0.8
    assert len(propagation_times) <= expected_num_times * 1.2

    assert "Non-Impacting" in propagated_variants
    for impact_body in impacts.collision_object.code.unique().to_pylist():
        assert impact_body in propagated_variants

    # Check that propagated orbits have the expected time points
    assert len(propagated_best_fit_orbit) == len(propagation_times) * len(orbit)


@pytest.mark.skipif(not HAS_ASSIST, reason="ASSISTPropagator not available")
@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
@pytest.mark.parametrize(
    "time_step,time_range",
    [
        (5, 60),  # Default values
        (10, 30),  # Different values
    ],
)
def test_plot_impact_simulation(time_step, time_range):
    """
    Test the plot_impact_simulation function using data from parquet files.

    Parameters
    ----------
    time_step : float
        Time step for propagation in minutes.
    time_range : float
        Time range for propagation in minutes.
    """
    # Load test data
    orbit, variant_orbits, impacts = load_test_data()

    # Create propagator
    propagator = ASSISTPropagator()

    # Generate data for visualization
    propagation_times, propagated_best_fit_orbit, propagated_variants = (
        generate_impact_visualization_data(
            orbit,
            variant_orbits,
            impacts,
            propagator,
            time_step=time_step,
            time_range=time_range,
            max_processes=1,
        )
    )

    # Test with different parameter combinations
    test_cases = [
        {
            "title": "Test Impact Simulation",
            "logo": True,
            "sample_impactors": 0.5,
            "sample_non_impactors": 0.5,
        },
        {
            "title": None,
            "logo": False,
            "show_impacting": False,
            "show_non_impacting": True,
        },
        {
            "title": "Minimal Test",
            "grid": False,
            "show_earth": True,
            "show_moon": False,
        },
    ]

    for params in test_cases:
        fig = plot_impact_simulation(
            propagation_times,
            propagated_best_fit_orbit,
            propagated_variants,
            impacts,
            **params,
        )

        # Verify the figure is created correctly
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) > 0

        # Check for expected elements based on params
        if params.get("show_earth", True):
            assert any(trace.name == "Earth" for trace in fig.data)

        if params.get("show_moon", True):
            assert any(trace.name == "Moon" for trace in fig.data)


@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
def test_plot_risk_corridor():
    """
    Test the plot_risk_corridor function using data from parquet files.
    """
    # Load test data
    _, _, impacts = load_test_data()

    # Test with different parameter combinations
    test_cases = [
        {"title": "Test Risk Corridor", "logo": True, "height": 600, "width": 800},
        {"title": None, "logo": False},
        {"title": "Custom Risk Corridor"},
    ]

    for params in test_cases:
        try:
            fig = plot_risk_corridor(impacts, **params)

            # Verify the figure is created correctly
            assert isinstance(fig, go.Figure)
            assert len(fig.frames) > 0

            # Check for expected title
            if params.get("title") is not None:
                assert params["title"] in fig.layout.title.text
            else:
                assert "Risk Corridor" in fig.layout.title.text

        except ValueError as e:
            # If no Earth impacts in the test data, this is expected
            if "No Earth impacts found" in str(e):
                pytest.skip(
                    "No Earth impacts in test data, skipping risk corridor test"
                )
            else:
                raise
