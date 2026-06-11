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
        assert any(
            key == impact_body or key.startswith(f"{impact_body} ")
            for key in propagated_variants
        )

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
            # If no Earth collision events in the test data, this is expected
            if "No Earth collision events found" in str(e):
                pytest.skip(
                    "No Earth collision events in test data, skipping risk corridor test"
                )
            else:
                raise


# ---------------------------------------------------------------------------
# Pure-numpy unit tests for the closest-approach window logic. These do not
# require adam_assist or a propagator.
# ---------------------------------------------------------------------------

from adam_core.coordinates import (  # noqa: E402
    CartesianCoordinates,
    Origin,
    SphericalCoordinates,
)
from adam_core.dynamics.plots import (  # noqa: E402
    _closest_event_time_window,
    prepare_propagated_variants,
)


def _make_collision_events(variant_ids, bodies, times_mjd, rhos, stopping):
    n = len(variant_ids)
    time = Timestamp.from_mjd(times_mjd, scale="tdb")
    return CollisionEvent.from_kwargs(
        orbit_id=["orbit"] * n,
        variant_id=variant_ids,
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0] * n,
            y=[0.0] * n,
            z=[0.0] * n,
            vx=[0.0] * n,
            vy=[0.0] * n,
            vz=[0.0] * n,
            time=time,
            origin=Origin.from_kwargs(code=["SUN"] * n),
            frame="ecliptic",
        ),
        condition_id=["cond"] * n,
        collision_object=Origin.from_kwargs(code=bodies),
        collision_coordinates=SphericalCoordinates.from_kwargs(
            rho=rhos,
            lon=[0.0] * n,
            lat=[0.0] * n,
            vrho=[0.0] * n,
            vlon=[0.0] * n,
            vlat=[0.0] * n,
            time=time,
            origin=Origin.from_kwargs(code=bodies),
            frame="ecliptic",
        ),
        stopping_condition=stopping,
    )


def test_closest_event_time_window_percentile_bracketing():
    n = 11
    times = [60000.0 + i for i in range(n)]
    events = _make_collision_events(
        variant_ids=[f"v{i}" for i in range(n)],
        bodies=["MOON"] * n,
        times_mjd=times,
        rhos=[1.0] * n,
        stopping=[False] * n,
    )

    start_mjd, end_mjd = _closest_event_time_window(
        events, window_percentiles=(10.0, 90.0), window_padding=0.0
    )

    np.testing.assert_allclose(start_mjd, 60001.0)
    np.testing.assert_allclose(end_mjd, 60009.0)


def test_closest_event_time_window_padding_is_minutes():
    events = _make_collision_events(
        variant_ids=["v0"],
        bodies=["MOON"],
        times_mjd=[60000.0],
        rhos=[1.0],
        stopping=[False],
    )

    start_mjd, end_mjd = _closest_event_time_window(
        events, window_percentiles=(0.0, 100.0), window_padding=30.0
    )

    np.testing.assert_allclose(start_mjd, 60000.0 - 30.0 / 60.0 / 24.0)
    np.testing.assert_allclose(end_mjd, 60000.0 + 30.0 / 60.0 / 24.0)


def test_closest_event_time_window_dedups_per_variant_by_min_rho():
    # v0 has two events; only its closest (smallest rho) should contribute.
    events = _make_collision_events(
        variant_ids=["v0", "v0", "v1"],
        bodies=["MOON"] * 3,
        times_mjd=[60005.0, 60020.0, 60020.0],
        rhos=[10.0, 5.0, 3.0],
        stopping=[False] * 3,
    )

    start_mjd, end_mjd = _closest_event_time_window(
        events, window_percentiles=(0.0, 100.0), window_padding=0.0
    )

    np.testing.assert_allclose(start_mjd, 60020.0)
    np.testing.assert_allclose(end_mjd, 60020.0)


def test_closest_event_time_window_default_focus_prefers_stopping_impacts(caplog):
    import logging

    events = _make_collision_events(
        variant_ids=["a", "b"],
        bodies=["EARTH", "MOON"],
        times_mjd=[60010.0, 60050.0],
        rhos=[1.0, 2.0],
        stopping=[True, False],
    )

    with caplog.at_level(logging.WARNING, logger="adam_core.dynamics.plots"):
        start_mjd, end_mjd = _closest_event_time_window(
            events, window_percentiles=(0.0, 100.0), window_padding=0.0
        )

    # The window centers on the EARTH stopping impact, not the MOON approach.
    np.testing.assert_allclose(start_mjd, 60010.0)
    np.testing.assert_allclose(end_mjd, 60010.0)
    assert any("focus_body" in record.message for record in caplog.records)


def test_closest_event_time_window_unknown_focus_body_raises():
    events = _make_collision_events(
        variant_ids=["a"],
        bodies=["EARTH"],
        times_mjd=[60010.0],
        rhos=[1.0],
        stopping=[True],
    )

    with pytest.raises(ValueError, match="focus_body"):
        _closest_event_time_window(events, focus_body="MOON")


def test_closest_event_time_window_falls_back_to_stopping_events():
    # The focus body has only stopping events; the window must fall back to
    # them rather than failing.
    events = _make_collision_events(
        variant_ids=["a", "b"],
        bodies=["EARTH", "EARTH"],
        times_mjd=[60010.0, 60012.0],
        rhos=[1.0, 1.0],
        stopping=[True, True],
    )

    start_mjd, end_mjd = _closest_event_time_window(
        events, focus_body="EARTH", window_percentiles=(0.0, 100.0), window_padding=0.0
    )

    np.testing.assert_allclose(start_mjd, 60010.0)
    np.testing.assert_allclose(end_mjd, 60012.0)


def test_closest_event_time_window_empty_events_raises():
    with pytest.raises(ValueError, match="No collision events"):
        _closest_event_time_window(CollisionEvent.empty())


def _make_propagated_variants(variant_ids, times_mjd):
    rows_variant = []
    rows_time = []
    for variant_id in variant_ids:
        for time in times_mjd:
            rows_variant.append(variant_id)
            rows_time.append(time)
    n = len(rows_variant)
    return Orbits.from_kwargs(
        orbit_id=rows_variant,
        object_id=["object"] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0] * n,
            y=[0.5] * n,
            z=[0.1] * n,
            vx=[0.0] * n,
            vy=[0.0] * n,
            vz=[0.0] * n,
            time=Timestamp.from_mjd(rows_time, scale="tdb"),
            origin=Origin.from_kwargs(code=["EARTH"] * n),
            frame="ecliptic",
        ),
    )


def test_prepare_propagated_variants_renders_each_variant_in_one_group():
    # v1 close-approaches the Moon AND impacts Earth (stopping); it must be
    # rendered only in the Earth Impacting group. v2 only approaches the Moon.
    variants = _make_propagated_variants(["v1", "v2"], [60000.0, 60002.0])
    events = _make_collision_events(
        variant_ids=["v1", "v1", "v2"],
        bodies=["EARTH", "MOON", "MOON"],
        times_mjd=[60001.0, 60000.0, 60000.0],
        rhos=[0.0001, 0.001, 0.001],
        stopping=[True, False, False],
    )

    prepared = prepare_propagated_variants(variants, events)

    assert set(prepared.keys()) == {
        "Non-Impacting",
        "EARTH Impacting",
        "MOON Close-Approaching",
    }
    assert set(prepared["EARTH Impacting"].orbit_id.to_pylist()) == {"v1"}
    assert set(prepared["MOON Close-Approaching"].orbit_id.to_pylist()) == {"v2"}
    assert len(prepared["Non-Impacting"]) == 0
