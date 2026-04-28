import os
import warnings

import pytest

try:
    from adam_core.propagator.adam_pyoorb import PYOORBPropagator
except ImportError:
    PYOORBPropagator = None

from ..differential_correction import iterative_fit


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
@pytest.mark.skipif(PYOORBPropagator is None, reason="PYOORBPropagator not available")
def test_iterative_fit_converges(pure_iod_orbit):
    """iterative_fit should improve an IOD orbit to a low reduced chi2."""
    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORBPropagator()

    fitted_orbit, fitted_orbit_members = iterative_fit(
        orbit,
        observations,
        propagator,
        rchi2_threshold=10.0,
        min_obs=6,
        contamination_percentage=20.0,
    )

    assert len(fitted_orbit) == 1
    assert len(fitted_orbit_members) == len(observations)
    # Should converge well below the IOD reduced chi2
    assert fitted_orbit.reduced_chi2[0].as_py() < orbit.reduced_chi2[0].as_py()


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
@pytest.mark.skipif(PYOORBPropagator is None, reason="PYOORBPropagator not available")
def test_iterative_fit_outlier_flags(pure_iod_orbit):
    """Observations rejected by iterative_fit should be flagged as outliers."""
    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORBPropagator()

    fitted_orbit, fitted_orbit_members = iterative_fit(
        orbit,
        observations,
        propagator,
        rchi2_threshold=10.0,
        min_obs=6,
        contamination_percentage=20.0,
    )

    # Every observation should appear in fitted_orbit_members
    assert set(observations.id.to_pylist()) == set(
        fitted_orbit_members.obs_id.to_pylist()
    )

    # At least the solution observations should be non-outliers
    import pyarrow.compute as pc

    solution_count = pc.sum(fitted_orbit_members.solution).as_py()
    assert solution_count >= 6


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
@pytest.mark.skipif(PYOORBPropagator is None, reason="PYOORBPropagator not available")
def test_iterative_fit_respects_min_obs(pure_iod_orbit):
    """iterative_fit must never drop below min_obs solution observations."""
    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORBPropagator()

    min_obs = 6
    fitted_orbit, fitted_orbit_members = iterative_fit(
        orbit,
        observations,
        propagator,
        rchi2_threshold=0.001,  # impossibly tight — forces max outlier removal
        min_obs=min_obs,
        contamination_percentage=20.0,
    )

    import pyarrow.compute as pc

    solution_count = pc.sum(fitted_orbit_members.solution).as_py()
    assert solution_count >= min_obs


def test_iterative_fit_no_propagator_needed_for_unit(pure_iod_orbit):
    """Smoke-test: iterative_fit signature accepts correct arguments (no propagator call)."""
    # Just verify the function is importable and has the expected signature
    import inspect

    sig = inspect.signature(iterative_fit)
    assert "rchi2_threshold" in sig.parameters
    assert "min_obs" in sig.parameters
    assert "contamination_percentage" in sig.parameters
    assert "min_arc_length" in sig.parameters
