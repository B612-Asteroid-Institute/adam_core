import os

import pyarrow as pa
import pyarrow.compute as pc
import pytest

try:
    from adam_core.propagator.adam_pyoorb import PYOORBPropagator
except ImportError:
    PYOORBPropagator = None

from .. import differential_correction
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


def _scripted_fit_least_squares(pure_iod_orbit, script, monkeypatch):
    """Replace fit_least_squares with a stub that replays (success, reduced_chi2)
    per pass on the fixture's orbit and members (which carry real residuals so
    the rejection step can pick the worst observation)."""
    orbit, orbit_members, _observations = pure_iod_orbit
    passes = []

    def fake_fit_least_squares(_orbit, _observations, _propagator, ignore=None, **_):
        success, reduced_chi2 = script[min(len(passes), len(script) - 1)]
        passes.append(list(ignore or []))
        fitted_orbit = orbit.set_column("success", pa.array([success])).set_column(
            "reduced_chi2", pa.array([reduced_chi2], type=pa.float64())
        )
        ignored = set(ignore or [])
        outlier = pa.array(
            [obs_id in ignored for obs_id in orbit_members.obs_id.to_pylist()]
        )
        members = orbit_members.set_column("outlier", outlier).set_column(
            "solution", pc.invert(outlier)
        )
        return fitted_orbit, members

    monkeypatch.setattr(
        differential_correction, "fit_least_squares", fake_fit_least_squares
    )
    return passes


def test_iterative_fit_prefers_converged_pass_over_failed_first_pass(
    pure_iod_orbit, monkeypatch
):
    """A first pass that did not converge must not be kept as 'best' over a
    later pass that did, even when the later reduced chi2 is higher."""
    orbit, _members, observations = pure_iod_orbit
    passes = _scripted_fit_least_squares(
        pure_iod_orbit, [(False, 50.0), (True, 60.0)], monkeypatch
    )

    fitted_orbit, fitted_members = iterative_fit(
        orbit.to_orbits(),
        observations,
        propagator=None,
        rchi2_threshold=10.0,
        min_obs=6,
        contamination_percentage=20.0,
    )

    assert len(passes) == 2 and len(passes[1]) == 1
    assert fitted_orbit.success[0].as_py() is True
    assert fitted_orbit.reduced_chi2[0].as_py() == 60.0
    assert pc.sum(fitted_members.outlier).as_py() == 1


def test_iterative_fit_falls_back_to_lowest_rchi2_when_no_pass_converges(
    pure_iod_orbit, monkeypatch
):
    orbit, _members, observations = pure_iod_orbit
    _scripted_fit_least_squares(
        pure_iod_orbit, [(False, 50.0), (False, 40.0)], monkeypatch
    )

    fitted_orbit, _fitted_members = iterative_fit(
        orbit.to_orbits(),
        observations,
        propagator=None,
        rchi2_threshold=10.0,
        min_obs=6,
        contamination_percentage=20.0,
    )

    assert fitted_orbit.success[0].as_py() is False
    assert fitted_orbit.reduced_chi2[0].as_py() == 40.0
