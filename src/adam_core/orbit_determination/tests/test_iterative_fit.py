from adam_core._rust.arrow import pa, pc

from ..algorithms.fitting import iterative_fit


def test_iterative_fit_no_propagator_needed_for_unit() -> None:
    """Smoke-test: iterative_fit signature accepts correct arguments (no propagator call)."""
    # Just verify the function is importable and has the expected signature
    import inspect

    sig = inspect.signature(iterative_fit)
    assert "rchi2_threshold" in sig.parameters
    assert "min_obs" in sig.parameters
    assert "contamination_percentage" in sig.parameters
    assert "min_arc_length" in sig.parameters


def test_iterative_fit_rejects_a_gross_outlier_through_the_capsule_route() -> None:
    """The Rust loop uses batched native propagation and rejects the outlier."""
    from .test_differential_correction import (
        TwoBodyPropagator,
        make_initial_guess,
        make_synthetic_observations,
    )

    observations = make_synthetic_observations()
    lat = observations.coordinates.lat.to_numpy(zero_copy_only=False).copy()
    lat[4] += 12.0 * 0.5 / 3600.0
    observations = observations.set_column("coordinates.lat", pa.array(lat))
    fitted_orbit, fitted_members = iterative_fit(
        make_initial_guess(),
        observations,
        TwoBodyPropagator(),
        rchi2_threshold=2.0,
        min_obs=6,
        contamination_percentage=20.0,
    )
    assert fitted_orbit.success[0].as_py() is True
    assert fitted_members.outlier.to_pylist()[4] is True
    assert pc.sum(fitted_members.outlier).as_py() == 1
    assert fitted_orbit.reduced_chi2[0].as_py() <= 2.0
