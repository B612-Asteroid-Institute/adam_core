import numpy as np
from adam_assist import ASSISTPropagator

from ..differential_correction import fit_least_squares


def test_fit_least_squares_dispatches_to_fused_work_unit(pure_iod_orbit):
    """The public `fit_least_squares` runs the Gauss-Newton fit AND the final
    evaluation behind one Rust crossing when the propagator exposes the fused
    `fit_least_squares_evaluated` work unit: outputs are bit-identical to the
    direct native call."""
    orbit, _orbit_members, observations = pure_iod_orbit
    propagator = ASSISTPropagator()

    fitted_orbit, fitted_members = fit_least_squares(
        orbit.to_orbits(), observations, propagator
    )
    direct = propagator.fit_least_squares_evaluated(orbit.to_orbits(), observations)

    np.testing.assert_array_equal(
        fitted_orbit.coordinates.values[0], np.asarray(direct["state"])
    )
    np.testing.assert_array_equal(
        fitted_orbit.coordinates.covariance.to_matrix()[0],
        np.asarray(direct["covariance"]).reshape(6, 6),
    )
    assert fitted_orbit.chi2[0].as_py() == direct["chi2"]
    assert fitted_orbit.reduced_chi2[0].as_py() == direct["reduced_chi2"]
    assert fitted_orbit.arc_length[0].as_py() == direct["arc_length"]
    assert fitted_orbit.num_obs[0].as_py() == direct["num_obs"]
    assert fitted_orbit.iterations[0].as_py() == direct["iterations"]
    assert fitted_orbit.success[0].as_py() == direct["converged"]
    np.testing.assert_array_equal(
        fitted_members.residuals.to_array(), np.asarray(direct["residual_values"])
    )
    assert fitted_members.solution.to_pylist() == [True] * len(observations)
    assert fitted_members.outlier.to_pylist() == [False] * len(observations)


def test_fit_least_squares_fused_ignore_marks_outliers(pure_iod_orbit):
    orbit, _orbit_members, observations = pure_iod_orbit
    propagator = ASSISTPropagator()
    ignore = observations.id.to_pylist()[-2:]

    fitted_orbit, fitted_members = fit_least_squares(
        orbit.to_orbits(), observations, propagator, ignore=ignore
    )
    assert fitted_orbit.num_obs[0].as_py() == len(observations) - 2
    assert fitted_members.outlier.to_pylist() == [
        obs_id in ignore for obs_id in observations.id.to_pylist()
    ]
    assert fitted_members.solution.to_pylist() == [
        obs_id not in ignore for obs_id in observations.id.to_pylist()
    ]
