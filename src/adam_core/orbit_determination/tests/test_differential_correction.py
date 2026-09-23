import os
import warnings

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest

try:
    from adam_core.propagator.adam_pyoorb import PYOORBPropagator
except ImportError:
    PYOORBPropagator = None

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.origin import Origin
from ...coordinates.residuals import Residuals
from ...coordinates.spherical import SphericalCoordinates
from ...dynamics.ephemeris import generate_ephemeris_2body
from ...dynamics.propagation import propagate_2body
from ...observers import Observers
from ...orbits.ephemeris import Ephemeris
from ...orbits.orbits import Orbits
from ...propagator.propagator import Propagator
from ...time import Timestamp
from ..differential_correction import (
    _analytic_jacobian,
    _analytic_jacobian_terms,
    _central_difference_jacobian,
    _observation_whitening_matrices,
    _validated_covariance,
    _weak_direction_delta_chi2,
    fit_least_squares,
    residual_function,
)
from ..evaluate import (
    OrbitDeterminationObservations,
    OrbitDeterminationPhotometry,
    evaluate_orbits,
)

TRUTH_STATE = np.array([0.9, 0.5, 0.05, -0.008, 0.012, 0.0005])
EPOCH_MJD_TDB = 61000.0
SIGMA_ARCSEC = 0.5


class TwoBodyPropagator(Propagator):
    """
    Kepler 2-body propagator over the Rust two-body kernels: the residual
    dynamics exactly match the model underlying the analytic Jacobian, so fits
    are exactly self-consistent. Ephemerides are emitted in (orbit_id, observer)
    blocks with each block in observer order, the pairing every OD entry point
    relies on.
    """

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def propagate_orbits(self, orbits: Orbits, times: Timestamp, **kwargs) -> Orbits:
        return propagate_2body(orbits, times)

    def generate_ephemeris(
        self, orbits: Orbits, observers: Observers, **kwargs
    ) -> Ephemeris:
        # Like every backend's ``generate_ephemeris(covariance=False)`` default,
        # emit no predicted covariance: the two-body kernels would otherwise
        # transport a fitted orbit's covariance into the ephemeris and
        # Residuals.calculate would add it to the observed covariance.
        orbits = orbits.sort_by(["orbit_id"]).set_column(
            "coordinates.covariance", CoordinateCovariances.nulls(len(orbits))
        )
        blocks = []
        for i in range(len(orbits)):
            propagated = propagate_2body(orbits[i : i + 1], observers.coordinates.time)
            blocks.append(generate_ephemeris_2body(propagated, observers))
        if len(blocks) == 1:
            return blocks[0]
        import quivr as qv

        return qv.concatenate(blocks)


def make_truth_orbit() -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=["truth"],
        object_id=["synthetic"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=TRUTH_STATE[0:1],
            y=TRUTH_STATE[1:2],
            z=TRUTH_STATE[2:3],
            vx=TRUTH_STATE[3:4],
            vy=TRUTH_STATE[4:5],
            vz=TRUTH_STATE[5:6],
            time=Timestamp.from_mjd([EPOCH_MJD_TDB], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


def make_synthetic_observations(
    n_obs: int = 25,
    sigma_arcsec: float = SIGMA_ARCSEC,
    correlation: float = 0.0,
    seed: int = 42,
) -> OrbitDeterminationObservations:
    """
    Generate noisy geocentric observations of the truth orbit with the 2-body
    pipeline and a known angular covariance.
    """
    rng = np.random.default_rng(seed)
    obs_mjd = np.sort(EPOCH_MJD_TDB + 5.0 + rng.uniform(0, 60, n_obs))
    times = Timestamp.from_mjd(obs_mjd, scale="tdb")
    observers = Observers.from_code("500", times)

    ephemeris = TwoBodyPropagator().generate_ephemeris(
        make_truth_orbit(), observers, max_processes=1
    )
    lon = ephemeris.coordinates.lon.to_numpy(zero_copy_only=False)
    lat = ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
    cos_lat = np.cos(np.radians(lat))

    sigma_deg = sigma_arcsec / 3600.0
    noise_sky = rng.multivariate_normal(
        [0.0, 0.0],
        sigma_deg**2 * np.array([[1.0, correlation], [correlation, 1.0]]),
        size=n_obs,
    )
    lon_obs = lon + noise_sky[:, 0] / cos_lat
    lat_obs = lat + noise_sky[:, 1]

    covariances = np.full((n_obs, 6, 6), np.nan)
    covariances[:, 1, 1] = sigma_deg**2 / cos_lat**2
    covariances[:, 2, 2] = sigma_deg**2
    if correlation != 0.0:
        covariances[:, 1, 2] = correlation * sigma_deg**2 / cos_lat
        covariances[:, 2, 1] = covariances[:, 1, 2]

    return OrbitDeterminationObservations.from_kwargs(
        id=[f"obs-{i:03d}" for i in range(n_obs)],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=lon_obs,
            lat=lat_obs,
            time=times,
            covariance=CoordinateCovariances.from_matrix(covariances),
            origin=Origin.from_kwargs(code=["500"] * n_obs),
            frame="equatorial",
        ),
        observers=observers,
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=[None] * n_obs, rmsmag=[None] * n_obs, band=[None] * n_obs
        ),
    )


def make_initial_guess() -> Orbits:
    x0 = TRUTH_STATE + np.array([1e-5, -8e-6, 5e-6, 1e-8, -1e-8, 5e-9])
    return Orbits.from_kwargs(
        orbit_id=["guess"],
        object_id=["synthetic"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=x0[0:1],
            y=x0[1:2],
            z=x0[2:3],
            vx=x0[3:4],
            vy=x0[4:5],
            vz=x0[5:6],
            time=Timestamp.from_mjd([EPOCH_MJD_TDB], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


class LinearPropagator:
    """
    Fake propagator whose predicted angles are an exact linear function of
    the state: pred_i(x) = obs_i + [A (x - x_star)]_i. The weighted
    least-squares solution is x_star exactly, with covariance
    inv(A^T Sigma^-1 A), computable by hand.

    Only duck-type compatible with residual_function / evaluate_orbits
    (implements generate_ephemeris only). All latitudes must be ~0 so that
    the cos(latitude) correction is the identity.
    """

    def __init__(
        self,
        observations: OrbitDeterminationObservations,
        design_matrix: np.ndarray,
        x_star: np.ndarray,
    ):
        self.lon_obs = observations.coordinates.lon.to_numpy(zero_copy_only=False)
        self.lat_obs = observations.coordinates.lat.to_numpy(zero_copy_only=False)
        self.times = observations.coordinates.time
        self.codes = observations.observers.code.to_pylist()
        self.design_matrix = design_matrix
        self.x_star = x_star

    def generate_ephemeris(self, orbits, observers, max_processes=1):
        state = orbits.coordinates.values[0]
        offsets = (self.design_matrix @ (state - self.x_star)).reshape(-1, 2)
        n = len(self.lon_obs)
        return Ephemeris.from_kwargs(
            orbit_id=[orbits.orbit_id[0].as_py()] * n,
            object_id=[orbits.object_id[0].as_py()] * n,
            coordinates=SphericalCoordinates.from_kwargs(
                lon=self.lon_obs + offsets[:, 0],
                lat=self.lat_obs + offsets[:, 1],
                time=self.times,
                origin=Origin.from_kwargs(code=self.codes),
                frame="equatorial",
            ),
        )


def make_linear_problem(n_obs: int = 12, seed: int = 7):
    """A linear least-squares problem with a hand-computable covariance."""
    rng = np.random.default_rng(seed)
    times = Timestamp.from_mjd(EPOCH_MJD_TDB + np.linspace(0, 30, n_obs), scale="tdb")
    sigma_lon_deg = rng.uniform(0.3, 1.0, n_obs) / 3600.0
    sigma_lat_deg = rng.uniform(0.3, 1.0, n_obs) / 3600.0
    covariances = np.full((n_obs, 6, 6), np.nan)
    covariances[:, 1, 1] = sigma_lon_deg**2
    covariances[:, 2, 2] = sigma_lat_deg**2

    observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"lin-{i:03d}" for i in range(n_obs)],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=10.0 + 0.01 * np.arange(n_obs),
            lat=np.zeros(n_obs),
            time=times,
            covariance=CoordinateCovariances.from_matrix(covariances),
            origin=Origin.from_kwargs(code=["500"] * n_obs),
            frame="equatorial",
        ),
        observers=Observers.from_code("500", times),
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=[None] * n_obs, rmsmag=[None] * n_obs, band=[None] * n_obs
        ),
    )

    x_star = TRUTH_STATE.copy()
    # Angular sensitivity ~arcsec-per-1e-6-au so the problem is well scaled
    design_matrix = rng.normal(0.0, 300.0, (2 * n_obs, 6))
    propagator = LinearPropagator(observations, design_matrix, x_star)

    sigma_flat = np.empty(2 * n_obs)
    sigma_flat[0::2] = sigma_lon_deg
    sigma_flat[1::2] = sigma_lat_deg
    whitened_design = design_matrix / sigma_flat[:, None]
    expected_covariance = np.linalg.inv(whitened_design.T @ whitened_design)
    return observations, propagator, x_star, expected_covariance


@pytest.fixture(scope="module")
def two_body_setup():
    observations = make_synthetic_observations()
    propagator = TwoBodyPropagator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted, members = fit_least_squares(
            make_initial_guess(), observations, propagator
        )
    return observations, propagator, fitted, members


class TestWhitenedResiduals:
    def test_sum_of_squares_matches_chi2(self):
        # The 2N whitened components must reproduce the Residuals.calculate
        # chi2 exactly, including with correlated (lon, lat) covariance.
        observations = make_synthetic_observations(n_obs=10, correlation=0.4)
        propagator = TwoBodyPropagator()
        residuals_whitened = residual_function(
            TRUTH_STATE, EPOCH_MJD_TDB, observations, propagator
        )
        assert residuals_whitened.shape == (20,)

        ephemeris = propagator.generate_ephemeris(
            make_truth_orbit(), observations.observers, max_processes=1
        )
        residuals = Residuals.calculate(observations.coordinates, ephemeris.coordinates)
        chi2_total = np.nansum(residuals.chi2.to_numpy(zero_copy_only=False))
        npt.assert_allclose(
            residuals_whitened @ residuals_whitened, chi2_total, rtol=1e-12
        )

        # Per-observation pairing: components 2i, 2i+1 belong to observation i
        chi2_pairs = (residuals_whitened.reshape(-1, 2) ** 2).sum(axis=1)
        npt.assert_allclose(
            chi2_pairs, residuals.chi2.to_numpy(zero_copy_only=False), rtol=1e-12
        )

    def test_non_finite_covariance_raises(self):
        observations = make_synthetic_observations(n_obs=6)
        covariances = observations.coordinates.covariance.to_matrix().copy()
        covariances[2, 1, 1] = np.nan
        observations = observations.set_column(
            "coordinates.covariance", CoordinateCovariances.from_matrix(covariances)
        )
        with pytest.raises(ValueError, match="non-finite"):
            _observation_whitening_matrices(observations)

    def test_non_positive_definite_covariance_raises(self):
        observations = make_synthetic_observations(n_obs=6)
        covariances = observations.coordinates.covariance.to_matrix().copy()
        covariances[3, 1, 2] = covariances[3, 2, 1] = 10.0  # |cov| >> sigmas
        observations = observations.set_column(
            "coordinates.covariance", CoordinateCovariances.from_matrix(covariances)
        )
        with pytest.raises(ValueError, match="positive-definite"):
            _observation_whitening_matrices(observations)


class TestSyntheticLinearCovariance:
    def test_covariance_matches_weighted_least_squares(self):
        # On an exactly linear problem the fitted covariance must equal the
        # hand-computed weighted least-squares covariance inv(A^T Sigma^-1 A),
        # and the weak-direction consistency check must pass silently.
        observations, propagator, x_star, expected = make_linear_problem()
        initial = make_initial_guess()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fitted, _ = fit_least_squares(
                initial, observations, propagator, jacobian="central"
            )
        state = np.concatenate([fitted.coordinates.r[0], fitted.coordinates.v[0]])
        npt.assert_allclose(state, x_star, atol=1e-12)
        npt.assert_allclose(
            fitted.coordinates.covariance.to_matrix()[0], expected, rtol=1e-6
        )

    def test_probe_measures_unity_on_quadratic_surface(self):
        observations, propagator, x_star, expected = make_linear_problem()
        residuals = residual_function(x_star, EPOCH_MJD_TDB, observations, propagator)
        delta_chi2 = _weak_direction_delta_chi2(
            x_star,
            EPOCH_MJD_TDB,
            observations,
            propagator,
            expected,
            float(residuals @ residuals),
        )
        npt.assert_allclose(delta_chi2, 1.0, rtol=1e-6)


class TestAnalyticJacobian:
    def test_matches_central_difference_on_2body_pipeline(self):
        # With 2-body residual dynamics the analytic (STM-chained) Jacobian
        # and central differences through the residual pipeline must agree.
        observations = make_synthetic_observations(n_obs=8)
        propagator = TwoBodyPropagator()
        terms = _analytic_jacobian_terms(observations)
        jac_analytic = _analytic_jacobian(TRUTH_STATE, EPOCH_MJD_TDB, terms)
        jac_central = _central_difference_jacobian(
            TRUTH_STATE, EPOCH_MJD_TDB, observations, propagator
        )
        assert jac_analytic.shape == (16, 6)
        scale = np.abs(jac_central).max()
        npt.assert_allclose(jac_analytic, jac_central, atol=1e-5 * scale)


class TestFitLeastSquares2Body:
    def test_fit_recovers_truth_within_uncertainty(self, two_body_setup):
        observations, propagator, fitted, members = two_body_setup
        assert fitted.success[0].as_py()
        state = np.concatenate([fitted.coordinates.r[0], fitted.coordinates.v[0]])
        covariance = fitted.coordinates.covariance.to_matrix()[0]
        error = state - TRUTH_STATE
        mahalanobis = np.sqrt(error @ np.linalg.solve(covariance, error))
        # 6-DOF: P(m > 4.6) ~ 1e-3; the seed is fixed so this is deterministic
        assert mahalanobis < 4.6

    def test_covariance_matches_central_difference(self, two_body_setup):
        observations, propagator, fitted, members = two_body_setup
        state = np.concatenate([fitted.coordinates.r[0], fitted.coordinates.v[0]])
        covariance = fitted.coordinates.covariance.to_matrix()[0]
        jac_central = _central_difference_jacobian(
            state, EPOCH_MJD_TDB, observations, propagator
        )
        covariance_central = np.linalg.inv(jac_central.T @ jac_central)
        sigma = np.sqrt(np.diag(covariance))
        sigma_central = np.sqrt(np.diag(covariance_central))
        npt.assert_allclose(sigma, sigma_central, rtol=1e-4)

    def test_chi2_displacement_scan_matches_weak_axis_sigma(self, two_body_setup):
        # Regression harness for the fabricated-confidence defect: the chi2
        # valley scale along the weakest direction must match the covariance's
        # claimed sigma, i.e. displacing by k sigma raises chi2 by ~k^2.
        # The broken inv(J^T J) covariance produced delta-chi2 ~ 1e-3 at
        # 1 claimed sigma on real arcs (curvature fabricated by FD noise).
        observations, propagator, fitted, members = two_body_setup
        state = np.concatenate([fitted.coordinates.r[0], fitted.coordinates.v[0]])
        covariance = fitted.coordinates.covariance.to_matrix()[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        sigma = np.sqrt(eigenvalues[-1])
        direction = eigenvectors[:, -1]

        residuals_0 = residual_function(state, EPOCH_MJD_TDB, observations, propagator)
        chi2_0 = residuals_0 @ residuals_0
        for k in (1.0, 2.0):
            chi2_displaced = []
            for sign in (+1.0, -1.0):
                displaced = state + sign * k * sigma * direction
                residuals_displaced = residual_function(
                    displaced, EPOCH_MJD_TDB, observations, propagator
                )
                chi2_displaced.append(residuals_displaced @ residuals_displaced)
            delta_chi2 = 0.5 * sum(chi2_displaced) - chi2_0
            npt.assert_allclose(delta_chi2, k**2, rtol=0.25)


class TestValidatedCovariance:
    def test_trustworthy_covariance_passes_silently(self):
        observations, propagator, x_star, expected = make_linear_problem()
        residuals = residual_function(x_star, EPOCH_MJD_TDB, observations, propagator)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = _validated_covariance(
                expected,
                "central",
                x_star,
                EPOCH_MJD_TDB,
                observations,
                propagator,
                float(residuals @ residuals),
            )
        npt.assert_array_equal(result, expected)

    def test_conditioning_warning_for_fd_covariance(self):
        # A covariance whose weak axis disagrees with the chi2 surface (here:
        # fabricated by construction, 100x too narrow) must trigger the
        # finite-difference conditioning warning in the legacy path.
        observations, propagator, x_star, expected = make_linear_problem()
        residuals = residual_function(x_star, EPOCH_MJD_TDB, observations, propagator)
        fabricated = expected / 100.0
        with pytest.warns(RuntimeWarning, match="noise floor"):
            result = _validated_covariance(
                fabricated,
                "2-point",
                x_star,
                EPOCH_MJD_TDB,
                observations,
                propagator,
                float(residuals @ residuals),
            )
        # Legacy path only warns; the covariance is returned unchanged
        npt.assert_array_equal(result, fabricated)

    def test_analytic_failure_falls_back_to_central_difference(self):
        # When the (nominally analytic) covariance fails the probe, the
        # analytic path recomputes it from central differences — here exact,
        # so the fallback recovers the true weighted least-squares covariance.
        observations, propagator, x_star, expected = make_linear_problem()
        residuals = residual_function(x_star, EPOCH_MJD_TDB, observations, propagator)
        fabricated = expected / 100.0
        with pytest.warns(RuntimeWarning, match="central-difference"):
            result = _validated_covariance(
                fabricated,
                "analytic",
                x_star,
                EPOCH_MJD_TDB,
                observations,
                propagator,
                float(residuals @ residuals),
            )
        npt.assert_allclose(result, expected, rtol=1e-6)


class TestFitLeastSquaresValidation:
    def test_rejects_non_angular_observations(self):
        observations = make_synthetic_observations(n_obs=6)
        values = observations.coordinates.rho.to_numpy(zero_copy_only=False).copy()
        values[0] = 1.0
        observations = observations.set_column("coordinates.rho", pa.array(values))
        with pytest.raises(ValueError, match="angular"):
            fit_least_squares(make_initial_guess(), observations, TwoBodyPropagator())

    def test_unknown_jacobian_raises(self):
        observations = make_synthetic_observations(n_obs=6)
        with pytest.raises(ValueError, match="jacobian"):
            fit_least_squares(
                make_initial_guess(),
                observations,
                TwoBodyPropagator(),
                jacobian="numerical",
            )


@pytest.mark.skipif(
    os.environ.get("OORB_DATA") is None, reason="OORB_DATA environment variable not set"
)
@pytest.mark.skipif(PYOORBPropagator is None, reason="PYOORBPropagator not available")
def test_fit_least_squares_pure_iod_orbit(pure_iod_orbit):
    # Test that fit_least_squares can fit and improve a pure orbit from an IOD
    # process using least squares

    orbit, orbit_members, observations = pure_iod_orbit
    propagator = PYOORBPropagator()

    fitted_orbit, fitted_orbit_members = fit_least_squares(
        orbit, observations, propagator
    )

    assert len(fitted_orbit) == 1
    assert len(fitted_orbit_members) == len(orbit_members) == len(observations)
    assert fitted_orbit.reduced_chi2[0].as_py() < (orbit.reduced_chi2[0].as_py() / 1e4)
    assert fitted_orbit.status_code[0].as_py() > 0
    assert fitted_orbit.iterations[0].as_py() <= 50


# ---------------------------------------------------------------------------
# Fused Rust work-unit dispatch (beads personal-cmy.7 / personal-dqk): reachable
# through the legacy forward-difference Jacobian with the linear loss on a
# propagator exposing ``fit_least_squares_evaluated`` (the Rust adam-assist
# backend). The default analytic-Jacobian path deliberately bypasses it.
# ---------------------------------------------------------------------------


def _assist_propagator():
    pytest.importorskip("adam_assist")
    from adam_assist import ASSISTPropagator

    propagator = ASSISTPropagator()
    if not hasattr(propagator, "fit_least_squares_evaluated"):
        pytest.skip(
            "installed adam_assist lacks the fused fit_least_squares_evaluated work unit"
        )
    return propagator


def test_fit_least_squares_dispatches_to_fused_work_unit(pure_iod_orbit):
    """With jacobian='2-point' the public `fit_least_squares` runs the
    Gauss-Newton fit AND the final evaluation behind one Rust crossing when
    the propagator exposes the fused `fit_least_squares_evaluated` work unit:
    outputs are bit-identical to the direct native call."""
    orbit, _orbit_members, observations = pure_iod_orbit
    propagator = _assist_propagator()

    fitted_orbit, fitted_members = fit_least_squares(
        orbit.to_orbits(), observations, propagator, jacobian="2-point"
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
    assert fitted_members.weight.to_pylist() == [1.0] * len(observations)


def test_fit_least_squares_fused_ignore_marks_outliers(pure_iod_orbit):
    orbit, _orbit_members, observations = pure_iod_orbit
    propagator = _assist_propagator()
    ignore = observations.id.to_pylist()[-2:]

    fitted_orbit, fitted_members = fit_least_squares(
        orbit.to_orbits(), observations, propagator, ignore=ignore, jacobian="2-point"
    )
    assert fitted_orbit.num_obs[0].as_py() == len(observations) - 2
    assert fitted_members.outlier.to_pylist() == [
        obs_id in ignore for obs_id in observations.id.to_pylist()
    ]
    assert fitted_members.solution.to_pylist() == [
        obs_id not in ignore for obs_id in observations.id.to_pylist()
    ]
    assert fitted_members.weight.to_pylist() == [
        0.0 if obs_id in ignore else 1.0 for obs_id in observations.id.to_pylist()
    ]


class FusedTwoBodyPropagator(TwoBodyPropagator):
    """
    `TwoBodyPropagator` posing as a Rust backend whose fused
    ``fit_least_squares_evaluated`` work unit returns a scripted state and
    covariance (the real work unit's forward-difference ``inv(J^T J)``), with
    the evaluation columns filled from `evaluate_orbits` at that state.
    """

    def __init__(self, state, covariance):
        self._state = np.asarray(state, dtype=np.float64)
        self._covariance = np.asarray(covariance, dtype=np.float64)

    def fit_least_squares_evaluated(self, orbit, observations, ignore_mask, **kwargs):
        fitted = Orbits.from_kwargs(
            orbit_id=orbit.orbit_id,
            object_id=orbit.object_id,
            coordinates=CartesianCoordinates.from_kwargs(
                x=self._state[0:1],
                y=self._state[1:2],
                z=self._state[2:3],
                vx=self._state[3:4],
                vy=self._state[4:5],
                vz=self._state[5:6],
                time=orbit.coordinates.time,
                origin=orbit.coordinates.origin,
                frame=orbit.coordinates.frame,
            ),
        )
        evaluated, members = evaluate_orbits(fitted, observations, self, parameters=6)
        return {
            "state": self._state,
            "covariance": self._covariance.reshape(36),
            "converged": True,
            "arc_length": evaluated.arc_length[0].as_py(),
            "num_obs": evaluated.num_obs[0].as_py(),
            "chi2": evaluated.chi2[0].as_py(),
            "reduced_chi2": evaluated.reduced_chi2[0].as_py(),
            "iterations": 3,
            "residual_values": members.residuals.to_array(),
            "residual_chi2": members.residuals.chi2.to_pylist(),
            "residual_dof": members.residuals.dof.to_pylist(),
            "residual_probability": members.residuals.probability.to_pylist(),
            "outlier": list(ignore_mask),
        }


def _consistency_warnings(records):
    return [
        r
        for r in records
        if issubclass(r.category, RuntimeWarning)
        and "weak-direction consistency check" in str(r.message)
    ]


def test_fused_path_validates_covariance_when_requested():
    """validate_covariance=True must probe the covariance handed back by a
    propagator's fused work unit and warn when the chi2 surface disagrees with
    it, as the scipy forward-difference path does; the fit itself is unchanged
    and validate_covariance=False keeps the path to its single native crossing."""
    observations = make_synthetic_observations()
    # Claims ~1e-10 au precision: chi2 is flat over many claimed sigma.
    fabricated = np.eye(6) * 1e-20
    propagator = FusedTwoBodyPropagator(TRUTH_STATE, fabricated)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        fitted_orbit, _members = fit_least_squares(
            make_initial_guess(), observations, propagator, jacobian="2-point"
        )
    assert len(_consistency_warnings(records)) == 1
    npt.assert_array_equal(
        fitted_orbit.coordinates.covariance.to_matrix()[0], fabricated
    )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        fit_least_squares(
            make_initial_guess(),
            observations,
            propagator,
            jacobian="2-point",
            validate_covariance=False,
        )
    assert _consistency_warnings(records) == []


def test_fused_path_accepts_consistent_covariance():
    """A native covariance consistent with the chi2 surface (here the exact
    2-body inv(J^T J) at the truth state) passes the probe silently."""
    observations = make_synthetic_observations()
    jacobian = _analytic_jacobian(
        TRUTH_STATE, EPOCH_MJD_TDB, _analytic_jacobian_terms(observations)
    )
    covariance = np.linalg.inv(jacobian.T @ jacobian)
    propagator = FusedTwoBodyPropagator(TRUTH_STATE, covariance)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        fit_least_squares(
            make_initial_guess(), observations, propagator, jacobian="2-point"
        )
    assert _consistency_warnings(records) == []
