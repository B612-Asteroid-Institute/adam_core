import os
import warnings

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest

from adam_core import _rust_native

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
from ...propagator.propagator import NativeBackend, Propagator
from ...time import Timestamp
from ..differential_correction import (
    _analytic_jacobian,
    _analytic_jacobian_terms,
    _observation_whitening_matrices,
    fit_least_squares,
    iterative_fit,
    residual_function,
)
from ..evaluate import (
    OrbitDeterminationObservations,
    OrbitDeterminationPhotometry,
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

    two_body_dynamics = True
    native_backend = NativeBackend(
        name="adam_core.two_body.external",
        version=_rust_native.two_body_propagator_semver(),
        work_units=frozenset({"propagator_capsule"}),
    )

    def propagator_capsule(self):
        from adam_core import _rust_native

        return _rust_native.two_body_propagator_capsule()

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
        # One propagation for every orbit, split per orbit with `take`: a
        # sliced table (``orbits[i : i + 1]``) keeps an Arrow offset that the
        # record-batch crossing ignores (see test_dynamics_table_slices.py).
        n = len(observers)
        propagated = propagate_2body(orbits, observers.coordinates.time)
        blocks = []
        for i in range(len(orbits)):
            block = propagated.take(list(range(i * n, (i + 1) * n)))
            blocks.append(generate_ephemeris_2body(block, observers))
        if len(blocks) == 1:
            return blocks[0]
        import quivr as qv

        return qv.concatenate(blocks)


class NativeTwoBodyPropagator(TwoBodyPropagator):
    """Compatibility spelling for the capsule-backed two-body test provider."""

    pass


def _central_difference_jacobian(
    state_vector, mjd_tdb, observations, propagator, rel_step=1e-6, abs_step_floor=1e-3
):
    """Central differences of `residual_function` (test cross-check for the
    autodiff Jacobian; the fit itself computes this in Rust)."""
    jacobian = np.empty((2 * len(observations), 6), dtype=np.float64)
    for k in range(6):
        step = rel_step * max(abs(state_vector[k]), abs_step_floor)
        plus = state_vector.copy()
        minus = state_vector.copy()
        plus[k] += step
        minus[k] -= step
        jacobian[:, k] = (
            residual_function(plus, mjd_tdb, observations, propagator)
            - residual_function(minus, mjd_tdb, observations, propagator)
        ) / (2.0 * step)
    return jacobian


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
def test_python_only_pyoorb_is_refused_for_native_od(pure_iod_orbit):
    orbit, _orbit_members, observations = pure_iod_orbit
    with pytest.raises(TypeError, match="no native orbit-determination backend"):
        fit_least_squares(orbit, observations, PYOORBPropagator())


# ---------------------------------------------------------------------------
# External propagation-provider integration.
# ---------------------------------------------------------------------------


def _assist_propagator():
    pytest.importorskip("adam_assist")
    from adam_assist import ASSISTPropagator

    from .._native_dispatch import native_work_unit

    propagator = ASSISTPropagator()
    if native_work_unit(propagator, "propagator_capsule") is None:
        pytest.skip("installed adam_assist has no propagation-provider capsule")
    return propagator


def test_fit_least_squares_uses_assist_capsule(pure_iod_orbit):
    orbit, _orbit_members, observations = pure_iod_orbit
    fitted_orbit, fitted_members = fit_least_squares(
        orbit.to_orbits(), observations, _assist_propagator(), jacobian="2-point"
    )
    assert len(fitted_orbit) == 1
    assert len(fitted_members) == len(observations)
    assert np.all(np.isfinite(fitted_orbit.coordinates.values))


class TestCapsuleWhitenedFitFacade:
    """Python validation around the single capsule-backed core fit driver."""

    def test_unsupported_solver_kwargs_raise(self):
        observations = make_synthetic_observations()
        propagator = TwoBodyPropagator()
        with pytest.raises(ValueError, match="x_scale=1.0"):
            fit_least_squares(
                make_initial_guess(), observations, propagator, x_scale=1.0
            )
        fitted, _ = fit_least_squares(
            make_initial_guess(), observations, propagator, ftol=1e-10
        )
        assert len(fitted) == 1


class TestCapsuleIterativeFitFacade:
    """Python validation around the capsule-backed worst-residual driver."""

    @staticmethod
    def _with_outlier(index: int, sigmas: float):
        observations = make_synthetic_observations()
        lat = observations.coordinates.lat.to_numpy(zero_copy_only=False).copy()
        lat[index] += sigmas * SIGMA_ARCSEC / 3600.0
        return observations.set_column("coordinates.lat", pa.array(lat))

    def test_unsupported_solver_kwargs_raise(self):
        observations = self._with_outlier(9, 12.0)
        with pytest.raises(ValueError, match="x_scale=1.0"):
            iterative_fit(
                make_initial_guess(),
                observations,
                TwoBodyPropagator(),
                x_scale=1.0,
            )

    def test_python_prediction_override_is_refused(self):
        class Broken(TwoBodyPropagator):
            def generate_ephemeris(self, orbits, observers, **kwargs):
                raise KeyError("must not be called")

        with pytest.raises(TypeError, match="not covered"):
            iterative_fit(make_initial_guess(), make_synthetic_observations(), Broken())
