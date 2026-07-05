"""N-body least-squares OD parity: backend-generic Rust driver vs legacy scipy.

Bead personal-cmy.7. The Gauss-Newton driver lives in the permissive core
(`fit_orbit_least_squares_barycentric`, generic over the Propagator trait);
this GPL package only instantiates it with the ASSIST propagator, per the
packaging decision that GPL is confined to the adam-assist equivalent.

Parity policy mirrors the 2-body LSQ gate: bit parity is architecturally
impossible (Gauss-Newton + FD Jacobians vs scipy trust-region-reflective, plus
cross-libassist C builds), so the gate is converged-minimum parity. Measured
2026-07-05 on the noise-free fixture: Rust recovers truth to 1.25e-7 AU
(legacy scipy+ASSIST: 5.9e-7), rust-vs-legacy state agreement 7.1e-7 AU,
Rust 22x faster (0.08 s vs 1.80 s; one batched 7-candidate same-epoch
ephemeris crossing per iteration vs seven sequential Python ASSIST calls).
"""

import numpy as np
import pytest
from adam_assist import ASSISTPropagator as PythonASSISTPropagator
from adam_core.coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    SphericalCoordinates,
)
from adam_core.observers import Observers
from adam_core.orbit_determination import fit_least_squares
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from adam_assist_rust import ASSISTPropagator as RustASSISTPropagator

EPOCH_MJD = 60000.0
TRUTH_STATE = np.array([1.2, 0.1, 0.05, -0.002, 0.016, 0.001])


def _make_orbit(state: np.ndarray, orbit_id: str) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[orbit_id],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[state[0]],
            y=[state[1]],
            z=[state[2]],
            vx=[state[3]],
            vy=[state[4]],
            vz=[state[5]],
            time=Timestamp.from_mjd([EPOCH_MJD], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


@pytest.fixture(scope="module")
def od_problem():
    """Noise-free ASSIST astrometry of a truth orbit + a perturbed start."""
    n = 8
    times = Timestamp.from_mjd(
        [EPOCH_MJD + 2.0 + i * 3.0 for i in range(n)], scale="utc"
    )
    observers = Observers.from_code("X05", times)
    truth = _make_orbit(TRUTH_STATE, "truth")
    python_propagator = PythonASSISTPropagator()
    predicted = python_propagator.generate_ephemeris(
        truth,
        observers,
        covariance=False,
        max_processes=1,
        predict_magnitudes=False,
        predict_phase_angle=False,
    ).coordinates
    arcsec = (1.0 / 3600.0) ** 2
    cov = np.tile(np.diag([1.0, arcsec, arcsec, 1.0, 1.0, 1.0]), (n, 1, 1))
    observed = SphericalCoordinates.from_kwargs(
        rho=predicted.rho.to_numpy(zero_copy_only=False),
        lon=predicted.lon.to_numpy(zero_copy_only=False),
        lat=predicted.lat.to_numpy(zero_copy_only=False),
        vrho=predicted.vrho.to_numpy(zero_copy_only=False),
        vlon=predicted.vlon.to_numpy(zero_copy_only=False),
        vlat=predicted.vlat.to_numpy(zero_copy_only=False),
        time=predicted.time,
        origin=predicted.origin,
        frame=predicted.frame,
        covariance=CoordinateCovariances.from_matrix(cov),
    )
    observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"obs-{i}" for i in range(n)],
        coordinates=observed,
        observers=observers,
    )
    initial = _make_orbit(
        TRUTH_STATE + np.array([1e-3, -1e-3, 5e-4, 1e-5, -1e-5, 1e-5]), "fit"
    )
    return python_propagator, observations, initial


def test_rust_od_recovers_truth_and_matches_legacy(od_problem):
    python_propagator, observations, initial = od_problem

    fitted_rust, chi2, iterations, converged = RustASSISTPropagator().fit_least_squares(
        initial, observations
    )
    assert converged
    assert chi2 < 1e-3
    rust_state = fitted_rust.coordinates.values[0]
    # Truth recovery at the Gauss-Newton FD-Jacobian floor (measured 1.25e-7).
    np.testing.assert_allclose(rust_state, TRUTH_STATE, rtol=0, atol=1e-6)

    fitted_legacy, _members = fit_least_squares(
        initial, observations, python_propagator
    )
    assert fitted_legacy.success[0].as_py()
    legacy_state = fitted_legacy.coordinates.values[0]
    # Converged-minimum parity: both optimizers settle around the same minimum
    # within their respective floors (measured 7.1e-7; 7x margin).
    np.testing.assert_allclose(rust_state, legacy_state, rtol=0, atol=5e-6)

    # The inv(J^T J) covariance is finite and positive on the diagonal.
    rust_cov = fitted_rust.coordinates.covariance.to_matrix()[0]
    assert np.isfinite(rust_cov).all()
    assert (np.diag(rust_cov) > 0).all()


def test_public_fit_least_squares_dispatches_to_native(od_problem):
    """The canonical public entry point
    ``adam_core.orbit_determination.fit_least_squares`` dispatches to the
    Rust-native driver when given the Rust-backed propagator (bead
    personal-cmy.13.1.4), preserving the legacy (FittedOrbits,
    FittedOrbitMembers) contract."""
    _python_propagator, observations, initial = od_problem
    rust_propagator = RustASSISTPropagator()

    fitted_orbit, fitted_members = fit_least_squares(
        initial, observations, rust_propagator
    )
    assert fitted_orbit.success[0].as_py()
    assert len(fitted_members) == len(observations)
    assert fitted_members.solution.to_pylist().count(True) == len(observations)

    # Bit-identical to the direct native fit: proves the native path ran
    # (the scipy path would only agree to its optimizer floor, not exactly).
    direct, _chi2, _iters, _conv = rust_propagator.fit_least_squares(
        initial, observations
    )
    np.testing.assert_array_equal(
        fitted_orbit.coordinates.values[0], direct.coordinates.values[0]
    )
