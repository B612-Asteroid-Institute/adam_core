"""Numerical qualification of the focused native refinement route."""

import numpy as np
import pyarrow.compute as pc
import pytest

from ...coordinates import CartesianCoordinates, CoordinateCovariances
from ...coordinates.origin import Origin
from ...coordinates.residuals import Residuals, calculate_reduced_chi2
from ...observers import Observers
from ...orbits import Orbits
from ...time import Timestamp
from ..evaluate import OrbitDeterminationObservations
from ._focused import refine_qualified


def test_least_squares_ignores_non_angular_covariance_in_native_code() -> None:
    guess = make_initial_guess()
    observations = make_synthetic_observations()
    covariance = observations.coordinates.covariance.to_matrix()
    covariance[:, 0, 0] = 123.0
    covariance[:, 3:, 3:] = np.eye(3) * 456.0
    with_extra_covariance = observations.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(covariance)
    )
    angular = np.full_like(covariance, np.nan)
    angular[:, 1:3, 1:3] = covariance[:, 1:3, 1:3]
    angular_only = observations.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(angular)
    )

    with pytest.warns(DeprecationWarning, match="LeastSquares is deprecated"):
        fitter = LeastSquares(use_central_difference=False)
    actual = fitter.least_squares(
        guess, with_extra_covariance, TwoBodyPropagator(), max_iterations=20
    )
    expected, _ = refine(
        guess,
        angular_only,
        TwoBodyPropagator(),
        rejection="none",
        fit_settings={"max_iterations": 20},
    )
    assert actual is not None
    np.testing.assert_array_equal(
        actual.coordinates.values, expected.coordinates.values
    )
    np.testing.assert_array_equal(
        actual.coordinates.covariance.to_matrix(),
        expected.coordinates.covariance.to_matrix(),
    )


def test_least_squares_does_not_materialize_covariance_in_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guess = make_initial_guess()
    observations = make_synthetic_observations()
    with pytest.warns(DeprecationWarning, match="LeastSquares is deprecated"):
        fitter = LeastSquares(use_central_difference=False)

    def forbidden_to_matrix(self):
        raise AssertionError("deprecated least-squares must not materialize covariance")

    monkeypatch.setattr(CoordinateCovariances, "to_matrix", forbidden_to_matrix)
    fitted = fitter.least_squares(
        guess, observations, TwoBodyPropagator(), max_iterations=20
    )
    assert fitted is not None
    assert np.isfinite(fitted.coordinates.values).all()


@pytest.fixture
def real_data():
    # Subset of actual observers for "2023 CL3".
    obstimes = Timestamp.from_kwargs(
        days=[59989, 59989, 59989, 59989, 59990, 59990],
        nanos=[
            78339000000000,
            80234800000000,
            82933632000000,
            85858272000000,
            68911862000000,
            70200173000000,
        ],
        scale="utc",
    )
    observers = Observers.from_codes(
        codes=["K63", "K63", "204", "204", "I93", "I93"], times=obstimes
    )
    orbit = Orbits.from_kwargs(
        orbit_id=["test1"],
        object_id=["2023 CL3"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[-0.875655],
            y=[1.066652],
            z=[0.085413],
            vx=[-0.007748],
            vy=[-0.014017],
            vz=[0.00026],
            time=Timestamp.from_kwargs(days=[60800], nanos=[0]),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
            covariance=CoordinateCovariances.from_sigmas(
                np.array(
                    [
                        [
                            1.63293252e-4,
                            2.95269724e-4,
                            5.70764235e-6,
                            2.06775246e-6,
                            2.51608186e-6,
                            2.01679472e-7,
                        ]
                    ]
                )
            ),
        ),
    )
    return observers, orbit


def _rchi2(orbit, observations, propagator) -> float:
    ephemeris = propagator.generate_ephemeris(
        orbit, observations.observers, chunk_size=1, max_processes=1
    )
    return calculate_reduced_chi2(
        Residuals.calculate(observations.coordinates, ephemeris.coordinates), 6
    )


def test_refinement_adjusts_and_improves_chi2(real_data) -> None:
    ASSISTPropagator = pytest.importorskip("adam_assist").ASSISTPropagator
    observers, orbit = real_data
    propagator = ASSISTPropagator()
    perfect_ephem = propagator.generate_ephemeris(
        orbit, observers, covariance=True, chunk_size=1, max_processes=1
    )
    observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"obs{i}" for i in range(len(observers))],
        coordinates=perfect_ephem.coordinates,
        observers=observers,
    )
    cometary = orbit.coordinates.to_cometary()
    guess = orbit.set_column(
        "coordinates",
        cometary.set_column("i", pc.multiply(cometary.i, 0.99)).to_cartesian(),
    )
    fitted, members = refine_qualified(guess, observations, propagator)
    assert len(fitted) == 1
    assert len(members) == len(observations)
    assert _rchi2(fitted.to_orbits(), observations, propagator) < _rchi2(
        guess, observations, propagator
    )


def test_refinement_handles_zero_state_component(real_data) -> None:
    ASSISTPropagator = pytest.importorskip("adam_assist").ASSISTPropagator
    observers, orbit = real_data
    propagator = ASSISTPropagator()
    observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"obs{i}" for i in range(len(observers))],
        coordinates=propagator.generate_ephemeris(
            orbit, observers, covariance=True, chunk_size=1, max_processes=1
        ).coordinates,
        observers=observers,
    )
    zeroed = orbit.set_column("coordinates", orbit.coordinates.set_column("vz", [0.0]))
    fitted, _ = refine_qualified(zeroed, observations, propagator)
    assert len(fitted) == 1
    assert np.all(np.isfinite(fitted.coordinates.values))
    assert _rchi2(fitted.to_orbits(), observations, propagator) < _rchi2(
        zeroed, observations, propagator
    )
