"""
The deprecated `LeastSquares` delegates to the native refinement stage,
`refine(..., rejection="none")`.
"""

import numpy as np
import pyarrow.compute as pc
import pytest

from ...coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
)
from ...coordinates.origin import Origin
from ...coordinates.residuals import Residuals, calculate_reduced_chi2
from ...observers import Observers
from ...orbit_determination.evaluate import OrbitDeterminationObservations
from ...orbits import Orbits
from ...time import Timestamp
from ..least_squares import LeastSquares
from ..od_orchestration import refine
from .test_differential_correction import (
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)


def test_least_squares_is_refine_without_rejection() -> None:
    guess = make_initial_guess()
    observations = make_synthetic_observations()
    with pytest.warns(DeprecationWarning, match="LeastSquares is deprecated"):
        fitter = LeastSquares(use_central_difference=False)
    debug_info: dict = {}
    legacy = fitter.least_squares(
        guess,
        observations,
        TwoBodyPropagator(),
        perturbation_initial_fraction=1e-3,
        max_iterations=20,
        debug_info=debug_info,
    )
    expected, _ = refine(
        guess,
        observations,
        TwoBodyPropagator(),
        rejection="none",
        fit_settings={"max_iterations": 20},
    )
    assert legacy is not None
    np.testing.assert_array_equal(
        legacy.coordinates.values, expected.coordinates.values
    )
    np.testing.assert_array_equal(
        legacy.coordinates.covariance.to_matrix(),
        expected.coordinates.covariance.to_matrix(),
    )
    assert debug_info["num_observations"] == len(observations)
    assert debug_info["success"]
    assert debug_info["reduced_chi2"] == expected.reduced_chi2[0].as_py()


@pytest.fixture
def real_data():
    # Subset of actual observers for "2023 CL3"
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
    obscodes = ["K63", "K63", "204", "204", "I93", "I93"]
    observers = Observers.from_codes(codes=obscodes, times=obstimes)

    # MPC orbit for the same object
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
                            1.63293252e-04,
                            2.95269724e-04,
                            5.70764235e-06,
                            2.06775246e-06,
                            2.51608186e-06,
                            2.01679472e-07,
                        ]
                    ]
                )
            ),
        ),
    )

    return observers, orbit


def test_least_squares_adjusts(real_data) -> None:
    ASSISTPropagator = pytest.importorskip("adam_assist").ASSISTPropagator
    observers, orbit = real_data

    # Make observations that fit the orbit perfectly
    propagator = ASSISTPropagator()
    perfect_ephem = propagator.generate_ephemeris(
        orbit, observers, covariance=True, chunk_size=1, max_processes=1
    )
    perfect_observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"obs{i}" for i in range(len(observers))],
        coordinates=perfect_ephem.coordinates,
        observers=observers,
    )

    with pytest.warns(DeprecationWarning, match="LeastSquares is deprecated"):
        fitter = LeastSquares(use_central_difference=True)

    # If we feed perfect observations to perfect orbit, there should be no change
    debug_info = {}
    unchanged_orbit = fitter.least_squares(
        orbit, perfect_observations, propagator, debug_info=debug_info
    )
    assert unchanged_orbit is not None
    assert debug_info["success"]
    np.testing.assert_allclose(
        unchanged_orbit.coordinates.values, orbit.coordinates.values, atol=1e-9
    )

    # Change the orbit a bit so that it still works as an initial guess, but doesn't fit observations perfectly.
    # This will take several iterations, with improvements
    original_cometary = orbit.coordinates.to_cometary()
    updated_cometary = original_cometary.set_column(
        "i", pc.multiply(original_cometary.i, 0.99)
    )
    guess_orbit = orbit.set_column("coordinates", updated_cometary.to_cartesian())
    debug_info = {}
    improved_orbit = fitter.least_squares(
        guess_orbit, perfect_observations, propagator, debug_info=debug_info
    )
    assert improved_orbit is not None

    assert debug_info["iterations"] >= 1
    guessed_ephem = propagator.generate_ephemeris(
        guess_orbit, observers, chunk_size=1, max_processes=1
    )
    guessed_rchi2 = calculate_reduced_chi2(
        Residuals.calculate(
            perfect_observations.coordinates, guessed_ephem.coordinates
        ),
        6,
    )
    improved_ephem = propagator.generate_ephemeris(
        improved_orbit, observers, chunk_size=1, max_processes=1
    )
    improved_rchi2 = calculate_reduced_chi2(
        Residuals.calculate(
            perfect_observations.coordinates, improved_ephem.coordinates
        ),
        6,
    )
    assert improved_rchi2 < guessed_rchi2

    # Check things don't blow up if we don't ask for debug info
    no_debug_orbit = fitter.least_squares(guess_orbit, perfect_observations, propagator)
    assert no_debug_orbit is not None


def test_least_squares_zero_base(real_data) -> None:
    ASSISTPropagator = pytest.importorskip("adam_assist").ASSISTPropagator
    observers, orbit = real_data

    # Set the last component of the orbit, which is already the smallest value, to 0.
    # This verifies that we don't end up with division by zero.
    zeroed_value = orbit.coordinates.set_column("vz", [0.0])
    zeroed_orbit = orbit.set_column("coordinates", zeroed_value)

    propagator = ASSISTPropagator()
    perfect_ephem = propagator.generate_ephemeris(
        orbit, observers, covariance=True, chunk_size=1, max_processes=1
    )
    perfect_observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"obs{i}" for i in range(len(observers))],
        coordinates=perfect_ephem.coordinates,
        observers=observers,
    )

    with pytest.warns(DeprecationWarning, match="LeastSquares is deprecated"):
        fitter = LeastSquares(use_central_difference=True)
    debug_info = {}
    improved_orbit = fitter.least_squares(
        zeroed_orbit, perfect_observations, propagator, debug_info=debug_info
    )
    assert improved_orbit is not None
    assert debug_info["iterations"] >= 1
    guessed_ephem = propagator.generate_ephemeris(
        zeroed_orbit, observers, chunk_size=1, max_processes=1
    )
    guessed_rchi2 = calculate_reduced_chi2(
        Residuals.calculate(
            perfect_observations.coordinates, guessed_ephem.coordinates
        ),
        6,
    )
    improved_ephem = propagator.generate_ephemeris(
        improved_orbit, observers, chunk_size=1, max_processes=1
    )
    improved_rchi2 = calculate_reduced_chi2(
        Residuals.calculate(
            perfect_observations.coordinates, improved_ephem.coordinates
        ),
        6,
    )
    assert improved_rchi2 < guessed_rchi2
