"""
The legacy finite-difference OD surfaces (`od.od`, `od.differential_correction`,
`LeastSquares`) are deprecated in favour of the Rust solver entry points and
say so on use, before doing any work.
"""

import pytest

from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..least_squares import LeastSquares
from ..od import differential_correction, od
from .test_differential_correction import (
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)


def test_least_squares_warns_on_construction() -> None:
    with pytest.warns(DeprecationWarning, match="LeastSquares is deprecated"):
        LeastSquares(use_central_difference=True)


def test_differential_correction_warns_before_doing_anything() -> None:
    with pytest.warns(
        DeprecationWarning, match="od.differential_correction is deprecated"
    ):
        orbits, members = differential_correction(
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            OrbitDeterminationObservations.empty(),
            TwoBodyPropagator,
        )
    assert len(orbits) == 0
    assert len(members) == 0


def test_od_warns_before_doing_anything() -> None:
    guess = make_initial_guess()
    fitted = FittedOrbits.from_kwargs(
        orbit_id=guess.orbit_id,
        object_id=guess.object_id,
        coordinates=guess.coordinates,
        arc_length=[0.0],
        num_obs=[0],
        chi2=[0.0],
        reduced_chi2=[0.0],
        iterations=[0],
        success=[False],
        status_code=[0],
    )
    # Fewer observations than the legacy loop's min_obs: nothing is fitted.
    observations = make_synthetic_observations(n_obs=4)
    with pytest.warns(DeprecationWarning, match="od.od is deprecated"):
        orbits, members = od(fitted, observations, TwoBodyPropagator)
    assert len(orbits) == 0
    assert len(members) == 0
