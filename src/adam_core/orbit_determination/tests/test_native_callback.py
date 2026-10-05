"""Observer identity must survive a propagator reordering its ephemerides."""

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest
import quivr as qv

from adam_core.coordinates import CartesianCoordinates, Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbit_determination._native_callback import predict_spherical
from adam_core.orbit_determination.differential_correction import fit_least_squares
from adam_core.orbit_determination.tests.test_differential_correction import (
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)
from adam_core.orbits import Ephemeris, Orbits
from adam_core.orbits.arrow_bridge import observers_to_ipc, orbits_to_ipc
from adam_core.time import Timestamp


@pytest.fixture
def callback_inputs():
    # Unsorted epochs, simultaneous stations, epochs separated by just 1 ns,
    # and a repeated request for the same observer (rows 2 and 4).
    times = Timestamp.from_kwargs(
        days=[61001, 61000, 61000, 61000, 61000],
        nanos=[30, 100, 100, 101, 100],
        scale="utc",
    )
    observers = Observers.from_kwargs(
        code=["500", "W84", "500", "500", "500"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0, 3.0, 4.0, 3.0],
            y=np.zeros(5),
            z=np.zeros(5),
            vx=np.zeros(5),
            vy=np.zeros(5),
            vz=np.zeros(5),
            time=times,
            origin=Origin.from_kwargs(code=["SUN"] * 5),
            frame="ecliptic",
        ),
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["z", "a"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[10.0, 20.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            vx=[0.0, 0.0],
            vy=[0.0, 0.0],
            vz=[0.0, 0.0],
            time=times.take([0, 0]),
            origin=Origin.from_kwargs(code=["SUN"] * 2),
            frame="ecliptic",
        ),
    )
    return orbits, observers


class ReorderingPropagator:
    """Return distinguishable predictions in orbit/time/station order."""

    def __init__(self, *, single_candidate=False, output_scale="utc"):
        self.single_candidate = single_candidate
        self.output_scale = output_scale
        self.calls = 0

    def generate_ephemeris(self, orbits, observers, **kwargs):
        self.calls += 1
        if self.single_candidate:
            orbits = orbits.take([0])
        blocks = []
        for orbit in orbits:
            lon = orbit.coordinates.x[0].as_py() + observers.coordinates.x.to_numpy(
                zero_copy_only=False
            )
            blocks.append(
                Ephemeris.from_kwargs(
                    orbit_id=[orbit.orbit_id[0].as_py()] * len(observers),
                    coordinates=SphericalCoordinates.from_kwargs(
                        lon=lon,
                        lat=-lon,
                        time=observers.coordinates.time.rescale(self.output_scale),
                        origin=Origin.from_kwargs(code=observers.code),
                        frame="equatorial",
                    ),
                )
            )
        return qv.concatenate(blocks).sort_by(
            [
                "orbit_id",
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )


class MjdRoundingPropagator(ReorderingPropagator):
    """Model a backend that returns epochs stored as a single float MJD."""

    def generate_ephemeris(self, orbits, observers, **kwargs):
        ephemeris = super().generate_ephemeris(orbits, observers, **kwargs)
        times = ephemeris.coordinates.time
        return ephemeris.set_column(
            "coordinates.time", Timestamp.from_mjd(times.mjd(), scale=times.scale)
        )


def predict(propagator, orbits, observers):
    return predict_spherical(
        propagator,
        orbits_to_ipc(orbits),
        observers_to_ipc(observers),
        len(orbits),
        len(observers),
    )


@pytest.mark.parametrize("single_candidate", [False, True])
@pytest.mark.parametrize("output_scale", ["utc", "tai"])
def test_callback_restores_candidate_and_observer_order(
    callback_inputs, single_candidate, output_scale
):
    orbits, observers = callback_inputs
    propagator = ReorderingPropagator(
        single_candidate=single_candidate, output_scale=output_scale
    )
    result = predict(propagator, orbits, observers)
    expected_lon = np.array([11, 12, 13, 14, 13, 21, 22, 23, 24, 23])
    npt.assert_array_equal(result[:, 1], expected_lon)
    npt.assert_array_equal(result[:, 2], -expected_lon)
    assert result.shape == (10, 6)
    assert result.flags.c_contiguous
    assert propagator.calls == (3 if single_candidate else 1)


@pytest.mark.parametrize("single_candidate", [False, True])
@pytest.mark.parametrize("output_scale", ["utc", "tai"])
def test_callback_matches_float_mjd_epochs(
    callback_inputs, single_candidate, output_scale
):
    orbits, observers = callback_inputs
    # Remove the distinct requests just 1 ns apart; these are ambiguous after
    # float MJD rounding and are covered by the rejection test below.
    observers = observers.take([0, 1, 2, 4])
    result = predict(
        MjdRoundingPropagator(
            single_candidate=single_candidate, output_scale=output_scale
        ),
        orbits,
        observers,
    )
    npt.assert_array_equal(result[:, 1], [11, 12, 13, 13, 21, 22, 23, 23])


@pytest.mark.parametrize("single_candidate", [False, True])
def test_callback_rejects_float_mjd_epoch_collisions(callback_inputs, single_candidate):
    orbits, observers = callback_inputs
    # Even identical predictions cannot identify distinct requested epochs
    # that collapse to the same float MJD; do not assign them by position.
    observers = observers.set_column("coordinates.x", pa.array([1, 2, 3, 3, 3.0]))
    with pytest.raises(ValueError, match="ambiguous.*observer"):
        predict(
            MjdRoundingPropagator(single_candidate=single_candidate),
            orbits,
            observers,
        )


@pytest.mark.parametrize("single_candidate", [False, True])
@pytest.mark.parametrize("mismatch", ["station", "epoch", "duplicate"])
def test_callback_rejects_mismatched_observers(
    callback_inputs, single_candidate, mismatch
):
    class Mismatched(ReorderingPropagator):
        def generate_ephemeris(self, orbits, observers, **kwargs):
            ephemeris = super().generate_ephemeris(orbits, observers, **kwargs)
            if mismatch == "station":
                codes = ephemeris.coordinates.origin.code.to_pylist()
                codes[0] = "XXX"
                return ephemeris.set_column(
                    "coordinates.origin.code", pa.array(codes, type=pa.large_string())
                )
            if mismatch == "epoch":
                nanos = ephemeris.coordinates.time.nanos.to_pylist()
                nanos[0] += 5
                return ephemeris.set_column("coordinates.time.nanos", pa.array(nanos))
            # Keep the total row count while replacing an expected observer
            # with an extra copy of another observer.
            return ephemeris.take([0] + list(range(len(ephemeris) - 1)))

    with pytest.raises(ValueError, match="observer"):
        predict(Mismatched(single_candidate=single_candidate), *callback_inputs)


def test_callback_rejects_ambiguous_observer_states(callback_inputs):
    orbits, observers = callback_inputs
    observers = observers.set_column("coordinates.x", pa.array([1, 2, 3, 4, 5.0]))
    with pytest.raises(ValueError, match="ambiguous.*observer"):
        predict(ReorderingPropagator(), orbits, observers)


def test_callback_rejects_conflicting_duplicate_predictions(callback_inputs):
    class Conflicting(ReorderingPropagator):
        def generate_ephemeris(self, orbits, observers, **kwargs):
            ephemeris = super().generate_ephemeris(orbits, observers, **kwargs)
            # Sorted rows 0 and 1 describe the repeated observer request.
            lon = ephemeris.coordinates.lon.to_pylist()
            lon[1] += 1.0
            return ephemeris.set_column("coordinates.lon", pa.array(lon))

    with pytest.raises(ValueError, match="ambiguous.*observer"):
        predict(Conflicting(), *callback_inputs)


@pytest.mark.parametrize("jacobian", ["analytic", "central"])
def test_callback_fit_is_invariant_to_observation_order(jacobian):
    class TimeSortedTwoBody(TwoBodyPropagator):
        def generate_ephemeris(self, orbits, observers, **kwargs):
            return (
                super()
                .generate_ephemeris(orbits, observers, **kwargs)
                .sort_by(
                    ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"]
                )
            )

    observations = make_synthetic_observations()
    permutation = np.random.default_rng(123).permutation(len(observations))
    shuffled = observations.take(permutation)
    propagator = TimeSortedTwoBody()
    fitted, members = fit_least_squares(
        make_initial_guess(), observations, propagator, jacobian=jacobian
    )
    fitted_shuffled, members_shuffled = fit_least_squares(
        make_initial_guess(), shuffled, propagator, jacobian=jacobian
    )
    assert fitted.success[0].as_py() is True
    assert fitted_shuffled.success[0].as_py() is True
    assert members_shuffled.obs_id.to_pylist() == shuffled.id.to_pylist()
    npt.assert_allclose(
        fitted_shuffled.coordinates.values,
        fitted.coordinates.values,
        rtol=1e-7,
        atol=1e-10,
    )
    npt.assert_allclose(
        fitted_shuffled.coordinates.covariance.to_matrix(),
        fitted.coordinates.covariance.to_matrix(),
        rtol=1e-5,
        atol=1e-15,
    )
    npt.assert_allclose(
        members_shuffled.residuals.chi2.to_numpy(zero_copy_only=False),
        members.take(permutation).residuals.chi2.to_numpy(zero_copy_only=False),
        rtol=1e-4,
        atol=1e-7,
    )
    assert (
        members_shuffled.outlier.to_pylist()
        == members.take(permutation).outlier.to_pylist()
    )
