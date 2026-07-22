"""
Tests for the observatory_bias_model parameter on the OD entry points.

Every OD entry point accepts observatory_bias_model (default None = nothing
changed) and applies it to the observations once, at entry, before fitting.
These tests use spy models to prove invocation, ordering, and single
application without running real fits.
"""

import importlib
import inspect
from typing import Any, Optional, cast

import numpy as np
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.origin import Origin
from ...propagator.propagator import Propagator
from ...time import Timestamp
from ..evaluate import OrbitDeterminationObservations, evaluate_orbits
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..native_orbit_fitter import NativeOrbitFitter
from ..observation_uncertainty import ObservationUncertaintyModel
from ..orbit_fitter import OrbitFitter
from .test_observation_uncertainty import make_observations

# Direct module references for monkeypatching; the package __init__ shadows
# some submodule names with same-named functions (e.g. iod).
dc_module = importlib.import_module(
    "adam_core.orbit_determination.differential_correction"
)
iod_module = importlib.import_module("adam_core.orbit_determination.iod")
native_module = importlib.import_module(
    "adam_core.orbit_determination.native_orbit_fitter"
)
od_module = importlib.import_module("adam_core.orbit_determination.od")

INFLATION_FACTOR = 4.0

# Placeholders for propagator arguments that are never reached: the spy
# models prove the bias model is applied before any fitting machinery.
UNUSED_PROPAGATOR = cast(Propagator, None)
UNUSED_PROPAGATOR_CLASS = cast(type[Propagator], object)


class InflatingSpy(ObservationUncertaintyModel):
    """Counts applications and inflates the RA/Dec covariance block."""

    def __init__(self, factor: float = INFLATION_FACTOR) -> None:
        self.calls = 0
        self.factor = factor

    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        self.calls += 1
        covariances = observations.coordinates.covariance.to_matrix().copy()
        covariances[:, 1:3, 1:3] *= self.factor
        return observations.set_column(
            "coordinates.covariance", CoordinateCovariances.from_matrix(covariances)
        )


class _Applied(Exception):
    """Sentinel raised by RaisingSpy to prove the model was invoked."""


class RaisingSpy(ObservationUncertaintyModel):
    """Raises a sentinel on apply: proves invocation at entry, before fitting."""

    def apply(
        self, observations: OrbitDeterminationObservations
    ) -> OrbitDeterminationObservations:
        raise _Applied()


def make_fitted_orbit() -> FittedOrbits:
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.5],
        z=[0.1],
        vx=[-0.005],
        vy=[0.01],
        vz=[0.001],
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )
    return FittedOrbits.from_kwargs(
        orbit_id=["orbit_01"],
        coordinates=coordinates,
        arc_length=[1.0],
        num_obs=[3],
        chi2=[1.0],
        reduced_chi2=[0.5],
    )


class TestSignatures:
    @pytest.mark.parametrize(
        "function",
        [
            od_module.od,
            od_module.differential_correction,
            iod_module.iod,
            iod_module.initial_orbit_determination,
            dc_module.fit_least_squares,
            dc_module.iterative_fit,
            evaluate_orbits,
            OrbitFitter.full_od,
            NativeOrbitFitter.initial_fit,
            NativeOrbitFitter.refine_fit,
        ],
        ids=lambda f: f.__qualname__,
    )
    def test_entry_point_accepts_observatory_bias_model(self, function: Any) -> None:
        parameter = inspect.signature(function).parameters.get("observatory_bias_model")
        assert parameter is not None, (
            f"{function.__qualname__} is missing the observatory_bias_model"
            " parameter"
        )
        assert parameter.default is None


class TestFullOD:
    def test_applies_model_once_before_fitting(self) -> None:
        observations = make_observations(["500", "F51", "W84"], [0.0, 30.0, -45.0])
        base_var = observations.coordinates.covariance.to_matrix()[:, 1, 1].copy()
        spy = InflatingSpy()
        received: dict[str, np.ndarray] = {}

        class FakeFitter(OrbitFitter):
            def __getstate__(self) -> dict[str, Any]:
                return self.__dict__

            def __setstate__(self, state: dict[str, Any]) -> None:
                self.__dict__.update(state)

            def initial_fit(
                self,
                object_id: Any,
                observations: OrbitDeterminationObservations,
            ) -> tuple[FittedOrbits, FittedOrbitMembers]:
                received["initial_fit"] = (
                    observations.coordinates.covariance.to_matrix()[:, 1, 1]
                )
                return make_fitted_orbit(), FittedOrbitMembers.empty()

            def refine_fit(
                self,
                fitted_orbit: FittedOrbits,
                observations: OrbitDeterminationObservations,
                propagator: Any,
            ) -> tuple[FittedOrbits, FittedOrbitMembers]:
                received["refine_fit"] = (
                    observations.coordinates.covariance.to_matrix()[:, 1, 1]
                )
                return fitted_orbit, FittedOrbitMembers.empty()

        fitter = FakeFitter()
        fitter.full_od(
            "obj",
            observations,
            propagator=UNUSED_PROPAGATOR,
            observatory_bias_model=spy,
        )

        # Applied exactly once, and both stages saw the same inflated
        # observations (factor 4, not 16 = no double application).
        assert spy.calls == 1
        np.testing.assert_allclose(
            received["initial_fit"], INFLATION_FACTOR * base_var, rtol=1e-12
        )
        np.testing.assert_allclose(
            received["refine_fit"], INFLATION_FACTOR * base_var, rtol=1e-12
        )

    def test_default_none_leaves_observations_unchanged(self) -> None:
        observations = make_observations(["500"], [0.0])
        received: dict[str, OrbitDeterminationObservations] = {}

        class FakeFitter(OrbitFitter):
            def __getstate__(self) -> dict[str, Any]:
                return self.__dict__

            def __setstate__(self, state: dict[str, Any]) -> None:
                self.__dict__.update(state)

            def initial_fit(
                self,
                object_id: Any,
                observations: OrbitDeterminationObservations,
            ) -> tuple[FittedOrbits, FittedOrbitMembers]:
                received["observations"] = observations
                return FittedOrbits.empty(), FittedOrbitMembers.empty()

            def refine_fit(
                self,
                fitted_orbit: FittedOrbits,
                observations: OrbitDeterminationObservations,
                propagator: Any,
            ) -> tuple[FittedOrbits, FittedOrbitMembers]:
                return fitted_orbit, FittedOrbitMembers.empty()

        FakeFitter().full_od("obj", observations, propagator=UNUSED_PROPAGATOR)
        assert received["observations"] is observations


class TestIterativeFit:
    def test_applies_model_and_does_not_forward_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observations = make_observations(["500"] * 6, [0.0] * 6)
        base_var = observations.coordinates.covariance.to_matrix()[:, 1, 1].copy()
        spy = InflatingSpy()
        captured: dict[str, Any] = {}

        def fake_fit_least_squares(
            orbit: Any,
            observations: OrbitDeterminationObservations,
            propagator: Any,
            ignore: Optional[list[str]] = None,
            **kwargs: Any,
        ) -> Any:
            captured["var"] = observations.coordinates.covariance.to_matrix()[:, 1, 1]
            captured["kwargs"] = kwargs
            raise _Applied()

        monkeypatch.setattr(dc_module, "fit_least_squares", fake_fit_least_squares)

        orbit = make_fitted_orbit().to_orbits()
        with pytest.raises(_Applied):
            dc_module.iterative_fit(
                orbit, observations, UNUSED_PROPAGATOR, observatory_bias_model=spy
            )

        assert spy.calls == 1
        np.testing.assert_allclose(
            captured["var"], INFLATION_FACTOR * base_var, rtol=1e-12
        )
        assert "observatory_bias_model" not in captured["kwargs"]


class TestNativeOrbitFitter:
    def test_initial_fit_applies_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        observations = make_observations(["500"], [0.0])
        base_var = observations.coordinates.covariance.to_matrix()[:, 1, 1].copy()
        spy = InflatingSpy()
        captured: dict[str, np.ndarray] = {}

        def fake_iod(
            observations: OrbitDeterminationObservations, *args: Any, **kwargs: Any
        ) -> Any:
            captured["var"] = observations.coordinates.covariance.to_matrix()[:, 1, 1]
            raise _Applied()

        monkeypatch.setattr(native_module, "iod", fake_iod)

        fitter = NativeOrbitFitter(propagator_class=UNUSED_PROPAGATOR_CLASS)
        with pytest.raises(_Applied):
            fitter.initial_fit("obj", observations, observatory_bias_model=spy)

        assert spy.calls == 1
        np.testing.assert_allclose(
            captured["var"], INFLATION_FACTOR * base_var, rtol=1e-12
        )

    def test_refine_fit_applies_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        observations = make_observations(["500"], [0.0])
        base_var = observations.coordinates.covariance.to_matrix()[:, 1, 1].copy()
        spy = InflatingSpy()
        captured: dict[str, Any] = {}

        def fake_iterative_fit(
            orbit: Any,
            observations: OrbitDeterminationObservations,
            propagator: Any,
            **kwargs: Any,
        ) -> Any:
            captured["var"] = observations.coordinates.covariance.to_matrix()[:, 1, 1]
            captured["kwargs"] = kwargs
            raise _Applied()

        monkeypatch.setattr(native_module, "iterative_fit", fake_iterative_fit)

        fitter = NativeOrbitFitter(propagator_class=UNUSED_PROPAGATOR_CLASS)
        with pytest.raises(_Applied):
            fitter.refine_fit(
                make_fitted_orbit(),
                observations,
                UNUSED_PROPAGATOR,
                observatory_bias_model=spy,
            )

        assert spy.calls == 1
        np.testing.assert_allclose(
            captured["var"], INFLATION_FACTOR * base_var, rtol=1e-12
        )
        assert "observatory_bias_model" not in captured["kwargs"]


class TestAppliedAtEntry:
    """
    The heavier entry points are exercised with a raising spy: the sentinel
    firing proves the model is invoked at entry, before any fitting machinery
    (the propagator argument is a placeholder that would fail if reached).
    """

    def test_od(self) -> None:
        observations = make_observations(["500", "F51", "W84"], [0.0, 30.0, -45.0])
        with pytest.raises(_Applied):
            od_module.od(
                make_fitted_orbit(),
                observations,
                propagator=UNUSED_PROPAGATOR,
                observatory_bias_model=RaisingSpy(),
            )

    def test_iod(self) -> None:
        observations = make_observations(["500", "F51", "W84"], [0.0, 30.0, -45.0])
        with pytest.raises(_Applied):
            iod_module.iod(
                observations,
                propagator=UNUSED_PROPAGATOR,
                observatory_bias_model=RaisingSpy(),
            )

    def test_fit_least_squares(self) -> None:
        observations = make_observations(["500"], [0.0])
        with pytest.raises(_Applied):
            dc_module.fit_least_squares(
                make_fitted_orbit().to_orbits(),
                observations,
                propagator=UNUSED_PROPAGATOR,
                observatory_bias_model=RaisingSpy(),
            )

    def test_evaluate_orbits(self) -> None:
        observations = make_observations(["500"], [0.0])
        with pytest.raises(_Applied):
            evaluate_orbits(
                make_fitted_orbit(),
                observations,
                propagator=UNUSED_PROPAGATOR,
                observatory_bias_model=RaisingSpy(),
            )

    def test_initial_orbit_determination(self) -> None:
        observations = make_observations(["500"], [0.0])
        with pytest.raises(_Applied):
            iod_module.initial_orbit_determination(
                observations,
                FittedOrbitMembers.empty(),
                propagator=UNUSED_PROPAGATOR,
                observatory_bias_model=RaisingSpy(),
            )

    def test_differential_correction(self) -> None:
        observations = make_observations(["500"], [0.0])
        with pytest.warns(DeprecationWarning):
            with pytest.raises(_Applied):
                od_module.differential_correction(
                    FittedOrbits.empty(),
                    FittedOrbitMembers.empty(),
                    observations,
                    propagator=UNUSED_PROPAGATOR,
                    observatory_bias_model=RaisingSpy(),
                )
