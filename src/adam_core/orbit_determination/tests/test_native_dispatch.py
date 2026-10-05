"""Fused OD must preserve custom model and fitter behavior."""

import numpy.testing as npt
import pytest

from adam_core.coordinates import CoordinateCovariances
from adam_core.orbit_determination.native_orbit_fitter import NativeOrbitFitter
from adam_core.orbit_determination.observation_uncertainty import (
    CompositeModel,
    IdentityModel,
    PerformanceWeightedModel,
)
from adam_core.orbit_determination.od_orchestration import run_od
from adam_core.orbit_determination.tests.test_differential_correction import (
    NativeTwoBodyPropagator,
    TwoBodyPropagator,
    make_synthetic_observations,
)
from adam_core.orbit_determination.tests.test_observation_uncertainty import (
    make_bias_table,
)


class RecordingNativePropagator(NativeTwoBodyPropagator):
    def __init__(self):
        self.run_calls = 0
        self.full_calls = 0

    def run_od(self, *args, **kwargs):
        self.run_calls += 1
        return super().run_od(*args, **kwargs)

    def full_od(self, *args, **kwargs):
        self.full_calls += 1
        return super().full_od(*args, **kwargs)


class ScaleSigmas(IdentityModel):
    factor = 2.0

    def apply(self, observations):
        self.calls = getattr(self, "calls", 0) + 1
        return observations.set_column(
            "coordinates.covariance",
            CoordinateCovariances.from_matrix(
                observations.coordinates.covariance.to_matrix() * self.factor**2
            ),
        )


class NativeScaleSigmas(ScaleSigmas):
    """Explicitly describe the inherited custom apply method for Rust."""

    def _native_specs(self):
        return PerformanceWeightedModel(
            make_bias_table([{"obs_code": "500", "chi2_per_obs": self.factor**2}])
        )._native_specs()


class ChangedNativeScaleSigmas(NativeScaleSigmas):
    def apply(self, observations):
        # This extra scaling is absent from the inherited native specification.
        return super().apply(super().apply(observations))


class ScalingComposite(CompositeModel):
    def apply(self, observations):
        return ScaleSigmas().apply(super().apply(observations))


class UnchangedIdentity(IdentityModel):
    pass


class UnchangedFitter(NativeOrbitFitter):
    pass


class CompatibleFitter(NativeOrbitFitter):
    def refine_fit(self, *args, **kwargs):
        return super().refine_fit(*args, **kwargs)

    def native_settings(self):
        # The custom hook is equivalent to the built-in refinement.
        return super().native_settings()


@pytest.fixture
def observations():
    return make_synthetic_observations()


def make_fitter(cls=NativeOrbitFitter, **kwargs):
    return cls(propagator_class=TwoBodyPropagator, iod_rchi2_threshold=1e6, **kwargs)


@pytest.mark.parametrize("backend", [TwoBodyPropagator, RecordingNativePropagator])
@pytest.mark.parametrize("form", ["direct", "sequence", "nested", "composite"])
def test_overridden_model_changes_used_uncertainties(observations, backend, form):
    model = ScaleSigmas()
    models = {
        "direct": model,
        "sequence": [IdentityModel(), model],
        "nested": CompositeModel(IdentityModel(), CompositeModel(model)),
        "composite": ScalingComposite(IdentityModel()),
    }[form]
    propagator = backend()
    fitted, members = run_od(observations, make_fitter(), models, propagator=propagator)
    assert fitted.success.to_pylist() == [True]
    for axis in ("lon", "lat"):
        npt.assert_allclose(
            getattr(members.used_astrometry, f"sigma_{axis}").to_numpy(),
            2.0 * getattr(members.original_astrometry, f"sigma_{axis}").to_numpy(),
        )
    if form != "composite":
        assert model.calls == 1
    if isinstance(propagator, RecordingNativePropagator):
        assert propagator.run_calls == 0
        # Only custom model application falls back; built-in fitting stays fused.
        assert propagator.full_calls == 1


@pytest.mark.parametrize(
    "model_class", [IdentityModel, UnchangedIdentity, NativeScaleSigmas]
)
def test_compatible_models_keep_fused_run_od(observations, model_class):
    model = model_class()
    propagator = RecordingNativePropagator()
    fitted, members = run_od(observations, make_fitter(), model, propagator=propagator)
    assert propagator.run_calls == 1
    assert fitted.success.to_pylist() == [True]
    npt.assert_allclose(
        members.used_astrometry.sigma_lon.to_numpy(),
        getattr(model, "factor", 1.0)
        * members.original_astrometry.sigma_lon.to_numpy(),
    )
    assert getattr(model, "calls", 0) == 0


def test_further_model_override_needs_new_spec(observations):
    propagator = RecordingNativePropagator()
    _, members = run_od(
        observations, make_fitter(), ChangedNativeScaleSigmas(), propagator=propagator
    )
    assert propagator.run_calls == 0
    npt.assert_allclose(
        members.used_astrometry.sigma_lon.to_numpy(),
        4.0 * members.original_astrometry.sigma_lon.to_numpy(),
    )


class CustomHookReached(Exception):
    pass


@pytest.mark.parametrize("hook", ["full_od", "initial_fit", "refine_fit"])
@pytest.mark.parametrize("entrypoint", ["run_od", "full_od"])
@pytest.mark.parametrize("override_kind", ["subclass", "instance", "further_subclass"])
def test_custom_fitter_hooks_are_called(
    observations, monkeypatch, hook, entrypoint, override_kind
):
    def custom_hook(*args, **kwargs):
        raise CustomHookReached(hook)

    if override_kind == "instance":
        fitter = make_fitter()
        monkeypatch.setattr(fitter, hook, custom_hook)
    else:
        parent = (
            CompatibleFitter
            if override_kind == "further_subclass"
            else NativeOrbitFitter
        )
        fitter = make_fitter(type("CustomFitter", (parent,), {hook: custom_hook}))
    propagator = RecordingNativePropagator()
    with pytest.raises(CustomHookReached, match=hook):
        if entrypoint == "run_od":
            run_od(observations, fitter, None, propagator=propagator)
        else:
            fitter.full_od("custom", observations, propagator)
    assert propagator.run_calls == 0
    assert propagator.full_calls == 0


@pytest.mark.parametrize(
    "fitter_class", [NativeOrbitFitter, UnchangedFitter, CompatibleFitter]
)
@pytest.mark.parametrize("entrypoint", ["run_od", "full_od"])
def test_compatible_fitters_keep_fused_paths(observations, fitter_class, entrypoint):
    fitter = make_fitter(fitter_class)
    propagator = RecordingNativePropagator()
    if entrypoint == "run_od":
        fitted, _ = run_od(observations, fitter, None, propagator=propagator)
        assert propagator.run_calls == 1
    else:
        fitted, _ = fitter.full_od("compatible", observations, propagator)
        assert propagator.full_calls == 1
    assert fitted.success.to_pylist() == [True]


def test_custom_fitter_runs_before_native_settings_validation(observations):
    class CustomFitter(NativeOrbitFitter):
        def full_od(self, *args, **kwargs):
            raise CustomHookReached("custom settings")

    fitter = make_fitter(CustomFitter, rejection_kwargs={"custom_option": True})
    with pytest.raises(CustomHookReached, match="custom settings"):
        run_od(observations, fitter, None, propagator=RecordingNativePropagator())


def test_instance_model_override_is_honored(observations, monkeypatch):
    model = IdentityModel()
    monkeypatch.setattr(model, "apply", ScaleSigmas().apply)
    propagator = RecordingNativePropagator()
    _, members = run_od(observations, make_fitter(), model, propagator=propagator)
    assert propagator.run_calls == 0
    npt.assert_allclose(
        members.used_astrometry.sigma_lon.to_numpy(),
        2.0 * members.original_astrometry.sigma_lon.to_numpy(),
    )
