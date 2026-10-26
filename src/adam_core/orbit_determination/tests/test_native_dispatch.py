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
    # Overriding work units voids the inherited declaration; recording calls
    # leaves the dynamics unchanged, so it is declared again.
    native_backend = NativeTwoBodyPropagator.native_backend

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
    return cls(TwoBodyPropagator(), iod_rchi2_threshold=1e6, **kwargs)


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


class ChangedDynamics(NativeTwoBodyPropagator):
    """Overrides propagation but inherits the work units and their
    declaration: the compiled work units would run the parent's dynamics."""

    def propagate_orbits(self, orbits, times, **kwargs):
        return super().propagate_orbits(orbits, times, **kwargs)


class UndeclaredMethods(TwoBodyPropagator):
    """Has a ``run_od`` method but declares no native backend."""

    def run_od(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("an undeclared work unit was dispatched")


def test_dispatch_requires_a_declared_backend():
    from adam_core.orbit_determination._native_dispatch import (
        native_backend,
        native_work_unit,
    )

    base = NativeTwoBodyPropagator()
    assert native_backend(base) is NativeTwoBodyPropagator.native_backend
    assert native_work_unit(base, "run_od") is not None
    # Not declared: a method of that name is not enough.
    assert native_work_unit(base, "fit_least_squares_evaluated") is None
    assert native_work_unit(UndeclaredMethods(), "run_od") is None
    # Overridden dynamics void the inherited declaration ...
    assert native_backend(ChangedDynamics()) is None
    assert native_work_unit(ChangedDynamics(), "run_od") is None
    # ... and so does an instance-level replacement of a work unit.
    patched = NativeTwoBodyPropagator()
    patched.run_od = lambda *args, **kwargs: None
    assert native_backend(patched) is None
    # A subclass that declares again is covered.
    assert native_work_unit(RecordingNativePropagator(), "run_od") is not None


def test_overridden_dynamics_run_the_composed_path():
    """run_od on a subclass with its own dynamics never enters the parent's
    compiled work unit, and still returns an orbit."""
    observations = make_synthetic_observations()
    calls = []

    class Spy(ChangedDynamics):
        def run_od(self, *args, **kwargs):  # pragma: no cover - must not run
            calls.append("run_od")
            return super().run_od(*args, **kwargs)

        def full_od(self, *args, **kwargs):  # pragma: no cover - must not run
            calls.append("full_od")
            return super().full_od(*args, **kwargs)

    propagator = Spy()
    orbits, _ = run_od(
        observations, NativeOrbitFitter(Spy()), None, propagator=propagator
    )
    assert calls == []
    assert len(orbits) == 1


@pytest.mark.parametrize(
    "module, name, replacement",
    [
        ("differential_correction", "fit_least_squares", "rejection='none'"),
        ("differential_correction", "iterative_fit", "rejection='worst_residual'"),
        ("rejection", "cmc2003_fit", "refine("),
        ("iod", "iod", "run_iod("),
        ("iod", "iod_worker", "initial_orbit_determination"),
    ],
)
def test_overlapping_entry_points_are_deprecated_aliases(module, name, replacement):
    """The overlapping entry points warn and name the stage they converge on;
    their signature is the implementation's, which internal code calls
    without a warning."""
    import importlib
    import inspect
    import warnings

    mod = importlib.import_module(f"adam_core.orbit_determination.{module}")
    alias = getattr(mod, name)
    implementation = getattr(mod, f"_{name}")
    assert inspect.signature(alias) == inspect.signature(implementation)
    assert alias.__doc__.startswith(".. deprecated::")
    with pytest.warns(DeprecationWarning, match=replacement.replace("(", r"\(")):
        try:
            alias()
        except TypeError:
            pass
    # The run_od stack itself never goes through a deprecated alias.
    observations = make_synthetic_observations()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        run_od(
            observations,
            NativeOrbitFitter(TwoBodyPropagator(), iod_rchi2_threshold=1e6),
            None,
            propagator=TwoBodyPropagator(),
        )


class CapsuleTwoBody(TwoBodyPropagator):
    """A Python propagator whose dynamics live in a native extension and are
    exposed as an external-propagator capsule (as ASSIST does): every OD
    stage runs them natively, never calling back into Python."""

    from adam_core.propagator import NativeBackend as _NativeBackend

    native_backend = _NativeBackend(
        name="adam_core.two_body.external",
        version="test",
        work_units=frozenset({"propagator_capsule"}),
    )
    ephemeris_calls = 0

    def propagator_capsule(self):
        from adam_core import _rust_native

        return _rust_native.two_body_propagator_capsule()

    def generate_ephemeris(self, orbits, observers, **kwargs):
        type(self).ephemeris_calls += 1
        return super().generate_ephemeris(orbits, observers, **kwargs)


def test_external_propagator_capsule_runs_od_natively():
    import numpy.testing as _npt

    observations = make_synthetic_observations()
    CapsuleTwoBody.ephemeris_calls = 0
    propagator = CapsuleTwoBody()
    native, native_members = run_od(
        observations,
        NativeOrbitFitter(propagator, iod_rchi2_threshold=1e6),
        None,
        propagator=propagator,
    )
    assert CapsuleTwoBody.ephemeris_calls == 0
    callback, callback_members = run_od(
        observations,
        NativeOrbitFitter(TwoBodyPropagator(), iod_rchi2_threshold=1e6),
        None,
        propagator=TwoBodyPropagator(),
    )
    _npt.assert_allclose(
        native.coordinates.values, callback.coordinates.values, rtol=0, atol=1e-8
    )
    assert native_members.outlier.to_pylist() == callback_members.outlier.to_pylist()
    # Provenance names the external backend.
    assert (
        native.provenance.refinement_backend[0]
        .as_py()
        .startswith("adam_core.two_body.external")
    )
    # A capsule of another name is refused.
    with pytest.raises(ValueError, match="capsule"):
        _check_bad_capsule()


def _check_bad_capsule():
    import ctypes

    from adam_core.orbit_determination.differential_correction import (
        fit_least_squares,
    )
    from adam_core.orbit_determination.tests.test_differential_correction import (
        make_initial_guess,
    )

    class Wrong(TwoBodyPropagator):
        from adam_core.propagator import NativeBackend as _NativeBackend

        native_backend = _NativeBackend(
            name="wrong", version="0", work_units=frozenset({"propagator_capsule"})
        )

        def propagator_capsule(self):
            PyCapsule_New = ctypes.pythonapi.PyCapsule_New
            PyCapsule_New.restype = ctypes.py_object
            PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
            return PyCapsule_New(ctypes.c_void_p(1), b"not.a.propagator", None)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fit_least_squares(make_initial_guess(), make_synthetic_observations(), Wrong())


@pytest.mark.native_only
def test_python_propagators_are_refused():
    """Orbit determination never calls back into Python per prediction: a
    propagator without a native backend is refused with a clear error; the
    native routes (adam-core's two-body propagator, a declared external
    propagator capsule) run."""
    from adam_core.orbit_determination import refine, run_iod

    observations = make_synthetic_observations()
    with pytest.raises(TypeError, match="no native orbit-determination backend"):
        run_iod(observations, TwoBodyPropagator())
    with pytest.raises(TypeError, match="no native orbit-determination backend"):
        run_od(
            observations,
            NativeOrbitFitter(TwoBodyPropagator(), iod_rchi2_threshold=1e6),
            None,
            propagator=TwoBodyPropagator(),
        )
    # A declared capsule runs natively, and a subclass that changes the
    # dynamics is not covered by it.
    orbit, _, report = run_iod(observations, CapsuleTwoBody(), rchi2_threshold=1e6)
    assert report.found
    refined, _ = refine(orbit, observations, CapsuleTwoBody())
    assert refined.success[0].as_py()

    class ChangedCapsule(CapsuleTwoBody):
        def propagate_orbits(self, *args, **kwargs):
            return super().propagate_orbits(*args, **kwargs)

    with pytest.raises(TypeError, match="not covered"):
        run_iod(observations, ChangedCapsule())
