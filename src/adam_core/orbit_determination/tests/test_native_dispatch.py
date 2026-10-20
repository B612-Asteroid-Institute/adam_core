"""Core-owned OD dispatch must preserve custom model and fitter behavior."""

from pathlib import Path

import numpy.testing as npt
import pytest

import adam_core
from adam_core import _rust_native
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
    """Capsule provider with forbidden OD-shaped methods that must be ignored."""

    def run_od(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("provider-owned run_od was dispatched")

    def full_od(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("provider-owned full_od was dispatched")


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


@pytest.mark.parametrize(
    "model_class", [IdentityModel, UnchangedIdentity, NativeScaleSigmas]
)
def test_compatible_models_keep_fused_run_od(observations, model_class):
    model = model_class()
    propagator = RecordingNativePropagator()
    fitted, members = run_od(observations, make_fitter(), model, propagator=propagator)
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
    npt.assert_allclose(
        members.used_astrometry.sigma_lon.to_numpy(),
        4.0 * members.original_astrometry.sigma_lon.to_numpy(),
    )


class CustomHookReached(Exception):
    pass


class CapsuleWiringReached(Exception):
    pass


@pytest.mark.parametrize("entrypoint", ["full_od", "run_od"])
def test_distinct_iod_and_refinement_capsules_are_obtained_and_forwarded(
    observations, monkeypatch, entrypoint
):
    from adam_core.propagator import NativeBackend

    requested = []
    iod_token = object()
    refinement_token = object()

    class IodProvider(TwoBodyPropagator):
        native_backend = NativeBackend(
            "test.iod", "1.2.3-rc.4", frozenset({"propagator_capsule"})
        )

        def propagator_capsule(self):
            requested.append("iod")
            return iod_token

    class RefinementProvider(TwoBodyPropagator):
        native_backend = NativeBackend(
            "test.refinement", "2.0.0-beta.2", frozenset({"propagator_capsule"})
        )

        def propagator_capsule(self):
            requested.append("refinement")
            return refinement_token

    def assert_wiring(refinement_descriptor, *args, iod_propagator=None, **kwargs):
        assert refinement_descriptor[:3] == (
            refinement_token,
            "test.refinement",
            "2.0.0-beta.2",
        )
        assert iod_propagator[:3] == (iod_token, "test.iod", "1.2.3-rc.4")
        assert refinement_descriptor[0] is not iod_propagator[0]
        raise CapsuleWiringReached

    monkeypatch.setattr(
        _rust_native,
        "full_od_ipc" if entrypoint == "full_od" else "run_od_ipc",
        assert_wiring,
    )
    fitter = NativeOrbitFitter(IodProvider(), iod_rchi2_threshold=1e6)
    refinement = RefinementProvider()
    with pytest.raises(CapsuleWiringReached):
        if entrypoint == "full_od":
            fitter.full_od("capsule-spy", observations, refinement)
        else:
            run_od(observations, fitter, None, propagator=refinement)
    assert requested == ["refinement", "iod"]


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


@pytest.mark.parametrize(
    "fitter_class", [NativeOrbitFitter, UnchangedFitter, CompatibleFitter]
)
@pytest.mark.parametrize("entrypoint", ["run_od", "full_od"])
def test_compatible_fitters_keep_fused_paths(observations, fitter_class, entrypoint):
    fitter = make_fitter(fitter_class)
    propagator = RecordingNativePropagator()
    if entrypoint == "run_od":
        fitted, _ = run_od(observations, fitter, None, propagator=propagator)
    else:
        fitted, _ = fitter.full_od("compatible", observations, propagator)
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

    native_backend = None

    def run_od(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("an undeclared work unit was dispatched")


def test_dispatch_requires_a_declared_backend():
    from adam_core.orbit_determination._native_dispatch import (
        native_backend,
        native_propagator,
        native_work_unit,
    )

    base = NativeTwoBodyPropagator()
    assert native_backend(base) is NativeTwoBodyPropagator.native_backend
    assert native_work_unit(base, "propagator_capsule") is not None
    assert native_work_unit(base, "run_od") is None
    assert native_work_unit(UndeclaredMethods(), "run_od") is None
    assert native_backend(ChangedDynamics()) is None
    patched = NativeTwoBodyPropagator()
    patched.propagator_capsule = lambda: None
    assert native_backend(patched) is None
    descriptor = native_propagator(base)
    assert descriptor[1:] == (
        "adam_core.two_body.external",
        _rust_native.two_body_propagator_semver(),
        True,
    )


def test_source_checkout_uses_canonical_native_semver_without_version_module(
    monkeypatch,
):
    from adam_core.orbit_determination._native_dispatch import native_propagator
    from adam_core.propagator import NativeBackend

    assert not Path(adam_core.__file__).with_name("_version.py").exists()
    monkeypatch.setattr(adam_core, "__version__", "1.2.3rc4")
    monkeypatch.setattr(
        _rust_native, "two_body_propagator_semver", lambda: "1.2.3-rc.4"
    )

    class PrereleaseProvider(TwoBodyPropagator):
        native_backend = NativeBackend(
            "test.prerelease",
            _rust_native.two_body_propagator_semver(),
            frozenset({"propagator_capsule"}),
        )

    descriptor = native_propagator(PrereleaseProvider())
    assert descriptor[2] == "1.2.3-rc.4"
    assert descriptor[2] != adam_core.__version__


def test_native_backend_rejects_algorithm_work_units():
    from adam_core.propagator import NativeBackend

    with pytest.raises(ValueError, match="propagation-only"):
        NativeBackend("provider", "1", frozenset({"run_od"}))


@pytest.mark.parametrize(("name", "version"), [("", "1"), ("provider", "")])
def test_native_backend_rejects_empty_identity(name, version):
    from adam_core.propagator import NativeBackend

    with pytest.raises(ValueError, match="must be non-empty"):
        NativeBackend(name, version, frozenset({"propagator_capsule"}))


def test_overridden_dynamics_fail_without_a_native_capsule():
    """A subclass with undeclared dynamics cannot fall back to Python calls."""
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
    with pytest.raises(TypeError, match="has no native orbit-determination backend"):
        run_od(observations, NativeOrbitFitter(Spy()), None, propagator=propagator)
    assert calls == []


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
        version=_rust_native.two_body_propagator_semver(),
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
    reference, reference_members = run_od(
        observations,
        NativeOrbitFitter(TwoBodyPropagator(), iod_rchi2_threshold=1e6),
        None,
        propagator=TwoBodyPropagator(),
    )
    _npt.assert_allclose(
        native.coordinates.values, reference.coordinates.values, rtol=0, atol=1e-8
    )
    assert native_members.outlier.to_pylist() == reference_members.outlier.to_pylist()
    # Provenance names the external backend.
    assert (
        native.provenance.refinement_backend[0]
        .as_py()
        .startswith("adam_core.two_body.external")
    )

    # A valid capsule cannot spoof the identity declared for provenance.
    class MismatchedIdentity(CapsuleTwoBody):
        from adam_core.propagator import NativeBackend as _NativeBackend

        native_backend = _NativeBackend(
            name="not-the-table",
            version="0",
            work_units=frozenset({"propagator_capsule"}),
        )

    from adam_core.orbit_determination import run_iod

    with pytest.raises(ValueError, match="identity mismatch"):
        run_iod(observations, MismatchedIdentity(), rchi2_threshold=1e6)

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


def test_python_propagators_are_refused():
    """Orbit determination never calls back into Python per prediction: a
    propagator without a native backend is refused with a clear error; the
    native routes (adam-core's two-body propagator, a declared external
    propagator capsule) run."""
    from adam_core.orbit_determination import refine, run_iod

    observations = make_synthetic_observations()

    class PythonOnly(TwoBodyPropagator):
        native_backend = None

    with pytest.raises(TypeError, match="no native orbit-determination backend"):
        run_iod(observations, PythonOnly())
    with pytest.raises(TypeError, match="no native orbit-determination backend"):
        run_od(
            observations,
            NativeOrbitFitter(PythonOnly(), iod_rchi2_threshold=1e6),
            None,
            propagator=PythonOnly(),
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
