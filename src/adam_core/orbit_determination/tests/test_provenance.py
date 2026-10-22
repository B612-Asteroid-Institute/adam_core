"""Run-level OD provenance (`OdProvenance`) on both run_od routes."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import adam_core
from adam_core import _rust_native
from adam_core.orbit_determination import NativeOrbitFitter, OdProvenance
from adam_core.orbit_determination.observation_uncertainty import (
    PerformanceWeightedModel,
)
from adam_core.orbit_determination.od_orchestration import run_od
from adam_core.orbit_determination.provenance import (
    _fit_input_sha256,
    backend_identity,
    complete_provenance,
    model_descriptor,
)
from adam_core.orbit_determination.tests.test_differential_correction import (
    NativeTwoBodyPropagator,
    TwoBodyPropagator,
    make_synthetic_observations,
)
from adam_core.orbit_determination.tests.test_observation_uncertainty import (
    make_bias_table,
)


def _record(orbits) -> dict:
    assert len(orbits) == 1
    return {
        name: orbits.provenance.column(name)[0].as_py()
        for name in OdProvenance.schema.names
    }


@pytest.mark.parametrize("route", ["composed", "fused"])
def test_run_od_records_provenance(route):
    observations = make_synthetic_observations()
    table = make_bias_table([{"obs_code": "500", "chi2_per_obs": 1.0}])

    class PythonAppliedModel(PerformanceWeightedModel):
        """Same arithmetic, with an override that selects core orchestration."""

        def apply(self, observations):
            return super().apply(observations)

    model = (
        PythonAppliedModel(table)
        if route == "composed"
        else PerformanceWeightedModel(table)
    )
    propagator_class = NativeTwoBodyPropagator
    orbits, members = run_od(
        observations,
        NativeOrbitFitter(
            propagator_class(),
            iod_rchi2_threshold=123.0,
            observation_selection_method="combinations",
            velocity_method="gibbs",
        ),
        [model],
        propagator=propagator_class(),
    )
    record = _record(orbits)
    assert record["adam_core_version"] == adam_core.__version__
    repo = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert record["adam_core_build"] == head
    assert len(record["input_fingerprint"]) == 64
    assert record["route"] == route
    assert record["fitter"].endswith("NativeOrbitFitter")
    assert record["refinement_backend"] == backend_identity(propagator_class())
    assert record["iod_backend"] == record["refinement_backend"]
    for field in ("iod_provider", "refinement_provider"):
        provider = json.loads(record[field])
        assert (
            provider["name"]
            in record[
                "iod_backend" if field == "iod_provider" else "refinement_backend"
            ]
        )
        assert provider["abi_version"] == 2
        assert provider["identity_validated_by_capsule"] is True
        assert provider["capabilities"] == provider["known_capabilities"]
        assert provider["integration_time_scale"] == "tdb"
        assert len(provider["configuration_fingerprint"]) == 64
        assert len(provider["data_fingerprint"]) == 64
    iod_diagnostics = json.loads(record["iod_diagnostics"])
    assert set(("pair_visits", "raw_pair_visits", "middle_visits")) <= set(
        iod_diagnostics
    )
    refinement_diagnostics = json.loads(record["refinement_diagnostics"])
    assert set(("passes", "n_rejected", "n_recovered", "flags")) <= set(
        refinement_diagnostics
    )
    settings = json.loads(record["settings"])
    assert set(settings) == {"iod", "refinement", "fit"}
    assert settings["iod"]["acceptance"] == "corrected"
    assert settings["iod"]["rchi2_threshold"] == 123.0
    assert settings["iod"]["observation_selection_method"] == "combinations"
    assert settings["iod"]["velocity_method"] == "gibbs"
    (descriptor,) = json.loads(record["models"])
    assert descriptor["type"].endswith(
        "PythonAppliedModel" if route == "composed" else "PerformanceWeightedModel"
    )
    assert all(len(h) == 64 for h in descriptor["tables_sha256"].values())
    assert descriptor["implementation_version"] == adam_core.__version__
    # The IOD's ambiguity reaches the final orbit on both routes.
    assert record["iod_ambiguous"] is False and record["iod_alternatives"] == 0
    rejected = sum(members.outlier.to_pylist())
    assert record["num_rejected"] == rejected
    assert record["rejected_fraction"] == pytest.approx(rejected / len(members))


def test_provenance_keeps_exact_native_identity_not_later_declarations():
    from adam_core.propagator import NativeBackend

    class IodProvider(NativeTwoBodyPropagator):
        native_backend = NativeBackend(
            "iod-stage", "1.2.3-rc.4", frozenset({"propagator_capsule"})
        )

    class RefinementProvider(NativeTwoBodyPropagator):
        native_backend = NativeBackend(
            "refinement-stage",
            "2.0.0-beta.2",
            frozenset({"propagator_capsule"}),
        )

    observations = make_synthetic_observations()
    fitted, members = run_od(
        observations,
        NativeOrbitFitter(TwoBodyPropagator()),
        None,
        propagator=TwoBodyPropagator(),
    )
    stamped = complete_provenance(
        fitted,
        members,
        fitter=NativeOrbitFitter(IodProvider()),
        propagator=RefinementProvider(),
        models=[],
        route="composed",
    )
    record = _record(stamped)
    assert record["iod_backend"].startswith("adam_core.two_body.external ")
    assert record["refinement_backend"].startswith("adam_core.two_body.external ")
    assert "iod-stage" not in record["iod_provider"]
    assert "refinement-stage" not in record["refinement_provider"]


def test_provenance_preflight_refuses_non_reproducible_builds(monkeypatch):
    from adam_core.orbit_determination.provenance import _preflight_provenance

    for identity in ("unknown", "dirty:" + "a" * 40, "A" * 40, "abc"):
        monkeypatch.setattr(
            _rust_native,
            "build_identity",
            lambda identity=identity: (adam_core.__version__, identity),
        )
        with pytest.raises(RuntimeError, match="exact clean source commit"):
            _preflight_provenance(object(), [])


def test_backend_identity_names_declared_and_python_backends():
    assert backend_identity(NativeTwoBodyPropagator()) == (
        f"adam_core.two_body.external {_rust_native.two_body_propagator_semver()}"
    )
    assert backend_identity(NativeTwoBodyPropagator) == backend_identity(
        NativeTwoBodyPropagator()
    )

    class PythonOnly(TwoBodyPropagator):
        native_backend = None

    python = backend_identity(PythonOnly())
    assert python.startswith("python:") and "PythonOnly" in python
    assert backend_identity(None) is None


def test_model_descriptor_hashes_table_content():
    same = model_descriptor(
        PerformanceWeightedModel(
            make_bias_table([{"obs_code": "500", "chi2_per_obs": 4.0}])
        )
    )
    again = model_descriptor(
        PerformanceWeightedModel(
            make_bias_table([{"obs_code": "500", "chi2_per_obs": 4.0}])
        )
    )
    other = model_descriptor(
        PerformanceWeightedModel(
            make_bias_table([{"obs_code": "500", "chi2_per_obs": 9.0}])
        )
    )
    assert same == again
    assert same["tables_sha256"] != other["tables_sha256"]


def test_model_descriptor_hashes_numpy_configuration_and_sets():
    import numpy as np

    class ArrayModel:
        def __init__(self, values, excluded):
            self.values = values
            self.excluded = excluded

    first = model_descriptor(ArrayModel(np.array([1.0, 2.0]), {"a", "b"}))
    same = model_descriptor(ArrayModel(np.array([1.0, 2.0]), {"b", "a"}))
    changed_array = model_descriptor(ArrayModel(np.array([1.0, 3.0]), {"a", "b"}))
    changed_set = model_descriptor(ArrayModel(np.array([1.0, 2.0]), {"a", "c"}))
    assert first == same
    assert first["descriptor_sha256"] != changed_array["descriptor_sha256"]
    assert first["descriptor_sha256"] != changed_set["descriptor_sha256"]


def test_model_descriptor_canonicalizes_colliding_sets_and_mapping_order():
    class Key:
        def __init__(self, name):
            self.name = name

        def __hash__(self):
            return 0

        def __eq__(self, other):
            return isinstance(other, Key) and self.name == other.name

    class Model:
        def __init__(self, values, mapping):
            self.values = values
            self.mapping = mapping

        def apply(self, observations):
            return observations

    names = ["alpha", "beta", "gamma", "delta", "epsilon"]
    first = Model(
        set(Key(name) for name in names),
        dict((Key(name), index) for index, name in enumerate(names)),
    )
    reverse = Model(
        set(Key(name) for name in reversed(names)),
        dict((Key(name), names.index(name)) for name in reversed(names)),
    )
    assert model_descriptor(first) == model_descriptor(reverse)


def test_input_fingerprint_hashes_caller_and_used_astrometry_independently():
    observations = make_synthetic_observations()
    fitted, members = run_od(
        observations,
        NativeOrbitFitter(TwoBodyPropagator()),
        None,
        propagator=TwoBodyPropagator(),
    )
    assert len(fitted) == 1
    baseline = _fit_input_sha256(observations, members)

    renamed = observations.set_column(
        "id",
        pa.array(
            [f"renamed-{row}" for row in range(len(observations))],
            type=observations.table.schema.field("id").type,
        ),
    )
    assert _fit_input_sha256(renamed, members) != baseline

    longitude = members.used_astrometry.lon.to_numpy(zero_copy_only=False).copy()
    longitude[0] = np.nextafter(longitude[0], np.inf)
    changed_members = members.set_column("used_astrometry.lon", pa.array(longitude))
    assert _fit_input_sha256(observations, changed_members) != baseline


def test_model_descriptor_rejects_unsupported_and_cyclic_configuration():
    class Model:
        pass

    unsupported = Model()
    unsupported.value = object()
    with pytest.raises(TypeError, match="unsupported provenance field"):
        model_descriptor(unsupported)

    cyclic = Model()
    cyclic.value = []
    cyclic.value.append(cyclic.value)
    with pytest.raises(ValueError, match="cyclic provenance"):
        model_descriptor(cyclic)


def test_other_entry_points_leave_provenance_null():
    from adam_core.orbit_determination import fit_least_squares
    from adam_core.orbit_determination.tests.test_differential_correction import (
        make_initial_guess,
    )

    orbits, _ = fit_least_squares(
        make_initial_guess(), make_synthetic_observations(), TwoBodyPropagator()
    )
    assert orbits.table.column("provenance").null_count == len(orbits)


def test_ambiguous_iod_survives_refinement():
    from adam_core.orbit_determination.provenance import (
        carry_iod_provenance,
        with_iod_provenance,
    )

    observations = make_synthetic_observations()
    orbits, _ = run_od(
        observations,
        NativeOrbitFitter(TwoBodyPropagator()),
        None,
        propagator=TwoBodyPropagator(),
    )
    source_record = _record(orbits)
    iod = with_iod_provenance(orbits, [4, 9])
    assert _record(iod)["iod_ambiguous"] is True
    assert _record(iod)["iod_alternatives"] == 2
    assert _record(iod)["iod_provider"] == source_record["iod_provider"]
    assert _record(iod)["iod_diagnostics"] == source_record["iod_diagnostics"]
    refined = orbits.set_column("provenance", OdProvenance.nulls(len(orbits)))
    carried = carry_iod_provenance(iod, refined)
    assert _record(carried)["iod_ambiguous"] is True
    assert _record(carried)["iod_alternatives"] == 2
    assert _record(carried)["iod_provider"] == source_record["iod_provider"]
    assert _record(carried)["iod_diagnostics"] == source_record["iod_diagnostics"]
    assert _record(carried)["refinement_provider"] is None
    # A refinement that recorded its own IOD fields keeps them.
    assert carry_iod_provenance(iod, orbits) is orbits
    assert _record(orbits)["iod_ambiguous"] is False


def test_model_descriptor_is_stable_across_python_hash_seeds(tmp_path):
    import os
    import sys

    module_path = tmp_path / "seed_model.py"
    module_path.write_text(
        "class Model:\n"
        "    def __init__(self):\n"
        "        self.excluded = {'V', 'a', 'b', 'c'}\n"
        "    def apply(self, observations):\n"
        "        return observations\n",
        encoding="utf-8",
    )
    script = (
        "from seed_model import Model; "
        "from adam_core.orbit_determination.provenance import model_descriptor; "
        "print(model_descriptor(Model())['descriptor_sha256'])"
    )
    hashes = set()
    for seed in ("0", "1", "12345", "random"):
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        pythonpath = os.pathsep.join(
            part for part in (str(tmp_path), environment.get("PYTHONPATH")) if part
        )
        environment["PYTHONPATH"] = pythonpath
        hashes.add(
            subprocess.check_output(
                [sys.executable, "-c", script],
                env=environment,
                text=True,
            ).strip()
        )
    assert len(hashes) == 1


def test_callable_identity_is_source_based_and_captures_delegated_helpers(
    tmp_path, monkeypatch
):
    import importlib
    import linecache
    import sys

    module_name = "_adam_core_provenance_helper_fixture"
    module_path = tmp_path / f"{module_name}.py"

    def write(helper_value: int) -> None:
        module_path.write_text(
            "def helper():\n"
            f"    return {helper_value}\n\n"
            "class Model:\n"
            "    def apply(self, observations):\n"
            "        helper()\n"
            "        return observations\n",
            encoding="utf-8",
        )
        importlib.invalidate_caches()
        linecache.clearcache()

    monkeypatch.syspath_prepend(str(tmp_path))
    write(1)
    module = importlib.import_module(module_name)
    first = model_descriptor(module.Model())
    implementation = first["implementation"]
    assert "bytecode_sha256" not in json.dumps(implementation)
    assert len(implementation["entrypoint_source_sha256"]) == 64
    assert len(implementation["module_source_sha256"]) == 64

    write(2)
    source = compile(module_path.read_text(encoding="utf-8"), str(module_path), "exec")
    spec = module.__spec__
    module.__dict__.clear()
    module.__dict__.update(
        {
            "__name__": module_name,
            "__file__": str(module_path),
            "__package__": "",
            "__spec__": spec,
        }
    )
    exec(source, module.__dict__)
    second = model_descriptor(module.Model())
    assert (
        first["implementation"]["entrypoint_source_sha256"]
        == second["implementation"]["entrypoint_source_sha256"]
    )
    assert (
        first["implementation"]["module_source_sha256"]
        != second["implementation"]["module_source_sha256"]
    )
    assert first["descriptor_sha256"] != second["descriptor_sha256"]
    sys.modules.pop(module_name, None)


def test_callable_identity_preserves_closure_defaults_path_and_enum(tmp_path):
    from enum import Enum

    def make_apply(marker):
        def apply(observations, default="default-value"):
            if marker:
                return observations
            return observations

        return apply

    class Mode(Enum):
        FIRST = "same-value"
        SECOND = "same-value-too"

    class Model:
        def __init__(self, marker, path, mode):
            self.apply = make_apply(marker)
            self.path = path
            self.mode = mode

    path = tmp_path / "configuration.dat"
    path.write_text("first", encoding="utf-8")
    first = model_descriptor(Model("closure-a", path, Mode.FIRST))
    implementation = first["implementation"]
    assert implementation["closure"] == ["closure-a"]
    assert implementation["defaults"] == ["default-value"]
    assert first["parameters"]["path"]["sha256"] == hashlib.sha256(b"first").hexdigest()
    assert first["parameters"]["mode"]["name"] == "FIRST"

    closure_changed = model_descriptor(Model("closure-b", path, Mode.FIRST))
    enum_changed = model_descriptor(Model("closure-a", path, Mode.SECOND))
    path.write_text("second", encoding="utf-8")
    path_changed = model_descriptor(Model("closure-a", path, Mode.FIRST))
    assert first["descriptor_sha256"] != closure_changed["descriptor_sha256"]
    assert first["descriptor_sha256"] != enum_changed["descriptor_sha256"]
    assert first["descriptor_sha256"] != path_changed["descriptor_sha256"]


def test_builtin_callable_identity_is_named_and_interpreter_independent():
    class Model:
        def __init__(self, callback):
            self.callback = callback

        def apply(self, observations):
            return observations

    length = model_descriptor(Model(len))
    absolute = model_descriptor(Model(abs))
    encoded = json.dumps(length["parameters"]["callback"], sort_keys=True)
    assert "bytecode" not in encoded
    assert '"implementation_kind": "builtin"' in encoded
    assert length["descriptor_sha256"] != absolute["descriptor_sha256"]


def test_dynamic_callable_without_module_fails_closed():
    class Model:
        def apply(self, observations):
            return observations

    Model.apply.__module__ = None
    with pytest.raises(TypeError, match="no reproducible Python module identity"):
        model_descriptor(Model())


def test_preflight_failure_happens_before_fitter_work(monkeypatch):
    import adam_core.orbit_determination.od_orchestration as orchestration

    calls = 0

    class CountingFitter:
        def full_od(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("fit must not run before provenance preflight")

    def refuse(*args, **kwargs):
        raise TypeError("deliberate provenance refusal")

    monkeypatch.setattr(orchestration, "_preflight_provenance", refuse)
    with pytest.raises(TypeError, match="deliberate provenance refusal"):
        run_od(
            make_synthetic_observations(),
            CountingFitter(),
            None,
            propagator=TwoBodyPropagator(),
        )
    assert calls == 0


def test_uniform_provider_rejects_disagreement_and_partial_omission():
    from adam_core.orbit_determination.provenance import _uniform_provider

    assert _uniform_provider(["a", "a"], "IOD") == "a"
    with pytest.raises(RuntimeError, match="omitted or disagreed"):
        _uniform_provider(["a", "b"], "IOD")
    with pytest.raises(RuntimeError, match="omitted or disagreed"):
        _uniform_provider(["a", None], "IOD")


def test_complete_provenance_keeps_distinct_native_stage_providers():
    observations = make_synthetic_observations()
    fitter = NativeOrbitFitter(TwoBodyPropagator())
    fitted, members = run_od(
        observations,
        fitter,
        None,
        propagator=TwoBodyPropagator(),
    )
    assert len(fitted) == 1
    base = fitted.provenance
    iod = {
        "name": "iod-native",
        "version": "1",
        "abi_version": 2,
    }
    refinement = {
        "name": "refinement-native",
        "version": "2",
        "abi_version": 2,
    }
    modified = fitted.set_column(
        "provenance",
        OdProvenance.from_kwargs(
            **{
                **{name: base.table.column(name) for name in OdProvenance.schema.names},
                "iod_provider": [json.dumps(iod, sort_keys=True)],
                "refinement_provider": [json.dumps(refinement, sort_keys=True)],
            }
        ),
    )
    stamped = complete_provenance(
        modified,
        members,
        fitter=fitter,
        propagator=TwoBodyPropagator(),
        models=[],
        route="fused",
        observations=observations,
    )
    record = _record(stamped)
    assert json.loads(record["iod_provider"])["name"] == "iod-native"
    assert json.loads(record["refinement_provider"])["name"] == "refinement-native"


def test_preflight_describes_actual_models_before_apply_or_fit():
    apply_calls = 0
    fit_calls = 0

    class UndescribableModel:
        def apply(self, observations):
            nonlocal apply_calls
            apply_calls += 1
            return observations

    class CountingFitter:
        def full_od(self, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            raise AssertionError("fit must not run before model provenance preflight")

    UndescribableModel.apply.__module__ = None
    with pytest.raises(TypeError, match="no reproducible Python module identity"):
        run_od(
            make_synthetic_observations(),
            CountingFitter(),
            [UndescribableModel()],
            propagator=TwoBodyPropagator(),
        )
    assert apply_calls == 0
    assert fit_calls == 0


def test_preflight_rejects_native_python_version_mismatch_before_work(monkeypatch):
    apply_calls = 0
    fit_calls = 0

    class Model:
        def apply(self, observations):
            nonlocal apply_calls
            apply_calls += 1
            return observations

    class CountingFitter:
        def full_od(self, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            raise AssertionError("fit must not run after version mismatch")

    monkeypatch.setattr(
        _rust_native,
        "build_identity",
        lambda: ("999.0.0", "0" * 40),
    )
    with pytest.raises(RuntimeError, match="does not match Python"):
        run_od(
            make_synthetic_observations(),
            CountingFitter(),
            [Model()],
            propagator=TwoBodyPropagator(),
        )
    assert apply_calls == 0
    assert fit_calls == 0


def test_sourceless_module_and_entrypoint_fail_closed(monkeypatch):
    import sys
    import types

    import adam_core.orbit_determination.provenance as provenance

    module_name = "_adam_core_sourceless_provenance_fixture"
    module = types.ModuleType(module_name)
    module.__file__ = None
    module.__spec__ = None
    exec(
        compile(
            "class Model:\n"
            "    def apply(self, observations):\n"
            "        return observations\n",
            "<dynamic-provenance>",
            "exec",
        ),
        module.__dict__,
    )
    sys.modules[module_name] = module
    try:
        with pytest.raises(TypeError, match="no reproducible source content"):
            model_descriptor(module.Model())

        monkeypatch.setattr(
            provenance.inspect,
            "getsource",
            lambda _value: "def apply(self, observations): return observations",
        )
        with pytest.raises(TypeError, match="no reproducible source content"):
            model_descriptor(module.Model())
    finally:
        sys.modules.pop(module_name, None)


def test_callable_identity_covers_kwdefaults_without_bytecode():
    class Model:
        def __init__(self, callback):
            self.callback = callback

        def apply(self, observations):
            return observations

    def first(*, scale=1):
        return scale

    def second(*, scale=2):
        return scale

    one = model_descriptor(Model(first))
    two = model_descriptor(Model(second))
    identity = one["parameters"]["callback"]["callable"]
    assert identity["kwdefaults"] == {"scale": 1}
    encoded = json.dumps(identity, sort_keys=True)
    assert "bytecode_sha256" not in encoded
    assert "code" not in identity
    assert one["descriptor_sha256"] != two["descriptor_sha256"]


def test_enum_class_type_and_nested_object_classes_are_distinct():
    from enum import Enum

    class FirstEnum(Enum):
        VALUE = "same"

    class SecondEnum(Enum):
        VALUE = "same"

    class FirstConfig:
        def __init__(self):
            self.value = 1

    class SecondConfig:
        def __init__(self):
            self.value = 1

    class Model:
        def __init__(self, parameter):
            self.parameter = parameter

        def apply(self, observations):
            return observations

    pairs = [
        (FirstEnum, SecondEnum),
        (FirstEnum.VALUE, SecondEnum.VALUE),
        (int, float),
        (np.float32, np.float64),
        (FirstConfig(), SecondConfig()),
    ]
    for left, right in pairs:
        assert (
            model_descriptor(Model(left))["descriptor_sha256"]
            != model_descriptor(Model(right))["descriptor_sha256"]
        )


def test_partial_and_stdlib_object_parameters_are_portable_and_distinct():
    import argparse
    import functools

    def helper(value):
        return value

    class Model:
        def __init__(self, parameter):
            self.parameter = parameter

        def apply(self, observations):
            return observations

    one = model_descriptor(Model(functools.partial(helper, 1)))
    two = model_descriptor(Model(functools.partial(helper, 2)))
    assert one["descriptor_sha256"] != two["descriptor_sha256"]
    namespace = model_descriptor(Model(argparse.Namespace(a=1)))
    encoded = json.dumps(namespace["parameters"]["parameter"], sort_keys=True)
    assert "stdlib-named-v1" in encoded
    assert "module_source_sha256" not in encoded


def test_model_identity_covers_mro_overrides_and_sibling_helpers(tmp_path):
    import os
    import sys

    base = (
        "class Base:\n"
        "    def apply(self, observations):\n"
        "        return self.transform(observations)\n"
        "    def transform(self, observations):\n"
        "        return observations\n"
    )
    user = (
        "import helper_module\n"
        "class User:\n"
        "    def apply(self, observations):\n"
        "        return helper_module.transform(observations)\n"
    )
    (tmp_path / "base_module.py").write_text(base, encoding="utf-8")
    (tmp_path / "user_module.py").write_text(user, encoding="utf-8")
    script = (
        "from subclass_module import Sub; from user_module import User; "
        "from adam_core.orbit_determination.provenance import model_descriptor; "
        "print(model_descriptor(Sub())['descriptor_sha256']); "
        "print(model_descriptor(User())['descriptor_sha256'])"
    )
    hashes = []
    for marker in ("first", "second"):
        (tmp_path / "subclass_module.py").write_text(
            "from base_module import Base\n"
            "class Sub(Base):\n"
            "    def transform(self, observations):\n"
            f"        marker = {marker!r}\n"
            "        return observations\n",
            encoding="utf-8",
        )
        (tmp_path / "helper_module.py").write_text(
            "def transform(observations):\n"
            f"    marker = {marker!r}\n"
            "    return observations\n",
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(tmp_path), environment.get("PYTHONPATH", "")]
        )
        hashes.append(
            subprocess.check_output(
                [sys.executable, "-c", script], env=environment, text=True
            )
            .strip()
            .splitlines()
        )
    assert hashes[0][0] != hashes[1][0]
    assert hashes[0][1] != hashes[1][1]

    # Unrelated sibling/test-like source is not part of the executing import
    # closure and therefore must not make an installed/source checkout differ.
    (tmp_path / "unrelated_module.py").write_text("marker = 'unrelated'\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), environment.get("PYTHONPATH", "")]
    )
    unrelated = (
        subprocess.check_output(
            [sys.executable, "-c", script], env=environment, text=True
        )
        .strip()
        .splitlines()
    )
    assert unrelated == hashes[1]
