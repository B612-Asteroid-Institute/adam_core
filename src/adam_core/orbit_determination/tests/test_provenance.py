"""Run-level OD provenance (`OdProvenance`) on both run_od routes."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import adam_core
from adam_core import _rust_native
from adam_core.orbit_determination import (
    ModelProvenance,
    NativeOrbitFitter,
    OdProvenance,
)
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

_TEST_IMPLEMENTATION = "1" * 64


class _ExplicitModel:
    """Test helper implementing the public custom-model contract."""

    implementation_fingerprint = _TEST_IMPLEMENTATION

    def model_provenance(self):
        return ModelProvenance(
            package="adam-core-tests",
            version="1.0",
            implementation_fingerprint=self.implementation_fingerprint,
            configuration=dict(vars(self)),
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

    class PythonAppliedModel(PerformanceWeightedModel, _ExplicitModel):
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
    if route == "fused":
        assert all(len(h) == 64 for h in descriptor["tables_sha256"].values())
    else:
        assert len(descriptor["parameters"]["bias_table"]["arrow_sha256"]) == 64
    assert descriptor["implementation_version"] == (
        "1.0" if route == "composed" else adam_core.__version__
    )
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

    class ArrayModel(_ExplicitModel):
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

    class Model(_ExplicitModel):
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
    class Model(_ExplicitModel):
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


def test_custom_descriptor_is_stable_across_hash_seed_and_import_order(tmp_path):
    import os
    import sys

    module_path = tmp_path / "seed_model.py"
    module_path.write_text(
        "from adam_core.orbit_determination import ModelProvenance\n"
        "class Model:\n"
        "    def __init__(self):\n"
        "        self.excluded = {'V', 'a', 'b', 'c'}\n"
        "    def apply(self, observations):\n"
        "        return observations\n"
        "    def model_provenance(self):\n"
        "        return ModelProvenance('example-model', '2.1', 'a' * 64, "
        "{'excluded': self.excluded})\n",
        encoding="utf-8",
    )
    script = (
        "{imports}; from seed_model import Model; "
        "from adam_core.orbit_determination import IdentityModel; "
        "from adam_core.orbit_determination.provenance import model_descriptor; "
        "print(model_descriptor(Model())['descriptor_sha256']); "
        "print(model_descriptor(IdentityModel())['descriptor_sha256'])"
    )
    hashes = set()
    for seed in ("0", "1", "12345", "random"):
        for imports in (
            "pass",
            "import scipy.optimize, pathlib, adam_core.coordinates._conversion",
        ):
            environment = dict(os.environ, PYTHONHASHSEED=seed)
            source_root = str(Path(__file__).resolve().parents[3])
            environment["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (
                    str(tmp_path),
                    source_root,
                    environment.get("PYTHONPATH"),
                )
                if part
            )
            hashes.add(
                tuple(
                    subprocess.check_output(
                        [sys.executable, "-c", script.format(imports=imports)],
                        env=environment,
                        text=True,
                    )
                    .strip()
                    .splitlines()
                )
            )
    assert len(hashes) == 1


def test_builtin_descriptor_uses_exact_build_and_not_import_history():
    table = make_bias_table([{"obs_code": "500", "chi2_per_obs": 4.0}])
    before = model_descriptor(PerformanceWeightedModel(table))
    import adam_core.coordinates._conversion  # noqa: F401
    import adam_core.observations.arrow_bridge  # noqa: F401

    after = model_descriptor(PerformanceWeightedModel(table))
    assert before == after
    implementation = before["implementation"]
    assert implementation == before["implementation_class"]
    assert implementation["kind"] == "adam-core-built-in-v1"
    assert implementation["entrypoint"] == "apply"
    assert implementation["build_commit"] == _rust_native.build_identity()[1]
    assert set(implementation) == {
        "kind",
        "build_commit",
        "class",
        "entrypoint",
    }
    encoded = json.dumps(before, sort_keys=True)
    for forbidden in (
        "bytecode",
        "sys.version",
        "module_source",
        "entrypoint_source",
        "source_tree",
    ):
        assert forbidden not in encoded


def test_custom_fingerprint_is_the_implementation_identity():
    class Model(_ExplicitModel):
        def __init__(self, fingerprint, scale=1):
            self.implementation_fingerprint = fingerprint
            self.scale = scale

        def apply(self, observations):
            return observations

    first = model_descriptor(Model("1" * 64))
    same = model_descriptor(Model("1" * 64))
    changed_science = model_descriptor(Model("2" * 64))
    changed_config = model_descriptor(Model("1" * 64, scale=2))
    assert first == same
    assert first["implementation"] == {
        "kind": "explicit-model-provenance-v1",
        "package": "adam-core-tests",
        "version": "1.0",
        "implementation_fingerprint": "1" * 64,
        "entrypoint": "apply",
    }
    assert changed_science["implementation"]["implementation_fingerprint"] == "2" * 64
    assert first["descriptor_sha256"] != changed_science["descriptor_sha256"]
    assert first["descriptor_sha256"] != changed_config["descriptor_sha256"]


def test_custom_contract_is_validated_exactly():
    class Model:
        def __init__(self, contract):
            self.contract = contract

        def apply(self, observations):
            return observations

        def model_provenance(self):
            return self.contract

    with pytest.raises(TypeError, match="must return ModelProvenance"):
        model_descriptor(Model({}))
    invalid = [
        ModelProvenance("bad package!", "1", "a" * 64, {}),
        ModelProvenance("package", "", "a" * 64, {}),
        ModelProvenance("package", "1", "A" * 64, {}),
        ModelProvenance("package", "1", "a" * 63, {}),
        ModelProvenance("package", "1", "a" * 64, []),
    ]
    for contract in invalid:
        with pytest.raises(TypeError, match="model provenance"):
            model_descriptor(Model(contract))


def test_common_configuration_values_are_portable_and_distinct(tmp_path):
    import datetime
    import functools
    from collections import namedtuple
    from enum import Enum

    class Mode(Enum):
        FIRST = "same"
        SECOND = "other"

    class OtherMode(Enum):
        FIRST = "same"

    class Nested:
        def __init__(self, value):
            self.value = value

    class OtherNested:
        def __init__(self, value):
            self.value = value

    Pair = namedtuple("Pair", "left right")
    OtherPair = namedtuple("OtherPair", "left right")

    def helper(value, *, scale=1):
        return value * scale

    path = tmp_path / "configuration.dat"
    path.write_text("content", encoding="utf-8")

    class Model(_ExplicitModel):
        def __init__(self, configuration):
            self.configuration = configuration

        def model_provenance(self):
            return ModelProvenance(
                "portable-config",
                "1",
                self.implementation_fingerprint,
                self.configuration,
            )

        def apply(self, observations):
            return observations

    configuration = {
        "int": np.int64(5),
        "float": np.float32(1.5),
        "type": np.float64,
        "ufunc": np.sqrt,
        "date": datetime.date(2024, 2, 29),
        "datetime": datetime.datetime(2024, 2, 29, 12, 1, 2, 3),
        "namedtuple_class": Pair,
        "namedtuple": Pair(1, 2),
        "enum_class": Mode,
        "enum": Mode.FIRST,
        "partial": functools.partial(helper, 3, scale=4),
        "nested_class": Nested,
        "nested": Nested(7),
        "path": path,
    }
    first = model_descriptor(Model(configuration))
    reordered = model_descriptor(Model(dict(reversed(list(configuration.items())))))
    assert first == reordered
    assert str(tmp_path) not in json.dumps(first, sort_keys=True)

    variants = [
        {**configuration, "int": np.int32(5)},
        {**configuration, "float": np.float64(1.5)},
        {**configuration, "type": np.float32},
        {**configuration, "ufunc": np.square},
        {**configuration, "date": datetime.date(2024, 3, 1)},
        {**configuration, "datetime": datetime.datetime(2024, 2, 29, 12, 1, 2, 4)},
        {**configuration, "namedtuple_class": OtherPair},
        {**configuration, "namedtuple": OtherPair(1, 2)},
        {**configuration, "namedtuple": (1, 2)},
        {**configuration, "enum_class": OtherMode},
        {**configuration, "enum": Mode.SECOND},
        {**configuration, "partial": functools.partial(helper, 4, scale=4)},
        {**configuration, "partial": functools.partial(helper, 3, scale=5)},
        {**configuration, "nested_class": OtherNested},
        {**configuration, "nested": OtherNested(7)},
    ]
    for changed in variants:
        assert (
            first["descriptor_sha256"]
            != model_descriptor(Model(changed))["descriptor_sha256"]
        )

    path.write_text("changed content", encoding="utf-8")
    assert (
        first["descriptor_sha256"]
        != model_descriptor(Model(configuration))["descriptor_sha256"]
    )


def test_custom_subclass_and_instance_apply_override_require_contract():
    table = make_bias_table([{"obs_code": "500", "chi2_per_obs": 1.0}])

    class Subclass(PerformanceWeightedModel):
        pass

    with pytest.raises(TypeError, match="must implement model_provenance"):
        model_descriptor(Subclass(table))

    instance = PerformanceWeightedModel(table)
    instance.apply = lambda observations: observations
    with pytest.raises(TypeError, match="must implement model_provenance"):
        model_descriptor(instance)


@pytest.mark.parametrize("module_name", ["__main__", None])
def test_unsupported_dynamic_model_has_documented_type_error(module_name):
    class Model:
        def apply(self, observations):
            return observations

    Model.__module__ = module_name
    with pytest.raises(TypeError, match="custom observation model|module identity"):
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
    iod = {"name": "iod-native", "version": "1", "abi_version": 2}
    refinement = {"name": "refinement-native", "version": "2", "abi_version": 2}
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


def test_preflight_describes_contract_before_apply_or_fit():
    apply_calls = 0
    contract_calls = 0
    fit_calls = 0

    class Model:
        def apply(self, observations):
            nonlocal apply_calls
            apply_calls += 1
            return observations

        def model_provenance(self):
            nonlocal contract_calls
            contract_calls += 1
            raise TypeError("deliberate contract refusal")

    class CountingFitter:
        def full_od(self, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            raise AssertionError("fit must not run before model provenance preflight")

    with pytest.raises(TypeError, match="deliberate contract refusal"):
        run_od(
            make_synthetic_observations(),
            CountingFitter(),
            [Model()],
            propagator=TwoBodyPropagator(),
        )
    assert contract_calls == 1
    assert apply_calls == 0
    assert fit_calls == 0


def test_preflight_rejects_native_python_version_mismatch_before_model_work(
    monkeypatch,
):
    contract_calls = 0
    apply_calls = 0

    class Model(_ExplicitModel):
        def model_provenance(self):
            nonlocal contract_calls
            contract_calls += 1
            return super().model_provenance()

        def apply(self, observations):
            nonlocal apply_calls
            apply_calls += 1
            return observations

    monkeypatch.setattr(_rust_native, "build_identity", lambda: ("999.0.0", "0" * 40))
    with pytest.raises(RuntimeError, match="does not match Python"):
        run_od(
            make_synthetic_observations(),
            object(),
            [Model()],
            propagator=TwoBodyPropagator(),
        )
    assert contract_calls == 0
    assert apply_calls == 0


def test_runtime_untracked_source_guard_precedes_models(monkeypatch):
    import adam_core.orbit_determination.provenance as provenance

    contract_calls = 0

    class Model(_ExplicitModel):
        def model_provenance(self):
            nonlocal contract_calls
            contract_calls += 1
            return super().model_provenance()

    monkeypatch.setattr(
        provenance,
        "_relevant_untracked_source",
        lambda: ("src/adam_core/relevant_untracked.py",),
    )
    with pytest.raises(RuntimeError, match="relevant_untracked.py"):
        provenance._preflight_provenance(object(), [Model()])
    assert contract_calls == 0
