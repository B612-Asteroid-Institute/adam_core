"""Run-level OD provenance (`OdProvenance`) on both run_od routes."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import adam_core
import adam_core.orbit_determination.provenance as provenance_module
from adam_core import _rust_native
from adam_core.orbit_determination import (
    ModelProvenance,
    NativeOrbitFitter,
    OdProvenance,
)
from adam_core.orbit_determination.observation_uncertainty import (
    ObservationUncertaintyModel,
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
_REAL_CUSTOM_CONTRACT = provenance_module._custom_contract


class _ExplicitModel(ObservationUncertaintyModel):
    """Test helper implementing the public custom-model contract."""

    implementation_fingerprint = _TEST_IMPLEMENTATION

    def apply(self, observations):
        return observations

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
        configuration = descriptor["parameters"]
        assert configuration["tag"] == "mapping"
        bias_table = next(
            value
            for key, value in configuration["entries"]
            if key == {"tag": "str", "value": "bias_table"}
        )
        assert bias_table["tag"] == "arrow-table"
        assert len(bias_table["sha256"]) == 64
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

    repository = Path(__file__).resolve().parents[4]
    example = repository / "migration/examples/acme_orbit_models"
    installed = tmp_path / "installed-example"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(installed),
            str(example),
        ],
        check=True,
    )
    script = (
        "{imports}; from acme_orbit_models import CalibratedModel as Model; "
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
                    str(installed),
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
    class Model(ObservationUncertaintyModel):
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
        "partial": functools.partial(sorted, [3, 1], reverse=True),
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
        {**configuration, "partial": functools.partial(sorted, [4, 1], reverse=True)},
        {**configuration, "partial": functools.partial(sorted, [3, 1], reverse=False)},
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

    class Model(ObservationUncertaintyModel):
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
        "_source_checkout_problems",
        lambda build_commit: (
            "untracked package Python source: src/adam_core/relevant_untracked.py",
        ),
    )
    with pytest.raises(RuntimeError, match="relevant_untracked.py"):
        provenance._preflight_provenance(object(), [Model()])
    assert contract_calls == 0


def test_builtin_behavior_overrides_and_nested_overrides_require_contract(monkeypatch):
    from adam_core.orbit_determination.observation_uncertainty import (
        CompositeModel,
        NightBatchDeweightingModel,
    )

    instance = NightBatchDeweightingModel(cap=2)
    instance._native_specs = lambda: [{"model": "night_batch", "cap": 8}]
    with pytest.raises(TypeError, match="model_provenance"):
        model_descriptor(instance)
    with pytest.raises(TypeError, match="model_provenance"):
        model_descriptor(CompositeModel(instance))

    monkeypatch.setattr(
        NightBatchDeweightingModel,
        "_native_specs",
        lambda self: [{"model": "night_batch", "cap": 1}],
    )
    with pytest.raises(TypeError, match="model_provenance"):
        model_descriptor(NightBatchDeweightingModel(cap=2))


def _assert_run_od_preflight_refuses_model(
    model, monkeypatch, match="model_provenance"
):
    monkeypatch.setattr(provenance_module, "_custom_contract", _REAL_CUSTOM_CONTRACT)
    fit_calls = 0
    provider_calls = 0

    class Fitter:
        def full_od(self, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            raise AssertionError("fitter must not run before provenance refusal")

    class Provider:
        def __getattribute__(self, name):
            nonlocal provider_calls
            if name.startswith("__"):
                return object.__getattribute__(self, name)
            provider_calls += 1
            raise AssertionError("provider must not run before provenance refusal")

    with pytest.raises(TypeError, match=match):
        run_od(
            make_synthetic_observations(),
            Fitter(),
            [model],
            propagator=Provider(),
        )
    assert fit_calls == 0
    assert provider_calls == 0


def test_exact_class_special_lookup_hooks_are_refused_before_execution(monkeypatch):
    from adam_core.orbit_determination.observation_uncertainty import IdentityModel

    monkeypatch.setattr(provenance_module, "_custom_contract", _REAL_CUSTOM_CONTRACT)
    for special in ("__getattribute__", "__getattr__"):
        calls = 0
        model = IdentityModel()
        baseline = model_descriptor(model)["descriptor_sha256"]

        def changed(self, name, _special=special):
            nonlocal calls
            calls += 1
            if _special == "__getattribute__":
                return object.__getattribute__(self, name)
            raise AttributeError(name)

        with monkeypatch.context() as context:
            context.setattr(IdentityModel, special, changed, raising=False)
            with pytest.raises(TypeError, match="model_provenance"):
                model_descriptor(model)
            _assert_run_od_preflight_refuses_model(model, context)
        assert calls == 0
        assert model_descriptor(model)["descriptor_sha256"] == baseline


def test_nested_lookup_special_resolution_hook_is_refused_before_execution(monkeypatch):
    from adam_core.orbit_determination.tests.test_veres2017 import SMALL_TABLE
    from adam_core.orbit_determination.veres2017 import (
        SigmaFillModel,
        VeresSigmaLookup,
    )

    calls = 0
    model = SigmaFillModel(SMALL_TABLE)

    def changed(self, name):
        nonlocal calls
        calls += 1
        return object.__getattribute__(self, name)

    monkeypatch.setattr(VeresSigmaLookup, "__getattribute__", changed)
    with pytest.raises(TypeError, match="model_provenance"):
        model_descriptor(model)
    _assert_run_od_preflight_refuses_model(model, monkeypatch)
    assert calls == 0


def test_composite_native_model_specs_delegate_mutation_is_refused(monkeypatch):
    import adam_core.orbit_determination.observation_uncertainty as uncertainty
    from adam_core.orbit_determination.observation_uncertainty import (
        CompositeModel,
        IdentityModel,
    )

    calls = 0
    model = CompositeModel(IdentityModel())
    baseline = model_descriptor(model)["descriptor_sha256"]

    def changed(models):
        nonlocal calls
        calls += 1
        return [{"model": "identity"}]

    monkeypatch.setattr(uncertainty, "native_model_specs", changed)
    with pytest.raises(TypeError, match="model_provenance"):
        model_descriptor(model)
    _assert_run_od_preflight_refuses_model(model, monkeypatch)
    assert calls == 0
    assert baseline


def test_bias_module_delegate_mutation_is_refused_before_execution(monkeypatch):
    import adam_core.orbit_determination.observation_uncertainty as uncertainty
    from adam_core.orbit_determination.observation_uncertainty import (
        PerformanceWeightedModel,
    )

    calls = 0
    model = PerformanceWeightedModel(
        make_bias_table([{"obs_code": "500", "chi2_per_obs": 2.0}])
    )

    def changed(observations):
        nonlocal calls
        calls += 1
        raise AssertionError("delegate executed")

    monkeypatch.setattr(uncertainty, "_covariances", changed)
    with pytest.raises(TypeError, match="model_provenance"):
        model_descriptor(model)
    _assert_run_od_preflight_refuses_model(model, monkeypatch)
    assert calls == 0


def test_native_kernel_delegate_mutation_is_refused_before_execution(monkeypatch):
    from adam_core import _rust_native
    from adam_core.orbit_determination.tests.test_veres2017 import SMALL_TABLE
    from adam_core.orbit_determination.veres2017 import SigmaFillModel

    calls = 0
    model = SigmaFillModel(SMALL_TABLE)

    def changed(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("kernel delegate executed")

    monkeypatch.setattr(_rust_native, "veres_model_apply_numpy", changed)
    with pytest.raises(TypeError, match="model_provenance"):
        model_descriptor(model)
    _assert_run_od_preflight_refuses_model(model, monkeypatch)
    assert calls == 0


def test_sigma_fill_model_constant_monkeypatch_is_custom_on_real_route(monkeypatch):
    """Exact fallback-audit reproducer: fill must not retain built-in identity."""
    from adam_core.orbit_determination.tests.test_veres2017 import SMALL_TABLE
    from adam_core.orbit_determination.veres2017 import SigmaFillModel

    model = SigmaFillModel(SMALL_TABLE)
    original = model_descriptor(model)
    monkeypatch.setattr(SigmaFillModel, "_MODEL", "replace")
    assert model._native_specs()[0]["kind"] == "replace"
    with pytest.raises(TypeError, match="model_provenance"):
        model_descriptor(model)
    assert original["implementation"]["kind"] == "adam-core-built-in-v1"
    _assert_run_od_preflight_refuses_model(model, monkeypatch)


def test_veres_lookup_payload_monkeypatch_is_custom_on_real_route(monkeypatch):
    """Exact fallback-audit reproducer: nested payload routing is attested."""
    from adam_core.orbit_determination.tests.test_veres2017 import SMALL_TABLE
    from adam_core.orbit_determination.veres2017 import (
        SigmaFillModel,
        VeresSigmaLookup,
    )

    model = SigmaFillModel(SMALL_TABLE)
    original = VeresSigmaLookup._table_arguments

    def replaced_arguments(self):
        arguments = original(self)
        return (*arguments[:-1], 19.0)

    monkeypatch.setattr(VeresSigmaLookup, "_table_arguments", replaced_arguments)
    assert model._native_specs()[0]["fallback_sigma_arcsec"] == 19.0
    with pytest.raises(TypeError, match="model_provenance"):
        model_descriptor(model)
    _assert_run_od_preflight_refuses_model(model, monkeypatch)


@pytest.mark.parametrize(
    ("owner_name", "member", "replacement"),
    [
        ("SigmaFillModel", "_DEFAULT_TABLE", "veres2017_working"),
        ("VeresSigmaLookup", "sigmas_for", lambda self, codes, catalogs: None),
        ("VeresSigmaLookup", "sigmas", lambda self, code, catalog: (9.0, 9.0)),
    ],
)
def test_veres_registered_class_mutations_are_custom(
    monkeypatch, owner_name, member, replacement
):
    from adam_core.orbit_determination.tests.test_veres2017 import SMALL_TABLE
    from adam_core.orbit_determination.veres2017 import (
        SigmaFillModel,
        VeresSigmaLookup,
    )

    model = SigmaFillModel(SMALL_TABLE)
    owner = {"SigmaFillModel": SigmaFillModel, "VeresSigmaLookup": VeresSigmaLookup}[
        owner_name
    ]
    monkeypatch.setattr(owner, member, replacement)
    _assert_run_od_preflight_refuses_model(model, monkeypatch)


@pytest.mark.parametrize(
    ("owner_name", "member"),
    [
        ("SigmaFillModel", "lookup"),
        ("SigmaFillModel", "fill_missing"),
        ("VeresSigmaLookup", "fallback_sigma_arcsec"),
    ],
)
def test_registered_instance_data_cannot_be_shadowed_on_the_class(
    monkeypatch, owner_name, member
):
    from adam_core.orbit_determination.tests.test_veres2017 import SMALL_TABLE
    from adam_core.orbit_determination.veres2017 import (
        SigmaFillModel,
        VeresSigmaLookup,
    )

    model = SigmaFillModel(SMALL_TABLE)
    owner = {"SigmaFillModel": SigmaFillModel, "VeresSigmaLookup": VeresSigmaLookup}[
        owner_name
    ]
    monkeypatch.setattr(owner, member, property(lambda self: None), raising=False)
    _assert_run_od_preflight_refuses_model(model, monkeypatch)


def test_bias_row_columns_monkeypatch_is_custom_on_real_route(monkeypatch):
    from adam_core.orbit_determination.observation_uncertainty import (
        PerformanceWeightedModel,
    )

    model = PerformanceWeightedModel(
        make_bias_table([{"obs_code": "500", "chi2_per_obs": 2.0}])
    )
    monkeypatch.setattr(
        PerformanceWeightedModel,
        "_ROW_COLUMNS",
        tuple(reversed(PerformanceWeightedModel._ROW_COLUMNS)),
    )
    _assert_run_od_preflight_refuses_model(model, monkeypatch)


def test_registered_instance_shadows_and_nested_subclass_are_custom(monkeypatch):
    from adam_core.orbit_determination.tests.test_veres2017 import SMALL_TABLE
    from adam_core.orbit_determination.veres2017 import (
        SigmaFillModel,
        VeresSigmaLookup,
    )

    for mutate in (
        lambda model: setattr(model, "_MODEL", "replace"),
        lambda model: setattr(model, "_DEFAULT_TABLE", "veres2017_working"),
        lambda model: setattr(
            model.lookup, "_table_arguments", lambda: ([], [], [], [], 3.0)
        ),
    ):
        model = SigmaFillModel(SMALL_TABLE)
        mutate(model)
        with pytest.raises(TypeError, match="model_provenance"):
            model_descriptor(model)

    class LookupSubclass(VeresSigmaLookup):
        pass

    model = SigmaFillModel(SMALL_TABLE)
    model.lookup = LookupSubclass(SMALL_TABLE)
    _assert_run_od_preflight_refuses_model(model, monkeypatch)


def test_nested_custom_model_contract_is_recursive_and_sensitive():
    from adam_core.orbit_determination.observation_uncertainty import CompositeModel

    class Custom(_ExplicitModel):
        def __init__(self, fingerprint, scale):
            self.implementation_fingerprint = fingerprint
            self.scale = scale

    first = model_descriptor(CompositeModel(Custom("1" * 64, 2)))
    science = model_descriptor(CompositeModel(Custom("2" * 64, 2)))
    config = model_descriptor(CompositeModel(Custom("1" * 64, 3)))
    nested = first["models"][0]
    assert nested["implementation"]["implementation_fingerprint"] == "1" * 64
    assert first["descriptor_sha256"] != science["descriptor_sha256"]
    assert first["descriptor_sha256"] != config["descriptor_sha256"]


def test_canonical_configuration_is_injective_and_user_mappings_cannot_forge_tags():
    class Model(_ExplicitModel):
        def __init__(self, configuration):
            self.configuration = configuration

    values = [
        [1, 2],
        (1, 2),
        {1, 2},
        frozenset({1, 2}),
        float("nan"),
        {"tag": "float64", "value": "nan"},
        np.dtype("<f8"),
        {"tag": "numpy-dtype", "value": "<f8"},
        {1: 2},
        {"tag": "mapping", "entries": [[1, 2]]},
    ]
    hashes = [
        model_descriptor(Model({"value": value}))["descriptor_sha256"]
        for value in values
    ]
    assert len(set(hashes)) == len(hashes)


def test_python_callable_configuration_requires_explicit_portable_data():
    class Model(_ExplicitModel):
        def __init__(self, function):
            self.function = function

    with pytest.raises(TypeError, match="explicit portable identity"):
        model_descriptor(Model(lambda value: value))


def test_public_pathlib_class_identity_is_normalized_across_private_relocation():
    from pathlib import PurePosixPath

    from adam_core.orbit_determination.provenance import _named_identity

    relocated = type("PurePosixPath", (), {})
    relocated.__module__ = "pathlib._local"
    assert _named_identity(relocated, kind="class-v1") == _named_identity(
        PurePosixPath, kind="class-v1"
    )


def test_contracted_moduleless_model_is_rejected():
    class Model(_ExplicitModel):
        pass

    Model.__module__ = None
    with pytest.raises(TypeError, match="module identity"):
        model_descriptor(Model())


def _write_installed_distribution(root, name, version, files, *, direct_url=None):
    normalized = name.replace("-", "_")
    dist_info = root / f"{normalized}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    records = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        records.append(f"{relative},,")
    if direct_url is not None:
        (dist_info / "direct_url.json").write_text(direct_url, encoding="utf-8")
        records.append(f"{dist_info.name}/direct_url.json,,")
    records.extend(
        [
            f"{dist_info.name}/METADATA,,",
            f"{dist_info.name}/RECORD,,",
        ]
    )
    (dist_info / "RECORD").write_text("\n".join(records) + "\n", encoding="utf-8")


def _dynamic_contracted_model(module, package, version, counters):
    def apply(self, observations):
        counters["apply"] += 1
        return observations

    def model_provenance(self):
        counters["contract"] += 1
        return ModelProvenance(package, version, "d" * 64, {"mode": "audit"})

    return type(
        "DynamicModel",
        (ObservationUncertaintyModel,),
        {
            "__module__": module,
            "__qualname__": "DynamicModel",
            "apply": apply,
            "model_provenance": model_provenance,
        },
    )


def _assert_live_python_refusal_precedes_work(model):
    counters = model._audit_counters

    class Fitter:
        def full_od(self, *args, **kwargs):
            counters["fit"] += 1
            raise AssertionError("fitter ran before custom-model refusal")

    class Provider:
        def __getattribute__(self, name):
            if name.startswith("__"):
                return object.__getattribute__(self, name)
            counters["provider"] += 1
            raise AssertionError("provider ran before custom-model refusal")

    with pytest.raises(TypeError, match="live Python behavior"):
        run_od(
            make_synthetic_observations(),
            Fitter(),
            [model],
            propagator=Provider(),
        )
    # No user provenance hook is executed: even that hook is mutable Python.
    assert counters == {"contract": 0, "apply": 0, "fit": 0, "provider": 0}


def _restore_real_custom_refusal(monkeypatch, allow_synthetic_model_distribution):
    import adam_core.orbit_determination.provenance as provenance

    monkeypatch.setattr(
        provenance, "_custom_contract", allow_synthetic_model_distribution
    )


def test_main_dynamic_nonexistent_contract_is_rejected_before_all_work(
    monkeypatch, allow_synthetic_model_distribution
):
    """Exact fallback-audit reproducer is refused without running its hook."""
    _restore_real_custom_refusal(monkeypatch, allow_synthetic_model_distribution)
    counters = {"contract": 0, "apply": 0, "fit": 0, "provider": 0}
    cls = _dynamic_contracted_model(
        "__main__", "nonexistent-audit-distribution", "999", counters
    )
    model = cls()
    model._audit_counters = counters
    _assert_live_python_refusal_precedes_work(model)


def test_local_contracted_class_is_rejected_before_all_work(
    monkeypatch, allow_synthetic_model_distribution
):
    _restore_real_custom_refusal(monkeypatch, allow_synthetic_model_distribution)
    counters = {"contract": 0, "apply": 0, "fit": 0, "provider": 0}

    class LocalModel(ObservationUncertaintyModel):
        def apply(self, observations):
            counters["apply"] += 1
            return observations

        def model_provenance(self):
            counters["contract"] += 1
            return ModelProvenance("nonexistent-local-model", "1", "e" * 64, {})

    model = LocalModel()
    model._audit_counters = counters
    _assert_live_python_refusal_precedes_work(model)


@pytest.mark.parametrize(
    "audit_probe",
    [
        "missing-distribution",
        "wrong-version",
        "dynamic-borrowed-module-qualname",
        "sourceless-module",
        "editable-direct-url",
        "duplicate-normalized-distribution",
        "overlapping-namespace-owner",
        "symlink-source",
        "resolved-path-escape",
        "missing-record",
        "tampered-record-hash",
        "record-size-mismatch",
        "unsupported-record-hash",
    ],
)
def test_distribution_identity_probes_fail_closed_before_all_work(
    tmp_path,
    monkeypatch,
    allow_synthetic_model_distribution,
    audit_probe,
):
    """No package metadata assertion can authenticate live executable Python."""
    _restore_real_custom_refusal(monkeypatch, allow_synthetic_model_distribution)
    # Materialize representative metadata/source so these are real filesystem
    # layouts, not merely malformed contract strings. Production intentionally
    # refuses before consulting any of this mutable provider-owned state.
    files = {"audit_models/models.py": "class DynamicModel: pass\n"}
    direct_url = None
    if audit_probe == "sourceless-module":
        files = {"audit_models/models.so": "opaque\n"}
    if audit_probe == "editable-direct-url":
        direct_url = '{"url":"file:///tmp/provider","dir_info":{"editable":true}}'
    _write_installed_distribution(
        tmp_path, "audit-models", "1.0", files, direct_url=direct_url
    )
    if audit_probe in {
        "duplicate-normalized-distribution",
        "overlapping-namespace-owner",
    }:
        other = tmp_path / "other"
        _write_installed_distribution(other, "audit_models", "1.0", files)
        monkeypatch.syspath_prepend(str(other))
    if audit_probe == "symlink-source":
        source = tmp_path / "audit_models/models.py"
        target = tmp_path / "outside.py"
        target.write_text("class DynamicModel: pass\n")
        source.unlink()
        source.symlink_to(target)
    if audit_probe == "resolved-path-escape":
        (tmp_path / "audit_models/models.py").unlink(missing_ok=True)
    dist_info = tmp_path / "audit_models-1.0.dist-info"
    record = dist_info / "RECORD"
    if audit_probe == "missing-record":
        record.unlink()
    elif audit_probe == "tampered-record-hash":
        record.write_text("audit_models/models.py,sha256=AAAA,1\n")
    elif audit_probe == "record-size-mismatch":
        record.write_text("audit_models/models.py,sha256=AAAA,999\n")
    elif audit_probe == "unsupported-record-hash":
        record.write_text("audit_models/models.py,md5=AAAA,1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    counters = {"contract": 0, "apply": 0, "fit": 0, "provider": 0}
    cls = _dynamic_contracted_model(
        "audit_models.models", "audit-models", "1.0", counters
    )
    model = cls()
    model._audit_counters = counters
    _assert_live_python_refusal_precedes_work(model)


def test_genuine_installed_class_and_apply_monkeypatches_are_refused(
    tmp_path, monkeypatch, allow_synthetic_model_distribution
):
    _restore_real_custom_refusal(monkeypatch, allow_synthetic_model_distribution)
    repository = Path(__file__).resolve().parents[4]
    installed = tmp_path / "installed-example"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(installed),
            str(repository / "migration/examples/acme_orbit_models"),
        ],
        check=True,
    )
    monkeypatch.syspath_prepend(str(installed))
    from acme_orbit_models import CalibratedModel

    for mutation in ("unmodified", "class-apply", "instance-apply"):
        counters = {"contract": 0, "apply": 0, "fit": 0, "provider": 0}
        model = CalibratedModel()
        model._audit_counters = counters
        if mutation == "class-apply":
            with monkeypatch.context() as context:
                context.setattr(
                    CalibratedModel, "apply", lambda self, observations: observations
                )
                _assert_live_python_refusal_precedes_work(model)
        elif mutation == "instance-apply":
            model.apply = lambda observations: observations
            _assert_live_python_refusal_precedes_work(model)
        else:
            _assert_live_python_refusal_precedes_work(model)


def test_real_git_source_checkout_detector_is_fail_closed(tmp_path):
    from adam_core.orbit_determination.provenance import _source_checkout_problems

    # Exercise the default path resolution against this real source checkout;
    # this kills fail-open and wrong-repository-root implementations rather
    # than only testing an explicitly supplied temporary path.
    assert any(
        "does not match" in problem for problem in _source_checkout_problems("0" * 40)
    )

    repo = tmp_path / "checkout"
    (repo / "src/adam_core/orbit_determination").mkdir(parents=True)
    (repo / "rust/example/src").mkdir(parents=True)
    (repo / "src/adam_core/runtime.py").write_text("VALUE = 1\n")
    (repo / "src/adam_core/_version.py").write_text("__version__ = '0'\n")
    (repo / "rust/example/src/lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
    (repo / ".gitignore").write_text("lib/\n*_version.py\nignored.rs\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    generated_target = repo / "rust/example/target/generated/src/lib.rs"
    generated_target.parent.mkdir(parents=True)
    generated_target.write_text("pub fn generated() {}\n")
    assert _source_checkout_problems(head, repo=repo) == ()

    runtime = repo / "src/adam_core/runtime.py"
    runtime.write_text("VALUE = 2\n")
    assert any(
        "tracked/staged" in item for item in _source_checkout_problems(head, repo=repo)
    )
    subprocess.run(["git", "add", str(runtime)], cwd=repo, check=True)
    assert any(
        "tracked/staged" in item for item in _source_checkout_problems(head, repo=repo)
    )
    subprocess.run(["git", "reset", "--hard", "-q", "HEAD"], cwd=repo, check=True)
    assert _source_checkout_problems(head, repo=repo) == ()

    visible = repo / "src/adam_core/visible.py"
    visible.write_text("VALUE = 3\n")
    assert any(
        "visible.py" in item for item in _source_checkout_problems(head, repo=repo)
    )
    visible.unlink()
    ignored = repo / "src/adam_core/orbit_determination/lib/override.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("VALUE = 4\n")
    assert any(
        "lib/override.py" in item for item in _source_checkout_problems(head, repo=repo)
    )
    ignored.unlink()
    ignored_version = repo / "src/adam_core/model_version.py"
    ignored_version.write_text("VALUE = 5\n")
    assert any(
        "model_version.py" in item
        for item in _source_checkout_problems(head, repo=repo)
    )
    ignored_version.unlink()
    ignored_rust = repo / "rust/example/src/ignored.rs"
    ignored_rust.write_text("pub fn ignored() {}\n")
    subprocess.run(
        ["git", "check-ignore", "-q", str(ignored_rust)], cwd=repo, check=True
    )
    assert any(
        "ignored.rs" in item for item in _source_checkout_problems(head, repo=repo)
    )
    ignored_rust.unlink()
    assert _source_checkout_problems(head, repo=repo) == ()

    (repo / "README.md").write_text("next\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "move head"], cwd=repo, check=True)
    assert any(
        "does not match" in item for item in _source_checkout_problems(head, repo=repo)
    )

    installed = tmp_path / "installed-wheel"
    installed.mkdir()
    assert _source_checkout_problems(head, repo=installed) == ()


def test_documented_custom_model_migration_policy_is_executable(
    monkeypatch, allow_synthetic_model_distribution
):
    _restore_real_custom_refusal(monkeypatch, allow_synthetic_model_distribution)
    repository = Path(__file__).resolve().parents[4]
    guide = (repository / "docs/source/use_cases/orbit_determination.rst").read_text()
    assert "Live custom Python observation models are not accepted" in guide
    guide_start = guide.index(
        "   from adam_core.orbit_determination.observation_uncertainty import IdentityModel"
    )
    guide_end = guide.index("\n\nSee ``migration", guide_start)
    guide_code = "\n".join(
        line[3:] if line.startswith("   ") else line
        for line in guide[guide_start:guide_end].splitlines()
    )
    namespace: dict[str, object] = {}
    exec(guide_code, namespace)
    assert model_descriptor(namespace["model"])["implementation"]["kind"] == (
        "adam-core-built-in-v1"
    )

    migration = (repository / "migration/custom_model_provenance.md").read_text()
    assert "does not accept live third-party Python" in migration
    migration_code = migration.split("```python\n", 1)[1].split("\n```", 1)[0]
    migration_namespace: dict[str, object] = {}
    exec(migration_code, migration_namespace)
    assert (
        model_descriptor(migration_namespace["model"])["implementation"]["kind"]
        == "adam-core-built-in-v1"
    )


def test_public_run_od_recursively_preflights_nested_custom_models():
    from adam_core.orbit_determination.observation_uncertainty import CompositeModel

    apply_calls = 0
    fit_calls = 0

    class Uncontracted(ObservationUncertaintyModel):
        def apply(self, observations):
            nonlocal apply_calls
            apply_calls += 1
            return observations

    class Fitter:
        def full_od(self, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            raise AssertionError("fit must not run before recursive model preflight")

    with pytest.raises(TypeError, match="model_provenance"):
        run_od(
            make_synthetic_observations(),
            Fitter(),
            [CompositeModel(Uncontracted())],
            propagator=TwoBodyPropagator(),
        )
    assert apply_calls == 0
    assert fit_calls == 0


pytestmark = pytest.mark.usefixtures("allow_synthetic_model_distribution")
