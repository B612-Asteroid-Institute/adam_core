"""Run-level OD provenance (`OdProvenance`) on both run_od routes."""

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
        dict((name, index) for index, name in enumerate(names)),
    )
    reverse = Model(
        set(Key(name) for name in reversed(names)),
        dict((name, names.index(name)) for name in reversed(names)),
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
