"""Run-level OD provenance (`OdProvenance`) on both run_od routes."""

import json

import pytest

import adam_core
from adam_core.orbit_determination import NativeOrbitFitter, OdProvenance
from adam_core.orbit_determination.observation_uncertainty import (
    PerformanceWeightedModel,
)
from adam_core.orbit_determination.od_orchestration import run_od
from adam_core.orbit_determination.provenance import (
    backend_identity,
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


@pytest.mark.parametrize(
    "propagator_class, route",
    [(TwoBodyPropagator, "composed"), (NativeTwoBodyPropagator, "fused")],
)
def test_run_od_records_provenance(propagator_class, route):
    observations = make_synthetic_observations()
    model = PerformanceWeightedModel(
        make_bias_table([{"obs_code": "500", "chi2_per_obs": 1.0}])
    )
    orbits, members = run_od(
        observations,
        NativeOrbitFitter(propagator_class),
        [model],
        propagator=propagator_class(),
    )
    record = _record(orbits)
    assert record["adam_core_version"] == adam_core.__version__
    assert record["route"] == route
    assert record["fitter"].endswith("NativeOrbitFitter")
    assert record["refinement_backend"] == backend_identity(propagator_class())
    assert record["iod_backend"] == record["refinement_backend"]
    settings = json.loads(record["settings"])
    assert set(settings) == {"iod", "refinement", "fit"}
    assert settings["iod"]["acceptance"] == "corrected"
    (descriptor,) = json.loads(record["models"])
    assert descriptor["type"].endswith("PerformanceWeightedModel")
    assert all(len(h) == 64 for h in descriptor["tables_sha256"].values())
    # The IOD's ambiguity reaches the final orbit on both routes.
    assert record["iod_ambiguous"] is False and record["iod_alternatives"] == 0
    rejected = sum(members.outlier.to_pylist())
    assert record["num_rejected"] == rejected
    assert record["rejected_fraction"] == pytest.approx(rejected / len(members))


def test_backend_identity_names_declared_and_python_backends():
    assert backend_identity(NativeTwoBodyPropagator()) == (
        "adam_core.tests.native_two_body 0"
    )
    assert backend_identity(NativeTwoBodyPropagator) == backend_identity(
        NativeTwoBodyPropagator()
    )
    python = backend_identity(TwoBodyPropagator())
    assert python.startswith("python:") and "TwoBodyPropagator" in python
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
        NativeOrbitFitter(TwoBodyPropagator),
        None,
        propagator=TwoBodyPropagator(),
    )
    iod = with_iod_provenance(orbits, [4, 9])
    assert _record(iod)["iod_ambiguous"] is True
    assert _record(iod)["iod_alternatives"] == 2
    refined = orbits.set_column("provenance", OdProvenance.nulls(len(orbits)))
    carried = carry_iod_provenance(iod, refined)
    assert _record(carried)["iod_ambiguous"] is True
    assert _record(carried)["iod_alternatives"] == 2
    # A refinement that recorded its own IOD fields keeps them.
    assert carry_iod_provenance(iod, orbits) is orbits
    assert _record(orbits)["iod_ambiguous"] is False
