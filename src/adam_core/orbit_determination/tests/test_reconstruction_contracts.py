"""Permanent adversarial oracles recovered from the 51ba61ef review."""

from __future__ import annotations

import subprocess
import warnings
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
    refine,
    run_iod,
)
from adam_core.orbit_determination.observation_uncertainty import (
    EFCC18DebiasModel,
    IdentityModel,
    ObservationUncertaintyModel,
    PerformanceWeightedModel,
)
from adam_core.orbit_determination.od_orchestration import run_od
from adam_core.orbit_determination.provenance import model_descriptor
from adam_core.orbit_determination.tests.test_differential_correction import (
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)
from adam_core.orbit_determination.tests.test_observation_uncertainty import (
    make_bias_table,
)
from adam_core.propagator.propagator import NativeBackend

REPO = str(Path(__file__).resolve().parents[4])


def _record(orbits) -> dict:
    return {
        n: orbits.provenance.column(n)[0].as_py() for n in OdProvenance.schema.names
    }


class PythonAppliedModel(PerformanceWeightedModel):
    def apply(self, observations):
        return super().apply(observations)

    def model_provenance(self):
        return ModelProvenance("adam-core-tests", "1", "2" * 64, dict(vars(self)))


class ExplicitStageComposition(NativeOrbitFitter):
    def full_od(self, object_id, observations, propagator):
        return super().full_od(object_id, observations, propagator)


def _fitter(**kw):
    return NativeOrbitFitter(TwoBodyPropagator(), iod_rchi2_threshold=1e6, **kw)


def _run(obs, models=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_od(
            obs, _fitter(**kw), models, propagator=TwoBodyPropagator(), object_id="x"
        )


# ------------------------------------------------------------------ build identity


def test_build_identity_is_exact_clean_head():
    head = subprocess.check_output(
        ["git", "-C", REPO, "rev-parse", "HEAD"], text=True
    ).strip()
    assert _rust_native.build_identity() == (adam_core.__version__, head)
    orbits, _ = _run(make_synthetic_observations())
    assert _record(orbits)["adam_core_build"] == head


# ------------------------------------------------------------------ input identity


def _set_np(obs, path, transform, dtype=None):
    parts = path.split(".")
    node = obs
    for p in parts:
        node = getattr(node, p)
    values = node.to_numpy(zero_copy_only=False).copy()
    values = transform(values)
    return obs.set_column(path, pa.array(values, type=dtype))


INPUT_MUTATIONS = {
    "observed time +1ns": lambda o: _set_np(
        o,
        "coordinates.time.nanos",
        lambda v: np.where(np.arange(len(v)) == 3, v + 1, v),
        pa.int64(),
    ),
    "observer time +1ns": lambda o: _set_np(
        o,
        "observers.coordinates.time.nanos",
        lambda v: np.where(np.arange(len(v)) == 3, v + 1, v),
        pa.int64(),
    ),
    "observer x +1e-12 au": lambda o: _set_np(
        o,
        "observers.coordinates.x",
        lambda v: np.where(np.arange(len(v)) == 5, v + 1e-12, v),
    ),
    "lon +1 ulp": lambda o: _set_np(
        o,
        "coordinates.lon",
        lambda v: np.where(np.arange(len(v)) == 2, np.nextafter(v, np.inf), v),
    ),
    "row order swap": lambda o: o.take(pa.array([1, 0] + list(range(2, len(o))))),
    "obs id rename": lambda o: o.set_column(
        "id", pa.array([f"z{i}" for i in range(len(o))], type=o.id.type)
    ),
}


def _covariance_tweak(o):
    from adam_core.coordinates.covariances import CoordinateCovariances

    cov = o.coordinates.covariance.to_matrix().copy()
    cov[4, 1, 1] *= 1.0 + 1e-12
    return o.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(cov)
    )


INPUT_MUTATIONS["lon variance *(1+1e-12)"] = _covariance_tweak


@pytest.mark.parametrize("name", sorted(INPUT_MUTATIONS))
def test_input_fingerprint_is_sensitive(name):
    obs = make_synthetic_observations()
    base = _record(_run(obs)[0])["input_fingerprint"]
    other = _record(_run(INPUT_MUTATIONS[name](obs))[0])["input_fingerprint"]
    assert base != other, name


def test_input_fingerprint_is_layout_canonical():
    """Logically identical inputs with different Arrow chunking must hash the same."""
    from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations

    obs = make_synthetic_observations()
    chunked = OrbitDeterminationObservations.from_pyarrow(
        pa.concat_tables([obs[:7].table, obs[7:].table])
    )
    assert chunked.table.column("id").num_chunks >= 2
    a = _record(_run(obs)[0])
    b = _record(_run(chunked)[0])
    assert (
        a["input_fingerprint"] == b["input_fingerprint"]
    ), "same content, different hash"


# ------------------------------------------------------------------ fused == composed


def test_fused_and_composed_share_native_model_science_and_provenance():
    obs = make_synthetic_observations()
    table = make_bias_table([{"obs_code": "500", "chi2_per_obs": 1.0}])
    fused, fused_members = _run(obs, [PerformanceWeightedModel(table)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        composed, composed_members = run_od(
            obs,
            ExplicitStageComposition(TwoBodyPropagator(), iod_rchi2_threshold=1e6),
            [PerformanceWeightedModel(table)],
            propagator=TwoBodyPropagator(),
            object_id="x",
        )
    f = _record(fused)
    c = _record(composed)
    assert f["route"] == "fused" and c["route"] == "composed"
    for field in (
        "adam_core_version",
        "adam_core_build",
        "input_fingerprint",
        "iod_backend",
        "refinement_backend",
        "settings",
        "models",
        "iod_ambiguous",
        "iod_alternatives",
        "num_rejected",
        "rejected_fraction",
    ):
        assert f[field] == c[field], field
    np.testing.assert_allclose(
        fused.coordinates.values,
        composed.coordinates.values,
        rtol=1e-12,
        atol=1e-12,
    )
    assert fused_members.outlier.to_pylist() == composed_members.outlier.to_pylist()


def test_explicit_stage_composition_route_completes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orbits, _ = run_od(
            make_synthetic_observations(),
            ExplicitStageComposition(TwoBodyPropagator(), iod_rchi2_threshold=1e6),
            None,
            propagator=TwoBodyPropagator(),
        )
    assert len(orbits) == 1


# ---------------------------------------------------------- rank-deficient arcs


def _two_epoch_arc():
    observations = make_synthetic_observations().take(pa.array([0, 0, 0, 1, 1, 1]))
    return observations.set_column(
        "id",
        pa.array(
            [f"two-epoch-{row}" for row in range(len(observations))],
            type=observations.table.schema.field("id").type,
        ),
    )


def test_two_distinct_epoch_arcs_are_refused_on_all_routes():
    observations = _two_epoch_arc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orbit, _, report = run_iod(
            observations,
            TwoBodyPropagator(),
            min_obs=6,
            min_arc_length=0.0,
        )
    assert len(orbit) == 0
    assert not report.found
    assert "2 distinct observation epochs" in report.not_found_reason

    for rejection in ("none", "worst_residual", "cmc2003"):
        with (
            warnings.catch_warnings(),
            pytest.raises(ValueError, match="2 distinct observation epochs"),
        ):
            warnings.simplefilter("ignore")
            refine(
                make_initial_guess(),
                observations,
                TwoBodyPropagator(),
                rejection=rejection,
            )

    routes = (
        (_fitter(min_arc_length=0.0), None),
        (
            ExplicitStageComposition(
                TwoBodyPropagator(),
                iod_rchi2_threshold=1e6,
                min_arc_length=0.0,
            ),
            [IdentityModel()],
        ),
    )
    for route_fitter, models in routes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted, members = run_od(
                observations,
                route_fitter,
                models,
                propagator=TwoBodyPropagator(),
            )
        assert len(fitted) == 0
        assert len(members) == 0


# ------------------------------------------------------------------ models


def test_model_descriptor_numpy_variants():
    class ArrayModel(ObservationUncertaintyModel):
        def __init__(self, values):
            self.values = values

        def apply(self, o):
            return o

        def model_provenance(self):
            return ModelProvenance(
                "adam-core-tests", "1", "4" * 64, {"values": self.values}
            )

    def sha(v):
        return model_descriptor(ArrayModel(v))["descriptor_sha256"]

    # Equal all-zero payload bytes make dtype and shape independently material.
    base = np.zeros((2, 3), dtype=np.float64)
    variants = {
        "f32": base.astype(np.float32),
        "i8": base.astype(np.int64),
        "shape": base.reshape(3, 2),
        "1ulp": np.where(np.arange(6).reshape(2, 3) == 5, np.nextafter(0.0, 1.0), base),
        "-0.0": np.where(base == 0, -0.0, base),
    }
    same = {
        "big-endian": base.astype(">f8"),
        "fortran": np.asfortranarray(base),
        "strided": np.arange(12, dtype=np.float64).reshape(2, 6)[:, ::2] * 0 + base,
    }
    ref = sha(base)
    assert not [k for k, v in variants.items() if sha(v) == ref]
    assert not [k for k, v in same.items() if sha(v) != ref]


def test_efcc_descriptor_content_version_exclusions():
    table = np.zeros((49152, 26, 4))
    a = model_descriptor(EFCC18DebiasModel(bias_table=table))
    t2 = table.copy()
    t2[123, 4, 2] = 1e-9
    b = model_descriptor(EFCC18DebiasModel(bias_table=t2))
    c = model_descriptor(EFCC18DebiasModel(bias_table=table, exclude_astcats=["V"]))
    unordered_model = EFCC18DebiasModel(
        bias_table=table,
        exclude_astcats={"V", "a", "b", "c"},
    )
    ordered_model = EFCC18DebiasModel(
        bias_table=table,
        exclude_astcats=["V", "a", "b", "c"],
    )
    assert unordered_model.exclude_astcats == tuple(sorted({"V", "a", "b", "c"}))
    assert model_descriptor(unordered_model) == model_descriptor(ordered_model)
    assert a["implementation_version"] == adam_core.__version__
    assert a["descriptor_sha256"] != b["descriptor_sha256"] != c["descriptor_sha256"]
    assert a["descriptor_sha256"] != c["descriptor_sha256"]


def test_provider_metadata_comes_from_capsule_not_python_declaration():
    class Declared(TwoBodyPropagator):
        native_backend = NativeBackend(
            name="declared-name",
            version="9.9.9",
            work_units=frozenset({"propagator_capsule"}),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            orbits, _ = run_od(
                make_synthetic_observations(),
                NativeOrbitFitter(Declared(), iod_rchi2_threshold=1e6),
                None,
                propagator=Declared(),
            )
        except Exception as error:  # identity mismatch must be rejected by Rust
            assert (
                "declared-name" in str(error) or "identity" in str(error).lower()
            ), error
            return
    rec = _record(orbits)
    assert "declared-name" not in rec["refinement_provider"]


pytestmark = pytest.mark.usefixtures("allow_synthetic_model_distribution")
