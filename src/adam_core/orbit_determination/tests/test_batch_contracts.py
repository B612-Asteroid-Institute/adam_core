"""Canonical batch ordering, atomicity, provider, and identity contracts."""

import json
import subprocess

import pyarrow as pa
import pytest

from adam_core import _rust_native
from adam_core.observations import Associations, SourceCatalog

from .. import evaluate_orbits, refine, run_iod, run_od
from .test_batch_workflows import (
    _assert_exact_results,
    _config,
    _not_converging_config,
    _two_body_source,
)
from .test_differential_correction import TwoBodyPropagator


class _Counter:
    calls = 0


@pytest.fixture
def provider_counter(monkeypatch):
    original = TwoBodyPropagator.propagator_capsule
    _Counter.calls = 0

    def counted(self):
        _Counter.calls += 1
        return original(self)

    monkeypatch.setattr(TwoBodyPropagator, "propagator_capsule", counted)
    return _Counter


def test_native_build_identity_is_exact_clean_head() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert _rust_native.build_identity() == ("0.7.0", head)


def _source_and_ids(pure_iod_orbit):
    source = _two_body_source(pure_iod_orbit)
    return source, source.id.to_pylist()


def test_overlap_first_appearance_source_order_and_no_iod_placeholder(
    pure_iod_orbit, provider_counter
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    n = len(ids)
    detections = [ids[1]] + list(reversed(ids)) + ids[:3] + list(reversed(ids))
    groups = ["small"] + ["zeta"] * n + ["small"] * 3 + ["alpha"] * n
    # Remove the duplicated small/ids[1] pair while preserving first appearance.
    del detections[n + 2]
    del groups[n + 2]
    associations = Associations.from_kwargs(detection_id=detections, object_id=groups)
    orbits, members, failures = run_iod(
        source, associations, TwoBodyPropagator(), _config().for_iod()
    )
    assert orbits.object_id.to_pylist() == ["zeta", "alpha"]
    assert failures.group_id.to_pylist() == ["small"]
    assert failures.orbit_id.to_pylist() == [None]
    assert failures.code.to_pylist() == ["no_solution"]
    by_orbit = {}
    for orbit_id, obs_id in zip(
        members.orbit_id.to_pylist(), members.obs_id.to_pylist(), strict=True
    ):
        by_orbit.setdefault(orbit_id, []).append(obs_id)
    for member_ids in by_orbit.values():
        assert member_ids == [obs_id for obs_id in ids if obs_id in member_ids]
    assert provider_counter.calls == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "null_group",
        "empty_group",
        "unknown_detection",
        "duplicate_pair",
        "nonunique_source",
    ],
)
def test_malformed_associations_abort_before_factory(
    pure_iod_orbit, provider_counter, mutation
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    detections = list(ids) + list(ids)
    groups = ["g1"] * len(ids) + ["g2"] * len(ids)
    if mutation == "null_group":
        groups[-1] = None
    elif mutation == "empty_group":
        groups[-1] = ""
    elif mutation == "unknown_detection":
        detections[-1] = "missing"
    elif mutation == "duplicate_pair":
        detections[-1] = detections[-2]
    else:
        duplicate_ids = list(ids)
        duplicate_ids[-1] = duplicate_ids[0]
        source = source.set_column("id", pa.array(duplicate_ids, type=source.id.type))
    associations = Associations.from_kwargs(detection_id=detections, object_id=groups)
    with pytest.raises(ValueError):
        run_od(source, associations, TwoBodyPropagator(), _config())
    assert provider_counter.calls == 0


@pytest.mark.parametrize("route", ["refine", "evaluate"])
@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_obs",
        "duplicate_member",
        "unknown_orbit",
        "orbit_without_members",
        "null_snapshot",
    ],
)
def test_malformed_members_abort_before_factory(
    pure_iod_orbit, provider_counter, route, mutation
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    config = _config()
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    orbits, members, failures = run_od(source, associations, None, config)
    assert len(failures) == 0
    if mutation == "unknown_obs":
        values = members.obs_id.to_pylist()
        values[-1] = "missing"
        members = members.set_column(
            "obs_id", pa.array(values, type=members.obs_id.type)
        )
    elif mutation == "duplicate_member":
        members = members.take(pa.array(list(range(len(members))) + [0]))
    elif mutation == "unknown_orbit":
        values = members.orbit_id.to_pylist()
        values[-1] = "missing"
        members = members.set_column(
            "orbit_id", pa.array(values, type=members.orbit_id.type)
        )
    elif mutation == "orbit_without_members":
        extra = orbits.set_column(
            "orbit_id", pa.array(["other"], type=orbits.orbit_id.type)
        )
        orbits = type(orbits).from_pyarrow(
            pa.concat_tables([orbits.table, extra.table]).combine_chunks()
        )
    else:
        values = members.used_astrometry.lon.to_pylist()
        values[0] = None
        members = members.set_column(
            "used_astrometry.lon", pa.array(values, type=pa.float64())
        )
    if route == "refine":

        def call():
            return refine(
                orbits,
                members,
                source,
                TwoBodyPropagator(),
                config.for_refinement(),
            )

    else:

        def call():
            return evaluate_orbits(
                orbits,
                members,
                source,
                TwoBodyPropagator(),
                config.for_evaluation(),
            )

    with pytest.raises(ValueError):
        call()
    assert provider_counter.calls == 0


def test_input_fingerprint_is_independent_of_arrow_chunk_layout(
    pure_iod_orbit,
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    split = len(source) // 2
    chunked = SourceCatalog.from_pyarrow(
        pa.concat_tables([source.table.slice(0, split), source.table.slice(split)])
    )
    assert chunked.table.column("id").num_chunks == 2
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    contiguous_result = run_iod(source, associations, None, config.for_iod())
    chunked_result = run_iod(chunked, associations, None, config.for_iod())
    _assert_exact_results(contiguous_result, chunked_result)


def test_provider_factory_counts_and_external_exact_composition(
    pure_iod_orbit, provider_counter
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    propagator = TwoBodyPropagator()
    iod = run_iod(source, associations, propagator, config.for_iod())
    composed = refine(iod[0], iod[1], source, propagator, config.for_refinement())
    fused = run_od(source, associations, propagator, config)
    _assert_exact_results(fused, composed)
    assert provider_counter.calls == 3


def test_direct_iod_output_is_statically_valid_for_evaluation(
    pure_iod_orbit, provider_counter
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    orbits, members, failures = run_iod(source, associations, None, config.for_iod())
    assert len(failures) == 0
    evaluated = evaluate_orbits(
        orbits,
        members,
        source,
        TwoBodyPropagator(),
        config.for_evaluation(),
    )
    assert [len(table) for table in evaluated] == [1, len(source), 0]
    assert evaluated[0].success.to_pylist() == [True]
    assert provider_counter.calls == 1


def test_explicit_evaluation_subset_and_stage_provenance(pure_iod_orbit) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    orbits, members, failures = run_od(source, associations, None, config)
    assert len(failures) == 0
    subset = members.take(pa.array([0, 2, 3, 5], type=pa.int64()))
    evaluated, evaluated_members, failures = evaluate_orbits(
        orbits, subset, source, None, config.for_evaluation()
    )
    assert len(failures) == 0
    assert evaluated_members.obs_id.to_pylist() == subset.obs_id.to_pylist()
    assert evaluated.num_obs.to_pylist() == [4]
    provenance = evaluated.provenance
    assert provenance.adam_core_build[0].as_py()
    assert provenance.evaluation_backend[0].as_py()
    assert provenance.evaluation_provider[0].as_py()
    assert provenance.state_fingerprint[0].as_py()
    assert provenance.provenance_fingerprint[0].as_py()
    assert evaluated.success.to_pylist() == [True]
    assert evaluated.status_code.to_pylist() == [1]
    assert evaluated_members.solution.to_pylist() == [
        not value for value in evaluated_members.outlier.to_pylist()
    ]


def test_membership_and_seed_group_fingerprints_are_enforced_before_factory(
    pure_iod_orbit, provider_counter
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    orbits, members, failures = run_iod(source, associations, None, config.for_iod())
    assert len(failures) == 0

    forged = members.set_column(
        "membership_fingerprint",
        pa.array(["0" * 64] * len(members), type=pa.large_string()),
    )
    with pytest.raises(ValueError, match="membership"):
        evaluate_orbits(
            orbits,
            forged,
            source,
            TwoBodyPropagator(),
            config.for_evaluation(),
        )
    assert provider_counter.calls == 0

    # Pair-level membership allows explicit subsets, but an IOD seed may be
    # composed only with its complete canonical group input.
    subset = members.take(pa.array(list(range(5)), type=pa.int64()))
    with pytest.raises(ValueError, match="canonical IOD input fingerprint"):
        refine(
            orbits,
            subset,
            source,
            TwoBodyPropagator(),
            config.for_refinement(),
        )
    assert provider_counter.calls == 0


@pytest.mark.parametrize("mutation", ["route", "configuration", "input"])
def test_route_configuration_and_input_provenance_are_cryptographically_bound(
    pure_iod_orbit, provider_counter, mutation
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    orbits, members, failures = run_od(source, associations, None, config)
    assert len(failures) == 0
    if mutation == "route":
        forged = orbits.set_column(
            "provenance.route", pa.array(["run_iod"], type=pa.large_string())
        )
    elif mutation == "configuration":
        value = json.loads(orbits.provenance.configuration[0].as_py())
        value["evaluation"]["covariance_dimension"] = 5
        forged = orbits.set_column(
            "provenance.configuration",
            pa.array([json.dumps(value)], type=pa.large_string()),
        )
    else:
        forged = orbits.set_column(
            "provenance.input_fingerprint",
            pa.array(["a" * 64], type=pa.large_string()),
        )
    with pytest.raises(ValueError, match="configuration|does not bind"):
        refine(
            forged,
            members,
            source,
            TwoBodyPropagator(),
            config.for_refinement(),
        )
    assert provider_counter.calls == 0


@pytest.mark.parametrize("route", ["run_iod", "run_od", "refine", "evaluate"])
def test_canonical_python_workflows_do_not_call_duplicate_native_preflight(
    pure_iod_orbit, monkeypatch, route
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    seed_orbits, seed_members, failures = run_od(source, associations, None, config)
    assert len(failures) == 0

    def duplicate_preflight(*_args, **_kwargs):
        raise AssertionError("canonical workflow called a duplicate planning pass")

    monkeypatch.setattr(_rust_native, "iod_od_batch_preflight_ipc", duplicate_preflight)
    monkeypatch.setattr(_rust_native, "refine_batch_preflight_ipc", duplicate_preflight)
    monkeypatch.setattr(
        _rust_native, "evaluate_orbits_batch_preflight_ipc", duplicate_preflight
    )
    if route == "run_iod":
        result = run_iod(source, associations, None, config.for_iod())
    elif route == "run_od":
        result = run_od(source, associations, None, config)
    elif route == "refine":
        result = refine(
            seed_orbits, seed_members, source, None, config.for_refinement()
        )
    else:
        result = evaluate_orbits(
            seed_orbits, seed_members, source, None, config.for_evaluation()
        )
    assert [len(table) for table in result] == [1, len(source), 0]


def test_nonconverged_failure_carries_output_and_input_orbit_identity(
    pure_iod_orbit,
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    orbits, _, failures = run_od(source, associations, None, _not_converging_config())
    assert len(orbits) == len(failures) == 1
    orbit_id = orbits.orbit_id[0].as_py()
    assert failures.orbit_id.to_pylist() == [orbit_id]
    assert failures.input_orbit_id.to_pylist() == [orbit_id]
    assert failures.best_state_retained.to_pylist() == [True]


def test_orbit_state_and_provenance_mutations_abort_before_factory(
    pure_iod_orbit, provider_counter
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    orbits, members, failures = run_od(source, associations, None, config)
    assert len(failures) == 0
    x = orbits.coordinates.x.to_numpy(zero_copy_only=False).copy()
    x[0] += 1e-12
    changed_coordinates = orbits.coordinates.set_column("x", pa.array(x))
    changed_orbits = orbits.set_column("coordinates", changed_coordinates)
    with pytest.raises(ValueError, match="state fingerprint"):
        evaluate_orbits(
            changed_orbits,
            members,
            source,
            TwoBodyPropagator(),
            config.for_evaluation(),
        )
    assert provider_counter.calls == 0

    missing = orbits.set_column(
        "provenance.state_fingerprint", pa.array([None], type=pa.large_string())
    )
    with pytest.raises(ValueError, match="state fingerprint"):
        refine(
            missing,
            members,
            source,
            TwoBodyPropagator(),
            config.for_refinement(),
        )
    assert provider_counter.calls == 0
