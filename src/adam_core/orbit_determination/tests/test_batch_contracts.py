"""Canonical batch ordering, atomicity, provider, and identity contracts."""

import pyarrow as pa
import pytest

from adam_core.observations import Associations

from .. import evaluate_orbits, refine, run_iod, run_od
from .test_batch_workflows import (
    _assert_exact_results,
    _config,
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
