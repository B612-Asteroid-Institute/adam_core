"""Canonical batch ordering, atomicity, provider, and identity contracts."""

import hashlib
import json
import re
import struct

import pytest

from adam_core._rust.arrow import pa
from adam_core.observations import Associations, SourceCatalog

from .. import FittedOrbits, evaluate_orbits, refine, run_iod, run_od
from .conftest import PureIodOrbit
from .test_batch_workflows import (
    BatchResult,
    _assert_exact_results,
    _config,
    _non_roundtripping_float_config,
    _not_converging_config,
    _two_body_source,
)
from .test_differential_correction import TwoBodyPropagator


class _Counter:
    calls = 0


def _put_digest_string(digest: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _membership_fingerprint(
    group_id: str, observation_id: str, input_fingerprint: str
) -> str:
    digest = hashlib.sha256()
    digest.update(b"adam-core-od-membership-pair/v2\0")
    for value in (group_id, observation_id, input_fingerprint):
        _put_digest_string(digest, value)
    return digest.hexdigest()


def _put_optional_digest_string(digest: "hashlib._Hash", value: str | None) -> None:
    if value is None:
        digest.update(b"\x00")
    else:
        digest.update(b"\x01")
        _put_digest_string(digest, value)


def _public_consistency_digest(
    orbit: FittedOrbits,
    *,
    configuration: str,
) -> str:
    """Reproduce the documented unkeyed consistency digest for attack tests."""

    complete_sha256 = hashlib.sha256(
        json.dumps(
            json.loads(configuration),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    digest = hashlib.sha256()
    digest.update(b"adam-core-od-provenance-consistency/v3\0")
    for value in (
        orbit.provenance.route[0].as_py(),
        json.loads(configuration)["configuration_sha256"],
        complete_sha256,
        orbit.provenance.input_fingerprint[0].as_py(),
    ):
        _put_digest_string(digest, value)
    _put_optional_digest_string(digest, orbit.provenance.state_fingerprint[0].as_py())
    for column in (
        orbit.provenance.iod_provider,
        orbit.provenance.refinement_provider,
        orbit.provenance.evaluation_provider,
        orbit.provenance.iod_diagnostics,
        orbit.provenance.refinement_diagnostics,
        orbit.provenance.evaluation_diagnostics,
    ):
        _put_optional_digest_string(digest, column[0].as_py())
    ambiguous = orbit.provenance.iod_ambiguous[0].as_py()
    alternatives = orbit.provenance.iod_alternatives[0].as_py()
    digest.update(b"\x00" if ambiguous is None else bytes((1, int(ambiguous))))
    if alternatives is None:
        digest.update(b"\x00")
    else:
        digest.update(b"\x01")
        digest.update(struct.pack("<Q", alternatives))
    # IOD has no refinement rejection count or fraction.
    digest.update(b"\x00\x00")
    return digest.hexdigest()


@pytest.fixture
def provider_counter(monkeypatch: pytest.MonkeyPatch) -> type[_Counter]:
    original = TwoBodyPropagator.propagator_capsule
    _Counter.calls = 0

    def counted(self: TwoBodyPropagator) -> object:
        _Counter.calls += 1
        return original(self)

    monkeypatch.setattr(TwoBodyPropagator, "propagator_capsule", counted)
    return _Counter


def _source_and_ids(pure_iod_orbit: PureIodOrbit) -> tuple[SourceCatalog, list[str]]:
    source = _two_body_source(pure_iod_orbit)
    raw_ids = source.id.to_pylist()
    ids = [value for value in raw_ids if isinstance(value, str)]
    if len(ids) != len(raw_ids):
        raise TypeError("source identifiers must be non-null strings")
    return source, ids


def test_overlap_first_appearance_source_order_and_no_iod_placeholder(
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter]
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
    by_orbit: dict[str, list[str]] = {}
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
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter], mutation: str
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    detections = list(ids) + list(ids)
    groups: list[str | None] = []
    groups.extend(["g1"] * len(ids))
    groups.extend(["g2"] * len(ids))
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
    pure_iod_orbit: PureIodOrbit,
    provider_counter: type[_Counter],
    route: str,
    mutation: str,
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

        def call() -> BatchResult:
            return refine(
                orbits,
                members,
                source,
                TwoBodyPropagator(),
                config.for_refinement(),
            )

    else:

        def call() -> BatchResult:
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
    pure_iod_orbit: PureIodOrbit,
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
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter]
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
    for result in (fused, composed):
        provenance = result[0].provenance
        assert provenance.iod_backend[0].as_py()
        assert provenance.iod_provider[0].as_py()
        assert provenance.refinement_backend[0].as_py()
        assert provenance.refinement_provider[0].as_py()
    for field in (
        "iod_backend",
        "iod_provider",
        "refinement_backend",
        "refinement_provider",
    ):
        assert (
            getattr(fused[0].provenance, field).to_pylist()
            == getattr(composed[0].provenance, field).to_pylist()
        )
    evaluated = evaluate_orbits(
        fused[0], fused[1], source, propagator, config.for_evaluation()
    )
    assert evaluated[0].provenance.evaluation_backend[0].as_py()
    assert evaluated[0].provenance.evaluation_provider[0].as_py()
    assert provider_counter.calls == 4


def test_direct_iod_output_is_statically_valid_for_evaluation(
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter]
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


def test_explicit_evaluation_subset_and_stage_provenance(
    pure_iod_orbit: PureIodOrbit,
) -> None:
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
    assert "adam_core_version" not in provenance.schema.names
    assert "adam_core_build" not in provenance.schema.names
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
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter]
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
def test_route_configuration_and_input_provenance_have_a_validated_consistency_digest(
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter], mutation: str
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
    with pytest.raises(ValueError, match="configuration|consistency digest"):
        refine(
            forged,
            members,
            source,
            TwoBodyPropagator(),
            config.for_refinement(),
        )
    assert provider_counter.calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "profile",
        "iod_provider",
        "iod_backend",
        "iod_diagnostics",
        "iod_ambiguous",
        "iod_alternatives",
    ],
)
def test_composed_seed_fields_have_a_validated_consistency_digest(
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter], field: str
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    orbits, members, failures = run_iod(source, associations, None, config.for_iod())
    assert len(failures) == 0
    if field == "profile":
        value = json.loads(orbits.provenance.configuration[0].as_py())
        value["profile"] = {
            "name": "ASTEROID_INSTITUTE_OD_V1",
            "version": 1,
            "definition_sha256": "f" * 64,
        }
        forged = orbits.set_column(
            "provenance.configuration",
            pa.array(
                [json.dumps(value, separators=(",", ":"))],
                type=pa.large_string(),
            ),
        )
    elif field in {"iod_provider", "iod_diagnostics"}:
        value = json.loads(getattr(orbits.provenance, field)[0].as_py())
        value["name" if field == "iod_provider" else "triplets_tried"] = (
            "forged-provider" if field == "iod_provider" else 123456
        )
        forged = orbits.set_column(
            f"provenance.{field}",
            pa.array([json.dumps(value)], type=pa.large_string()),
        )
    elif field == "iod_backend":
        forged = orbits.set_column(
            "provenance.iod_backend",
            pa.array(["forged-provider 9.9.9"], type=pa.large_string()),
        )
    elif field == "iod_ambiguous":
        forged = orbits.set_column(
            "provenance.iod_ambiguous",
            pa.array([not orbits.provenance.iod_ambiguous[0].as_py()], type=pa.bool_()),
        )
    else:
        forged = orbits.set_column(
            "provenance.iod_alternatives",
            pa.array(
                [orbits.provenance.iod_alternatives[0].as_py() + 41], type=pa.int64()
            ),
        )
    with pytest.raises(ValueError, match="consistency digest"):
        refine(
            forged,
            members,
            source,
            TwoBodyPropagator(),
            config.for_refinement(),
        )
    assert provider_counter.calls == 0


@pytest.mark.parametrize("route", ["refine", "evaluate"])
def test_consistently_renamed_iod_seed_is_accepted_and_preserved(
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter], route: str
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    orbits, members, failures = run_iod(source, associations, None, config.for_iod())
    assert len(failures) == 0
    old_id = orbits.orbit_id[0].as_py()
    relabelled_id = "relabelled-seed"
    renamed_orbits = orbits.set_column(
        "orbit_id", pa.array([relabelled_id], type=pa.large_string())
    )
    renamed_members = members.set_column(
        "orbit_id",
        pa.array([relabelled_id] * len(members), type=pa.large_string()),
    )
    assert old_id != relabelled_id
    assert renamed_orbits.object_id.to_pylist() == ["g"]
    expected_membership = [
        _membership_fingerprint("g", obs_id, input_fingerprint)
        for obs_id, input_fingerprint in zip(
            members.obs_id.to_pylist(),
            members.input_fingerprint.to_pylist(),
            strict=True,
        )
    ]
    assert members.membership_fingerprint.to_pylist() == expected_membership
    assert renamed_members.membership_fingerprint.to_pylist() == expected_membership

    if route == "refine":
        baseline = refine(orbits, members, source, None, config.for_refinement())
        renamed = refine(
            renamed_orbits,
            renamed_members,
            source,
            None,
            config.for_refinement(),
        )
    else:
        baseline = evaluate_orbits(
            orbits, members, source, None, config.for_evaluation()
        )
        renamed = evaluate_orbits(
            renamed_orbits,
            renamed_members,
            source,
            None,
            config.for_evaluation(),
        )

    baseline_orbits, baseline_members, baseline_failures = baseline
    renamed_out, renamed_out_members, renamed_failures = renamed
    assert len(baseline_failures) == len(renamed_failures) == 0
    assert provider_counter.calls == 0
    assert renamed_out.orbit_id.to_pylist() == [relabelled_id]
    assert renamed_out.object_id.to_pylist() == ["g"]
    assert renamed_out_members.orbit_id.to_pylist() == [relabelled_id] * len(
        renamed_out_members
    )
    assert (
        renamed_out.provenance.input_fingerprint.to_pylist()
        == baseline_orbits.provenance.input_fingerprint.to_pylist()
    )
    assert (
        renamed_out.provenance.state_fingerprint.to_pylist()
        == baseline_orbits.provenance.state_fingerprint.to_pylist()
    )
    assert (
        renamed_out.provenance.provenance_fingerprint.to_pylist()
        == baseline_orbits.provenance.provenance_fingerprint.to_pylist()
    )
    assert (
        renamed_out_members.membership_fingerprint.to_pylist()
        == baseline_members.membership_fingerprint.to_pylist()
        == expected_membership
    )


@pytest.mark.parametrize("route", ["refine", "evaluate"])
def test_one_direct_label_edit_in_a_multi_orbit_batch_is_preserved(
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter], route: str
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids + ids,
        object_id=["g-1"] * len(ids) + ["g-2"] * len(ids),
    )
    config = _config()
    orbits, members, failures = run_iod(source, associations, None, config.for_iod())
    assert len(failures) == 0
    original_ids = orbits.orbit_id.to_pylist()
    relabelled = "one-direct-edit"
    edited_orbits = orbits.set_column(
        "orbit_id",
        pa.array([relabelled, original_ids[1]], type=pa.large_string()),
    )
    edited_member_ids = [
        relabelled if value == original_ids[0] else value
        for value in members.orbit_id.to_pylist()
    ]
    edited_members = members.set_column(
        "orbit_id", pa.array(edited_member_ids, type=pa.large_string())
    )

    if route == "refine":
        baseline = refine(
            orbits, members, source, TwoBodyPropagator(), config.for_refinement()
        )
        edited = refine(
            edited_orbits,
            edited_members,
            source,
            TwoBodyPropagator(),
            config.for_refinement(),
        )
    else:
        baseline = evaluate_orbits(
            orbits, members, source, TwoBodyPropagator(), config.for_evaluation()
        )
        edited = evaluate_orbits(
            edited_orbits,
            edited_members,
            source,
            TwoBodyPropagator(),
            config.for_evaluation(),
        )

    baseline_orbits, baseline_members, baseline_failures = baseline
    edited_orbits_out, edited_members_out, edited_failures = edited
    assert len(baseline_failures) == len(edited_failures) == 0
    assert provider_counter.calls == 2
    assert edited_orbits_out.orbit_id.to_pylist() == [relabelled, original_ids[1]]
    assert edited_orbits_out.object_id.to_pylist() == ["g-1", "g-2"]
    assert edited_members_out.orbit_id.to_pylist() == edited_member_ids
    assert (
        edited_orbits_out.provenance.input_fingerprint.to_pylist()
        == baseline_orbits.provenance.input_fingerprint.to_pylist()
    )
    assert (
        edited_orbits_out.provenance.state_fingerprint.to_pylist()
        == baseline_orbits.provenance.state_fingerprint.to_pylist()
    )
    assert (
        edited_orbits_out.provenance.provenance_fingerprint.to_pylist()
        == baseline_orbits.provenance.provenance_fingerprint.to_pylist()
    )
    assert (
        edited_members_out.membership_fingerprint.to_pylist()
        == baseline_members.membership_fingerprint.to_pylist()
    )


@pytest.mark.parametrize("mutation", ["dangling", "empty", "duplicate"])
def test_inconsistent_or_ambiguous_orbit_labels_fail_before_provider(
    pure_iod_orbit: PureIodOrbit,
    provider_counter: type[_Counter],
    mutation: str,
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    config = _config()
    if mutation == "duplicate":
        associations = Associations.from_kwargs(
            detection_id=ids + ids,
            object_id=["g-1"] * len(ids) + ["g-2"] * len(ids),
        )
    else:
        associations = Associations.from_kwargs(
            detection_id=ids, object_id=["g"] * len(ids)
        )
    orbits, members, failures = run_iod(source, associations, None, config.for_iod())
    assert len(failures) == 0

    if mutation == "dangling":
        forged_orbits = orbits.set_column(
            "orbit_id", pa.array(["renamed"], type=pa.large_string())
        )
        forged_members = members
        expected = "unknown orbit_id"
    elif mutation == "empty":
        forged_orbits = orbits.set_column(
            "orbit_id", pa.array([""], type=pa.large_string())
        )
        forged_members = members.set_column(
            "orbit_id", pa.array([""] * len(members), type=pa.large_string())
        )
        expected = "orbit ID.*empty"
    else:
        forged_orbits = orbits.set_column(
            "orbit_id", pa.array(["duplicate", "duplicate"], type=pa.large_string())
        )
        forged_members = members.set_column(
            "orbit_id",
            pa.array(["duplicate"] * len(members), type=pa.large_string()),
        )
        expected = "orbit ID.*not unique"

    with pytest.raises(ValueError, match=expected):
        evaluate_orbits(
            forged_orbits,
            forged_members,
            source,
            TwoBodyPropagator(),
            config.for_evaluation(),
        )
    assert provider_counter.calls == 0


@pytest.mark.parametrize("route", ["refine", "evaluate"])
def test_seed_configuration_content_must_match_its_science_sha_before_provider(
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter], route: str
) -> None:
    """A recomputed public digest cannot bless stale configuration_sha256."""

    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _config()
    orbits, members, failures = run_iod(source, associations, None, config.for_iod())
    assert len(failures) == 0

    changed = json.loads(orbits.provenance.configuration[0].as_py())
    changed["evaluation"]["covariance_dimension"] = 5
    # Keep the stale science hash deliberately, then recompute the public,
    # unkeyed consistency digest over the altered complete configuration.
    assert (
        changed["configuration_sha256"]
        == json.loads(orbits.provenance.configuration[0].as_py())[
            "configuration_sha256"
        ]
    )
    changed_json = json.dumps(changed, separators=(",", ":"))
    forged = orbits.set_column(
        "provenance.configuration",
        pa.array([changed_json], type=pa.large_string()),
    ).set_column(
        "provenance.provenance_fingerprint",
        pa.array(
            [_public_consistency_digest(orbits, configuration=changed_json)],
            type=pa.large_string(),
        ),
    )

    if route == "refine":

        def call() -> BatchResult:
            return refine(
                forged,
                members,
                source,
                TwoBodyPropagator(),
                config.for_refinement(),
            )

    else:

        def call() -> BatchResult:
            return evaluate_orbits(
                forged,
                members,
                source,
                TwoBodyPropagator(),
                config.for_evaluation(),
            )

    with pytest.raises(ValueError, match="configuration_sha256"):
        call()
    assert provider_counter.calls == 0


def test_equivalent_seed_json_is_canonicalized_before_composed_provenance(
    pure_iod_orbit: PureIodOrbit,
) -> None:
    source, ids = _source_and_ids(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=ids, object_id=["g"] * len(ids)
    )
    config = _non_roundtripping_float_config()
    iod = run_iod(source, associations, None, config.for_iod())
    raw = iod[0].provenance.configuration[0].as_py()
    value = json.loads(raw)
    rho = value["iod"]["search_correction_ranging"]["ranging"]["rho_min_au"]
    canonical_spelling = json.dumps(rho)
    equivalent_spelling = format(rho, ".30f").rstrip("0").rstrip(".")
    assert float(equivalent_spelling).hex() == rho.hex()
    assert equivalent_spelling != canonical_spelling
    altered = re.sub(
        r'("rho_min_au":)[^,}]+',
        rf"\g<1>{equivalent_spelling}",
        raw,
        count=1,
    )
    altered = "\n  " + altered + "\n"
    assert equivalent_spelling in altered
    assert altered != raw and json.loads(altered) == value
    altered_orbits = iod[0].set_column(
        "provenance.configuration",
        pa.array([altered], type=pa.large_string()),
    )

    composed = refine(
        altered_orbits,
        iod[1],
        source,
        None,
        config.for_refinement(),
    )
    fused = run_od(source, associations, None, config)
    _assert_exact_results(fused, composed)
    emitted = composed[0].provenance.configuration[0].as_py()
    assert emitted == raw
    assert emitted == json.dumps(json.loads(emitted), separators=(",", ":"))


def test_nonconverged_failure_carries_output_and_input_orbit_identity(
    pure_iod_orbit: PureIodOrbit,
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
    pure_iod_orbit: PureIodOrbit, provider_counter: type[_Counter]
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
