"""Canonical batch-native workflow composition and preflight contracts."""

import dataclasses
import json

import numpy as np
import pyarrow as pa
import pytest

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.observations import Associations, SourceCatalog
from adam_core.time import Timestamp

from .. import (
    EvaluationConfig,
    FittedOrbitMembers,
    FittedOrbits,
    IodRunConfig,
    NightlyBatchDeweighting,
    ObservationPipeline,
    OdConfig,
    OrbitDeterminationFailures,
    RefinementRunConfig,
    ResourceBindings,
    WorstResidualRefinement,
    evaluate_orbits,
    refine,
    run_iod,
    run_od,
)
from .test_differential_correction import TwoBodyPropagator


def _config() -> OdConfig:
    return OdConfig.explicit(
        observation_pipeline=ObservationPipeline.as_supplied(),
        iod=IodRunConfig.core_v1(),
        refinement=RefinementRunConfig.core_v1(),
        evaluation=EvaluationConfig.six_dimensional(),
        resources=ResourceBindings({}),
    )


def _two_body_source(pure_iod_orbit) -> SourceCatalog:
    orbit, _, observations = pure_iod_orbit
    ephemeris = TwoBodyPropagator().generate_ephemeris(
        orbit.to_orbits(), observations.observers, max_processes=1
    )
    n = len(observations)
    sigma_deg = 0.3 / 3600.0
    rng = np.random.default_rng(7)
    lon = ephemeris.coordinates.lon.to_numpy() + 0.3 * sigma_deg * rng.standard_normal(
        n
    )
    lat = ephemeris.coordinates.lat.to_numpy() + 0.3 * sigma_deg * rng.standard_normal(
        n
    )
    covariance = np.full((n, 6, 6), np.nan)
    covariance[:, 1, 1] = sigma_deg**2
    covariance[:, 2, 2] = sigma_deg**2
    coordinates = (
        observations.coordinates.set_column("lon", pa.array(lon))
        .set_column("lat", pa.array(lat))
        .set_column("covariance", CoordinateCovariances.from_matrix(covariance))
    )
    return SourceCatalog.from_kwargs(
        id=observations.id,
        time=coordinates.time,
        ra=coordinates.lon,
        dec=coordinates.lat,
        ra_sigma=[0.3] * n,
        dec_sigma=[0.3] * n,
        observatory_code=observations.observers.code,
        object_id=["object-1"] * n,
        catalog_id=["synthetic-two-body"] * n,
        astcat=observations.astcat,
    )


def _canonical_ipc(table) -> bytes:
    """Canonical Arrow IPC comparator preserving exact floating-point bits."""

    arrow = table.table.combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, arrow.schema) as writer:
        writer.write_table(arrow)
    return sink.getvalue().to_pybytes()


def _assert_exact_results(left, right) -> None:
    for left_table, right_table in zip(left, right, strict=True):
        assert _canonical_ipc(left_table) == _canonical_ipc(right_table)


def _concat_failures(*failures) -> OrbitDeterminationFailures:
    tables = [value.table for value in failures if len(value)]
    if not tables:
        return OrbitDeterminationFailures.empty()
    return OrbitDeterminationFailures.from_pyarrow(
        pa.concat_tables(tables).combine_chunks()
    )


def _sorted_failure_ipc(failures: OrbitDeterminationFailures) -> bytes:
    group_ids = failures.group_id.to_pylist()
    stages = failures.stage.to_pylist()
    codes = failures.code.to_pylist()
    order = sorted(
        range(len(failures)),
        key=lambda row: (group_ids[row], stages[row], codes[row]),
    )
    return _canonical_ipc(failures.take(pa.array(order, type=pa.int64())))


def _assert_batch_evaluation_matches_singletons(
    orbits: FittedOrbits,
    members: FittedOrbitMembers,
    source: SourceCatalog,
    config: EvaluationConfig,
):
    batch = evaluate_orbits(orbits, members, source, None, config)
    evaluated, evaluated_members, failures = batch
    assert len(evaluated) == len(orbits)
    assert len(failures) == 0
    evaluated_ids = evaluated.orbit_id.to_pylist()
    input_states = orbits.coordinates.values
    for row, orbit_id in enumerate(orbits.orbit_id.to_pylist()):
        output_row = evaluated_ids.index(orbit_id)
        np.testing.assert_array_equal(
            evaluated.coordinates.values[output_row].view(np.uint64),
            input_states[row].view(np.uint64),
        )
        input_members = members.apply_mask(pa.compute.equal(members.orbit_id, orbit_id))
        singleton = evaluate_orbits(
            orbits.take(pa.array([row], type=pa.int64())),
            input_members,
            source,
            None,
            config,
        )
        assert len(singleton[2]) == 0
        assert _canonical_ipc(
            evaluated.take(pa.array([output_row], type=pa.int64()))
        ) == _canonical_ipc(singleton[0])
        batch_members = evaluated_members.apply_mask(
            pa.compute.equal(evaluated_members.orbit_id, orbit_id)
        )
        assert _canonical_ipc(batch_members) == _canonical_ipc(singleton[1])
    return batch


def _not_converging_config() -> OdConfig:
    config = _config()
    declaration = config.for_refinement().refinement
    fit = dataclasses.replace(
        declaration.fit,
        max_iterations=1,
        max_nfev=1,
        xtol=1e-300,
        ftol=1e-300,
        gtol=1e-300,
    )
    refinement = RefinementRunConfig(dataclasses.replace(declaration, fit=fit))
    return OdConfig.explicit(
        observation_pipeline=config.observation_pipeline,
        iod=config.for_iod(),
        refinement=refinement,
        evaluation=config.for_evaluation(),
        resources=ResourceBindings({}),
    )


def _non_roundtripping_float_config() -> OdConfig:
    """Known serde_json edge values retained exactly in both OD stages."""

    base = _config()
    iod_declaration = base.for_iod()
    search = iod_declaration.search_correction_ranging
    ranging = dataclasses.replace(search.ranging, rho_min_au=1.8135371537546276e-10)
    iod = IodRunConfig(
        dataclasses.replace(search, ranging=ranging),
        iod_declaration.candidate_correction_fit,
    )
    refinement_declaration = base.for_refinement().refinement
    refinement = RefinementRunConfig(
        dataclasses.replace(refinement_declaration, psd_floor_frac=1e-30)
    )
    return OdConfig.explicit(
        observation_pipeline=base.observation_pipeline,
        iod=iod,
        refinement=refinement,
        evaluation=base.for_evaluation(),
        resources=ResourceBindings({}),
    )


def _assert_non_roundtripping_values_in_provenance(orbits: FittedOrbits) -> None:
    value = json.loads(orbits.provenance.configuration[0].as_py())
    assert (
        value["iod"]["search_correction_ranging"]["ranging"]["rho_min_au"].hex()
        == (1.8135371537546276e-10).hex()
    )
    assert value["refinement"]["psd_floor_frac"].hex() == (1e-30).hex()
    assert orbits.provenance.provenance_fingerprint[0].as_py()


def test_non_roundtripping_config_floats_are_exact_across_every_route(
    pure_iod_orbit,
) -> None:
    source = _two_body_source(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=source.id, object_id=["group-1"] * len(source)
    )
    config = _non_roundtripping_float_config()
    assert (
        config.for_iod().search_correction_ranging.ranging.rho_min_au.hex()
        == (1.8135371537546276e-10).hex()
    )
    assert config.for_refinement().refinement.psd_floor_frac.hex() == (1e-30).hex()

    iod = run_iod(source, associations, None, config.for_iod())
    fused = run_od(source, associations, None, config)
    composed = refine(iod[0], iod[1], source, None, config.for_refinement())
    _assert_exact_results(fused, composed)
    for result in (iod, fused, composed):
        assert len(result[0]) == 1
        _assert_non_roundtripping_values_in_provenance(result[0])

    evaluated_iod = evaluate_orbits(
        iod[0], iod[1], source, None, config.for_evaluation()
    )
    evaluated_fused = evaluate_orbits(
        fused[0], fused[1], source, None, config.for_evaluation()
    )
    evaluated_composed = evaluate_orbits(
        composed[0], composed[1], source, None, config.for_evaluation()
    )
    _assert_exact_results(evaluated_fused, evaluated_composed)
    for result in (evaluated_iod, evaluated_fused, evaluated_composed):
        assert len(result[0]) == 1
        _assert_non_roundtripping_values_in_provenance(result[0])


def test_run_od_is_exact_run_iod_then_refine(pure_iod_orbit) -> None:
    source = _two_body_source(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=source.id,
        object_id=["group-1"] * len(source),
    )
    config = _config()

    iod_orbits, iod_members, iod_failures = run_iod(
        source, associations, None, config.for_iod()
    )
    assert len(iod_orbits) == 1
    assert len(iod_members) == len(source)
    assert len(iod_failures) == 0

    composed = refine(
        iod_orbits,
        iod_members,
        source,
        None,
        config.for_refinement(),
    )
    fused = run_od(source, associations, None, config)

    _assert_exact_results(fused, composed)


@pytest.mark.parametrize("force_failure", [False, True])
def test_profile_shaped_deweighting_is_exact_across_fused_and_composed_routes(
    pure_iod_orbit, force_failure
) -> None:
    base_source = _two_body_source(pure_iod_orbit)
    # Repeat the same valid seven-epoch arc three times with distinct detection
    # IDs. The station-night containing two rows now has six rows, so the
    # profile's cap=4 transform genuinely changes its variance while preserving
    # the original distinct-epoch science problem.
    source = SourceCatalog.from_pyarrow(
        pa.concat_tables([base_source.table] * 3).combine_chunks()
    )
    source = source.set_column(
        "id",
        pa.array(
            [
                f"{obs_id}-repeat-{repeat}"
                for repeat in range(3)
                for obs_id in base_source.id.to_pylist()
            ],
            type=pa.large_string(),
        ),
    )
    # Make the persisted covariance bits discriminating for both former lossy
    # paths: snapshotting variance as sqrt(variance)^2 and reconstructing it as
    # sigma^2. Preserve heterogeneous cross covariance as part of the handoff.
    rng = np.random.default_rng(11)
    source = (
        source.set_column(
            "ra_sigma", pa.array(0.3 * (1.0 + 0.5 * rng.random(len(source))))
        )
        .set_column("dec_sigma", pa.array(0.3 * (1.0 + 0.5 * rng.random(len(source)))))
        .set_column("radec_corr", pa.array(np.linspace(-0.4, 0.4, len(source))))
    )
    source_covariance = source.coordinates().covariance.to_matrix()
    source_variance_lon = source_covariance[:, 1, 1]
    source_variance_lat = source_covariance[:, 2, 2]
    source_cross = source_covariance[:, 1, 2]
    associations = Associations.from_kwargs(
        detection_id=source.id,
        object_id=["group-1"] * len(source),
    )
    base = _not_converging_config() if force_failure else _config()
    config = OdConfig.explicit(
        observation_pipeline=ObservationPipeline((NightlyBatchDeweighting(4),)),
        iod=base.for_iod(),
        refinement=base.for_refinement(),
        evaluation=base.for_evaluation(),
        resources=ResourceBindings({}),
    )

    fused = run_od(source, associations, None, config)
    iod = run_iod(source, associations, None, config.for_iod())
    composed = refine(iod[0], iod[1], source, None, config.for_refinement())

    original_variance = iod[1].original_astrometry.variance_lon.to_numpy(
        zero_copy_only=False
    )
    original_variance_lat = iod[1].original_astrometry.variance_lat.to_numpy(
        zero_copy_only=False
    )
    used_variance = iod[1].used_astrometry.variance_lon.to_numpy(zero_copy_only=False)
    used_variance_lat = iod[1].used_astrometry.variance_lat.to_numpy(
        zero_copy_only=False
    )
    used_cross = iod[1].used_astrometry.cov_lonlat.to_numpy(zero_copy_only=False)
    changed = original_variance.view(np.uint64) != used_variance.view(np.uint64)
    assert np.any(changed)
    np.testing.assert_array_equal(
        original_variance.view(np.uint64), source_variance_lon.view(np.uint64)
    )
    np.testing.assert_array_equal(
        original_variance_lat.view(np.uint64), source_variance_lat.view(np.uint64)
    )
    expected_used_variance = source_variance_lon.copy()
    expected_used_variance[changed] *= 1.5
    expected_used_variance_lat = source_variance_lat.copy()
    expected_used_variance_lat[changed] *= 1.5
    expected_used_cross = source_cross.copy()
    expected_used_cross[changed] *= 1.5
    np.testing.assert_array_equal(
        used_variance.view(np.uint64), expected_used_variance.view(np.uint64)
    )
    np.testing.assert_array_equal(
        used_variance_lat.view(np.uint64), expected_used_variance_lat.view(np.uint64)
    )
    np.testing.assert_array_equal(
        used_cross.view(np.uint64), expected_used_cross.view(np.uint64)
    )
    assert np.any(
        (np.sqrt(used_variance) ** 2).view(np.uint64) != used_variance.view(np.uint64)
    )
    assert np.all(np.isfinite(used_cross)) and np.any(used_cross != 0.0)
    np.testing.assert_array_equal(
        fused[1]
        .used_astrometry.variance_lon.to_numpy(zero_copy_only=False)
        .view(np.uint64),
        used_variance.view(np.uint64),
    )
    _assert_exact_results(fused, composed)
    if force_failure:
        assert fused[0].success.to_pylist() == [False]
        assert fused[2].best_state_retained.to_pylist() == [True]


@pytest.mark.parametrize("force_failure", [False, True])
def test_overlapping_multigroup_ordered_pipeline_exact_composition_and_evaluation(
    pure_iod_orbit, force_failure
) -> None:
    """Pin shared-row, multi-hypothesis composition and both former mutants."""

    base_source = _two_body_source(pure_iod_orbit)
    source = SourceCatalog.from_pyarrow(
        pa.concat_tables([base_source.table] * 3).combine_chunks()
    )
    source = source.set_column(
        "id",
        pa.array(
            [
                f"{obs_id}-multi-{repeat}"
                for repeat in range(3)
                for obs_id in base_source.id.to_pylist()
            ],
            type=pa.large_string(),
        ),
    )
    rng = np.random.default_rng(217)
    source = (
        source.set_column("ra_sigma", pa.array(0.2 + 0.3 * rng.random(len(source))))
        .set_column("dec_sigma", pa.array(0.25 + 0.35 * rng.random(len(source))))
        .set_column("radec_corr", pa.array(np.linspace(-0.45, 0.45, len(source))))
    )
    covariance = source.coordinates().covariance.to_matrix()
    assert np.all(np.isfinite(covariance[:, 1, 2]))
    assert np.any(covariance[:, 1, 2] != 0.0)

    ids = source.id.to_pylist()
    group_a = list(reversed(ids))
    group_b_indices = [3, 0, 4, 1, 5, 2] + list(range(6, 14))
    group_b = [ids[index] for index in group_b_indices]
    group_c = ids[-3:]
    associations = Associations.from_kwargs(
        detection_id=group_a + group_b + group_c,
        object_id=(
            ["group-a"] * len(group_a)
            + ["group-b"] * len(group_b)
            + ["group-c-too-small"] * len(group_c)
        ),
    )
    base = _not_converging_config() if force_failure else _config()
    config = OdConfig.explicit(
        observation_pipeline=ObservationPipeline.ordered(
            (NightlyBatchDeweighting(4), NightlyBatchDeweighting(2))
        ),
        iod=base.for_iod(),
        refinement=base.for_refinement(),
        evaluation=base.for_evaluation(),
        resources=ResourceBindings({}),
    )

    iod = run_iod(source, associations, None, config.for_iod())
    composed = refine(iod[0], iod[1], source, None, config.for_refinement())
    fused = run_od(source, associations, None, config)
    expected = (composed[0], composed[1], _concat_failures(iod[2], composed[2]))

    assert _canonical_ipc(fused[0]) == _canonical_ipc(expected[0])
    assert _canonical_ipc(fused[1]) == _canonical_ipc(expected[1])
    # Stage concatenation and fused group traversal need not encounter the
    # small IOD failure in the same row position; failure rows themselves are
    # bit-exact after deterministic group/stage ordering.
    assert _sorted_failure_ipc(fused[2]) == _sorted_failure_ipc(expected[2])
    assert iod[0].object_id.to_pylist() == ["group-a", "group-b"]
    assert iod[2].group_id.to_pylist() == ["group-c-too-small"]
    assert iod[2].code.to_pylist() == ["no_solution"]

    by_group: dict[str, list[str]] = {}
    orbit_to_group = dict(
        zip(iod[0].orbit_id.to_pylist(), iod[0].object_id.to_pylist(), strict=True)
    )
    for orbit_id, obs_id in zip(
        iod[1].orbit_id.to_pylist(), iod[1].obs_id.to_pylist(), strict=True
    ):
        by_group.setdefault(orbit_to_group[orbit_id], []).append(obs_id)
    assert by_group["group-a"] == ids
    assert by_group["group-b"] == ids[:14]

    shared = ids[0]
    used_variance_by_group = {}
    for orbit_id, obs_id, variance in zip(
        iod[1].orbit_id.to_pylist(),
        iod[1].obs_id.to_pylist(),
        iod[1].used_astrometry.variance_lon.to_pylist(),
        strict=True,
    ):
        if obs_id == shared:
            used_variance_by_group[orbit_to_group[orbit_id]] = variance
    assert used_variance_by_group["group-a"] != used_variance_by_group["group-b"]
    provenance = json.loads(iod[0].provenance.configuration[0].as_py())
    assert [
        transform["effective_observation_cap"] for transform in provenance["transforms"]
    ] == [4, 2]

    evaluated_fused = _assert_batch_evaluation_matches_singletons(
        fused[0], fused[1], source, config.for_evaluation()
    )
    reversed_orbits = fused[0].take(pa.array([1, 0], type=pa.int64()))
    _assert_batch_evaluation_matches_singletons(
        reversed_orbits, fused[1], source, config.for_evaluation()
    )
    evaluated_composed = evaluate_orbits(
        composed[0], composed[1], source, None, config.for_evaluation()
    )
    _assert_exact_results(evaluated_fused, evaluated_composed)
    assert evaluated_fused[0].object_id.to_pylist() == ["group-a", "group-b"]

    if force_failure:
        assert fused[0].success.to_pylist() == [False, False]
        assert set(fused[2].group_id.to_pylist()) == {
            "group-a",
            "group-b",
            "group-c-too-small",
        }
    else:
        assert fused[0].success.to_pylist() == [True, True]
        assert fused[2].group_id.to_pylist() == ["group-c-too-small"]


def test_input_fingerprint_changes_when_pipeline_changes_only_used_astrometry(
    pure_iod_orbit,
) -> None:
    base_source = _two_body_source(pure_iod_orbit)
    source = SourceCatalog.from_pyarrow(
        pa.concat_tables([base_source.table] * 3).combine_chunks()
    ).set_column(
        "id",
        pa.array(
            [
                f"{obs_id}-fingerprint-{repeat}"
                for repeat in range(3)
                for obs_id in base_source.id.to_pylist()
            ],
            type=pa.large_string(),
        ),
    )
    associations = Associations.from_kwargs(
        detection_id=source.id,
        object_id=["fingerprint-group"] * len(source),
    )
    as_supplied = _config()
    transformed = OdConfig.explicit(
        observation_pipeline=ObservationPipeline.ordered((NightlyBatchDeweighting(4),)),
        iod=as_supplied.for_iod(),
        refinement=as_supplied.for_refinement(),
        evaluation=as_supplied.for_evaluation(),
        resources=ResourceBindings({}),
    )

    original = run_iod(source, associations, None, as_supplied.for_iod())
    changed = run_iod(source, associations, None, transformed.for_iod())
    assert _canonical_ipc(original[1].original_astrometry) == _canonical_ipc(
        changed[1].original_astrometry
    )
    assert _canonical_ipc(original[1].used_astrometry) != _canonical_ipc(
        changed[1].used_astrometry
    )
    assert (
        original[0].provenance.input_fingerprint.to_pylist()
        != changed[0].provenance.input_fingerprint.to_pylist()
    )
    assert (
        original[1].input_fingerprint.to_pylist()
        != changed[1].input_fingerprint.to_pylist()
    )


def test_run_od_failure_is_exact_run_iod_then_refine(pure_iod_orbit) -> None:
    source = _two_body_source(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=source.id,
        object_id=["group-1"] * len(source),
    )
    config = _not_converging_config()
    iod = run_iod(source, associations, None, config.for_iod())
    composed = refine(iod[0], iod[1], source, None, config.for_refinement())
    fused = run_od(source, associations, None, config)

    _assert_exact_results(fused, composed)
    assert fused[0].success.to_pylist() == [False]
    assert fused[0].status_code.to_pylist() == [0]
    assert fused[2].group_id.to_pylist() == ["group-1"]
    assert fused[2].best_state_retained.to_pylist() == [True]


def test_worst_residual_provenance_counts_actual_rejections(pure_iod_orbit) -> None:
    source = _two_body_source(pure_iod_orbit)
    ra = source.ra.to_numpy(zero_copy_only=False).copy()
    ra[3] += 0.05
    source = source.set_column("ra", pa.array(ra))
    associations = Associations.from_kwargs(
        detection_id=source.id,
        object_id=["group-1"] * len(source),
    )
    base = _config()
    fit = base.for_refinement().refinement.fit
    refinement = RefinementRunConfig(
        WorstResidualRefinement(
            method="worst_residual",
            rchi2_threshold=3.0,
            min_obs=5,
            min_arc_length=1.0,
            contamination_percentage=30.0,
            fit=fit,
        )
    )
    config = OdConfig.explicit(
        observation_pipeline=base.observation_pipeline,
        iod=base.for_iod(),
        refinement=refinement,
        evaluation=base.for_evaluation(),
        resources=ResourceBindings({}),
    )
    orbits, members, failures = run_od(source, associations, None, config)
    assert len(orbits) == 1
    assert len(failures) == 0
    n_rejected = int(np.sum(members.outlier.to_numpy(zero_copy_only=False)))
    assert n_rejected > 0
    assert orbits.provenance.num_rejected.to_pylist() == [n_rejected]
    assert orbits.provenance.rejected_fraction.to_pylist() == [n_rejected / len(source)]
    diagnostics = json.loads(orbits.provenance.refinement_diagnostics[0].as_py())
    assert diagnostics["n_rejected"] == n_rejected


def test_refine_rejects_source_astrometry_changed_since_iod_before_provider_binding(
    pure_iod_orbit,
) -> None:
    source = _two_body_source(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=source.id,
        object_id=["group-1"] * len(source),
    )
    config = _config()
    iod_orbits, iod_members, failures = run_iod(
        source, associations, None, config.for_iod()
    )
    assert len(failures) == 0
    changed_ra = source.ra.to_numpy(zero_copy_only=False).copy()
    changed_ra[0] += 1e-7
    changed_source = source.set_column("ra", pa.array(changed_ra))

    with pytest.raises(
        ValueError, match="does not match persisted original_astrometry"
    ):
        refine(
            iod_orbits,
            iod_members,
            changed_source,
            object(),  # type: ignore[arg-type]
            config.for_refinement(),
        )


@pytest.mark.parametrize("route", ["run_iod", "run_od", "refine", "evaluate"])
def test_empty_batches_are_rust_owned_and_never_bind_provider(route) -> None:
    config = _config()
    if route == "run_iod":
        result = run_iod(
            SourceCatalog.empty(),
            Associations.empty(),
            object(),  # type: ignore[arg-type]
            config.for_iod(),
        )
    elif route == "run_od":
        result = run_od(
            SourceCatalog.empty(),
            Associations.empty(),
            object(),  # type: ignore[arg-type]
            config,
        )
    elif route == "refine":
        result = refine(
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            SourceCatalog.empty(),
            object(),  # type: ignore[arg-type]
            config.for_refinement(),
        )
    else:
        result = evaluate_orbits(
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            SourceCatalog.empty(),
            object(),  # type: ignore[arg-type]
            config.for_evaluation(),
        )
    assert [len(table) for table in result] == [0, 0, 0]


@pytest.mark.parametrize("route", ["refine", "evaluate"])
@pytest.mark.parametrize("mutation", ["time", "observatory", "filter", "id"])
def test_member_source_identity_mutations_fail_before_provider_binding(
    pure_iod_orbit, route, mutation
) -> None:
    source = _two_body_source(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=source.id,
        object_id=["group-1"] * len(source),
    )
    config = _config()
    orbits, members, failures = run_od(source, associations, None, config)
    assert len(failures) == 0
    if mutation == "time":
        days = source.time.days.to_pylist()
        nanos = source.time.nanos.to_pylist()
        nanos[0] += 3_600_000_000_000
        mutated = source.set_column(
            "time",
            Timestamp.from_kwargs(days=days, nanos=nanos, scale=source.time.scale),
        )
    elif mutation == "observatory":
        codes = source.observatory_code.to_pylist()
        codes[0] = "500" if codes[0] != "500" else "I41"
        mutated = source.set_column(
            "observatory_code", pa.array(codes, type=source.observatory_code.type)
        )
    elif mutation == "filter":
        filters = source.filter.to_pylist()
        filters[0] = "r" if filters[0] != "r" else "g"
        mutated = source.set_column(
            "filter", pa.array(filters, type=source.filter.type)
        )
    else:
        ids = source.id.to_pylist()
        ids[0] = f"{ids[0]}-changed"
        mutated = source.set_column("id", pa.array(ids, type=source.id.type))
    if route == "refine":

        def call():
            return refine(
                orbits,
                members,
                mutated,
                object(),  # type: ignore[arg-type]
                config.for_refinement(),
            )

    else:

        def call():
            return evaluate_orbits(
                orbits,
                members,
                mutated,
                object(),  # type: ignore[arg-type]
                config.for_evaluation(),
            )

    with pytest.raises(ValueError):
        call()


def test_evaluate_rejects_null_outlier_before_provider_binding(pure_iod_orbit) -> None:
    source = _two_body_source(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=source.id,
        object_id=["group-1"] * len(source),
    )
    config = _config()
    orbits, members, failures = run_od(source, associations, None, config)
    assert len(failures) == 0
    members = members.set_column(
        "outlier", pa.array([None] * len(members), type=pa.bool_())
    )
    with pytest.raises(ValueError, match="outlier is null"):
        evaluate_orbits(
            orbits,
            members,
            source,
            object(),  # type: ignore[arg-type]
            config.for_evaluation(),
        )


def test_typed_mixed_group_evaluation_continues_without_placeholder(
    pure_iod_orbit,
) -> None:
    source = _two_body_source(pure_iod_orbit)
    detection_ids = source.id.to_pylist()
    associations = Associations.from_kwargs(
        detection_id=detection_ids + detection_ids,
        object_id=["failed-group"] * len(source) + ["good-group"] * len(source),
    )
    config = _config()
    orbits, members, failures = run_od(source, associations, None, config)
    assert len(orbits) == 2
    assert len(failures) == 0
    failed_orbit_id = orbits.orbit_id[0].as_py()
    flags = [orbit_id == failed_orbit_id for orbit_id in members.orbit_id.to_pylist()]
    members = members.set_column("outlier", pa.array(flags, type=pa.bool_()))
    evaluated, evaluated_members, failures = evaluate_orbits(
        orbits, members, source, None, config.for_evaluation()
    )
    assert evaluated.object_id.to_pylist() == ["good-group"]
    assert set(evaluated_members.orbit_id.to_pylist()) == {
        evaluated.orbit_id[0].as_py()
    }
    assert failures.group_id.to_pylist() == ["failed-group"]
    assert failures.code.to_pylist() == ["no_included_observations"]
    assert failures.best_state_retained.to_pylist() == [False]
    assert failures.recoverable.to_pylist() == [False]
    assert failures.orbit_id.to_pylist() == [None]
    assert failures.input_orbit_id.to_pylist() == [orbits.orbit_id[0].as_py()]


def test_local_refinement_error_emits_failure_only(pure_iod_orbit) -> None:
    source = _two_body_source(pure_iod_orbit)
    associations = Associations.from_kwargs(
        detection_id=source.id,
        object_id=["group-1"] * len(source),
    )
    config = _config()
    orbits, members, failures = run_od(source, associations, None, config)
    assert len(failures) == 0
    subset = members.take(pa.array([0, 1], type=pa.int64()))
    refined, refined_members, failures = refine(
        orbits, subset, source, None, config.for_refinement()
    )
    assert len(refined) == 0
    assert len(refined_members) == 0
    assert failures.group_id.to_pylist() == ["group-1"]
    assert failures.code.to_pylist() == ["rank_deficient"]
    assert failures.best_state_retained.to_pylist() == [False]
    assert failures.recoverable.to_pylist() == [False]
    assert failures.orbit_id.to_pylist() == [None]
    assert failures.input_orbit_id.to_pylist() == [orbits.orbit_id[0].as_py()]
