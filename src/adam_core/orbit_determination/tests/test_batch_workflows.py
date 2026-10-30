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
    used_variance = iod[1].used_astrometry.variance_lon.to_numpy(zero_copy_only=False)
    assert np.any(original_variance.view(np.uint64) != used_variance.view(np.uint64))
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
