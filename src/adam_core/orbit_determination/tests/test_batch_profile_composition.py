"""Resource-gated exact composition through the shipped Rust-owned profiles."""

import dataclasses
import os

import numpy as np
import pyarrow as pa
import pytest

from adam_core.observations import Associations, SourceCatalog

from .. import (
    ASTEROID_INSTITUTE_OD_V1,
    CORE_OD_V1,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
    evaluate_orbits,
    refine,
    run_iod,
    run_od,
)
from .test_batch_workflows import _assert_exact_results, _two_body_source

_RESOURCE_PATH_ENV = {
    "efcc18_catalog_bias_table": "ADAM_CORE_OD_RESOURCE_EFCC18_CATALOG_BIAS_TABLE",
    "station_catalog_uncertainty_table": (
        "ADAM_CORE_OD_RESOURCE_STATION_CATALOG_UNCERTAINTY_TABLE"
    ),
    "station_band_residual_covariance_table": (
        "ADAM_CORE_OD_RESOURCE_STATION_BAND_RESIDUAL_COVARIANCE_TABLE"
    ),
}


def _bindings(profile) -> ResourceBindings:
    paths = {}
    for requirement in profile.requirements:
        env_name = _RESOURCE_PATH_ENV[requirement.slot]
        path = os.environ.get(env_name)
        if not path:
            pytest.skip(f"semantic resource path {env_name} is not supplied")
        paths[requirement.slot] = path
    # profile.bind below performs the exact Rust size/SHA/schema validation;
    # this helper performs no Python-side parsing or fallback discovery.
    return ResourceBindings(paths)


def _profile_source(pure_iod_orbit, *, fill_missing: bool) -> SourceCatalog:
    base = _two_body_source(pure_iod_orbit)
    source = SourceCatalog.from_pyarrow(
        pa.concat_tables([base.table] * 3).combine_chunks()
    )
    source = source.set_column(
        "id",
        pa.array(
            [
                f"{obs_id}-profile-{repeat}"
                for repeat in range(3)
                for obs_id in base.id.to_pylist()
            ],
            type=pa.large_string(),
        ),
    ).set_column("astcat", pa.array(["UCAC4"] * len(source), type=pa.large_string()))
    if fill_missing:
        ra_sigma = source.ra_sigma.to_pylist()
        dec_sigma = source.dec_sigma.to_pylist()
        for row in (1, 9):
            ra_sigma[row] = None
            dec_sigma[row] = None
        source = source.set_column(
            "ra_sigma", pa.array(ra_sigma, type=pa.float64())
        ).set_column("dec_sigma", pa.array(dec_sigma, type=pa.float64()))
    return source


def _forced_failure(config: OdConfig, bindings: ResourceBindings) -> OdConfig:
    declaration = config.for_refinement().refinement
    fit = dataclasses.replace(
        declaration.fit,
        max_iterations=1,
        max_nfev=1,
        xtol=1e-300,
        ftol=1e-300,
        gtol=1e-300,
    )
    return OdConfig.explicit(
        observation_pipeline=config.observation_pipeline,
        iod=config.for_iod(),
        refinement=RefinementRunConfig(dataclasses.replace(declaration, fit=fit)),
        evaluation=config.for_evaluation(),
        resources=bindings,
    )


@pytest.mark.parametrize("force_failure", [False, True])
@pytest.mark.parametrize(
    "profile",
    [CORE_OD_V1, ASTEROID_INSTITUTE_OD_V1],
    ids=lambda profile: profile.name,
)
def test_shipped_profile_exact_composition_and_evaluation(
    pure_iod_orbit, profile, force_failure
) -> None:
    bindings = _bindings(profile)
    # This is the public Rust validation boundary: every required semantic slot
    # is content-pinned and parsed before any workflow/provider work.
    config = profile.bind(bindings)
    if force_failure:
        config = _forced_failure(config, bindings)
    source = _profile_source(
        pure_iod_orbit,
        fill_missing=profile is ASTEROID_INSTITUTE_OD_V1,
    )
    associations = Associations.from_kwargs(
        detection_id=source.id, object_id=["profile-group"] * len(source)
    )

    fused = run_od(source, associations, None, config)
    iod = run_iod(source, associations, None, config.for_iod())
    assert len(iod[0]) == 1, iod[2].message.to_pylist()
    composed = refine(iod[0], iod[1], source, None, config.for_refinement())
    _assert_exact_results(fused, composed)

    original = iod[1].original_astrometry
    used = iod[1].used_astrometry
    changed_fields = []
    for field in ("lon", "lat", "variance_lon", "variance_lat", "cov_lonlat"):
        before = getattr(original, field).to_numpy(zero_copy_only=False)
        after = getattr(used, field).to_numpy(zero_copy_only=False)
        changed_fields.append(np.any(before.view(np.uint64) != after.view(np.uint64)))
    assert any(changed_fields)
    if not force_failure:
        assert profile.name in fused[0].provenance.configuration[0].as_py()

    if force_failure:
        assert fused[0].success.to_pylist() == [False]
        assert fused[2].best_state_retained.to_pylist() == [True]
    else:
        assert fused[0].success.to_pylist() == [True]
        assert len(fused[2]) == 0

    evaluated_fused = evaluate_orbits(
        fused[0], fused[1], source, None, config.for_evaluation()
    )
    evaluated_composed = evaluate_orbits(
        composed[0], composed[1], source, None, config.for_evaluation()
    )
    evaluated_iod = evaluate_orbits(
        iod[0], iod[1], source, None, config.for_evaluation()
    )
    _assert_exact_results(evaluated_fused, evaluated_composed)
    assert len(evaluated_iod[0]) == 1
