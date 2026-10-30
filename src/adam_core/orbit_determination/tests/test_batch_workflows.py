"""Canonical batch-native workflow composition contracts."""

import numpy as np
import pyarrow as pa
import pytest

from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.observations import Associations, SourceCatalog

from .. import (
    EvaluationConfig,
    IodRunConfig,
    ObservationPipeline,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
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

    for fused_table, composed_table in zip(fused, composed, strict=True):
        assert fused_table.table.equals(composed_table.table)


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
