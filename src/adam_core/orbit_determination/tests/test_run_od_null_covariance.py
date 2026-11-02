"""Missing astrometric covariance fails closed without an explicit fill transform."""

from __future__ import annotations

import numpy as np
import pytest

from adam_core._rust.arrow import pa

from ...coordinates.covariances import CoordinateCovariances
from ..algorithms.fitting import fit_least_squares
from ..config import (
    EvaluationConfig,
    IodRunConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
)
from ..evaluate import OrbitDeterminationObservations
from ..observation_pipeline import (
    FillMissingAstrometricUncertainty,
    ObservationPipeline,
)
from .conftest import PureIodOrbit
from .test_differential_correction import TwoBodyPropagator, make_initial_guess

FILL_SIGMA_ARCSEC = 0.3
SIGMA_DEG = 0.3 / 3600.0


@pytest.fixture
def two_body_arc(pure_iod_orbit: PureIodOrbit) -> OrbitDeterminationObservations:
    """
    The pinned IOD arc (seven W84 observations over 21 days) re-observed
    with two-body dynamics, uniform 0.3" sigmas and 0.3-sigma noise, so the
    two-body IOD and refinement are exactly consistent with the data.
    """
    orbit, _, observations = pure_iod_orbit
    ephemeris = TwoBodyPropagator().generate_ephemeris(
        orbit.to_orbits(), observations.observers, max_processes=1
    )
    n = len(observations)
    rng = np.random.default_rng(7)
    lon = ephemeris.coordinates.lon.to_numpy() + 0.3 * SIGMA_DEG * rng.standard_normal(
        n
    )
    lat = ephemeris.coordinates.lat.to_numpy() + 0.3 * SIGMA_DEG * rng.standard_normal(
        n
    )
    covariances = np.full((n, 6, 6), np.nan)
    covariances[:, 1, 1] = SIGMA_DEG**2
    covariances[:, 2, 2] = SIGMA_DEG**2
    return (
        observations.set_column("coordinates.lon", pa.array(lon))
        .set_column("coordinates.lat", pa.array(lat))
        .set_column(
            "coordinates.covariance", CoordinateCovariances.from_matrix(covariances)
        )
    )


@pytest.fixture
def null_covariance_arc(
    two_body_arc: OrbitDeterminationObservations,
) -> OrbitDeterminationObservations:
    return two_body_arc.set_column(
        "coordinates.covariance", CoordinateCovariances.nulls(len(two_body_arc))
    )


def test_without_a_fill_transform_the_fit_is_refused(
    null_covariance_arc: OrbitDeterminationObservations,
) -> None:
    with pytest.raises(ValueError, match="covariance|finite|uncertainty"):
        fit_least_squares(
            make_initial_guess(), null_covariance_arc, TwoBodyPropagator()
        )


def test_fill_transform_requires_explicit_validated_resource() -> None:
    with pytest.raises(Exception, match="station_catalog_uncertainty_table"):
        OdConfig.explicit(
            observation_pipeline=ObservationPipeline.ordered(
                [FillMissingAstrometricUncertainty()]
            ),
            iod=IodRunConfig.core_v1(),
            refinement=RefinementRunConfig.core_v1(),
            evaluation=EvaluationConfig.six_dimensional(),
            resources=ResourceBindings({}),
        )
