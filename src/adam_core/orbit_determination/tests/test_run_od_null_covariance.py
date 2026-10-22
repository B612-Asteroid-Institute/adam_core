"""
`run_od` with observations that carry no covariance (PR #217 review round 2,
phase 1, item 3): the one-crossing Rust route must run observation models
before requiring a finite covariance, so that `SigmaFillModel` can supply
the missing sigmas exactly as when adam-core applies the model first; without
a model that fills them the fit is refused with an error naming the remedy.
"""

from __future__ import annotations

import warnings

import numpy as np
import pyarrow as pa
import pytest

from ...coordinates.covariances import CoordinateCovariances
from ..evaluate import OrbitDeterminationObservations
from ..model_provenance import ModelProvenance
from ..native_orbit_fitter import NativeOrbitFitter
from ..od_orchestration import run_od
from ..veres2017 import VERES2017_SIGMA_TABLE_SCHEMA, SigmaFillModel
from .test_differential_correction import TwoBodyPropagator

FILL_SIGMA_ARCSEC = 0.3
SIGMA_DEG = 0.3 / 3600.0


@pytest.fixture
def two_body_arc(pure_iod_orbit) -> OrbitDeterminationObservations:
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


def global_fill_table() -> pa.Table:
    return pa.table(
        {
            "obs_code": pa.array([None], pa.large_string()),
            "astcat": pa.array([None], pa.large_string()),
            "sigma_ra_arcsec": pa.array([FILL_SIGMA_ARCSEC], pa.float64()),
            "sigma_dec_arcsec": pa.array([FILL_SIGMA_ARCSEC], pa.float64()),
        },
        schema=VERES2017_SIGMA_TABLE_SCHEMA,
    )


@pytest.fixture
def null_covariance_arc(two_body_arc):
    return two_body_arc.set_column(
        "coordinates.covariance", CoordinateCovariances.nulls(len(two_body_arc))
    )


def fitter() -> NativeOrbitFitter:
    return NativeOrbitFitter(
        TwoBodyPropagator(),
        min_obs=6,
        rchi2_threshold=10.0,
        iod_rchi2_threshold=1e4,
    )


def test_sigma_fill_supplies_the_missing_covariance(null_covariance_arc) -> None:
    model = SigmaFillModel(global_fill_table(), fallback_sigma_arcsec=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted, members = run_od(
            null_covariance_arc,
            fitter(),
            [model],
            propagator=TwoBodyPropagator(),
            object_id="null-cov",
        )
    assert len(fitted) == 1
    assert fitted.success[0].as_py()
    assert len(members) == len(null_covariance_arc)
    original = members.original_astrometry
    used = members.used_astrometry
    assert np.all(np.isnan(original.sigma_lon.to_numpy(zero_copy_only=False)))
    assert np.all(np.isnan(original.sigma_lat.to_numpy(zero_copy_only=False)))
    np.testing.assert_allclose(
        used.sigma_lat.to_numpy(zero_copy_only=False),
        FILL_SIGMA_ARCSEC / 3600.0,
        rtol=1e-12,
    )
    assert np.all(np.isfinite(used.sigma_lon.to_numpy(zero_copy_only=False)))


def test_without_a_filling_model_the_fit_is_refused(null_covariance_arc) -> None:
    with pytest.raises(ValueError, match="SigmaFillModel"):
        run_od(null_covariance_arc, fitter(), None, propagator=TwoBodyPropagator())


def test_native_and_python_applied_model_routes_agree_with_filled_sigmas(
    null_covariance_arc,
) -> None:
    model = SigmaFillModel(global_fill_table(), fallback_sigma_arcsec=None)

    class PythonAppliedSigmaFillModel(SigmaFillModel):
        """Same operation, with an override that disables model fusion."""

        def apply(self, observations):
            return super().apply(observations)

        def model_provenance(self):
            return ModelProvenance("adam-core-tests", "1", "5" * 64, dict(vars(self)))

    python_model = PythonAppliedSigmaFillModel(
        global_fill_table(), fallback_sigma_arcsec=None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fused, fused_members = run_od(
            null_covariance_arc, fitter(), [model], propagator=TwoBodyPropagator()
        )
        composed, composed_members = run_od(
            null_covariance_arc,
            fitter(),
            [python_model],
            propagator=TwoBodyPropagator(),
        )
    np.testing.assert_allclose(
        fused.coordinates.values, composed.coordinates.values, rtol=1e-7, atol=1e-12
    )
    assert fused_members.outlier.to_pylist() == composed_members.outlier.to_pylist()
    assert fused.provenance.route.to_pylist() == ["fused"]
    assert composed.provenance.route.to_pylist() == ["composed"]
    for name in ("sigma_lon", "sigma_lat"):
        np.testing.assert_allclose(
            getattr(fused_members.used_astrometry, name).to_numpy(zero_copy_only=False),
            getattr(composed_members.used_astrometry, name).to_numpy(
                zero_copy_only=False
            ),
            rtol=1e-12,
        )
