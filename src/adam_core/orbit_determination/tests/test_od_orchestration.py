"""
Tests for fit-time OD orchestration (`run_od`) and the original / used
astrometry provenance recorded on `FittedOrbitMembers`.

Provenance is proven with models that already exist on the branch
(`IdentityModel`, `EmpiricalCovarianceModel`, `NightBatchDeweightingModel`):
first with a deterministic recording fitter (plumbing, joins, back-compat),
then with a real `NativeOrbitFitter` run (Gauss IOD + differential
correction) on synthetic two-body observations, which needs no external
propagator. Synthetic hand-built inputs only: no real bias numbers appear.
"""

from __future__ import annotations

import importlib
from importlib.resources import files
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest
import quivr as qv

from ...coordinates.covariances import CoordinateCovariances
from ...dynamics.propagation import propagate_2body
from ...orbits import Orbits
from ...propagator.propagator import Propagator
from ...propagator.types import OrbitType, TimestampType
from ...time import Timestamp
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits, ObservationAstrometry
from ..native_orbit_fitter import NativeOrbitFitter
from ..observation_uncertainty import (
    ARCSEC_PER_DEG,
    CompositeModel,
    EmpiricalCovarianceModel,
    IdentityModel,
    NightBatchDeweightingModel,
    ObservationUncertaintyModel,
)
from ..od_orchestration import (
    apply_observation_models,
    attach_observation_provenance,
    run_od,
)
from ..orbit_fitter import OrbitFitter
from .test_observation_uncertainty import make_bias_table, make_observations
from .test_observatory_bias_model_wiring import (
    UNUSED_PROPAGATOR,
    InflatingSpy,
    make_fitted_orbit,
)

ASTROMETRY_FIELDS = ("lon", "lat", "sigma_lon", "sigma_lat", "cov_lonlat")

# Synthetic per-station residual statistics (arcsec², cos(dec)-corrected RA)
RESID_VAR_RA = 0.25
RESID_VAR_DEC = 0.16
RESID_COV_RA_DEC = 0.01


def astrometry_arrays(astrometry: ObservationAstrometry) -> dict[str, np.ndarray]:
    return {
        name: astrometry.table[name].to_numpy(zero_copy_only=False)
        for name in ASTROMETRY_FIELDS
    }


def assert_astrometry_equal(
    first: ObservationAstrometry, second: ObservationAstrometry
) -> None:
    """NaN-aware equality of every astrometry field."""
    first_arrays = astrometry_arrays(first)
    second_arrays = astrometry_arrays(second)
    for name in ASTROMETRY_FIELDS:
        npt.assert_array_equal(first_arrays[name], second_arrays[name], err_msg=name)


def expected_empirical_sigmas(
    sigma_lon: np.ndarray, sigma_lat: np.ndarray, lat_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Used (sigma_lon, sigma_lat, cov_lonlat) for EmpiricalCovarianceModel(mode='add')
    with a NaN (i.e. zero) baseline cross-term."""
    cos_dec = np.cos(np.deg2rad(lat_deg))
    used_sigma_lon = np.sqrt(
        sigma_lon**2 + RESID_VAR_RA / (ARCSEC_PER_DEG**2 * cos_dec**2)
    )
    used_sigma_lat = np.sqrt(sigma_lat**2 + RESID_VAR_DEC / ARCSEC_PER_DEG**2)
    used_cov = RESID_COV_RA_DEC / (ARCSEC_PER_DEG**2 * cos_dec)
    return used_sigma_lon, used_sigma_lat, used_cov


def empirical_model(obs_code: str) -> EmpiricalCovarianceModel:
    table = make_bias_table(
        [
            {
                "obs_code": obs_code,
                "resid_var_ra": RESID_VAR_RA,
                "resid_var_dec": RESID_VAR_DEC,
                "resid_cov_ra_dec": RESID_COV_RA_DEC,
                "resid_cov_n": 100,
            }
        ]
    )
    return EmpiricalCovarianceModel(table)


def with_astcat(
    observations: OrbitDeterminationObservations, astcat: list[str | None]
) -> OrbitDeterminationObservations:
    return observations.set_column("astcat", pa.array(astcat, pa.large_string()))


class RecordingFitter(OrbitFitter):
    """
    Deterministic fitter: records the object ids and observations it receives
    and returns one orbit plus members for every observation (optionally in
    reverse order and with the given ids flagged as outliers). Residuals are
    left null, as a fitter without a residual model would.
    """

    def __init__(
        self, reverse: bool = False, outlier_ids: tuple[str, ...] = ()
    ) -> None:
        self.reverse = reverse
        self.outlier_ids = outlier_ids
        self.received: list[OrbitDeterminationObservations] = []
        self.object_ids: list[Any] = []

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _members(
        self, observations: OrbitDeterminationObservations
    ) -> FittedOrbitMembers:
        ids = observations.id.to_pylist()
        if self.reverse:
            ids = ids[::-1]
        outlier = [obs_id in self.outlier_ids for obs_id in ids]
        return FittedOrbitMembers.from_kwargs(
            orbit_id=["orbit_01"] * len(ids),
            obs_id=ids,
            solution=[not flag for flag in outlier],
            outlier=outlier,
        )

    def initial_fit(
        self,
        object_id: Any,
        observations: OrbitDeterminationObservations,
    ) -> tuple[FittedOrbits, FittedOrbitMembers]:
        self.object_ids.append(object_id)
        self.received.append(observations)
        return make_fitted_orbit(), self._members(observations)

    def refine_fit(
        self,
        fitted_orbit: FittedOrbits,
        observations: OrbitDeterminationObservations,
        propagator: Any,
    ) -> tuple[FittedOrbits, FittedOrbitMembers]:
        self.received.append(observations)
        return fitted_orbit, self._members(observations)


class TestObservationAstrometry:
    def test_from_spherical_snapshots_position_and_uncertainty(self) -> None:
        observations = make_observations(
            ["500", "F51"],
            [0.0, 30.0],
            sigma_lon_deg=1e-4,
            sigma_lat_deg=2e-4,
            cov_lonlat_deg2=1e-9,
        )
        astrometry = ObservationAstrometry.from_spherical(observations.coordinates)

        arrays = astrometry_arrays(astrometry)
        npt.assert_array_equal(arrays["lon"], observations.coordinates.lon.to_numpy())
        npt.assert_array_equal(arrays["lat"], observations.coordinates.lat.to_numpy())
        npt.assert_allclose(arrays["sigma_lon"], 1e-4, rtol=1e-12)
        npt.assert_allclose(arrays["sigma_lat"], 2e-4, rtol=1e-12)
        npt.assert_allclose(arrays["cov_lonlat"], 1e-9, rtol=1e-12)

    def test_from_spherical_missing_covariance_terms_are_nan(self) -> None:
        observations = make_observations(["500"], [0.0])  # no cross-term
        coordinates = observations.coordinates.set_column(
            "covariance",
            CoordinateCovariances.from_matrix(np.full((1, 6, 6), np.nan)),
        )
        astrometry = ObservationAstrometry.from_spherical(coordinates)

        arrays = astrometry_arrays(astrometry)
        assert np.isnan(arrays["sigma_lon"]).all()
        assert np.isnan(arrays["sigma_lat"]).all()
        assert np.isnan(arrays["cov_lonlat"]).all()
        assert np.isnan(
            astrometry_arrays(
                ObservationAstrometry.from_spherical(observations.coordinates)
            )["cov_lonlat"]
        ).all()

    def test_from_spherical_empty(self) -> None:
        empty = ObservationAstrometry.from_spherical(
            OrbitDeterminationObservations.empty().coordinates
        )
        assert len(empty) == 0


class TestFittedOrbitMembersBackCompat:
    def test_members_without_provenance_have_null_columns(self) -> None:
        members = FittedOrbitMembers.from_kwargs(
            orbit_id=["o", "o"], obs_id=["a", "b"], outlier=[False, True]
        )
        for name in ("original_astrometry", "used_astrometry", "astcat"):
            assert members.table[name].null_count == 2
        assert members.original_astrometry.lon.to_pylist() == [None, None]
        assert members.outlier.to_pylist() == [False, True]

    def test_members_with_and_without_provenance_concatenate(self) -> None:
        observations = make_observations(["500", "F51"], [0.0, 30.0])
        plain = FittedOrbitMembers.from_kwargs(orbit_id=["o", "o"], obs_id=["a", "b"])
        enriched = attach_observation_provenance(
            FittedOrbitMembers.from_kwargs(
                orbit_id=["p", "p"], obs_id=observations.id.to_pylist()
            ),
            observations,
            observations,
        )

        combined = qv.concatenate([plain, enriched])
        assert len(combined) == 4
        assert combined.table["original_astrometry"].null_count == 2
        npt.assert_array_equal(
            combined.used_astrometry.lon.to_numpy(zero_copy_only=False)[2:],
            observations.coordinates.lon.to_numpy(),
        )

    def test_legacy_members_parquet_loads_with_null_provenance(self) -> None:
        # Pinned file predates the provenance columns.
        members = FittedOrbitMembers.from_parquet(
            str(
                files("adam_core.orbit_determination.tests.data").joinpath(
                    "pure_iod_orbit_members.parquet"
                )
            )
        )
        assert len(members) == 7
        assert members.table["original_astrometry"].null_count == 7
        assert members.table["used_astrometry"].null_count == 7
        assert members.table["astcat"].null_count == 7
        assert members.outlier.to_pylist() == [False] * 7

    def test_direct_fitter_path_leaves_provenance_null(self) -> None:
        # Fitters called directly (no run_od) keep working and record no
        # provenance: the new columns are null-safe.
        observations = make_observations(["500", "F51", "W84"], [0.0, 30.0, -45.0])
        fitter = RecordingFitter(outlier_ids=("obs_02",))
        _, members = fitter.full_od("obj", observations, UNUSED_PROPAGATOR)

        assert members.outlier.to_pylist() == [False, False, True]
        assert members.solution.to_pylist() == [True, True, False]
        assert members.table["original_astrometry"].null_count == 3
        assert members.table["used_astrometry"].null_count == 3
        assert members.table["astcat"].null_count == 3


class TestApplyObservationModels:
    def test_none_and_empty_sequence_return_input(self) -> None:
        observations = make_observations(["500"], [0.0])
        assert apply_observation_models(observations, None) is observations
        assert apply_observation_models(observations, []) is observations

    def test_sequence_is_applied_in_order_like_a_composite(self) -> None:
        observations = make_observations(["F51", "500"], [30.0, 0.0])
        model = empirical_model("F51")
        spy = InflatingSpy(4.0)

        used = apply_observation_models(observations, [model, spy])
        composite = CompositeModel(model, spy).apply(observations)

        npt.assert_array_equal(
            used.coordinates.covariance.to_matrix(),
            composite.coordinates.covariance.to_matrix(),
        )
        # (baseline + measured residual variance) * 4 for the F51 row: the
        # inflation is applied after the addition, so order is observable.
        sigma_lon = observations.coordinates.covariance.sigmas[0, 1]
        cos_dec = np.cos(np.deg2rad(30.0))
        expected = 4.0 * (
            sigma_lon**2 + RESID_VAR_RA / (ARCSEC_PER_DEG**2 * cos_dec**2)
        )
        npt.assert_allclose(
            used.coordinates.covariance.to_matrix()[0, 1, 1], expected, rtol=1e-12
        )

    def test_model_dropping_observations_raises(self) -> None:
        class Dropping(ObservationUncertaintyModel):
            def apply(
                self, observations: OrbitDeterminationObservations
            ) -> OrbitDeterminationObservations:
                return observations[:-1]

        observations = make_observations(["500", "F51"], [0.0, 30.0])
        with pytest.raises(ValueError, match="preserve the observations' ids"):
            apply_observation_models(observations, Dropping())

    def test_model_reordering_observations_raises(self) -> None:
        class Reordering(ObservationUncertaintyModel):
            def apply(
                self, observations: OrbitDeterminationObservations
            ) -> OrbitDeterminationObservations:
                return observations.take(pa.array([1, 0]))

        observations = make_observations(["500", "F51"], [0.0, 30.0])
        with pytest.raises(ValueError, match="preserve the observations' ids"):
            apply_observation_models(observations, Reordering())


class TestRunOdProvenance:
    """Provenance plumbing with a deterministic fitter."""

    def test_identity_model_records_used_equal_to_original(self) -> None:
        observations = with_astcat(
            make_observations(["500", "F51", "W84"], [0.0, 30.0, -45.0]),
            ["Gaia2", "UCAC4", None],
        )
        fitter = RecordingFitter()

        fitted_orbits, members = run_od(
            observations, fitter, IdentityModel(), propagator=UNUSED_PROPAGATOR
        )

        assert len(fitted_orbits) == 1
        assert len(members) == 3
        assert members.obs_id.to_pylist() == observations.id.to_pylist()
        assert_astrometry_equal(members.original_astrometry, members.used_astrometry)
        assert_astrometry_equal(
            members.original_astrometry,
            ObservationAstrometry.from_spherical(observations.coordinates),
        )
        assert members.astcat.to_pylist() == ["Gaia2", "UCAC4", None]
        # A true no-op: the fitter saw the very observations that were passed in.
        assert all(received is observations for received in fitter.received)

    def test_empirical_covariance_inflates_used_sigma_and_keeps_positions(
        self,
    ) -> None:
        observations = make_observations(["500", "F51", "F51"], [0.0, 30.0, -45.0])
        fitter = RecordingFitter()

        _, members = run_od(
            observations, fitter, empirical_model("F51"), propagator=UNUSED_PROPAGATOR
        )

        original = astrometry_arrays(members.original_astrometry)
        used = astrometry_arrays(members.used_astrometry)
        # Positions never change under a sigma-only model
        npt.assert_array_equal(used["lon"], original["lon"])
        npt.assert_array_equal(used["lat"], original["lat"])
        # Original sigmas are the input sigmas
        npt.assert_allclose(original["sigma_lon"], 1e-4, rtol=1e-12)
        npt.assert_allclose(original["sigma_lat"], 2e-4, rtol=1e-12)
        assert np.isnan(original["cov_lonlat"]).all()
        # Station 500 is absent from the table: used == original
        assert used["sigma_lon"][0] == original["sigma_lon"][0]
        assert used["sigma_lat"][0] == original["sigma_lat"][0]
        assert np.isnan(used["cov_lonlat"][0])
        # F51 rows: measured residual covariance added to the baseline
        lat = original["lat"][1:]
        expected_lon, expected_lat, expected_cov = expected_empirical_sigmas(
            original["sigma_lon"][1:], original["sigma_lat"][1:], lat
        )
        npt.assert_allclose(used["sigma_lon"][1:], expected_lon, rtol=1e-12)
        npt.assert_allclose(used["sigma_lat"][1:], expected_lat, rtol=1e-12)
        npt.assert_allclose(used["cov_lonlat"][1:], expected_cov, rtol=1e-12)
        assert (used["sigma_lon"][1:] > original["sigma_lon"][1:]).all()
        assert (used["sigma_lat"][1:] > original["sigma_lat"][1:]).all()
        # The fitter was handed exactly the used uncertainties, never the model
        for received in fitter.received:
            covariances = received.coordinates.covariance.to_matrix()
            npt.assert_allclose(
                np.sqrt(covariances[:, 1, 1]), used["sigma_lon"], rtol=1e-12
            )
            npt.assert_allclose(
                np.sqrt(covariances[:, 2, 2]), used["sigma_lat"], rtol=1e-12
            )

    def test_night_batch_deweighting_records_scaled_sigma(self) -> None:
        # Six same-station, same-night observations with cap 4: factor 1.5 on
        # the covariance, i.e. sqrt(1.5) on the sigmas.
        observations = make_observations(["W84"] * 6, [10.0] * 6)

        _, members = run_od(
            observations,
            RecordingFitter(),
            NightBatchDeweightingModel(cap=4),
            propagator=UNUSED_PROPAGATOR,
        )

        original = astrometry_arrays(members.original_astrometry)
        used = astrometry_arrays(members.used_astrometry)
        npt.assert_allclose(
            used["sigma_lon"], np.sqrt(1.5) * original["sigma_lon"], rtol=1e-12
        )
        npt.assert_allclose(
            used["sigma_lat"], np.sqrt(1.5) * original["sigma_lat"], rtol=1e-12
        )
        npt.assert_array_equal(used["lon"], original["lon"])

    def test_no_models_records_originals_and_keeps_fitter_flags(self) -> None:
        observations = make_observations(["500", "F51", "W84"], [0.0, 30.0, -45.0])
        fitter = RecordingFitter(outlier_ids=("obs_02",))

        _, members = run_od(observations, fitter, propagator=UNUSED_PROPAGATOR)

        assert members.outlier.to_pylist() == [False, False, True]
        assert members.solution.to_pylist() == [True, True, False]
        assert members.table["residuals"].null_count == 3  # fitter set none
        assert_astrometry_equal(members.original_astrometry, members.used_astrometry)
        assert members.table["original_astrometry"].null_count == 0

    def test_provenance_joins_by_obs_id_when_fitter_reorders_members(self) -> None:
        observations = with_astcat(
            make_observations(["500", "F51", "W84"], [0.0, 30.0, -45.0]),
            ["A", "B", "C"],
        )

        _, members = run_od(
            observations,
            RecordingFitter(reverse=True),
            IdentityModel(),
            propagator=UNUSED_PROPAGATOR,
        )

        assert members.obs_id.to_pylist() == observations.id.to_pylist()[::-1]
        lon_by_id = dict(
            zip(observations.id.to_pylist(), observations.coordinates.lon.to_pylist())
        )
        astcat_by_id = dict(
            zip(observations.id.to_pylist(), observations.astcat.to_pylist())
        )
        for obs_id, lon, astcat in zip(
            members.obs_id.to_pylist(),
            members.original_astrometry.lon.to_pylist(),
            members.astcat.to_pylist(),
        ):
            assert lon == lon_by_id[obs_id]
            assert astcat == astcat_by_id[obs_id]

    def test_member_with_unknown_obs_id_raises(self) -> None:
        observations = make_observations(["500"], [0.0])
        members = FittedOrbitMembers.from_kwargs(orbit_id=["o"], obs_id=["bogus"])
        with pytest.raises(ValueError, match="absent from the original"):
            attach_observation_provenance(members, observations, observations)

    def test_duplicate_observation_ids_raise(self) -> None:
        observations = make_observations(["500", "F51"], [0.0, 30.0])
        observations = observations.set_column(
            "id", pa.array(["dup", "dup"], pa.large_string())
        )
        with pytest.raises(ValueError, match="unique"):
            run_od(observations, RecordingFitter(), propagator=UNUSED_PROPAGATOR)

    def test_empty_observations_return_empty_tables(self) -> None:
        fitter = RecordingFitter()
        fitted_orbits, members = run_od(
            OrbitDeterminationObservations.empty(), fitter, propagator=UNUSED_PROPAGATOR
        )
        assert len(fitted_orbits) == 0
        assert len(members) == 0
        assert fitter.received == []

    def test_object_id_is_passed_to_backend_and_stamped_when_null(self) -> None:
        observations = make_observations(["500"], [0.0])
        fitter = RecordingFitter()

        fitted_orbits, _ = run_od(
            observations, fitter, propagator=UNUSED_PROPAGATOR, object_id="2024 XY"
        )
        assert fitter.object_ids == ["2024 XY"]
        assert fitted_orbits.object_id.to_pylist() == ["2024 XY"]

        fitter = RecordingFitter()
        fitted_orbits, _ = run_od(observations, fitter, propagator=UNUSED_PROPAGATOR)
        assert isinstance(fitter.object_ids[0], str) and fitter.object_ids[0]
        assert fitted_orbits.object_id.to_pylist() == [None]

    def test_run_od_is_exported_from_the_package(self) -> None:
        package = importlib.import_module("adam_core.orbit_determination")
        assert package.run_od is run_od
        assert package.ObservationAstrometry is ObservationAstrometry


class TwoBodyPropagator(Propagator):
    """Minimal Keplerian propagator so the native fitter runs without plugins."""

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _propagate_orbits(self, orbits: OrbitType, times: TimestampType) -> OrbitType:
        assert isinstance(orbits, Orbits)
        assert isinstance(times, Timestamp)
        return propagate_2body(orbits, times)


SYNTHETIC_SIGMA_DEG = 0.3 / ARCSEC_PER_DEG


@pytest.fixture
def two_body_observations(
    pure_iod_orbit: tuple[
        FittedOrbits, FittedOrbitMembers, OrbitDeterminationObservations
    ],
) -> OrbitDeterminationObservations:
    """
    Seven W84 observations over 21 days generated from the pinned IOD orbit
    with two-body dynamics, uniform 0.3" sigmas and 0.3-sigma Gaussian noise.
    Exactly consistent with TwoBodyPropagator, so the native fit converges.
    """
    orbit, _, observations = pure_iod_orbit
    ephemeris = TwoBodyPropagator().generate_ephemeris(
        orbit.to_orbits(), observations.observers, max_processes=1
    )
    assert ephemeris.coordinates.time.equals(observations.coordinates.time)

    n = len(observations)
    rng = np.random.default_rng(42)
    lon = ephemeris.coordinates.lon.to_numpy() + 0.3 * SYNTHETIC_SIGMA_DEG * (
        rng.standard_normal(n)
    )
    lat = ephemeris.coordinates.lat.to_numpy() + 0.3 * SYNTHETIC_SIGMA_DEG * (
        rng.standard_normal(n)
    )
    covariances = np.full((n, 6, 6), np.nan)
    covariances[:, 1, 1] = SYNTHETIC_SIGMA_DEG**2
    covariances[:, 2, 2] = SYNTHETIC_SIGMA_DEG**2
    return with_astcat(
        observations.set_column("coordinates.lon", pa.array(lon))
        .set_column("coordinates.lat", pa.array(lat))
        .set_column(
            "coordinates.covariance", CoordinateCovariances.from_matrix(covariances)
        ),
        ["Gaia2"] * n,
    )


def native_fitter() -> NativeOrbitFitter:
    # A Gauss triplet orbit over a 21-day arc leaves arcsecond-level residuals
    # at 0.3" sigmas, so the IOD acceptance threshold is relaxed; DC then
    # converges to the two-body truth.
    return NativeOrbitFitter(
        propagator_class=TwoBodyPropagator,
        min_obs=6,
        rchi2_threshold=10.0,
        iod_rchi2_threshold=1e4,
    )


class TestRunOdNativeOrbitFitter:
    """Real Gauss IOD + differential correction through run_od."""

    def test_empirical_covariance_records_used_sigma_and_is_consumed_by_the_fit(
        self, two_body_observations: OrbitDeterminationObservations
    ) -> None:
        propagator = TwoBodyPropagator()

        orbits_plain, members_plain = run_od(
            two_body_observations, native_fitter(), propagator=propagator
        )
        orbits_model, members_model = run_od(
            two_body_observations,
            native_fitter(),
            empirical_model("W84"),
            propagator=propagator,
            object_id="synthetic",
        )

        for orbits in (orbits_plain, orbits_model):
            assert len(orbits) == 1
            assert orbits.success[0].as_py() is True
            assert orbits.reduced_chi2[0].as_py() < 10.0
        assert orbits_model.object_id.to_pylist() == ["synthetic"]

        assert len(members_model) == len(two_body_observations)
        assert set(members_model.obs_id.to_pylist()) == set(
            two_body_observations.id.to_pylist()
        )
        assert members_model.table["residuals"].null_count == 0
        assert members_model.outlier.to_pylist() == [False] * len(members_model)
        assert members_model.astcat.to_pylist() == ["Gaia2"] * len(members_model)

        original = astrometry_arrays(members_model.original_astrometry)
        used = astrometry_arrays(members_model.used_astrometry)
        npt.assert_array_equal(used["lon"], original["lon"])
        npt.assert_array_equal(used["lat"], original["lat"])
        npt.assert_allclose(original["sigma_lon"], SYNTHETIC_SIGMA_DEG, rtol=1e-12)
        npt.assert_allclose(original["sigma_lat"], SYNTHETIC_SIGMA_DEG, rtol=1e-12)
        expected_lon, expected_lat, expected_cov = expected_empirical_sigmas(
            original["sigma_lon"], original["sigma_lat"], original["lat"]
        )
        npt.assert_allclose(used["sigma_lon"], expected_lon, rtol=1e-12)
        npt.assert_allclose(used["sigma_lat"], expected_lat, rtol=1e-12)
        npt.assert_allclose(used["cov_lonlat"], expected_cov, rtol=1e-12)
        assert (used["sigma_lon"] > original["sigma_lon"]).all()

        # The plain fit records used == original ...
        assert_astrometry_equal(
            members_plain.original_astrometry, members_plain.used_astrometry
        )
        # ... and the inflated sigmas really reached the fit: same residuals
        # in the sky, smaller chi2.
        assert (
            orbits_model.reduced_chi2[0].as_py() < orbits_plain.reduced_chi2[0].as_py()
        )

    def test_identity_model_matches_no_model_fit(
        self, two_body_observations: OrbitDeterminationObservations
    ) -> None:
        propagator = TwoBodyPropagator()
        orbits_plain, _ = run_od(
            two_body_observations, native_fitter(), propagator=propagator
        )
        orbits_identity, members = run_od(
            two_body_observations,
            native_fitter(),
            IdentityModel(),
            propagator=propagator,
        )
        npt.assert_array_equal(
            orbits_identity.coordinates.values, orbits_plain.coordinates.values
        )
        assert_astrometry_equal(members.original_astrometry, members.used_astrometry)


def test_efcc18_composite_records_shifted_positions_and_inflated_sigma(
    two_body_observations: OrbitDeterminationObservations,
) -> None:
    """
    End-to-end provenance with a position-modifying model:
    CompositeModel(EFCC18DebiasModel, EmpiricalCovarianceModel) must leave
    members whose used lon/lat are the EFCC18-shifted positions (original
    lon/lat untouched) and whose used sigmas are inflated by the station's
    residual covariance.

    Self-gated: skips until the ``adam_core.observations.efcc18`` module (bead
    v32) is present on this branch and the published ``bias.dat`` is installed
    (not bundled; resolved via ``ADAM_CORE_EFCC18_BIAS_DAT`` or the EFCC18 cache
    directory), mirroring bead v32's own real-file test.
    """
    efcc18 = pytest.importorskip("adam_core.observations.efcc18")
    try:
        bias_dat = efcc18.resolve_bias_dat()
    except FileNotFoundError:
        pytest.skip("EFCC18 bias.dat not installed")
    uncertainty = importlib.import_module(
        "adam_core.orbit_determination.observation_uncertainty"
    )
    debias = cast(
        ObservationUncertaintyModel, uncertainty.EFCC18DebiasModel(bias_dat=bias_dat)
    )

    # UCAC4 is tabulated in EFCC18 with non-trivial corrections at this epoch.
    n = len(two_body_observations)
    observations = with_astcat(two_body_observations, ["UCAC4"] * n)
    debiased = debias.apply(observations)
    assert debiased is not observations, "EFCC18 did not correct UCAC4 positions"

    _, members = run_od(
        observations,
        native_fitter(),
        CompositeModel(debias, empirical_model("W84")),
        propagator=TwoBodyPropagator(),
    )

    # Compare in observation order regardless of the member order returned.
    members = members.sort_by("obs_id")
    observations = observations.sort_by("id")
    debiased = debiased.sort_by("id")
    assert members.obs_id.to_pylist() == observations.id.to_pylist()
    assert members.table["residuals"].null_count == 0
    assert members.astcat.to_pylist() == ["UCAC4"] * n

    original = astrometry_arrays(members.original_astrometry)
    used = astrometry_arrays(members.used_astrometry)
    # Original == input positions; used == EFCC18-shifted positions, and the
    # shift is real but sub-arcsecond-scale.
    npt.assert_array_equal(original["lon"], observations.coordinates.lon.to_numpy())
    npt.assert_array_equal(original["lat"], observations.coordinates.lat.to_numpy())
    npt.assert_array_equal(used["lon"], debiased.coordinates.lon.to_numpy())
    npt.assert_array_equal(used["lat"], debiased.coordinates.lat.to_numpy())
    shift_ra_arcsec = (
        (used["lon"] - original["lon"])
        * np.cos(np.deg2rad(original["lat"]))
        * ARCSEC_PER_DEG
    )
    shift_dec_arcsec = (used["lat"] - original["lat"]) * ARCSEC_PER_DEG
    assert np.any(shift_ra_arcsec != 0.0) or np.any(shift_dec_arcsec != 0.0)
    assert np.max(np.abs(shift_ra_arcsec)) < 5.0
    assert np.max(np.abs(shift_dec_arcsec)) < 5.0
    # Used sigmas: original sigmas inflated by the W84 residual covariance
    # (EFCC18 leaves the covariance block untouched).
    npt.assert_allclose(original["sigma_lon"], SYNTHETIC_SIGMA_DEG, rtol=1e-12)
    expected_lon, expected_lat, expected_cov = expected_empirical_sigmas(
        original["sigma_lon"], original["sigma_lat"], used["lat"]
    )
    npt.assert_allclose(used["sigma_lon"], expected_lon, rtol=1e-12)
    npt.assert_allclose(used["sigma_lat"], expected_lat, rtol=1e-12)
    npt.assert_allclose(used["cov_lonlat"], expected_cov, rtol=1e-12)
