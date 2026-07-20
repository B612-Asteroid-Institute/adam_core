from typing import Callable, Optional

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest

from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.origin import Origin
from ...coordinates.spherical import SphericalCoordinates
from ...observers import Observers
from ...time import Timestamp
from ..evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry
from ..observation_uncertainty import (
    ARCSEC_PER_DEG,
    BIAS_TABLE_SCHEMA,
    EmpiricalCovarianceModel,
    IdentityModel,
    ObservationUncertaintyModel,
    PerformanceWeightedModel,
    SigmaFloorModel,
    assert_positions_unchanged,
    validate_bias_table,
)

# Synthetic hand-built inputs only: no real bias numbers appear in these tests.

_ROW_DEFAULTS: dict[str, object] = {
    "band": None,
    "n_obs": 1000,
    "n_objects": 100,
    "bias_ra_arcsec": 0.0,
    "bias_ra_ci_low": 0.0,
    "bias_ra_ci_high": 0.0,
    "bias_dec_arcsec": 0.0,
    "bias_dec_ci_low": 0.0,
    "bias_dec_ci_high": 0.0,
    "resid_var_ra": 0.0,
    "resid_var_dec": 0.0,
    "resid_cov_ra_dec": 0.0,
    "resid_cov_n": 100,
    "rms_ra_arcsec": 0.5,
    "rms_dec_arcsec": 0.5,
    "chi2_per_obs": 1.0,
    "bias_significant": False,
    "high_confidence": True,
    "confidence_score": 1.0,
}


def make_bias_table(rows: list[dict[str, object]]) -> pa.Table:
    """Build a synthetic bias table from per-row overrides of _ROW_DEFAULTS."""
    full_rows = []
    for overrides in rows:
        row = dict(_ROW_DEFAULTS)
        row.update(overrides)
        full_rows.append(row)
    columns = {
        name: [row[name] for row in full_rows] for name in BIAS_TABLE_SCHEMA.names
    }
    return pa.table(columns, schema=BIAS_TABLE_SCHEMA)


def make_observations(
    codes: list[str],
    lats: list[float],
    sigma_lon_deg: float = 1e-4,
    sigma_lat_deg: float = 2e-4,
    cov_lonlat_deg2: Optional[float] = None,
    bands: Optional[list[Optional[str]]] = None,
) -> OrbitDeterminationObservations:
    """Build synthetic observations at the given stations and declinations."""
    n = len(codes)
    times = Timestamp.from_mjd(60000.0 + 0.01 * np.arange(n), scale="utc")
    covariances = np.full((n, 6, 6), np.nan)
    covariances[:, 1, 1] = sigma_lon_deg**2
    covariances[:, 2, 2] = sigma_lat_deg**2
    if cov_lonlat_deg2 is not None:
        covariances[:, 1, 2] = cov_lonlat_deg2
        covariances[:, 2, 1] = cov_lonlat_deg2
    coordinates = SphericalCoordinates.from_kwargs(
        lon=10.0 + np.arange(n),
        lat=np.asarray(lats, dtype=np.float64),
        covariance=CoordinateCovariances.from_matrix(covariances),
        time=times,
        origin=Origin.from_kwargs(code=codes),
        frame="equatorial",
    )
    observers = Observers.from_codes(times=times, codes=codes)
    if bands is None:
        bands = [None] * n
    photometry = OrbitDeterminationPhotometry.from_kwargs(
        mag=[None] * n, rmsmag=[None] * n, band=bands
    )
    return OrbitDeterminationObservations.from_kwargs(
        id=[f"obs_{i:02d}" for i in range(n)],
        coordinates=coordinates,
        observers=observers,
        photometry=photometry,
    )


def assert_observations_identical(
    before: OrbitDeterminationObservations,
    after: OrbitDeterminationObservations,
) -> None:
    """NaN-aware full equality (pa.Table.equals treats NaN != NaN)."""
    assert_positions_unchanged(before, after)
    npt.assert_array_equal(
        before.coordinates.covariance.to_matrix(),
        after.coordinates.covariance.to_matrix(),
    )


class TestValidateBiasTable:
    def test_valid_table_passes(self) -> None:
        table = make_bias_table([{"obs_code": "500"}, {"obs_code": "F51"}])
        validated = validate_bias_table(table)
        assert validated.schema.equals(BIAS_TABLE_SCHEMA)
        assert len(validated) == 2

    def test_missing_column_raises(self) -> None:
        table = make_bias_table([{"obs_code": "500"}]).drop_columns(["resid_var_ra"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_bias_table(table)

    def test_uncastable_column_raises(self) -> None:
        table = make_bias_table([{"obs_code": "500"}])
        index = table.column_names.index("resid_var_ra")
        table = table.set_column(
            index, "resid_var_ra", pa.array(["not a number"], pa.large_string())
        )
        with pytest.raises(ValueError, match="could not be cast"):
            validate_bias_table(table)

    def test_null_obs_code_raises(self) -> None:
        table = make_bias_table([{"obs_code": "500"}])
        index = table.column_names.index("obs_code")
        table = table.set_column(index, "obs_code", pa.array([None], pa.large_string()))
        with pytest.raises(ValueError, match="obs_code"):
            validate_bias_table(table)

    def test_negative_variance_raises(self) -> None:
        table = make_bias_table([{"obs_code": "500", "resid_var_ra": -0.1}])
        with pytest.raises(ValueError, match="resid_var_ra"):
            validate_bias_table(table)

    def test_extra_columns_dropped(self) -> None:
        table = make_bias_table([{"obs_code": "500"}])
        table = table.append_column("extra", pa.array([1.0]))
        validated = validate_bias_table(table)
        assert "extra" not in validated.column_names

    def test_compatible_types_cast(self) -> None:
        table = make_bias_table([{"obs_code": "500"}])
        cast_schema = pa.schema(
            [
                (
                    field.with_type(pa.string())
                    if field.name == "obs_code"
                    else (
                        field.with_type(pa.int32()) if field.name == "n_obs" else field
                    )
                )
                for field in BIAS_TABLE_SCHEMA
            ]
        )
        validated = validate_bias_table(table.cast(cast_schema))
        assert validated.schema.equals(BIAS_TABLE_SCHEMA)


class TestIdentityModel:
    def test_noop(self) -> None:
        observations = make_observations(["500", "F51"], [0.0, 45.0])
        model = IdentityModel()
        result = model.apply(observations)
        assert result is observations
        assert_positions_unchanged(observations, result)

    def test_optional_table_validated(self) -> None:
        IdentityModel(make_bias_table([{"obs_code": "500"}]))
        with pytest.raises(ValueError, match="missing required columns"):
            IdentityModel(pa.table({"obs_code": pa.array(["500"])}))


class TestEmpiricalCovarianceModel:
    def test_add_mode_at_equator(self) -> None:
        # dec = 0 so cos(dec) = 1: pure arcsec² -> deg² unit conversion.
        sigma_lon, sigma_lat = 1e-4, 2e-4
        base_cross = 1e-9
        observations = make_observations(
            ["500"], [0.0], sigma_lon, sigma_lat, cov_lonlat_deg2=base_cross
        )
        table = make_bias_table(
            [
                {
                    "obs_code": "500",
                    "resid_var_ra": 0.36,
                    "resid_var_dec": 0.25,
                    "resid_cov_ra_dec": 0.09,
                }
            ]
        )
        result = EmpiricalCovarianceModel(table).apply(observations)
        cov = result.coordinates.covariance.to_matrix()[0]
        npt.assert_allclose(
            cov[1, 1], sigma_lon**2 + 0.36 / ARCSEC_PER_DEG**2, rtol=1e-9
        )
        npt.assert_allclose(
            cov[2, 2], sigma_lat**2 + 0.25 / ARCSEC_PER_DEG**2, rtol=1e-9
        )
        npt.assert_allclose(cov[1, 2], base_cross + 0.09 / ARCSEC_PER_DEG**2, rtol=1e-9)
        npt.assert_allclose(cov[2, 1], cov[1, 2], rtol=1e-15)
        assert_positions_unchanged(observations, result)

    def test_add_mode_high_declination(self) -> None:
        # dec = 60 so cos(dec) = 0.5: RA variance scales by 1/cos², the
        # cross-term by 1/cos (once), Dec by units only. Getting this wrong
        # by a cos(dec) factor is the failure mode this test pins down.
        sigma_lon, sigma_lat = 1e-4, 2e-4
        observations = make_observations(
            ["500"], [60.0], sigma_lon, sigma_lat, cov_lonlat_deg2=0.0
        )
        table = make_bias_table(
            [
                {
                    "obs_code": "500",
                    "resid_var_ra": 0.36,
                    "resid_var_dec": 0.25,
                    "resid_cov_ra_dec": 0.09,
                }
            ]
        )
        result = EmpiricalCovarianceModel(table).apply(observations)
        cov = result.coordinates.covariance.to_matrix()[0]
        cos_dec = np.cos(np.deg2rad(60.0))
        npt.assert_allclose(
            cov[1, 1],
            sigma_lon**2 + 0.36 / (ARCSEC_PER_DEG**2 * cos_dec**2),
            rtol=1e-9,
        )
        npt.assert_allclose(
            cov[2, 2], sigma_lat**2 + 0.25 / ARCSEC_PER_DEG**2, rtol=1e-9
        )
        npt.assert_allclose(cov[1, 2], 0.09 / (ARCSEC_PER_DEG**2 * cos_dec), rtol=1e-9)
        assert_positions_unchanged(observations, result)

    def test_replace_mode(self) -> None:
        observations = make_observations(
            ["500"], [0.0], 1e-4, 2e-4, cov_lonlat_deg2=1e-9
        )
        table = make_bias_table(
            [
                {
                    "obs_code": "500",
                    "resid_var_ra": 0.36,
                    "resid_var_dec": 0.25,
                    "resid_cov_ra_dec": 0.09,
                }
            ]
        )
        result = EmpiricalCovarianceModel(table, mode="replace").apply(observations)
        cov = result.coordinates.covariance.to_matrix()[0]
        npt.assert_allclose(cov[1, 1], 0.36 / ARCSEC_PER_DEG**2, rtol=1e-9)
        npt.assert_allclose(cov[2, 2], 0.25 / ARCSEC_PER_DEG**2, rtol=1e-9)
        npt.assert_allclose(cov[1, 2], 0.09 / ARCSEC_PER_DEG**2, rtol=1e-9)

    def test_invalid_mode_raises(self) -> None:
        table = make_bias_table([{"obs_code": "500"}])
        with pytest.raises(ValueError, match="mode"):
            EmpiricalCovarianceModel(table, mode="subtract")  # type: ignore[arg-type]

    def test_nan_base_cross_term_treated_as_zero(self) -> None:
        observations = make_observations(["500"], [0.0], 1e-4, 2e-4)
        assert np.isnan(observations.coordinates.covariance.to_matrix()[0, 1, 2])
        table = make_bias_table(
            [{"obs_code": "500", "resid_var_ra": 0.36, "resid_cov_ra_dec": 0.09}]
        )
        result = EmpiricalCovarianceModel(table).apply(observations)
        cov = result.coordinates.covariance.to_matrix()[0]
        npt.assert_allclose(cov[1, 2], 0.09 / ARCSEC_PER_DEG**2, rtol=1e-9)

    def test_absent_station_passes_through(self) -> None:
        observations = make_observations(["500", "F51"], [0.0, 30.0])
        table = make_bias_table(
            [{"obs_code": "X05", "resid_var_ra": 1.0, "resid_var_dec": 1.0}]
        )
        result = EmpiricalCovarianceModel(table).apply(observations)
        assert_observations_identical(observations, result)

    def test_low_resid_cov_n_passes_through(self) -> None:
        observations = make_observations(["500"], [0.0])
        table = make_bias_table(
            [{"obs_code": "500", "resid_var_ra": 1.0, "resid_cov_n": 5}]
        )
        result = EmpiricalCovarianceModel(table).apply(observations)
        assert_observations_identical(observations, result)

        # A lower threshold admits the same station.
        result = EmpiricalCovarianceModel(table, min_resid_cov_n=5).apply(observations)
        cov = result.coordinates.covariance.to_matrix()[0]
        npt.assert_allclose(cov[1, 1], 1e-8 + 1.0 / ARCSEC_PER_DEG**2, rtol=1e-9)

    def test_null_resid_cov_n_passes_through(self) -> None:
        observations = make_observations(["500"], [0.0])
        table = make_bias_table(
            [{"obs_code": "500", "resid_var_ra": 1.0, "resid_cov_n": None}]
        )
        result = EmpiricalCovarianceModel(table).apply(observations)
        assert_observations_identical(observations, result)

    def test_band_specific_row_preferred(self) -> None:
        observations = make_observations(
            ["F51", "F51", "F51"],
            [0.0, 0.0, 0.0],
            bands=["g", "r", None],
        )
        table = make_bias_table(
            [
                {"obs_code": "F51", "band": None, "resid_var_ra": 0.25},
                {"obs_code": "F51", "band": "g", "resid_var_ra": 1.0},
            ]
        )
        result = EmpiricalCovarianceModel(table).apply(observations)
        cov = result.coordinates.covariance.to_matrix()
        # band g -> band-specific row; band r and band null -> station rollup.
        npt.assert_allclose(cov[0, 1, 1], 1e-8 + 1.0 / ARCSEC_PER_DEG**2, rtol=1e-9)
        npt.assert_allclose(cov[1, 1, 1], 1e-8 + 0.25 / ARCSEC_PER_DEG**2, rtol=1e-9)
        npt.assert_allclose(cov[2, 1, 1], 1e-8 + 0.25 / ARCSEC_PER_DEG**2, rtol=1e-9)

    def test_duplicate_station_rows_raise(self) -> None:
        table = make_bias_table([{"obs_code": "500"}, {"obs_code": "500"}])
        with pytest.raises(ValueError, match="duplicate"):
            EmpiricalCovarianceModel(table)

    def test_empty_observations(self) -> None:
        observations = make_observations(["500"], [0.0])[:0]
        table = make_bias_table([{"obs_code": "500", "resid_var_ra": 1.0}])
        result = EmpiricalCovarianceModel(table).apply(observations)
        assert len(result) == 0


class TestPerformanceWeightedModel:
    def test_scales_sigmas_by_sqrt_chi2(self) -> None:
        sigma_lon, sigma_lat = 1e-4, 2e-4
        base_cross = 1e-9
        observations = make_observations(
            ["500"], [45.0], sigma_lon, sigma_lat, cov_lonlat_deg2=base_cross
        )
        table = make_bias_table([{"obs_code": "500", "chi2_per_obs": 4.0}])
        result = PerformanceWeightedModel(table).apply(observations)
        cov = result.coordinates.covariance.to_matrix()[0]
        npt.assert_allclose(cov[1, 1], 4.0 * sigma_lon**2, rtol=1e-9)
        npt.assert_allclose(cov[2, 2], 4.0 * sigma_lat**2, rtol=1e-9)
        npt.assert_allclose(cov[1, 2], 4.0 * base_cross, rtol=1e-9)
        sigmas = result.coordinates.covariance.sigmas[0]
        npt.assert_allclose(sigmas[1], 2.0 * sigma_lon, rtol=1e-9)
        assert_positions_unchanged(observations, result)

    def test_chi2_below_one_clamps_to_unchanged(self) -> None:
        observations = make_observations(["500"], [45.0])
        table = make_bias_table([{"obs_code": "500", "chi2_per_obs": 0.25}])
        result = PerformanceWeightedModel(table).apply(observations)
        assert_observations_identical(observations, result)


class TestSigmaFloorModel:
    def test_floors_each_axis_independently(self) -> None:
        # Baseline sigmas: lon 1e-4 deg = 0.36", lat 2e-4 deg = 0.72".
        # RA bias 2" floors lon; Dec bias 0.1" is below baseline -> lat kept.
        base_cross = 1e-9
        observations = make_observations(
            ["500"], [0.0], 1e-4, 2e-4, cov_lonlat_deg2=base_cross
        )
        table = make_bias_table(
            [{"obs_code": "500", "bias_ra_arcsec": -2.0, "bias_dec_arcsec": 0.1}]
        )
        result = SigmaFloorModel(table).apply(observations)
        cov = result.coordinates.covariance.to_matrix()[0]
        npt.assert_allclose(cov[1, 1], (2.0 / ARCSEC_PER_DEG) ** 2, rtol=1e-9)
        npt.assert_allclose(cov[2, 2], (2e-4) ** 2, rtol=1e-9)
        npt.assert_allclose(cov[1, 2], base_cross, rtol=1e-9)
        assert_positions_unchanged(observations, result)

    def test_high_declination_ra_floor(self) -> None:
        # At dec = 60, an RA bias of 2" (cos(dec)-corrected) corresponds to
        # a lon floor of 2"/cos(dec) = 4".
        observations = make_observations(["500"], [60.0], 1e-4, 2e-4)
        table = make_bias_table([{"obs_code": "500", "bias_ra_arcsec": 2.0}])
        result = SigmaFloorModel(table).apply(observations)
        cov = result.coordinates.covariance.to_matrix()[0]
        cos_dec = np.cos(np.deg2rad(60.0))
        npt.assert_allclose(
            cov[1, 1], (2.0 / (ARCSEC_PER_DEG * cos_dec)) ** 2, rtol=1e-9
        )

    def test_bias_below_baseline_is_noop(self) -> None:
        observations = make_observations(["500"], [0.0], 1e-4, 2e-4)
        table = make_bias_table(
            [{"obs_code": "500", "bias_ra_arcsec": 0.01, "bias_dec_arcsec": 0.01}]
        )
        result = SigmaFloorModel(table).apply(observations)
        assert_observations_identical(observations, result)


class TestPositionPreservation:
    @pytest.mark.parametrize(
        "model_factory",
        [
            lambda table: IdentityModel(table),
            lambda table: EmpiricalCovarianceModel(table),
            lambda table: EmpiricalCovarianceModel(table, mode="replace"),
            lambda table: PerformanceWeightedModel(table),
            lambda table: SigmaFloorModel(table),
        ],
        ids=[
            "identity",
            "empirical_add",
            "empirical_replace",
            "performance_weighted",
            "sigma_floor",
        ],
    )
    def test_all_models_preserve_positions(
        self,
        model_factory: Callable[[pa.Table], ObservationUncertaintyModel],
    ) -> None:
        observations = make_observations(
            ["500", "F51", "W84"], [0.0, 60.0, -45.0], cov_lonlat_deg2=1e-9
        )
        table = make_bias_table(
            [
                {
                    "obs_code": "500",
                    "resid_var_ra": 0.36,
                    "resid_var_dec": 0.25,
                    "resid_cov_ra_dec": 0.09,
                    "bias_ra_arcsec": 2.0,
                    "bias_dec_arcsec": 1.5,
                    "chi2_per_obs": 4.0,
                },
                {
                    "obs_code": "F51",
                    "resid_var_ra": 0.04,
                    "resid_var_dec": 0.04,
                    "resid_cov_ra_dec": -0.01,
                    "bias_ra_arcsec": -1.0,
                    "bias_dec_arcsec": 0.5,
                    "chi2_per_obs": 2.5,
                },
            ]
        )
        model = model_factory(table)
        result = model.apply(observations)
        assert_positions_unchanged(observations, result)

    def test_assert_positions_unchanged_detects_shift(self) -> None:
        observations = make_observations(["500", "F51"], [0.0, 45.0])
        lon = observations.coordinates.lon.to_numpy(zero_copy_only=False).copy()
        lon[0] += 1e-6
        shifted = observations.set_column("coordinates.lon", pa.array(lon))
        with pytest.raises(AssertionError, match="lon"):
            assert_positions_unchanged(observations, shifted)

    def test_assert_positions_unchanged_detects_length_change(self) -> None:
        observations = make_observations(["500", "F51"], [0.0, 45.0])
        with pytest.raises(AssertionError, match="Number of observations"):
            assert_positions_unchanged(observations, observations[:1])
