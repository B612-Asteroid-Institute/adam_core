"""
Tests for the VFC2017 station/catalog sigma interpreters (`VeresFloorModel`,
`VeresReplaceModel`) and the bundled sigma table. Synthetic observations only.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest

from ...coordinates.covariances import CoordinateCovariances
from ..evaluate import OrbitDeterminationObservations
from ..observation_uncertainty import (
    ARCSEC_PER_DEG,
    CompositeModel,
    NightBatchDeweightingModel,
    assert_positions_unchanged,
)
from ..veres2017 import (
    VERES2017_FALLBACK_SIGMA_ARCSEC,
    VERES2017_SIGMA_TABLE_SCHEMA,
    VeresFloorModel,
    VeresReplaceModel,
    VeresSigmaLookup,
    validate_veres_sigma_table,
    veres2017_sigma_table,
)
from .test_observation_uncertainty import make_observations


def table(rows: list[tuple[str | None, str, float, float]]) -> pa.Table:
    return pa.table(
        {
            "obs_code": pa.array([r[0] for r in rows], pa.large_string()),
            "astcat": pa.array([r[1] for r in rows], pa.large_string()),
            "sigma_ra_arcsec": pa.array([r[2] for r in rows], pa.float64()),
            "sigma_dec_arcsec": pa.array([r[3] for r in rows], pa.float64()),
        },
        schema=VERES2017_SIGMA_TABLE_SCHEMA,
    )


SMALL_TABLE = table(
    [
        (None, "Gaia2", 0.20, 0.20),
        (None, "UCAC4", 0.30, 0.40),
        ("703", "Gaia2", 0.34, 0.34),
    ]
)


def observations_with(
    codes: list[str], astcats: list[str | None], lats: list[float]
) -> OrbitDeterminationObservations:
    observations = make_observations(
        codes, lats, sigma_lon_deg=1e-5, sigma_lat_deg=2e-5
    )
    return observations.set_column("astcat", pa.array(astcats, pa.large_string()))


class TestBundledTable:
    def test_schema_and_sanity(self) -> None:
        bundled = veres2017_sigma_table()
        assert bundled.schema.equals(VERES2017_SIGMA_TABLE_SCHEMA)
        sigma_ra = np.array(bundled["sigma_ra_arcsec"].to_pylist())
        sigma_dec = np.array(bundled["sigma_dec_arcsec"].to_pylist())
        assert np.all(sigma_ra > 0) and np.all(sigma_dec > 0)
        assert np.all(sigma_ra <= 1.0) and np.all(sigma_dec <= 1.0)
        keys = list(zip(bundled["obs_code"].to_pylist(), bundled["astcat"].to_pylist()))
        assert len(keys) == len(set(keys))
        # Validation of the bundled table is clean
        validate_veres_sigma_table(bundled)

    def test_known_entries(self) -> None:
        lookup = VeresSigmaLookup()
        assert lookup.sigmas("703", "Gaia2") == (0.34, 0.34)  # station override
        assert lookup.sigmas("F51", "UCAC4") == (0.30, 0.30)  # catalog default
        assert lookup.sigmas("704", "USNOA2") == (0.60, 0.75)  # asymmetric override
        assert lookup.sigmas("X99", "NoSuchCatalog") == (
            VERES2017_FALLBACK_SIGMA_ARCSEC,
            VERES2017_FALLBACK_SIGMA_ARCSEC,
        )


class TestLookup:
    def test_lookup_order(self) -> None:
        lookup = VeresSigmaLookup(SMALL_TABLE, fallback_sigma_arcsec=0.9)
        assert lookup.sigmas("703", "Gaia2") == (0.34, 0.34)
        assert lookup.sigmas("F51", "Gaia2") == (0.20, 0.20)
        assert lookup.sigmas(None, "UCAC4") == (0.30, 0.40)
        assert lookup.sigmas("F51", "PPMXL") == (0.9, 0.9)
        assert lookup.sigmas("F51", None) == (0.9, 0.9)

    def test_no_fallback_returns_none(self) -> None:
        lookup = VeresSigmaLookup(SMALL_TABLE, fallback_sigma_arcsec=None)
        assert lookup.sigmas("F51", "PPMXL") is None
        assert lookup.sigmas("F51", None) is None
        assert lookup.sigmas("F51", "Gaia2") == (0.20, 0.20)

    def test_duplicate_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            VeresSigmaLookup(
                table([(None, "Gaia2", 0.2, 0.2), (None, "Gaia2", 0.3, 0.3)])
            )
        with pytest.raises(ValueError, match="Duplicate"):
            VeresSigmaLookup(
                table([("703", "Gaia2", 0.2, 0.2), ("703", "Gaia2", 0.3, 0.3)])
            )


class TestValidateTable:
    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            validate_veres_sigma_table(SMALL_TABLE.drop_columns(["sigma_dec_arcsec"]))

    def test_null_astcat_raises(self) -> None:
        bad = SMALL_TABLE.set_column(
            1, "astcat", pa.array(["Gaia2", None, "Gaia2"], pa.large_string())
        )
        with pytest.raises(ValueError, match="astcat"):
            validate_veres_sigma_table(bad)

    def test_non_positive_sigma_raises(self) -> None:
        bad = SMALL_TABLE.set_column(
            2, "sigma_ra_arcsec", pa.array([0.2, 0.0, 0.34], pa.float64())
        )
        with pytest.raises(ValueError, match="positive finite"):
            validate_veres_sigma_table(bad)


class TestVeresFloorModel:
    def test_floors_small_sigmas_and_keeps_large_ones(self) -> None:
        # Reported sigma_lon 1e-5 deg (0.036") is below every table sigma;
        # sigma_lat 2e-5 deg (0.072") too, so both axes are floored.
        observations = observations_with(
            ["F51", "703"], ["Gaia2", "Gaia2"], [0.0, 60.0]
        )
        result = VeresFloorModel(SMALL_TABLE).apply(observations)
        assert_positions_unchanged(observations, result)
        covariances = result.coordinates.covariance.to_matrix()
        cos60 = np.cos(np.deg2rad(60.0))
        npt.assert_allclose(covariances[0, 1, 1], (0.20 / ARCSEC_PER_DEG) ** 2)
        npt.assert_allclose(covariances[0, 2, 2], (0.20 / ARCSEC_PER_DEG) ** 2)
        npt.assert_allclose(
            covariances[1, 1, 1], (0.34 / (ARCSEC_PER_DEG * cos60)) ** 2
        )
        npt.assert_allclose(covariances[1, 2, 2], (0.34 / ARCSEC_PER_DEG) ** 2)
        # A large reported sigma is not lowered
        loose = make_observations(
            ["F51"], [0.0], sigma_lon_deg=1e-3, sigma_lat_deg=1e-3
        )
        loose = loose.set_column("astcat", pa.array(["Gaia2"], pa.large_string()))
        assert VeresFloorModel(SMALL_TABLE).apply(loose) is loose

    def test_cross_term_is_preserved(self) -> None:
        observations = make_observations(
            ["F51"],
            [0.0],
            sigma_lon_deg=1e-5,
            sigma_lat_deg=1e-5,
            cov_lonlat_deg2=1e-11,
        ).set_column("astcat", pa.array(["Gaia2"], pa.large_string()))
        result = VeresFloorModel(SMALL_TABLE).apply(observations)
        assert result.coordinates.covariance.to_matrix()[0, 1, 2] == 1e-11

    def test_missing_sigma_left_unless_fill_missing(self) -> None:
        observations = observations_with(["F51"], ["Gaia2"], [0.0])
        nan_cov = np.full((1, 6, 6), np.nan)
        observations = observations.set_column(
            "coordinates.covariance", CoordinateCovariances.from_matrix(nan_cov)
        )
        assert VeresFloorModel(SMALL_TABLE).apply(observations) is observations
        filled = VeresFloorModel(SMALL_TABLE, fill_missing=True).apply(observations)
        npt.assert_allclose(
            filled.coordinates.covariance.to_matrix()[0, 1, 1],
            (0.20 / ARCSEC_PER_DEG) ** 2,
        )

    def test_unknown_catalog_uses_fallback_or_passes_through(self) -> None:
        observations = observations_with(["F51", "F51"], [None, "PPMXL"], [0.0, 0.0])
        result = VeresFloorModel(SMALL_TABLE).apply(observations)
        npt.assert_allclose(
            result.coordinates.covariance.to_matrix()[:, 2, 2],
            (VERES2017_FALLBACK_SIGMA_ARCSEC / ARCSEC_PER_DEG) ** 2,
        )
        assert (
            VeresFloorModel(SMALL_TABLE, fallback_sigma_arcsec=None).apply(observations)
            is observations
        )


class TestVeresReplaceModel:
    def test_replaces_both_axes_and_zeroes_cross_term(self) -> None:
        observations = make_observations(
            ["703", "F51"],
            [30.0, 0.0],
            sigma_lon_deg=1e-3,
            sigma_lat_deg=1e-3,
            cov_lonlat_deg2=1e-7,
        ).set_column("astcat", pa.array(["Gaia2", "UCAC4"], pa.large_string()))
        result = VeresReplaceModel(SMALL_TABLE).apply(observations)
        assert_positions_unchanged(observations, result)
        covariances = result.coordinates.covariance.to_matrix()
        cos30 = np.cos(np.deg2rad(30.0))
        npt.assert_allclose(
            covariances[0, 1, 1], (0.34 / (ARCSEC_PER_DEG * cos30)) ** 2
        )
        npt.assert_allclose(covariances[0, 2, 2], (0.34 / ARCSEC_PER_DEG) ** 2)
        npt.assert_allclose(covariances[1, 1, 1], (0.30 / ARCSEC_PER_DEG) ** 2)
        npt.assert_allclose(covariances[1, 2, 2], (0.40 / ARCSEC_PER_DEG) ** 2)
        npt.assert_array_equal(covariances[:, 1, 2], 0.0)
        npt.assert_array_equal(covariances[:, 2, 1], 0.0)

    def test_composes_with_other_models(self) -> None:
        observations = observations_with(["F51"], ["Gaia2"], [0.0])
        composite = CompositeModel(
            VeresReplaceModel(SMALL_TABLE), NightBatchDeweightingModel(cap=4)
        )
        result = composite.apply(observations)
        npt.assert_allclose(
            result.coordinates.covariance.to_matrix()[0, 2, 2],
            (0.20 / ARCSEC_PER_DEG) ** 2,
        )
