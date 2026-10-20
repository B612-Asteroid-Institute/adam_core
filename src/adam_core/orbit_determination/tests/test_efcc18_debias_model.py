"""
Tests for EFCC18DebiasModel.

This model MODIFIES POSITIONS by design, so unlike the other uncertainty
models it is tested for: covariance unchanged, positions shifted by the
injected correction, and pass-through of uncovered / null catalogs.
Synthetic bias tables only; no published EFCC18 numbers appear here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.testing as npt
import pytest

from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.origin import Origin
from ...coordinates.spherical import SphericalCoordinates
from ...observations.efcc18 import (
    EFCC18_CATALOG_CODES,
    EFCC18_JPL_UNDEBIASED_ASTCATS,
    EFCC18_N_CATALOGS,
    EFCC18_N_TILES,
    ra_dec_to_healpix,
)
from ...observers.observers import Observers
from ...time import Timestamp
from ..evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry
from ..observation_uncertainty import (
    ARCSEC_PER_DEG,
    CompositeModel,
    EFCC18DebiasModel,
    EmpiricalCovarianceModel,
    assert_positions_unchanged,
)
from .test_observation_uncertainty import make_bias_table

JD_J2000 = 2451545.0
TEN_YEARS_JD = JD_J2000 + 10.0 * 365.25

# Injected cell values: (dRA arcsec, dDec arcsec, pmRA mas/yr, pmDec mas/yr)
D_RA, D_DEC, PM_RA, PM_DEC = 0.5, -0.3, 20.0, -10.0
# Expected total correction ten years after J2000, in arcsec (cos(dec) frame)
EXPECTED_RA_ARCSEC = D_RA + 10.0 * PM_RA / 1000.0  # 0.7
EXPECTED_DEC_ARCSEC = D_DEC + 10.0 * PM_DEC / 1000.0  # -0.4


def make_observations(
    lons: list[float],
    lats: list[float],
    astcats: list[Optional[str]],
    jd_tdb: float = TEN_YEARS_JD,
    sigma_lon_deg: float = 1e-4,
    sigma_lat_deg: float = 2e-4,
) -> OrbitDeterminationObservations:
    n = len(lons)
    times = Timestamp.from_jd(np.full(n, jd_tdb) + 1e-3 * np.arange(n), scale="tdb")
    codes = ["500"] * n
    covariances = np.full((n, 6, 6), np.nan)
    covariances[:, 1, 1] = sigma_lon_deg**2
    covariances[:, 2, 2] = sigma_lat_deg**2
    covariances[:, 1, 2] = covariances[:, 2, 1] = 1e-9
    coordinates = SphericalCoordinates.from_kwargs(
        lon=np.asarray(lons, dtype=np.float64),
        lat=np.asarray(lats, dtype=np.float64),
        covariance=CoordinateCovariances.from_matrix(covariances),
        time=times,
        origin=Origin.from_kwargs(code=codes),
        frame="equatorial",
    )
    return OrbitDeterminationObservations.from_kwargs(
        id=[f"obs_{i:02d}" for i in range(n)],
        coordinates=coordinates,
        observers=Observers.from_codes(codes, times),
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=[None] * n, rmsmag=[None] * n, band=[None] * n
        ),
        astcat=astcats,
    )


def injected_table(
    lons: list[float], lats: list[float], astcat: str = "UCAC4"
) -> np.ndarray:
    """Zero table with the injected cell at each (tile of lon/lat, catalog)."""
    table = np.zeros((EFCC18_N_TILES, EFCC18_N_CATALOGS, 4), dtype=np.float32)
    from ...observations.efcc18 import MPC_ASTCAT_TO_EFCC18

    column = EFCC18_CATALOG_CODES.index(MPC_ASTCAT_TO_EFCC18[astcat])
    for tile in ra_dec_to_healpix(lons, lats):
        table[tile, column] = [D_RA, D_DEC, PM_RA, PM_DEC]
    return table


def assert_everything_but_positions_unchanged(
    before: OrbitDeterminationObservations, after: OrbitDeterminationObservations
) -> None:
    assert len(before) == len(after)
    assert before.id.to_pylist() == after.id.to_pylist()
    assert before.astcat.to_pylist() == after.astcat.to_pylist()
    assert before.coordinates.time.equals(after.coordinates.time)
    assert before.observers.code.to_pylist() == after.observers.code.to_pylist()
    assert before.coordinates.frame == after.coordinates.frame
    assert (
        before.coordinates.origin.code.to_pylist()
        == after.coordinates.origin.code.to_pylist()
    )
    npt.assert_array_equal(
        before.coordinates.covariance.to_matrix(),
        after.coordinates.covariance.to_matrix(),
    )


class TestEFCC18DebiasModel:
    def test_injected_correction_round_trip(self) -> None:
        # Equatorial-ish and a high-declination case: cos(dec) de-scaling of the
        # RA correction is the thing most likely to be wrong.
        lons = [100.0, 50.0]
        lats = [30.0, 80.0]
        observations = make_observations(lons, lats, ["UCAC4", "UCAC4"])
        model = EFCC18DebiasModel(bias_table=injected_table(lons, lats))
        result = model.apply(observations)

        cos_dec = np.cos(np.deg2rad(np.asarray(lats)))
        expected_lon = np.asarray(lons) - EXPECTED_RA_ARCSEC / (
            ARCSEC_PER_DEG * cos_dec
        )
        expected_lat = np.asarray(lats) - EXPECTED_DEC_ARCSEC / ARCSEC_PER_DEG
        npt.assert_allclose(
            result.coordinates.lon.to_numpy(zero_copy_only=False),
            expected_lon,
            rtol=0,
            atol=1e-10,
        )
        npt.assert_allclose(
            result.coordinates.lat.to_numpy(zero_copy_only=False),
            expected_lat,
            rtol=0,
            atol=1e-10,
        )
        # The high-dec RA shift is larger than the low-dec one by 1/cos(dec).
        shifts = np.asarray(lons) - result.coordinates.lon.to_numpy(
            zero_copy_only=False
        )
        # (rows are staggered by 1e-3 d, so the pm term differs at the 1e-7 level)
        npt.assert_allclose(shifts[1] / shifts[0], cos_dec[0] / cos_dec[1], rtol=1e-6)
        # Dec shift is the same on both rows and has the right sign (bias negative -> lat up).
        npt.assert_allclose(
            result.coordinates.lat.to_numpy(zero_copy_only=False) - np.asarray(lats),
            0.4 / ARCSEC_PER_DEG,
        )

        assert_everything_but_positions_unchanged(observations, result)
        # And this model is, by design, NOT position-preserving:
        with pytest.raises(AssertionError, match="positions changed"):
            assert_positions_unchanged(observations, result)
        # Input untouched (new table, not in-place edit)
        npt.assert_array_equal(
            observations.coordinates.lon.to_numpy(zero_copy_only=False), lons
        )

    def test_covariance_unchanged_including_nan_pattern(self) -> None:
        lons, lats = [10.0], [-20.0]
        observations = make_observations(lons, lats, ["PPMXL"])
        result = EFCC18DebiasModel(
            bias_table=injected_table(lons, lats, "PPMXL")
        ).apply(observations)
        before = observations.coordinates.covariance.to_matrix()
        after = result.coordinates.covariance.to_matrix()
        npt.assert_array_equal(np.isnan(before), np.isnan(after))
        npt.assert_array_equal(before[~np.isnan(before)], after[~np.isnan(after)])
        npt.assert_allclose(
            result.coordinates.covariance.sigmas[:, 1:3],
            observations.coordinates.covariance.sigmas[:, 1:3],
        )

    def test_uncovered_and_null_catalogs_pass_through(self) -> None:
        lons, lats = [100.0, 100.0, 100.0], [30.0, 30.0, 30.0]
        observations = make_observations(lons, lats, ["Gaia2", None, "ATLAS2"])
        model = EFCC18DebiasModel(bias_table=injected_table(lons, lats))
        result = model.apply(observations)
        assert result is observations

    def test_mixed_rows_only_covered_move(self) -> None:
        lons = [100.0, 100.0, 100.0, 100.0]
        lats = [30.0, 30.0, 30.0, 30.0]
        astcats: list[Optional[str]] = ["UCAC4", "Gaia2", None, "UCAC4"]
        observations = make_observations(lons, lats, astcats)
        result = EFCC18DebiasModel(bias_table=injected_table(lons, lats)).apply(
            observations
        )
        lon = result.coordinates.lon.to_numpy(zero_copy_only=False)
        lat = result.coordinates.lat.to_numpy(zero_copy_only=False)
        moved = np.array([True, False, False, True])
        assert np.all((lon != 100.0) == moved)
        assert np.all((lat != 30.0) == moved)
        assert_everything_but_positions_unchanged(observations, result)

    def test_exclude_astcats(self) -> None:
        lons, lats = [100.0, 100.0], [30.0, 30.0]
        observations = make_observations(lons, lats, ["Tycho", "UCAC4"])
        table = injected_table(lons, lats, "Tycho") + injected_table(
            lons, lats, "UCAC4"
        )
        default = EFCC18DebiasModel(bias_table=table).apply(observations)
        assert np.all(default.coordinates.lon.to_numpy(zero_copy_only=False) != 100.0)
        jpl = EFCC18DebiasModel(
            bias_table=table, exclude_astcats=EFCC18_JPL_UNDEBIASED_ASTCATS
        ).apply(observations)
        lon = jpl.coordinates.lon.to_numpy(zero_copy_only=False)
        assert lon[0] == 100.0 and lon[1] != 100.0
        assert jpl.astcat.to_pylist() == ["Tycho", "UCAC4"]

    def test_zero_correction_returns_input(self) -> None:
        observations = make_observations([100.0], [30.0], ["UCAC4"])
        zeros = np.zeros((EFCC18_N_TILES, EFCC18_N_CATALOGS, 4), dtype=np.float32)
        assert EFCC18DebiasModel(bias_table=zeros).apply(observations) is observations

    def test_ra_wraps_into_range(self) -> None:
        lons, lats = [1e-6, 359.99999], [10.0, 10.0]
        # Positive RA bias moves lon downward across 0; negative moves it up across 360.
        observations = make_observations(lons, lats, ["UCAC4", "UCAC4"])
        table = injected_table(lons, lats)
        result = EFCC18DebiasModel(bias_table=table).apply(observations)
        lon = result.coordinates.lon.to_numpy(zero_copy_only=False)
        assert np.all((lon >= 0.0) & (lon < 360.0))
        assert lon[0] > 359.0  # wrapped below zero -> just under 360

        table_neg = -table
        result_neg = EFCC18DebiasModel(bias_table=table_neg).apply(observations)
        lon_neg = result_neg.coordinates.lon.to_numpy(zero_copy_only=False)
        assert np.all((lon_neg >= 0.0) & (lon_neg < 360.0))
        assert lon_neg[1] < 1.0  # wrapped above 360 -> just over 0

    def test_pole_and_nonfinite_rows_pass_through(self) -> None:
        lons, lats = [100.0, 100.0, 100.0], [90.0, 30.0, np.nan]
        observations = make_observations(lons, lats, ["UCAC4"] * 3)
        table = injected_table([100.0], [30.0])
        table[
            ra_dec_to_healpix([100.0], [90.0])[0], EFCC18_CATALOG_CODES.index("q")
        ] = [D_RA, D_DEC, PM_RA, PM_DEC]
        result = EFCC18DebiasModel(bias_table=table).apply(observations)
        lon = result.coordinates.lon.to_numpy(zero_copy_only=False)
        lat = result.coordinates.lat.to_numpy(zero_copy_only=False)
        assert (
            lon[0] == 100.0 and lat[0] == 90.0
        )  # pole: RA correction undefined -> untouched
        assert lon[1] != 100.0 and lat[1] != 30.0
        assert lon[2] == 100.0 and np.isnan(lat[2])

    def test_empty_observations(self) -> None:
        empty = OrbitDeterminationObservations.empty()
        zeros = np.zeros((EFCC18_N_TILES, EFCC18_N_CATALOGS, 4), dtype=np.float32)
        assert EFCC18DebiasModel(bias_table=zeros).apply(empty) is empty

    def test_epoch_matters(self) -> None:
        # Same table, J2000 epoch -> only the position term applies (no pm term).
        lons, lats = [100.0], [30.0]
        observations = make_observations(lons, lats, ["UCAC4"], jd_tdb=JD_J2000)
        result = EFCC18DebiasModel(bias_table=injected_table(lons, lats)).apply(
            observations
        )
        cos_dec = np.cos(np.deg2rad(30.0))
        npt.assert_allclose(
            result.coordinates.lon.to_numpy(zero_copy_only=False)[0],
            100.0 - D_RA / (ARCSEC_PER_DEG * cos_dec),
            rtol=0,
            atol=1e-10,
        )
        npt.assert_allclose(
            result.coordinates.lat.to_numpy(zero_copy_only=False)[0],
            30.0 - D_DEC / ARCSEC_PER_DEG,
            rtol=0,
            atol=1e-10,
        )

    def test_constructor_validation(self, tmp_path: "pytest.TempPathFactory") -> None:
        zeros = np.zeros((EFCC18_N_TILES, EFCC18_N_CATALOGS, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="not both"):
            EFCC18DebiasModel(bias_table=zeros, bias_dat="/nonexistent/bias.dat")
        with pytest.raises(ValueError, match="bias_table has shape"):
            EFCC18DebiasModel(bias_table=np.zeros((3, 26, 4)))
        with pytest.raises(FileNotFoundError):
            EFCC18DebiasModel(bias_dat="/nonexistent/bias.dat")

    def test_composes_with_covariance_models(self) -> None:
        lons, lats = [100.0], [30.0]
        observations = make_observations(lons, lats, ["UCAC4"])
        bias_table = make_bias_table(
            [
                {
                    "obs_code": "500",
                    "resid_var_ra": 0.36,
                    "resid_var_dec": 0.25,
                    "resid_cov_ra_dec": 0.0,
                }
            ]
        )
        model = CompositeModel(
            EFCC18DebiasModel(bias_table=injected_table(lons, lats)),
            EmpiricalCovarianceModel(bias_table),
        )
        result = model.apply(observations)
        # Positions moved by EFCC18...
        assert result.coordinates.lon.to_numpy(zero_copy_only=False)[0] != 100.0
        # ...and the covariance was inflated by the station model.
        cov_before = observations.coordinates.covariance.to_matrix()[0]
        cov_after = result.coordinates.covariance.to_matrix()[0]
        assert cov_after[1, 1] > cov_before[1, 1] and cov_after[2, 2] > cov_before[2, 2]
        assert result.astcat.to_pylist() == ["UCAC4"]
        assert result.id.to_pylist() == observations.id.to_pylist()
