import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import quivr as qv

from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.origin import Origin
from ...coordinates.spherical import SphericalCoordinates
from ...observations.ades import ADESObservations
from ...observers.observers import Observers
from ...time import Timestamp
from ..evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry
from ..observation_uncertainty import IdentityModel, SigmaFloorModel
from .test_observation_uncertainty import make_bias_table


@pytest.fixture
def ades_observations() -> ADESObservations:
    return ADESObservations.from_kwargs(
        permID=["3000", "3000", "3001", "3001"],
        trkSub=["a1234b", "a1234b", "a2345b", "a2345b"],
        obsSubID=["obs01", "obs02", "obs03", "obs04"],
        obsTime=Timestamp.from_mjd([60434.0, 60434.1, 60435.0, 60435.2], scale="utc"),
        ra=[240.00, 240.05, 15.00, 15.05],
        dec=[-15.00, -15.05, 10.00, 60.00],
        rmsRACosDec=[0.9659, 0.9657, None, 0.5],
        rmsDec=[1.0, 1.0, None, 0.25],
        rmsCorr=[0.2, None, None, -0.5],
        mag=[20.0, 20.3, None, 21.4],
        rmsMag=[0.1, 0.2, None, 0.3],
        band=["r", "g", None, "r"],
        stn=["W84", "W84", "V00", "695"],
        mode=["CCD", "CCD", "CCD", "CCD"],
        astCat=["Gaia2", "Gaia2", "UCAC4", "PPMXL"],
    )


def _od_observations_without_astcat(n: int = 3) -> OrbitDeterminationObservations:
    # Mimics a pre-astcat construction site: no astcat kwarg at all.
    times = Timestamp.from_mjd(60000.0 + np.arange(n) * 0.1, scale="utc")
    codes = ["W84"] * n
    covariance = np.full((n, 6, 6), np.nan)
    covariance[:, 1, 1] = (0.3 / 3600.0) ** 2
    covariance[:, 2, 2] = (0.3 / 3600.0) ** 2
    return OrbitDeterminationObservations.from_kwargs(
        id=[f"legacy{i}" for i in range(n)],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=np.linspace(10.0, 11.0, n),
            lat=np.linspace(-5.0, -4.0, n),
            time=times,
            covariance=CoordinateCovariances.from_matrix(covariance),
            origin=Origin.from_kwargs(code=codes),
            frame="equatorial",
        ),
        observers=Observers.from_codes(codes, times),
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=[20.0] * n, rmsmag=[0.1] * n, band=["r"] * n
        ),
    )


def test_from_ades_retains_astcat(ades_observations: ADESObservations) -> None:
    # ADES with astCat -> OD observations carries the catalog code verbatim,
    # in input order, alongside a faithful mapping of the astrometry.
    od_obs = OrbitDeterminationObservations.from_ades(ades_observations)

    assert len(od_obs) == len(ades_observations)
    assert od_obs.astcat.to_pylist() == ["Gaia2", "Gaia2", "UCAC4", "PPMXL"]
    assert od_obs.astcat.null_count == 0

    # IDs come from obsSubID when fully populated
    assert od_obs.id.to_pylist() == ["obs01", "obs02", "obs03", "obs04"]

    # Astrometry, time and station mapping
    np.testing.assert_array_equal(
        od_obs.coordinates.lon.to_numpy(zero_copy_only=False),
        ades_observations.ra.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_array_equal(
        od_obs.coordinates.lat.to_numpy(zero_copy_only=False),
        ades_observations.dec.to_numpy(zero_copy_only=False),
    )
    assert od_obs.coordinates.time.equals(ades_observations.obsTime)
    assert od_obs.coordinates.time.scale == "utc"
    assert od_obs.coordinates.frame == "equatorial"
    assert od_obs.coordinates.origin.code.to_pylist() == ["W84", "W84", "V00", "695"]
    assert od_obs.observers.code.to_pylist() == ["W84", "W84", "V00", "695"]
    assert od_obs.observers.coordinates.time.equals(ades_observations.obsTime)

    # Photometry
    assert od_obs.photometry.mag.to_pylist() == [20.0, 20.3, None, 21.4]
    assert od_obs.photometry.rmsmag.to_pylist() == [0.1, 0.2, None, 0.3]
    assert od_obs.photometry.band.to_pylist() == ["r", "g", None, "r"]


def test_from_ades_uncertainty_conversion(
    ades_observations: ADESObservations,
) -> None:
    # rmsRACosDec / rmsDec (arcsec, RA scaled by cos(dec)) -> lon/lat covariance
    # in deg^2 on the raw angles; rmsCorr -> correlation (null -> 0); null rms -> NaN.
    od_obs = OrbitDeterminationObservations.from_ades(ades_observations)
    cov = od_obs.coordinates.covariance.to_matrix()

    dec = ades_observations.dec.to_numpy(zero_copy_only=False)
    rms_ra_cosdec = np.array([0.9659, 0.9657, np.nan, 0.5])
    rms_dec = np.array([1.0, 1.0, np.nan, 0.25])
    corr = np.array([0.2, 0.0, 0.0, -0.5])
    sigma_lon = rms_ra_cosdec / 3600.0 / np.cos(np.radians(dec))
    sigma_lat = rms_dec / 3600.0

    np.testing.assert_allclose(cov[:, 1, 1], sigma_lon**2, rtol=1e-12)
    np.testing.assert_allclose(cov[:, 2, 2], sigma_lat**2, rtol=1e-12)
    np.testing.assert_allclose(cov[:, 1, 2], corr * sigma_lon * sigma_lat, rtol=1e-12)
    np.testing.assert_allclose(cov[:, 2, 1], cov[:, 1, 2], rtol=1e-12)

    # The high-dec row must show the cos(dec) de-scaling (sigma_lon > rmsRACosDec/3600)
    assert cov[3, 1, 1] > (0.5 / 3600.0) ** 2
    # Row with null rms carries NaN variances (nothing invented)
    assert np.isnan(cov[2, 1, 1]) and np.isnan(cov[2, 2, 2])
    # Everything outside the lon/lat block is NaN (unset)
    assert np.all(np.isnan(cov[:, 0, :])) and np.all(np.isnan(cov[:, 3:, 3:]))

    # sigmas as exposed on the coordinates
    np.testing.assert_allclose(
        od_obs.coordinates.covariance.sigmas[:, 1], sigma_lon, rtol=1e-12
    )
    np.testing.assert_allclose(
        od_obs.coordinates.covariance.sigmas[:, 2], sigma_lat, rtol=1e-12
    )


def test_from_ades_ids(ades_observations: ADESObservations) -> None:
    # Explicit ids override obsSubID
    od_obs = OrbitDeterminationObservations.from_ades(
        ades_observations, ids=["a", "b", "c", "d"]
    )
    assert od_obs.id.to_pylist() == ["a", "b", "c", "d"]
    assert od_obs.astcat.to_pylist() == ["Gaia2", "Gaia2", "UCAC4", "PPMXL"]

    od_obs = OrbitDeterminationObservations.from_ades(
        ades_observations, ids=pa.array(["a", "b", "c", "d"])
    )
    assert od_obs.id.to_pylist() == ["a", "b", "c", "d"]
    assert od_obs.id.type == pa.large_string()

    # Missing obsSubID on any row -> positional ids
    ades_no_subid = ades_observations.set_column(
        "obsSubID", pa.array(["obs01", None, "obs03", "obs04"], type=pa.large_string())
    )
    od_obs = OrbitDeterminationObservations.from_ades(ades_no_subid)
    assert od_obs.id.to_pylist() == ["0", "1", "2", "3"]

    # Duplicate / wrong-length / null ids are rejected
    with pytest.raises(ValueError, match="unique"):
        OrbitDeterminationObservations.from_ades(
            ades_observations, ids=["a", "a", "c", "d"]
        )
    with pytest.raises(ValueError, match="Expected 4"):
        OrbitDeterminationObservations.from_ades(ades_observations, ids=["a", "b"])
    with pytest.raises(ValueError, match="null"):
        OrbitDeterminationObservations.from_ades(
            ades_observations, ids=pa.array(["a", None, "c", "d"])
        )
    ades_dup_subid = ades_observations.set_column(
        "obsSubID", pa.array(["x", "x", "y", "z"], type=pa.large_string())
    )
    with pytest.raises(ValueError, match="unique"):
        OrbitDeterminationObservations.from_ades(ades_dup_subid)


def test_from_ades_empty() -> None:
    od_obs = OrbitDeterminationObservations.from_ades(ADESObservations.empty())
    assert len(od_obs) == 0
    assert "astcat" in od_obs.table.schema.names


def test_astcat_nullable_and_back_compatible(
    ades_observations: ADESObservations, tmp_path: pathlib.Path
) -> None:
    # Construction sites that predate astcat keep working and yield null astcat
    legacy = _od_observations_without_astcat()
    assert "astcat" in legacy.table.schema.names
    assert legacy.astcat.null_count == len(legacy)
    assert OrbitDeterminationObservations.empty().astcat.null_count == 0

    # Concatenating a null-astcat table with a populated one preserves both
    populated = OrbitDeterminationObservations.from_ades(ades_observations)
    combined = qv.concatenate([legacy, populated])
    assert combined.astcat.to_pylist() == [None] * len(legacy) + [
        "Gaia2",
        "Gaia2",
        "UCAC4",
        "PPMXL",
    ]

    # Masking / sorting keep astcat aligned with the rows
    subset = combined.apply_mask(pc.equal(combined.observers.code, "W84"))
    assert subset.astcat.to_pylist() == [None] * len(legacy) + ["Gaia2", "Gaia2"]
    assert combined.sort_by([("id", "descending")]).select(
        "id", "obs03"
    ).astcat.to_pylist() == ["UCAC4"]

    # Parquet round trip retains astcat (nulls and values)
    path = tmp_path / "od_obs.parquet"
    combined.to_parquet(str(path))
    reloaded = OrbitDeterminationObservations.from_parquet(str(path))
    assert reloaded.astcat.to_pylist() == combined.astcat.to_pylist()
    assert reloaded.id.to_pylist() == combined.id.to_pylist()
    assert reloaded.observers.code.to_pylist() == combined.observers.code.to_pylist()
    assert reloaded.coordinates.time.equals(combined.coordinates.time)
    # Whole-table equality is not used here: the covariance carries NaNs and
    # pa.Table.equals treats NaN != NaN, so compare the values NaN-aware instead.
    np.testing.assert_array_equal(
        reloaded.coordinates.values, combined.coordinates.values
    )
    np.testing.assert_array_equal(
        reloaded.coordinates.covariance.to_matrix(),
        combined.coordinates.covariance.to_matrix(),
    )

    # A parquet file written before astcat existed (column absent) still loads,
    # with astcat filled as null.
    legacy_table = legacy.table.drop_columns(["astcat"])
    assert "astcat" not in legacy_table.schema.names
    legacy_path = tmp_path / "legacy_od_obs.parquet"
    pa.parquet.write_table(legacy_table, str(legacy_path))
    reloaded_legacy = OrbitDeterminationObservations.from_parquet(str(legacy_path))
    assert len(reloaded_legacy) == len(legacy)
    assert reloaded_legacy.astcat.null_count == len(legacy)
    assert reloaded_legacy.id.to_pylist() == legacy.id.to_pylist()


def test_astcat_survives_uncertainty_models(
    ades_observations: ADESObservations,
) -> None:
    # Uncertainty models rebuild the coordinates column only; astcat must pass
    # through untouched (both populated and null), including when the model
    # actually changes the covariance.
    populated = OrbitDeterminationObservations.from_ades(ades_observations)
    legacy = _od_observations_without_astcat()

    # Synthetic bias table: a W84 floor well above the baseline sigmas so the
    # model engages on the W84 rows.
    bias_table = make_bias_table(
        [{"obs_code": "W84", "bias_ra_arcsec": 3.0, "bias_dec_arcsec": 3.0}]
    )
    for model in (IdentityModel(), SigmaFloorModel(bias_table)):
        after = model.apply(populated)
        assert after.astcat.to_pylist() == populated.astcat.to_pylist()
        assert after.id.to_pylist() == populated.id.to_pylist()
        after_legacy = model.apply(legacy)
        assert after_legacy.astcat.null_count == len(legacy)

    # Confirm the floor really changed W84 sigmas (so the pass-through above was
    # exercised under a genuine covariance rewrite)
    floored = SigmaFloorModel(bias_table).apply(populated)
    assert np.all(
        floored.coordinates.covariance.sigmas[:2, 2]
        > populated.coordinates.covariance.sigmas[:2, 2]
    )
