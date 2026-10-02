"""
Local regression pins for the OD covariance fix, checked against agency
references on two real objects.

These tests require local data (MPC observations, ADAM precovery detections,
saved prior fits, and the private v2 observatory bias table) plus the
adam-assist propagator, and are skipped unless the environment variable
``ADAM_CORE_OD_PIN_DATA`` points at a directory containing the files listed
in its README.

Agency references (retrieved 2026-07-27):

- 2026 AC4, encounter-timing sigma at the 2026-06-27 Earth encounter
  (MJD 61148): JPL SBDB 15.3 s, NEOCC 14.5 s. The broken inv(J^T J)
  covariance claimed 0.83 s; the corrected covariance gives ~4 s. The pin
  asserts the corrected order of magnitude, not agency equality — the
  remaining factor ~3 is under study (weight philosophy: reported ADES rms
  vs Veres-2017-style floors).
- 2012 TF79, position sigma at epoch: JPL 227 km, FindOrb 231-261 km. The
  broken covariance claimed 21 km; the corrected covariance gives ~300 km.

The fits here are seeded from our own previously converged solutions (saved
in the data directory), not from any external orbit.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pytest

DATA_DIR = os.environ.get("ADAM_CORE_OD_PIN_DATA")

try:
    import rebound  # noqa: F401
    from adam_assist.propagator import ASSISTPropagator
except ImportError:
    ASSISTPropagator = None

pytestmark = [
    pytest.mark.skipif(
        DATA_DIR is None or not os.path.isdir(DATA_DIR),
        reason="ADAM_CORE_OD_PIN_DATA not set or not a directory",
    ),
    pytest.mark.skipif(ASSISTPropagator is None, reason="adam-assist not available"),
]

KM_P_AU = 149_597_870.7
S_P_DAY = 86400.0
DEFAULT_SIGMA_ARCSEC = 1.0
AC4_ENCOUNTER_MJD = 61148.0


def _build_observations(ids, mjds, codes, ra, dec, var_lon, var_lat, mag, rmsmag, band):
    import quivr as qv

    from ...coordinates.covariances import CoordinateCovariances
    from ...coordinates.origin import Origin
    from ...coordinates.spherical import SphericalCoordinates
    from ...observers import Observers
    from ...time import Timestamp
    from ..evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry

    n = len(ids)
    times = Timestamp.from_mjd(np.array(mjds), scale="utc")
    cov = np.full((n, 6, 6), np.nan)
    cov[:, 1, 1] = var_lon
    cov[:, 2, 2] = var_lat
    coords = SphericalCoordinates.from_kwargs(
        lon=ra,
        lat=dec,
        time=times,
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=codes),
        frame="equatorial",
    )
    observers_parts, order = [], []
    codes_arr = np.array(codes)
    for code in np.unique(codes_arr):
        idx = np.flatnonzero(codes_arr == code)
        observers_parts.append(Observers.from_code(str(code), times.take(idx)))
        order.extend(idx.tolist())
    observers = qv.concatenate(observers_parts)
    inverse = np.argsort(np.array(order), kind="stable")
    observers = observers.take(inverse)
    obs = OrbitDeterminationObservations.from_kwargs(
        id=ids,
        coordinates=coords,
        observers=observers,
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=mag, rmsmag=rmsmag, band=band
        ),
    )
    return obs.sort_by(
        [
            ("coordinates.time.days", "ascending"),
            ("coordinates.time.nanos", "ascending"),
        ]
    )


def observations_from_ades(path, prefix):
    rows = json.load(open(path))[0]["ADES_DF"]
    ids, mjds, codes = [], [], []
    ra, dec, var_lon, var_lat = [], [], [], []
    mag, rmsmag, band = [], [], []
    for i, r in enumerate(rows):
        t = datetime.strptime(r["obstime"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
        mjd = (
            t - datetime(1858, 11, 17, tzinfo=timezone.utc)
        ).total_seconds() / 86400.0
        ra_i, dec_i = float(r["ra"]), float(r["dec"])
        cosd = np.cos(np.deg2rad(dec_i))
        # ADES rmsra is cos(dec)-corrected arcsec; lon is not cos-corrected.
        s_ra = float(r["rmsra"]) if r.get("rmsra") else DEFAULT_SIGMA_ARCSEC
        s_dec = float(r["rmsdec"]) if r.get("rmsdec") else DEFAULT_SIGMA_ARCSEC
        ids.append(f"{prefix}-mpc-{i:03d}")
        mjds.append(mjd)
        codes.append(r["stn"])
        ra.append(ra_i)
        dec.append(dec_i)
        var_lon.append((s_ra / 3600.0) ** 2 / cosd**2)
        var_lat.append((s_dec / 3600.0) ** 2)
        mag.append(float(r["mag"]) if r.get("mag") else None)
        rmsmag.append(float(r["rmsmag"]) if r.get("rmsmag") else None)
        band.append(r.get("band"))
    return _build_observations(
        ids, mjds, codes, ra, dec, var_lon, var_lat, mag, rmsmag, band
    )


def observations_from_precovery(path, prefix, obscode):
    import pyarrow.parquet as pq

    rows = pq.ParquetFile(path).read().to_pylist()
    ids, mjds, codes = [], [], []
    ra, dec, var_lon, var_lat = [], [], [], []
    mag, rmsmag, band = [], [], []
    for r in rows:
        mjd = r["time"]["days"] + r["time"]["nanos"] / 86400e9
        ra_i, dec_i = r["ra"], r["dec"]
        cosd = np.cos(np.deg2rad(dec_i))
        # ra_sigma/dec_sigma are sky-plane degrees (cos-corrected convention).
        ids.append(f"{prefix}-prec-{r['id']}")
        mjds.append(mjd)
        codes.append(obscode)
        ra.append(ra_i)
        dec.append(dec_i)
        var_lon.append(r["ra_sigma"] ** 2 / cosd**2)
        var_lat.append(r["dec_sigma"] ** 2)
        mag.append(r["mag"])
        rmsmag.append(r["mag_sigma"])
        b = None
        if ".wrp." in r["exposure_id"]:
            b = r["exposure_id"].split(".wrp.")[1].split(".")[0]
        band.append(b)
    return _build_observations(
        ids, mjds, codes, ra, dec, var_lon, var_lat, mag, rmsmag, band
    )


@pytest.fixture(scope="module")
def bias_model():
    import pyarrow.parquet as pq

    from ..observation_uncertainty import EmpiricalCovarianceModel

    table = pq.ParquetFile(os.path.join(DATA_DIR, "v2_full.parquet")).read()
    if "program_code" in table.column_names:
        table = table.drop_columns(["program_code"])
    return EmpiricalCovarianceModel(table)


def _fit(prefix, precovery_file, precovery_code, bias_model):
    import quivr as qv

    from ...orbits.orbits import Orbits
    from ..differential_correction import fit_least_squares
    from ..fitted_orbits import FittedOrbits

    mpc_obs = observations_from_ades(
        os.path.join(DATA_DIR, f"{prefix}_obs.json"), prefix
    )
    prec_obs = observations_from_precovery(
        os.path.join(DATA_DIR, precovery_file), prefix, precovery_code
    )
    observations = bias_model.apply(
        qv.concatenate([mpc_obs, prec_obs]).sort_by(
            [
                ("coordinates.time.days", "ascending"),
                ("coordinates.time.nanos", "ascending"),
            ]
        )
    )
    saved = FittedOrbits.from_parquet(
        os.path.join(DATA_DIR, f"{prefix}_after_fit.parquet")
    )
    initial = Orbits.from_kwargs(
        orbit_id=saved.orbit_id,
        object_id=saved.object_id,
        coordinates=saved.coordinates,
    )
    fitted, members = fit_least_squares(initial, observations, ASSISTPropagator())
    return fitted, saved


@pytest.fixture(scope="module")
def ac4_fit(bias_model):
    return _fit("ac4", "probably_2026_AC4.parquet", "I41", bias_model)


@pytest.fixture(scope="module")
def tf79_fit(bias_model):
    return _fit("tf79", "probably_2012_TF79.parquet", "F51", bias_model)


def _encounter_timing_sigma(fitted, target_mjd):
    from ...dynamics.propagation import propagate_2body
    from ...orbits.orbits import Orbits
    from ...time import Timestamp

    orbits = Orbits.from_kwargs(
        orbit_id=fitted.orbit_id,
        object_id=fitted.object_id,
        coordinates=fitted.coordinates,
    )
    propagated = propagate_2body(orbits, Timestamp.from_mjd([target_mjd], scale="tdb"))
    covariance = propagated.coordinates.covariance.to_matrix()[0]
    velocity = propagated.coordinates.v[0]
    vhat = velocity / np.linalg.norm(velocity)
    sigma_along_km = float(np.sqrt(vhat @ covariance[:3, :3] @ vhat)) * KM_P_AU
    speed_km_s = float(np.linalg.norm(velocity)) * KM_P_AU / S_P_DAY
    return sigma_along_km / speed_km_s


class Test2026AC4:
    def test_encounter_timing_sigma_order_of_magnitude(self, ac4_fit):
        fitted, _ = ac4_fit
        timing_sigma = _encounter_timing_sigma(fitted, AC4_ENCOUNTER_MJD)
        # Corrected value ~4 s (agencies 14.5-15.3 s). The broken covariance
        # claimed 0.83 s; anything below 2 s means the fabricated-confidence
        # defect is back.
        assert 2.0 < timing_sigma < 8.0, f"timing sigma {timing_sigma:.2f} s"

    def test_weak_axis_scale(self, ac4_fit):
        fitted, _ = ac4_fit
        covariance = fitted.coordinates.covariance.to_matrix()[0]
        weak_axis_km = float(
            np.sqrt(np.linalg.eigvalsh(covariance[:3, :3])[-1]) * KM_P_AU
        )
        # Corrected ~414 km at the converged solution (broken: 1.6 km).
        assert 250.0 < weak_axis_km < 700.0, f"weak axis {weak_axis_km:.1f} km"

    def test_chi2_not_worse_than_saved_solution(self, ac4_fit):
        # The analytic-Jacobian solver must reach at least as deep a minimum
        # as the saved fit (the legacy solver under-converged along the flat
        # valley: chi2 106.5 vs ~97.7 reachable).
        fitted, saved = ac4_fit
        assert fitted.chi2[0].as_py() <= saved.chi2[0].as_py() + 0.1


class Test2012TF79:
    def test_position_sigma_matches_agencies(self, tf79_fit):
        fitted, _ = tf79_fit
        covariance = fitted.coordinates.covariance.to_matrix()[0]
        pos_sigma_km = float(np.sqrt(np.trace(covariance[:3, :3])) * KM_P_AU)
        # JPL 227 km, FindOrb 231-261 km; corrected native fit ~300-360 km
        # (broken: 21 km).
        assert 200.0 < pos_sigma_km < 450.0, f"pos sigma {pos_sigma_km:.1f} km"

    def test_chi2_not_worse_than_saved_solution(self, tf79_fit):
        fitted, saved = tf79_fit
        assert fitted.chi2[0].as_py() <= saved.chi2[0].as_py() + 0.1
