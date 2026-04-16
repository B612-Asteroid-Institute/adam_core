import numpy as np
import pytest

try:
    from adam_assist import ASSISTPropagator
except Exception:  # pragma: no cover
    ASSISTPropagator = None

from ...coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    Residuals,
    SphericalCoordinates,
)
from ...constants import Constants as c
from ...observations.ades import ADESObservations
from ...observers import Observers
from ...orbits import Orbits
from ...time import Timestamp
from ..short_arc import (
    ShortArcRangingConfig,
    _approximate_candidate_chi2,
    _construct_heliocentric_state,
    _fit_attributable,
    _gnomonic_inverse,
    ades_to_od_observations,
    propagate_sky_plane,
    systematic_ranging_short_arc,
)


def _make_fake_observers(times: Timestamp) -> Observers:
    n = len(times)
    coords = CartesianCoordinates.from_kwargs(
        x=np.linspace(1.0, 1.0 + 1e-3, n),
        y=np.linspace(0.5, 0.5 + 1e-3, n),
        z=np.linspace(0.1, 0.1 + 1e-3, n),
        vx=np.full(n, 1e-4),
        vy=np.full(n, -2e-4),
        vz=np.full(n, 3e-4),
        time=times,
        origin=Origin.from_kwargs(code=np.full(n, "SUN", dtype=object)),
        frame="ecliptic",
    )
    return Observers.from_kwargs(
        code=np.full(n, "500", dtype=object), coordinates=coords
    )


def _make_od_observations(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    times: Timestamp,
    sigma_arcsec: float = 0.3,
):
    n = len(ra_deg)
    cov = np.full((n, 6, 6), np.nan, dtype=np.float64)
    sigma_deg = sigma_arcsec / 3600.0
    cov[:, 1, 1] = sigma_deg**2
    cov[:, 2, 2] = sigma_deg**2

    coordinates = SphericalCoordinates.from_kwargs(
        rho=None,
        lon=ra_deg,
        lat=dec_deg,
        vrho=None,
        vlon=None,
        vlat=None,
        time=times,
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=np.full(n, "500", dtype=object)),
        frame="equatorial",
    )

    from ..evaluate import OrbitDeterminationObservations

    return OrbitDeterminationObservations.from_kwargs(
        id=np.array([f"obs{i}" for i in range(n)], dtype=object),
        coordinates=coordinates,
        observers=_make_fake_observers(times),
    )


def _angular_error_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    dra = ((ra1_deg - ra2_deg + 180.0) % 360.0) - 180.0
    dra *= np.cos(np.radians(dec2_deg))
    ddec = dec1_deg - dec2_deg
    return np.sqrt(dra**2 + ddec**2) * 3600.0


def test_ades_to_od_observations_covariance_conversion() -> None:
    times = Timestamp.from_mjd([60200.0, 60200.01, 60200.02, 60200.03], scale="utc")
    ades = ADESObservations.from_kwargs(
        permID=[None, None, None, None],
        provID=[None, None, None, None],
        trkSub=["trk1", "trk1", "trk1", "trk1"],
        obsSubID=["a", "b", "c", "d"],
        obsTime=times,
        ra=[120.0, 120.01, 120.02, 120.03],
        dec=[10.0, 10.01, 10.02, 10.03],
        rmsRACosDec=[0.6, 0.7, None, 0.8],
        rmsDec=[0.5, None, 0.6, 0.7],
        rmsCorr=[0.1, None, -0.2, 0.0],
        stn=["W84", "W84", "W84", "W84"],
        mode=["CCD", "CCD", "CCD", "CCD"],
        astCat=["Gaia2", "Gaia2", "Gaia2", "Gaia2"],
    )

    od = ades_to_od_observations(ades)
    assert len(od) == 4
    assert np.all(
        od.observers.code.to_numpy(zero_copy_only=False) == np.array(["W84"] * 4)
    )

    sig = od.coordinates.covariance.sigmas
    expected_ra0 = (0.6 / np.cos(np.radians(10.0))) / 3600.0
    expected_dec0 = 0.5 / 3600.0
    assert np.isclose(sig[0, 1], expected_ra0)
    assert np.isclose(sig[0, 2], expected_dec0)

    # Missing RMS should fall back to 1 arcsec.
    assert np.isclose(sig[1, 2], 1.0 / 3600.0)
    assert np.isclose(sig[2, 1], 1.0 / 3600.0)


def test_fit_attributable_recovers_quadratic_rates() -> None:
    t_days = np.array([0.0, 0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    times = Timestamp.from_mjd(60000.0 + t_days, scale="utc")

    ra = 20.0 + 0.30 * t_days + 0.04 * t_days**2
    dec = -5.0 + 0.10 * t_days - 0.03 * t_days**2
    observations = _make_od_observations(ra, dec, times, sigma_arcsec=0.1)

    attributable = _fit_attributable(observations, order=2, epoch="first")

    assert np.isclose(attributable.ra_deg, 20.0, atol=5e-6)
    assert np.isclose(attributable.dec_deg, -5.0, atol=5e-6)
    assert np.isclose(attributable.ra_rate_deg_per_day, 0.30, atol=5e-5)
    assert np.isclose(attributable.dec_rate_deg_per_day, 0.10, atol=5e-5)
    assert np.isclose(attributable.ra_acc_deg_per_day2, 0.08, atol=5e-4)
    assert np.isclose(attributable.dec_acc_deg_per_day2, -0.06, atol=5e-4)


def test_construct_heliocentric_state() -> None:
    observer_position = np.array([1.0, 2.0, 3.0])
    observer_velocity = np.array([0.1, 0.2, 0.3])
    line_of_sight = np.array([1.0, 0.0, 0.0])
    line_of_sight_rate = np.array([0.0, 1.0, 0.0])

    state = _construct_heliocentric_state(
        observer_position,
        observer_velocity,
        line_of_sight,
        line_of_sight_rate,
        rho=2.0,
        rho_dot=0.5,
    )

    expected = np.array([3.0, 2.0, 3.0, 0.6, 2.2, 0.3])
    np.testing.assert_allclose(state, expected)


def test_approximate_candidate_chi2_rotates_ecliptic_to_equatorial() -> None:
    state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dt_days = np.array([0.0], dtype=np.float64)
    observer_positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    topocentric_equatorial = c.TRANSFORM_EC2EQ @ state[:3]
    distance = np.linalg.norm(topocentric_equatorial)
    ux, uy, uz = topocentric_equatorial / distance

    observed_ra_deg = np.array([np.degrees(np.arctan2(uy, ux)) % 360.0], dtype=np.float64)
    observed_dec_deg = np.array(
        [np.degrees(np.arcsin(np.clip(uz, -1.0, 1.0)))], dtype=np.float64
    )

    chi2 = _approximate_candidate_chi2(
        state,
        dt_days,
        observer_positions,
        observed_ra_deg,
        observed_dec_deg,
        sigma_ra_deg=np.array([1e-4], dtype=np.float64),
        sigma_dec_deg=np.array([1e-4], dtype=np.float64),
    )

    assert chi2 < 1e-10


def test_propagate_sky_plane_order2_beats_order1() -> None:
    t_obs = np.array([0.0, 0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    t_target = np.array([0.06, 0.07, 0.08], dtype=np.float64)
    times_obs = Timestamp.from_mjd(60250.0 + t_obs, scale="utc")
    times_target = Timestamp.from_mjd(60250.0 + t_target, scale="utc")

    ra0_rad = np.radians(120.0)
    dec0_rad = np.radians(-5.0)

    xi_obs = 1.5e-3 * t_obs + 2.0e-4 * t_obs**2
    eta_obs = -8.0e-4 * t_obs + 1.0e-4 * t_obs**2
    ra_obs_rad, dec_obs_rad = _gnomonic_inverse(xi_obs, eta_obs, ra0_rad, dec0_rad)

    xi_true = 1.5e-3 * t_target + 2.0e-4 * t_target**2
    eta_true = -8.0e-4 * t_target + 1.0e-4 * t_target**2
    ra_true_rad, dec_true_rad = _gnomonic_inverse(xi_true, eta_true, ra0_rad, dec0_rad)

    observations = _make_od_observations(
        np.degrees(ra_obs_rad),
        np.degrees(dec_obs_rad),
        times_obs,
        sigma_arcsec=0.05,
    )

    linear = propagate_sky_plane(observations, times_target, order=1, epoch="first")
    quadratic = propagate_sky_plane(observations, times_target, order=2, epoch="first")

    err_linear = _angular_error_arcsec(
        linear.predictions.ra.to_numpy(zero_copy_only=False),
        linear.predictions.dec.to_numpy(zero_copy_only=False),
        np.degrees(ra_true_rad),
        np.degrees(dec_true_rad),
    )
    err_quadratic = _angular_error_arcsec(
        quadratic.predictions.ra.to_numpy(zero_copy_only=False),
        quadratic.predictions.dec.to_numpy(zero_copy_only=False),
        np.degrees(ra_true_rad),
        np.degrees(dec_true_rad),
    )

    assert np.mean(err_quadratic) < np.mean(err_linear)
    assert np.max(err_quadratic) < 0.5

    sigma_ra = quadratic.predictions.sigma_ra.to_numpy(zero_copy_only=False)
    sigma_dec = quadratic.predictions.sigma_dec.to_numpy(zero_copy_only=False)
    assert np.all(np.isfinite(sigma_ra))
    assert np.all(np.isfinite(sigma_dec))


@pytest.mark.skipif(ASSISTPropagator is None, reason="ASSISTPropagator not available")
def test_short_arc_assist_integration_accuracy() -> None:
    orbit = Orbits.from_kwargs(
        orbit_id=["orbit0"],
        object_id=["test-object"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[0.85],
            y=[-0.32],
            z=[0.14],
            vx=[0.012],
            vy=[0.008],
            vz=[-0.003],
            time=Timestamp.from_mjd([60300.0], scale="utc"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    dt_minutes = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 30.0], dtype=np.float64)
    times = Timestamp.from_mjd(60300.0 + dt_minutes / 1440.0, scale="utc")
    observers = Observers.from_code("W84", times)

    prop = ASSISTPropagator()
    eph = prop.generate_ephemeris(orbit, observers, max_processes=1, covariance=False)

    ra = eph.coordinates.lon.to_numpy(zero_copy_only=False)
    dec = eph.coordinates.lat.to_numpy(zero_copy_only=False)

    ades_all = ADESObservations.from_kwargs(
        trkSub=["T0001"] * len(times),
        obsSubID=[f"obs{i}" for i in range(len(times))],
        obsTime=times,
        ra=ra,
        dec=dec,
        rmsRACosDec=[0.4] * len(times),
        rmsDec=[0.4] * len(times),
        stn=["W84"] * len(times),
        mode=["CCD"] * len(times),
        astCat=["Gaia2"] * len(times),
    )

    fitted_orbits, _ = systematic_ranging_short_arc(
        ades_all[:4],
        ASSISTPropagator,
        config=ShortArcRangingConfig(
            n_rho=20,
            n_rho_dot=21,
            max_seed_candidates=80,
            max_refine_candidates=10,
        ),
        max_candidates=5,
        refine_with_least_squares=True,
    )

    assert len(fitted_orbits) >= 1
    best_orbit = fitted_orbits.sort_by([("reduced_chi2", "ascending")])[:1].to_orbits()

    od_all = ades_to_od_observations(ades_all)
    holdout = od_all[4:]

    eph_holdout = prop.generate_ephemeris(
        best_orbit, holdout.observers, max_processes=1
    )
    residuals = Residuals.calculate(holdout.coordinates, eph_holdout.coordinates)
    residual_values = np.stack(residuals.values.to_numpy(zero_copy_only=False))
    holdout_error = (
        np.sqrt(residual_values[:, 1] ** 2 + residual_values[:, 2] ** 2) * 3600.0
    )

    assert np.max(holdout_error) < 120.0

    sky = propagate_sky_plane(od_all[:4], holdout.coordinates.time, order=2)
    sky_error = _angular_error_arcsec(
        sky.predictions.ra.to_numpy(zero_copy_only=False),
        sky.predictions.dec.to_numpy(zero_copy_only=False),
        holdout.coordinates.lon.to_numpy(zero_copy_only=False),
        holdout.coordinates.lat.to_numpy(zero_copy_only=False),
    )

    assert np.max(sky_error) < 120.0
    assert np.all(np.isfinite(sky.predictions.sigma_ra.to_numpy(zero_copy_only=False)))
    assert np.all(np.isfinite(sky.predictions.sigma_dec.to_numpy(zero_copy_only=False)))


def test_validation_errors() -> None:
    times_short = Timestamp.from_mjd([60100.0, 60100.01], scale="utc")
    obs_short = _make_od_observations(
        np.array([15.0, 15.01]),
        np.array([-10.0, -10.01]),
        times_short,
    )

    with pytest.raises(ValueError, match="At least 3 observations"):
        systematic_ranging_short_arc(obs_short, object())

    with pytest.raises(ValueError, match="At least 3 observations"):
        propagate_sky_plane(obs_short, times_short, order=2)

    times_ok = Timestamp.from_mjd([60110.0, 60110.01, 60110.02], scale="utc")
    obs_ok = _make_od_observations(
        np.array([30.0, 30.01, 30.02]),
        np.array([5.0, 5.01, 5.02]),
        times_ok,
    )

    with pytest.raises(ValueError, match="order must be 1 or 2"):
        propagate_sky_plane(obs_ok, times_ok, order=3)

    pole_ades = ADESObservations.from_kwargs(
        trkSub=["POLE"],
        obsSubID=["pole-1"],
        obsTime=Timestamp.from_mjd([60120.0], scale="utc"),
        ra=[10.0],
        dec=[90.0],
        rmsRACosDec=[0.2],
        rmsDec=[0.2],
        stn=["W84"],
        mode=["CCD"],
        astCat=["Gaia2"],
    )

    with pytest.raises(ValueError, match=r"cos\(dec\)"):
        ades_to_od_observations(pole_ades)

    with pytest.raises(ValueError, match="default_rms_arcsec must be positive"):
        ades_to_od_observations(pole_ades, default_rms_arcsec=0.0)
