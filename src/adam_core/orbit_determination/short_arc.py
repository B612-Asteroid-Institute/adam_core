from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

from ..constants import Constants as c
from ..coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    OriginCodes,
    SphericalCoordinates,
    transform_coordinates,
)
from ..observations.ades import ADESObservations
from ..observers import Observers
from ..orbits import Orbits
from ..propagator.propagator import Propagator
from ..time import Timestamp
from .differential_correction import fit_least_squares
from .evaluate import OrbitDeterminationObservations, evaluate_orbits
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, drop_duplicate_orbits

DEFAULT_ASTROMETRIC_RMS_ARCSEC = 1.0
_TRANSFORM_EC2EQ = np.asarray(c.TRANSFORM_EC2EQ, dtype=np.float64)


@dataclass(frozen=True)
class ShortArcRangingConfig:
    rho_min: float = 0.02
    rho_max: float = 8.0
    n_rho: int = 80
    rho_dot_min: float = -0.08
    rho_dot_max: float = 0.08
    n_rho_dot: int = 81
    max_seed_candidates: int = 300
    max_refine_candidates: int = 25


class SkyPropagationPredictions(qv.Table):
    time = Timestamp.as_column()
    ra = qv.Float64Column()
    dec = qv.Float64Column()
    sigma_ra = qv.Float64Column(nullable=True)
    sigma_dec = qv.Float64Column(nullable=True)
    frame = qv.StringAttribute(default="equatorial")


@dataclass(frozen=True)
class SkyPropagationResult:
    predictions: SkyPropagationPredictions
    reference_time: Timestamp
    order: int


@dataclass(frozen=True)
class _Attributable:
    reference_index: int
    reference_mjd: float
    ra_deg: float
    dec_deg: float
    ra_rate_deg_per_day: float
    dec_rate_deg_per_day: float
    ra_acc_deg_per_day2: float
    dec_acc_deg_per_day2: float


def _coerce_propagator(
    propagator: Union[Type[Propagator], Propagator],
    propagator_kwargs: dict,
) -> Propagator:
    if isinstance(propagator, Propagator):
        if propagator_kwargs:
            raise ValueError(
                "propagator_kwargs must be empty when passing an instantiated propagator."
            )
        return propagator

    if isinstance(propagator, type) and issubclass(propagator, Propagator):
        return propagator(**propagator_kwargs)

    raise TypeError("propagator must be a Propagator instance or Propagator subclass.")


def _make_obs_ids(ades: ADESObservations) -> npt.NDArray[np.str_]:
    obs_sub_id = ades.obsSubID.to_numpy(zero_copy_only=False)
    days = ades.obsTime.days.to_numpy(zero_copy_only=False)
    nanos = ades.obsTime.nanos.to_numpy(zero_copy_only=False)
    stn = ades.stn.to_numpy(zero_copy_only=False)

    ids = np.empty(len(ades), dtype=object)
    for i in range(len(ades)):
        candidate = obs_sub_id[i]
        if candidate is None or str(candidate).strip() == "":
            candidate = f"{stn[i]}|{days[i]}|{nanos[i]}|{i}"
        ids[i] = str(candidate)

    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    duplicate_mask = counts[inverse] > 1
    if np.any(duplicate_mask):
        occurrence = np.zeros(len(ids), dtype=np.int64)
        for i, dup in enumerate(duplicate_mask):
            if dup:
                occurrence[i] += np.sum(inverse[:i] == inverse[i])
                ids[i] = f"{ids[i]}#{occurrence[i]}"

    return ids.astype(str)


def _extract_astrometric_sigmas_deg(
    observations: OrbitDeterminationObservations,
    default_rms_arcsec: float = DEFAULT_ASTROMETRIC_RMS_ARCSEC,
) -> tuple[np.ndarray, np.ndarray]:
    if default_rms_arcsec <= 0:
        raise ValueError("default_rms_arcsec must be positive.")

    sigma_default = default_rms_arcsec / 3600.0
    sigmas = observations.coordinates.covariance.sigmas[:, 1:3]
    sigma_ra_deg = np.array(sigmas[:, 0], dtype=np.float64)
    sigma_dec_deg = np.array(sigmas[:, 1], dtype=np.float64)

    for sigma in (sigma_ra_deg, sigma_dec_deg):
        invalid = ~np.isfinite(sigma) | (sigma <= 0)
        sigma[invalid] = sigma_default

    if np.any(~np.isfinite(sigma_ra_deg)) or np.any(~np.isfinite(sigma_dec_deg)):
        raise ValueError("Astrometric uncertainties must be finite after fallback.")

    return sigma_ra_deg, sigma_dec_deg


def ades_to_od_observations(
    ades: ADESObservations,
    default_rms_arcsec: float = DEFAULT_ASTROMETRIC_RMS_ARCSEC,
) -> OrbitDeterminationObservations:
    """
    Convert ADES observations to orbit-determination observations.

    Parameters
    ----------
    ades
        ADES observations containing optical RA/Dec astrometry.
    default_rms_arcsec
        Default 1-sigma uncertainty in arcseconds used when RMS columns are missing.

    Returns
    -------
    OrbitDeterminationObservations
    """
    if len(ades) == 0:
        return OrbitDeterminationObservations.empty()

    if default_rms_arcsec <= 0:
        raise ValueError("default_rms_arcsec must be positive.")

    dec_deg = ades.dec.to_numpy(zero_copy_only=False)
    dec_rad = np.radians(dec_deg)
    cos_dec = np.cos(dec_rad)

    rms_ra_cos = np.array(
        ades.rmsRACosDec.to_numpy(zero_copy_only=False), dtype=np.float64
    )
    rms_dec = np.array(ades.rmsDec.to_numpy(zero_copy_only=False), dtype=np.float64)
    rms_corr = np.array(ades.rmsCorr.to_numpy(zero_copy_only=False), dtype=np.float64)

    sigma_ra_arcsec = np.full(len(ades), default_rms_arcsec, dtype=np.float64)
    sigma_dec_arcsec = np.full(len(ades), default_rms_arcsec, dtype=np.float64)

    has_ra = np.isfinite(rms_ra_cos)
    has_dec = np.isfinite(rms_dec)

    tiny_cos = np.abs(cos_dec) < 1e-10
    if np.any(has_ra & tiny_cos):
        raise ValueError(
            "Cannot convert rmsRACosDec to RMS RA near dec=+/-90 deg where cos(dec) is ~0."
        )

    sigma_ra_arcsec[has_ra] = rms_ra_cos[has_ra] / np.abs(cos_dec[has_ra])
    sigma_dec_arcsec[has_dec] = rms_dec[has_dec]

    if np.any(~np.isfinite(sigma_ra_arcsec)) or np.any(~np.isfinite(sigma_dec_arcsec)):
        raise ValueError("Astrometric RMS values must be finite.")
    if np.any(sigma_ra_arcsec <= 0) or np.any(sigma_dec_arcsec <= 0):
        raise ValueError("Astrometric RMS values must be positive.")

    corr = np.where(np.isfinite(rms_corr), rms_corr, 0.0)
    if np.any(np.abs(corr) > 1):
        raise ValueError("rmsCorr must be in [-1, 1].")

    sigma_ra_deg = sigma_ra_arcsec / 3600.0
    sigma_dec_deg = sigma_dec_arcsec / 3600.0

    covariance_matrix = np.full((len(ades), 6, 6), np.nan, dtype=np.float64)
    covariance_matrix[:, 1, 1] = sigma_ra_deg**2
    covariance_matrix[:, 2, 2] = sigma_dec_deg**2
    covariance_matrix[:, 1, 2] = corr * sigma_ra_deg * sigma_dec_deg
    covariance_matrix[:, 2, 1] = covariance_matrix[:, 1, 2]

    coordinates = SphericalCoordinates.from_kwargs(
        rho=None,
        lon=ades.ra,
        lat=ades.dec,
        vrho=None,
        vlon=None,
        vlat=None,
        time=ades.obsTime,
        covariance=CoordinateCovariances.from_matrix(covariance_matrix),
        origin=Origin.from_kwargs(code=ades.stn),
        frame="equatorial",
    )
    observers = Observers.from_codes(codes=ades.stn, times=ades.obsTime)
    return OrbitDeterminationObservations.from_kwargs(
        id=_make_obs_ids(ades),
        coordinates=coordinates,
        observers=observers,
    )


def _coerce_od_observations(
    observations: Union[ADESObservations, OrbitDeterminationObservations],
    default_rms_arcsec: float = DEFAULT_ASTROMETRIC_RMS_ARCSEC,
) -> OrbitDeterminationObservations:
    if isinstance(observations, ADESObservations):
        od_obs = ades_to_od_observations(
            observations,
            default_rms_arcsec=default_rms_arcsec,
        )
    elif isinstance(observations, OrbitDeterminationObservations):
        od_obs = observations
    else:
        raise TypeError(
            "observations must be ADESObservations or OrbitDeterminationObservations."
        )

    return od_obs.sort_by(
        [
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.origin.code",
        ]
    )


def _resolve_reference_index(
    num_obs: int, epoch: Literal["first", "middle", "last"]
) -> int:
    if epoch == "first":
        return 0
    if epoch == "middle":
        return num_obs // 2
    if epoch == "last":
        return num_obs - 1
    raise ValueError("epoch must be one of {'first', 'middle', 'last'}.")


def _weighted_polyfit(
    t_days: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    X = np.vander(t_days, N=order + 1, increasing=True)
    w = 1.0 / np.square(sigma)
    weighted_X = X * np.sqrt(w)[:, None]
    weighted_y = values * np.sqrt(w)

    coeff, *_ = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)

    normal = X.T @ (w[:, None] * X)
    cov = np.linalg.pinv(normal)

    residual = values - X @ coeff
    dof = max(len(values) - len(coeff), 1)
    residual_var = float(np.sum(np.square(residual)) / dof)
    return coeff, cov, residual_var, residual


def _fit_attributable(
    observations: OrbitDeterminationObservations,
    order: int = 2,
    epoch: Literal["first", "middle", "last"] = "middle",
    default_rms_arcsec: float = DEFAULT_ASTROMETRIC_RMS_ARCSEC,
) -> _Attributable:
    if len(observations) < 3:
        raise ValueError(
            "At least 3 observations are required for attributable fitting."
        )

    reference_index = _resolve_reference_index(len(observations), epoch)
    times_mjd = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    reference_mjd = float(times_mjd[reference_index])
    t_days = times_mjd - reference_mjd

    sigma_ra_deg, sigma_dec_deg = _extract_astrometric_sigmas_deg(
        observations,
        default_rms_arcsec=default_rms_arcsec,
    )

    ra_rad = np.unwrap(
        np.radians(observations.coordinates.lon.to_numpy(zero_copy_only=False))
    )
    dec_rad = np.radians(observations.coordinates.lat.to_numpy(zero_copy_only=False))

    effective_order = min(order, len(observations) - 1)
    coeff_ra, _, _, _ = _weighted_polyfit(
        t_days,
        ra_rad,
        np.radians(sigma_ra_deg),
        effective_order,
    )
    coeff_dec, _, _, _ = _weighted_polyfit(
        t_days,
        dec_rad,
        np.radians(sigma_dec_deg),
        effective_order,
    )

    ra0_rad = coeff_ra[0]
    dec0_rad = coeff_dec[0]
    ra_rate = coeff_ra[1] if len(coeff_ra) > 1 else 0.0
    dec_rate = coeff_dec[1] if len(coeff_dec) > 1 else 0.0
    ra_acc = 2 * coeff_ra[2] if len(coeff_ra) > 2 else 0.0
    dec_acc = 2 * coeff_dec[2] if len(coeff_dec) > 2 else 0.0

    return _Attributable(
        reference_index=reference_index,
        reference_mjd=reference_mjd,
        ra_deg=float(np.degrees(ra0_rad) % 360.0),
        dec_deg=float(np.degrees(dec0_rad)),
        ra_rate_deg_per_day=float(np.degrees(ra_rate)),
        dec_rate_deg_per_day=float(np.degrees(dec_rate)),
        ra_acc_deg_per_day2=float(np.degrees(ra_acc)),
        dec_acc_deg_per_day2=float(np.degrees(dec_acc)),
    )


def _line_of_sight_and_rate_ecliptic(
    reference_time: Timestamp,
    ra_deg: float,
    dec_deg: float,
    ra_rate_deg_per_day: float,
    dec_rate_deg_per_day: float,
) -> tuple[np.ndarray, np.ndarray]:
    los_equatorial = SphericalCoordinates.from_kwargs(
        rho=[1.0],
        lon=[ra_deg],
        lat=[dec_deg],
        vrho=[0.0],
        vlon=[ra_rate_deg_per_day],
        vlat=[dec_rate_deg_per_day],
        time=reference_time,
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="equatorial",
    )
    los_ecliptic = transform_coordinates(
        los_equatorial,
        representation_out=CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )
    return los_ecliptic.r[0], los_ecliptic.v[0]


def _construct_heliocentric_state(
    observer_position: np.ndarray,
    observer_velocity: np.ndarray,
    line_of_sight: np.ndarray,
    line_of_sight_rate: np.ndarray,
    rho: float,
    rho_dot: float,
) -> np.ndarray:
    position = observer_position + rho * line_of_sight
    velocity = observer_velocity + rho_dot * line_of_sight + rho * line_of_sight_rate
    return np.concatenate([position, velocity]).astype(np.float64)


def _build_orbits_from_states(states: np.ndarray, reference_time: Timestamp) -> Orbits:
    if len(states) == 0:
        return Orbits.empty()

    ref_day = reference_time.days[0].as_py()
    ref_nano = reference_time.nanos[0].as_py()
    time = Timestamp.from_kwargs(
        days=np.full(len(states), ref_day, dtype=np.int64),
        nanos=np.full(len(states), ref_nano, dtype=np.int64),
        scale=reference_time.scale,
    )

    coordinates = CartesianCoordinates.from_kwargs(
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        vx=states[:, 3],
        vy=states[:, 4],
        vz=states[:, 5],
        time=time,
        origin=Origin.from_kwargs(code=np.full(len(states), "SUN", dtype=object)),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(coordinates=coordinates)


def _approximate_candidate_chi2(
    state: np.ndarray,
    dt_days: np.ndarray,
    observer_positions: np.ndarray,
    observed_ra_deg: np.ndarray,
    observed_dec_deg: np.ndarray,
    sigma_ra_deg: np.ndarray,
    sigma_dec_deg: np.ndarray,
) -> float:
    chi2 = 0.0
    for i, dt in enumerate(dt_days):
        object_position = state[:3] + dt * state[3:]
        topocentric = object_position - observer_positions[i]
        distance = np.linalg.norm(topocentric)
        if not np.isfinite(distance) or distance <= 0:
            return np.inf

        topocentric_equatorial = _TRANSFORM_EC2EQ @ topocentric
        ux, uy, uz = topocentric_equatorial / distance
        ra_pred = np.degrees(np.arctan2(uy, ux)) % 360.0
        dec_pred = np.degrees(np.arcsin(np.clip(uz, -1.0, 1.0)))

        ra_resid = ((observed_ra_deg[i] - ra_pred + 180.0) % 360.0) - 180.0
        ra_resid *= np.cos(np.radians(observed_dec_deg[i]))
        dec_resid = observed_dec_deg[i] - dec_pred

        chi2 += (ra_resid / sigma_ra_deg[i]) ** 2 + (dec_resid / sigma_dec_deg[i]) ** 2

    return float(chi2)


def _set_members_solution_flags(
    orbit_members: FittedOrbitMembers,
) -> FittedOrbitMembers:
    if len(orbit_members) == 0:
        return orbit_members

    outlier = pc.fill_null(orbit_members.outlier, False)
    return orbit_members.set_column("solution", pc.invert(outlier))


def _merge_refined_candidates(
    seed_orbits: FittedOrbits,
    seed_members: FittedOrbitMembers,
    refined_orbits: list[FittedOrbits],
    refined_members: list[FittedOrbitMembers],
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    if len(refined_orbits) == 0:
        return seed_orbits, seed_members

    refined_orbits_table = qv.concatenate(refined_orbits)
    refined_members_table = qv.concatenate(refined_members)
    replace_ids = refined_orbits_table.orbit_id

    keep_seed_orbits = seed_orbits.apply_mask(
        pc.invert(pc.is_in(seed_orbits.orbit_id, replace_ids))
    )
    keep_seed_members = seed_members.apply_mask(
        pc.invert(pc.is_in(seed_members.orbit_id, replace_ids))
    )

    merged_orbits = qv.concatenate([keep_seed_orbits, refined_orbits_table])
    merged_members = qv.concatenate([keep_seed_members, refined_members_table])
    return merged_orbits, merged_members


def systematic_ranging_short_arc(
    observations: Union[ADESObservations, OrbitDeterminationObservations],
    propagator: Union[Type[Propagator], Propagator],
    *,
    config: Optional[ShortArcRangingConfig] = None,
    max_candidates: int = 50,
    refine_with_least_squares: bool = True,
    propagator_kwargs: Optional[dict] = None,
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Generate short-arc state-vector candidates via systematic ranging and score them.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1.")

    config = config or ShortArcRangingConfig()
    propagator_kwargs = propagator_kwargs or {}

    od_obs = _coerce_od_observations(observations)
    if len(od_obs) < 3:
        raise ValueError("At least 3 observations are required for systematic ranging.")

    attributable = _fit_attributable(od_obs, order=2, epoch="middle")
    ref_idx = attributable.reference_index
    reference_time = od_obs.coordinates.time[ref_idx : ref_idx + 1]

    observer_position = od_obs.observers.coordinates.r[ref_idx]
    observer_velocity = od_obs.observers.coordinates.v[ref_idx]
    line_of_sight, line_of_sight_rate = _line_of_sight_and_rate_ecliptic(
        reference_time,
        attributable.ra_deg,
        attributable.dec_deg,
        attributable.ra_rate_deg_per_day,
        attributable.dec_rate_deg_per_day,
    )

    times_mjd = od_obs.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    dt_days = times_mjd - attributable.reference_mjd
    observer_positions = od_obs.observers.coordinates.r

    observed_ra_deg = od_obs.coordinates.lon.to_numpy(zero_copy_only=False)
    observed_dec_deg = od_obs.coordinates.lat.to_numpy(zero_copy_only=False)
    sigma_ra_deg, sigma_dec_deg = _extract_astrometric_sigmas_deg(od_obs)

    rho_values = np.geomspace(config.rho_min, config.rho_max, num=config.n_rho)
    rho_dot_values = np.linspace(
        config.rho_dot_min, config.rho_dot_max, num=config.n_rho_dot
    )

    num_grid = config.n_rho * config.n_rho_dot
    states = np.empty((num_grid, 6), dtype=np.float64)
    approx_chi2 = np.empty(num_grid, dtype=np.float64)

    k = 0
    for rho in rho_values:
        for rho_dot in rho_dot_values:
            state = _construct_heliocentric_state(
                observer_position,
                observer_velocity,
                line_of_sight,
                line_of_sight_rate,
                float(rho),
                float(rho_dot),
            )
            states[k] = state
            approx_chi2[k] = _approximate_candidate_chi2(
                state,
                dt_days,
                observer_positions,
                observed_ra_deg,
                observed_dec_deg,
                sigma_ra_deg,
                sigma_dec_deg,
            )
            k += 1

    candidate_order = np.argsort(approx_chi2)
    candidate_order = candidate_order[np.isfinite(approx_chi2[candidate_order])]
    if len(candidate_order) == 0:
        return FittedOrbits.empty(), FittedOrbitMembers.empty()

    n_seed = min(config.max_seed_candidates, len(candidate_order))
    seed_states = states[candidate_order[:n_seed]]
    seed_orbits = _build_orbits_from_states(seed_states, reference_time)

    prop = _coerce_propagator(propagator, propagator_kwargs)
    fitted_seed_orbits, fitted_seed_members = evaluate_orbits(seed_orbits, od_obs, prop)
    fitted_seed_members = _set_members_solution_flags(fitted_seed_members)

    if len(fitted_seed_orbits) == 0:
        return fitted_seed_orbits, fitted_seed_members

    refined_orbits: list[FittedOrbits] = []
    refined_members: list[FittedOrbitMembers] = []
    if refine_with_least_squares:
        ranked = fitted_seed_orbits.sort_by(
            [("reduced_chi2", "ascending"), ("chi2", "ascending")]
        )
        n_refine = min(config.max_refine_candidates, len(ranked))

        for orbit_id in ranked.orbit_id[:n_refine]:
            orbit_id_str = orbit_id.as_py()
            seed_orbit = ranked.select("orbit_id", orbit_id_str).to_orbits()
            seed_metric = (
                ranked.select("orbit_id", orbit_id_str).reduced_chi2[0].as_py()
            )
            try:
                refined_orbit, refined_member = fit_least_squares(
                    seed_orbit, od_obs, prop
                )
            except Exception:
                continue

            if len(refined_orbit) == 0:
                continue
            if refined_orbit.reduced_chi2[0].as_py() < seed_metric:
                refined_orbits.append(refined_orbit)
                refined_members.append(refined_member)

    merged_orbits, merged_members = _merge_refined_candidates(
        fitted_seed_orbits,
        fitted_seed_members,
        refined_orbits,
        refined_members,
    )

    merged_orbits, merged_members = drop_duplicate_orbits(merged_orbits, merged_members)
    merged_members = _set_members_solution_flags(merged_members)

    ranked_orbits = merged_orbits.sort_by(
        [("reduced_chi2", "ascending"), ("chi2", "ascending")]
    )
    top_orbits = ranked_orbits[:max_candidates]
    top_members = merged_members.apply_mask(
        pc.is_in(merged_members.orbit_id, top_orbits.orbit_id)
    )
    top_members = _set_members_solution_flags(top_members)
    return top_orbits, top_members


def _gnomonic_forward(
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    ra0_rad: float,
    dec0_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta_ra = ra_rad - ra0_rad
    sin_dec = np.sin(dec_rad)
    cos_dec = np.cos(dec_rad)
    sin_dec0 = np.sin(dec0_rad)
    cos_dec0 = np.cos(dec0_rad)

    denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * np.cos(delta_ra)
    if np.any(np.abs(denom) < 1e-14):
        raise ValueError(
            "Projection singularity encountered in tangent-plane projection."
        )

    xi = cos_dec * np.sin(delta_ra) / denom
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * np.cos(delta_ra)) / denom
    return xi, eta


def _gnomonic_inverse(
    xi: np.ndarray,
    eta: np.ndarray,
    ra0_rad: float,
    dec0_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    sin_dec0 = np.sin(dec0_rad)
    cos_dec0 = np.cos(dec0_rad)

    denom = cos_dec0 - eta * sin_dec0
    ra = ra0_rad + np.arctan2(xi, denom)
    dec = np.arctan2(
        sin_dec0 + eta * cos_dec0,
        np.sqrt(np.square(denom) + np.square(xi)),
    )

    ra = np.mod(ra, 2 * np.pi)
    return ra, dec


def _build_design_matrix(t_days: np.ndarray, order: int) -> np.ndarray:
    return np.vander(t_days, N=order + 1, increasing=True)


def propagate_sky_plane(
    observations: Union[ADESObservations, OrbitDeterminationObservations],
    target_times: Timestamp,
    *,
    order: int = 2,
    epoch: Literal["first", "middle", "last"] = "middle",
) -> SkyPropagationResult:
    """
    Fit a weighted polynomial in the tangent plane and propagate to target times.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")

    od_obs = _coerce_od_observations(observations)
    if len(od_obs) < 3:
        raise ValueError(
            "At least 3 observations are required for sky-plane propagation."
        )
    if len(target_times) == 0:
        return SkyPropagationResult(
            predictions=SkyPropagationPredictions.empty(),
            reference_time=od_obs.coordinates.time[:1],
            order=order,
        )

    if target_times.scale != od_obs.coordinates.time.scale:
        target_times = target_times.rescale(od_obs.coordinates.time.scale)

    reference_index = _resolve_reference_index(len(od_obs), epoch)
    reference_time = od_obs.coordinates.time[reference_index : reference_index + 1]

    times_mjd = od_obs.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    t_ref = float(times_mjd[reference_index])
    t_obs = times_mjd - t_ref

    ra_obs_rad = np.radians(od_obs.coordinates.lon.to_numpy(zero_copy_only=False))
    dec_obs_rad = np.radians(od_obs.coordinates.lat.to_numpy(zero_copy_only=False))

    ra0_rad = float(ra_obs_rad[reference_index])
    dec0_rad = float(dec_obs_rad[reference_index])

    xi_obs, eta_obs = _gnomonic_forward(ra_obs_rad, dec_obs_rad, ra0_rad, dec0_rad)

    sigma_ra_deg, sigma_dec_deg = _extract_astrometric_sigmas_deg(od_obs)
    sigma_xi = np.radians(sigma_ra_deg * np.cos(dec0_rad))
    sigma_eta = np.radians(sigma_dec_deg)

    effective_order = min(order, len(od_obs) - 1)
    coeff_xi, cov_xi, residual_var_xi, _ = _weighted_polyfit(
        t_obs,
        xi_obs,
        sigma_xi,
        effective_order,
    )
    coeff_eta, cov_eta, residual_var_eta, _ = _weighted_polyfit(
        t_obs,
        eta_obs,
        sigma_eta,
        effective_order,
    )

    target_mjd = target_times.mjd().to_numpy(zero_copy_only=False)
    t_target = target_mjd - t_ref
    X_target = _build_design_matrix(t_target, effective_order)

    xi_pred = X_target @ coeff_xi
    eta_pred = X_target @ coeff_eta

    var_xi = np.einsum("ij,jk,ik->i", X_target, cov_xi, X_target) + residual_var_xi
    var_eta = np.einsum("ij,jk,ik->i", X_target, cov_eta, X_target) + residual_var_eta
    sigma_xi_pred = np.sqrt(np.clip(var_xi, 0.0, np.inf))
    sigma_eta_pred = np.sqrt(np.clip(var_eta, 0.0, np.inf))

    ra_pred_rad, dec_pred_rad = _gnomonic_inverse(xi_pred, eta_pred, ra0_rad, dec0_rad)
    sigma_dec_rad = sigma_eta_pred
    sigma_ra_rad = sigma_xi_pred / np.maximum(np.cos(dec_pred_rad), 1e-8)

    predictions = SkyPropagationPredictions.from_kwargs(
        time=target_times,
        ra=np.degrees(ra_pred_rad),
        dec=np.degrees(dec_pred_rad),
        sigma_ra=np.degrees(sigma_ra_rad),
        sigma_dec=np.degrees(sigma_dec_rad),
    )

    return SkyPropagationResult(
        predictions=predictions,
        reference_time=reference_time,
        order=effective_order,
    )


__all__ = [
    "ShortArcRangingConfig",
    "SkyPropagationPredictions",
    "SkyPropagationResult",
    "ades_to_od_observations",
    "systematic_ranging_short_arc",
    "propagate_sky_plane",
]
