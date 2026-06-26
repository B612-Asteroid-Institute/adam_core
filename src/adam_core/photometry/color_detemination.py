# Obtaining different colors and matching the color mix to asteroid class.

from __future__ import annotations

from typing import Optional

import numpy as np
import pyarrow.compute as pc
import quivr as qv
import scipy.optimize
from mpcq import MPCObservations
from mpcq.orbits import MPCOrbits

from ..dynamics.propagation import propagate_2body
from ..observations.detections import PointSourceDetections
from ..observations.exposures import Exposures
from ..observers.observers import Observers
from .absolute_magnitude import estimate_absolute_magnitude_v_from_detections
from .hg12star import hg12star_correction


class ColorFit(qv.Table):
    object_id = qv.LargeStringColumn()
    abs_mag = qv.Float64Column(nullable=True)
    g_mag = qv.Float64Column(nullable=True)
    i_mag = qv.Float64Column(nullable=True)
    r_mag = qv.Float64Column(nullable=True)
    u_mag = qv.Float64Column(nullable=True)
    g_r = qv.Float64Column(nullable=True)
    g_i = qv.Float64Column(nullable=True)
    r_i = qv.Float64Column(nullable=True)
    num_obs = qv.Int64Column(nullable=True)
    num_outliers = qv.Int64Column(nullable=True)
    color_class = qv.StringColumn(nullable=True)


def _obs_to_detections_and_exposures(
    obs: MPCObservations,
) -> tuple[PointSourceDetections, Exposures]:
    """
    Convert a single-object MPCObservations slice into aligned
    PointSourceDetections + Exposures (one row per observation).
    """
    n = len(obs)
    times = obs.obstime

    obsids = np.asarray(obs.obsid.to_numpy(zero_copy_only=False), dtype=object).astype(
        str
    )
    bands = np.asarray(obs.band.to_numpy(zero_copy_only=False), dtype=object).astype(
        str
    )
    stn = np.asarray(obs.stn.to_numpy(zero_copy_only=False), dtype=object).astype(str)
    mag = obs.mag.to_numpy(zero_copy_only=False).astype(np.float64)
    rmsmag = obs.rmsmag.to_numpy(zero_copy_only=False).astype(np.float64)
    ra = obs.ra.to_numpy(zero_copy_only=False).astype(np.float64)
    dec = obs.dec.to_numpy(zero_copy_only=False).astype(np.float64)

    exposures = Exposures.from_kwargs(
        id=obsids.tolist(),
        start_time=times,
        duration=np.zeros(n, dtype=np.float64),
        filter=bands.tolist(),
        observatory_code=stn.tolist(),
    )

    detections = PointSourceDetections.from_kwargs(
        id=obsids.tolist(),
        exposure_id=obsids.tolist(),
        time=times,
        ra=ra.tolist(),
        dec=dec.tolist(),
        mag=mag.tolist(),
        mag_sigma=rmsmag.tolist(),
    )

    return detections, exposures


def _estimate_abs_mag(
    obs: MPCObservations,
    object_coords,
    G: float,
) -> Optional[float]:
    """Estimate H_V via existing photometry pipeline for a single object."""
    detections, exposures = _obs_to_detections_and_exposures(obs)
    try:
        pp = estimate_absolute_magnitude_v_from_detections(
            detections,
            exposures,
            object_coords,
            composition="NEO",
            G=G,
        )
        return float(pp.H_v[0].as_py())
    except Exception:
        return None


def _compute_geometry(
    object_coords,
    observer_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute heliocentric distance r (AU), topocentric distance delta (AU),
    and phase angle alpha (degrees) from Cartesian positions.

    object_coords: CartesianCoordinates, heliocentric, aligned with observer_pos rows.
    observer_pos: N×3 array of heliocentric observer positions in AU.
    """
    obj_pos = object_coords.r  # N×3
    T = obj_pos - observer_pos  # vector from observer to object
    r = np.linalg.norm(obj_pos, axis=1)
    delta = np.linalg.norm(T, axis=1)
    # phase angle at object between Sun direction and observer direction
    cos_alpha = np.einsum("ij,ij->i", obj_pos, T) / (r * delta)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha_deg = np.rad2deg(np.arccos(cos_alpha))
    return r, delta, alpha_deg


def _fit_per_band_h_g12star(
    m_red: np.ndarray,
    alpha_deg: np.ndarray,
    bands: np.ndarray,
    root_weights: np.ndarray,
) -> dict[str, float]:
    """
    Fit per-band absolute magnitudes (H_g, H_i, H_r, H_u) and G12* simultaneously
    using the Penttilä (2016) HG12* phase function with g(t)=0 (no rotation term).

    Returns dict with keys "G12star", "H_g", "H_i", "H_r", "H_u", "num_obs", "num_outliers".
    """
    n = len(m_red)
    # Band selector matrix: columns for H_g, H_i, H_r, H_u
    A = np.zeros((n, 4))
    A[:, 0] = (bands == "g").astype(float)
    A[:, 1] = (bands == "i").astype(float)
    A[:, 2] = (bands == "r").astype(float)
    A[:, 3] = (bands == "u").astype(float)

    Aw = A * root_weights[:, None]
    Bw = m_red * root_weights
    full_weights = root_weights**2

    # Initial H per band: unweighted mean of reduced mags in that band
    H_init = np.zeros(4)
    for bi, bname in enumerate(["g", "i", "r", "u"]):
        mask = bands == bname
        if np.any(mask):
            H_init[bi] = float(np.mean(m_red[mask]))

    params0 = np.concatenate([[0.15], H_init])
    included = np.ones(n, dtype=bool)

    def func(par):
        corr = hg12star_correction(alpha_deg, g12star=par[0])
        return (
            Aw[included] @ par[1:]
            + (corr * root_weights)[included]
            - Bw[included]
        )

    converged = False
    values = params0.copy()
    num_outliers = 0
    while not converged:
        result = scipy.optimize.least_squares(func, params0, verbose=0)
        values = result.x
        corr = hg12star_correction(alpha_deg, g12star=values[0])
        res = (A @ values[1:] + corr - m_red) ** 2 * full_weights
        n_incl = int(np.sum(included))
        sigma2 = np.dot(res, included) / (n_incl - 5) if n_incl > 5 else np.inf
        outliers = res > 9 * sigma2
        new_outliers = outliers & included
        converged = not np.any(new_outliers)
        included &= ~outliers
        params0 = values
        num_outliers = np.sum(outliers)

    return {
        "G12star": float(values[0]),
        "H_g": float(values[1]),
        "H_i": float(values[2]),
        "H_r": float(values[3]),
        "H_u": float(values[4]),
        "num_obs": n,
        "num_outliers": num_outliers,
    }


def _fit_per_band_h_c1c2(
    m_red: np.ndarray,
    alpha_deg: np.ndarray,
    bands: np.ndarray,
    root_weights: np.ndarray,
) -> dict:
    """
    Fit per-band absolute magnitudes (H_g, H_i, H_r, H_u) with the polynomial
    phase correction phi(alpha) = c1*alpha + c2*alpha^2 (alpha in radians).

    Purely linear; solved with weighted least squares and iterative sigma-clipping.

    Returns dict with keys "c1", "c2", "H_g", "H_i", "H_r", "H_u", "num_obs", "num_outliers".
    """
    n = len(m_red)
    alpha_rad = np.deg2rad(alpha_deg)

    # A matrix: [c1_col, c2_col, H_g_sel, H_i_sel, H_r_sel, H_u_sel]
    A = np.zeros((n, 6))
    A[:, 0] = alpha_rad
    A[:, 1] = alpha_rad**2
    A[:, 2] = (bands == "g").astype(float)
    A[:, 3] = (bands == "i").astype(float)
    A[:, 4] = (bands == "r").astype(float)
    A[:, 5] = (bands == "u").astype(float)

    full_weights = root_weights**2
    included = np.ones(n, dtype=bool)
    values = np.zeros(6)

    converged = False
    num_outliers = 0
    while not converged:
        Aw = A[included] * root_weights[included, None]
        Bw = m_red[included] * root_weights[included]
        values, _, _, _ = np.linalg.lstsq(Aw, Bw, rcond=None)
        res = (A @ values - m_red) ** 2 * full_weights
        n_incl = int(np.sum(included))
        sigma2 = np.dot(res, included) / (n_incl - 6) if n_incl > 6 else np.inf
        outliers = res > 9 * sigma2
        new_outliers = outliers & included
        converged = not np.any(new_outliers)
        included &= ~outliers
        num_outliers = int(np.sum(~included))

    return {
        "c1": float(values[0]),
        "c2": float(values[1]),
        "H_g": float(values[2]),
        "H_i": float(values[3]),
        "H_r": float(values[4]),
        "H_u": float(values[5]),
        "num_obs": n,
        "num_outliers": num_outliers,
    }


def _hg_correction(alpha_deg: np.ndarray, G: float) -> np.ndarray:
    """
    Standard Bowell et al. (1989) H-G phase function correction, using
    tan(alpha/2) basis functions (the model already used elsewhere in
    `photometry.magnitude` for predicting apparent magnitudes).

    Returns -2.5*log10((1-G)*phi1 + G*phi2).
    """
    alpha_rad = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    cos_alpha = np.cos(alpha_rad)
    tan_half = np.sqrt((1.0 - cos_alpha) / (1.0 + cos_alpha))
    phi1 = np.exp(-3.33 * tan_half**0.63)
    phi2 = np.exp(-1.87 * tan_half**1.22)
    phase_function = (1.0 - G) * phi1 + G * phi2
    phase_function = np.maximum(phase_function, 1e-10)
    return -2.5 * np.log10(phase_function)


def _fit_per_band_h_hg(
    m_red: np.ndarray,
    alpha_deg: np.ndarray,
    bands: np.ndarray,
    root_weights: np.ndarray,
) -> dict:
    """
    Fit per-band absolute magnitudes (H_g, H_i, H_r, H_u) and G simultaneously
    using the standard Bowell et al. H-G phase function with g(t)=0 (no rotation term).

    Returns dict with keys "G", "H_g", "H_i", "H_r", "H_u", "num_obs", "num_outliers".
    """
    n = len(m_red)
    A = np.zeros((n, 4))
    A[:, 0] = (bands == "g").astype(float)
    A[:, 1] = (bands == "i").astype(float)
    A[:, 2] = (bands == "r").astype(float)
    A[:, 3] = (bands == "u").astype(float)

    Aw = A * root_weights[:, None]
    Bw = m_red * root_weights
    full_weights = root_weights**2

    H_init = np.zeros(4)
    for bi, bname in enumerate(["g", "i", "r", "u"]):
        mask = bands == bname
        if np.any(mask):
            H_init[bi] = float(np.mean(m_red[mask]))

    params0 = np.concatenate([[0.15], H_init])
    included = np.ones(n, dtype=bool)

    def func(par):
        corr = _hg_correction(alpha_deg, par[0])
        return (
            Aw[included] @ par[1:]
            + (corr * root_weights)[included]
            - Bw[included]
        )

    converged = False
    values = params0.copy()
    num_outliers = 0
    while not converged:
        result = scipy.optimize.least_squares(func, params0, verbose=0)
        values = result.x
        corr = _hg_correction(alpha_deg, values[0])
        res = (A @ values[1:] + corr - m_red) ** 2 * full_weights
        n_incl = int(np.sum(included))
        sigma2 = np.dot(res, included) / (n_incl - 5) if n_incl > 5 else np.inf
        outliers = res > 9 * sigma2
        new_outliers = outliers & included
        converged = not np.any(new_outliers)
        included &= ~outliers
        params0 = values
        num_outliers = int(np.sum(outliers))

    return {
        "G": float(values[0]),
        "H_g": float(values[1]),
        "H_i": float(values[2]),
        "H_r": float(values[3]),
        "H_u": float(values[4]),
        "num_obs": n,
        "num_outliers": num_outliers,
    }


def _prepare_geometry(
    obs: MPCObservations,
    object_coords,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract geometry and photometry arrays needed for per-band H fitting.

    Returns (mag, rmsmag, bands, r, delta, alpha_deg, valid_mask).
    valid_mask selects rows with finite mag, finite positive rmsmag.
    """
    stn = np.asarray(obs.stn.to_numpy(zero_copy_only=False), dtype=object).astype(str)
    observers = Observers.from_codes(stn, obs.obstime)
    observer_pos = observers.coordinates.r

    mag = obs.mag.to_numpy(zero_copy_only=False).astype(np.float64)
    rmsmag = obs.rmsmag.to_numpy(zero_copy_only=False).astype(np.float64)
    bands = np.asarray(obs.band.to_numpy(zero_copy_only=False), dtype=object).astype(str)

    r, delta, alpha_deg = _compute_geometry(object_coords, observer_pos)
    valid = np.isfinite(mag) & np.isfinite(rmsmag) & (rmsmag > 0)
    return mag, rmsmag, bands, r, delta, alpha_deg, valid


def estimate_colors(
    observations: MPCObservations,
    orbits: MPCOrbits,
    phi_type: str, # = "HG12star",
) -> ColorFit:
    """
    Estimate per-band absolute magnitudes and colors for each object.

    Inputs can contain data for multiple objects, multiple observers, and
    multiple color bands.

    Parameters
    ----------
    observations
        MPC astrometric/photometric observations.  Must have valid ``requested_provid``,
        ``obstime``, ``mag``, ``band``, and ``stn`` columns.
    orbits
        MPC fitted orbits for the same objects.  Used to propagate positions
        to each observation epoch.
    phi_type
        Phase function type: "HG12star" (Penttilä 2016), "HG" (standard H-G),
        or "c1c2" (polynomial).

    Returns
    -------
    ColorFit
        One row per unique object found in both ``observations`` and ``orbits``.
    """
    # TODO check phi_type is in the supported set

    len_before = len(observations)
    observations = observations.apply_mask(pc.is_valid(observations.band))
    observations = observations.apply_mask(pc.is_valid(observations.mag))
    if len(observations) != len_before:
        print(f"Removed {len_before-len(observations)} null bands")
    unique_ids = [x for x in pc.unique(observations.requested_provid).to_pylist() if x is not None]

    out_ids: list[str] = []
    # out_abs_mag: list[Optional[float]] = []
    out_g_mag: list[Optional[float]] = []
    out_i_mag: list[Optional[float]] = []
    out_r_mag: list[Optional[float]] = []
    out_u_mag: list[Optional[float]] = []
    out_g_r: list[Optional[float]] = []
    out_g_i: list[Optional[float]] = []
    out_r_i: list[Optional[float]] = []
    out_num_obs: list[Optional[int]] = []
    out_num_outliers: list[Optional[int]] = []

    for obj_id in unique_ids:
        obs_mask = pc.equal(observations.requested_provid, obj_id)
        obs = observations.apply_mask(obs_mask)

        orb_mask = pc.equal(orbits.requested_provid, obj_id)
        orb = orbits.apply_mask(orb_mask)
        if len(orb) == 0:
            continue

        adam_orbits = orb.orbits()
        times = obs.obstime

        propagated = propagate_2body(adam_orbits, times)
        object_coords = propagated.coordinates

        G_raw = orb.g[0].as_py()
        G = float(G_raw) if G_raw is not None else 0.15

        # H_V = _estimate_abs_mag(obs, object_coords, G)

        H_g: Optional[float] = None
        H_i: Optional[float] = None
        H_r: Optional[float] = None
        H_u: Optional[float] = None
        g_r: Optional[float] = None
        g_i: Optional[float] = None
        r_i: Optional[float] = None
        num_obs: Optional[int] = None
        num_outliers: Optional[int] = None

        if phi_type in ("HG12star", "c1c2", "HG"):
            try:
                mag, rmsmag, bands, r, delta, alpha_deg, valid = _prepare_geometry(
                    obs, object_coords
                )
                if np.any(valid):
                    m_red = mag[valid] - 5.0 * np.log10(r[valid] * delta[valid])
                    root_weights = 1.0 / rmsmag[valid]
                    if phi_type == "HG12star":
                        fit = _fit_per_band_h_g12star(
                            m_red, alpha_deg[valid], bands[valid], root_weights
                        )
                    elif phi_type == "HG":
                        fit = _fit_per_band_h_hg(
                            m_red, alpha_deg[valid], bands[valid], root_weights
                        )
                    else:
                        fit = _fit_per_band_h_c1c2(
                            m_red, alpha_deg[valid], bands[valid], root_weights
                        )
                    H_g = fit["H_g"]
                    H_i = fit["H_i"]
                    H_r = fit["H_r"]
                    H_u = fit["H_u"]
                    num_obs = int(fit["num_obs"])
                    num_outliers = int(fit["num_outliers"])
                    g_r = H_g - H_r
                    g_i = H_g - H_i
                    r_i = H_r - H_i
            except Exception as e:
                print(f"Problem when fitting colors {e}")
                pass

        out_ids.append(obj_id)
        # out_abs_mag.append(H_V)
        out_g_mag.append(H_g)
        out_i_mag.append(H_i)
        out_r_mag.append(H_r)
        out_u_mag.append(H_u)
        out_g_r.append(g_r)
        out_g_i.append(g_i)
        out_r_i.append(r_i)
        out_num_obs.append(num_obs)
        out_num_outliers.append(num_outliers)

    n_out = len(out_ids)
    return ColorFit.from_kwargs(
        object_id=out_ids,
        # abs_mag=out_abs_mag,
        g_mag=out_g_mag,
        i_mag=out_i_mag,
        r_mag=out_r_mag,
        u_mag=out_u_mag,
        g_r=out_g_r,
        g_i=out_g_i,
        r_i=out_r_i,
        num_obs=out_num_obs,
        num_outliers=out_num_outliers,
        color_class=[None] * n_out,
    )
