# Obtaining different colors for asteroids.

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pyarrow.compute as pc
import quivr as qv
import scipy.optimize
from mpcq import MPCObservations
from mpcq.orbits import MPCOrbits

from ..dynamics.propagation import propagate_2body
from ..observers.observers import Observers
from .hg12star import hg12star_correction
from .magnitude import calculate_phase_angle

_BANDS = ("g", "i", "r", "u")
_PHI_TYPES = ("HG12star", "HG", "c1c2")
# If fewer than this fraction of an object's observations survive validity
# filtering, band-recognition filtering, and outlier rejection, the fit is
# not trustworthy enough to report silently.
_MIN_RETAINED_FRACTION = 0.5


class ColorFit(qv.Table):
    object_id = qv.LargeStringColumn()
    g_mag = qv.Float64Column(nullable=True)
    i_mag = qv.Float64Column(nullable=True)
    r_mag = qv.Float64Column(nullable=True)
    u_mag = qv.Float64Column(nullable=True)
    g_r = qv.Float64Column(nullable=True)
    g_i = qv.Float64Column(nullable=True)
    r_i = qv.Float64Column(nullable=True)
    num_obs = qv.Int64Column(nullable=True)
    num_outliers = qv.Int64Column(nullable=True)


def _compute_geometry(
    object_coords,
    observers: Observers,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute heliocentric distance r (AU), topocentric distance delta (AU),
    and phase angle alpha (degrees).

    object_coords: CartesianCoordinates, heliocentric, aligned with observers rows.
    observers: Observers, heliocentric, aligned with object_coords rows.

    Phase angle is computed via `calculate_phase_angle`, which also validates
    that the input geometry is finite and physically sensible (r > 0, delta > 0)
    and raises otherwise.
    """
    obj_pos = object_coords.r  # N×3
    observer_pos = observers.coordinates.r  # N×3
    r = np.linalg.norm(obj_pos, axis=1)
    delta = np.linalg.norm(obj_pos - observer_pos, axis=1)
    alpha_deg = calculate_phase_angle(object_coords, observers)
    return r, delta, alpha_deg


def _prepare_geometry(
    obs: MPCObservations,
    object_coords,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Extract geometry and photometry arrays needed for per-band H fitting.

    Returns (mag, rmsmag, bands, r, delta, alpha_deg, valid_mask).
    valid_mask selects rows with finite mag, finite positive rmsmag.
    """
    stn = np.asarray(obs.stn.to_numpy(zero_copy_only=False), dtype=object).astype(str)
    observers = Observers.from_codes(stn, obs.obstime)

    mag = obs.mag.to_numpy(zero_copy_only=False).astype(np.float64)
    rmsmag = obs.rmsmag.to_numpy(zero_copy_only=False).astype(np.float64)
    bands = np.asarray(obs.band.to_numpy(zero_copy_only=False), dtype=object).astype(
        str
    )

    r, delta, alpha_deg = _compute_geometry(object_coords, observers)
    valid = np.isfinite(mag) & np.isfinite(rmsmag) & (rmsmag > 0)
    return mag, rmsmag, bands, r, delta, alpha_deg, valid


def _band_selector_matrix(bands: np.ndarray) -> np.ndarray:
    """N×4 selector matrix for H_g, H_i, H_r, H_u columns."""
    return np.column_stack([(bands == b).astype(float) for b in _BANDS])


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


def _fit_per_band_h(
    m_red: np.ndarray,
    alpha_deg: np.ndarray,
    bands: np.ndarray,
    root_weights: np.ndarray,
    phi_type: Literal["HG12star", "HG", "c1c2"],
) -> dict[str, float]:
    """
    Fit per-band absolute magnitudes (H_g, H_i, H_r, H_u) with g(t)=0 (no rotation
    term), using one of three phase-function models:

    - "HG12star": Penttilä (2016) HG12* phase function; G12* fit jointly (nonlinear).
    - "HG": standard Bowell et al. H-G phase function; G fit jointly (nonlinear).
    - "c1c2": polynomial phase correction c1*alpha + c2*alpha^2 (alpha in radians);
      purely linear.

    Observations whose band is not one of "g", "i", "r", "u" are excluded up
    front and counted as outliers. A band with zero surviving observations is
    reported as NaN. If, after all exclusions and outlier rejection, fewer than
    `_MIN_RETAINED_FRACTION` of the input rows remain, the fit is considered
    unreliable and raises.

    In all cases the fit is solved with iterative 9-sigma outlier rejection.

    Returns dict with keys "H_g", "H_i", "H_r", "H_u", "num_obs", "num_outliers".
    """
    n = len(m_red)
    H_sel = _band_selector_matrix(bands)
    full_weights = root_weights**2

    known_band_mask = np.isin(bands, _BANDS)
    if not np.all(known_band_mask):
        unknown_bands = sorted(set(bands[~known_band_mask].tolist()))
        print(
            f"Excluding {int(np.sum(~known_band_mask))} observations with "
            f"unrecognized band codes: {unknown_bands}"
        )
    included = known_band_mask.copy()

    if phi_type == "c1c2":
        alpha_rad = np.deg2rad(alpha_deg)
        A = np.column_stack([alpha_rad, alpha_rad**2, H_sel])
        H_idx = 2
    else:
        A = H_sel
        correction_fn = (
            hg12star_correction if phi_type == "HG12star" else _hg_correction
        )
        H_init = np.array(
            [
                float(np.mean(m_red[bands == b])) if np.any(bands == b) else 0.0
                for b in _BANDS
            ]
        )
        params0 = np.concatenate([[0.15], H_init])
        H_idx = 1
        # Loop-invariant: A, root_weights, and m_red never change across
        # outlier-rejection iterations, only the `included` mask does.
        Aw = A * root_weights[:, None]
        Bw = m_red * root_weights

    num_params = A.shape[1] + (0 if phi_type == "c1c2" else 1)
    values = np.zeros(num_params)
    converged = False
    while not converged:
        if phi_type == "c1c2":
            Aw = A[included] * root_weights[included, None]
            Bw = m_red[included] * root_weights[included]
            values, _, _, _ = np.linalg.lstsq(Aw, Bw, rcond=None)
            res = (A @ values - m_red) ** 2 * full_weights
        else:

            def func(par):
                corr = correction_fn(alpha_deg, par[0])
                return (
                    Aw[included] @ par[1:]
                    + (corr * root_weights)[included]
                    - Bw[included]
                )

            result = scipy.optimize.least_squares(func, params0, verbose=0)
            values = result.x
            corr = correction_fn(alpha_deg, values[0])
            res = (A @ values[1:] + corr - m_red) ** 2 * full_weights
            params0 = values

        n_incl = int(np.sum(included))
        sigma2 = (
            np.dot(res, included) / (n_incl - num_params)
            if n_incl > num_params
            else np.inf
        )
        outliers = res > 9 * sigma2
        new_outliers = outliers & included
        converged = not np.any(new_outliers)
        included &= ~outliers

    num_outliers = int(np.sum(~included))
    if n - num_outliers < _MIN_RETAINED_FRACTION * n:
        raise ValueError(
            f"Outlier/band rejection removed {num_outliers}/{n} observations "
            f"(more than {1 - _MIN_RETAINED_FRACTION:.0%} of the data); fit is unreliable."
        )

    H_values = [float(values[H_idx + i]) for i in range(len(_BANDS))]
    for i, b in enumerate(_BANDS):
        if not np.any(bands[known_band_mask] == b):
            H_values[i] = float("nan")

    return {
        "H_g": H_values[0],
        "H_i": H_values[1],
        "H_r": H_values[2],
        "H_u": H_values[3],
        "num_obs": n,
        "num_outliers": num_outliers,
    }


def estimate_colors(
    observations: MPCObservations,
    orbits: MPCOrbits,
    phi_type: Literal["HG12star", "HG", "c1c2"],
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
    if phi_type not in _PHI_TYPES:
        raise ValueError(
            f"Unsupported phi_type {phi_type!r}; expected one of {_PHI_TYPES}"
        )

    len_before = len(observations)
    observations = observations.apply_mask(pc.is_valid(observations.band))
    observations = observations.apply_mask(pc.is_valid(observations.mag))
    if len(observations) != len_before:
        print(f"Removed {len_before - len(observations)} null bands")
    unique_ids = [
        x for x in pc.unique(observations.requested_provid).to_pylist() if x is not None
    ]

    out_ids: list[str] = []
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
        if len(orb) > 1:
            raise ValueError(f"Expected exactly one orbit for {obj_id}, got {len(orb)}")

        adam_orbits = orb.orbits()
        propagated = propagate_2body(adam_orbits, obs.obstime)
        object_coords = propagated.coordinates

        H_g: Optional[float] = None
        H_i: Optional[float] = None
        H_r: Optional[float] = None
        H_u: Optional[float] = None
        g_r: Optional[float] = None
        g_i: Optional[float] = None
        r_i: Optional[float] = None
        num_obs: int = len(obs)
        num_outliers: Optional[int] = None

        try:
            mag, rmsmag, bands, r, delta, alpha_deg, valid = _prepare_geometry(
                obs, object_coords
            )
            n_invalid = len(obs) - int(np.sum(valid))
            if np.any(valid):
                m_red = mag[valid] - 5.0 * np.log10(r[valid] * delta[valid])
                root_weights = 1.0 / rmsmag[valid]
                fit = _fit_per_band_h(
                    m_red, alpha_deg[valid], bands[valid], root_weights, phi_type
                )
                H_g = fit["H_g"]
                H_i = fit["H_i"]
                H_r = fit["H_r"]
                H_u = fit["H_u"]
                num_outliers = n_invalid + int(fit["num_outliers"])
                g_r = H_g - H_r
                g_i = H_g - H_i
                r_i = H_r - H_i
            else:
                num_outliers = n_invalid
        except Exception:
            print(f"Problem when fitting colors for {obj_id}")
            raise

        out_ids.append(obj_id)
        out_g_mag.append(H_g)
        out_i_mag.append(H_i)
        out_r_mag.append(H_r)
        out_u_mag.append(H_u)
        out_g_r.append(g_r)
        out_g_i.append(g_i)
        out_r_i.append(r_i)
        out_num_obs.append(num_obs)
        out_num_outliers.append(num_outliers)

    return ColorFit.from_kwargs(
        object_id=out_ids,
        g_mag=out_g_mag,
        i_mag=out_i_mag,
        r_mag=out_r_mag,
        u_mag=out_u_mag,
        g_r=out_g_r,
        g_i=out_g_i,
        r_i=out_r_i,
        num_obs=out_num_obs,
        num_outliers=out_num_outliers,
    )
