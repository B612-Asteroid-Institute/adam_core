# Obtaining different colors for asteroids.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow.compute as pc
import quivr as qv
import scipy.optimize

from ..dynamics.propagation import propagate_2body
from ..observers.observers import Observers
from .bandpasses.api import bandpass_delta_mag, map_to_canonical_filter_bands
from .hg12star import hg12star_correction
from .lightcurve import reduced_magnitude
from .magnitude import calculate_phase_angle
from .magnitude_common import hg_phase_correction

if TYPE_CHECKING:
    # mpcq is an optional dependency (install via `adam_core[mpc]`); it is only
    # referenced in type annotations here, which `from __future__ import
    # annotations` keeps from being evaluated at runtime. This keeps
    # `import adam_core.photometry` working without mpcq installed.
    from mpcq import MPCObservations
    from mpcq.orbits import MPCOrbits

logger = logging.getLogger(__name__)

# Color channels we fit an absolute magnitude for. These are the base band
# letters of the canonical vendored filter IDs (e.g. SDSS_g, LSST_g, PS1_g all
# reduce to the "g" channel); see `_resolve_channels`.
_BANDS = ("g", "i", "r", "u")
_PHI_TYPES = ("HG12star", "HG", "c1c2")
# If fewer than this fraction of an object's observations survive validity
# filtering, band-recognition filtering, and outlier rejection, the fit is
# not trustworthy enough to report silently.
_MIN_RETAINED_FRACTION = 0.5
# Physically meaningful range of the H-G / HG12* slope parameter.  Both G
# (Bowell et al. 1989) and G12* (Penttilä 2016) are defined on [0, 1].
_G_BOUNDS = (0.0, 1.0)
_G_LABELS = {"HG12star": "G12*", "HG": "G"}


def _validate_g_bounds(
    G: float,
    phi_type: str,
    obj_id: str,
    force_g_bounds: bool,
) -> None:
    """
    Check that a fitted slope parameter lies within its physical range.

    "c1c2" has no slope parameter (``G`` is NaN) and is skipped.  When ``G`` is
    out of range: raise ``ValueError`` if ``force_g_bounds`` is True, otherwise
    log a warning and keep the fit.
    """
    if phi_type == "c1c2" or not np.isfinite(G):
        return
    lo, hi = _G_BOUNDS
    if lo <= G <= hi:
        return
    label = _G_LABELS[phi_type]
    msg = (
        f"Fitted {label} = {G:.4f} for {obj_id} is outside the physical "
        f"[{lo:g}, {hi:g}] range"
    )
    if force_g_bounds:
        raise ValueError(msg)
    logger.warning("%s; keeping it because force_g_bounds=False", msg)


class ColorFit(qv.Table):
    object_id = qv.LargeStringColumn()
    g_mag = qv.Float64Column(nullable=True)
    i_mag = qv.Float64Column(nullable=True)
    r_mag = qv.Float64Column(nullable=True)
    u_mag = qv.Float64Column(nullable=True)
    # 1-sigma formal uncertainties on the per-band absolute magnitudes, rescaled
    # to the observed scatter (see `_fit_per_band_h`). NaN for unobserved bands.
    g_mag_sigma = qv.Float64Column(nullable=True)
    i_mag_sigma = qv.Float64Column(nullable=True)
    r_mag_sigma = qv.Float64Column(nullable=True)
    u_mag_sigma = qv.Float64Column(nullable=True)
    g_r = qv.Float64Column(nullable=True)
    g_i = qv.Float64Column(nullable=True)
    r_i = qv.Float64Column(nullable=True)
    # Color uncertainties, propagated from the full parameter covariance (so the
    # H_x/H_y correlation through the shared phase parameter is accounted for).
    g_r_sigma = qv.Float64Column(nullable=True)
    g_i_sigma = qv.Float64Column(nullable=True)
    r_i_sigma = qv.Float64Column(nullable=True)
    # Fitted phase slope parameter (G for "HG", G12* for "HG12star"; NaN for
    # "c1c2") and its 1-sigma uncertainty.
    phase_param = qv.Float64Column(nullable=True)
    phase_param_sigma = qv.Float64Column(nullable=True)
    # Fit-quality diagnostics over the finally-included observations.
    chi2 = qv.Float64Column(nullable=True)
    reduced_chi2 = qv.Float64Column(nullable=True)
    dof = qv.Int64Column(nullable=True)
    rank = qv.Int64Column(nullable=True)
    converged = qv.BooleanColumn(nullable=True)
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


def _resolve_channels(
    stn: np.ndarray, bands: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map raw (observatory_code, reported_band) pairs to g/i/r/u color channels.

    Rather than matching MPC band strings literally, this routes each observation
    through the shared `map_to_canonical_filter_bands` utility, which resolves
    `(observatory_code, band)` to a canonical vendored filter ID (handling MPC/ADES
    label quirks, e.g. G96 "G" -> SDSS_g, ATLAS "o" -> ATLAS_o, LSST encodings, etc).
    The canonical filter's base band letter is then taken as the color channel, so
    every g-like filter (SDSS_g, LSST_g, PS1_g, DECam_g, so on) contributes to the "g"
    fit, and filters outside the g/i/r/u set (V, PS1_w, SkyMapper_v) or rows with
    no resolvable filter are returned as ``None`` (excluded downstream).

    Returns ``(channels, filter_ids)``, each an object array of length
    ``len(bands)``. ``channels`` entries are one of ``"g"``, ``"i"``, ``"r"``,
    ``"u"`` or ``None``; ``filter_ids`` holds the canonical vendored filter ID (or
    ``None``) and is kept so callers can apply inter-system color-term corrections
    (see `_apply_color_terms`).
    """
    # on_unknown="skip" leaves unresolvable rows as None instead of raising, so
    # unfiltered reports and unmapped bands are simply dropped from the fit.
    filter_ids = map_to_canonical_filter_bands(stn, bands, on_unknown="skip")
    channels = np.empty(len(filter_ids), dtype=object)
    for i, fid in enumerate(filter_ids.tolist()):
        if fid is None:
            channels[i] = None
            continue
        base = str(fid).rsplit("_", 1)[-1].lower()
        channels[i] = base if base in _BANDS else None

    unresolved = channels == None  # noqa: E711
    if np.any(unresolved):
        dropped = sorted(
            {f"{s}|{b}" for s, b in zip(stn[unresolved], bands[unresolved])}
        )
        logger.warning(
            "Excluding %d observation(s) whose (station, band) does not resolve to a "
            "g/i/r/u color channel: %s",
            int(np.sum(unresolved)),
            dropped,
        )
    return channels, filter_ids


def _apply_color_terms(
    m_red: np.ndarray,
    filter_ids: np.ndarray,
    channels: np.ndarray,
    composition: str | tuple[float, float],
) -> np.ndarray:
    """
    Reconcile reduced magnitudes onto a single reference filter per color channel.

    A channel may pool observations taken through different but same-letter filters
    (e.g. SDSS_g and LSST_g both feed the "g" channel). Merging them directly biases
    the per-channel H by the inter-system color term. This converts every row onto
    the channel's reference filter using `bandpass_delta_mag`:

        m_red_ref = m_red + Δm(composition, filter_id -> reference_filter)

    Because the reference is the dominant filter, rows already in it are unchanged,
    and a channel containing a single filter system is a no-op. ``composition`` (a
    template id "C"/"S"/"NEO"/"MBA" or a ``(weight_C, weight_S)`` tuple) therefore
    only influences channels that genuinely mix filter systems.
    """
    out = np.asarray(m_red, dtype=np.float64).copy()
    fid_str = np.array([str(f) for f in filter_ids.tolist()], dtype=object)
    for ch in _BANDS:
        in_ch = channels == ch
        if not np.any(in_ch):
            continue
        uniq, counts = np.unique(fid_str[in_ch], return_counts=True)
        if len(uniq) < 2:
            continue  # single filter system in this channel: nothing to reconcile
        ref = str(uniq[int(np.argmax(counts))])
        for src in uniq.tolist():
            if src == ref:
                continue
            delta = bandpass_delta_mag(composition, src, ref)
            out[in_ch & (fid_str == src)] += delta
            logger.debug(
                f"Color-term correction {src} -> {ref} ({ch} channel): {delta:.4f} mag"
            )
    return out


def _prepare_geometry(
    obs: MPCObservations,
    object_coords,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Extract geometry and photometry arrays needed for per-band H fitting.

    Returns (mag, rmsmag, channels, filter_ids, r, delta, alpha_deg, valid_mask).
    ``channels`` holds the resolved g/i/r/u color channel (or ``None``) for each
    row and ``filter_ids`` the canonical vendored filter ID (or ``None``); see
    `_resolve_channels`. valid_mask selects rows with finite mag, finite positive
    rmsmag.
    """
    stn = np.asarray(obs.stn.to_numpy(zero_copy_only=False), dtype=object).astype(str)
    observers = Observers.from_codes(stn, obs.obstime)

    mag = obs.mag.to_numpy(zero_copy_only=False).astype(np.float64)
    rmsmag = obs.rmsmag.to_numpy(zero_copy_only=False).astype(np.float64)
    bands = np.asarray(obs.band.to_numpy(zero_copy_only=False), dtype=object).astype(
        str
    )
    channels, filter_ids = _resolve_channels(stn, bands)

    r, delta, alpha_deg = _compute_geometry(object_coords, observers)
    valid = np.isfinite(mag) & np.isfinite(rmsmag) & (rmsmag > 0)
    return mag, rmsmag, channels, filter_ids, r, delta, alpha_deg, valid


def _band_selector_matrix(channels: np.ndarray) -> np.ndarray:
    """N×4 selector matrix for H_g, H_i, H_r, H_u columns."""
    return np.column_stack([(channels == b).astype(float) for b in _BANDS])


def _fit_per_band_h(
    m_red: np.ndarray,
    alpha_deg: np.ndarray,
    channels: np.ndarray,
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

    ``channels`` is the resolved g/i/r/u color channel for each row (or ``None``);
    see `_resolve_channels`. Observations whose channel is not one of "g", "i", "r",
    "u" are excluded up front and counted as outliers. A channel with zero surviving
    observations is reported as NaN. If, after all exclusions and outlier rejection,
    fewer than `_MIN_RETAINED_FRACTION` of the input rows remain, the fit is
    considered unreliable and raises.

    In all cases the fit is solved with iterative 3-sigma outlier rejection.

    Returns a dict of fit results and diagnostics:

    - "H_g"/"H_i"/"H_r"/"H_u" and their "_sigma": per-band absolute magnitudes and
      1-sigma uncertainties (NaN for an unobserved band).
    - "g_r_sigma"/"g_i_sigma"/"r_i_sigma": color uncertainties, propagated from the
      full parameter covariance so the H_x/H_y correlation is included.
    - "G"/"G_sigma": fitted slope parameter (G for "HG", G12* for "HG12star"; NaN
      for "c1c2") and its uncertainty.
    - "chi2"/"reduced_chi2"/"dof"/"rank": goodness-of-fit over the finally-included
      rows and the design-matrix rank.
    - "converged": whether the (nonlinear) optimizer reported success; always True
      for the linear "c1c2" solve.
    - "num_obs"/"num_outliers".

    Uncertainties come from the (J'*W*J)^-1 covariance rescaled by the reduced
    chi-square, i.e. errors are matched to the observed scatter rather than trusting
    the absolute rmsmag calibration.
    """
    n = len(m_red)
    H_sel = _band_selector_matrix(channels)
    full_weights = root_weights**2

    # Rows whose (station, band) did not resolve to a color channel are already
    # logged in `_resolve_channels`; here they are simply excluded from the fit.
    known_band_mask = np.isin(channels, _BANDS)
    included = known_band_mask.copy()

    if phi_type == "c1c2":
        alpha_rad = np.deg2rad(alpha_deg)
        A = np.column_stack([alpha_rad, alpha_rad**2, H_sel])
        H_idx = 2
    else:
        A = H_sel
        correction_fn = (
            hg12star_correction if phi_type == "HG12star" else hg_phase_correction
        )
        H_init = np.array(
            [
                float(np.mean(m_red[channels == b])) if np.any(channels == b) else 0.0
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

    # Fit diagnostics: goodness of fit and parameter covariance.
    #
    # chi2 is the weighted sum of squared residuals over the finally-included
    # rows; dof = n_incl - num_params. `J` is the weighted design matrix (c1c2)
    # or the optimizer's residual Jacobian (nonlinear), both equal
    # d(weighted residual)/d(params), so cov = (J'J)^-1, rescaled by the reduced
    # chi-square to match the observed scatter. pinv keeps this well-defined when
    # an unobserved band leaves its H column at zero (rank-deficient normal
    # matrix); those bands are then masked out to NaN below.
    n_incl = int(np.sum(included))
    dof = n_incl - num_params
    chi2 = float(np.dot(res, included))
    reduced_chi2 = chi2 / dof if dof > 0 else float("nan")

    if phi_type == "c1c2":
        J = A[included] * root_weights[included, None]
        optimizer_converged = True
    else:
        J = np.asarray(result.jac, dtype=np.float64)
        optimizer_converged = bool(result.success)
    rank = int(np.linalg.matrix_rank(J)) if J.size else 0

    if dof > 0:
        cov = np.linalg.pinv(J.T @ J) * reduced_chi2
        param_sigma = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    else:
        cov = np.full((num_params, num_params), np.nan)
        param_sigma = np.full(num_params, np.nan)

    band_present = [bool(np.any(channels[known_band_mask] == b)) for b in _BANDS]
    H_values = [
        float(values[H_idx + i]) if band_present[i] else float("nan")
        for i in range(len(_BANDS))
    ]
    H_sigma = [
        float(param_sigma[H_idx + i]) if band_present[i] else float("nan")
        for i in range(len(_BANDS))
    ]

    def _color_sigma(i: int, j: int) -> float:
        if not (band_present[i] and band_present[j]):
            return float("nan")
        a, b = H_idx + i, H_idx + j
        var = float(cov[a, a] + cov[b, b] - 2.0 * cov[a, b])
        return float(np.sqrt(var)) if var > 0 else float("nan")

    # _BANDS order is (g, i, r, u) -> indices g=0, i=1, r=2, u=3.
    g_r_sigma = _color_sigma(0, 2)
    g_i_sigma = _color_sigma(0, 1)
    r_i_sigma = _color_sigma(2, 1)

    G_fit = float(values[0]) if phi_type != "c1c2" else float("nan")
    G_sigma = float(param_sigma[0]) if phi_type != "c1c2" else float("nan")

    return {
        "H_g": H_values[0],
        "H_i": H_values[1],
        "H_r": H_values[2],
        "H_u": H_values[3],
        "H_g_sigma": H_sigma[0],
        "H_i_sigma": H_sigma[1],
        "H_r_sigma": H_sigma[2],
        "H_u_sigma": H_sigma[3],
        "g_r_sigma": g_r_sigma,
        "g_i_sigma": g_i_sigma,
        "r_i_sigma": r_i_sigma,
        "G": G_fit,
        "G_sigma": G_sigma,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "dof": dof,
        "rank": rank,
        "converged": optimizer_converged,
        "num_obs": n,
        "num_outliers": num_outliers,
    }


def estimate_colors(
    observations: MPCObservations,
    orbits: MPCOrbits,
    phi_type: Literal["HG12star", "HG", "c1c2"],
    force_g_bounds: bool = True,
    color_term_composition: str | tuple[float, float] | None = None,
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
    force_g_bounds
        Whether to enforce the physical [0, 1] range on the fitted slope
        parameter (G for "HG", G12* for "HG12star"; ignored for "c1c2").  If
        True (default), an out-of-range fit raises ``ValueError``.  If False, it
        is logged as a warning and the out-of-range value is kept -- some
        analyses (e.g. Greenstreet et al.) only reproduce when values outside
        [0, 1] are allowed.
    color_term_composition
        If set, reconcile observations from different filter systems within a
        color channel (e.g. SDSS_g and LSST_g both feeding "g") onto the channel's
        most-observed filter using `bandpass_delta_mag`, assuming this reflectance
        spectrum: a template id ("C", "S", "NEO", "MBA") or a ``(weight_C,
        weight_S)`` tuple. Channels observed through a single filter system are
        unaffected, so this is a no-op unless a channel actually mixes systems.
        If ``None`` (default), no color-term correction is applied and same-letter
        filters are pooled directly (griz inter-system terms are ~0.01 mag; see
        `_apply_color_terms`).

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
        logger.info("Removed %d null bands", len_before - len(observations))
    unique_ids = [
        x for x in pc.unique(observations.requested_provid).to_pylist() if x is not None
    ]

    rows: list[dict[str, object]] = []

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

        # Per-object outputs default to None (no fit produced) and are overwritten
        # when a fit runs. G_fit stays NaN so `_validate_g_bounds` skips objects
        # without a slope parameter (no valid data, or phi_type="c1c2").
        row: dict[str, object] = {
            "object_id": obj_id,
            "g_mag": None,
            "i_mag": None,
            "r_mag": None,
            "u_mag": None,
            "g_mag_sigma": None,
            "i_mag_sigma": None,
            "r_mag_sigma": None,
            "u_mag_sigma": None,
            "g_r": None,
            "g_i": None,
            "r_i": None,
            "g_r_sigma": None,
            "g_i_sigma": None,
            "r_i_sigma": None,
            "phase_param": None,
            "phase_param_sigma": None,
            "chi2": None,
            "reduced_chi2": None,
            "dof": None,
            "rank": None,
            "converged": None,
            "num_obs": len(obs),
            "num_outliers": None,
        }
        G_fit: float = float("nan")

        try:
            mag, rmsmag, channels, filter_ids, r, delta, alpha_deg, valid = (
                _prepare_geometry(obs, object_coords)
            )
            n_invalid = len(obs) - int(np.sum(valid))
            if np.any(valid):
                m_red = reduced_magnitude(mag[valid], r[valid], delta[valid])
                if color_term_composition is not None:
                    m_red = _apply_color_terms(
                        m_red,
                        filter_ids[valid],
                        channels[valid],
                        color_term_composition,
                    )
                root_weights = 1.0 / rmsmag[valid]
                fit = _fit_per_band_h(
                    m_red, alpha_deg[valid], channels[valid], root_weights, phi_type
                )
                G_fit = fit["G"]
                row.update(
                    g_mag=fit["H_g"],
                    i_mag=fit["H_i"],
                    r_mag=fit["H_r"],
                    u_mag=fit["H_u"],
                    g_mag_sigma=fit["H_g_sigma"],
                    i_mag_sigma=fit["H_i_sigma"],
                    r_mag_sigma=fit["H_r_sigma"],
                    u_mag_sigma=fit["H_u_sigma"],
                    g_r=fit["H_g"] - fit["H_r"],
                    g_i=fit["H_g"] - fit["H_i"],
                    r_i=fit["H_r"] - fit["H_i"],
                    g_r_sigma=fit["g_r_sigma"],
                    g_i_sigma=fit["g_i_sigma"],
                    r_i_sigma=fit["r_i_sigma"],
                    phase_param=G_fit,
                    phase_param_sigma=fit["G_sigma"],
                    chi2=fit["chi2"],
                    reduced_chi2=fit["reduced_chi2"],
                    dof=fit["dof"],
                    rank=fit["rank"],
                    converged=fit["converged"],
                    num_outliers=n_invalid + int(fit["num_outliers"]),
                )
            else:
                row["num_outliers"] = n_invalid
        except Exception:
            logger.exception("Problem when fitting colors for %s", obj_id)
            raise

        _validate_g_bounds(G_fit, phi_type, obj_id, force_g_bounds)
        rows.append(row)

    def _col(name: str) -> list[object]:
        return [row[name] for row in rows]

    return ColorFit.from_kwargs(
        object_id=_col("object_id"),
        g_mag=_col("g_mag"),
        i_mag=_col("i_mag"),
        r_mag=_col("r_mag"),
        u_mag=_col("u_mag"),
        g_mag_sigma=_col("g_mag_sigma"),
        i_mag_sigma=_col("i_mag_sigma"),
        r_mag_sigma=_col("r_mag_sigma"),
        u_mag_sigma=_col("u_mag_sigma"),
        g_r=_col("g_r"),
        g_i=_col("g_i"),
        r_i=_col("r_i"),
        g_r_sigma=_col("g_r_sigma"),
        g_i_sigma=_col("g_i_sigma"),
        r_i_sigma=_col("r_i_sigma"),
        phase_param=_col("phase_param"),
        phase_param_sigma=_col("phase_param_sigma"),
        chi2=_col("chi2"),
        reduced_chi2=_col("reduced_chi2"),
        dof=_col("dof"),
        rank=_col("rank"),
        converged=_col("converged"),
        num_obs=_col("num_obs"),
        num_outliers=_col("num_outliers"),
    )
