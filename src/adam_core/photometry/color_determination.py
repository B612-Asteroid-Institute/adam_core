# Obtaining different colors for asteroids.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow.compute as pc
import quivr as qv
import scipy.optimize

from ..dynamics.propagation import propagate_2body
from ..observers.observers import Observers
from .bandpasses.api import bandpass_delta_mag, map_to_canonical_filter_bands
from .hg12star import _hg12star_correction_dg12star, hg12star_correction
from .lightcurve import reduced_magnitude
from .magnitude import observing_geometry
from .magnitude_common import _hg_phase_correction_dg, hg_phase_correction

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
# A parameter combination counts as non-fittable once its component along the
# weighted Jacobian's null space exceeds this fraction of its own norm.  The
# Jacobian is analytic (see `_hg12star_correction_dg12star`), so an exactly
# degenerate direction leaks at rounding level and a determined one at zero.
_FITABILITY_TOL = 1e-8
# A 3-sigma clip is meaningless once the model reproduces every included
# magnitude to a rounding-level fraction of its quoted uncertainty: the observed
# "scatter" is then pure floating-point noise, and clipping against it runs away,
# rejecting good observations until the residuals are identically zero. Below
# this reduced chi-square the data are treated as exactly fit and nothing is
# clipped.
_MIN_CLIP_REDUCED_CHI2 = 1e-12
# The two nonlinear phase models, each paired with its analytic slope derivative.
# "c1c2" is absent because it is linear: its phase terms are design-matrix
# columns rather than a nonlinear parameter (see `_design_matrix`).
_PhaseCorrection = Callable[[np.ndarray, float], np.ndarray]
_PHASE_MODELS: dict[str, tuple[_PhaseCorrection, _PhaseCorrection]] = {
    "HG12star": (hg12star_correction, _hg12star_correction_dg12star),
    "HG": (hg_phase_correction, _hg_phase_correction_dg),
}


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
    # to the observed scatter (see `_fit_per_band_h`). NaN for a band that was not
    # observed or whose magnitude the data cannot determine on its own.
    g_mag_sigma = qv.Float64Column(nullable=True)
    i_mag_sigma = qv.Float64Column(nullable=True)
    r_mag_sigma = qv.Float64Column(nullable=True)
    u_mag_sigma = qv.Float64Column(nullable=True)
    g_r = qv.Float64Column(nullable=True)
    g_i = qv.Float64Column(nullable=True)
    r_i = qv.Float64Column(nullable=True)
    # Color uncertainties, propagated from the full parameter covariance (so the
    # H_x/H_y correlation through the shared phase parameter is accounted for).
    # A color is reported whenever the contrast is fitable, which can be the
    # case even when neither absolute magnitude individually is.
    g_r_sigma = qv.Float64Column(nullable=True)
    g_i_sigma = qv.Float64Column(nullable=True)
    r_i_sigma = qv.Float64Column(nullable=True)
    # Fitted phase slope parameter (G for "HG", G12* for "HG12star"; NaN for
    # "c1c2") and its 1-sigma uncertainty.
    phase_param = qv.Float64Column(nullable=True)
    phase_param_sigma = qv.Float64Column(nullable=True)
    # Fit-quality diagnostics over the finally-included observations.  `dof` is
    # the number of included observations minus the rank of the weighted
    # Jacobian, and `num_params` counts the parameters actually fitted (phase
    # terms plus one magnitude per observed band).  `rank < num_params` flags a
    # rank-deficient fit, whose non-fitable parameters are reported as NaN.
    chi2 = qv.Float64Column(nullable=True)
    reduced_chi2 = qv.Float64Column(nullable=True)
    dof = qv.Int64Column(nullable=True)
    rank = qv.Int64Column(nullable=True)
    num_params = qv.Int64Column(nullable=True)
    converged = qv.BooleanColumn(nullable=True)
    num_obs = qv.Int64Column(nullable=True)
    num_outliers = qv.Int64Column(nullable=True)


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

    r, delta, alpha_deg = observing_geometry(object_coords, observers)
    valid = np.isfinite(mag) & np.isfinite(rmsmag) & (rmsmag > 0)
    return mag, rmsmag, channels, filter_ids, r, delta, alpha_deg, valid


def _active_bands(channels: np.ndarray, included: np.ndarray) -> list[str]:
    """Bands with at least one included observation, in `_BANDS` order."""
    return [b for b in _BANDS if np.any(channels[included] == b)]


def _design_matrix(
    phi_type: str,
    alpha_rad: np.ndarray,
    channels: np.ndarray,
    active_bands: list[str],
) -> np.ndarray:
    """
    Linear design matrix for the currently active parameters, all rows.

    Columns follow the parameter order: the ``c1``/``c2`` polynomial terms for
    "c1c2" (the nonlinear models contribute their slope column through the
    Jacobian instead), then one selector column per observed band. A band with
    no observations gets no column at all. An all-zero column would inflate the
    nominal parameter count without constraining anything, which would corrupt
    the degrees of freedom.
    """
    cols = [alpha_rad, alpha_rad**2] if phi_type == "c1c2" else []
    cols += [(channels == b).astype(float) for b in active_bands]
    return np.column_stack(cols)


def _rank_and_null_space(J: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Numerical rank of ``J`` and an orthonormal basis for its null space.

    Both are read off a single SVD using the same singular-value threshold as
    ``numpy.linalg.matrix_rank``, so the rank plus the number of returned null
    directions always equals the number of columns (parameters).
    """
    n_rows, n_cols = J.shape
    if J.size == 0:
        return 0, np.eye(n_cols)
    _, sv, Vt = np.linalg.svd(J, full_matrices=True)
    tol = float(sv[0]) * max(n_rows, n_cols) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(sv > tol))
    return rank, Vt[rank:].T


def _is_fitable(null_basis: np.ndarray, vector: np.ndarray) -> bool:
    """
    Whether the parameter combination ``vector`` is determined by the data.

    A combination is fitable only if it lies in the row space of the weighted
    Jacobian, i.e. has no component along a direction the data leaves free.
    Individual parameters can be non-fitable while contrasts between them (the
    colors) are perfectly well determined: with every band observed at a single
    common phase angle, for instance, the slope trades off against all the
    absolute magnitudes by the same amount, which cancels in ``H_x - H_y``. The
    pseudoinverse would otherwise hand back a finite, arbitrary number for every
    parameter and imply they were all identified.
    """
    if null_basis.shape[1] == 0:
        return True
    leak = float(np.linalg.norm(null_basis.T @ vector))
    return leak <= _FITABILITY_TOL * float(np.linalg.norm(vector))


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

    Only bands that actually have included observations become parameters, and the
    active set is recomputed on every iteration (clipping can empty a band). The
    scatter driving the clip, the final degrees of freedom and the reported
    uncertainties are all based on the rank of the weighted Jacobian rather than
    on a nominal parameter count, so a sparse or single-band fit still gets a
    finite scatter estimate and a working outlier clip.

    Rank alone is not sufficient, though: a full set of active parameters can still
    be individually non-identifiable (e.g. a single phase angle leaves the slope
    and the absolute magnitudes free to trade off). Every reported quantity is
    therefore checked for fitability against the Jacobian's null space and
    reported as NaN when the data do not determine it, separately for the
    absolute magnitudes, the slope, and each color contrast, since a contrast can
    be well determined when neither of its magnitudes is.

    Returns a dict of fit results and diagnostics:

    - "H_g"/"H_i"/"H_r"/"H_u" and their "_sigma": per-band absolute magnitudes and
      1-sigma uncertainties (NaN for an unobserved or non-fitable band).
    - "g_r"/"g_i"/"r_i" and their "_sigma": colors and their uncertainties,
      propagated from the full parameter covariance so the H_x/H_y correlation is
      included.
    - "G"/"G_sigma": fitted slope parameter (G for "HG", G12* for "HG12star"; NaN
      for "c1c2") and its uncertainty.
    - "chi2"/"reduced_chi2"/"dof"/"rank"/"num_params": goodness of fit over the
      finally-included rows, the Jacobian rank and the number of parameters
      actually fitted. ``rank < num_params`` flags a rank-deficient fit.
    - "converged": whether the (nonlinear) optimizer reported success; always True
      for the linear "c1c2" solve.
    - "num_obs"/"num_outliers".

    Uncertainties come from the (J'*W*J)^-1 covariance rescaled by the reduced
    chi-square, i.e. errors are matched to the observed scatter rather than trusting
    the absolute rmsmag calibration.
    """
    n = len(m_red)
    m_red = np.asarray(m_red, dtype=np.float64)
    alpha_deg = np.asarray(alpha_deg, dtype=np.float64)
    alpha_rad = np.deg2rad(alpha_deg)
    root_weights = np.asarray(root_weights, dtype=np.float64)
    full_weights = root_weights**2
    Bw = m_red * root_weights

    # Rows whose (station, band) did not resolve to a color channel are already
    # logged in `_resolve_channels`; here they are simply excluded from the fit.
    included = np.isin(channels, _BANDS)
    if not np.any(included):
        raise ValueError(
            f"None of the {n} observations resolve to a g/i/r/u color channel; "
            "there is nothing to fit."
        )

    # Parameters preceding the per-band magnitudes: (c1, c2) for the polynomial
    # model, the single slope for H-G / HG12*.  The nonlinear models also need
    # d(correction)/d(slope); supplying it analytically keeps the Jacobian (and
    # so the rank and fitability tests below) exact.
    n_phase = 2 if phi_type == "c1c2" else 1
    correction_fn, correction_dg = _PHASE_MODELS.get(phi_type, (None, None))

    slope_guess = 0.15
    while True:
        active = _active_bands(channels, included)
        A = _design_matrix(phi_type, alpha_rad, channels, active)
        Aw = A * root_weights[:, None]

        if phi_type == "c1c2":
            values, _, _, _ = np.linalg.lstsq(Aw[included], Bw[included], rcond=None)
            model = A @ values
            J = Aw[included]
            optimizer_converged = True
        else:
            assert correction_fn is not None and correction_dg is not None
            H_init = [float(np.mean(m_red[included & (channels == b)])) for b in active]
            params0 = np.concatenate([[slope_guess], H_init])

            def residual(par: np.ndarray) -> np.ndarray:
                corr_w = correction_fn(alpha_deg, par[0]) * root_weights
                return np.asarray(
                    Aw[included] @ par[1:] + corr_w[included] - Bw[included]
                )

            def residual_jac(par: np.ndarray) -> np.ndarray:
                # d(weighted residual)/d(params): the slope column, then the
                # (parameter-independent) weighted band selectors.
                dcorr_w = correction_dg(alpha_deg, par[0]) * root_weights
                return np.column_stack([dcorr_w[included], Aw[included]])

            result = scipy.optimize.least_squares(
                residual, params0, jac=residual_jac, verbose=0
            )
            values = result.x
            # Warm-start the next iteration's slope; the band parameters are
            # re-seeded above because the active set may have shrunk.
            slope_guess = float(values[0])
            model = A @ values[1:] + correction_fn(alpha_deg, values[0])
            J = np.asarray(result.jac, dtype=np.float64)
            optimizer_converged = bool(result.success)

        res = (model - m_red) ** 2 * full_weights
        n_incl = int(np.sum(included))
        rank, null_basis = _rank_and_null_space(J)
        chi2 = float(np.dot(res, included))
        dof = n_incl - rank
        # `res` and `sigma2` are both squared, so `9 *` is a 3-sigma clip. Using
        # the Jacobian rank keeps the scatter finite for small, few-band fits
        # where a nominal five-parameter count would give dof <= 0 and silently
        # disable clipping altogether.
        sigma2 = chi2 / dof if dof > 0 else np.inf
        if sigma2 < _MIN_CLIP_REDUCED_CHI2:
            # No usable scatter estimate (dof <= 0, or an exact fit); clip nothing.
            sigma2 = np.inf
        outliers = res > 9 * sigma2
        # if no new outliers, we have converged
        if not np.any(outliers & included):
            break
        included &= ~outliers

    num_outliers = int(np.sum(~included))
    if n - num_outliers < _MIN_RETAINED_FRACTION * n:
        raise ValueError(
            f"Outlier/band rejection removed {num_outliers}/{n} observations "
            f"(more than {1 - _MIN_RETAINED_FRACTION:.0%} of the data); fit is unreliable."
        )

    # Fit diagnostics: goodness of fit and parameter covariance.
    #
    # `J` is the weighted design matrix (c1c2) or the optimizer's residual
    # Jacobian (nonlinear), both equal d(weighted residual)/d(params), so
    # cov = (J'J)^-1 rescaled by the reduced chi-square to match the observed
    # scatter. pinv keeps that well-defined when the Jacobian is rank deficient;
    # the parameters it cannot actually determine are masked to NaN below.
    num_params = int(J.shape[1])
    reduced_chi2 = chi2 / dof if dof > 0 else float("nan")
    if dof > 0:
        cov = np.linalg.pinv(J.T @ J) * reduced_chi2
    else:
        cov = np.full((num_params, num_params), np.nan)

    def _unit(index: int) -> np.ndarray:
        vector = np.zeros(num_params)
        vector[index] = 1.0
        return vector

    band_param = {b: n_phase + i for i, b in enumerate(active)}

    if rank < num_params:
        labels = (["c1", "c2"] if phi_type == "c1c2" else [_G_LABELS[phi_type]]) + [
            f"H_{b}" for b in active
        ]
        non_fit = ", ".join(
                        label
                        for i, label in enumerate(labels)
                        if not _is_fitable(null_basis, _unit(i)))
        logger.warning(f"Weighted Jacobian for the {phi_type} fit is rank deficient "
                       f"(rank {rank} of {num_params} parameters from {n_incl} observation(s)"
                       f" in band(s) {''.join(active)}); reporting the "
                       f"non-fitable parameter(s) {non_fit} as NaN")

    def _param(index: int) -> tuple[float, float]:
        """(value, 1-sigma) for one fitted parameter; NaN when non-fitable."""
        if not _is_fitable(null_basis, _unit(index)):
            return float("nan"), float("nan")
        if dof <= 0:
            return float(values[index]), float("nan")
        var = float(cov[index, index])
        return float(values[index]), (float(np.sqrt(var)) if var > 0 else float("nan"))

    def _color(band_a: str, band_b: str) -> tuple[float, float]:
        """
        (H_a - H_b, 1-sigma) for one color, read off the raw solution.

        Taken from `values`/`cov` rather than from the per-band results below, so
        a fitable contrast survives even when neither absolute magnitude
        individually is.
        """
        if band_a not in band_param or band_b not in band_param:
            return float("nan"), float("nan")
        i, j = band_param[band_a], band_param[band_b]
        if not _is_fitable(null_basis, _unit(i) - _unit(j)):
            return float("nan"), float("nan")
        color = float(values[i] - values[j])
        if dof <= 0:
            return color, float("nan")
        var = float(cov[i, i] + cov[j, j] - 2.0 * cov[i, j])
        return color, (float(np.sqrt(var)) if var > 0 else float("nan"))

    H_fit: dict[str, tuple[float, float]] = {}
    for band in _BANDS:
        H_fit[band] = (
            _param(band_param[band])
            if band in band_param
            else (float("nan"), float("nan"))
        )

    g_r, g_r_sigma = _color("g", "r")
    g_i, g_i_sigma = _color("g", "i")
    r_i, r_i_sigma = _color("r", "i")

    if phi_type == "c1c2":
        G_fit, G_sigma = float("nan"), float("nan")
    else:
        G_fit, G_sigma = _param(0)

    return {
        "H_g": H_fit["g"][0],
        "H_i": H_fit["i"][0],
        "H_r": H_fit["r"][0],
        "H_u": H_fit["u"][0],
        "H_g_sigma": H_fit["g"][1],
        "H_i_sigma": H_fit["i"][1],
        "H_r_sigma": H_fit["r"][1],
        "H_u_sigma": H_fit["u"][1],
        "g_r": g_r,
        "g_i": g_i,
        "r_i": r_i,
        "g_r_sigma": g_r_sigma,
        "g_i_sigma": g_i_sigma,
        "r_i_sigma": r_i_sigma,
        "G": G_fit,
        "G_sigma": G_sigma,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "dof": dof,
        "rank": rank,
        "num_params": num_params,
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
            "num_params": None,
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
                    g_r=fit["g_r"],
                    g_i=fit["g_i"],
                    r_i=fit["r_i"],
                    g_r_sigma=fit["g_r_sigma"],
                    g_i_sigma=fit["g_i_sigma"],
                    r_i_sigma=fit["r_i_sigma"],
                    phase_param=G_fit,
                    phase_param_sigma=fit["G_sigma"],
                    chi2=fit["chi2"],
                    reduced_chi2=fit["reduced_chi2"],
                    dof=fit["dof"],
                    rank=fit["rank"],
                    num_params=fit["num_params"],
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
        num_params=_col("num_params"),
        converged=_col("converged"),
        num_obs=_col("num_obs"),
        num_outliers=_col("num_outliers"),
    )
