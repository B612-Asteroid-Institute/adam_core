# Obtaining different colors for asteroids.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow.compute as pc
import quivr as qv
import scipy.optimize

from ..dynamics.propagation import propagate_2body
from ..observers.observers import Observers
from .bandpasses.api import bandpass_delta_mag, map_to_canonical_filter_bands
from .hg12star import (
    _G12STAR_BOUNDS,
    _hg12star_correction_dg12star,
    hg12star_correction,
)
from .lightcurve import reduced_magnitude
from .magnitude import observing_geometry
from .magnitude_common import (
    _HG_G_BOUNDS,
    _hg_phase_correction_dg,
    hg_phase_correction,
)

if TYPE_CHECKING:
    # mpcq is an optional dependency (install via `adam_core[mpc]`); it is only
    # referenced in type annotations here, which `from __future__ import
    # annotations` keeps from being evaluated at runtime. This keeps
    # `import adam_core.photometry` working without mpcq installed.
    from mpcq import MPCObservations
    from mpcq.orbits import MPCOrbits

logger = logging.getLogger(__name__)

# Color channels we fit an absolute magnitude for.
_BANDS = ("g", "i", "r", "u")
# The canonical vendored filter IDs allowed to feed each color channel.  This is
# an explicit allow-list rather than the filter ID's base letter, which would
# also pool e.g. Bessell_R (a Vega-system Cousins R) into "r".  A filter is
# listed when, for every vendored reflectance template, its color term onto the
# channel's target filter (`_TARGET_FILTERS`) is within 0.03 mag, so pooling it
# uncorrected costs no more than that.  Excluded on those grounds: SkyMapper_g
# (up to 0.08 mag) and every non-LSST u filter (0.2-0.4 mag).
_CHANNEL_FILTERS: dict[str, tuple[str, ...]] = {
    "g": ("LSST_g", "SDSS_g", "PS1_g", "ZTF_g", "DECam_g", "BASS_g"),
    "i": ("LSST_i", "SDSS_i", "PS1_i", "ZTF_i", "DECam_i", "SkyMapper_i"),
    "r": ("LSST_r", "SDSS_r", "PS1_r", "ZTF_r", "DECam_r", "BASS_r", "SkyMapper_r"),
    "u": ("LSST_u",),
}
_FILTER_CHANNEL = {fid: ch for ch, fids in _CHANNEL_FILTERS.items() for fid in fids}
# The single photometric system that `color_term_composition` normalizes every
# channel onto.
_TARGET_FILTERS = {ch: f"LSST_{ch}" for ch in _BANDS}
_PHI_TYPES = ("HG12star", "HG", "c1c2")
# If fewer than this fraction of an object's eligible observations survive
# outlier rejection, the fit is not trustworthy enough to report silently.
# Eligible means it could have entered the fit at all: finite magnitude and
# uncertainty, through a filter that resolves to a g/i/r/u color channel.
# Observations dropped for being invalid or for coming through an unsupported
# filter say nothing about the quality of the fit (an object observed cleanly
# in g and r alongside a pile of unfiltered reports is still cleanly fit) so
# they are reported separately and kept out of this denominator.
_MIN_RETAINED_FRACTION = 0.5
# Validity bounds on each model's slope parameter.  They are model-specific:
# G12* is defined on [0, 1], while for G the bounds are where the implemented H-G
# formula still dims monotonically with phase angle (see `_HG_G_BOUNDS`).
_G_BOUNDS = {"HG12star": _G12STAR_BOUNDS, "HG": _HG_G_BOUNDS}
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


def _g_bounds_violation(G: float, phi_type: str, obj_id: str) -> str | None:
    """
    Describe how a fitted slope parameter violates its model's validity bounds.

    Returns ``None`` when ``G`` is within bounds, or when there is nothing to check
    ("c1c2" has no slope parameter, and ``G`` is NaN when no fit was produced).
    """
    if phi_type == "c1c2" or not np.isfinite(G):
        return None
    lo, hi = _G_BOUNDS[phi_type]
    if lo <= G <= hi:
        return None
    label = _G_LABELS[phi_type]
    return (
        f"Fitted {label} = {G:.4f} for {obj_id} is outside the {phi_type} "
        f"validity bounds [{lo:g}, {hi:g}]"
    )


class _UnreliableFitError(ValueError):
    """Outlier rejection discarded too much of an object's photometry to trust."""


# Per-object fit outcomes reported in `ColorFit.status`; see `estimate_colors`.
_STATUSES = (
    "ok",
    "no_orbit",
    "multiple_orbits",
    "no_usable_photometry",
    "too_many_outliers",
    "phase_param_out_of_bounds",
    "failed",
)
# Results withheld when a fit is rejected for an out-of-bounds slope parameter.
# The slope itself and the fit diagnostics are kept to show why.
_PHOTOMETRIC_FIELDS = tuple(
    [f"{b}_mag" for b in _BANDS]
    + [f"{b}_mag_sigma" for b in _BANDS]
    + ["g_r", "g_i", "r_i", "g_r_sigma", "g_i_sigma", "r_i_sigma"]
)


def _mark_failed(row: dict[str, object], status: str, error: str) -> None:
    """Record why an object's fit produced no usable result, and log it."""
    row.update(status=status, error=error)
    logger.warning("No colors for %s (%s): %s", row["object_id"], status, error)


class ColorFit(qv.Table):
    object_id = qv.LargeStringColumn()
    # Outcome of this object's fit: "ok" or the reason it produced no usable
    # result (one of `_STATUSES`; see `estimate_colors`), with a human-readable
    # explanation in `error` whenever the status is not "ok".
    status = qv.LargeStringColumn()
    error = qv.LargeStringColumn(nullable=True)
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
    # The photometric system each band's magnitude is in: the canonical filter ID
    # of the observations used (e.g. "LSST_g"), or several IDs joined with "+"
    # when the band pooled filters without a color-term correction.  With
    # `color_term_composition` set this is always the target filter (LSST_g,
    # LSST_i, LSST_r, LSST_u).  ``None`` for a band with no used observations.
    # Colors are system-specific: g_r is in the system of g_filter minus r_filter.
    g_filter = qv.LargeStringColumn(nullable=True)
    i_filter = qv.LargeStringColumn(nullable=True)
    r_filter = qv.LargeStringColumn(nullable=True)
    u_filter = qv.LargeStringColumn(nullable=True)
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
    # Where this object's observations went.  These partition `num_obs` exactly:
    # num_obs = num_invalid + num_unsupported_filter + num_clipped + num_used.
    num_obs = qv.Int64Column(nullable=True)
    # Non-finite magnitude, or a missing/non-positive reported uncertainty.
    num_invalid = qv.Int64Column(nullable=True)
    # Valid photometry whose (station, band) does not resolve to a g/i/r/u
    # channel: V, PS1_w, unfiltered reports.  Never entered the fit, and so is
    # not a statement about its quality (see `_MIN_RETAINED_FRACTION`). Invalid
    # takes precedence, so this can be smaller than the count `_resolve_channels`
    # logs: a row with no magnitude is unfittable whatever its filter.
    num_unsupported_filter = qv.Int64Column(nullable=True)
    # Eligible observations rejected by the iterative 3-sigma clip.
    num_clipped = qv.Int64Column(nullable=True)
    # Observations the reported fit is actually based on, in total and per band.
    num_used = qv.Int64Column(nullable=True)
    num_used_g = qv.Int64Column(nullable=True)
    num_used_i = qv.Int64Column(nullable=True)
    num_used_r = qv.Int64Column(nullable=True)
    num_used_u = qv.Int64Column(nullable=True)


def _resolve_channels(
    stn: np.ndarray, bands: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map raw (observatory_code, reported_band) pairs to g/i/r/u color channels.

    Rather than matching MPC band strings literally, this routes each observation
    through the shared `map_to_canonical_filter_bands` utility, which resolves
    `(observatory_code, band)` to a canonical vendored filter ID (handling MPC/ADES
    label quirks, e.g. G96 "G" -> SDSS_g, ATLAS "o" -> ATLAS_o, LSST encodings, etc).
    The canonical filter is then looked up in the `_CHANNEL_FILTERS` allow-list, so
    only filters close to the channel's target passband contribute to its fit.
    Anything else (V, PS1_w, Bessell_R, SkyMapper_g, SDSS_u, ...) and rows with no
    resolvable filter are returned as ``None`` (excluded downstream).

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
        channels[i] = None if fid is None else _FILTER_CHANNEL.get(str(fid))

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
    Normalize reduced magnitudes onto the channel's target filter (`_TARGET_FILTERS`).

    A channel may pool observations taken through different filters (e.g. SDSS_g
    and LSST_g both feed the "g" channel), and different channels may be observed
    in different systems (e.g. LSST_g with SDSS_r). This converts every row onto
    its channel's target filter using `bandpass_delta_mag`:

        m_red_target = m_red + Δm(composition, filter_id -> target_filter)

    so all channels, and hence the colors between them, end up in one defined
    system regardless of which filters the data happen to contain. Rows already
    in the target filter are unchanged. ``composition`` is a template id
    "C"/"S"/"NEO"/"MBA" or a ``(weight_C, weight_S)`` tuple.
    """
    out = np.asarray(m_red, dtype=np.float64).copy()
    fid_str = np.array([str(f) for f in filter_ids.tolist()], dtype=object)
    for ch in _BANDS:
        in_ch = channels == ch
        if not np.any(in_ch):
            continue
        ref = _TARGET_FILTERS[ch]
        for src in np.unique(fid_str[in_ch]).tolist():
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


def _band_filters(
    filter_ids: np.ndarray, channels: np.ndarray, included: np.ndarray
) -> dict[str, str | None]:
    """
    The photometric system each band's magnitude is in, from the rows used.

    A single filter ID when every used row of the band shares it, or the sorted
    IDs joined with "+" (e.g. "LSST_g+SDSS_g") when the band pools filters
    without a color-term correction; ``None`` for a band with no used rows.
    """
    out: dict[str, str | None] = {}
    for b in _BANDS:
        fids = sorted({str(f) for f in filter_ids[included & (channels == b)]})
        out[b] = "+".join(fids) if fids else None
    return out


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
) -> dict[str, Any]:
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
    - "num_obs"/"num_unsupported_filter"/"num_clipped"/"num_used": where the rows
      passed in ended up; these three partition "num_obs". (Rows that
      failed validity filtering never reach this function, so `estimate_colors`
      adds the invalid count itself.)
    - "num_used_*": per-band retained counts, summing to "num_used".
    - "included": boolean mask of the rows the final fit is based on.

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
    # They are ineligible rather than rejected: they are counted and reported
    # apart from the statistical clip, and kept out of its retention threshold.
    eligible = np.isin(channels, _BANDS)
    n_eligible = int(np.sum(eligible))
    n_unsupported = n - n_eligible
    if n_eligible == 0:
        raise ValueError(
            f"None of the {n} observations resolve to a g/i/r/u color channel; "
            "there is nothing to fit."
        )
    included = eligible.copy()

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

    num_clipped = n_eligible - n_incl
    if n_incl < _MIN_RETAINED_FRACTION * n_eligible:
        raise _UnreliableFitError(
            f"Outlier rejection removed {num_clipped} of the {n_eligible} "
            f"observation(s) eligible for the color fit (more than "
            f"{1 - _MIN_RETAINED_FRACTION:.0%} of them); fit is unreliable. "
            f"A further {n_unsupported} observation(s) had no g/i/r/u filter and "
            "are excluded from this threshold."
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
            if not _is_fitable(null_basis, _unit(i))
        )
        logger.warning(
            f"Weighted Jacobian for the {phi_type} fit is rank deficient "
            f"(rank {rank} of {num_params} parameters from {n_incl} observation(s)"
            f" in band(s) {''.join(active)}); reporting the "
            f"non-fitable parameter(s) {non_fit} as NaN"
        )

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

    num_used = {b: int(np.sum(channels[included] == b)) for b in _BANDS}

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
        "num_unsupported_filter": n_unsupported,
        "num_clipped": num_clipped,
        "num_used": n_incl,
        "num_used_g": num_used["g"],
        "num_used_i": num_used["i"],
        "num_used_r": num_used["r"],
        "num_used_u": num_used["u"],
        "included": included,
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
        Whether to enforce the model's validity bounds on the fitted slope
        parameter (ignored for "c1c2"): [0, 1] for G12* ("HG12star"), and
        [-0.255, 1] for G ("HG"), the range over which the implemented H-G
        formula dims monotonically with phase angle.  If True (default), an
        out-of-bounds fit is rejected: its status is "phase_param_out_of_bounds"
        and its magnitudes and colors are withheld (the slope and fit
        diagnostics are kept to show why).  If False, it is logged as a warning
        and the fit is kept. Some analyses (e.g. Greenstreet et al.) only
        reproduce when G12* outside [0, 1] is allowed.
    color_term_composition
        If set, convert every observation onto one defined photometric system,
        LSST g/i/r/u, using `bandpass_delta_mag` and assuming this reflectance
        spectrum: a template id ("C", "S", "NEO", "MBA") or a ``(weight_C,
        weight_S)`` tuple.  Magnitudes and colors are then LSST-system whatever
        filters the data contain.  If ``None`` (default), no color-term correction
        is applied: each band is fit in the filter(s) it was observed through, and
        the allowed filters of a channel are pooled directly (their color terms
        onto LSST are within 0.03 mag; see `_CHANNEL_FILTERS`).

    Returns
    -------
    ColorFit
        One row per unique object in ``observations``.  Objects are fit
        independently, and one object's failure never aborts the batch: its row
        carries a ``status`` other than "ok" and an ``error`` message, and no
        magnitudes or colors.  The statuses are:

        - "ok": fit succeeded.
        - "no_orbit" / "multiple_orbits": ``orbits`` has no, or more than one,
          orbit for the object.
        - "no_usable_photometry": no observation has a valid magnitude and
          uncertainty through a g/i/r/u filter.
        - "too_many_outliers": outlier rejection discarded more than half of the
          eligible observations, so the fit is not trusted.
        - "phase_param_out_of_bounds": the fitted slope parameter is outside its
          validity bounds and ``force_g_bounds`` is True.
        - "failed": any other error while fitting this object (the traceback is
          logged).

        Magnitudes and colors are system-specific: the ``*_filter`` columns give
        the filter each band's magnitude is in (always LSST with
        ``color_term_composition`` set), and a color such as ``g_r`` is in the
        system of ``g_filter`` minus ``r_filter``.

    Raises
    ------
    ValueError
        If ``phi_type`` is not supported.  Invalid arguments affect every object,
        so they still fail fast.
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

        # Per-object outputs default to None (no fit produced) and are overwritten
        # when a fit runs.
        row: dict[str, object] = {
            "object_id": obj_id,
            "status": "ok",
            "error": None,
            "g_mag": None,
            "i_mag": None,
            "r_mag": None,
            "u_mag": None,
            "g_mag_sigma": None,
            "i_mag_sigma": None,
            "r_mag_sigma": None,
            "u_mag_sigma": None,
            "g_filter": None,
            "i_filter": None,
            "r_filter": None,
            "u_filter": None,
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
            "num_invalid": None,
            "num_unsupported_filter": None,
            "num_clipped": None,
            "num_used": None,
            "num_used_g": None,
            "num_used_i": None,
            "num_used_r": None,
            "num_used_u": None,
        }
        rows.append(row)

        orb_mask = pc.equal(orbits.requested_provid, obj_id)
        orb = orbits.apply_mask(orb_mask)
        if len(orb) != 1:
            if len(orb) == 0:
                _mark_failed(row, "no_orbit", "No orbit given for this object")
            else:
                _mark_failed(
                    row,
                    "multiple_orbits",
                    f"Expected exactly one orbit, got {len(orb)}",
                )
            continue

        try:
            adam_orbits = orb.orbits()
            propagated = propagate_2body(adam_orbits, obs.obstime)
            object_coords = propagated.coordinates

            mag, rmsmag, channels, filter_ids, r, delta, alpha_deg, valid = (
                _prepare_geometry(obs, object_coords)
            )
            n_invalid = len(obs) - int(np.sum(valid))
            row.update(num_invalid=n_invalid)
            if not np.any(valid & np.isin(channels, _BANDS)):
                # Nothing to fit: every observation is invalid or comes through a
                # filter outside the g/i/r/u channels.
                row.update(
                    num_unsupported_filter=int(np.sum(valid)),
                    num_clipped=0,
                    num_used=0,
                    num_used_g=0,
                    num_used_i=0,
                    num_used_r=0,
                    num_used_u=0,
                )
                _mark_failed(
                    row,
                    "no_usable_photometry",
                    f"None of the {len(obs)} observation(s) has a valid magnitude "
                    "through a g/i/r/u filter",
                )
                continue

            m_red = reduced_magnitude(mag[valid], r[valid], delta[valid])
            # The filter each row's magnitude is expressed in once it enters
            # the fit: as observed, or the channel's target if converted.
            row_filters = filter_ids[valid]
            if color_term_composition is not None:
                m_red = _apply_color_terms(
                    m_red,
                    filter_ids[valid],
                    channels[valid],
                    color_term_composition,
                )
                row_filters = np.array(
                    [_TARGET_FILTERS.get(ch) for ch in channels[valid]],
                    dtype=object,
                )
            root_weights = 1.0 / rmsmag[valid]
            fit = _fit_per_band_h(
                m_red, alpha_deg[valid], channels[valid], root_weights, phi_type
            )
        except _UnreliableFitError as exc:
            _mark_failed(row, "too_many_outliers", str(exc))
            continue
        except Exception as exc:
            # Anything else is unexpected; keep the traceback in the log, but do
            # not let one object abort the whole batch.
            logger.exception("Problem when fitting colors for %s", obj_id)
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            continue

        G_fit = fit["G"]
        band_filters = _band_filters(row_filters, channels[valid], fit["included"])
        row.update(
            g_mag=fit["H_g"],
            i_mag=fit["H_i"],
            r_mag=fit["H_r"],
            u_mag=fit["H_u"],
            g_mag_sigma=fit["H_g_sigma"],
            i_mag_sigma=fit["H_i_sigma"],
            r_mag_sigma=fit["H_r_sigma"],
            u_mag_sigma=fit["H_u_sigma"],
            g_filter=band_filters["g"],
            i_filter=band_filters["i"],
            r_filter=band_filters["r"],
            u_filter=band_filters["u"],
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
            num_unsupported_filter=int(fit["num_unsupported_filter"]),
            num_clipped=int(fit["num_clipped"]),
            num_used=int(fit["num_used"]),
            num_used_g=int(fit["num_used_g"]),
            num_used_i=int(fit["num_used_i"]),
            num_used_r=int(fit["num_used_r"]),
            num_used_u=int(fit["num_used_u"]),
        )

        violation = _g_bounds_violation(G_fit, phi_type, obj_id)
        if violation is not None:
            if force_g_bounds:
                row.update(dict.fromkeys(_PHOTOMETRIC_FIELDS))
                _mark_failed(row, "phase_param_out_of_bounds", violation)
            else:
                logger.warning("%s; keeping it because force_g_bounds=False", violation)

    def _col(name: str) -> list[object]:
        return [row[name] for row in rows]

    return ColorFit.from_kwargs(
        object_id=_col("object_id"),
        status=_col("status"),
        error=_col("error"),
        g_mag=_col("g_mag"),
        i_mag=_col("i_mag"),
        r_mag=_col("r_mag"),
        u_mag=_col("u_mag"),
        g_mag_sigma=_col("g_mag_sigma"),
        i_mag_sigma=_col("i_mag_sigma"),
        r_mag_sigma=_col("r_mag_sigma"),
        u_mag_sigma=_col("u_mag_sigma"),
        g_filter=_col("g_filter"),
        i_filter=_col("i_filter"),
        r_filter=_col("r_filter"),
        u_filter=_col("u_filter"),
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
        num_invalid=_col("num_invalid"),
        num_unsupported_filter=_col("num_unsupported_filter"),
        num_clipped=_col("num_clipped"),
        num_used=_col("num_used"),
        num_used_g=_col("num_used_g"),
        num_used_i=_col("num_used_i"),
        num_used_r=_col("num_used_r"),
        num_used_u=_col("num_used_u"),
    )
