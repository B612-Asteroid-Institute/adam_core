"""
Runtime APIs for the bandpass-driven photometry implementation.

Bandpass curves are vendored from the SVO Filter Profile Service. Please see
`REFERENCES.md` for the required acknowledgement and citations when using this
service.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Final, Iterable

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .tables import (
    AsteroidTemplates,
    BandpassCurves,
    ObservatoryBandMap,
    TemplateBandpassIntegrals,
)

_DATA_DIR = files("adam_core.photometry.bandpasses").joinpath("data")

_BANDPASS_CURVES_FILE: Final[str] = "bandpass_curves.parquet"
_OBS_BAND_MAP_FILE: Final[str] = "observatory_band_map.parquet"
_TEMPLATES_FILE: Final[str] = "asteroid_templates.parquet"
_INTEGRALS_FILE: Final[str] = "template_bandpass_integrals.parquet"
_SOLAR_SPECTRUM_FILE: Final[str] = "solar_spectrum.parquet"


def _to_string_array(
    values: pa.Array | pa.ChunkedArray | npt.NDArray[np.object_] | Iterable[str],
) -> pa.Array | pa.ChunkedArray:
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return pc.cast(values, pa.large_string())
    arr = pa.array(list(values), type=pa.large_string())
    return arr


def _key_from_code_band(
    code: pa.Array | pa.ChunkedArray, band: pa.Array | pa.ChunkedArray
) -> pa.Array | pa.ChunkedArray:
    # Avoid Python loops; Arrow string concatenation is fast.
    sep = pa.scalar("|", type=pa.large_string())
    return pc.binary_join_element_wise(code, band, sep)


def _normalize_reported_band_for_station(
    codes: pa.Array | pa.ChunkedArray,
    bands: pa.Array | pa.ChunkedArray,
    *,
    only_if_code: str,
) -> pa.Array | pa.ChunkedArray:
    """
    Normalize station-specific reported-band encodings to the "reported band" strings
    expected by `ObservatoryBandMap`.

    Currently this is only needed for LSST (X05), where MPC/ADES encodings can include
    'Lg'/'Lr'/... and 'LSST_g'/... and sometimes 'Y' for y-band.

    Notes
    -----
    - This must be applied *after* canonical `filter_id` pass-through checks.
    - This should be lightweight and Arrow-native (no Python loops).
    """
    mask = pc.equal(codes, pa.scalar(str(only_if_code), type=pa.large_string()))
    if not bool(pc.any(mask).as_py()):
        return bands

    b = pc.cast(bands, pa.large_string())
    b = pc.utf8_trim_whitespace(b)

    # Strip "LSST_" prefix: LSST_g -> g
    has_lsst_prefix = pc.match_substring_regex(b, "^LSST_")
    b_lsst = pc.utf8_slice_codeunits(b, 5)
    b = pc.if_else(has_lsst_prefix, b_lsst, b)

    # Strip leading 'L' for 2-character encodings: Lg -> g, Ly -> y, etc.
    is_Lx = pc.match_substring_regex(b, "^L[ugrizy]$")
    b_Lx = pc.utf8_slice_codeunits(b, 1)
    b = pc.if_else(is_Lx, b_Lx, b)

    # Accept uppercase variants too: LY -> y, etc.
    is_LX = pc.match_substring_regex(b, "^L[UGRIZY]$")
    b_LX = pc.utf8_lower(pc.utf8_slice_codeunits(b, 1))
    b = pc.if_else(is_LX, b_LX, b)

    # Normalize Y -> y (both as raw 'Y' and as 'LY' after stripping doesn't apply).
    b = pc.if_else(pc.equal(b, "Y"), "y", b)

    # Apply only for the targeted station code.
    return pc.if_else(mask, b, bands)


@lru_cache(maxsize=1)
def load_bandpass_curves() -> BandpassCurves:
    path = _DATA_DIR.joinpath(_BANDPASS_CURVES_FILE)
    return BandpassCurves.from_parquet(path)


@lru_cache(maxsize=1)
def load_observatory_band_map() -> ObservatoryBandMap:
    path = _DATA_DIR.joinpath(_OBS_BAND_MAP_FILE)
    return ObservatoryBandMap.from_parquet(path)


@lru_cache(maxsize=1)
def load_asteroid_templates() -> AsteroidTemplates:
    path = _DATA_DIR.joinpath(_TEMPLATES_FILE)
    return AsteroidTemplates.from_parquet(path)


@lru_cache(maxsize=1)
def load_template_integrals() -> TemplateBandpassIntegrals:
    path = _DATA_DIR.joinpath(_INTEGRALS_FILE)
    return TemplateBandpassIntegrals.from_parquet(path)


@lru_cache(maxsize=1)
def _load_solar_spectrum() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Load the adopted solar spectrum used for all precomputed integrals.

    Returns
    -------
    wavelength_nm, flux : ndarray
        Flux is arbitrary units suitable for relative photometry.
    """
    path = _DATA_DIR.joinpath(_SOLAR_SPECTRUM_FILE)
    table = pq.read_table(path)
    wl = np.asarray(
        table.column("wavelength_nm").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    flux = np.asarray(
        table.column("flux").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    return wl, flux


def _compute_filter_solar_norm_photon(
    filter_wavelength_nm: npt.NDArray[np.float64],
    filter_throughput: npt.NDArray[np.float64],
) -> float:
    """
    Compute the solar-weighted photon-counting normalization for a filter:

        D = ∫ F_sun(λ) * T(λ) * λ dλ

    This is used to convert our precomputed template×filter photon integrals
    (which are proportional to total photon counts) into a *band-averaged* quantity:

        <R>_filter = (∫ F_sun(λ) * R_ast(λ) * T(λ) * λ dλ) / D

    Using <R> makes V→filter color terms depend on the *shape* of the bandpass and
    the reflectance spectrum, rather than on absolute throughput scaling or filter width.
    """
    solar_wl, solar_flux = _load_solar_spectrum()

    wl = np.asarray(filter_wavelength_nm, dtype=np.float64)
    thr = np.asarray(filter_throughput, dtype=np.float64)
    if wl.ndim != 1 or thr.ndim != 1 or len(wl) != len(thr):
        raise ValueError("filter arrays must be 1D and have the same length")
    if len(wl) < 2:
        return float("nan")

    wl_min = max(float(solar_wl.min()), float(wl.min()))
    wl_max = min(float(solar_wl.max()), float(wl.max()))
    if wl_max <= wl_min:
        return float("nan")

    mask = (solar_wl >= wl_min) & (solar_wl <= wl_max)
    grid = solar_wl[mask]
    sun = solar_flux[mask]
    if len(grid) < 2:
        return float("nan")

    t = _interp_to_grid(wl, thr, grid)
    return float(np.trapezoid(sun * t * grid, grid))


@lru_cache(maxsize=1)
def _solar_norm_by_filter_id() -> dict[str, float]:
    """
    Cache solar-weighted throughput normalizations for all vendored filters.
    """
    curves = load_bandpass_curves()
    out: dict[str, float] = {}
    for fid, wl_list, thr_list in zip(
        curves.filter_id.to_pylist(),
        curves.wavelength_nm.to_pylist(),
        curves.throughput.to_pylist(),
    ):
        wl = np.asarray(wl_list, dtype=np.float64)
        thr = np.asarray(thr_list, dtype=np.float64)
        d = _compute_filter_solar_norm_photon(wl, thr)
        if not np.isfinite(d) or d <= 0.0:
            raise ValueError(f"Invalid solar normalization for filter_id '{fid}'")
        out[str(fid)] = float(d)
    return out


def _solar_norm_for_filter_ids(
    filter_ids: npt.NDArray[np.object_],
) -> npt.NDArray[np.float64]:
    norms = _solar_norm_by_filter_id()
    filt = np.asarray(filter_ids, dtype=object)
    out = np.empty(len(filt), dtype=np.float64)
    for i, fid in enumerate(filt.tolist()):
        d = norms.get(str(fid))
        if d is None:
            raise ValueError(f"Unknown filter_id for solar normalization: {fid}")
        out[i] = float(d)
    return out


def map_to_canonical_filter_bands(
    observatory_codes: (
        pa.Array | pa.ChunkedArray | npt.NDArray[np.object_] | Iterable[str]
    ),
    bands: pa.Array | pa.ChunkedArray | npt.NDArray[np.object_] | Iterable[str],
    *,
    allow_fallback_filters: bool = True,
) -> npt.NDArray[np.object_]:
    """
    Suggest canonical (vendored) bandpass filter IDs for a set of observations.

    This function is intended to be called by users *before* passing filters into
    bandpass-driven magnitude APIs. It applies the following resolution strategy:

    - If the provided band is already a canonical vendored `filter_id` (i.e., we have a
      curve for it), it is passed through unchanged.
    - Else, if (observatory_code, band) is present in `ObservatoryBandMap`, that mapping
      is used.
    - Else, apply a conservative fallback for generic bands:
        u/g/r/i/z -> SDSS_u/g/r/i/z
        y         -> PS1_y

    Parameters
    ----------
    observatory_codes : array-like
        MPC observatory codes.
    bands : array-like
        Reported band labels OR canonical vendored filter IDs.
    allow_fallback_filters : bool, optional
        If True, allow generic-band fallbacks when no (observatory_code, band) mapping is
        available:
          u/g/r/i/z -> SDSS_u/g/r/i/z
          y         -> PS1_y
        If False, raise if any row would require those fallbacks. Canonical `filter_id`
        inputs are always passed through. Defaults to True.

    Returns
    -------
    ndarray
        Canonical vendored `filter_id` strings.
    """
    codes = _to_string_array(observatory_codes)
    b = _to_string_array(bands)
    if len(codes) != len(b):
        raise ValueError(
            f"observatory_codes length ({len(codes)}) must match bands length ({len(b)})"
        )

    curves = load_bandpass_curves()
    mapping = load_observatory_band_map()

    # Pass-through if already a known filter_id.
    b_idx = pc.fill_null(pc.index_in(b, value_set=curves.filter_id), -1)
    is_known = np.asarray(b_idx.to_numpy(zero_copy_only=False), dtype=np.int32) >= 0

    out = np.empty(len(b), dtype=object)
    b_np = np.asarray(b.to_numpy(zero_copy_only=False), dtype=object)
    out[is_known] = b_np[is_known]

    # Try mapping table for the rest.
    need_map = ~is_known
    used_fallback = np.zeros(len(b), dtype=bool)
    if np.any(need_map):
        need_map_arr = pa.array(need_map.tolist(), type=pa.bool_())
        b_need = pc.if_else(need_map_arr, b, b)
        b_for_map = _normalize_reported_band_for_station(codes, b_need, only_if_code="X05")

        keys = _key_from_code_band(codes, b_for_map)
        idx = pc.fill_null(pc.index_in(keys, value_set=mapping.key), -1)
        idx_np = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)

        mapped = need_map & (idx_np >= 0)
        if np.any(mapped):
            mapped_vals = mapping.filter_id.take(
                pa.array(idx_np[mapped], type=pa.int32())
            )
            out[mapped] = np.asarray(
                mapped_vals.to_numpy(zero_copy_only=False), dtype=object
            )

        missing = need_map & (idx_np < 0)
        if np.any(missing):
            missing_idx = np.nonzero(missing)[0]
            missing_bands = b_np[missing].astype(str)
            codes_np = np.asarray(
                codes.to_numpy(zero_copy_only=False), dtype=object
            ).astype(str)
            missing_codes = codes_np[missing].astype(str)

            fallback_map = {
                "u": "SDSS_u",
                "g": "SDSS_g",
                "r": "SDSS_r",
                "i": "SDSS_i",
                "z": "SDSS_z",
                "y": "PS1_y",
            }

            for j, code, band in zip(
                missing_idx.tolist(), missing_codes.tolist(), missing_bands.tolist()
            ):
                band_l = band.lower()
                fb = fallback_map.get(band_l)
                if fb is None:
                    continue
                out[j] = fb
                used_fallback[j] = True

    # Validate outputs are all present and correspond to known vendored curves.
    missing_out = np.nonzero(out == None)[0]  # noqa: E711
    if len(missing_out) > 0:
        codes_np = np.asarray(
            codes.to_numpy(zero_copy_only=False), dtype=object
        ).astype(str)
        unknown = [f"{codes_np[i]}|{str(b_np[i])}" for i in missing_out.tolist()]
        raise ValueError(
            "Unable to suggest canonical filter_id(s) for: " + ", ".join(unknown)
        )

    if (not allow_fallback_filters) and np.any(used_fallback):
        codes_np = np.asarray(
            codes.to_numpy(zero_copy_only=False), dtype=object
        ).astype(str)
        fallback_pairs = [
            f"{codes_np[i]}|{str(b_np[i])}"
            for i in np.nonzero(used_fallback)[0].tolist()
        ]
        raise ValueError(
            "No non-fallback mapping found for: "
            + ", ".join(fallback_pairs)
            + ". Set allow_fallback_filters=True to allow SDSS/PS1 fallbacks."
        )

    # Final guarantee: every output has a curve.
    out_arr = pa.array(out.tolist(), type=pa.large_string())
    out_idx = pc.fill_null(pc.index_in(out_arr, value_set=curves.filter_id), -1)
    out_idx_np = np.asarray(out_idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(out_idx_np < 0):
        bad = np.unique(
            np.asarray(out, dtype=object)[out_idx_np < 0].astype(str)
        ).tolist()
        raise ValueError(
            "Suggested filter_id(s) do not have vendored curves: " + repr(bad)
        )

    return np.asarray(out, dtype=object)


def assert_filter_ids_have_curves(
    filter_ids: pa.Array | pa.ChunkedArray | npt.NDArray[np.object_] | Iterable[str],
) -> None:
    """
    Raise if any `filter_id` is not present in vendored `BandpassCurves`.
    """
    curves = load_bandpass_curves()
    arr = _to_string_array(filter_ids)
    idx = pc.fill_null(pc.index_in(arr, value_set=curves.filter_id), -1)
    idx_np = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(idx_np < 0):
        raw = np.asarray(arr.to_numpy(zero_copy_only=False), dtype=object)
        missing = np.unique(raw[idx_np < 0].astype(str)).tolist()
        raise ValueError(
            "Unknown filter_id(s) (no vendored bandpass curve): "
            + repr(missing)
            + ". Run map_to_canonical_filter_bands() first to map observatory bands to canonical filter_ids."
        )


def _get_integrals_precomputed(
    template_id: str, filter_ids: npt.NDArray[np.object_]
) -> npt.NDArray[np.float64]:
    """
    Return solar-weighted mean reflectance values for a known template_id across filter_ids.
    """
    filt = np.asarray(filter_ids, dtype=object)
    if filt.ndim != 1:
        raise ValueError("filter_ids must be a 1D array")
    if len(filt) == 0:
        return np.asarray([], dtype=np.float64)

    integrals = load_template_integrals()
    key = pa.array(
        [f"{template_id}|{str(x)}" for x in filt.tolist()], type=pa.large_string()
    )
    sep = pa.scalar("|", type=pa.large_string())
    value_set = pc.binary_join_element_wise(
        integrals.template_id, integrals.filter_id, sep
    )
    idx = pc.index_in(key, value_set=value_set)
    idx = pc.fill_null(idx, -1)
    idx_np = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(idx_np < 0):
        missing = np.unique(
            np.asarray(key.to_numpy(zero_copy_only=False), dtype=object)[idx_np < 0]
        )
        raise ValueError(
            f"Missing precomputed integrals for template '{template_id}' and filters: {missing.tolist()}"
        )
    vals = integrals.integral_photon.take(pa.array(idx_np, type=pa.int32()))
    numer = np.asarray(vals.to_numpy(zero_copy_only=False), dtype=np.float64)
    denom = _solar_norm_for_filter_ids(filt)
    return numer / denom


def get_integrals(
    template_id: str, filter_ids: npt.NDArray[np.object_]
) -> npt.NDArray[np.float64]:
    """
    Return solar-weighted mean reflectance values for `template_id` across `filter_ids`.

    Supports both vendored templates (precomputed) and custom templates registered
    at runtime via `register_custom_template`.
    """
    tid = str(template_id)
    tmpl = _CUSTOM_TEMPLATES.get(tid)
    if tmpl is None:
        return _get_integrals_precomputed(tid, filter_ids)

    curves = load_bandpass_curves()
    filt = np.asarray(filter_ids, dtype=object)
    if filt.ndim != 1:
        raise ValueError("filter_ids must be a 1D array")

    denom = _solar_norm_for_filter_ids(filt)
    out = np.empty(len(filt), dtype=np.float64)
    for i, fid in enumerate(filt.tolist()):
        key = (tid, str(fid))
        cached = _CUSTOM_INTEGRALS.get(key)
        if cached is not None:
            out[i] = float(cached) / float(denom[i])
            continue
        mask = pc.equal(curves.filter_id, str(fid))
        sel = curves.apply_mask(mask)
        if len(sel) != 1:
            raise ValueError(
                f"Unknown filter_id '{fid}' for custom template integral computation"
            )
        wl = np.asarray(sel.wavelength_nm[0].as_py(), dtype=np.float64)
        thr = np.asarray(sel.throughput[0].as_py(), dtype=np.float64)
        val = _compute_integral_photon(wl, thr, tmpl.wavelength_nm, tmpl.reflectance)
        _CUSTOM_INTEGRALS[key] = val
        out[i] = float(val) / float(denom[i])
    return out


def compute_mix_integrals(
    weight_C: float,
    weight_S: float,
    filter_ids: npt.NDArray[np.object_],
) -> npt.NDArray[np.float64]:
    """
    Compute integrals for a C/S linear mix using precomputed base integrals.

    This avoids recomputing any convolution and supports arbitrary weights.
    """
    w_c = float(weight_C)
    w_s = float(weight_S)
    if not np.isfinite(w_c) or not np.isfinite(w_s):
        raise ValueError("weights must be finite")
    if w_c < 0.0 or w_s < 0.0:
        raise ValueError("weights must be non-negative")
    s = w_c + w_s
    if s <= 0.0:
        raise ValueError("at least one weight must be > 0")
    w_c /= s
    w_s /= s

    filt = np.asarray(filter_ids, dtype=object)
    ints_c = get_integrals("C", filt)
    ints_s = get_integrals("S", filt)
    return w_c * ints_c + w_s * ints_s


def bandpass_delta_mag(
    composition: str | tuple[float, float],
    source_filter_id: str,
    target_filter_id: str,
) -> float:
    """
    Compute a constant magnitude offset between two canonical filters for a composition.

    The delta is defined as:

        Δm = m_target - m_source = -2.5 log10(<R>_target / <R>_source)

    where <R> is the solar-weighted band-averaged reflectance for the composition
    (see `get_integrals`).
    """
    src = str(source_filter_id)
    tgt = str(target_filter_id)
    if not src or not tgt:
        raise ValueError("source_filter_id and target_filter_id must be non-empty")
    if src == tgt:
        return 0.0

    ids = np.asarray([src, tgt], dtype=object)
    if isinstance(composition, str):
        integrals = get_integrals(str(composition), ids)
    else:
        w_c, w_s = composition
        integrals = compute_mix_integrals(float(w_c), float(w_s), ids)

    i_src = float(integrals[0])
    i_tgt = float(integrals[1])
    if not np.isfinite(i_src) or not np.isfinite(i_tgt) or i_src <= 0.0 or i_tgt <= 0.0:
        raise ValueError(f"Invalid integrals for delta magnitude {src} -> {tgt}")
    return float(-2.5 * np.log10(i_tgt / i_src))


def bandpass_color_terms(
    composition: str | tuple[float, float],
    *,
    source_filter_id: str = "V",
    target_filter_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    """
    Return delta magnitudes relative to `source_filter_id` for a set of canonical filters.

    Returns
    -------
    dict
        Mapping: target_filter_id -> Δm where Δm = m_target - m_source.
    """
    src = str(source_filter_id)
    if not src:
        raise ValueError("source_filter_id must be non-empty")

    if target_filter_ids is None:
        targets = [str(x) for x in load_bandpass_curves().filter_id.to_pylist()]
    else:
        targets = [str(x) for x in target_filter_ids]

    targets = [t for t in targets if t and t != src]
    if not targets:
        return {}

    ids = np.asarray([src] + targets, dtype=object)
    if isinstance(composition, str):
        integrals = get_integrals(str(composition), ids)
    else:
        w_c, w_s = composition
        integrals = compute_mix_integrals(float(w_c), float(w_s), ids)

    i_src = float(integrals[0])
    if not np.isfinite(i_src) or i_src <= 0.0:
        raise ValueError(f"Invalid integral for source filter '{src}'")

    deltas = -2.5 * np.log10(np.asarray(integrals[1:], dtype=np.float64) / i_src)
    return {t: float(d) for t, d in zip(targets, deltas)}


@dataclass(frozen=True)
class _CustomTemplate:
    wavelength_nm: npt.NDArray[np.float64]
    reflectance: npt.NDArray[np.float64]


_CUSTOM_TEMPLATES: dict[str, _CustomTemplate] = {}
_CUSTOM_INTEGRALS: dict[tuple[str, str], float] = {}


def register_custom_template(
    template_id: str,
    wavelength_nm: npt.NDArray[np.float64],
    reflectance: npt.NDArray[np.float64],
) -> None:
    """
    Register a custom reflectance template in-process and compute missing integrals lazily.

    Notes
    -----
    - This does not modify vendored Parquet data.
    - This is intended for NumPy paths. JAX-compiled tables in `magnitude.py`
      cannot be extended at runtime without recompilation.
    """
    tid = str(template_id)
    if not tid:
        raise ValueError("template_id must be non-empty")

    wl = np.asarray(wavelength_nm, dtype=np.float64)
    rf = np.asarray(reflectance, dtype=np.float64)
    if wl.ndim != 1 or rf.ndim != 1:
        raise ValueError("wavelength_nm and reflectance must be 1D arrays")
    if len(wl) != len(rf):
        raise ValueError("wavelength_nm and reflectance must have the same length")
    if len(wl) < 2:
        raise ValueError("template arrays must have at least 2 points")

    order = np.argsort(wl)
    wl = wl[order]
    rf = rf[order]
    if np.any(~np.isfinite(wl)) or np.any(~np.isfinite(rf)):
        raise ValueError("template arrays must be finite")
    if np.any(np.diff(wl) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")

    _CUSTOM_TEMPLATES[tid] = _CustomTemplate(wavelength_nm=wl, reflectance=rf)
    _clear_custom_cache()


def _interp_to_grid(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x_new: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.interp(x_new, x, y, left=0.0, right=0.0).astype(np.float64, copy=False)


def _compute_integral_photon(
    filter_wavelength_nm: npt.NDArray[np.float64],
    filter_throughput: npt.NDArray[np.float64],
    template_wavelength_nm: npt.NDArray[np.float64],
    template_reflectance: npt.NDArray[np.float64],
) -> float:
    solar_wl, solar_flux = _load_solar_spectrum()

    wl_min = max(
        float(solar_wl.min()),
        float(filter_wavelength_nm.min()),
        float(template_wavelength_nm.min()),
    )
    wl_max = min(
        float(solar_wl.max()),
        float(filter_wavelength_nm.max()),
        float(template_wavelength_nm.max()),
    )
    if wl_max <= wl_min:
        return float("nan")

    # Use the solar spectrum sampling as the shared grid (already dense and stable).
    mask = (solar_wl >= wl_min) & (solar_wl <= wl_max)
    wl = solar_wl[mask]
    sun = solar_flux[mask]
    if len(wl) < 2:
        return float("nan")

    t = _interp_to_grid(filter_wavelength_nm, filter_throughput, wl)
    r = _interp_to_grid(template_wavelength_nm, template_reflectance, wl)
    integrand = sun * r * t * wl
    return float(np.trapezoid(integrand, wl))


def _clear_custom_cache() -> None:
    _CUSTOM_INTEGRALS.clear()
