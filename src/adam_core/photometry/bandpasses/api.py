"""
Runtime APIs for the bandpass-driven photometry implementation.

Bandpass curves are vendored from the SVO Filter Profile Service. Please see
`REFERENCES.md` for the required acknowledgement and citations when using this
service.

All runtime compute (solar normalizations, photon-counting integrals,
template/mix integrals, canonical band mapping, and the custom-template
registry) runs in the Rust backend (bead personal-cmy.24); these functions are
thin bindings that pass the vendored data directory to Rust once and marshal
arguments. The quivr table loaders remain Python for table-level access.

Documented deviation: integral-derived values match legacy to <= 1e-12
relative (numpy's trapezoid uses pairwise summation; Rust sums sequentially),
gated by the frozen legacy fixture.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Final, Iterable

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

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


def _data_dir_str() -> str:
    return str(_DATA_DIR)


def _to_string_array(
    values: pa.Array | pa.ChunkedArray | npt.NDArray[np.object_] | Iterable[str],
) -> pa.Array | pa.ChunkedArray:
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return pc.cast(values, pa.large_string())
    arr = pa.array(list(values), type=pa.large_string())
    return arr


def _to_string_list(
    values: pa.Array | pa.ChunkedArray | npt.NDArray[np.object_] | Iterable[str],
) -> list[str]:
    return [str(value) for value in _to_string_array(values).to_pylist()]


def _load_table(cls, filename: str):
    from adam_core import _rust_native as _rn

    batches = _rn.bandpasses_load_table(_data_dir_str(), filename)
    return cls.from_pyarrow(pa.Table.from_batches(batches))


@lru_cache(maxsize=1)
def load_bandpass_curves() -> BandpassCurves:
    return _load_table(BandpassCurves, _BANDPASS_CURVES_FILE)


@lru_cache(maxsize=1)
def load_observatory_band_map() -> ObservatoryBandMap:
    return _load_table(ObservatoryBandMap, _OBS_BAND_MAP_FILE)


@lru_cache(maxsize=1)
def load_asteroid_templates() -> AsteroidTemplates:
    return _load_table(AsteroidTemplates, _TEMPLATES_FILE)


@lru_cache(maxsize=1)
def load_template_integrals() -> TemplateBandpassIntegrals:
    return _load_table(TemplateBandpassIntegrals, _INTEGRALS_FILE)


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
        available. If False, raise if any row would require those fallbacks. Canonical
        `filter_id` inputs are always passed through. Defaults to True.

    Returns
    -------
    ndarray
        Canonical vendored `filter_id` strings.
    """
    from adam_core import _rust_native as _rn

    out = _rn.bandpasses_map_to_canonical(
        _data_dir_str(),
        _to_string_list(observatory_codes),
        _to_string_list(bands),
        bool(allow_fallback_filters),
    )
    return np.asarray(out, dtype=object)


def assert_filter_ids_have_curves(
    filter_ids: pa.Array | pa.ChunkedArray | npt.NDArray[np.object_] | Iterable[str],
) -> None:
    """
    Raise if any `filter_id` is not present in vendored `BandpassCurves`.
    """
    from adam_core import _rust_native as _rn

    _rn.bandpasses_assert_filter_ids(_data_dir_str(), _to_string_list(filter_ids))


def get_integrals(
    template_id: str, filter_ids: npt.NDArray[np.object_]
) -> npt.NDArray[np.float64]:
    """
    Return solar-weighted mean reflectance values for `template_id` across `filter_ids`.

    Supports both vendored templates (precomputed) and custom templates registered
    at runtime via `register_custom_template`.
    """
    from adam_core import _rust_native as _rn

    filt = np.asarray(filter_ids, dtype=object)
    if filt.ndim != 1:
        raise ValueError("filter_ids must be a 1D array")
    out = _rn.bandpasses_get_integrals(
        _data_dir_str(), str(template_id), [str(x) for x in filt.tolist()]
    )
    return np.asarray(out, dtype=np.float64)


def compute_mix_integrals(
    weight_C: float,
    weight_S: float,
    filter_ids: npt.NDArray[np.object_],
) -> npt.NDArray[np.float64]:
    """
    Compute integrals for a C/S linear mix using precomputed base integrals.

    This avoids recomputing any convolution and supports arbitrary weights.
    """
    from adam_core import _rust_native as _rn

    filt = np.asarray(filter_ids, dtype=object)
    if filt.ndim != 1:
        raise ValueError("filter_ids must be a 1D array")
    out = _rn.bandpasses_compute_mix_integrals(
        _data_dir_str(),
        float(weight_C),
        float(weight_S),
        [str(x) for x in filt.tolist()],
    )
    return np.asarray(out, dtype=np.float64)


def _composition_args(
    composition: str | tuple[float, float],
) -> tuple[str | None, tuple[float, float] | None]:
    if isinstance(composition, str):
        return str(composition), None
    w_c, w_s = composition
    return None, (float(w_c), float(w_s))


def bandpass_delta_mag(
    composition: str | tuple[float, float],
    source_filter_id: str,
    target_filter_id: str,
) -> float:
    """
    Compute a constant magnitude offset between two canonical filters for a composition.

    The delta is defined as:

        \u0394m = m_target - m_source = -2.5 log10(<R>_target / <R>_source)

    where <R> is the solar-weighted band-averaged reflectance for the composition
    (see `get_integrals`).
    """
    from adam_core import _rust_native as _rn

    template_id, mix = _composition_args(composition)
    return float(
        _rn.bandpasses_delta_mag(
            _data_dir_str(),
            str(source_filter_id),
            str(target_filter_id),
            template_id=template_id,
            mix=mix,
        )
    )


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
        Mapping: target_filter_id -> \u0394m where \u0394m = m_target - m_source.
    """
    from adam_core import _rust_native as _rn

    template_id, mix = _composition_args(composition)
    targets = None if target_filter_ids is None else [str(x) for x in target_filter_ids]
    out = _rn.bandpasses_color_terms(
        _data_dir_str(),
        str(source_filter_id),
        template_id=template_id,
        mix=mix,
        target_filter_ids=targets,
    )
    return {target: float(delta) for target, delta in out}


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
    - The registry lives in the Rust backend and is process-wide, matching the
      legacy in-process dict registry.
    """
    from adam_core import _rust_native as _rn

    wl = np.asarray(wavelength_nm, dtype=np.float64)
    rf = np.asarray(reflectance, dtype=np.float64)
    if wl.ndim != 1 or rf.ndim != 1:
        raise ValueError("wavelength_nm and reflectance must be 1D arrays")
    _rn.bandpasses_register_custom_template(str(template_id), wl.tolist(), rf.tolist())
