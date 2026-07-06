from __future__ import annotations

from functools import lru_cache
from typing import TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from .bandpasses.api import _composition_args, _data_dir_str
from .bandpasses.api import assert_filter_ids_have_curves  # noqa: F401
from .bandpasses.api import compute_mix_integrals as _compute_bandpass_mix_integrals
from .bandpasses.api import get_integrals as _get_bandpass_integrals

BandpassComposition: TypeAlias = Union[str, tuple[float, float]]


@lru_cache(maxsize=1)
def bandpass_filter_id_table() -> tuple[
    tuple[str, ...],
    pa.Array,
    dict[str, int],
    int,
]:
    """
    Return (filter_ids, filter_ids_arrow, filter_to_id, v_id) for bandpass conversions.

    We intentionally build this lazily (rather than at import time) since it requires
    reading packaged Parquet data (Rust-side since bead personal-cmy.24).
    """
    from adam_core import _rust_native as _rn

    filter_ids = tuple(_rn.bandpasses_filter_ids(_data_dir_str()))
    if "V" not in filter_ids:
        raise ValueError("Bandpass curves must include a canonical 'V' filter_id.")

    filter_ids_arrow = pa.array(list(filter_ids), type=pa.large_string())
    filter_to_id = {name: i for i, name in enumerate(filter_ids)}
    v_id = int(filter_to_id["V"])
    return filter_ids, filter_ids_arrow, filter_to_id, v_id


def bandpass_integrals_for_composition(
    composition: BandpassComposition, filter_ids: npt.NDArray[np.object_]
) -> npt.NDArray[np.float64]:
    if isinstance(composition, str):
        return _get_bandpass_integrals(composition, filter_ids)
    try:
        w_c, w_s = composition
    except Exception as e:
        raise TypeError(
            "composition must be either a template_id string (e.g. 'C') "
            "or a (weight_C, weight_S) tuple"
        ) from e
    return _compute_bandpass_mix_integrals(float(w_c), float(w_s), filter_ids)


def bandpass_composition_key(composition: BandpassComposition) -> BandpassComposition:
    if isinstance(composition, str):
        if not composition:
            raise ValueError("composition template_id must be non-empty")
        return composition
    try:
        w_c, w_s = composition
    except Exception as e:
        raise TypeError(
            "composition must be either a template_id string (e.g. 'C') "
            "or a (weight_C, weight_S) tuple"
        ) from e
    w_c = float(w_c)
    w_s = float(w_s)
    if not np.isfinite(w_c) or not np.isfinite(w_s):
        raise ValueError("composition weights must be finite")
    if w_c < 0.0 or w_s < 0.0:
        raise ValueError("composition weights must be non-negative")
    s = w_c + w_s
    if s <= 0.0:
        raise ValueError("at least one composition weight must be > 0")
    return (w_c / s, w_s / s)


@lru_cache(maxsize=None)
def bandpass_delta_table_for_composition_cached(
    composition_key: BandpassComposition,
) -> npt.NDArray[np.float64]:
    """
    Compute per-filter delta magnitudes relative to V for the given composition:

        delta[filter] = m_filter - m_V

    Computed in the Rust backend (bead personal-cmy.24).
    """
    from adam_core import _rust_native as _rn

    template_id, mix = _composition_args(composition_key)
    delta = _rn.bandpasses_delta_table(
        _data_dir_str(), template_id=template_id, mix=mix
    )
    return np.asarray(delta, dtype=np.float64)


def bandpass_delta_table_for_composition(
    composition: BandpassComposition,
) -> npt.NDArray[np.float64]:
    return bandpass_delta_table_for_composition_cached(
        bandpass_composition_key(composition)
    )
