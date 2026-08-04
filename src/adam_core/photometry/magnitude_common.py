from __future__ import annotations

from functools import lru_cache
from typing import TypeAlias, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from .bandpasses.api import assert_filter_ids_have_curves  # noqa: F401
from .bandpasses.api import compute_mix_integrals as _compute_bandpass_mix_integrals
from .bandpasses.api import get_integrals as _get_bandpass_integrals
from .bandpasses.api import load_bandpass_curves as _load_bandpass_curves

JAX_CHUNK_SIZE = 8192

BandpassComposition: TypeAlias = Union[str, tuple[float, float]]

# IAU two-parameter (H, G) phase-function coefficients (Bowell et al. 1989).  Defined
# once here so the NumPy and JAX implementations below cannot drift apart.
_HG_PHI1_SCALE, _HG_PHI1_EXP = 3.33, 0.63
_HG_PHI2_SCALE, _HG_PHI2_EXP = 1.87, 1.22
_HG_PHASE_FLOOR = 1.0e-12


def hg_phase_correction(
    alpha_deg: npt.NDArray[np.float64] | float,
    G: float,
) -> npt.NDArray[np.float64]:
    """H-G phase correction in magnitudes for solar phase angle ``alpha_deg`` and slope ``G``.

    Returns ``-2.5 * log10[(1 - G) * phi1 + G * phi2]`` -- the term added to
    ``H + 5 * log10(r_au * delta_au)`` to get the reduced/apparent V magnitude, where
    ``phi_i = exp(-A_i * tan(alpha/2) ** B_i)``.  Zero at opposition (alpha = 0) and
    positive (fainter) for larger phase angles.  NumPy implementation for CPU callers.
    """
    alpha_rad = np.radians(np.asarray(alpha_deg, dtype=np.float64))
    tan_half = np.tan(0.5 * alpha_rad)
    phi1 = np.exp(-_HG_PHI1_SCALE * np.power(tan_half, _HG_PHI1_EXP))
    phi2 = np.exp(-_HG_PHI2_SCALE * np.power(tan_half, _HG_PHI2_EXP))
    phase = np.clip((1.0 - G) * phi1 + G * phi2, _HG_PHASE_FLOOR, None)
    return -2.5 * np.log10(phase)


def _hg_phase_correction_from_cos_jax(
    cos_phase: jnp.ndarray,
    G: jnp.ndarray,
) -> jnp.ndarray:
    """H-G phase correction in magnitudes, from ``cos`` of the phase angle (JAX kernels).

    Same coefficients as :func:`hg_phase_correction`, but takes ``cos_phase`` directly
    (avoiding ``arccos``) via the identity ``tan(alpha/2) = sqrt((1 - cos) / (1 + cos))``.
    """
    tan_half = jnp.sqrt((1.0 - cos_phase) / (1.0 + cos_phase))
    phi1 = jnp.exp(-_HG_PHI1_SCALE * tan_half**_HG_PHI1_EXP)
    phi2 = jnp.exp(-_HG_PHI2_SCALE * tan_half**_HG_PHI2_EXP)
    phase_function = (1.0 - G) * phi1 + G * phi2
    return -2.5 * jnp.log10(phase_function)


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
    reading packaged Parquet data.
    """
    curves = _load_bandpass_curves()
    filter_ids = tuple(curves.filter_id.to_pylist())
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
    """
    filter_ids, _, _, v_id = bandpass_filter_id_table()
    ids = np.asarray(filter_ids, dtype=object)
    integrals = bandpass_integrals_for_composition(composition_key, ids)

    i_v = float(integrals[v_id])
    if not np.isfinite(i_v) or i_v <= 0.0:
        raise ValueError("Invalid V-band integral for bandpass conversion.")

    with np.errstate(divide="raise", invalid="raise"):
        delta = -2.5 * np.log10(np.asarray(integrals, dtype=np.float64) / i_v)
    return np.asarray(delta, dtype=np.float64)


@lru_cache(maxsize=None)
def bandpass_delta_table_jax_for_composition_cached(
    composition_key: BandpassComposition,
) -> jax.Array:
    delta = bandpass_delta_table_for_composition_cached(composition_key)
    return jnp.asarray(delta, dtype=jnp.float64)


def bandpass_delta_table_for_composition(
    composition: BandpassComposition,
) -> npt.NDArray[np.float64]:
    return bandpass_delta_table_for_composition_cached(
        bandpass_composition_key(composition)
    )
