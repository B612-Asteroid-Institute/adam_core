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
    from adam_core import _rust_native as _rn

    if isinstance(composition, str):
        template_id, mix = _rn.bandpasses_composition_key(composition, None)
        assert mix is None
        return str(template_id)
    try:
        w_c, w_s = composition
    except Exception as e:
        raise TypeError(
            "composition must be either a template_id string (e.g. 'C') "
            "or a (weight_C, weight_S) tuple"
        ) from e
    template_id, mix = _rn.bandpasses_composition_key(None, (float(w_c), float(w_s)))
    assert template_id is None
    return (float(mix[0]), float(mix[1]))


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
