from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from jax import jit

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import OriginCodes
from ..observations.exposures import Exposures
from ..observers.observers import Observers
from ..utils.chunking import process_in_chunks
from .magnitude_common import (
    JAX_CHUNK_SIZE,
    BandpassComposition,
    assert_filter_ids_have_curves,
)
from .magnitude_common import bandpass_composition_key as _bandpass_composition_key
from .magnitude_common import (
    bandpass_delta_table_for_composition as _bandpass_delta_table_for_composition,
)
from .magnitude_common import (
    bandpass_delta_table_for_composition_cached as _bandpass_delta_table_for_composition_cached,
)
from .magnitude_common import (
    bandpass_delta_table_jax_for_composition_cached as _bandpass_delta_table_jax_for_composition_cached,
)
from .magnitude_common import bandpass_filter_id_table as _bandpass_filter_id_table


def _validate_hg_geometry(
    *,
    object_pos: npt.NDArray[np.float64],
    observer_pos: npt.NDArray[np.float64],
) -> None:
    """
    Validate geometry inputs for the H-G apparent magnitude model.

    We treat non-finite or non-positive distances as invalid inputs and raise, since
    they represent physically impossible (or mis-framed) states for solar system photometry.
    """
    obj = np.asarray(object_pos, dtype=np.float64)
    obs = np.asarray(observer_pos, dtype=np.float64)
    if obj.ndim != 2 or obj.shape[1] != 3:
        raise ValueError("object_pos must have shape (N, 3)")
    if obs.shape != obj.shape:
        raise ValueError("observer_pos must have shape (N, 3) and match object_pos")

    r = np.sqrt(np.sum(obj * obj, axis=1))
    delta_vec = obj - obs
    delta = np.sqrt(np.sum(delta_vec * delta_vec, axis=1))

    invalid = (~np.isfinite(r)) | (~np.isfinite(delta)) | (r <= 0.0) | (delta <= 0.0)
    if np.any(invalid):
        n_bad = int(np.count_nonzero(invalid))
        raise ValueError(
            "Invalid photometry geometry for H-G model: "
            f"{n_bad} rows have non-finite or non-positive distances (r<=0 or delta<=0)."
        )


def calculate_phase_angle(
    object_coords: CartesianCoordinates,
    observers: Observers,
) -> npt.NDArray[np.float64]:
    """
    Calculate the solar phase angle (Sun–object–observer) in degrees.

    "Phase angle" here is the angle at the object between the Sun direction and the
    observer direction. It is commonly used for simple photometry/visibility metrics.

    Notes
    -----
    This helper expects **heliocentric** coordinates (origin = `OriginCodes.SUN`) for both
    the object and the observer. If you have barycentric coordinates, transform first.

    Parameters
    ----------
    object_coords
        Object Cartesian coordinates in AU (origin must be SUN).
    observers
        Observer states (origin must be SUN).

    Returns
    -------
    phase_angle_deg
        Phase angle in degrees for each paired row.

    Examples
    --------
    Given an `Ephemeris` `eph` from a propagator and corresponding `Observers` `obs`:

    - Use `eph.coordinates` for on-sky (RA/Dec, rho) values.
    - Use `eph.aberrated_coordinates` for emission-time geometry, and transform to heliocentric:

    ```python
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.origin import OriginCodes
    from adam_core.coordinates.transform import transform_coordinates
    from adam_core.photometry import calculate_phase_angle
    from adam_core.observers import Observers

    observers_eph = Observers.from_codes(eph.coordinates.origin.code, eph.coordinates.time)

    obj_helio = transform_coordinates(
        eph.aberrated_coordinates,
        CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )
    obs_helio = observers_eph.set_column(
        "coordinates",
        transform_coordinates(
            observers_eph.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        ),
    )
    alpha_deg = calculate_phase_angle(obj_helio, obs_helio)
    ```
    """
    n_obj = len(object_coords)
    n_obs = len(observers)
    if n_obs != n_obj:
        raise ValueError(
            f"observers length ({n_obs}) must match object_coords length ({n_obj})"
        )

    if not np.all(object_coords.origin == OriginCodes.SUN):
        raise ValueError(
            "object_coords must be heliocentric (origin=SUN). "
            "Use transform_coordinates(..., origin_out=OriginCodes.SUN) first."
        )
    if not np.all(observers.coordinates.origin == OriginCodes.SUN):
        raise ValueError(
            "observers.coordinates must be heliocentric (origin=SUN). "
            "Use transform_coordinates(..., origin_out=OriginCodes.SUN) first."
        )
    if object_coords.frame != observers.coordinates.frame:
        raise ValueError(
            "object_coords and observers must be expressed in the same frame "
            f"(got {object_coords.frame!r} vs {observers.coordinates.frame!r})."
        )

    object_pos = np.asarray(object_coords.r, dtype=np.float64)
    observer_pos = np.asarray(observers.coordinates.r, dtype=np.float64)
    _validate_hg_geometry(object_pos=object_pos, observer_pos=observer_pos)

    # -------------------------------------------------------------------------
    # JAX compute: padded/chunked to a fixed shape to avoid recompiles.
    # -------------------------------------------------------------------------
    chunk_size = JAX_CHUNK_SIZE
    padded_n = int(((n_obj + chunk_size - 1) // chunk_size) * chunk_size)
    out = np.empty((padded_n,), dtype=np.float64)

    chunks: list[jax.Array] = []
    for obj_chunk, obs_chunk in zip(
        process_in_chunks(object_pos, chunk_size),
        process_in_chunks(observer_pos, chunk_size),
    ):
        chunks.append(_calculate_phase_angle_core_jax(obj_chunk, obs_chunk))

    host_chunks = jax.device_get(chunks)
    offset = 0
    for alpha_chunk in host_chunks:
        out[offset : offset + chunk_size] = alpha_chunk
        offset += chunk_size

    return out[:n_obj]


def convert_magnitude(
    magnitude: npt.NDArray[np.float64],
    source_filter_id: npt.NDArray[np.object_],
    target_filter_id: npt.NDArray[np.object_],
    *,
    composition: BandpassComposition,
) -> npt.NDArray[np.float64]:
    """
    Convert magnitudes between *canonical* bandpass filter IDs using template integrals.

    Parameters
    ----------
    magnitude : ndarray
        1D array of magnitudes in `source_filter_id`.
    source_filter_id : ndarray
        1D array of canonical source filter IDs (e.g., 'V', 'DECam_g', 'LSST_r').
    target_filter_id : ndarray
        1D array of canonical target filter IDs.
    composition : str or (float, float)
        Required. Either a template ID ('C', 'S', 'NEO', 'MBA', or a registered custom
        template), or a (weight_C, weight_S) tuple for a linear C/S mix.

    Returns
    -------
    ndarray
        Magnitudes in `target_filter_id`.
    """
    mags = np.asarray(magnitude, dtype=np.float64)
    if mags.ndim != 1:
        raise ValueError("magnitude must be a 1D ndarray")

    src = np.asarray(source_filter_id, dtype=object)
    tgt = np.asarray(target_filter_id, dtype=object)
    if src.ndim != 1 or tgt.ndim != 1:
        raise ValueError("source_filter_id and target_filter_id must be 1D ndarrays")
    if len(src) != len(mags) or len(tgt) != len(mags):
        raise ValueError(
            "source_filter_id/target_filter_id must match magnitude length"
        )

    # Contract: these are canonical vendored filter IDs (call find_suggested_filter_bands first).
    assert_filter_ids_have_curves(src)
    assert_filter_ids_have_curves(tgt)

    filter_ids, filter_ids_arrow, _, _ = _bandpass_filter_id_table()
    delta_table = _bandpass_delta_table_for_composition(composition)
    if int(delta_table.shape[0]) != len(filter_ids):
        raise ValueError("Bandpass delta table length mismatch.")

    # Fast Arrow mapping: filter_id strings -> integer IDs.
    src_arr = pa.array(src, type=pa.large_string())
    tgt_arr = pa.array(tgt, type=pa.large_string())
    src_ids_arr = pc.fill_null(pc.index_in(src_arr, value_set=filter_ids_arrow), -1)
    tgt_ids_arr = pc.fill_null(pc.index_in(tgt_arr, value_set=filter_ids_arrow), -1)
    src_ids = np.asarray(src_ids_arr.to_numpy(zero_copy_only=False), dtype=np.int32)
    tgt_ids = np.asarray(tgt_ids_arr.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(src_ids < 0) or np.any(tgt_ids < 0):
        missing_src = np.unique(
            np.asarray(src, dtype=object)[src_ids < 0].astype(str)
        ).tolist()
        missing_tgt = np.unique(
            np.asarray(tgt, dtype=object)[tgt_ids < 0].astype(str)
        ).tolist()
        raise ValueError(
            f"Unknown canonical filter_ids for bandpass conversion. "
            f"missing source={missing_src}, missing target={missing_tgt}. "
            "Run find_suggested_filter_bands() first to map observatory bands to canonical filter_ids."
        )

    delta_src = delta_table[src_ids]
    delta_tgt = delta_table[tgt_ids]
    return mags + (delta_tgt - delta_src)


@jit
def _calculate_apparent_magnitude_core_jax(
    H_v: jnp.ndarray,
    object_pos: jnp.ndarray,
    observer_pos: jnp.ndarray,
    G: jnp.ndarray,
) -> jnp.ndarray:
    """
    JAX core computation for apparent magnitude in V-band.

    Notes
    -----
    This function is intentionally "array-only" (no ADAM classes) to keep it
    JIT-friendly. Use `calculate_apparent_magnitude_v` for the public API.
    """
    # Heliocentric distance r (AU)
    # (manual norm is typically a bit leaner than jnp.linalg.norm for small fixed dims)
    r = jnp.sqrt(jnp.sum(object_pos * object_pos, axis=1))

    # Observer-to-object distance delta (AU)
    delta_vec = object_pos - observer_pos
    delta = jnp.sqrt(jnp.sum(delta_vec * delta_vec, axis=1))

    # Phase angle
    observer_sun_dist = jnp.sqrt(jnp.sum(observer_pos * observer_pos, axis=1))
    numer = r**2 + delta**2 - observer_sun_dist**2
    denom = 2.0 * r * delta
    cos_phase = jnp.clip(numer / denom, -1.0, 1.0)
    # H-G phase function
    #
    # Best practice (perf): avoid arccos() + tan() since we only need tan(phase/2).
    # Use identity: tan(phase/2) = sqrt((1 - cos_phase) / (1 + cos_phase)).
    tan_half = jnp.sqrt((1.0 - cos_phase) / (1.0 + cos_phase))
    phi1 = jnp.exp(-3.33 * tan_half**0.63)
    phi2 = jnp.exp(-1.87 * tan_half**1.22)
    phase_function = (1.0 - G) * phi1 + G * phi2

    return H_v + 5.0 * jnp.log10(r * delta) - 2.5 * jnp.log10(phase_function)


@jit
def _calculate_phase_angle_core_jax(
    object_pos: jnp.ndarray,
    observer_pos: jnp.ndarray,
) -> jnp.ndarray:
    """
    JAX core computation for phase angle in degrees.

    Notes
    -----
    This is intentionally array-only and expects paired rows (N, 3).
    """
    r = jnp.sqrt(jnp.sum(object_pos * object_pos, axis=1))
    delta_vec = object_pos - observer_pos
    delta = jnp.sqrt(jnp.sum(delta_vec * delta_vec, axis=1))
    observer_sun_dist = jnp.sqrt(jnp.sum(observer_pos * observer_pos, axis=1))

    numer = r * r + delta * delta - observer_sun_dist * observer_sun_dist
    denom = 2.0 * r * delta
    cos_alpha = jnp.clip(numer / denom, -1.0, 1.0)

    # Stable conversion from cos(alpha) -> alpha without arccos().
    y = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_alpha))
    x = jnp.sqrt(jnp.maximum(0.0, 1.0 + cos_alpha))
    alpha_rad = 2.0 * jnp.arctan2(y, x)
    return alpha_rad * (180.0 / jnp.pi)


@jit
def _calculate_apparent_magnitude_and_phase_core_jax(
    H_v: jnp.ndarray,
    object_pos: jnp.ndarray,
    observer_pos: jnp.ndarray,
    G: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX core computation for (apparent V magnitude, phase angle in degrees).
    """
    r = jnp.sqrt(jnp.sum(object_pos * object_pos, axis=1))
    delta_vec = object_pos - observer_pos
    delta = jnp.sqrt(jnp.sum(delta_vec * delta_vec, axis=1))
    observer_sun_dist = jnp.sqrt(jnp.sum(observer_pos * observer_pos, axis=1))

    numer = r * r + delta * delta - observer_sun_dist * observer_sun_dist
    denom = 2.0 * r * delta
    cos_phase = jnp.clip(numer / denom, -1.0, 1.0)

    # Magnitude H-G phase function (uses tan(phase/2)).
    tan_half = jnp.sqrt((1.0 - cos_phase) / (1.0 + cos_phase))
    phi1 = jnp.exp(-3.33 * tan_half**0.63)
    phi2 = jnp.exp(-1.87 * tan_half**1.22)
    phase_function = (1.0 - G) * phi1 + G * phi2
    mags_v = H_v + 5.0 * jnp.log10(r * delta) - 2.5 * jnp.log10(phase_function)

    # Phase angle in degrees.
    y = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_phase))
    x = jnp.sqrt(jnp.maximum(0.0, 1.0 + cos_phase))
    alpha_rad = 2.0 * jnp.arctan2(y, x)
    alpha_deg = alpha_rad * (180.0 / jnp.pi)
    return mags_v, alpha_deg


@jit
def _predict_magnitudes_bandpass_core_jax(
    H_v: jnp.ndarray,
    object_pos: jnp.ndarray,
    observer_pos: jnp.ndarray,
    G: jnp.ndarray,
    target_ids: jnp.ndarray,
    delta_table: jnp.ndarray,
) -> jnp.ndarray:
    """
    JAX core computation for per-exposure magnitudes (V-band geometry + bandpass conversion).

    This fuses:
    1) apparent V magnitude calculation (H-G system)
    2) V -> target filter conversion via a per-filter delta magnitude table
    """
    mags_v = _calculate_apparent_magnitude_core_jax(
        H_v=H_v, object_pos=object_pos, observer_pos=observer_pos, G=G
    )
    delta = delta_table[target_ids]
    return mags_v + delta


def calculate_apparent_magnitude_v(
    H_v: Union[float, npt.NDArray[np.float64]],
    object_coords: CartesianCoordinates,
    observer: Observers,
    G: Union[float, npt.NDArray[np.float64]] = 0.15,
) -> npt.NDArray[np.float64]:
    """
    Calculate apparent V-band magnitudes.

    Notes
    -----
    This function is JAX-backed (numpy-sandwich pattern) and returns a NumPy array.
    """
    # -------------------------------------------------------------------------
    # Numpy sandwich input + validation
    # -------------------------------------------------------------------------
    n = len(object_coords)
    n_obs = len(observer)
    if n_obs != n:
        raise ValueError(
            f"observer length ({n_obs}) must match object_coords length ({n})"
        )

    object_pos = np.asarray(object_coords.r, dtype=np.float64)
    observer_pos = np.asarray(observer.coordinates.r, dtype=np.float64)
    _validate_hg_geometry(object_pos=object_pos, observer_pos=observer_pos)

    H_v_arr = np.asarray(H_v, dtype=np.float64)
    if H_v_arr.ndim == 0:
        H_v_arr = np.full(n, float(H_v_arr), dtype=np.float64)
    elif len(H_v_arr) != n:
        raise ValueError(
            f"H array length ({len(H_v_arr)}) must match object_coords length ({n})"
        )

    G_arr = np.asarray(G, dtype=np.float64)
    if G_arr.ndim == 0:
        G_arr = np.full(n, float(G_arr), dtype=np.float64)
    elif len(G_arr) != n:
        raise ValueError(
            f"G array length ({len(G_arr)}) must match H array length ({n})"
        )

    # -------------------------------------------------------------------------
    # JAX compute: padded/chunked to a fixed shape to avoid recompiles.
    # -------------------------------------------------------------------------
    chunk_size = JAX_CHUNK_SIZE
    padded_n = int(((n + chunk_size - 1) // chunk_size) * chunk_size)
    out = np.empty((padded_n,), dtype=np.float64)

    chunks: list[jax.Array] = []
    for H_chunk, obj_chunk, obs_chunk, G_chunk in zip(
        process_in_chunks(H_v_arr, chunk_size),
        process_in_chunks(object_pos, chunk_size),
        process_in_chunks(observer_pos, chunk_size),
        process_in_chunks(G_arr, chunk_size),
    ):
        chunks.append(
            _calculate_apparent_magnitude_core_jax(
                H_v=H_chunk,
                object_pos=obj_chunk,
                observer_pos=obs_chunk,
                G=G_chunk,
            )
        )

    host_chunks = jax.device_get(chunks)
    offset = 0
    for mags_v_chunk in host_chunks:
        out[offset : offset + chunk_size] = mags_v_chunk
        offset += chunk_size

    return out[:n]


def calculate_apparent_magnitude_v_and_phase_angle(
    H_v: Union[float, npt.NDArray[np.float64]],
    object_coords: CartesianCoordinates,
    observer: Observers,
    G: Union[float, npt.NDArray[np.float64]] = 0.15,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate apparent V-band magnitudes and phase angles (degrees) together.

    Why: when both are needed, the H-G model already computes the phase geometry.
    This combined function avoids redoing the same law-of-cosines computation twice.
    """
    n = len(object_coords)
    n_obs = len(observer)
    if n_obs != n:
        raise ValueError(
            f"observer length ({n_obs}) must match object_coords length ({n})"
        )

    object_pos = np.asarray(object_coords.r, dtype=np.float64)
    observer_pos = np.asarray(observer.coordinates.r, dtype=np.float64)
    _validate_hg_geometry(object_pos=object_pos, observer_pos=observer_pos)

    H_v_arr = np.asarray(H_v, dtype=np.float64)
    if H_v_arr.ndim == 0:
        H_v_arr = np.full(n, float(H_v_arr), dtype=np.float64)
    elif len(H_v_arr) != n:
        raise ValueError(
            f"H array length ({len(H_v_arr)}) must match object_coords length ({n})"
        )

    G_arr = np.asarray(G, dtype=np.float64)
    if G_arr.ndim == 0:
        G_arr = np.full(n, float(G_arr), dtype=np.float64)
    elif len(G_arr) != n:
        raise ValueError(
            f"G array length ({len(G_arr)}) must match H array length ({n})"
        )

    chunk_size = JAX_CHUNK_SIZE
    padded_n = int(((n + chunk_size - 1) // chunk_size) * chunk_size)
    out_mag = np.empty((padded_n,), dtype=np.float64)
    out_alpha = np.empty((padded_n,), dtype=np.float64)

    chunks: list[tuple[jax.Array, jax.Array]] = []
    for H_chunk, obj_chunk, obs_chunk, G_chunk in zip(
        process_in_chunks(H_v_arr, chunk_size),
        process_in_chunks(object_pos, chunk_size),
        process_in_chunks(observer_pos, chunk_size),
        process_in_chunks(G_arr, chunk_size),
    ):
        chunks.append(
            _calculate_apparent_magnitude_and_phase_core_jax(
                H_v=H_chunk,
                object_pos=obj_chunk,
                observer_pos=obs_chunk,
                G=G_chunk,
            )
        )

    host_chunks = jax.device_get(chunks)
    offset = 0
    for mags_v_chunk, alpha_chunk in host_chunks:
        out_mag[offset : offset + chunk_size] = mags_v_chunk
        out_alpha[offset : offset + chunk_size] = alpha_chunk
        offset += chunk_size

    return out_mag[:n], out_alpha[:n]

def predict_magnitudes(
    H: Union[float, npt.NDArray[np.float64]],
    object_coords: CartesianCoordinates,
    exposures: Exposures,
    G: Union[float, npt.NDArray[np.float64]] = 0.15,
    reference_filter: str = "V",
    *,
    composition: BandpassComposition,
) -> npt.NDArray[np.float64]:
    """
    Predict apparent magnitudes for objects observed during exposures using bandpass-based conversions.

    This:
    - compute apparent V-band magnitudes using the H-G system + geometry, then
    - convert V -> exposure filter.

    Notes
    -----
    - `exposures.filter` must contain canonical bandpass `filter_id` values (e.g. 'LSST_i', 'DECam_g').
    - The V -> target conversion is computed from precomputed template×filter integrals, and requires
      an explicit asteroid composition (template_id or C/S mix weights).

    Parameters
    ----------
    H : float or ndarray
        Absolute magnitude(s) of the object(s) in `reference_filter` (canonical bandpass filter ID).
    object_coords : CartesianCoordinates
        Cartesian coordinates of the object(s) at the exposure times.
    exposures : Exposures
        Exposure table. `exposures.filter` must be a canonical bandpass `filter_id`.
    G : float or ndarray, optional
        Slope parameter for the H-G system, defaults to 0.15.
    reference_filter : str, optional
        Canonical filter ID in which H is defined. Defaults to "V".
    composition : str or (float, float)
        Required. Either a template ID ('C', 'S', 'NEO', 'MBA', or a registered custom template),
        or a (weight_C, weight_S) tuple for a linear C/S mix.

    Returns
    -------
    ndarray
        Predicted apparent magnitudes in the exposures' filters.
    """
    if len(object_coords) != len(exposures):
        raise ValueError(
            f"object_coords length ({len(object_coords)}) must match exposures length ({len(exposures)})"
        )

    # Contract: exposures.filter and reference_filter are canonical vendored filter IDs
    # (call find_suggested_filter_bands first to map observatory bands).
    assert_filter_ids_have_curves(exposures.filter)
    assert_filter_ids_have_curves([reference_filter])

    observers = exposures.observers()

    n = len(object_coords)
    object_pos = np.asarray(object_coords.r, dtype=np.float64)
    observer_pos = np.asarray(observers.coordinates.r, dtype=np.float64)
    _validate_hg_geometry(object_pos=object_pos, observer_pos=observer_pos)

    H_arr = np.asarray(H, dtype=np.float64)
    if H_arr.ndim == 0:
        H_arr = np.full(n, float(H_arr), dtype=np.float64)
    elif len(H_arr) != n:
        raise ValueError(
            f"H array length ({len(H_arr)}) must match object_coords length ({n})"
        )

    G_arr = np.asarray(G, dtype=np.float64)
    if G_arr.ndim == 0:
        G_arr = np.full(n, float(G_arr), dtype=np.float64)
    elif len(G_arr) != n:
        raise ValueError(
            f"G array length ({len(G_arr)}) must match object_coords length ({n})"
        )

    # -------------------------------------------------------------------------
    # Bandpass conversion: build delta table and map exposures -> target IDs
    # -------------------------------------------------------------------------
    filter_ids, filter_ids_arrow, filter_to_id, v_id = _bandpass_filter_id_table()
    comp_key = _bandpass_composition_key(composition)
    delta_table = _bandpass_delta_table_for_composition_cached(comp_key)
    if int(delta_table.shape[0]) != len(filter_ids):
        raise ValueError("Bandpass delta table length mismatch.")

    # Convert H into V-band absolute magnitude for internal V-centric calculation.
    if reference_filter == "V":
        H_v_arr = H_arr
    else:
        ref_id = filter_to_id.get(str(reference_filter))
        if ref_id is None:
            raise ValueError(
                f"Unknown reference_filter for bandpass conversion: {reference_filter}"
            )
        H_v_arr = H_arr - float(delta_table[int(ref_id)])

    # Map canonical filter_id -> integer ID.
    tgt_idx = pc.index_in(exposures.filter, value_set=filter_ids_arrow)
    tgt_idx = pc.fill_null(tgt_idx, -1)
    target_ids = np.asarray(tgt_idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(target_ids < 0):
        target_raw = exposures.filter.to_numpy(zero_copy_only=False)
        missing = np.unique(
            np.asarray(target_raw, dtype=object)[target_ids < 0].astype(str)
        )
        raise ValueError(
            "Unknown canonical filter_ids for bandpass prediction: "
            + missing.tolist().__repr__()
        )

    # -------------------------------------------------------------------------
    # JAX compute: padded/chunked to a fixed shape to avoid recompiles.
    # -------------------------------------------------------------------------
    delta_table_jax = _bandpass_delta_table_jax_for_composition_cached(comp_key)
    chunk_size = JAX_CHUNK_SIZE
    padded_n = int(((n + chunk_size - 1) // chunk_size) * chunk_size)
    out = np.empty((padded_n,), dtype=np.float64)

    chunks: list[jax.Array] = []
    for H_chunk, obj_chunk, obs_chunk, G_chunk, tgt_chunk in zip(
        process_in_chunks(H_v_arr, chunk_size),
        process_in_chunks(object_pos, chunk_size),
        process_in_chunks(observer_pos, chunk_size),
        process_in_chunks(G_arr, chunk_size),
        process_in_chunks(target_ids, chunk_size),
    ):
        chunks.append(
            _predict_magnitudes_bandpass_core_jax(
                H_v=H_chunk,
                object_pos=obj_chunk,
                observer_pos=obs_chunk,
                G=G_chunk,
                target_ids=tgt_chunk,
                delta_table=delta_table_jax,
            )
        )

    host_chunks = jax.device_get(chunks)
    offset = 0
    for mags_out_chunk in host_chunks:
        out[offset : offset + chunk_size] = mags_out_chunk
        offset += chunk_size

    return out[:n]
