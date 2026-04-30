import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Literal, Optional, Set

import numpy as np
import pyarrow.compute as pc
from naif_de440 import de440
from naif_earth_itrf93 import earth_itrf93
from naif_eop_high_prec import eop_high_prec
from naif_eop_historical import eop_historical
from naif_eop_predict import eop_predict
from naif_leapseconds import leapseconds

from ..constants import KM_P_AU, S_P_DAY
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..time import Timestamp
from .bounded_lru import bounded_lru_get, bounded_lru_put
from .spice_backend import get_backend

DEFAULT_KERNELS = [
    leapseconds,
    de440,
    eop_predict,
    eop_historical,
    eop_high_prec,
    earth_itrf93,
]

# Global state for tracking custom kernels
_REGISTERED_KERNELS: Set[str] = set()

J2000_TDB_JD = 2451545.0


@dataclass(frozen=True)
class _SpkezCacheKey:
    target: int
    observer: int
    frame: str
    days: int
    nanos: int


_SPKEZ_CACHE_MAXSIZE = int(os.environ.get("ADAM_CORE_SPKEZ_CACHE_MAXSIZE", "200000"))
_SPKEZ_CACHE: "OrderedDict[_SpkezCacheKey, np.ndarray]" = OrderedDict()


# Backward-compatible shims: existing callers and tests monkeypatch these
# names directly, so keep them as thin delegates onto the active backend
# rather than inlining get_backend() at every call site.


def _query_pxform_itrf93_batch(frame_spice: str, ets: np.ndarray) -> np.ndarray:
    """Batched 3×3 rotation ITRF93 → inertial. Returns shape `(N, 3, 3)`."""
    ets = np.ascontiguousarray(ets, dtype=np.float64)
    return get_backend().pxform_batch("ITRF93", frame_spice, ets)


def _query_sxform_itrf93_batch(
    frame_from: str, frame_to: str, ets: np.ndarray
) -> np.ndarray:
    """Batched 6×6 state transform across ITRF93 ↔ inertial. Shape `(N, 6, 6)`."""
    ets = np.ascontiguousarray(ets, dtype=np.float64)
    return get_backend().sxform_batch(frame_from, frame_to, ets)


def _query_states_km_kms_batch(
    reader, target: int, center: int, frame_spice: str, ets: np.ndarray
) -> np.ndarray:
    """Batched (N, 6) state query in km / km-s.

    The ``reader`` positional argument is retained for backward
    compatibility but ignored — the active backend owns its own readers
    and routes Rust-first, CSPICE-fallback under the covers.
    """
    ets = np.ascontiguousarray(ets, dtype=np.float64)
    return get_backend().spkez_batch(int(target), int(center), frame_spice, ets)


def _spkez_cache_get(key: _SpkezCacheKey) -> np.ndarray | None:
    return bounded_lru_get(_SPKEZ_CACHE, key, maxsize=_SPKEZ_CACHE_MAXSIZE)


def _spkez_cache_put(key: _SpkezCacheKey, state_au_aud: np.ndarray) -> None:
    bounded_lru_put(_SPKEZ_CACHE, key, state_au_aud, maxsize=_SPKEZ_CACHE_MAXSIZE)


def clear_spkez_cache() -> None:
    """
    Clear the in-process SPICE state cache.

    This is primarily intended for testing and benchmarking.
    """
    _SPKEZ_CACHE.clear()


def _jd_tdb_to_et(jd_tdb: np.ndarray) -> np.ndarray:
    """
    Convert TDB-scaled JD times to an ephemeris time (ET) in seconds.

    Parameters
    ----------
    jd_tdb : `~numpy.ndarray` (N)
        Times in JD TDB.

    Returns
    -------
    et : `~numpy.ndarray` (N)
        Times in ET in seconds.
    """
    # Convert to days since J2000 (noon on January 1, 2000)
    days_since_j2000 = jd_tdb - J2000_TDB_JD

    # Convert to seconds since J2000
    # (SPICE format)
    et = days_since_j2000 * S_P_DAY
    return et


def setup_SPICE(kernels: Optional[List[str]] = None, force: bool = False):
    """
    Load SPICE kernels.

    This function checks to see if SPICE has already been initialized for the current process.
    If it has, then it does nothing. If it has not, then it loads the desired kernels into SPICE.
    If force is set to True, then the kernels will be loaded regardless of whether or not SPICE
    has already been initialized. SPICE has a limit on the number of kernels that can be loaded
    at once, so it is recommended to only load the kernels that are needed for the current
    calculation (calling sp.furnsh multiple times will load the same kernel multiple times, which
    will cause an error.)

    The default kernels loaded are those provided by the NAIF data packages:

    - Leapsecond data (`naif-leapseconds <https://pypi.org/project/naif-leapseconds/>`_)
    - DE440 ephemeris data (`naif-de440 <https://pypi.org/project/naif-de440/>`_)
    - Longterm Earth Orientation Parameter Predictions (`naif-eop-predict <https://pypi.org/project/naif-eop-predict/>`_)
    - Historical Earth Orientation Parameters (`naif-eop-historical <https://pypi.org/project/naif-eop-historical/>`_)
    - High Precision Earth Orientation Parameters (`naif-eop-high-prec <https://pypi.org/project/naif-eop-high-prec/>`_)
    - Earth Body-fixed Reference Frame/Body Association (`naif-earth-itrf93 <https://pypi.org/project/naif-earth-itrf93/>`_)

    Parameters
    ----------
    kernels :
        List of SPICE kernels to load into SPICE. If None, then the default kernels will be loaded.

    """
    if kernels is None:
        kernels = DEFAULT_KERNELS

    process_id = os.getpid()
    env_var = f"ADAM_CORE_SPICE_INITIALIZED_{process_id}"
    if env_var in os.environ and not force:
        return

    for kernel in kernels:
        register_spice_kernel(kernel)
    os.environ[env_var] = "True"
    return


def get_perturber_state(
    perturber: OriginCodes,
    times: Timestamp,
    frame: Literal["ecliptic", "equatorial", "itrf93"] = "ecliptic",
    origin: OriginCodes = OriginCodes.SUN,
) -> CartesianCoordinates:
    """
    Query the JPL ephemeris files loaded in SPICE for the state vectors of desired perturbers.

    Parameters
    ----------
    perturber : OriginCodes
        The NAIF ID of the perturber.
    times : Timestamp (N)
        Times at which to get state vectors.
    frame : {'equatorial', 'ecliptic', 'itrf93'}
        Return perturber state in the equatorial or ecliptic J2000 frames.
    origin :  OriginCodes
        The NAIF ID of the origin.

    Returns
    -------
    states : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        The state vectors of the perturber in the desired frame
        and measured from the desired origin.
    """
    if frame == "ecliptic":
        frame_spice = "ECLIPJ2000"
    elif frame == "equatorial":
        frame_spice = "J2000"
    elif frame == "itrf93":
        frame_spice = "ITRF93"
    else:
        err = "frame should be one of {'equatorial', 'ecliptic', 'itrf93'}"
        raise ValueError(err)

    # Make sure SPICE is ready to roll
    setup_SPICE()

    N = int(len(times))
    if N == 0:
        return CartesianCoordinates.empty()

    # Build stable time keys in TDB using integer (days,nanos).
    times_tdb = times.rescale("tdb")
    days = times_tdb.days.to_numpy(zero_copy_only=False).astype(np.int64)
    nanos = times_tdb.nanos.to_numpy(zero_copy_only=False).astype(np.int64)
    time_key = times_tdb.key(scale=None)

    uniq_keys, rep_idx, inv = np.unique(
        time_key, return_index=True, return_inverse=True
    )

    epochs_et = times_tdb.et().to_numpy(zero_copy_only=False).astype(np.float64)
    uniq_states = np.empty((uniq_keys.shape[0], 6), dtype=np.float64)

    # First pass: cache probe for every unique epoch; collect misses for a
    # single batched query.
    cache_hit = np.zeros(uniq_keys.shape[0], dtype=bool)
    miss_idx: list[int] = []
    for i_u in range(int(uniq_keys.shape[0])):
        i0 = int(rep_idx[i_u])
        key = _SpkezCacheKey(
            target=int(perturber.value),
            observer=int(origin.value),
            frame=str(frame_spice),
            days=int(days[i0]),
            nanos=int(nanos[i0]),
        )
        cached = _spkez_cache_get(key)
        if cached is not None:
            uniq_states[i_u, :] = cached
            cache_hit[i_u] = True
            continue

        rev_key = _SpkezCacheKey(
            target=int(origin.value),
            observer=int(perturber.value),
            frame=str(frame_spice),
            days=int(days[i0]),
            nanos=int(nanos[i0]),
        )
        cached_rev = _spkez_cache_get(rev_key)
        if cached_rev is not None:
            s = -cached_rev
            uniq_states[i_u, :] = s
            _spkez_cache_put(key, s)
            cache_hit[i_u] = True
            continue

        miss_idx.append(i_u)

    if miss_idx:
        miss_i0 = rep_idx[np.asarray(miss_idx, dtype=np.int64)]
        miss_ets = epochs_et[miss_i0]
        batched = _query_states_km_kms_batch(
            None,
            int(perturber.value),
            int(origin.value),
            frame_spice,
            miss_ets,
        )
        scale = np.array([KM_P_AU, KM_P_AU, KM_P_AU, KM_P_AU / S_P_DAY,
                          KM_P_AU / S_P_DAY, KM_P_AU / S_P_DAY], dtype=np.float64)
        miss_states = batched / scale
        for local_i, i_u in enumerate(miss_idx):
            i0 = int(rep_idx[i_u])
            s = miss_states[local_i]
            uniq_states[i_u, :] = s
            key = _SpkezCacheKey(
                target=int(perturber.value),
                observer=int(origin.value),
                frame=str(frame_spice),
                days=int(days[i0]),
                nanos=int(nanos[i0]),
            )
            rev_key = _SpkezCacheKey(
                target=int(origin.value),
                observer=int(perturber.value),
                frame=str(frame_spice),
                days=int(days[i0]),
                nanos=int(nanos[i0]),
            )
            _spkez_cache_put(key, s)
            _spkez_cache_put(rev_key, -s)

    states = uniq_states[inv]

    return CartesianCoordinates.from_kwargs(
        time=times,
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        vx=states[:, 3],
        vy=states[:, 4],
        vz=states[:, 5],
        frame=frame,
        origin=Origin.from_kwargs(code=[origin.name] * N),
    )


def list_registered_kernels() -> Set[str]:
    """
    Get the set of currently registered custom SPICE kernels.

    Returns
    -------
    kernels : set[str]
        Set of kernel file paths that are currently registered
    """
    return _REGISTERED_KERNELS.copy()


def register_spice_kernel(kernel_path: str) -> None:
    """
    Register and load a custom SPICE kernel.

    Parameters
    ----------
    kernel_path : str
        Path to the SPICE kernel file
    """
    if kernel_path not in _REGISTERED_KERNELS:
        get_backend().furnsh(kernel_path)
        _REGISTERED_KERNELS.add(kernel_path)


def unregister_spice_kernel(kernel_path: str) -> None:
    """
    Unregister and unload a custom SPICE kernel.

    Parameters
    ----------
    kernel_path : str
        Path to the SPICE kernel file
    """
    if kernel_path in _REGISTERED_KERNELS:
        get_backend().unload(kernel_path)
        _REGISTERED_KERNELS.remove(kernel_path)


def get_spice_body_state(
    body_id: int,
    times: Timestamp,
    frame: Literal["ecliptic", "equatorial", "itrf93"] = "ecliptic",
    origin: OriginCodes = OriginCodes.SUN,
) -> CartesianCoordinates:
    """
    Get state vectors for a body using its SPICE ID.

    Parameters
    ----------
    body_id : int
        The SPICE ID of the body
    times : Timestamp
        Times at which to get state vectors
    frame : {'equatorial', 'ecliptic', 'itrf93'}
        Reference frame for returned state vectors
    origin : OriginCodes
        The origin for the state vectors

    Returns
    -------
    states : CartesianCoordinates
        The state vectors in the desired frame

    Raises
    ------
    ValueError
        If the body ID is not found in any loaded kernel or if state data
        cannot be retrieved for the requested times
    """
    if frame == "ecliptic":
        frame_spice = "ECLIPJ2000"
    elif frame == "equatorial":
        frame_spice = "J2000"
    elif frame == "itrf93":
        frame_spice = "ITRF93"
    else:
        raise ValueError("frame should be one of {'equatorial', 'ecliptic', 'itrf93'}")

    # Make sure SPICE is ready
    setup_SPICE()

    # Convert epochs to ET in TDB
    epochs_et = times.et()
    unique_epochs_et = epochs_et.unique()
    N = len(times)
    states = np.empty((N, 6), dtype=np.float64)

    # One batched query per unique epoch set — the active backend routes
    # Rust-first for J2000/ECLIPJ2000 and falls back to CSPICE for ITRF93
    # or any body the Rust DE440 reader does not cover.
    unique_ets_np = unique_epochs_et.to_numpy(zero_copy_only=False).astype(np.float64)
    try:
        unique_states_km = _query_states_km_kms_batch(
            None, int(body_id), int(origin.value), frame_spice, unique_ets_np
        )
    except Exception as e:
        raise ValueError(
            f"Could not get state data for body ID {body_id}: {str(e)}"
        )

    for i, epoch in enumerate(unique_epochs_et):
        mask = pc.equal(epochs_et, epoch).to_numpy(False)
        states[mask, :] = unique_states_km[i]

    # Convert units (vectorized operations)
    states = states / KM_P_AU
    states[:, 3:] *= S_P_DAY

    return CartesianCoordinates.from_kwargs(
        time=times,
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        vx=states[:, 3],
        vy=states[:, 4],
        vz=states[:, 5],
        frame=frame,
        origin=Origin.from_kwargs(code=[origin.name] * N),
    )
