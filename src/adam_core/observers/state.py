from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..constants import Constants as c
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..time import Timestamp
from ..utils.bounded_lru import bounded_lru_get, bounded_lru_put
from ..utils.spice import (
    _query_pxform_itrf93_batch,
    get_perturber_state,
    get_spice_body_state,
    setup_SPICE,
)
from ..utils.spice_backend import NotCovered, get_backend
from .observers import OBSERVATORY_CODES, OBSERVATORY_PARALLAX_COEFFICIENTS

R_EARTH_EQUATORIAL = c.R_EARTH_EQUATORIAL
OMEGA_EARTH = 2 * np.pi / 0.997269675925926
Z_AXIS = np.array([0, 0, 1])


@dataclass(frozen=True)
class _ObserverStateCacheKey:
    code: str
    frame: str
    origin: str
    n: int
    first_key: int
    last_key: int
    time_digest: int


_OBSERVER_STATE_CACHE_MAXSIZE = int(
    os.environ.get("ADAM_CORE_OBSERVER_STATE_CACHE_MAXSIZE", "256")
)
_OBSERVER_STATE_CACHE: "OrderedDict[_ObserverStateCacheKey, CartesianCoordinates]" = (
    OrderedDict()
)


def _observer_cache_get(key: _ObserverStateCacheKey) -> CartesianCoordinates | None:
    return bounded_lru_get(
        _OBSERVER_STATE_CACHE, key, maxsize=_OBSERVER_STATE_CACHE_MAXSIZE
    )


def _observer_cache_put(
    key: _ObserverStateCacheKey, coords: CartesianCoordinates
) -> None:
    bounded_lru_put(
        _OBSERVER_STATE_CACHE, key, coords, maxsize=_OBSERVER_STATE_CACHE_MAXSIZE
    )


def clear_observer_state_cache() -> None:
    """
    Clear the in-process observer-state cache (primarily for tests/benchmarks).
    """
    _OBSERVER_STATE_CACHE.clear()


def get_mpc_observer_state(
    code: str,
    times: Timestamp,
    frame: Literal["ecliptic", "equatorial", "itrf93"] = "ecliptic",
    origin: OriginCodes = OriginCodes.SUN,
) -> CartesianCoordinates:
    """
    Get the state of an MPC observatory on Earth.

    Combines Earth SPICE state and rotation with MPC observatory parallax coefficients.

    Parameters
    ----------
    code : str
        The MPC observatory code of the observer.
    times : Timestamp
        The times at which to get the observer state.
    frame : {'ecliptic', 'equatorial', 'itrf93'}
        The frame in which to return the observer state.
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

    # Observatory codes use the geodedics from the MPC observatory code table
    # Get observatory geodetic information
    parallax_coeffs = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", code)
    if len(parallax_coeffs) == 0:
        raise ValueError(
            f"Observatory code '{code}' not found in the observatory parallax coefficients table."
        )
    parallax_coeffs = parallax_coeffs.table.to_pylist()[0]

    # Unpack geodetic information
    longitude = parallax_coeffs["longitude"]
    cos_phi = parallax_coeffs["cos_phi"]
    sin_phi = parallax_coeffs["sin_phi"]

    if np.any(np.isnan([longitude, cos_phi, sin_phi])):
        err = (
            f"{code} is missing information on Earth-based geodetic coordinates. The MPC Obs Code\n"
            "file may be missing this information or the observer is a space-based observatory.\n"
            "Space observatories are currently not supported.\n"
        )
        raise ValueError(err)

    # Calculate pointing vector from geocenter to observatory
    sin_longitude = np.sin(np.radians(longitude))
    cos_longitude = np.cos(np.radians(longitude))
    o_hat_ITRF93 = np.array([cos_longitude * cos_phi, sin_longitude * cos_phi, sin_phi])

    # Multiply pointing vector with Earth radius to get actual vector
    o_vec_ITRF93 = np.dot(R_EARTH_EQUATORIAL, o_hat_ITRF93)

    n, first, last, _ = times.signature(scale="tdb")
    cache_key = _ObserverStateCacheKey(
        code=str(code),
        frame=str(frame),
        origin=str(origin.name),
        n=int(n),
        first_key=int(first),
        last_key=int(last),
        time_digest=int(times.cache_digest(scale="tdb")),
    )
    cached = _observer_cache_get(cache_key)
    if cached is not None:
        return cached

    # If ITRF93 frame is requested, we can directly use the ITRF93 values
    if frame == "itrf93":
        # For ITRF93, which is Earth-fixed, position is constant but velocity comes from Earth's rotation
        N = len(times)
        r_obs = np.tile(o_vec_ITRF93, (N, 1))

        # Calculate velocity due to Earth's rotation
        rotation_direction = np.cross(o_hat_ITRF93, Z_AXIS)
        v_obs = np.tile(-OMEGA_EARTH * R_EARTH_EQUATORIAL * rotation_direction, (N, 1))

        # For ITRF93, we still need Earth's state relative to the requested origin
        if origin != OriginCodes.EARTH:
            earth_state = get_perturber_state(
                OriginCodes.EARTH, times, frame="itrf93", origin=origin
            )
            r_obs += earth_state.r
            v_obs += earth_state.v

        observer_states = CartesianCoordinates.from_kwargs(
            time=times,
            x=r_obs[:, 0],
            y=r_obs[:, 1],
            z=r_obs[:, 2],
            vx=v_obs[:, 0],
            vy=v_obs[:, 1],
            vz=v_obs[:, 2],
            frame=frame,
            origin=Origin.from_kwargs(
                code=pa.array(
                    pa.repeat(origin.name, len(times)), type=pa.large_string()
                )
            ),
        )

        _observer_cache_put(cache_key, observer_states)
        return observer_states

    # For other frames, continue with existing implementation
    # Grab Earth state vector (this function handles duplicate times)
    state = get_perturber_state(OriginCodes.EARTH, times, frame=frame, origin=origin)

    # If the code is 500 (geocenter), we can just return the Earth state vector
    if code == "500":
        return state

    # If not then we need to add a topocentric correction.
    # Warning! Converting times to ET will incur a loss of precision.
    epochs_et_np = times.et().to_numpy(zero_copy_only=False).astype(np.float64)
    unique_ets, inv = np.unique(epochs_et_np, return_inverse=True)

    rotation_direction = np.cross(o_hat_ITRF93, Z_AXIS)
    v_offset_ITRF93 = -OMEGA_EARTH * R_EARTH_EQUATORIAL * rotation_direction

    # ITRF93 high-accuracy Earth rotation: precession (IAU-1976), nutation
    # (IAU-1980 with IERS corrections), true sidereal time, polar motion.
    unique_rot = _query_pxform_itrf93_batch(frame_spice, unique_ets)

    rot = unique_rot[inv]  # shape (N, 3, 3)
    # rot @ vec broadcasts to (N, 3) with BLAS-compatible FP ordering, so it
    # matches a per-epoch `M @ v` loop bit-for-bit.
    r_offsets = rot @ o_vec_ITRF93
    v_offsets = rot @ v_offset_ITRF93
    r_obs = state.r + r_offsets
    v_obs = state.v + v_offsets

    observer_states = CartesianCoordinates.from_kwargs(
        time=times,
        x=r_obs[:, 0],
        y=r_obs[:, 1],
        z=r_obs[:, 2],
        vx=v_obs[:, 0],
        vy=v_obs[:, 1],
        vz=v_obs[:, 2],
        frame=frame,
        origin=Origin.from_kwargs(
            code=pa.array(pa.repeat(origin.name, len(times)), type=pa.large_string())
        ),
    )

    _observer_cache_put(cache_key, observer_states)
    return observer_states


def get_observer_state(
    code: Union[str, OriginCodes],
    times: Timestamp,
    frame: Literal["ecliptic", "equatorial", "itrf93"] = "ecliptic",
    origin: OriginCodes = OriginCodes.SUN,
) -> CartesianCoordinates:
    """
    Find state vectors for an observer at a given time in the given frame and measured from the given origin.

    For NAIF Origin Codes this function uses the SPICE kernels to find the state vectors.
    See `~adam_core.utils.spice.get_perturber_state` for more information.

    For MPC observatory codes this function currently only supports ground-based observers on Earth.
    In this case, the Earth body-fixed frame used for calculations is the standard ITRF93,
    which takes into account:

      - precession (IAU-1976)
      - nutation (IAU-1980 with IERS corrections)
      - polar motion

    This frame is retrieved through SPICE.

    Parameters
    ----------
    code : Union[str, OriginCodes]
        MPC observatory code as string, custom SPICE kernel NAIF code as string,
        or NAIF origin code for which to find the states.
    times : Timestamp (N)
        Epochs for which to find the observatory locations.
    frame : {'equatorial', 'ecliptic'}
        Return observer state in the equatorial or ecliptic J2000 frames.
    origin : OriginCodes
        The NAIF ID of the origin.

    Returns
    -------
    observer_states : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        The state vectors of the observer in the desired frame
        and measured from the desired origin.
    """
    assert isinstance(code, (str, OriginCodes)), "code must be a string or OriginCodes"

    # Make sure SPICE is ready to go
    setup_SPICE()

    # If the code is an OriginCode, we can just use the get_perturber_state function
    if isinstance(code, OriginCodes):
        return get_perturber_state(code, times, frame=frame, origin=origin)

    # Prioritize MPC observatory codes over custom SPICE kernel NAIF codes
    if code in OBSERVATORY_CODES:
        return get_mpc_observer_state(code, times, frame=frame, origin=origin)

    # Try to retrieve the body ID from SPICE, could be a custom SPICE kernel NAIF code
    try:
        body_id = get_backend().bodn2c(code)
    except (NotCovered, RuntimeError):
        err = f"{code} is not a valid MPC observatory code and was not found in SPICE kernels."
        raise ValueError(err)
    return get_spice_body_state(
        body_id=body_id, times=times, frame=frame, origin=origin
    )
