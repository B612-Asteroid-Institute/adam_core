from typing import Literal, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import spiceypy as sp

from ..constants import Constants as c
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..time import Timestamp
from ..utils.spice import get_perturber_state, get_spice_body_state, setup_SPICE
from .observers import OBSERVATORY_CODES, OBSERVATORY_PARALLAX_COEFFICIENTS

R_EARTH_EQUATORIAL = c.R_EARTH_EQUATORIAL
OMEGA_EARTH = 2 * np.pi / 0.997269675925926
Z_AXIS = np.array([0, 0, 1])


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

        return observer_states

    # For other frames, continue with existing implementation
    # Grab Earth state vector (this function handles duplicate times)
    state = get_perturber_state(OriginCodes.EARTH, times, frame=frame, origin=origin)

    # If the code is 500 (geocenter), we can just return the Earth state vector
    if code == "500":
        return state

    # If not then we need to add a topocentric correction
    # Warning! Converting times to ET will incur a loss of precision.
    epochs_et = times.et()
    unique_epochs_et_tdb = epochs_et.unique()

    N = len(epochs_et)
    r_obs = np.empty((N, 3), dtype=np.float64)
    v_obs = np.empty((N, 3), dtype=np.float64)
    r_geo = state.r
    v_geo = state.v
    for epoch in unique_epochs_et_tdb:
        # Grab rotation matrices from ITRF93 to ecliptic J2000
        # The ITRF93 high accuracy Earth rotation model takes into account:
        # Precession:  1976 IAU model from Lieske.
        # Nutation:  1980 IAU model, with IERS corrections due to Herring et al.
        # True sidereal time using accurate values of TAI-UT1
        # Polar motion
        rotation_matrix = sp.pxform("ITRF93", frame_spice, epoch.as_py())

        # Find indices of epochs that match the current unique epoch
        mask = pc.equal(epochs_et, epoch).to_numpy(False)

        # Add o_vec + r_geo to get r_obs (thank you numpy broadcasting)
        r_obs[mask] = r_geo[mask] + rotation_matrix @ o_vec_ITRF93

        # Calculate the velocity (thank you numpy broadcasting)
        rotation_direction = np.cross(o_hat_ITRF93, Z_AXIS)
        v_obs[mask] = v_geo[mask] + rotation_matrix @ (
            -OMEGA_EARTH * R_EARTH_EQUATORIAL * rotation_direction
        )

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
        body_id = sp.bodn2c(code)
    except sp.SpiceyError:
        err = f"{code} is not a valid MPC observatory code and was not found in SPICE kernels."
        raise ValueError(err)
    else:
        return get_spice_body_state(
            body_id=body_id, times=times, frame=frame, origin=origin
        )
