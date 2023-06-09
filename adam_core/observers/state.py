from typing import Literal

import numpy as np
import spiceypy as sp
from astropy.time import Time

from ..constants import Constants as c
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.times import Times
from ..utils.spice import get_perturber_state, setup_SPICE
from .observers import OBSERVATORY_CODES, OBSERVATORY_GEODETICS

R_EARTH = c.R_EARTH
OMEGA_EARTH = 2 * np.pi / 0.997269675925926
Z_AXIS = np.array([0, 0, 1])


def get_observer_state(
    code: str,
    times: Time,
    frame: Literal["ecliptic", "equatorial"] = "ecliptic",
    origin: OriginCodes = OriginCodes.SUN,
) -> CartesianCoordinates:
    """
    Find state vectors for an observer at a given time in the given frame and measured from the given origin.

    Currently only supports ground-based observers.

    The Earth body-fixed frame used for calculations is the standard ITRF93, which takes into account:
        - precession (IAU-1976)
        - nutation (IAU-1980 with IERS corrections)
        - polar motion
    This frame is retrieved through SPICE.

    Parameters
    ----------
    code : str
        MPC observatory code for which to find the states.
    observation_times : `~astropy.time.core.Time` (N)
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
    if code not in OBSERVATORY_CODES:
        err = f"{code} is not a valid MPC observatory code."
        raise ValueError(err)

    if frame == "ecliptic":
        frame_spice = "ECLIPJ2000"
    elif frame == "equatorial":
        frame_spice = "J2000"
    else:
        err = "frame should be one of {'equatorial', 'ecliptic'}"
        raise ValueError(err)

    # Make sure SPICE is ready to go
    setup_SPICE()

    # Get observatory geodetic information
    geodetics = OBSERVATORY_GEODETICS.select("code", code).values[0]

    if np.any(np.isnan(geodetics)):
        err = (
            f"{code} is missing information on Earth-based geodetic coordinates. The MPC Obs Code\n"
            "file may be missing this information or the observer is a space-based observatory.\n"
            "Space observatories are currently not supported.\n"
        )
        raise ValueError(err)

    # Grab Earth state vector (this function handles duplicate times)
    state = get_perturber_state(OriginCodes.EARTH, times, frame=frame, origin=origin)

    # If the code is 500 (geocenter), we can just return the Earth state vector
    if code == "500":
        return state

    # If not then we need to add a topocentric correction
    else:
        # Get observer location on Earth
        longitude = geodetics[0]
        cos_phi = geodetics[1]
        sin_phi = geodetics[2]
        sin_longitude = np.sin(np.radians(longitude))
        cos_longitude = np.cos(np.radians(longitude))

        # Calculate pointing vector from geocenter to observatory
        o_hat_ITRF93 = np.array(
            [cos_longitude * cos_phi, sin_longitude * cos_phi, sin_phi]
        )

        # Multiply pointing vector with Earth radius to get actual vector
        o_vec_ITRF93 = np.dot(R_EARTH, o_hat_ITRF93)

        # Convert MJD epochs in TDB to ET in TDB
        epochs_tdb = times.tdb.jd
        unique_epochs_tdb = np.unique(epochs_tdb)
        unique_epochs_et = np.array(
            [sp.str2et("JD {:.16f} TDB".format(i)) for i in unique_epochs_tdb]
        )

        N = len(epochs_tdb)
        r_obs = np.empty((N, 3), dtype=np.float64)
        v_obs = np.empty((N, 3), dtype=np.float64)
        r_geo = state.r
        v_geo = state.v
        for i, epoch in enumerate(unique_epochs_et):
            # Grab rotaton matrices from ITRF93 to ecliptic J2000
            # The ITRF93 high accuracy Earth rotation model takes into account:
            # Precession:  1976 IAU model from Lieske.
            # Nutation:  1980 IAU model, with IERS corrections due to Herring et al.
            # True sidereal time using accurate values of TAI-UT1
            # Polar motion
            rotation_matrix = sp.pxform("ITRF93", frame_spice, epoch)

            # Find indices of epochs that match the current unique epoch
            mask = np.where(epochs_tdb == unique_epochs_tdb[i])[0]

            # Add o_vec + r_geo to get r_obs (thank you numpy broadcasting)
            r_obs[mask] = r_geo[mask] + rotation_matrix @ o_vec_ITRF93

            # Calculate the velocity (thank you numpy broadcasting)
            rotation_direction = np.cross(o_hat_ITRF93, Z_AXIS)
            v_obs[mask] = v_geo[mask] + rotation_matrix @ (
                -OMEGA_EARTH * R_EARTH * rotation_direction
            )

    return CartesianCoordinates.from_kwargs(
        times=Times.from_astropy(times),
        x=r_obs[:, 0],
        y=r_obs[:, 1],
        z=r_obs[:, 2],
        vx=v_obs[:, 0],
        vy=v_obs[:, 1],
        vz=v_obs[:, 2],
        frame=frame,
        origin=Origin.from_kwargs(code=[origin.name for i in range(len(times))]),
    )
