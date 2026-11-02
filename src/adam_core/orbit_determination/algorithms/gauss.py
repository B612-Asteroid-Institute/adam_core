"""
Gauss's method for initial orbit determination — public Python surface
over the fused Rust kernel.

The full IOD algorithm (equatorial→ecliptic unit-vector rotation, Milani
A/B/V/C0/h0 coefficients, 8th-order polynomial roots via Durand-Kerner +
Newton polish, per-root orbit construction with optional light-time
correction) lives in `adam_core._rust_native.gauss_iod_fused_numpy`.

`calcGauss` returns the velocity at the middle observation using the
classical Gauss f/g approximation, computed in Rust.
"""

import numpy as np
import numpy.typing as npt

from adam_core._rust import calc_gauss_numpy, gauss_iod_orbits_arrow
from adam_core._rust.arrow import table_from_record_batch
from adam_core.constants import Constants as c
from adam_core.orbits import Orbits

from ._validation import three_position_vectors

__all__ = [
    "calcGauss",
    "gaussIOD",
]

MU = c.MU
C = c.C


def calcGauss(
    r1: npt.ArrayLike,
    r2: npt.ArrayLike,
    r3: npt.ArrayLike,
    t1: float,
    t2: float,
    t3: float,
) -> npt.NDArray[np.float64]:
    r"""
    Calculate the velocity vector at the middle position (r2) using Gauss's
    classical f/g Lagrange-coefficient approximation.

    .. math::
        f_1 \approx 1 - \frac{1}{2}\frac{\mu}{r_2^3} (t_1 - t_2)^2 \\
        f_3 \approx 1 - \frac{1}{2}\frac{\mu}{r_2^3} (t_3 - t_2)^2 \\
        g_1 \approx (t_1 - t_2) - \frac{1}{6}\frac{\mu}{r_2^3} (t_1 - t_2)^3 \\
        g_3 \approx (t_3 - t_2) - \frac{1}{6}\frac{\mu}{r_2^3} (t_3 - t_2)^3 \\
        \vec{v}_2 = \frac{1}{f_1 g_3 - f_3 g_1} (-f_3 \vec{r}_1 + f_1 \vec{r}_3)

    See chapter 5 in Howard D. Curtis' "Orbital Mechanics for Engineering
    Students".

    Parameters
    ----------
    r1, r2, r3 : ~numpy.ndarray (3,)
        Heliocentric position vectors at times 1, 2, 3 in AU.
    t1, t2, t3 : float
        Times in decimal days (any consistent units).

    Returns
    -------
    v2 : ~numpy.ndarray (3,)
        Velocity at r2 / t2 in AU per day.
    """
    r1_np, r2_np, r3_np = three_position_vectors(r1, r2, r3)
    rust_out = calc_gauss_numpy(r1_np, r2_np, r3_np, t1, t2, t3, MU)
    return np.asarray(rust_out, dtype=np.float64)


def gaussIOD(
    coords: npt.NDArray[np.float64],
    observation_times: npt.ArrayLike,
    coords_obs: npt.ArrayLike,
    velocity_method: str = "gibbs",
    light_time: bool = True,
    mu: float = MU,
    max_iter: int = 10,
    tol: float = 1e-15,
) -> Orbits:
    """
    Compute up to three preliminary orbits using three observations in angular
    equatorial coordinates.

    Parameters
    ----------
    coords : ~numpy.ndarray (3, 2)
        RA and Dec of three observations in degrees.
    observation_times : ~numpy.ndarray (3,)
        Times of the three observations in decimal days (MJD or JD).
    coords_obs : ~numpy.ndarray (3, 3)
        Heliocentric position vectors of the observer at the three times in AU.
    velocity_method : {'gauss', 'gibbs', 'herrick+gibbs'}, optional
        Method for the velocity at the second observation. [Default 'gibbs']
    light_time : bool, optional
        Correct for light-travel time. [Default True]
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in AU**3 / day**2.
        The same value is used for candidate geometry and velocity. This fixes
        the legacy implementation's physically inconsistent split-``mu`` path.
    max_iter : int, optional
        Reserved for future use; currently ignored by the Rust kernel.
    tol : float, optional
        Reserved for future use; currently ignored by the Rust kernel.

    Returns
    -------
    orbits : `~adam_core.orbits.Orbits`
        Up to three preliminary orbits (zero rows if no real-positive root).
    """
    # One crossing: the fused kernel, non-finite filtering, orbit-id
    # generation, and finished Orbits table assembly all run in Rust.
    result = gauss_iod_orbits_arrow(
        np.ascontiguousarray(coords[:, 0], dtype=np.float64),
        np.ascontiguousarray(coords[:, 1], dtype=np.float64),
        np.ascontiguousarray(np.asarray(observation_times, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(coords_obs, dtype=np.float64)),
        velocity_method,
        bool(light_time),
        float(mu),
        float(C),
    )
    if result.num_rows == 0:
        return Orbits.empty()
    return table_from_record_batch(Orbits, result)
