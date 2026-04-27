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

from adam_core._rust import calc_gauss_numpy, gauss_iod_fused_numpy
from adam_core.constants import Constants as c
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

__all__ = [
    "calcGauss",
    "gaussIOD",
]

MU = c.MU
C = c.C


def calcGauss(r1, r2, r3, t1, t2, t3):
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
    r1_np = np.ascontiguousarray(np.asarray(r1, dtype=np.float64))
    r2_np = np.ascontiguousarray(np.asarray(r2, dtype=np.float64))
    r3_np = np.ascontiguousarray(np.asarray(r3, dtype=np.float64))
    if r1_np.shape != (3,) or r2_np.shape != (3,) or r3_np.shape != (3,):
        raise ValueError("r1, r2, and r3 must each have shape (3,)")

    rust_out = calc_gauss_numpy(r1_np, r2_np, r3_np, t1, t2, t3, MU)
    assert rust_out is not None
    return rust_out


def gaussIOD(
    coords,
    observation_times,
    coords_obs,
    velocity_method="gibbs",
    light_time=True,
    mu=MU,
    max_iter=10,
    tol=1e-15,
):
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
    max_iter : int, optional
        Reserved for future use; currently ignored by the Rust kernel.
    tol : float, optional
        Reserved for future use; currently ignored by the Rust kernel.

    Returns
    -------
    orbits : `~adam_core.orbits.Orbits`
        Up to three preliminary orbits (zero rows if no real-positive root).
    """
    fused_out = gauss_iod_fused_numpy(
        np.ascontiguousarray(coords[:, 0], dtype=np.float64),
        np.ascontiguousarray(coords[:, 1], dtype=np.float64),
        np.ascontiguousarray(np.asarray(observation_times, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(coords_obs, dtype=np.float64)),
        velocity_method,
        bool(light_time),
        float(mu),
        float(C),
    )
    assert fused_out is not None
    epochs, orbits = fused_out
    if len(orbits) == 0:
        return Orbits.empty()
    finite_mask = ~np.isnan(orbits).any(axis=1)
    epochs = epochs[finite_mask]
    orbits = orbits[finite_mask]
    if len(orbits) == 0:
        return Orbits.empty()
    return Orbits.from_kwargs(
        coordinates=CartesianCoordinates.from_kwargs(
            x=orbits[:, 0],
            y=orbits[:, 1],
            z=orbits[:, 2],
            vx=orbits[:, 3],
            vy=orbits[:, 4],
            vz=orbits[:, 5],
            time=Timestamp.from_mjd(epochs, scale="utc"),
            origin=Origin.from_kwargs(
                code=np.full(len(orbits), "SUN", dtype="object")
            ),
            frame="ecliptic",
        )
    )
