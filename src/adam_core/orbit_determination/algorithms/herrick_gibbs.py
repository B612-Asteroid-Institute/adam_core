import numpy as np

from adam_core._rust import calc_herrick_gibbs_numpy
from adam_core.constants import Constants as c

__all__ = ["calcHerrickGibbs"]

MU = c.MU


def calcHerrickGibbs(r1, r2, r3, t1, t2, t3):
    r"""
    Calculates the velocity vector at the location of the second position vector (r2) using the
    Herrick-Gibbs formula.

    .. math::
        \vec{v}_2 =
            -\Delta t_{32} \left ( \frac{1}{ \Delta t_{21} \Delta t_{31}}
                + \frac{\mu}{12 r_1^3}  \right ) \vec{r}_2
            + ( \Delta t_{32} - \Delta t_{21}) \left ( \frac{1}{ \Delta t_{21} \Delta t_{32}}
                + \frac{\mu}{12 r_2^3}  \right ) \vec{r}_2
            + \Delta t_{21} \left ( \frac{1}{ \Delta t_{32} \Delta t_{31}}
                + \frac{\mu}{12 r_3^3}  \right ) \vec{r}_3

    For more details on theory see Chapter 4 in David A. Vallado's "Fundamentals of Astrodynamics
    and Applications".

    Parameters
    ----------
    r1 : `~numpy.ndarray` (3)
        Heliocentric position vector at time 1 in cartesian coordinates in units
        of AU.
    r2 : `~numpy.ndarray` (3)
        Heliocentric position vector at time 2 in cartesian coordinates in units
        of AU.
    r3 : `~numpy.ndarray` (3)
        Heliocentric position vector at time 3 in cartesian coordinates in units
        of AU.
    t1 : float
        Time at r1. Units of MJD or JD work or any decimal time format (one day is 1.00) as
        long as all times are given in the same format.
    t2 : float
        Time at r2. Units of MJD or JD work or any decimal time format (one day is 1.00) as
        long as all times are given in the same format.
    t3 : float
        Time at r3. Units of MJD or JD work or any decimal time format (one day is 1.00) as
        long as all times are given in the same format.

    Returns
    -------
    v2 : `~numpy.ndarray` (3)
        Velocity of object at position r2 at time t2 in units of AU per day.
    """
    r1_np = np.ascontiguousarray(np.asarray(r1, dtype=np.float64))
    r2_np = np.ascontiguousarray(np.asarray(r2, dtype=np.float64))
    r3_np = np.ascontiguousarray(np.asarray(r3, dtype=np.float64))
    if r1_np.shape != (3,) or r2_np.shape != (3,) or r3_np.shape != (3,):
        raise ValueError("r1, r2, and r3 must each have shape (3,)")

    rust_out = calc_herrick_gibbs_numpy(r1_np, r2_np, r3_np, t1, t2, t3, MU)
    return rust_out
