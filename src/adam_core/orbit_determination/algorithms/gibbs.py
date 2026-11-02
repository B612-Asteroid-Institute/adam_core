import numpy as np
import numpy.typing as npt

from adam_core._rust import calc_gibbs_numpy
from adam_core.constants import Constants as c

from ._validation import three_position_vectors

__all__ = ["calcGibbs"]

MU = c.MU


def calcGibbs(
    r1: npt.ArrayLike, r2: npt.ArrayLike, r3: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the velocity vector at the location of the second position vector (r2) using the
    Gibbs method.

    .. math::
        \vec{D} = \vec{r}_1 \times \vec{r}_2  +  \vec{r}_2 \times \vec{r}_3 +  \vec{r}_3 \times \vec{r}_1

        \vec{N} = r_1 (\vec{r}_2 \times \vec{r}_3) + r_2 (\vec{r}_3 \times \vec{r}_1) + r_3 (\vec{r}_1 \times \vec{r}_2)

        \vec{B} \equiv \vec{D} \times \vec{r}_2

        L_g \equiv \sqrt{\frac{\mu}{ND}}

        \vec{v}_2 = \frac{L_g}{r_2} \vec{B} + L_g \vec{S}

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

    Returns
    -------
    v2 : `~numpy.ndarray` (3)
        Velocity of object at position r2 at time t2 in units of AU per day.
    """
    r1_np, r2_np, r3_np = three_position_vectors(r1, r2, r3)
    rust_out = calc_gibbs_numpy(r1_np, r2_np, r3_np, MU)
    return np.asarray(rust_out, dtype=np.float64)
