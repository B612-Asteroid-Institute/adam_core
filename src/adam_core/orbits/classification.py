from typing import Union

import numpy as np

from ..coordinates.cometary import CometaryCoordinates
from ..coordinates.keplerian import KeplerianCoordinates


def calc_orbit_class(
    elements: Union[KeplerianCoordinates, CometaryCoordinates]
) -> np.ndarray:
    """
    Calculate the orbital class for each Keplerian or Cometary orbit.

    Based on the classification scheme defined by the Planetary Data System Small
    Bodies Node (see: https://pdssbn.astro.umd.edu/data_other/objclass.shtml).

    TODO: Classification is currently limited to asteroid dynamical classes. Cometary class
    have not yet been implemented.

    Parameters
    ----------
    elements : KeplerianCoordinates or CometaryCoordinates
        Keplerian orbits for which to find classes.

    Returns
    -------
    orbit_class : `~numpy.ndarray`
        Class for each orbit.
    """
    if isinstance(elements, CometaryCoordinates):
        a = elements.a
        e = elements.e.to_numpy(zero_copy_only=False)
        q = elements.q.to_numpy(zero_copy_only=False)
        Q = elements.Q

    elif isinstance(elements, KeplerianCoordinates):
        a = elements.a.to_numpy(zero_copy_only=False)
        e = elements.e.to_numpy(zero_copy_only=False)
        q = elements.q
        Q = elements.Q

    orbit_class = np.array(["AST" for i in range(len(elements))])

    orbit_class_dict = {
        "AMO": np.where((a > 1.0) & (q > 1.017) & (q < 1.3)),
        "APO": np.where((a > 1.0) & (q < 1.017)),
        "ATE": np.where((a < 1.0) & (Q > 0.983)),
        "CEN": np.where((a > 5.5) & (a < 30.1)),
        "IEO": np.where((Q < 0.983)),
        "IMB": np.where((a < 2.0) & (q > 1.666)),
        "MBA": np.where((a > 2.0) & (a < 3.2) & (q > 1.666)),
        "MCA": np.where((a < 3.2) & (q > 1.3) & (q < 1.666)),
        "OMB": np.where((a > 3.2) & (a < 4.6)),
        "TJN": np.where((a > 4.6) & (a < 5.5) & (e < 0.3)),
        "TNO": np.where((a > 30.1)),
        "PAA": np.where((e == 1)),
        "HYA": np.where((e > 1)),
    }
    for c, v in orbit_class_dict.items():
        orbit_class[v] = c

    return orbit_class
