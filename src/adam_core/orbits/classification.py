from typing import Union

import numpy as np

from .._rust import classify_orbits_numpy as _rust_classify_orbits_numpy
from ..coordinates.cometary import CometaryCoordinates
from ..coordinates.keplerian import KeplerianCoordinates


# Maps the rust kernel's int32 class codes (see classification.rs) to the
# PDS Small Bodies Node string labels.
CLASS_CODE_TO_NAME: tuple[str, ...] = (
    "AST",  # 0  default
    "AMO",  # 1
    "APO",  # 2
    "ATE",  # 3
    "CEN",  # 4
    "IEO",  # 5
    "IMB",  # 6
    "MBA",  # 7
    "MCA",  # 8
    "OMB",  # 9
    "TJN",  # 10
    "TNO",  # 11
    "PAA",  # 12
    "HYA",  # 13
)


def calc_orbit_class(
    elements: Union[KeplerianCoordinates, CometaryCoordinates],
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
        a = np.ascontiguousarray(np.asarray(elements.a, dtype=np.float64))
        e = elements.e.to_numpy(zero_copy_only=False).astype(np.float64)
        q = elements.q.to_numpy(zero_copy_only=False).astype(np.float64)
        q_apo = np.ascontiguousarray(np.asarray(elements.Q, dtype=np.float64))
    elif isinstance(elements, KeplerianCoordinates):
        a = elements.a.to_numpy(zero_copy_only=False).astype(np.float64)
        e = elements.e.to_numpy(zero_copy_only=False).astype(np.float64)
        q = np.ascontiguousarray(np.asarray(elements.q, dtype=np.float64))
        q_apo = np.ascontiguousarray(np.asarray(elements.Q, dtype=np.float64))
    else:
        raise TypeError(
            f"elements must be KeplerianCoordinates or CometaryCoordinates, "
            f"got {type(elements).__name__}"
        )

    codes = _rust_classify_orbits_numpy(a, e, q, q_apo)

    name_lookup = np.asarray(CLASS_CODE_TO_NAME, dtype=object)
    return name_lookup[np.asarray(codes, dtype=np.int64)]
