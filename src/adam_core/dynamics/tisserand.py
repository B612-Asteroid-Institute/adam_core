"""
This code generates the dictionary of semi-major axes for the
third body needed for the Tisserand parameter

from adam_core.orbits.query import _get_horizons_elements

ids = ["199", "299", "399", "499", "599", "699", "799", "899"]
elements = _get_horizons_elements(ids, times, id_type="majorbody")

MAJOR_BODIES = {}
for i, r in elements[["targetname", "a"]].iterrows():
   body_name = r["targetname"].split(" ")[0].lower()
   MAJOR_BODIES[body_name] = r["a"]

"""

import numpy as np

from .._rust import tisserand_parameter_numpy as _rust_tisserand_parameter_numpy

MAJOR_BODIES = {
    "mercury": 0.3870970330236769,
    "venus": 0.723341974974844,
    "earth": 0.9997889954736553,
    "mars": 1.523803685638066,
    "jupiter": 5.203719697535582,
    "saturn": 9.579110220472034,
    "uranus": 19.18646168457971,
    "neptune": 30.22486701698071,
}


def calc_tisserand_parameter(a, e, i, third_body="jupiter"):
    """
    Calculate Tisserand's parameter used to identify potential comets.

    Tp = a_p/a + 2·cos(i)·sqrt((a/a_p)·(1−e²))

    Objects with Jupiter Tisserand parameter > 3 are typically asteroids;
    Jupiter-family comets fall between 2 and 3; Damocloids are below 2.

    Parameters
    ----------
    a : float or `~numpy.ndarray` (N)
        Semi-major axis in au.
    e : float or `~numpy.ndarray` (N)
        Eccentricity.
    i : float or `~numpy.ndarray` (N)
        Inclination in degrees.
    third_body : str
        Name of planet with respect to which Tisserand's parameter
        should be calculated.

    Returns
    -------
    Tp : float or `~numpy.ndarray` (N)
        Tisserand's parameter.
    """
    if third_body not in MAJOR_BODIES:
        valid = ",".join(MAJOR_BODIES.keys())
        raise ValueError(f"third_body should be one of {valid}")

    ap = MAJOR_BODIES[third_body]

    # Coerce scalar inputs to 1-element arrays so the rust kernel can be
    # invoked uniformly; squeeze back at the end for scalar-in / scalar-out.
    a_arr = np.atleast_1d(np.asarray(a, dtype=np.float64))
    e_arr = np.atleast_1d(np.asarray(e, dtype=np.float64))
    i_arr = np.atleast_1d(np.asarray(i, dtype=np.float64))

    out = _rust_tisserand_parameter_numpy(a_arr, e_arr, i_arr, ap)
    assert out is not None, "rust tisserand_parameter unavailable"

    if np.isscalar(a) and np.isscalar(e) and np.isscalar(i):
        return float(out[0])
    return np.asarray(out, dtype=np.float64)
