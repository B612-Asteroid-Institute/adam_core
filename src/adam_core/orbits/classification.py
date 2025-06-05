from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

from ..coordinates.cometary import CometaryCoordinates
from ..coordinates.keplerian import KeplerianCoordinates
from ..dynamics.tisserand import calc_tisserand_parameter


@dataclass
class MPCClassification:
    """MPC orbit classification values and their corresponding integer codes."""

    # Inner Solar System
    ATIRA: int = 0
    ATEN: int = 1
    APOLLO: int = 2
    AMOR: int = 3
    INNER_OTHER: int = 9

    # Middle Solar System
    MARS_CROSSER: int = 10
    MAIN_BELT: int = 11
    JUPITER_TROJAN: int = 12
    MIDDLE_OTHER: int = 19

    # Outer Solar System
    JUPITER_COUPLED: int = 20
    NEPTUNE_TROJAN: int = 21
    CENTAURS: int = 22
    TNO: int = 23

    # Special cases
    PARABOLIC: int = 31
    HYPERBOLIC: int = 30
    OTHER: int = 99


def mpc_class_to_int(classes: np.ndarray) -> np.ndarray:
    """
    Convert MPC classification strings to their corresponding integer values.

    Parameters
    ----------
    classes : `~numpy.ndarray`
        Array of MPC classification strings.

    Returns
    -------
    int_classes : `~numpy.ndarray`
        Array of integer classification values.
    """
    classes_int = np.zeros(len(classes), dtype=int)
    for class_name, class_value in MPCClassification.__dict__.items():
        if not class_name.startswith("__"):
            classes_int[classes == class_name] = class_value
    return classes_int


def calc_orbit_class(
    elements: Union[KeplerianCoordinates, CometaryCoordinates],
    style: Literal["PDS", "MPC"] = "PDS",
) -> np.ndarray:
    """
    Calculate the orbital class for each Keplerian or Cometary orbit.

    Based on either:
    1. The classification scheme defined by the Planetary Data System Small
       Bodies Node (see: https://pdssbn.astro.umd.edu/data_other/objclass.shtml).
    2. The classification scheme defined by the Minor Planet Center
       (see: https://www.minorplanetcenter.net/mpcops/documentation/orbit-types/).

    Parameters
    ----------
    elements : KeplerianCoordinates or CometaryCoordinates
        Keplerian orbits for which to find classes.
    style : {"PDS", "MPC"}, optional
        Classification scheme to use. Default is "PDS".

    Returns
    -------
    orbit_class : `~numpy.ndarray`
        Class for each orbit.
    """
    if style == "PDS":
        return _calc_pds_class(elements)
    elif style == "MPC":
        return _calc_mpc_class(elements)
    else:
        raise ValueError(f"Invalid classification style: {style}")


def _calc_pds_class(
    elements: Union[KeplerianCoordinates, CometaryCoordinates],
) -> np.ndarray:
    """Calculate PDS classification scheme."""
    if isinstance(elements, CometaryCoordinates):
        a = elements.a
        e = elements.e.to_numpy(zero_copy_only=False)
        q = elements.q.to_numpy(zero_copy_only=False)
        Q = elements.Q
    else:  # KeplerianCoordinates
        a = elements.a.to_numpy(zero_copy_only=False)
        e = elements.e.to_numpy(zero_copy_only=False)
        q = elements.q
        Q = elements.Q

    orbit_class = np.array(["AST" for _ in range(len(elements))])

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


def _calc_mpc_class(
    elements: Union[KeplerianCoordinates, CometaryCoordinates],
) -> np.ndarray:
    """
    Calculate MPC classification scheme.

    Parameters
    ----------
    elements : KeplerianCoordinates or CometaryCoordinates
        Keplerian or Cometary orbits for which to find classes.

    Returns
    -------
    orbit_class : `~numpy.ndarray`
        Class for each orbit.

    See Also
    --------
    mpc_class_to_int : Convert MPC classification strings to integer values.
    """
    if isinstance(elements, CometaryCoordinates):
        a = elements.a
        e = elements.e.to_numpy(zero_copy_only=False)
        q = elements.q.to_numpy(zero_copy_only=False)
        Q = elements.Q
        i = elements.i.to_numpy(zero_copy_only=False)
    else:  # KeplerianCoordinates
        a = elements.a.to_numpy(zero_copy_only=False)
        e = elements.e.to_numpy(zero_copy_only=False)
        q = elements.q
        Q = elements.Q
        i = elements.i.to_numpy(zero_copy_only=False)

    # Constants from MPC documentation
    q_mars = 1.405
    a_jupiter = 5.204
    a_neptune = 30.178

    # Calculate Tisserand parameter with respect to Jupiter
    Tp = calc_tisserand_parameter(a, e, i, third_body="jupiter")

    orbit_class = np.array(["Other" for _ in range(len(elements))], dtype="<U16")

    # Classification order following MPC documentation
    # Inner Solar System
    orbit_class = np.where(
        (a < 1.0) & (Q < 0.983) & (orbit_class == "Other"), "Atira", orbit_class
    )
    orbit_class = np.where(
        (a < 1.0) & (Q >= 0.983) & (orbit_class == "Other"), "Aten", orbit_class
    )
    orbit_class = np.where(
        (a >= 1.0) & (q < 1.017) & (orbit_class == "Other"), "Apollo", orbit_class
    )
    orbit_class = np.where(
        (a >= 1.0) & (q >= 1.017) & (q < 1.3) & (orbit_class == "Other"),
        "Amor",
        orbit_class,
    )

    # Middle Solar System
    orbit_class = np.where(
        (a >= 1.0) & (a < 3.2) & (q > 1.3) & (q < 1.666) & (orbit_class == "Other"),
        "Mars Crosser",
        orbit_class,
    )
    orbit_class = np.where(
        (a >= 1.0) & (a < 3.27831) & (orbit_class == "Other"), "Main Belt", orbit_class
    )
    orbit_class = np.where(
        (a > 4.8) & (a < 5.4) & (e < 0.3) & (orbit_class == "Other"),
        "Jupiter Trojan",
        orbit_class,
    )

    # Outer Solar System
    orbit_class = np.where(
        (a >= 1.0) & (Tp >= 2) & (Tp <= 3) & (orbit_class == "Other"),
        "Jupiter Coupled",
        orbit_class,
    )
    orbit_class = np.where(
        (a > 29.8) & (a < 30.4) & (orbit_class == "Other"),
        "Neptune Trojan",
        orbit_class,
    )
    orbit_class = np.where(
        (a >= a_jupiter) & (a < a_neptune) & (orbit_class == "Other"),
        "Centaur",
        orbit_class,
    )
    orbit_class = np.where(
        (a >= a_neptune) & (orbit_class == "Other"), "TNO", orbit_class
    )

    # Special cases
    orbit_class = np.where(
        (e == 1) & (orbit_class == "Other"), "Parabolic", orbit_class
    )
    orbit_class = np.where(
        (e > 1) & (orbit_class == "Other"), "Hyperbolic", orbit_class
    )

    # Inner Other and Middle Other are applied last
    orbit_class = np.where(
        (a >= 1.0) & (Q < q_mars) & (orbit_class == "Other"), "Inner Other", orbit_class
    )
    orbit_class = np.where(
        (a < a_jupiter) & (orbit_class == "Other"), "Middle Other", orbit_class
    )

    return orbit_class
