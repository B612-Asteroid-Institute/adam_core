import logging
from enum import Enum

import numpy as np
import pyarrow.compute as pc
import quivr as qv

from ..constants import KM_P_AU, S_P_DAY

logger = logging.getLogger(__name__)


class OriginCodes(Enum):
    SOLAR_SYSTEM_BARYCENTER = 0
    MERCURY_BARYCENTER = 1
    VENUS_BARYCENTER = 2
    EARTH_MOON_BARYCENTER = 3
    MARS_BARYCENTER = 4
    JUPITER_BARYCENTER = 5
    SATURN_BARYCENTER = 6
    URANUS_BARYCENTER = 7
    NEPTUNE_BARYCENTER = 8
    SUN = 10
    MERCURY = 199
    VENUS = 299
    EARTH = 399
    MOON = 301
    MARS = 499
    JUPITER = 599
    SATURN = 699
    URANUS = 799
    NEPTUNE = 899


def _convert_mu_units(mu: float):
    """
    Convert mu in km^3 / s^2 to au^3 / day^2.
    """
    return mu / KM_P_AU**3 * S_P_DAY**2


class OriginGravitationalParameters(float, Enum):
    # Taken from https://iopscience.iop.org/article/10.3847/1538-3881/abd414
    MERCURY_BARYCENTER = _convert_mu_units(22032.080486418)
    VENUS_BARYCENTER = _convert_mu_units(324858.592000)
    MARS_BARYCENTER = _convert_mu_units(42828.375816)
    JUPITER_BARYCENTER = _convert_mu_units(126712764.100000)
    SATURN_BARYCENTER = _convert_mu_units(37940584.841800)
    URANUS_BARYCENTER = _convert_mu_units(5794556.400000)
    NEPTUNE_BARYCENTER = _convert_mu_units(6836527.100580)
    PLUTO_BARYCENTER = _convert_mu_units(975.500000)
    SUN = _convert_mu_units(132712440041.279419)
    MERCURY = _convert_mu_units(22031.868551)
    VENUS = _convert_mu_units(324858.592000)
    EARTH = _convert_mu_units(398600.435507)
    MOON = _convert_mu_units(4902.800118)

    @classmethod
    def SOLAR_SYSTEM_BARYCENTER(cls) -> float:
        """
        Return the gravitational parameter of the Solar System barycenter as approximated
        by adding the gravitational parameters of the Sun, Mercury, Venus, Earth, Moon,
        Mars, Jupiter, Uranus, and Neptune.

        Returns
        -------
        mu : float
            The gravitational parameter of the Solar System barycenter in au^3 / day^2.
        """
        return (
            cls.SUN
            + cls.MERCURY_BARYCENTER
            + cls.VENUS_BARYCENTER
            + cls.EARTH
            + cls.MOON
            + cls.MARS_BARYCENTER
            + cls.JUPITER_BARYCENTER
            + cls.URANUS_BARYCENTER
            + cls.NEPTUNE_BARYCENTER
            + cls.PLUTO_BARYCENTER
        )


# TODO: Replace with DictionaryColumn or similar
#       Investigate whether this class is even necessary
class Origin(qv.Table):
    code = qv.LargeStringColumn()

    def as_OriginCodes(self) -> OriginCodes:
        """
        Convert the origin codes to an `~adam_core.coordinates.origin.OriginCodes` object.

        Returns
        -------
        OriginCodes
            Origin codes as an `~adam_core.coordinates.origin.OriginCodes` object.
        """
        assert (
            len(self.code.unique()) == 1
        ), "Only one origin code can be converted at a time."
        return OriginCodes[self.code.unique()[0].as_py()]

    def __eq__(self, other: object) -> np.ndarray:
        if isinstance(other, (str, np.ndarray)):
            codes = self.code.to_numpy(zero_copy_only=False)
            return codes == other
        elif isinstance(other, OriginCodes):
            codes = self.code.to_numpy(zero_copy_only=False)
            return codes == other.name
        elif isinstance(other, Origin):
            codes = self.code.to_numpy(zero_copy_only=False)
            other_codes = other.code.to_numpy(zero_copy_only=False)
            return codes == other_codes
        else:
            raise TypeError(f"Cannot compare Origin to type: {type(other)}")

    def __ne__(self, other: object) -> np.ndarray:
        return ~self.__eq__(other)

    def mu(self) -> np.ndarray:
        """
        Return the gravitational parameter of the origin.

        Returns
        -------
        mu : `~numpy.ndarray` (N)
            The gravitational parameter of the origin in au^3 / day^2.

        Raises
        ------
        ValueError
            If the origin code is not recognized.
        """
        mu = np.empty(len(self.code), dtype=np.float64)
        for code in pc.unique(self.code):
            code = code.as_py()
            mask = pc.equal(self.code, code).to_numpy(zero_copy_only=False)
            if code == "SOLAR_SYSTEM_BARYCENTER":
                mu[mask] = OriginGravitationalParameters.SOLAR_SYSTEM_BARYCENTER()
            elif code in OriginGravitationalParameters.__members__:
                mu[mask] = OriginGravitationalParameters[code].value
            else:
                raise ValueError(f"Unknown origin code: {code}")

        return mu
