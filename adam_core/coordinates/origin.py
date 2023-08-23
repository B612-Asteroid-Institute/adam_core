import logging
from enum import Enum
from typing import Optional

import numpy as np
import pyarrow as pa
from quivr import StringColumn, Table

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
    SUN = _convert_mu_units(132712440041.279419)
    MERCURY = _convert_mu_units(22031.868551)
    VENUS = _convert_mu_units(324858.592000)
    EARTH = _convert_mu_units(398600.435507)
    MOON = _convert_mu_units(4902.800118)


# TODO: Replace with DictionaryColumn or similar
#       Investigate whether this class is even necessary
class Origin(Table):

    code = StringColumn()

    def __init__(self, table: pa.Table, mu: Optional[float] = None):
        super().__init__(table)
        self._mu = mu

    def with_table(self, table: pa.Table) -> "Origin":
        return super().with_table(table, mu=self.mu)

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

    @property
    def mu(self):
        if self._mu is None:
            logger.debug(
                "Origin.mu called without mu set. Finding mu in OriginGravitationalParameters."
            )
            codes = np.array(self.code)
            if len(np.unique(codes)) > 1:
                raise ValueError("Origin.mu called on table with multiple origins.")

            try:
                return OriginGravitationalParameters[codes[0]].value
            except KeyError:
                raise ValueError(
                    "Origin.mu called on table with unrecognized origin code."
                )
        else:
            return self._mu

    @mu.setter
    def mu(self, mu: float):
        self._mu = mu

    @mu.deleter
    def mu(self):
        self._mu = None
