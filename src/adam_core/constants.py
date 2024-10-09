import numpy as np

__all__ = [
    "KM_P_AU",
    "S_P_DAY",
    "Constants",
    "DE44X",
]

# km in an au
KM_P_AU = 149597870.700
# seconds in a day
S_P_DAY = 86400.0


class _Constants:
    def __init__(
        self,
        C=None,
        MU=None,
        R_EARTH_EQUATORIAL=None,
        R_EARTH_POLAR=None,
        OBLIQUITY=None,
    ):
        self.C = C
        self.MU = MU
        self.R_EARTH_EQUATORIAL = R_EARTH_EQUATORIAL
        self.R_EARTH_POLAR = R_EARTH_POLAR
        self.OBLIQUITY = OBLIQUITY

        # Transformation matrix from Equatorial J2000 to Ecliptic J2000
        self.TRANSFORM_EQ2EC = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.OBLIQUITY), np.sin(self.OBLIQUITY)],
                [0, -np.sin(self.OBLIQUITY), np.cos(self.OBLIQUITY)],
            ]
        )

        # Transformation matrix from Ecliptic J2000 to Equatorial J2000
        self.TRANSFORM_EC2EQ = self.TRANSFORM_EQ2EC.T
        return


DE44X_CONSTANTS = {
    # Speed of Light : au / d (299792.458 km / s -- DE430/DE431)
    "C": 299792.458 / KM_P_AU * S_P_DAY,
    # Standard Gravitational Parameter -- Sun :  au**3 / d**2 (0.29591220828411956E-03 -- DE441/DE440)
    "MU": 0.29591220828411956e-03,
    # Earth Equatorial Radius: au (6378.1363 km -- DE431/DE430)
    "R_EARTH_EQUATORIAL": 6378.1363 / KM_P_AU,
    # Earth Polar Radius: au (6356.7523 km)
    "R_EARTH_POLAR": 6356.7523 / KM_P_AU,
    # Mean Obliquity at J2000: radians (84381.448 arcseconds -- DE431/DE430)
    "OBLIQUITY": 84381.448 * np.pi / (180.0 * 3600.0),
}
DE44X = Constants = _Constants(**DE44X_CONSTANTS)
