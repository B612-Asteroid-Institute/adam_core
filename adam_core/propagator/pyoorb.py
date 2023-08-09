try:
    import pyoorb as oo
except ImportError:
    raise ImportError("PYOORB is not installed.")

import enum
import os
import warnings
from typing import Optional, Union

import numpy as np
from astropy.time import Time

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin
from ..coordinates.times import Times
from ..orbits.orbits import Orbits
from .propagator import Propagator
from .utils import _assert_times_almost_equal


class OpenOrbTimescale(enum.Enum):
    UTC = 1
    UT1 = 2
    TT = 3
    TAI = 4


class OpenOrbOrbitType(enum.Enum):
    CARTESIAN = 1
    COMETARY = 2
    KEPLERIAN = 3


class PYOORB(Propagator):
    def __init__(
        self, *, dynamical_model: str = "N", ephemeris_file: str = "de430.dat"
    ):

        self.dynamical_model = dynamical_model
        self.ephemeris_file = ephemeris_file

        env_var = "ADAM_CORE_PYOORB_INITIALIZED"
        if env_var in os.environ.keys() and os.environ[env_var] == "True":
            pass
        else:
            if os.environ.get("OORB_DATA") is None:
                if os.environ.get("CONDA_PREFIX") is None:
                    raise RuntimeError(
                        "Cannot find OORB_DATA directory. Please set the OORB_DATA environment variable."
                    )
                else:
                    os.environ["OORB_DATA"] = os.path.join(
                        os.environ["CONDA_PREFIX"], "share/openorb"
                    )

            oorb_data = os.environ["OORB_DATA"]

            # Prepare pyoorb
            ephfile = os.path.join(oorb_data, self.ephemeris_file)
            err = oo.pyoorb.oorb_init(ephfile)
            if err == 0:
                os.environ[env_var] = "True"
            else:
                warnings.warn(f"PYOORB returned error code: {err}")

        return

    @staticmethod
    def _configure_orbits(
        orbits: np.ndarray,
        t0: np.ndarray,
        orbit_type: OpenOrbOrbitType,
        time_scale: OpenOrbTimescale,
        magnitude: Optional[Union[float, np.ndarray]] = None,
        slope: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Convert an array of orbits into the format expected by PYOORB.

        Parameters
        ----------
        orbits : `~numpy.ndarray` (N, 6)
            Orbits to convert. See orbit_type for expected input format.
        t0 : `~numpy.ndarray` (N)
            Epoch in MJD at which the orbits are defined.
        orbit_type : OpenOrbOrbitType
            Orbital element representation of the provided orbits.
            If cartesian:
                x : heliocentric ecliptic J2000 x position in AU
                y : heliocentric ecliptic J2000 y position in AU
                z : heliocentric ecliptic J2000 z position in AU
                vx : heliocentric ecliptic J2000 x velocity in AU per day
                vy : heliocentric ecliptic J2000 y velocity in AU per day
                vz : heliocentric ecliptic J2000 z velocity in AU per day
            If keplerian:
                a : semi-major axis in AU
                e : eccentricity in degrees
                i : inclination in degrees
                Omega : longitude of the ascending node in degrees
                omega : argument of periapsis in degrees
                M0 : mean anomaly in degrees
            If cometary:
                p : perihelion distance in AU
                e : eccentricity in degrees
                i : inclination in degrees
                Omega : longitude of the ascending node in degrees
                omega : argument of periapsis in degrees
                T0 : time of perihelion passage in MJD
        time_scale : OpenOrbTimescale
            Time scale of the MJD epochs.
        magnitude : float or `~numpy.ndarray` (N), optional
            Absolute H-magnitude or M1 magnitude.
        slope : float or `~numpy.ndarray` (N), optional
            Photometric slope parameter G or K1.

        Returns
        -------
        orbits_pyoorb : `~numpy.ndarray` (N, 12)
            Orbits formatted in the format expected by PYOORB.
                orbit_id : index of input orbits
                elements x6: orbital elements of propagated orbits
                orbit_type : orbit type
                epoch_mjd : epoch of the propagate orbit
                time_scale : time scale of output epochs
                H/M1 : absolute magnitude
                G/K1 : photometric slope parameter
        """
        orbits_ = orbits.copy()
        num_orbits = orbits_.shape[0]

        orbit_type_ = np.array([orbit_type.value for i in range(num_orbits)])

        time_scale_ = np.array([time_scale.value for i in range(num_orbits)])

        if isinstance(slope, (float, int)):
            slope_ = np.array([slope for i in range(num_orbits)])
        elif isinstance(slope, list):
            slope_ = np.array(slope)
        elif isinstance(slope, np.ndarray):
            slope_ = slope
        else:
            slope_ = np.array([0.15 for i in range(num_orbits)])

        if isinstance(magnitude, (float, int)):
            magnitude_ = np.array([magnitude for i in range(num_orbits)])
        elif isinstance(magnitude, list):
            magnitude_ = np.array(magnitude)
        elif isinstance(magnitude, np.ndarray):
            magnitude_ = magnitude
        else:
            magnitude_ = np.array([20.0 for i in range(num_orbits)])

        ids = np.array([i for i in range(num_orbits)])

        orbits_pyoorb = np.zeros((num_orbits, 12), dtype=np.double, order="F")
        orbits_pyoorb[:, 0] = ids
        orbits_pyoorb[:, 1:7] = orbits_
        orbits_pyoorb[:, 7] = orbit_type_
        orbits_pyoorb[:, 8] = t0
        orbits_pyoorb[:, 9] = time_scale_
        orbits_pyoorb[:, 10] = magnitude_
        orbits_pyoorb[:, 11] = slope_

        return orbits_pyoorb

    @staticmethod
    def _configure_epochs(
        epochs: np.ndarray, time_scale: OpenOrbTimescale
    ) -> np.ndarray:
        """
        Convert an array of orbits into the format expected by PYOORB.

        Parameters
        ----------
        epochs : `~numpy.ndarray` (N)
            Epoch in MJD to convert.
        time_scale : OpenOrbTimescale
            Time scale of the MJD epochs.

        Returns
        -------
        epochs_pyoorb : `~numpy.ndarray` (N, 2)
            Epochs converted into the PYOORB format.
        """
        num_times = len(epochs)
        time_scale_list = [time_scale.value for i in range(num_times)]
        epochs_pyoorb = np.array(
            list(np.vstack([epochs, time_scale_list]).T), dtype=np.double, order="F"
        )
        return epochs_pyoorb

    def _propagate_orbits(self, orbits: Orbits, times: Time) -> Orbits:
        """
        Propagate orbits using PYOORB.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            Orbits to propagate.
        times : `~astropy.time.core.Time` (M)
            Times to which to propagate orbits.

        Returns
        -------
        propagated : `~adam_core.orbits.orbits.Orbits` (N * M)
            Orbits propagated to each time in times.
        """
        # Convert orbits into PYOORB format
        orbits_pyoorb = self._configure_orbits(
            orbits.coordinates.values,
            orbits.coordinates.time.to_astropy().tt.mjd,
            OpenOrbOrbitType.CARTESIAN,
            OpenOrbTimescale.TT,
            magnitude=None,
            slope=None,
        )

        # Convert epochs into PYOORB format
        epochs_pyoorb = self._configure_epochs(times.tt.mjd, OpenOrbTimescale.TT)

        # Propagate orbits to each epoch and append to list
        # of new states
        states_list = []
        orbits_pyoorb_i = orbits_pyoorb.copy()
        for epoch in epochs_pyoorb:
            orbits_pyoorb_i, err = oo.pyoorb.oorb_propagation(
                in_orbits=orbits_pyoorb_i,
                in_epoch=epoch,
                in_dynmodel=self.dynamical_model,
            )
            states_list.append(orbits_pyoorb_i)

        # Convert list of new states into a pandas data frame
        # These states at the moment will always be return as cartesian
        # state vectors
        # elements = ["x", "y", "z", "vx", "vy", "vz"]
        # Other PYOORB state vector representations:
        # "keplerian":
        #    elements = ["a", "e", "i", "Omega", "omega", "M0"]
        # "cometary":
        #    elements = ["q", "e", "i", "Omega", "omega", "T0"]
        states = np.concatenate(states_list)

        # Extract cartesian states from PYOORB results
        orbit_ids_ = states[:, 0].astype(int)
        x = states[:, 1]
        y = states[:, 2]
        z = states[:, 3]
        vx = states[:, 4]
        vy = states[:, 5]
        vz = states[:, 6]
        mjd_tt = states[:, 8]

        # Check to make sure the desired times are within an acceptable
        # tolerance
        _assert_times_almost_equal(mjd_tt, np.repeat(epochs_pyoorb[:, 0], len(orbits)))

        # Convert output epochs to TDB
        times_ = Time(mjd_tt, format="mjd", scale="tt")
        times_ = times_.tdb

        # Map the object and orbit IDs back to the input arrays
        object_ids = orbits.object_id.to_numpy(zero_copy_only=False)[orbit_ids_]
        orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)[orbit_ids_]

        propagated_orbits = Orbits.from_kwargs(
            orbit_id=orbit_ids,
            object_id=object_ids,
            coordinates=CartesianCoordinates.from_kwargs(
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                vz=vz,
                time=Times.from_astropy(times_),
                origin=Origin.from_kwargs(code=["SUN" for i in range(len(times_))]),
                frame="ecliptic",
            ),
        )
        return propagated_orbits

    def _generate_ephemeris(self, orbits: Orbits, observers):
        raise NotImplementedError(
            "Ephemeris generation is not yet implemented for PYOORB."
        )
