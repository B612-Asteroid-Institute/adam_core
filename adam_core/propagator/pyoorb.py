try:
    import pyoorb as oo
except ImportError:
    raise ImportError("PYOORB is not installed.")

import os
import warnings
from typing import Optional

import numpy as np
from astropy.time import Time

from ..coordinates.cartesian import CartesianCoordinates
from ..orbits.orbits import Orbits
from .propagator import Propagator

PYOORB_CONFIG = {"dynamical_model": "N", "ephemeris_file": "de430.dat"}


class PYOORB(Propagator):
    def __init__(self, **kwargs):
        # Make sure only the correct kwargs are passed to the constructor
        allowed_kwargs = PYOORB_CONFIG.keys()
        for k in kwargs:
            if k not in allowed_kwargs:
                warnings.warn(
                    f"Invalid argument {k} passed to PYOORB propagator. Ignoring..."
                )

        # If an allowed kwarg is missing, add the default
        for k in allowed_kwargs:
            if k not in kwargs:
                kwargs[k] = PYOORB_CONFIG[k]

        self.dynamical_model = kwargs["dynamical_model"]
        self.ephemeris_file = kwargs["ephemeris_file"]

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

            # Prepare pyoorb
            ephfile = os.path.join(os.getenv("OORB_DATA"), self.ephemeris_file)
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
        orbit_type: str,
        time_scale: str,
        magnitude: Optional[float] = None,
        slope: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convert an array of orbits into the format expected by PYOORB.

        Parameters
        ----------
        orbits : `~numpy.ndarray` (N, 6)
            Orbits to convert. See orbit_type for expected input format.
        t0 : `~numpy.ndarray` (N)
            Epoch in MJD at which the orbits are defined.
        orbit_type : {'cartesian', 'keplerian', 'cometary'}, optional
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
        time_scale : {'UTC', 'UT1', 'TT', 'TAI'}, optional
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

        if orbit_type == "cartesian":
            orbit_type_ = np.array([1 for i in range(num_orbits)])
        elif orbit_type == "cometary":
            orbit_type_ = np.array([2 for i in range(num_orbits)])
            # H = M1
            # G = K1
            orbits_[:, 1:5] = np.radians(orbits_[:, 1:5])
        elif orbit_type == "keplerian":
            orbit_type_ = np.array([3 for i in range(num_orbits)])
            orbits_[:, 1:] = np.radians(orbits_[:, 1:])
        else:
            raise ValueError(
                "orbit_type should be one of {'cartesian', 'keplerian', 'cometary'}"
            )

        if time_scale == "UTC":
            time_scale_ = np.array([1 for i in range(num_orbits)])
        elif time_scale == "UT1":
            time_scale_ = np.array([2 for i in range(num_orbits)])
        elif time_scale == "TT":
            time_scale_ = np.array([3 for i in range(num_orbits)])
        elif time_scale == "TAI":
            time_scale_ = np.array([4 for i in range(num_orbits)])
        else:
            raise ValueError("time_scale should be one of {'UTC', 'UT1', 'TT', 'TAI'}")

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
    def _configure_epochs(epochs: np.ndarray, time_scale: str) -> np.ndarray:
        """
        Convert an array of orbits into the format expected by PYOORB.

        Parameters
        ----------
        epochs : `~numpy.ndarray` (N)
            Epoch in MJD to convert.
        time_scale : {'UTC', 'UT1', 'TT', 'TAI'}
            Time scale of the MJD epochs.

        Returns
        -------
        epochs_pyoorb : `~numpy.ndarray` (N, 2)
            Epochs converted into the PYOORB format.
        """
        num_times = len(epochs)
        if time_scale == "UTC":
            time_scale_list = [1 for i in range(num_times)]
        elif time_scale == "UT1":
            time_scale_list = [2 for i in range(num_times)]
        elif time_scale == "TT":
            time_scale_list = [3 for i in range(num_times)]
        elif time_scale == "TAI":
            time_scale_list = [4 for i in range(num_times)]
        else:
            raise ValueError("time_scale should be one of {'UTC', 'UT1', 'TT', 'TAI'}")

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
            orbits.cartesian.values.filled(),
            orbits.cartesian.times.tt.mjd,
            "cartesian",
            "TT",
            magnitude=None,
            slope=None,
        )

        # Convert epochs into PYOORB format
        epochs_pyoorb = self._configure_epochs(times.tt.mjd, "TT")

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

        # Convert output epochs to TDB
        times_ = Time(mjd_tt, format="mjd", scale="tt")
        times_ = times_.tdb

        if orbits.object_ids is not None:
            object_ids = orbits.object_ids[orbit_ids_]
        else:
            object_ids = None

        if orbits.orbit_ids is not None:
            orbit_ids = orbits.orbit_ids[orbit_ids_]
        else:
            orbit_ids = None

        propagated_orbits = Orbits(
            CartesianCoordinates(
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                vz=vz,
                times=times_,
                origin="heliocenter",
                frame="ecliptic",
            ),
            orbit_ids=orbit_ids,
            object_ids=object_ids,
        )
        return propagated_orbits

    def _generate_ephemeris(self, orbits: Orbits, observers):
        raise NotImplementedError(
            "Ephemeris generation is not yet implemented for PYOORB."
        )
