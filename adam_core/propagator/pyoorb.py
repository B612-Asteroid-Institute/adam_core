try:
    import pyoorb as oo
except ImportError:
    raise ImportError("PYOORB is not installed.")

import enum
import logging
import os
from typing import Optional, Union

import numpy as np
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin
from ..coordinates.spherical import SphericalCoordinates
from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..orbits.variants import VariantEphemeris, VariantOrbits
from ..time import Timestamp
from .propagator import EphemerisType, OrbitType, Propagator
from .utils import _assert_times_almost_equal

logger = logging.getLogger(__name__)


class OpenOrbTimescale(enum.Enum):
    UTC = 1
    UT1 = 2
    TT = 3
    TAI = 4


class OpenOrbOrbitType(enum.Enum):
    CARTESIAN = 1
    COMETARY = 2
    KEPLERIAN = 3


PYOORB_INIT_CACHCE = {}


def process_safe_oorb_init(ephfile):
    """
    Initializes pyoorb only if it hasn't been initialized in this process before
    """
    pid = os.getpid()
    if pid in PYOORB_INIT_CACHCE:
        logger.debug(f"PYOORB already initialized for process {pid}")
        return

    logger.debug(f"Initializing PYOORB for process {pid}")
    PYOORB_INIT_CACHCE[pid] = True
    err = oo.pyoorb.oorb_init(ephfile)
    if err != 0:
        raise RuntimeError(f"PYOORB returned error code: {err}")


class PYOORB(Propagator):
    def __init__(
        self, *, dynamical_model: str = "N", ephemeris_file: str = "de430.dat"
    ):
        self.dynamical_model = dynamical_model
        self.ephemeris_file = ephemeris_file

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
        process_safe_oorb_init(ephfile)

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
        Convert an array of times into the format expected by PYOORB.

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

    def _propagate_orbits(self, orbits: OrbitType, times: Timestamp) -> OrbitType:
        """
        Propagate orbits using PYOORB.

        Parameters
        ----------
        orbits : {`~adam_core.orbits.orbits.Orbits`, `~adam_core.orbits.orbits.VariantOrbits`} (N)
            Orbits to propagate.
        times : Timestamp (M)
            Times to which to propagate orbits.

        Returns
        -------
        propagated : {`~adam_core.orbits.orbits.Orbits`, `~adam_core.orbits.orbits.VariantOrbits`} (N * M)
            Orbits propagated to each time in times.
        """
        # Convert orbits into PYOORB format
        orbits_pyoorb = self._configure_orbits(
            orbits.coordinates.values,
            orbits.coordinates.time.rescale("tt").mjd(),
            OpenOrbOrbitType.CARTESIAN,
            OpenOrbTimescale.TT,
            magnitude=None,
            slope=None,
        )

        # Convert epochs into PYOORB format
        epochs_pyoorb = self._configure_epochs(
            times.rescale("tt").mjd(), OpenOrbTimescale.TT
        )

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

            if err != 0:
                raise RuntimeError(f"PYOORB propagation failed with error code {err}.")

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

        # Check to make sure the desired times are within 1 microsecond
        # of the times returned by PYOORB
        _assert_times_almost_equal(
            mjd_tt, np.repeat(epochs_pyoorb[:, 0], len(orbits)), tolerance=0.001
        )

        # Convert output epochs to TDB
        times_ = Timestamp.from_mjd(mjd_tt, scale="tt").rescale("tdb")

        if isinstance(orbits, Orbits):
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
                    time=times_,
                    origin=Origin.from_kwargs(code=["SUN" for i in range(len(times_))]),
                    frame="ecliptic",
                ),
            )

        elif isinstance(orbits, VariantOrbits):
            # Map the object and orbit IDs back to the input arrays
            object_ids = orbits.object_id.to_numpy(zero_copy_only=False)[orbit_ids_]
            orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)[orbit_ids_]
            weights = orbits.weights.to_numpy()[orbit_ids_]
            weights_cov = orbits.weights_cov.to_numpy()[orbit_ids_]

            propagated_orbits = VariantOrbits.from_kwargs(
                orbit_id=orbit_ids,
                object_id=object_ids,
                weights=weights,
                weights_cov=weights_cov,
                coordinates=CartesianCoordinates.from_kwargs(
                    x=x,
                    y=y,
                    z=z,
                    vx=vx,
                    vy=vy,
                    vz=vz,
                    time=times_,
                    origin=Origin.from_kwargs(code=["SUN" for i in range(len(times_))]),
                    frame="ecliptic",
                ),
            )

        return propagated_orbits

    def _generate_ephemeris(
        self, orbits: OrbitType, observers: Observers
    ) -> EphemerisType:
        """
        Generate ephemerides for orbits as viewed from observers using PYOORB.

        Parameters
        ----------
        orbits : {`~adam_core.orbits.orbits.Orbits`, `~adam_core.orbits.orbits.VariantOrbits`} (N)
            Orbits to propagate.
        observers : `~adam_core.observers.observers.Observers` (M)
            Observers to generate ephemerides for.

        Returns
        -------
        ephemeris : `~adam_core.orbits.ephemeris.Ephemeris` (M)
            Ephemerides for each orbit as viewed from each observer.
        """
        # Convert orbits into PYOORB format
        orbits_pyoorb = self._configure_orbits(
            orbits.coordinates.values,
            orbits.coordinates.time.rescale("tt").mjd(),
            OpenOrbOrbitType.CARTESIAN,
            OpenOrbTimescale.TT,
            magnitude=None,
            slope=None,
        )

        # Check if observation times are defined in UTC
        if observers.coordinates.time.scale != "utc":
            observers_utc = Observers.from_kwargs(
                code=observers.code,
                coordinates=CartesianCoordinates.from_kwargs(
                    x=observers.coordinates.x,
                    y=observers.coordinates.y,
                    z=observers.coordinates.z,
                    vx=observers.coordinates.vx,
                    vy=observers.coordinates.vy,
                    vz=observers.coordinates.vz,
                    covariance=observers.coordinates.covariance,
                    time=observers.coordinates.time.rescale("utc"),
                    origin=observers.coordinates.origin,
                    frame=observers.coordinates.frame,
                ),
            )
        else:
            observers_utc = observers

        ephemeris_list = []
        # Iterate over unique observatory codes and their times
        for code_i, observer_i in observers_utc.iterate_codes():
            # Extract obervation times
            mjd_utc = observer_i.coordinates.time.mjd()

            # Convert epochs into PYOORB format (we want UTC as output)
            epochs_pyoorb = self._configure_epochs(mjd_utc, OpenOrbTimescale.UTC)

            # Generate ephemeris
            ephemeris_array_3D, err = oo.pyoorb.oorb_ephemeris_full(
                in_orbits=orbits_pyoorb,
                in_obscode=code_i,
                in_date_ephems=epochs_pyoorb,
                in_dynmodel=self.dynamical_model,
            )

            if err != 0:
                raise RuntimeError(
                    f"PYOORB ephemeris generation failed with error code {err}."
                )

            # Stack 3D ephemeris into single 2D array
            ephemeris_array = np.vstack(ephemeris_array_3D)

            # PYOORB returns ephemerides for each orbit, so lets reconstruct orbit IDs
            ids = np.arange(0, len(orbits))
            orbit_ids_idx = np.repeat(ids, len(mjd_utc))

            # Columns returned by PYOORB, we will only use a subset
            # columns = [
            #     "mjd_utc",
            #     "RA_deg",
            #     "Dec_deg",
            #     "vRAcosDec",
            #     "vDec",
            #     "PhaseAngle_deg",
            #     "SolarElon_deg",
            #     "r_au",
            #     "delta_au",
            #     "VMag",
            #     "PosAngle_deg",
            #     "TLon_deg",
            #     "TLat_deg",
            #     "TOCLon_deg",
            #     "TOCLat_deg",
            #     "HLon_deg",
            #     "HLat_deg",
            #     "HOCLon_deg",
            #     "HOCLat_deg",
            #     "Alt_deg",
            #     "SolarAlt_deg",
            #     "LunarAlt_deg",
            #     "LunarPhase",
            #     "LunarElon_deg",
            #     "obj_x",
            #     "obj_y",
            #     "obj_z",
            #     "obj_vx",
            #     "obj_vy",
            #     "obj_vz",
            #     "obs_x",
            #     "obs_y",
            #     "obs_z",
            #     "TrueAnom",
            # ]
            codes = np.empty(len(ephemeris_array), dtype=object)
            codes[:] = code_i

            # Check to make sure the desired times are within 1 microsecond
            # of the times returned by PYOORB
            _assert_times_almost_equal(
                np.tile(mjd_utc, len(orbits)),
                ephemeris_array[:, 0],
                tolerance=0.001,  # FIXME: This is 0.001 days, which is 86.4 seconds, not 1 microsecond?
            )

            if isinstance(orbits, Orbits):
                # Map the object and orbit IDs back to the input arrays
                orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)[
                    orbit_ids_idx
                ]
                object_ids = orbits.object_id.to_numpy(zero_copy_only=False)[
                    orbit_ids_idx
                ]

                ephemeris = Ephemeris.from_kwargs(
                    orbit_id=orbit_ids,
                    object_id=object_ids,
                    coordinates=SphericalCoordinates.from_kwargs(
                        time=Timestamp.from_mjd(
                            ephemeris_array[:, 0],
                            scale="utc",
                        ),
                        rho=None,  # PYOORB rho (delta_au) is geocentric not topocentric
                        lon=ephemeris_array[:, 1],
                        lat=ephemeris_array[:, 2],
                        vlon=ephemeris_array[:, 3]
                        / np.cos(np.radians(ephemeris_array[:, 2])),
                        vlat=ephemeris_array[:, 4],
                        vrho=None,  # PYOORB doesn't calculate observer velocity so it can't calulate vrho
                        origin=Origin.from_kwargs(code=codes),
                        frame="equatorial",
                    ),
                    alpha=ephemeris_array[:, 5],
                    aberrated_coordinates=CartesianCoordinates.from_kwargs(
                        x=ephemeris_array[:, 24],
                        y=ephemeris_array[:, 25],
                        z=ephemeris_array[:, 26],
                        vx=ephemeris_array[:, 27],
                        vy=ephemeris_array[:, 28],
                        vz=ephemeris_array[:, 29],
                        time=Timestamp.from_mjd(
                            ephemeris_array[:, 0],
                            scale="utc",
                        ),
                        origin=Origin.from_kwargs(
                            code=["SUN" for i in range(len(ephemeris_array))]
                        ),
                        frame="ecliptic",
                    ),
                )

                ephemeris_list.append(ephemeris)

            elif isinstance(orbits, VariantOrbits):
                # Map the object and orbit IDs back to the input arrays
                object_ids = orbits.object_id.to_numpy(zero_copy_only=False)[
                    orbit_ids_idx
                ]
                orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)[
                    orbit_ids_idx
                ]
                weights = orbits.weights.to_numpy()[orbit_ids_idx]
                weights_cov = orbits.weights_cov.to_numpy()[orbit_ids_idx]

                variant_ephemeris = VariantEphemeris.from_kwargs(
                    orbit_id=orbit_ids,
                    object_id=object_ids,
                    coordinates=SphericalCoordinates.from_kwargs(
                        time=Timestamp.from_mjd(
                            ephemeris_array[:, 0],
                            scale="utc",
                        ),
                        rho=None,  # PYOORB rho (delta_au) is geocentric not topocentric
                        lon=ephemeris_array[:, 1],
                        lat=ephemeris_array[:, 2],
                        vlon=ephemeris_array[:, 3]
                        / np.cos(np.radians(ephemeris_array[:, 2])),
                        vlat=ephemeris_array[:, 4],
                        vrho=None,  # PYOORB doesn't calculate observer velocity so it can't calulate vrho
                        origin=Origin.from_kwargs(code=codes),
                        frame="equatorial",
                    ),
                    weights=weights,
                    weights_cov=weights_cov,
                )

                ephemeris_list.append(variant_ephemeris)

        return qv.concatenate(ephemeris_list)
