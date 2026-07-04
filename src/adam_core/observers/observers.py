import warnings
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from mpc_obscodes import mpc_obscodes
from timezonefinder import TimezoneFinder
from typing_extensions import Self

from ..constants import Constants as c
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..time import Timestamp

R_EARTH_EQUATORIAL = c.R_EARTH_EQUATORIAL
R_EARTH_POLAR = c.R_EARTH_POLAR
E_EARTH = np.sqrt(1 - (R_EARTH_POLAR / R_EARTH_EQUATORIAL) ** 2)


class ObservatoryParallaxCoefficients(qv.Table):
    code = qv.LargeStringColumn()
    longitude = qv.Float64Column()
    cos_phi = qv.Float64Column()
    sin_phi = qv.Float64Column()
    name = qv.LargeStringColumn()

    def lon_lat(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Return the longitude and latitude of the observatories in degrees.

        This is only valid for Earth-based observatories.

        Returns
        -------
        longitude : np.ndarray
            The longitude of the observatories in degrees. In the range -180 to 180 degrees,
            with positive values east of the prime meridian.
        latitude : np.ndarray
            The latitude of the observatories in degrees. In the range -90 to 90 degrees,
            with positive values north of the equator.
        """
        # Filter out Space-based observatories
        mask = pc.is_nan(self.longitude).to_numpy(zero_copy_only=False)

        longitude = np.where(
            mask, np.nan, self.longitude.to_numpy(zero_copy_only=False)
        )
        tan_phi_geo = np.where(
            mask,
            np.nan,
            self.sin_phi.to_numpy(zero_copy_only=False)
            / self.cos_phi.to_numpy(zero_copy_only=False),
        )
        latitude_geodetic = np.arctan(tan_phi_geo / (1 - E_EARTH**2))

        # Scale longitude to -180 to 180
        longitude = np.where(longitude > 180, longitude - 360, longitude)

        return longitude, np.degrees(latitude_geodetic)

    def timezone(self) -> npt.NDArray[np.str_]:
        """
        Return the timezone of the observatories in hours.

        Returns
        -------
        timezone : np.ndarray
            The timezone of the observatories in hours.
        """
        tf = TimezoneFinder()
        lon, lat = self.lon_lat()
        time_zones = np.array(
            [
                tz if not np.isnan(lon_i) else "None"
                for lon_i, lat_i in zip(lon, lat)
                for tz in [tf.timezone_at(lng=lon_i, lat=lat_i)]
            ]
        )
        return time_zones


# Read MPC extended observatory codes file
# Ignore warning about pandas deprecation
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated",
    )
    OBSCODES = pd.read_json(
        mpc_obscodes,
        orient="index",
        dtype={"Longitude": float, "cos": float, "sin": float, "Name": str},
        encoding_errors="strict",
        precise_float=True,
    )
    OBSCODES.reset_index(inplace=True, names=["code"])

OBSERVATORY_PARALLAX_COEFFICIENTS = ObservatoryParallaxCoefficients.from_kwargs(
    code=OBSCODES["code"].values,
    longitude=OBSCODES["Longitude"].values,
    cos_phi=OBSCODES["cos"].values,
    sin_phi=OBSCODES["sin"].values,
    name=OBSCODES["Name"].values,
)

OBSERVATORY_CODES = {
    x for x in OBSERVATORY_PARALLAX_COEFFICIENTS.code.to_numpy(zero_copy_only=False)
}

_OBSCODES_LOADED_BACKENDS: set = set()


def _ensure_obscodes_loaded(backend) -> None:
    """Load the MPC observatory parallax table into the Rust SPICE backend
    once per backend instance (the backend caches the parsed sites)."""
    key = id(backend)
    if key in _OBSCODES_LOADED_BACKENDS:
        return
    with open(mpc_obscodes) as obscodes_file:
        backend.load_mpc_obscodes(obscodes_file.read())
    _OBSCODES_LOADED_BACKENDS.add(key)


class Observers(qv.Table):
    code = qv.LargeStringColumn(nullable=False)
    coordinates = CartesianCoordinates.as_column()

    @classmethod
    def from_codes(
        cls, codes: Union[list, npt.NDArray[np.str_], pa.Array], times: Timestamp
    ) -> Self:
        """
        Create an Observers table from a list of codes and times. The codes and times
        do not need to be unique and are assumed to belong to each other in an element-wise fashion.
        The observer state will be calculated  correctly matched to the input times and
        replicated for duplicate times.

        Parameters
        ----------
        codes : Union[list, npt.NDArray[np.str], pa.Array] (N)
            MPC observatory codes for which to find the states.
        times : Timestamp (N)
            Epochs for which to find the observatory locations.

        Returns
        -------
        observers : `~adam_core.observers.observers.Observers` (N)
            The observer and its state at each time.
        """
        if len(codes) != len(times):
            raise ValueError("codes and times must have the same length.")

        if not isinstance(codes, pa.Array):
            codes = pa.array([str(code) for code in codes], type=pa.large_string())

        # Single Rust crossing: obscodes lookup, DE440 Earth state, and the
        # ITRF93 ground-offset rotation all run in the spicekit backend
        # (bead personal-cmy.6). Requests containing codes the Rust ground
        # table cannot serve (special space observatories such as JWST, or
        # unknown codes) fall back to the legacy per-code assembly, which
        # either computes them through their dedicated paths or raises the
        # exact legacy error.
        from ..utils.spice import get_backend, setup_SPICE

        setup_SPICE()
        backend = get_backend()
        _ensure_obscodes_loaded(backend)
        encoded = pc.dictionary_encode(codes)
        unique_codes = [str(code) for code in encoded.dictionary.to_pylist()]
        code_indices = encoded.indices.to_numpy(zero_copy_only=False)
        try:
            values = np.asarray(
                backend.observer_states_from_codes(
                    unique_codes,
                    code_indices,
                    times.scale,
                    times.days.to_numpy(zero_copy_only=False),
                    times.nanos.to_numpy(zero_copy_only=False),
                    "ecliptic",
                    "SUN",
                ),
                dtype=np.float64,
            )
        except Exception:
            return cls._from_codes_legacy(codes, times)

        coordinates = CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=times,
            origin=Origin.from_kwargs(code=np.full(len(codes), "SUN", dtype="object")),
            frame="ecliptic",
        )
        return cls.from_kwargs(code=codes, coordinates=coordinates)

    @classmethod
    def _from_codes_legacy(cls, codes: pa.Array, times: Timestamp) -> Self:
        """Legacy per-code assembly: used when a request includes codes the
        Rust ground-site table cannot serve (special space observatories,
        unknown codes). Preserves exact legacy semantics and errors."""
        from .state import get_observer_state

        class IndexedObservers(qv.Table):
            index = qv.UInt64Column()
            observers = Observers.as_column()

        indexed_observers = IndexedObservers.empty()

        for code in pc.unique(codes):
            indices = pc.indices_nonzero(pc.equal(codes, code))
            times_code = times.take(indices)
            observers_i = cls.from_kwargs(
                code=[code.as_py()] * len(times_code),
                coordinates=get_observer_state(code.as_py(), times_code),
            )
            indexed_observers_i = IndexedObservers.from_kwargs(
                index=indices,
                observers=observers_i,
            )
            indexed_observers = qv.concatenate([indexed_observers, indexed_observers_i])
            if indexed_observers.fragmented():
                indexed_observers = qv.defragment(indexed_observers)

        return indexed_observers.sort_by("index").observers

    @classmethod
    def from_code(cls, code: Union[str, OriginCodes], times: Timestamp) -> Self:
        """
        Instantiate an Observers table with a single code and multiple times.
        Times do not need to be unique. The observer state will be calculated
        for each time and correctly matched to the input times and replicated for
        duplicate times.

        To load multiple codes, use `from_code` and then concatenate the tables.

        Note that NAIF origin codes may not be supported by `~adam_core.propagator.Propagator`
        classes such as PYOORB.

        Parameters
        ----------
        code : Union[str, OriginCodes]
            MPC observatory code or NAIF origin code for which to find the states.
        times : Timetamp (N)
            Epochs for which to find the observatory locations.

        Returns
        -------
        observers : `~adam_core.observers.observers.Observers` (N)
            The observer and its state at each time.

        Examples
        --------
        >>> import numpy as np
        >>> from adam_core.time import Timestamp
        >>> from adam_core.observers import Observers
        >>> times = Timestamp.from_mjd(np.arange(59000, 59000 + 100), scale="tdb")
        >>> observers = Observers.from_code("X05", times)
        """
        from .state import get_observer_state

        if isinstance(code, OriginCodes):
            code_str = code.name
        elif isinstance(code, str):
            code_str = code
        else:
            err = "Code should be a str or an `~adam_core.coordinates.origin.OriginCodes`."
            raise ValueError(err)

        if isinstance(code, str):
            # Route MPC observatory codes through the single-crossing Rust
            # path used by from_codes; NAIF OriginCodes keep the perturber
            # state path below.
            return cls.from_codes([code_str] * len(times), times)

        return cls.from_kwargs(
            code=[code_str] * len(times),
            coordinates=get_observer_state(code, times),
        )

    def iterate_codes(self):
        """
        Iterate over the codes in the Observers table.

        Yields
        ------
        code : str
            The code for observer.
        observers : `~adam_core.observers.observers.Observers`
            The Observers table for this observer.
        """
        for code in self.code.unique().sort():
            yield code.as_py(), self.select("code", code)
