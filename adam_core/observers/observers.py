import warnings
from typing import Union

import pandas as pd
import quivr as qv
from mpc_obscodes import mpc_obscodes
from typing_extensions import Self

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import OriginCodes
from ..time import Timestamp


class ObservatoryGeodetics(qv.Table):
    code = qv.LargeStringColumn()
    longitude = qv.Float64Column()
    cos_phi = qv.Float64Column()
    sin_phi = qv.Float64Column()
    name = qv.LargeStringColumn()


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

OBSERVATORY_GEODETICS = ObservatoryGeodetics.from_kwargs(
    code=OBSCODES["code"].values,
    longitude=OBSCODES["Longitude"].values,
    cos_phi=OBSCODES["cos"].values,
    sin_phi=OBSCODES["sin"].values,
    name=OBSCODES["Name"].values,
)

OBSERVATORY_CODES = {
    x for x in OBSERVATORY_GEODETICS.code.to_numpy(zero_copy_only=False)
}


class Observers(qv.Table):
    code = qv.LargeStringColumn(nullable=False)
    coordinates = CartesianCoordinates.as_column()

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
