import warnings
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
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
            codes = pa.array(codes, type=pa.large_string())

        # Create a table with the codes and times and add
        # and index column to track the original order
        table = pa.Table.from_pydict(
            {
                "index": pa.array(range(len(codes)), type=pa.uint64()),
                "code": codes,
                "times.days": times.days,
                "times.nanos": times.nanos,
            }
        )

        # Expected observers schema with the addition of a
        # column that tracks the original index
        observers_schema = pa.schema(
            [
                pa.field("code", pa.large_string(), nullable=False),
                pa.field(
                    "coordinates",
                    pa.struct(
                        [
                            pa.field("x", pa.float64()),
                            pa.field("y", pa.float64()),
                            pa.field("z", pa.float64()),
                            pa.field("vx", pa.float64()),
                            pa.field("vy", pa.float64()),
                            pa.field("vz", pa.float64()),
                            pa.field(
                                "time",
                                pa.struct(
                                    [
                                        pa.field("days", pa.int64()),
                                        pa.field("nanos", pa.int64()),
                                    ]
                                ),
                            ),
                            pa.field(
                                "covariance",
                                pa.struct(
                                    [pa.field("values", pa.large_list(pa.float64()))]
                                ),
                            ),
                            pa.field(
                                "origin",
                                pa.struct([pa.field("code", pa.large_string())]),
                            ),
                        ]
                    ),
                ),
                pa.field("index", pa.uint64()),
            ],
            metadata={
                "coordinates.time.scale": times.scale,
                "coordinates.frame": "ecliptic",
            },
        )

        # Create an empty table with the expected schema
        observers_table = observers_schema.empty_table()

        # Loop through each unique code and calculate the observer's
        # state for each time (these can be non-unique as cls.from_code
        # will handle this)
        for code in table["code"].unique():

            times_code = table.filter(pc.equal(table["code"], code))

            observers = cls.from_code(
                code.as_py(),
                Timestamp.from_kwargs(
                    days=times_code["times.days"],
                    nanos=times_code["times.nanos"],
                    scale=times.scale,
                ),
            )

            observers_table_i = observers.table.append_column(
                "index", times_code["index"]
            )
            observers_table = pa.concat_tables(
                [observers_table, observers_table_i]
            ).combine_chunks()

        observers_table = observers_table.sort_by(("index")).drop_columns(["index"])
        return cls.from_pyarrow(observers_table)

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
