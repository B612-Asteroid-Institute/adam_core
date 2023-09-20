from typing import List

import pyarrow as pa
import pyarrow.compute as pc
from quivr import StringColumn, Table

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.spherical import SphericalCoordinates


class Ephemeris(Table):

    orbit_id = StringColumn()
    object_id = StringColumn(nullable=True)
    coordinates = SphericalCoordinates.as_column()

    # The coordinates as observed by the observer will be the result of
    # light emitted or reflected from the object at the time of the observation.
    # Light, however, has a finite speed and so the object's observed cooordinates
    # will be different from its actual geometric coordinates at the time of observation.
    # Aberrated coordinates are coordinates that account for the light travel time
    # from the time of emission/reflection to the time of observation
    aberrated_coordinates = CartesianCoordinates.as_column(nullable=True)

    def sort_by(
        self, by: List[str] = ["orbit_id", "time", "code"], ascending: bool = True
    ) -> "Ephemeris":
        """
        Sort the Ephemeris table the desired columns.
        Column options are "orbit_id", "object_id", "time", and "code".

        Parameters
        ----------
        by : List[str], optional
            The column(s) to sort by. Default is ["orbit_id", "time", "code"].
        ascending : bool, optional
            Whether to sort in ascending or descending order.

        Returns
        -------
        ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
            The sorted ephemeris table.

        Raises
        ------
        ValueError: If an invalid column is passed.
        """
        values = []
        names = []
        for col in by:
            if col == "orbit_id":
                values.append(self.orbit_id)
            elif col == "object_id":
                values.append(self.object_id)
            elif col == "time":
                values.append(self.coordinates.time.mjd())
            elif col == "code":
                values.append(self.coordinates.origin.code)
            else:
                raise ValueError(
                    f"Invalid column {col}. Valid columns are 'orbit_id', 'object_id', 'time' and 'code'"
                )

            names.append(col)

        table = pa.table(values, names=names)
        if ascending:
            order = [(name, "ascending") for name in names]
        else:
            order = [(name, "descending") for name in names]

        sort_indices = pc.sort_indices(table, order)
        return self.take(sort_indices)
