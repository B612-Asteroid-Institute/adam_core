from __future__ import annotations

from typing import Iterator, Literal

import astropy.time
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from quivr.validators import and_, ge, le

from ..coordinates import cartesian, origin, times
from ..observers import state


class Exposures(qv.Table):
    """
    Exposures is a table of data about exposures that provide point source observations.
    """

    id = qv.StringColumn()
    start_time = times.Times.as_column()
    duration = qv.Float64Column(validator=and_(ge(0), le(3600)))
    filter = qv.DictionaryColumn(index_type=pa.uint32(), value_type=pa.string())

    observatory_code = qv.StringColumn()

    def group_by_observatory_code(self) -> Iterator[tuple[str, Exposures]]:
        """
        Groups the exposures by observatory code.
        """
        unique_codes = self.observatory_code.unique()
        for code in unique_codes:
            mask = pc.equal(self.observatory_code, code)
            yield code, self.apply_mask(mask)

    def observer_states(
        self,
        frame: Literal["ecliptic", "equatorial"] = "ecliptic",
        origin: origin.OriginCodes = origin.OriginCodes.SUN,
    ) -> cartesian.CartesianCoordinates:
        """
        Return the observer location at each exposure midpoint.
        """

        # bunch of bookkeeping here to return states with the same
        # indexing as self
        states = []
        indices = []

        unique_codes = self.observatory_code.unique().to_pylist()
        for code in unique_codes:
            mask = pc.equal(self.observatory_code, code)
            exposures_for_code = self.apply_mask(mask)
            indices_for_code = pc.indices_nonzero(mask)

            times_for_code = exposures_for_code.midpoint().to_astropy()
            states_for_code = state.get_observer_state(
                code, times_for_code, frame=frame, origin=origin
            )

            states.append(states_for_code)
            indices.append(indices_for_code)

        states_table = qv.concatenate(states)
        indices = pa.concat_arrays(indices)
        original_ordering = pc.array_sort_indices(indices)
        return states_table.take(original_ordering)

    def midpoint(self) -> times.Times:
        """
        Returns the midpoint of the exposure.
        """
        delta = astropy.time.TimeDelta(self.duration.to_numpy() / 2, format="sec")
        return self.start_time.add(delta)
