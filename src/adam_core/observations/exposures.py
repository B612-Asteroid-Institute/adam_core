from __future__ import annotations

from typing import Iterator, Literal

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from quivr.validators import and_, ge, le

from ..coordinates.origin import OriginCodes
from ..observers import observers, state
from ..time import Timestamp


class Exposures(qv.Table):
    """
    Exposures is a table of data about exposures that provide point source observations.
    """

    id = qv.LargeStringColumn()
    start_time = Timestamp.as_column()
    duration = qv.Float64Column(validator=and_(ge(0), le(3600)))
    filter = qv.LargeStringColumn()
    observatory_code = qv.LargeStringColumn()
    seeing = qv.Float64Column(nullable=True)
    depth_5sigma = qv.Float64Column(nullable=True)

    def group_by_observatory_code(self) -> Iterator[tuple[str, Exposures]]:
        """
        Groups the exposures by observatory code.
        """
        unique_codes = self.observatory_code.unique()
        for code in unique_codes:
            mask = pc.equal(self.observatory_code, code)
            yield code, self.apply_mask(mask)

    def observers(
        self,
        frame: Literal["ecliptic", "equatorial"] = "ecliptic",
        origin: OriginCodes = OriginCodes.SUN,
    ) -> observers.Observers:
        """
        Return the observer location at each exposure midpoint.
        """

        # bunch of bookkeeping here to return states with the same
        # indexing as self
        coords = []
        indices = []
        codes = []

        unique_codes = self.observatory_code.unique().to_pylist()
        for code in unique_codes:
            mask = pc.equal(self.observatory_code, code)
            exposures_for_code = self.apply_mask(mask)
            indices_for_code = pc.indices_nonzero(mask)

            times_for_code = exposures_for_code.midpoint()
            coords_for_code = state.get_observer_state(
                code, times_for_code, frame=frame, origin=origin
            )

            coords.append(coords_for_code)
            indices.append(indices_for_code)
            codes.append(pa.array([code] * len(coords_for_code)))

        observers_table = observers.Observers.from_kwargs(
            code=pa.concat_arrays(codes),
            coordinates=qv.concatenate(coords),
        )

        indices = pa.concat_arrays(indices)
        original_ordering = pc.array_sort_indices(indices)
        return observers_table.take(original_ordering)

    def midpoint(self) -> Timestamp:
        """
        Returns the midpoint of the exposure.
        """
        return self.start_time.add_seconds(pc.divide(self.duration, 2.0))
