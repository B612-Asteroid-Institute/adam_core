from __future__ import annotations

from typing import Iterator, Literal

import pyarrow as pa
import quivr as qv
from quivr.validators import and_, ge, le

from ..coordinates.origin import OriginCodes
from ..observers import observers
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
        from adam_core import _rust_native

        from .arrow_bridge import observations_from_ipc, observations_to_ipc

        for code, raw in _rust_native.exposure_groups_ipc(observations_to_ipc(self)):
            yield code, observations_from_ipc(raw, Exposures)

    def observers(
        self,
        frame: Literal["ecliptic", "equatorial", "itrf93"] = "ecliptic",
        origin: OriginCodes = OriginCodes.SUN,
    ) -> observers.Observers:
        """
        Return the observer location at each exposure midpoint.
        """

        from .._rust.arrow import ensure_spice_backend, table_from_record_batch

        time_table = self.start_time.table.combine_chunks()
        exposure_table = self.table.combine_chunks()
        batch = pa.RecordBatch.from_arrays(
            [
                exposure_table.column("observatory_code")
                .chunk(0)
                .cast(pa.large_string()),
                time_table.column("days").chunk(0),
                time_table.column("nanos").chunk(0),
                exposure_table.column("duration").chunk(0),
            ],
            names=["code", "days", "nanos", "duration"],
        )
        result = ensure_spice_backend().observer_states_from_exposures_arrow(
            batch, self.start_time.scale, frame, origin.name
        )
        return table_from_record_batch(observers.Observers, result)

    def midpoint(self) -> Timestamp:
        """
        Returns the midpoint of the exposure.
        """
        from adam_core import _rust_native

        from .arrow_bridge import observations_to_ipc

        days, nanos = _rust_native.exposure_midpoint_ipc(observations_to_ipc(self))
        return Timestamp.from_kwargs(
            days=days, nanos=nanos, scale=self.start_time.scale
        )
