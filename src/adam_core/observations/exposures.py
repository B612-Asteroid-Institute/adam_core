from __future__ import annotations

from typing import Iterator, Literal

import pyarrow as pa
import pyarrow.compute as pc
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
            # Legacy iterated ``observatory_code.unique()`` and therefore
            # yielded pyarrow large_string scalars; preserve that type.
            yield pa.scalar(code, type=pa.large_string()), observations_from_ipc(
                raw, Exposures
            )

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
        # Only SPICE coverage errors (space-based / unknown observatory codes
        # the Rust ground table cannot serve) fall back to the legacy per-code
        # assembly, mirroring Observers.from_codes.
        try:
            result = ensure_spice_backend().observer_states_from_exposures_arrow(
                batch, self.start_time.scale, frame, origin.name
            )
        except (RuntimeError, ValueError):
            return self._observers_legacy(frame=frame, origin=origin)
        return table_from_record_batch(observers.Observers, result)

    def _observers_legacy(
        self,
        frame: Literal["ecliptic", "equatorial", "itrf93"] = "ecliptic",
        origin: OriginCodes = OriginCodes.SUN,
    ) -> observers.Observers:
        """Legacy per-code assembly: used when a request includes codes the
        Rust ground-site table cannot serve (special space observatories,
        unknown codes). Preserves exact legacy semantics and errors."""
        from ..observers import state

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
        from adam_core import _rust_native

        from .arrow_bridge import observations_to_ipc

        days, nanos = _rust_native.exposure_midpoint_ipc(observations_to_ipc(self))
        return Timestamp.from_kwargs(
            days=days, nanos=nanos, scale=self.start_time.scale
        )
