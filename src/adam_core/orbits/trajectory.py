"""Piecewise, validity-bounded trajectory of orbit states.

A :class:`Trajectory` is an ordered set of segments. Each segment is one
anchor :class:`~adam_core.orbits.orbits.Orbits` state plus a half-open coverage
window over which that state is authoritative. Computation is Rust-owned; this
module defines the compatible quivr schema and thin method veneers.
"""

from typing import Optional, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import quivr as qv

from ..time import Timestamp
from .orbits import Orbits


class Trajectory(qv.Table):
    """A set of validity-bounded orbit segments (rows are segments).

    Segments belonging to the same ``object_id`` must have non-overlapping
    coverage windows; gaps between them are allowed and are treated as
    "no valid state" (fail closed) rather than silently extrapolated.
    """

    object_id = qv.LargeStringColumn()
    segment_id = qv.LargeStringColumn()
    coverage_start = Timestamp.as_column()
    coverage_end = Timestamp.as_column()
    orbit = Orbits.as_column()
    source = qv.LargeStringColumn(nullable=True)
    source_version = qv.LargeStringColumn(nullable=True)
    max_propagation_days = qv.Float64Column(nullable=True)
    is_maneuver_boundary = qv.BooleanColumn(nullable=True)

    def _native_batch(self) -> pa.RecordBatch:
        table = self.table.combine_chunks()
        metadata = dict(table.schema.metadata or {})
        defaults = {
            b"coverage_start.scale": b"tai",
            b"coverage_end.scale": b"tai",
            b"orbit.coordinates.time.scale": b"tai",
            b"orbit.coordinates.frame": b"unspecified",
        }
        if any(key not in metadata for key in defaults):
            for key, value in defaults.items():
                metadata.setdefault(key, value)
            table = table.replace_schema_metadata(metadata)
        batches = table.to_batches()
        if batches:
            return batches[0]
        return pa.RecordBatch.from_arrays(
            [pa.array([], type=field.type) for field in table.schema],
            schema=table.schema,
        )

    def coverage_start_mjd(self) -> npt.NDArray[np.float64]:
        """Coverage-window start times as TDB MJD."""
        from adam_core import _rust_native  # type: ignore[attr-defined]

        return cast(
            npt.NDArray[np.float64],
            _rust_native.trajectory_mjd_arrow(self._native_batch(), "coverage_start"),
        )

    def coverage_end_mjd(self) -> npt.NDArray[np.float64]:
        """Coverage-window end times as TDB MJD."""
        from adam_core import _rust_native  # type: ignore[attr-defined]

        return cast(
            npt.NDArray[np.float64],
            _rust_native.trajectory_mjd_arrow(self._native_batch(), "coverage_end"),
        )

    def epoch_mjd(self) -> npt.NDArray[np.float64]:
        """Anchor-state epochs as TDB MJD."""
        from adam_core import _rust_native  # type: ignore[attr-defined]

        return cast(
            npt.NDArray[np.float64],
            _rust_native.trajectory_mjd_arrow(self._native_batch(), "epoch"),
        )

    def object_ids(self) -> list[str]:
        """Distinct object ids, in first-seen order."""
        from adam_core import _rust_native  # type: ignore[attr-defined]

        return cast(
            list[str],
            _rust_native.trajectory_object_ids_arrow(self._native_batch()),
        )

    def validate_coverage(self) -> "Trajectory":
        """Check structural invariants; raise ``ValueError`` on violation.

        Enforces, per object: ``coverage_end`` strictly after
        ``coverage_start``, each anchor epoch inside its own coverage window,
        and non-overlapping coverage windows (touching endpoints are allowed
        because windows are half-open ``[start, end)``).
        """
        from adam_core import _rust_native  # type: ignore[attr-defined]

        _rust_native.trajectory_validate_arrow(self._native_batch())
        return self

    def segment_for(
        self, time_mjd_tdb: float, object_id: Optional[str] = None
    ) -> Optional["Trajectory"]:
        """Return the single segment whose coverage window contains ``time``.

        Coverage windows are half-open ``[coverage_start, coverage_end)``.
        Returns ``None`` when no segment covers the time (a gap; fail closed).
        Raises ``ValueError`` when more than one segment matches (an invalid,
        overlapping trajectory) or when the trajectory holds multiple objects
        and ``object_id`` was not supplied.
        """
        from adam_core import _rust_native  # type: ignore[attr-defined]

        index = _rust_native.trajectory_segment_index_arrow(
            self._native_batch(), float(time_mjd_tdb), object_id
        )
        if index is None:
            return None
        return self.take(np.asarray([index], dtype=np.int64))
