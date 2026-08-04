"""Piecewise, validity-bounded trajectory of orbit states.

A :class:`Trajectory` is an ordered set of *segments*. Each segment is one
anchor :class:`~adam_core.orbits.orbits.Orbits` state (position, velocity and
6x6 covariance at a single epoch) plus a *coverage window* -- the time interval
over which that anchor state is the authoritative source for the object. This
mirrors the CCSDS OEM / SPICE notion of an ephemeris segment's useable
interval (see :mod:`adam_core.orbits.oem_io`).

This is the in-memory type a search consumes when an object's state is only
valid piecewise (e.g. spacecraft trajectories reconstructed from an SPK, where
solar-radiation-pressure and maneuvers make a single propagated state diverge
over time). For a given search span the caller binds the one segment whose
coverage window contains the span's reference time via :meth:`segment_for`, and
fails closed on coverage gaps and overlaps.

A conventional single-state orbit is simply a one-segment trajectory whose
coverage window spans the whole search, so callers that never build a
:class:`Trajectory` are unaffected.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
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
    # Coverage window [coverage_start, coverage_end) over which ``orbit`` is the
    # authoritative state for the object (OEM USEABLE_START/STOP semantics).
    coverage_start = Timestamp.as_column()
    coverage_end = Timestamp.as_column()
    # Anchor state + 6x6 covariance + epoch + frame + origin for this segment.
    orbit = Orbits.as_column()
    # Provenance / constraints.
    source = qv.LargeStringColumn(nullable=True)
    source_version = qv.LargeStringColumn(nullable=True)
    max_propagation_days = qv.Float64Column(nullable=True)
    is_maneuver_boundary = qv.BooleanColumn(nullable=True)

    def coverage_start_mjd(self) -> npt.NDArray[np.float64]:
        """Coverage-window start times as TDB MJD."""
        return np.asarray(
            self.coverage_start.rescale("tdb").mjd().to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )

    def coverage_end_mjd(self) -> npt.NDArray[np.float64]:
        """Coverage-window end times as TDB MJD."""
        return np.asarray(
            self.coverage_end.rescale("tdb").mjd().to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )

    def epoch_mjd(self) -> npt.NDArray[np.float64]:
        """Anchor-state epochs as TDB MJD."""
        return np.asarray(
            self.orbit.coordinates.time.rescale("tdb")
            .mjd()
            .to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )

    def object_ids(self) -> list[str]:
        """Distinct object ids, in first-seen order."""
        seen: dict[str, None] = {}
        for value in self.object_id.to_pylist():
            if value is not None:
                seen.setdefault(str(value), None)
        return list(seen)

    def validate_coverage(self) -> "Trajectory":
        """Check structural invariants; raise ``ValueError`` on violation.

        Enforces, per object: ``coverage_end`` strictly after
        ``coverage_start``, each anchor epoch inside its own coverage window,
        and non-overlapping coverage windows (touching endpoints are allowed
        because windows are half-open ``[start, end)``).
        """
        starts = self.coverage_start_mjd()
        ends = self.coverage_end_mjd()
        epochs = self.epoch_mjd()
        if bool(np.any(ends <= starts)):
            raise ValueError(
                "Trajectory coverage_end must be strictly after coverage_start"
            )
        if bool(np.any((epochs < starts) | (epochs > ends))):
            raise ValueError(
                "Trajectory segment orbit epoch must lie within its coverage window"
            )
        objects: npt.NDArray[np.object_] = np.asarray(
            self.object_id.to_pylist(), dtype=object
        )
        for obj in self.object_ids():
            mask = objects == obj
            order = np.argsort(starts[mask], kind="stable")
            s = starts[mask][order]
            e = ends[mask][order]
            if s.size > 1 and bool(np.any(s[1:] < e[:-1])):
                raise ValueError(
                    f"Trajectory has overlapping coverage windows for object {obj!r}"
                )
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
        traj: "Trajectory" = self
        if object_id is not None:
            objects: npt.NDArray[np.object_] = np.asarray(
                self.object_id.to_pylist(), dtype=object
            )
            traj = self.take(np.nonzero(objects == object_id)[0])
        elif len(traj.object_ids()) > 1:
            raise ValueError(
                "segment_for requires object_id when the trajectory holds "
                "multiple objects"
            )
        starts = traj.coverage_start_mjd()
        ends = traj.coverage_end_mjd()
        covers = (time_mjd_tdb >= starts) & (time_mjd_tdb < ends)
        idx = np.nonzero(covers)[0]
        if idx.size == 0:
            return None
        if idx.size > 1:
            raise ValueError(
                f"Overlapping coverage windows match t={time_mjd_tdb}; "
                "trajectory is invalid"
            )
        return traj.take(idx)
