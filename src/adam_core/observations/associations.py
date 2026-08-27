from __future__ import annotations

from typing import Iterator

import quivr as qv

from .detections import PointSourceDetections


class Associations(qv.Table):

    detection_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)

    # TODO: We may want to create a derivative class called "ProbabilisticAssociations" that
    # includes residuals with respect to an orbit
    # orbit_id = qv.LargeStringColumn(nullable=True)
    # residuals = Residuals.as_column(nullable=True) # from adam_core.coordinates.residuals import Residuals

    def group_by_object(self) -> Iterator["Associations"]:
        """
        Group the associations by object ID. Any null object IDs will be returned last.

        Returns
        -------
        associations : Iterator[`~adam_core.observations.associations.Associations`]
            Associations grouped by object ID.
        """
        from adam_core import _rust_native

        from .arrow_bridge import observations_from_ipc, observations_to_ipc

        # One Rust crossing owns the grouping: non-null object IDs in
        # first-appearance order, then any null group last.
        for raw in _rust_native.association_object_groups_ipc(
            observations_to_ipc(self)
        ):
            yield observations_from_ipc(raw, Associations)

    def link_to_detections(
        self, detections: PointSourceDetections
    ) -> qv.Linkage[Associations, PointSourceDetections]:
        """
        Link the associations to a table of detections.

        Parameters
        ----------
        detections : `~adam_core.observations.detections.PointSourceDetections`
            Table of detections to link to.
        """
        # NOTE: Unsure if this is actually useful since its 1 to 1
        return qv.Linkage(self, detections, self.detection_id, detections.id)
