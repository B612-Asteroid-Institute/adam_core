from __future__ import annotations

from typing import Iterator

import pyarrow.compute as pc
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
        # Gather unique exposure IDs
        object_ids = self.object_id.unique()
        sorted = self.table.sort_by("object_id")

        # Return non-null object IDs first
        for object_id in pc.drop_null(object_ids):
            mask = pc.equal(sorted.column("object_id"), object_id)
            table = sorted.filter(mask)
            yield Associations.from_pyarrow(table)

        # If there are any null object IDs, return them last
        if object_ids.null_count > 0:
            mask = pc.is_null(sorted.column("object_id"))
            table = sorted.filter(mask)
            yield Associations.from_pyarrow(table)

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
