from __future__ import annotations

from typing import Iterator

import pyarrow.compute as pc
import quivr as qv

from .detections import PointSourceDetections


class Associations(qv.Table):

    detection_id = qv.StringColumn()
    object_id = qv.StringColumn(nullable=True)

    # TODO: We may want to create a derivative class called "ProbabilisticAssociations" that
    # includes residuals with respect to an orbit
    # orbit_id = qv.StringColumn(nullable=True)
    # residuals = Residuals.as_column(nullable=True) # from adam_core.coordinates.residuals import Residuals

    def group_by_object(self) -> Iterator["Associations"]:
        """
        Returns an iterator of PointSourceDetections, each grouped by exposure_id.
        """
        # Gather unique exposure IDs
        object_ids = self.object_id.unique()
        sorted = self.table.sort_by("object_id")
        for object_id in object_ids:
            mask = pc.equal(sorted.column("object_id"), object_id)
            table = sorted.filter(mask)
            yield Associations(table)

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
