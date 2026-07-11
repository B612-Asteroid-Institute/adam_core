from __future__ import annotations

from typing import Iterator

import numpy as np
import numpy.typing as npt
import quivr as qv
from quivr.validators import and_, ge, le

from ..time import Timestamp
from .exposures import Exposures


class PointSourceDetections(qv.Table):
    """
    PointSourceDetections is a table of data about point source detections.

    """

    id = qv.LargeStringColumn()
    exposure_id = qv.LargeStringColumn(nullable=True)
    time = Timestamp.as_column()

    ra = qv.Float64Column(validator=and_(ge(0), le(360)))
    ra_sigma = qv.Float64Column(nullable=True)

    dec = qv.Float64Column(validator=and_(ge(-90), le(90)))
    dec_sigma = qv.Float64Column(nullable=True)

    mag = qv.Float64Column(nullable=True, validator=and_(ge(-10), le(30)))
    mag_sigma = qv.Float64Column(nullable=True)

    def group_by_exposure(self) -> Iterator["PointSourceDetections"]:
        """
        Returns an iterator of PointSourceDetections, each grouped by exposure_id.
        """
        from adam_core import _rust_native

        from .arrow_bridge import observations_from_ipc, observations_to_ipc

        # One Rust crossing owns the grouping: unique exposure IDs in
        # first-appearance order; a null unique ID yields an empty group
        # exactly like the legacy null-equality mask.
        for raw in _rust_native.detection_exposure_groups_ipc(
            observations_to_ipc(self)
        ):
            yield observations_from_ipc(raw, PointSourceDetections)

    def healpixels(self, nside: int, nest: bool = True) -> npt.NDArray[np.int64]:
        """
        Returns an array of healpixels for each observation.
        """
        from adam_core import _rust_native

        return _rust_native.detections_healpixels_numpy(
            self.ra.to_numpy(zero_copy_only=False),
            self.dec.to_numpy(zero_copy_only=False),
            nside,
            nest,
        )

    def group_by_healpixel(
        self, nside: int, nest: bool = True
    ) -> Iterator[tuple[int, PointSourceDetections]]:
        """
        Returns an iterator of PointSourceDetections, each grouped by healpixel.
        """
        from adam_core import _rust_native

        from .arrow_bridge import observations_from_ipc, observations_to_ipc

        # One Rust crossing owns pixel assignment and ascending-pixel
        # grouping; the yielded key stays a numpy int64 scalar like the
        # legacy ``np.unique`` iteration.
        for pixel, raw in _rust_native.detection_healpixel_groups_ipc(
            observations_to_ipc(self), nside, nest
        ):
            yield (np.int64(pixel), observations_from_ipc(raw, PointSourceDetections))

    def link_to_exposures(
        self, exposures: Exposures
    ) -> qv.Linkage[PointSourceDetections, Exposures]:
        """
        Links the detections to the exposures.
        """
        return qv.Linkage(self, exposures, self.exposure_id, exposures.id)
