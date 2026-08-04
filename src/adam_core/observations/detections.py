from __future__ import annotations

from typing import Iterator

import healpy
import numpy as np
import numpy.typing as npt
import pyarrow
import pyarrow.compute
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
        # Gather unique exposure IDs
        exposure_ids = self.exposure_id.unique()
        sorted = self.table.sort_by("exposure_id")
        for exposure_id in exposure_ids:
            mask = pyarrow.compute.equal(sorted.column("exposure_id"), exposure_id)
            table = sorted.filter(mask)
            yield PointSourceDetections(table)

    def healpixels(self, nside: int, nest: bool = True) -> npt.NDArray[np.int64]:
        """
        Returns an array of healpixels for each observation.
        """
        return healpy.ang2pix(nside, self.ra, self.dec, nest=nest, lonlat=True)

    def group_by_healpixel(
        self, nside: int, nest: bool = True
    ) -> Iterator[tuple[int, PointSourceDetections]]:
        """
        Returns an iterator of PointSourceDetections, each grouped by healpixel.
        """
        # Gather unique healpixels
        healpixels = self.healpixels(nside, nest)
        unique_healpixels = np.unique(healpixels)
        for healpixel in unique_healpixels:
            mask = pyarrow.compute.equal(healpixels, healpixel)
            yield (healpixel, self.apply_mask(mask))

    def link_to_exposures(
        self, exposures: Exposures
    ) -> qv.Linkage[PointSourceDetections, Exposures]:
        """
        Links the detections to the exposures.
        """
        return qv.Linkage(self, exposures, self.exposure_id, exposures.id)
