import healpy

from ...time import Timestamp
from ..detections import PointSourceDetections
from ..exposures import Exposures


def test_detections_link_to_exposures():
    start_times = Timestamp.from_iso8601(
        [
            "2000-01-01T00:00:00",
            "2000-01-02T00:00:00",
        ],
        scale="utc",
    )
    exp = Exposures.from_kwargs(
        id=["e1", "e2"],
        start_time=start_times,
        duration=[60, 30],
        filter=["g", "r"],
        observatory_code=["I41", "I41"],
    )

    det = PointSourceDetections.from_kwargs(
        id=["d1", "d2", "d3", "d4", "d5"],
        exposure_id=["e1", "e1", "e2", "e1", "e2"],
        time=None,
        ra=[0, 1, 2, 3, 4],
        dec=[5, 6, 7, 8, 9],
        mag=[10, 11, 12, 13, 14],
    )

    link = det.link_to_exposures(exp)

    # one link per unique exposure id
    assert len(link) == 2

    have_det, have_exp = link.select("e1")
    assert len(have_exp) == 1
    assert len(have_det) == 3
    assert have_det.id.to_pylist() == ["d1", "d2", "d4"]

    have_det, have_exp = link.select("e2")
    assert len(have_exp) == 1
    assert len(have_det) == 2
    assert have_det.id.to_pylist() == ["d3", "d5"]


def test_detection_healpixels():
    ra, dec = healpy.pixelfunc.pix2ang(
        nside=16, ipix=[1, 2, 3, 4, 5], nest=True, lonlat=True
    )
    det = PointSourceDetections.from_kwargs(
        id=["d1", "d2", "d3", "d4", "d5"],
        exposure_id=["e1", "e1", "e2", "e1", "e2"],
        time=None,
        ra=ra,
        dec=dec,
        mag=[10, 11, 12, 13, 14],
    )

    healpixels = det.healpixels(nside=16, nest=True)

    assert len(healpixels) == 5
    assert healpixels.tolist() == [1, 2, 3, 4, 5]


def test_detection_group_by_healpixel():
    healpixels = [3, 1, 1, 2, 1, 3, 1, 2]
    ids = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"]
    ra, dec = healpy.pixelfunc.pix2ang(
        nside=16, ipix=healpixels, nest=True, lonlat=True
    )
    det = PointSourceDetections.from_kwargs(
        id=ids,
        exposure_id=[""] * 8,
        ra=ra,
        dec=dec,
        mag=[0] * 8,
    )

    groups = dict(det.group_by_healpixel(nside=16, nest=True))
    assert len(groups) == 3

    assert 1 in groups
    assert len(groups[1]) == 4
    assert groups[1].id.to_pylist() == ["d2", "d3", "d5", "d7"]

    assert 2 in groups
    assert len(groups[2]) == 2
    assert groups[2].id.to_pylist() == ["d4", "d8"]

    assert 3 in groups
    assert len(groups[3]) == 2
    assert groups[3].id.to_pylist() == ["d1", "d6"]


def test_detections_group_by_exposure():
    detections = PointSourceDetections.from_kwargs(
        id=["d1", "d2", "d3", "d4", "d5"],
        exposure_id=["e1", "e1", "e2", "e1", "e2"],
        time=None,
        ra=[0, 1, 2, 3, 4],
        dec=[5, 6, 7, 8, 9],
        mag=[10, 11, 12, 13, 14],
    )

    groups = list(detections.group_by_exposure())
    assert len(groups) == 2

    assert groups[0].id.to_pylist() == ["d1", "d2", "d4"]
    assert groups[0].exposure_id.to_pylist() == ["e1", "e1", "e1"]
    assert groups[0].ra.to_pylist() == [0, 1, 3]
    assert groups[0].dec.to_pylist() == [5, 6, 8]

    assert groups[1].id.to_pylist() == ["d3", "d5"]
    assert groups[1].exposure_id.to_pylist() == ["e2", "e2"]
    assert groups[1].ra.to_pylist() == [2, 4]
    assert groups[1].dec.to_pylist() == [7, 9]
