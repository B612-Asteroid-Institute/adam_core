from ..associations import Associations
from ..detections import PointSourceDetections


def test_associations_link_to_detections():
    # Test that we can link associations to detections
    detections = PointSourceDetections.from_kwargs(
        id=["d1", "d2", "d3", "d4", "d5"],
        exposure_id=["e1", "e1", "e2", "e1", "e2"],
        time=None,
        ra=[0, 1, 2, 3, 4],
        dec=[5, 6, 7, 8, 9],
        mag=[10, 11, 12, 13, 14],
    )

    associations = Associations.from_kwargs(
        detection_id=["d1", "d2", "d3", "d4", "d5"],
        object_id=["o1", "o1", "o2", "o2", None],
    )

    link = associations.link_to_detections(detections)
    assert link.left_table == associations
    assert link.right_table == detections

    for i in range(5):
        associations_i, detetections_i = link.select(f"d{i + 1}")
        assert len(associations_i) == 1
        assert len(detetections_i) == 1


def test_associations_group_by_object():
    # Test that we can group associations by object
    associations = Associations.from_kwargs(
        detection_id=["d1", "d2", "d3", "d4", "d5"],
        object_id=["o1", "o1", "o2", "o2", None],
    )

    groups = list(associations.group_by_object())
    assert len(groups) == 3
    assert groups[0].object_id.to_pylist() == ["o1", "o1"]
    assert groups[0].detection_id.to_pylist() == ["d1", "d2"]

    assert groups[1].object_id.to_pylist() == ["o2", "o2"]
    assert groups[1].detection_id.to_pylist() == ["d3", "d4"]

    # What is the expected return here?
    assert groups[2].object_id.to_pylist() == [None]
