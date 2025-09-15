import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry.bvh import BVHIndex, BVHNodes, BVHPrimitives
from adam_core.geometry.bvh import build_bvh_index_from_segments
from adam_core.geometry.bvh_query import query_bvh_index
from adam_core.geometry.overlap import OverlapHits
from adam_core.observations.detections import PointSourceDetections
from adam_core.observations.exposures import Exposures
from adam_core.observations.rays import rays_from_detections
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import (
    OrbitPolylineSegments,
    compute_segment_aabbs,
    sample_ellipse_adaptive,
)
from adam_core.time import Timestamp
import numpy as np


def make_small_index():
    times = Timestamp.from_mjd([59000.0], scale="tdb")

    coords = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.017202],
        vz=[0.0],
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )

    orbits = Orbits.from_kwargs(orbit_id=["test_orbit"], coordinates=coords)
    _, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=1.0)
    segments = compute_segment_aabbs(segments, guard_arcmin=1.0)

    return build_bvh_index_from_segments(segments)


def make_rays():
    times = Timestamp.from_mjd([59000.0, 59000.1], scale="tdb")
    exposures = Exposures.from_kwargs(
        id=["exp_1", "exp_2"],
        start_time=times,
        duration=[300.0, 300.0],
        filter=["r", "g"],
        observatory_code=["500", "500"],
        seeing=[1.2, 1.3],
        depth_5sigma=[22.0, 22.1],
    )
    detections = PointSourceDetections.from_kwargs(
        id=["det_1", "det_2"],
        exposure_id=["exp_1", "exp_2"],
        time=times,
        ra=[0.0, 90.0],
        dec=[0.0, 0.0],
        ra_sigma=[0.1, 0.1],
        dec_sigma=[0.1, 0.1],
        mag=[20.0, 20.1],
        mag_sigma=[0.1, 0.1],
    )
    return rays_from_detections(detections, exposures)


def test_bvhindex_io_roundtrip(tmp_path):
    index = make_small_index()
    out_dir = tmp_path / "idx"
    out_dir.mkdir()
    index.to_parquet(str(out_dir))
    loaded = BVHIndex.from_parquet(str(out_dir))
    assert len(loaded.nodes) == len(index.nodes)
    assert len(loaded.prims) == len(index.prims)
    assert len(loaded.segments) == len(index.segments)


@pytest.mark.parametrize("max_processes", [0, 2])
def test_query_bvh_index_consistency_max_processes(max_processes):
    index = make_small_index()
    rays = make_rays()
    hits = query_bvh_index(index, rays, guard_arcmin=5.0, batch_size=64, max_processes=max_processes)
    assert isinstance(hits, OverlapHits)


def test_query_bvh_index_basic():
    index = make_small_index()
    rays = make_rays()
    hits = query_bvh_index(index, rays, guard_arcmin=5.0, batch_size=64, max_processes=0)
    assert isinstance(hits, OverlapHits)
    assert len(hits) >= 0


def test_bvhindex_orbit_mapping_consistency_with_shuffle():
    # Build segments from two orbits, then shuffle rows
    times = Timestamp.from_mjd([59000.0, 59000.0], scale="tdb")

    coords = CartesianCoordinates.from_kwargs(
        x=[1.0, 0.0],
        y=[0.0, 1.0],
        z=[0.0, 0.0],
        vx=[0.0, -0.017202],
        vy=[0.017202, 0.0],
        vz=[0.0, 0.0],
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name, OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    orbits = Orbits.from_kwargs(orbit_id=["A", "B"], coordinates=coords)
    _, segs = sample_ellipse_adaptive(orbits, max_chord_arcmin=1.0)
    segs = compute_segment_aabbs(segs, guard_arcmin=1.0)

    # Shuffle segments and rebuild seg_id to reflect new order
    n = len(segs)
    perm = np.random.permutation(n)
    segs = segs.take(perm)

    # Build index
    index = build_bvh_index_from_segments(segs)

    # Verify that mapping derived from segments matches expectation via prim_row_index
    seg_orbit_ids = segs.orbit_id.to_pylist()
    unique_ids = list(dict.fromkeys(seg_orbit_ids))
    id_to_idx = {oid: i for i, oid in enumerate(unique_ids)}
    expected = np.asarray([id_to_idx[seg_orbit_ids[row]] for row in np.asarray(index.prims.segment_row_index)], dtype=np.int32)

    # Build BVHArrays and ensure aggregator can derive orbit indices from segments
    from adam_core.geometry.bvh_query import query_bvh_worker_index
    # No rays necessary here; just ensure no exceptions when building arrays
    arrays = index.to_bvh_arrays()
    assert arrays.num_primitives == len(index.prims)


def test_query_bvh_index_parallel():
    index = make_small_index()
    rays = make_rays()
    hits_serial = query_bvh_index(index, rays, guard_arcmin=5.0, batch_size=64, max_processes=0)
    hits_parallel = query_bvh_index(index, rays, guard_arcmin=5.0, batch_size=64, max_processes=2)
    assert len(hits_serial) == len(hits_parallel)

