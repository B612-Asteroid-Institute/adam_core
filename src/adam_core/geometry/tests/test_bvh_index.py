import os
import os as _os
from pathlib import Path

import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry.bvh.index import (
    BVHIndex,
    _fused_aabb_from_arrays_range,
    build_bvh_index,
    build_bvh_index_from_segments,
    get_leaf_primitives_numpy,
)
from adam_core.geometry.bvh.query import (
    calc_ray_segment_distance_and_guard,
)
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import (
    OrbitPolylineSegments,
    compute_segment_aabbs,
    sample_ellipse_adaptive,
)
from adam_core.time import Timestamp

# Grid infrastructure removed - using fixed optimal parameters


def _make_small_orbits(n: int = 3) -> Orbits:
    times = Timestamp.from_mjd([59000.0] * n, scale="tdb")
    x = np.linspace(0.9, 1.3, n)
    y = np.linspace(-0.1, 0.1, n)
    z = np.linspace(0.0, 0.05, n)
    vx = np.zeros(n)
    vy = np.linspace(0.015, 0.02, n)
    vz = np.zeros(n)
    coords = CartesianCoordinates.from_kwargs(
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(orbit_id=[f"o{i}" for i in range(n)], coordinates=coords)


# ---------------------------- Build from segments ----------------------------


def test_empty_index_from_segments():
    empty_segments = OrbitPolylineSegments.empty()
    index = build_bvh_index_from_segments(empty_segments)
    assert len(index.nodes) == 0
    assert len(index.prims) == 0


def test_build_bvh_index_from_segments_single():
    segments = OrbitPolylineSegments.from_kwargs(
        orbit_id=["test"],
        seg_id=[0],
        x0=[0.0],
        y0=[0.0],
        z0=[0.0],
        x1=[1.0],
        y1=[0.0],
        z1=[0.0],
        r_mid_au=[0.5],
        n_x=[0.0],
        n_y=[0.0],
        n_z=[1.0],
    )
    index = build_bvh_index_from_segments(segments)
    assert len(index.nodes) == 1
    assert len(index.prims) == 1
    assert int(index.nodes.left_child[0].as_py()) == -1
    assert int(index.nodes.right_child[0].as_py()) == -1


def test_build_bvh_index_missing_aabbs_no_longer_required():
    segments = OrbitPolylineSegments.from_kwargs(
        orbit_id=["t1"],
        seg_id=[0],
        x0=[0.0],
        y0=[0.0],
        z0=[0.0],
        x1=[1.0],
        y1=[0.0],
        z1=[0.0],
        r_mid_au=[0.5],
        n_x=[0.0],
        n_y=[0.0],
        n_z=[1.0],
    )
    # Should not raise; AABBs computed during build
    build_bvh_index_from_segments(segments)


def test_build_bvh_index_from_segments_multiple_structure(segments_aabbs):
    index = build_bvh_index_from_segments(segments_aabbs, max_leaf_size=4)
    assert len(index.nodes) > 0
    assert len(index.prims) == len(segments_aabbs)

    # All primitives accounted for across leaves
    leaf_mask = np.asarray(index.nodes.left_child) == -1
    counts = np.asarray(index.nodes.prim_count, dtype=np.int32)
    assert int(counts[leaf_mask].sum()) == len(segments_aabbs)

    # Parent AABB contains children
    left = np.asarray(index.nodes.left_child, dtype=np.int32)
    right = np.asarray(index.nodes.right_child, dtype=np.int32)
    nodes_min = np.column_stack(
        [
            np.asarray(index.nodes.nodes_min_x),
            np.asarray(index.nodes.nodes_min_y),
            np.asarray(index.nodes.nodes_min_z),
        ]
    )
    nodes_max = np.column_stack(
        [
            np.asarray(index.nodes.nodes_max_x),
            np.asarray(index.nodes.nodes_max_y),
            np.asarray(index.nodes.nodes_max_z),
        ]
    )
    for i in range(len(index.nodes)):
        if left[i] >= 0:
            assert np.all(nodes_min[i] <= nodes_min[left[i]] + 1e-10)
            assert np.all(nodes_max[i] >= nodes_max[left[i]] - 1e-10)
        if right[i] >= 0:
            assert np.all(nodes_min[i] <= nodes_min[right[i]] + 1e-10)
            assert np.all(nodes_max[i] >= nodes_max[right[i]] - 1e-10)


# ------------------------------- Node helpers --------------------------------


def test_nodes_helpers_min_max_and_is_leaf(segments_aabbs):
    index = build_bvh_index_from_segments(segments_aabbs, max_leaf_size=8)
    mn, mx = index.nodes.min_max_numpy()
    assert mn.shape == mx.shape == (len(index.nodes), 3)
    leaves = index.nodes.is_leaf_numpy()
    assert leaves.dtype == bool
    left = np.asarray(index.nodes.left_child)
    right = np.asarray(index.nodes.right_child)
    assert np.all(right[leaves] == -1)
    assert np.all((left == -1) == leaves)


def test_get_leaf_primitives_numpy_and_table(segments_aabbs):
    index = build_bvh_index_from_segments(segments_aabbs, max_leaf_size=8)
    leaves = np.nonzero(index.nodes.is_leaf_numpy())[0]
    rows, segs, leafs = get_leaf_primitives_numpy(index.nodes, index.prims, leaves)
    assert rows.dtype == np.int32 and segs.dtype == np.int32 and leafs.dtype == np.int32
    counts = np.asarray(index.nodes.prim_count, dtype=np.int32)
    expected = int(counts[leaves].sum())
    assert len(rows) == len(segs) == len(leafs) == expected


# ---------------------- Build from orbits: serial vs parallel -----------------
def test_build_bvh_index_from_orbits_serial_vs_parallel():
    orbits = _make_small_orbits(5)
    idx_serial = build_bvh_index(
        orbits,
        max_chord_arcmin=1.0,
        guard_arcmin=0.5,
        max_leaf_size=8,
        chunk_size_orbits=2,
        max_processes=0,
    )
    idx_parallel = build_bvh_index(
        orbits,
        max_chord_arcmin=1.0,
        guard_arcmin=0.5,
        max_leaf_size=8,
        chunk_size_orbits=2,
        max_processes=2,
    )

    # Fundamental checks
    assert len(idx_serial.segments) == len(idx_parallel.segments)
    assert len(idx_serial.prims) == len(idx_parallel.prims)
    assert len(idx_serial.nodes) > 0 and len(idx_parallel.nodes) > 0

    # Compare root AABB (order-invariant)
    if len(idx_serial.nodes) > 0 and len(idx_parallel.nodes) > 0:
        root_min_s = np.array(
            [
                idx_serial.nodes.nodes_min_x[0].as_py(),
                idx_serial.nodes.nodes_min_y[0].as_py(),
                idx_serial.nodes.nodes_min_z[0].as_py(),
            ]
        )
        root_max_s = np.array(
            [
                idx_serial.nodes.nodes_max_x[0].as_py(),
                idx_serial.nodes.nodes_max_y[0].as_py(),
                idx_serial.nodes.nodes_max_z[0].as_py(),
            ]
        )
        root_min_p = np.array(
            [
                idx_parallel.nodes.nodes_min_x[0].as_py(),
                idx_parallel.nodes.nodes_min_y[0].as_py(),
                idx_parallel.nodes.nodes_min_z[0].as_py(),
            ]
        )
        root_max_p = np.array(
            [
                idx_parallel.nodes.nodes_max_x[0].as_py(),
                idx_parallel.nodes.nodes_max_y[0].as_py(),
                idx_parallel.nodes.nodes_max_z[0].as_py(),
            ]
        )
        assert np.allclose(root_min_s, root_min_p)
        assert np.allclose(root_max_s, root_max_p)

    # Compare primitive sets (order can differ)
    seg_ids_s = set(np.asarray(idx_serial.prims.prim_seg_ids, dtype=np.int32).tolist())
    seg_ids_p = set(
        np.asarray(idx_parallel.prims.prim_seg_ids, dtype=np.int32).tolist()
    )
    assert seg_ids_s == seg_ids_p

    # Validate invariants
    idx_serial.validate()
    idx_parallel.validate()


def test_bvhindex_io_roundtrip(tmp_path, index_small):
    out_dir = tmp_path / "idx"
    out_dir.mkdir()
    index_small.to_parquet(str(out_dir))
    loaded = BVHIndex.from_parquet(str(out_dir))
    assert len(loaded.nodes) == len(index_small.nodes)
    assert len(loaded.prims) == len(index_small.prims)
    assert len(loaded.segments) == len(index_small.segments)


def test_prim_mapping_matches_segments(index_small):
    seg_rows = index_small.prims.segment_row_index.to_numpy()
    seg_ids = index_small.prims.prim_seg_ids.to_numpy()
    # seg_id at each row index matches prim_seg_ids
    seg_ids_from_segments = index_small.segments.seg_id.to_numpy()[seg_rows]
    assert np.array_equal(seg_ids_from_segments, seg_ids)


def test_nodes_attributes(index_small):
    # build_max_leaf_size is set and positive
    assert int(index_small.nodes.build_max_leaf_size) > 0
    # bvh_max_depth positive when index has nodes
    if len(index_small.nodes) > 0:
        assert int(index_small.nodes.bvh_max_depth) > 0


def generate_pairs_window(W: int = 4096, seed: int = 42):
    """Generate flattened pairs window (ray/segment arrays) for JAX pairwise kernel."""
    rng = np.random.RandomState(seed)
    ro = rng.uniform(-5, 5, (W, 3))
    rd = rng.normal(0, 1, (W, 3))
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    s0 = rng.uniform(-10, 10, (W, 3))
    s1 = rng.uniform(-10, 10, (W, 3))
    return ro, rd, s0, s1


# Indexing related benchmarks
@pytest.mark.benchmark
@pytest.mark.parametrize("theta_guard", [0.5, 2.0])
@pytest.mark.parametrize("W", [1024, 4096, 8192])
def test_distances_and_guard_pairs_jax(benchmark, theta_guard, W):
    ro, rd, s0, s1 = generate_pairs_window(W=W)
    result = benchmark(
        lambda: calc_ray_segment_distance_and_guard(
            ro, rd, s0, s1, 1.0, 1.0, theta_guard
        )
    )
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("max_chord_arcmin", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("max_segments_per_orbit", [8192, 65536])
def test_sample_ellipse_adaptive(
    benchmark, orbits_synthetic_stratified_ci, max_chord_arcmin, max_segments_per_orbit
):
    orbits_plane_params, orbits_polyline_segments = benchmark(
        lambda: sample_ellipse_adaptive(
            orbits_synthetic_stratified_ci,
            max_chord_arcmin=max_chord_arcmin,
            max_segments_per_orbit=max_segments_per_orbit,
        )
    )
    assert orbits_plane_params is not None
    assert orbits_polyline_segments is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("guard_arcmin", [0.1, 1.0])
@pytest.mark.parametrize("epsilon_n_au", [1e-8, 1e-4])
@pytest.mark.parametrize("padding_method", ["baseline", "sagitta_guard"])
def test_compute_segment_aabbs(
    benchmark, segments_aabbs, guard_arcmin, epsilon_n_au, padding_method
):
    result = benchmark(
        lambda: compute_segment_aabbs(
            segments_aabbs,
            guard_arcmin=guard_arcmin,
            epsilon_n_au=epsilon_n_au,
            padding_method=padding_method,
        )
    )
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("max_leaf_size", [32, 64, 128, 256])
def test_build_bvh_index_from_segments(benchmark, segments_aabbs, max_leaf_size):
    result = benchmark(
        lambda: build_bvh_index_from_segments(
            segments_aabbs, max_leaf_size=max_leaf_size
        )
    )


def test_build_bvh_index_e2e(orbits_synthetic_stratified_ci):
    index = build_bvh_index(
        orbits_synthetic_stratified_ci,
        max_chord_arcmin=5.0,
        guard_arcmin=0.65,
        max_leaf_size=64,
        max_processes=1,
        max_segments_per_orbit=512,
        epsilon_n_au=1e-9,
        padding_method="baseline",
    )
    index.validate()


@pytest.mark.benchmark
def test_build_bvh_index_benchmark(benchmark, orbits_synthetic_stratified_ci):
    """Benchmark BVH index building with optimal parameters."""

    def _build() -> BVHIndex:
        return build_bvh_index(
            orbits_synthetic_stratified_ci,
            max_chord_arcmin=5.0,  # Optimal from analysis
            guard_arcmin=0.65,  # Optimal from analysis
            max_leaf_size=64,  # Optimal from analysis
            max_processes=1,
            max_segments_per_orbit=512,  # Optimal from analysis
            epsilon_n_au=1e-9,  # Optimal from analysis
            padding_method="baseline",  # Optimal from analysis
        )

    # Capture the built index from the benchmarked call (benchmark returns function result)
    idx = benchmark(_build)

    # Collect metrics
    nodes_count = len(idx.nodes)
    prims_count = len(idx.prims)
    segs_count = len(idx.segments)

    # Compute index size
    cache_dir = Path(__file__).parent / "cache" / "indices" / "optimal"
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(str(cache_dir))
    total_bytes = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())

    # Add metrics as benchmark extra info
    benchmark.extra_info.update(
        {
            "nodes_count": nodes_count,
            "prims_count": prims_count,
            "segments_count": segs_count,
            "index_size_bytes": total_bytes,
            "max_chord_arcmin": 5.0,
            "index_guard_arcmin": 0.65,
            "max_leaf_size": 64,
            "max_segments_per_orbit": 512,
            "epsilon_n_au": 1e-9,
            "padding_method": "baseline",
        }
    )
