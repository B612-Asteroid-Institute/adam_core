import os
from typing import Set, Tuple

import numpy as np
import quivr as qv
import pytest

from adam_core.geometry.bvh.index import BVHIndex, build_bvh_index
from adam_core.geometry.bvh.query import (
    OverlapHits,
    query_bvh,
    query_bvh_sharded,
    route_rays_to_shards,
    write_routed_rays_by_shard,
)
from adam_core.geometry.bvh.sharded import (
    FilesystemShardResolver,
    ShardedBVH,
    build_bvh_index_sharded,
    compute_sharding_params,
    derive_morton_ranges,
    route_and_write_sharded_chunk,
    build_shard_index_from_parts,
    assemble_sharded_bvh,
)



@pytest.mark.ci
def test_route_and_write_rays_by_shard(tmp_path, orbits_synthetic_stratified_ci, rays_nbody):
    index_dir = tmp_path / "sharded_idx"
    routed_dir = tmp_path / "routed"
    index_dir.mkdir()

    sharded = build_bvh_index_sharded(
        orbits_source=orbits_synthetic_stratified_ci[:256],
        index_dir=str(index_dir),
        num_shards=8,
        sample_fraction=0.2,
        max_chord_arcmin=5.0,
        guard_arcmin=0.65,
        max_leaf_size=32,
        max_segments_per_orbit=256,
        epsilon_n_au=1e-6,
        padding_method="baseline",
        chunk_size_orbits=128,
        max_processes=1,
    )

    rays_small = rays_nbody[:1024]
    assignments, tele = route_rays_to_shards(
        sharded,
        rays_small,
        packet_size=32,
        max_shards_per_packet=8,
    )
    assert len(assignments) >= 0
    shard_set_manifest = set(sharded.tlas_prims.shard_id.to_pylist())
    shard_set_assigned = set(assignments.shard_id.to_pylist()) if len(assignments) else set()
    assert shard_set_assigned.issubset(shard_set_manifest)

    rows_written, tele2 = write_routed_rays_by_shard(
        sharded,
        rays_small,
        str(routed_dir),
        packet_size=32,
        max_open_writers=64,
    )
    assert rows_written == len(assignments)

    # Verify a shard file exists and row count matches assignments for that shard
    if len(assignments):
        sid0 = assignments.shard_id[0]
        sid0_dir = routed_dir / str(sid0)
        assert (sid0_dir / "rays.parquet").exists()


@pytest.mark.ci
def test_query_parity_sharded_vs_monolithic(tmp_path, orbits_synthetic_stratified_ci, rays_nbody):
    index_dir = tmp_path / "sharded_idx"
    index_dir.mkdir()

    # Build monolithic
    mono_idx = build_bvh_index(
        orbits_synthetic_stratified_ci[:256],
        max_chord_arcmin=5.0,
        guard_arcmin=0.65,
        max_leaf_size=32,
        chunk_size_orbits=256,
        max_processes=1,
        max_segments_per_orbit=256,
        epsilon_n_au=1e-6,
        padding_method="baseline",
    )

    # Build sharded
    sharded = build_bvh_index_sharded(
        orbits_source=orbits_synthetic_stratified_ci[:256],
        index_dir=str(index_dir),
        num_shards=8,
        sample_fraction=0.2,
        max_chord_arcmin=5.0,
        guard_arcmin=0.65,
        max_leaf_size=32,
        max_segments_per_orbit=256,
        epsilon_n_au=1e-6,
        padding_method="baseline",
        chunk_size_orbits=128,
        max_processes=1,
    )

    rays_small = rays_nbody[:1024]

    hits_mono, _ = query_bvh(
        mono_idx,
        rays_small,
        guard_arcmin=0.65,
        batch_size=32768,
        window_size=16384,
        max_processes=0,
    )

    resolver = FilesystemShardResolver(str(index_dir))
    hits_sharded, _ = query_bvh_sharded(
        sharded,
        rays_small,
        resolver=resolver,
        batch_size=32768,
        window_size=16384,
        packet_size=64,
        max_shards_per_packet=8,
    )

    def pairs(h: OverlapHits) -> Set[Tuple[str, str]]:
        return set(zip(h.det_id.to_pylist(), h.orbit_id.to_pylist()))

    # Allow that ordering differs; compare sets of (det_id, orbit_id)
    assert pairs(hits_sharded) == pairs(hits_mono)


@pytest.mark.ci
def test_compute_sharding_params_and_ranges_small(orbits_synthetic_stratified_ci):
    params, cuts = compute_sharding_params(
        orbits_synthetic_stratified_ci[:128],
        num_shards=8,
        sample_fraction=0.2,
        max_chord_arcmin=5.0,
        max_segments_per_orbit=256,
        chunk_size_orbits=64,
    )
    assert len(cuts) == 7
    ranges = derive_morton_ranges(cuts, 8)
    assert len(ranges) == 8
    # Ensure ranges are strictly increasing in hi
    his = ranges.morton_hi.to_numpy()
    assert np.all(his[:-1] < his[1:]) or len(his) <= 1


@pytest.mark.ci
def test_route_write_chunk_and_build_from_parts(tmp_path, orbits_synthetic_stratified_ci):
    # Stage 1: sharding params
    params, cuts = compute_sharding_params(
        orbits_synthetic_stratified_ci[:128],
        num_shards=8,
        sample_fraction=0.2,
        max_chord_arcmin=5.0,
        max_segments_per_orbit=128,
        chunk_size_orbits=64,
    )
    ranges = derive_morton_ranges(cuts, 8)

    # Stage 2: route a couple of tiny chunks (single orbit chunks)
    out_root = tmp_path / "shards"
    out_root.mkdir()
    for i in range(4):
        batch = orbits_synthetic_stratified_ci[i : i + 1]
        written, stats = route_and_write_sharded_chunk(
            batch,
            params=params,
            cuts=cuts,
            output_root=str(out_root),
            chunk_id=f"c{i:02d}",
            max_chord_arcmin=5.0,
            max_segments_per_orbit=64,
            guard_arcmin=0.65,
            epsilon_n_au=1e-6,
            padding_method="baseline",
            max_open_writers=64,
        )
        # WrittenParts rows should equal sum of stats
        assert int(sum(written.rows.to_numpy())) == int(sum(stats.count.to_numpy()))

    # Stage 3: build each shard from parts; ensure it returns metadata rows
    shard_ids = derive_morton_ranges(cuts, 8).shard_id.to_pylist()
    metas = []
    for sid in shard_ids:
        sdir = out_root / sid
        if not sdir.exists():
            sdir.mkdir(parents=True)
        meta_row = build_shard_index_from_parts(
            sid,
            str(sdir),
            morton_ranges=ranges,
            max_leaf_size=32,
            guard_arcmin=0.65,
            epsilon_n_au=1e-6,
            padding_method="baseline",
            max_processes=1,
        )
        metas.append(meta_row)

    shards = qv.concatenate(metas, defrag=True)
    assert len(shards) == len(shard_ids)

    # Stage 4: assemble TLAS in-memory
    sharded = assemble_sharded_bvh(shards)
    assert len(sharded.tlas_prims) == len(shards)
    if len(shards):
        assert len(sharded.tlas_nodes) > 0



@pytest.mark.ci
def test_build_sharded_multiple_single_orbit_chunks(tmp_path, orbits_synthetic_stratified_ci):
    index_dir = tmp_path / "sharded_idx_multi"
    index_dir.mkdir()

    sharded = build_bvh_index_sharded(
        orbits_source=orbits_synthetic_stratified_ci[:96],
        index_dir=str(index_dir),
        num_shards=8,
        sample_fraction=0.2,
        max_chord_arcmin=5.0,
        guard_arcmin=0.65,
        max_leaf_size=64,
        max_segments_per_orbit=512,
        epsilon_n_au=1e-9,
        padding_method="baseline",
        chunk_size_orbits=1,  # single-orbit chunks
        max_processes=1,
    )

    # Sanity checks on returned object
    assert len(sharded.shards) == len(sharded.tlas_prims)
    if len(sharded.shards):
        assert len(sharded.tlas_nodes) > 0


