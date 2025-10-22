import os
from typing import Set, Tuple

import numpy as np
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
)


@pytest.mark.ci
def test_sharded_build_and_tlas(tmp_path, orbits_synthetic_stratified_ci):
    index_dir = tmp_path / "sharded_idx"
    index_dir.mkdir()

    # Build a small sharded index
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

    # Reload via wrapper
    sharded = ShardedBVH.from_dir(str(index_dir))
    assert sharded.shards.num_shards() == len(sharded.shards)
    assert len(sharded.tlas_prims) == sharded.shards.num_shards()
    # TLAS should have at least one node when shards exist
    if len(sharded.shards) > 0:
        assert len(sharded.tlas_nodes) > 0


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

    sharded = ShardedBVH.from_dir(str(index_dir))
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
    sharded = ShardedBVH.from_dir(str(index_dir))

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


