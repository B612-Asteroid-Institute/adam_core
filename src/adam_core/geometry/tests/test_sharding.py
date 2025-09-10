"""
Tests for BVH sharding system.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.keplerian import KeplerianCoordinates
from adam_core.coordinates.origin import OriginCodes
from adam_core.geometry import (
    ShardManifest,
    build_bvh_shards,
    estimate_query_memory,
    estimate_ray_query_resources,
    estimate_shard_bytes,
    load_shard,
    query_manifest_local,
    query_manifest_ray,
    save_manifest,
)
from adam_core.observations.rays import ObservationRays
from adam_core.orbits.orbits import Orbits
from adam_core.time import Timestamp


def create_test_orbits(n_orbits: int = 10) -> Orbits:
    """Create test orbits for sharding tests."""
    # Create simple test orbits
    orbit_ids = [f"test_orbit_{i:04d}" for i in range(n_orbits)]

    # Semi-major axes from 1-3 AU
    a = np.linspace(1.0, 3.0, n_orbits)
    e = np.full(n_orbits, 0.1)  # Low eccentricity
    i = np.full(n_orbits, 0.1)  # Low inclination
    Omega = np.linspace(0, 2 * np.pi, n_orbits)
    omega = np.full(n_orbits, 0.0)
    M0 = np.full(n_orbits, 0.0)

    epoch = Timestamp.from_mjd([60000.0] * n_orbits, scale="tdb")

    # Create Keplerian coordinates first
    from adam_core.coordinates.origin import Origin

    kep_coords = KeplerianCoordinates.from_kwargs(
        a=a,
        e=e,
        i=np.degrees(i),  # Convert to degrees
        raan=np.degrees(Omega),  # Convert to degrees
        ap=np.degrees(omega),  # Convert to degrees
        M=np.degrees(M0),  # Convert to degrees
        time=epoch,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n_orbits),
        frame="ecliptic",
    )

    # Convert to Cartesian
    cart_coords = CartesianCoordinates.from_keplerian(kep_coords)

    return Orbits.from_kwargs(
        orbit_id=orbit_ids,
        coordinates=cart_coords,
    )


def create_test_rays(n_rays: int = 100) -> ObservationRays:
    """Create test observation rays."""
    det_ids = [f"det_{i:06d}" for i in range(n_rays)]
    times = Timestamp.from_mjd([60000.0] * n_rays, scale="utc")

    # Random unit vectors
    np.random.seed(42)
    u_vectors = np.random.randn(n_rays, 3)
    u_vectors = u_vectors / np.linalg.norm(u_vectors, axis=1, keepdims=True)

    # Observer at origin
    from adam_core.coordinates.origin import Origin

    observer_coords = CartesianCoordinates.from_kwargs(
        x=np.zeros(n_rays),
        y=np.zeros(n_rays),
        z=np.zeros(n_rays),
        vx=np.zeros(n_rays),
        vy=np.zeros(n_rays),
        vz=np.zeros(n_rays),
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n_rays),
        frame="ecliptic",
    )

    return ObservationRays.from_kwargs(
        det_id=det_ids,
        time=times,
        observer_code=["X05"] * n_rays,
        observer=observer_coords,
        u_x=u_vectors[:, 0],
        u_y=u_vectors[:, 1],
        u_z=u_vectors[:, 2],
    )


class TestShardingTypes:
    """Test sharding data types."""

    def test_shard_meta_serialization(self):
        """Test ShardMeta to/from dict conversion."""
        from adam_core.geometry.sharding_types import ShardMeta

        meta = ShardMeta(
            shard_id="test_shard_001",
            orbit_id_start="orbit_001",
            orbit_id_end="orbit_100",
            num_orbits=100,
            num_segments=5000,
            num_bvh_nodes=10000,
            max_chord_arcmin=60.0,
            float_dtype="float64",
            segments_npz="test_shard_001_segments.npz",
            bvh_npz="test_shard_001_bvh.npz",
            orbit_ids_json="test_shard_001_orbit_ids.json",
            segments_bytes=10_000_000,
            bvh_bytes=8_000_000,
            orbit_ids_bytes=50_000,
            total_bytes=18_050_000,
            file_hashes={},
            estimated_bytes=50_000_000,
        )

        # Test serialization
        data = meta.to_dict()
        assert data["shard_id"] == "test_shard_001"
        assert data["num_orbits"] == 100

        # Test deserialization
        meta2 = ShardMeta.from_dict(data)
        assert meta2 == meta

    def test_manifest_serialization(self):
        """Test ShardManifest to/from dict conversion."""
        from adam_core.geometry.sharding_types import ShardManifest, ShardMeta

        meta1 = ShardMeta(
            shard_id="shard_001",
            orbit_id_start="orbit_001",
            orbit_id_end="orbit_050",
            num_orbits=50,
            num_segments=2500,
            num_bvh_nodes=5000,
            max_chord_arcmin=60.0,
            float_dtype="float64",
            segments_npz="shard_001_segments.npz",
            bvh_npz="shard_001_bvh.npz",
            orbit_ids_json="shard_001_orbit_ids.json",
            segments_bytes=5_000_000,
            bvh_bytes=4_000_000,
            orbit_ids_bytes=25_000,
            total_bytes=9_025_000,
            file_hashes={},
            estimated_bytes=25_000_000,
        )

        manifest = ShardManifest(
            version="1.0.0",
            build_time="2024-01-01T00:00:00",
            max_chord_arcmin=60.0,
            float_dtype="float64",
            shards=[meta1],
            total_orbits=50,
            total_segments=2500,
            total_bvh_nodes=5000,
            total_estimated_bytes=25_000_000,
        )

        # Test serialization
        data = manifest.to_dict()
        assert data["version"] == "1.0.0"
        assert len(data["shards"]) == 1

        # Test deserialization
        manifest2 = ShardManifest.from_dict(data)
        assert manifest2 == manifest


class TestShardBuilder:
    """Test shard building functionality."""

    def test_estimate_shard_bytes(self):
        """Test shard size estimation."""
        # Test with known parameters
        bytes_est = estimate_shard_bytes(
            num_orbits=100,
            seg_per_orbit=50,
            float_dtype="float64",
        )

        # Should be reasonable size (segments + BVH + overhead)
        assert bytes_est > 0
        assert bytes_est < 100_000_000  # Less than 100 MB for small test

        # float32 should be smaller
        bytes_est_f32 = estimate_shard_bytes(
            num_orbits=100,
            seg_per_orbit=50,
            float_dtype="float32",
        )

        assert bytes_est_f32 < bytes_est

    def test_build_small_shards(self):
        """Test building shards from small orbit set."""
        orbits = create_test_orbits(n_orbits=5)

        # Build shards with small target size to force multiple shards
        shards = build_bvh_shards(
            orbits=orbits,
            max_chord_arcmin=60.0,
            target_shard_bytes=1_000_000,  # 1 MB - very small
            float_dtype="float64",
        )

        # Should create at least one shard
        assert len(shards) >= 1

        # Check shard properties
        total_orbits = sum(len(shard.orbit_mapping.id_to_index) for shard in shards)
        assert total_orbits == len(orbits)

        # Each shard should have valid data
        for shard in shards:
            assert len(shard.segments) > 0
            assert len(shard.bvh.nodes_min) > 0
            assert shard.max_chord_arcmin == 60.0
            assert shard.float_dtype == "float64"


class TestShardPersistence:
    """Test shard saving and loading."""

    def test_save_and_load_manifest(self):
        """Test saving and loading complete manifest."""
        orbits = create_test_orbits(n_orbits=3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Build shards
            shards = build_bvh_shards(
                orbits=orbits,
                max_chord_arcmin=60.0,
                target_shard_bytes=10_000_000,  # 10 MB
                float_dtype="float64",
            )

            # Save manifest
            manifest = save_manifest(
                out_dir=tmp_path,
                shards=shards,
                max_chord_arcmin=60.0,
                float_dtype="float64",
            )

            # Check files exist
            assert (tmp_path / "manifest.json").exists()
            for meta in manifest.shards:
                assert (tmp_path / meta.segments_npz).exists()
                assert (tmp_path / meta.bvh_npz).exists()

            # Load manifest
            loaded_manifest = ShardManifest.load(tmp_path / "manifest.json")
            assert loaded_manifest.total_orbits == len(orbits)
            assert loaded_manifest.max_chord_arcmin == 60.0

            # Test loading individual shards
            for meta in loaded_manifest.shards:
                segments, bvh_shard = load_shard(tmp_path, meta)
                assert len(segments) > 0
                assert len(bvh_shard.nodes_min) > 0


class TestShardedQuery:
    """Test sharded query functionality."""

    def test_query_small_manifest(self):
        """Test querying a small sharded manifest."""
        orbits = create_test_orbits(n_orbits=5)
        rays = create_test_rays(n_rays=10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Build and save shards
            shards = build_bvh_shards(
                orbits=orbits,
                max_chord_arcmin=60.0,
                target_shard_bytes=5_000_000,  # 5 MB
                float_dtype="float64",
            )

            manifest = save_manifest(
                out_dir=tmp_path,
                shards=shards,
                max_chord_arcmin=60.0,
                float_dtype="float64",
            )

            # Query manifest
            hits = query_manifest_local(
                manifest=manifest,
                rays=rays,
                guard_arcmin=1.0,  # Large guard for testing
                alpha=0.0,
                ray_batch_size=5,  # Small batch for testing
                manifest_dir=tmp_path,
            )

            # Should return valid hits table (may be empty)
            assert hasattr(hits, "det_id")
            assert hasattr(hits, "orbit_id")
            assert hasattr(hits, "distance_au")

    def test_estimate_query_memory(self):
        """Test query memory estimation."""
        from adam_core.geometry.sharding_types import ShardManifest, ShardMeta

        # Create mock manifest
        meta = ShardMeta(
            shard_id="test",
            orbit_id_start="001",
            orbit_id_end="100",
            num_orbits=100,
            num_segments=5000,
            num_bvh_nodes=10000,
            max_chord_arcmin=60.0,
            float_dtype="float64",
            segments_npz="test.npz",
            bvh_npz="test.npz",
            orbit_ids_json="test_orbit_ids.json",
            segments_bytes=20_000_000,
            bvh_bytes=15_000_000,
            orbit_ids_bytes=50_000,
            total_bytes=35_050_000,
            file_hashes={},
            estimated_bytes=50_000_000,
        )

        manifest = ShardManifest(
            version="1.0.0",
            build_time="2024-01-01T00:00:00",
            max_chord_arcmin=60.0,
            float_dtype="float64",
            shards=[meta],
            total_orbits=100,
            total_segments=5000,
            total_bvh_nodes=10000,
            total_estimated_bytes=50_000_000,
        )

        # Estimate memory
        memory_est = estimate_query_memory(manifest, ray_batch_size=1000)

        assert memory_est["max_shard_bytes"] == 50_000_000
        assert memory_est["ray_batch_bytes"] > 0
        assert memory_est["peak_query_bytes"] > memory_est["max_shard_bytes"]


@pytest.mark.benchmark
class TestShardingBenchmarks:
    """Benchmark tests for sharding performance."""

    def test_shard_build_benchmark(self, benchmark):
        """Benchmark shard building."""
        orbits = create_test_orbits(n_orbits=20)

        def build_shards():
            return build_bvh_shards(
                orbits=orbits,
                max_chord_arcmin=60.0,
                target_shard_bytes=10_000_000,
                float_dtype="float64",
            )

        shards = benchmark(build_shards)
        assert len(shards) >= 1

    def test_shard_query_benchmark(self, benchmark):
        """Benchmark sharded query."""
        orbits = create_test_orbits(n_orbits=10)
        rays = create_test_rays(n_rays=50)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Build shards
            shards = build_bvh_shards(
                orbits=orbits,
                max_chord_arcmin=60.0,
                target_shard_bytes=5_000_000,
                float_dtype="float64",
            )

            manifest = save_manifest(
                out_dir=tmp_path,
                shards=shards,
                max_chord_arcmin=60.0,
                float_dtype="float64",
            )

            def query_shards():
                return query_manifest_local(
                    manifest=manifest,
                    rays=rays,
                    guard_arcmin=1.0,
                    alpha=0.0,
                    ray_batch_size=25,
                    manifest_dir=tmp_path,
                )

            hits = benchmark(query_shards)
            assert hasattr(hits, "det_id")


class TestShardingSmoke:
    """End-to-end smoke test for sharded BVH query from Ephemeris."""

    def test_end_to_end_smoke(self):
        """Build shards, create ephemeris->rays, and query manifest."""
        # Tiny orbit set
        orbits = create_test_orbits(n_orbits=3)

        # Build and save shards
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            shards = build_bvh_shards(
                orbits=orbits,
                max_chord_arcmin=60.0,
                target_shard_bytes=5_000_000,
                float_dtype="float64",
            )

            manifest = save_manifest(
                out_dir=tmp_path,
                shards=shards,
                max_chord_arcmin=60.0,
                float_dtype="float64",
            )

            # Create observers and epochs
            from adam_core.coordinates.origin import Origin, OriginCodes
            from adam_core.coordinates.spherical import SphericalCoordinates
            from adam_core.geometry import ephemeris_to_rays
            from adam_core.observers.observers import Observers
            from adam_core.orbits.ephemeris import Ephemeris

            station_codes = ["X05", "T08", "I41"]
            n_epochs = 3
            times = Timestamp.from_mjd(
                [60000.0 + i for i in range(n_epochs * len(station_codes))], scale="utc"
            )

            # Repeat station codes per epoch
            codes = station_codes * n_epochs

            observers = Observers.from_codes(times=times, codes=codes)

            # Build simple RA/Dec pattern (spread across sky)
            n = len(times)
            ra = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
            dec = np.linspace(-0.3, 0.3, n)

            sph = SphericalCoordinates.from_kwargs(
                rho=np.ones(n),
                lon=ra,
                lat=dec,
                vrho=np.zeros(n),
                vlon=np.zeros(n),
                vlat=np.zeros(n),
                time=times,
                origin=observers.coordinates.origin,
                frame="equatorial",
            )

            ephem = Ephemeris.from_kwargs(
                orbit_id=["smoke"] * n,
                object_id=["smoke"] * n,
                coordinates=sph,
            )

            rays = ephemeris_to_rays(ephem)

            # Query manifest
            hits = query_manifest_local(
                manifest=manifest,
                rays=rays,
                guard_arcmin=1.0,
                alpha=0.0,
                ray_batch_size=10,
                manifest_dir=tmp_path,
            )

            # Table integrity checks
            assert hasattr(hits, "det_id")
            assert hasattr(hits, "orbit_id")
            assert hasattr(hits, "seg_id")
            assert hasattr(hits, "distance_au")

            # We may or may not have hits with this synthetic setup, but the call must succeed
            assert isinstance(len(hits), int)


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestShardedQueryRay:
    """Test Ray-parallel sharded query functionality."""

    def setup_method(self):
        """Initialize Ray for each test."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)

    def teardown_method(self):
        """Shutdown Ray after each test."""
        if ray.is_initialized():
            ray.shutdown()

    def test_ray_vs_local_consistency(self):
        """Test that Ray and local queries produce identical results."""
        orbits = create_test_orbits(n_orbits=4)
        rays = create_test_rays(n_rays=8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Build and save shards
            shards = build_bvh_shards(
                orbits=orbits,
                max_chord_arcmin=60.0,
                target_shard_bytes=2_000_000,  # Force multiple shards
                float_dtype="float64",
            )

            manifest = save_manifest(
                out_dir=tmp_path,
                shards=shards,
                max_chord_arcmin=60.0,
                float_dtype="float64",
            )

            # Query with local implementation
            local_hits = query_manifest_local(
                manifest=manifest,
                rays=rays,
                guard_arcmin=1.0,
                alpha=0.0,
                ray_batch_size=4,
                manifest_dir=tmp_path,
            )

            # Query with Ray implementation
            ray_hits = query_manifest_ray(
                manifest=manifest,
                rays=rays,
                guard_arcmin=1.0,
                alpha=0.0,
                ray_batch_size=4,
                max_concurrency=2,
                manifest_dir=tmp_path,
            )

            # Results should be identical
            assert len(local_hits) == len(ray_hits)

            if len(local_hits) > 0:
                # Compare sorted results
                local_det_ids = sorted(local_hits.det_id.to_pylist())
                ray_det_ids = sorted(ray_hits.det_id.to_pylist())
                assert local_det_ids == ray_det_ids

                local_distances = local_hits.distance_au.to_numpy(zero_copy_only=False)
                ray_distances = ray_hits.distance_au.to_numpy(zero_copy_only=False)
                np.testing.assert_array_almost_equal(
                    np.sort(local_distances), np.sort(ray_distances)
                )

    def test_ray_query_determinism(self):
        """Test that repeated Ray queries produce identical results."""
        orbits = create_test_orbits(n_orbits=3)
        rays = create_test_rays(n_rays=6)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            shards = build_bvh_shards(
                orbits=orbits,
                max_chord_arcmin=60.0,
                target_shard_bytes=3_000_000,
                float_dtype="float64",
            )

            manifest = save_manifest(
                out_dir=tmp_path,
                shards=shards,
                max_chord_arcmin=60.0,
                float_dtype="float64",
            )

            # Run query twice
            hits1 = query_manifest_ray(
                manifest=manifest,
                rays=rays,
                guard_arcmin=1.0,
                alpha=0.0,
                ray_batch_size=3,
                manifest_dir=tmp_path,
            )

            hits2 = query_manifest_ray(
                manifest=manifest,
                rays=rays,
                guard_arcmin=1.0,
                alpha=0.0,
                ray_batch_size=3,
                manifest_dir=tmp_path,
            )

            # Results should be identical
            assert len(hits1) == len(hits2)

            if len(hits1) > 0:
                det_ids1 = hits1.det_id.to_pylist()
                det_ids2 = hits2.det_id.to_pylist()
                assert det_ids1 == det_ids2

                distances1 = hits1.distance_au.to_numpy(zero_copy_only=False)
                distances2 = hits2.distance_au.to_numpy(zero_copy_only=False)
                np.testing.assert_array_equal(distances1, distances2)

    def test_estimate_ray_query_resources(self):
        """Test Ray query resource estimation."""
        from adam_core.geometry.sharding_types import ShardManifest, ShardMeta

        # Create mock manifest with multiple shards
        shards = []
        for i in range(3):
            meta = ShardMeta(
                shard_id=f"shard_{i:03d}",
                orbit_id_start=f"orbit_{i*100:06d}",
                orbit_id_end=f"orbit_{(i+1)*100-1:06d}",
                num_orbits=100,
                num_segments=5000,
                num_bvh_nodes=10000,
                max_chord_arcmin=60.0,
                float_dtype="float64",
                segments_npz=f"shard_{i:03d}_segments.npz",
                bvh_npz=f"shard_{i:03d}_bvh.npz",
                orbit_ids_json=f"shard_{i:03d}_orbit_ids.json",
                segments_bytes=20_000_000,
                bvh_bytes=15_000_000,
                orbit_ids_bytes=50_000,
                total_bytes=35_050_000,
                file_hashes={},
                estimated_bytes=30_000_000,
            )
            shards.append(meta)

        manifest = ShardManifest(
            version="1.1.0",
            build_time="2024-01-01T00:00:00",
            max_chord_arcmin=60.0,
            float_dtype="float64",
            shards=shards,
            total_orbits=300,
            total_segments=15000,
            total_bvh_nodes=30000,
            total_estimated_bytes=105_150_000,
        )

        # Estimate resources
        resources = estimate_ray_query_resources(
            manifest, ray_batch_size=1000, max_concurrency=2
        )

        assert resources["total_shards"] == 3
        assert resources["concurrent_shards"] == 2
        assert resources["max_shard_bytes"] == 35_050_000
        assert resources["ray_batch_bytes"] > 0
        assert resources["peak_memory_bytes"] > resources["max_shard_bytes"]
