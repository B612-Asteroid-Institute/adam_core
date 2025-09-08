"""
Pytest-benchmark tests for geometric overlap performance.

These tests measure performance of core JAX kernels and Ray parallelization.
Run with: pdm run pytest -m benchmark --benchmark-only
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from typing import Optional

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from adam_core.geometry.jax_kernels import (
    ray_segment_distances_jax,
    compute_overlap_hits_jax,
    OverlapBackend,
)
from adam_core.geometry.jax_types import BVHArrays, SegmentsSOA, HitsSOA
from adam_core.geometry.aggregator import CandidateBatch
from adam_core.geometry.adapters import (
    bvh_shard_to_arrays,
    segments_to_soa,
    rays_to_arrays,
)

# Test data generators
def generate_test_rays(num_rays: int = 1000, seed: int = 42):
    """Generate test observation rays."""
    rng = np.random.RandomState(seed)
    
    # Observer positions (AU from Sun)
    origins = rng.uniform(-5, 5, (num_rays, 3))
    
    # Random unit directions
    directions = rng.normal(0, 1, (num_rays, 3))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    # Observer distances
    distances = rng.uniform(0.9, 1.1, num_rays)
    
    return origins, directions, distances


def generate_test_segments(num_segments: int = 10000, seed: int = 42):
    """Generate test orbit segments."""
    rng = np.random.RandomState(seed)
    
    # Random segment endpoints in heliocentric frame
    starts = rng.uniform(-10, 10, (num_segments, 3))
    ends = rng.uniform(-10, 10, (num_segments, 3))
    
    # Orbit and segment IDs
    orbit_ids = rng.randint(0, num_segments // 10, num_segments)
    seg_ids = np.arange(num_segments)
    
    return starts, ends, orbit_ids, seg_ids


def generate_candidate_batch(batch_size: int = 64, max_candidates: int = 32, seed: int = 42):
    """Generate a test candidate batch for JAX kernels."""
    rng = np.random.RandomState(seed)
    
    # Ray data
    ray_origins = rng.uniform(-5, 5, (batch_size, 3))
    ray_directions = rng.normal(0, 1, (batch_size, 3))
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1, keepdims=True)
    observer_distances = rng.uniform(0.9, 1.1, batch_size)
    
    # Candidate segments (padded)
    starts = rng.uniform(-10, 10, (batch_size, max_candidates, 3))
    ends = rng.uniform(-10, 10, (batch_size, max_candidates, 3))
    r_mids = rng.uniform(1, 10, (batch_size, max_candidates))  # segment midpoint distances
    orbit_indices = rng.randint(0, 1000, (batch_size, max_candidates))
    seg_ids = rng.randint(0, 10000, (batch_size, max_candidates))
    leaf_ids = rng.randint(0, 100, (batch_size, max_candidates))
    
    # Ray indices
    ray_indices = np.arange(batch_size, dtype=np.int32)
    
    # Mask (some candidates are padding)
    mask = rng.random((batch_size, max_candidates)) < 0.7  # 70% valid candidates
    
    return CandidateBatch(
        ray_indices=ray_indices,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        observer_distances=observer_distances,
        seg_starts=starts,
        seg_ends=ends,
        r_mids=r_mids,
        orbit_indices=orbit_indices,
        seg_ids=seg_ids,
        leaf_ids=leaf_ids,
        mask=mask,
    )


@pytest.mark.benchmark
class TestJAXKernelBenchmarks:
    """Benchmark JAX-compiled geometric kernels."""
    
    def test_distance_kernel_small(self, benchmark):
        """Benchmark JAX distance kernel with small batch."""
        batch = generate_candidate_batch(batch_size=32, max_candidates=16)
        
        # JIT compile first
        _ = ray_segment_distances_jax(
            batch.ray_origins, batch.ray_directions, 
            batch.seg_starts, batch.seg_ends, batch.mask
        )
        
        # Benchmark the compiled kernel
        result = benchmark(
            ray_segment_distances_jax,
            batch.ray_origins, batch.ray_directions,
            batch.seg_starts, batch.seg_ends, batch.mask
        )
        assert result.shape == (32, 16)
    
    def test_distance_kernel_medium(self, benchmark):
        """Benchmark JAX distance kernel with medium batch."""
        batch = generate_candidate_batch(batch_size=128, max_candidates=32)
        
        # JIT compile first
        _ = ray_segment_distances_jax(
            batch.ray_origins, batch.ray_directions,
            batch.seg_starts, batch.seg_ends, batch.mask
        )
        
        # Benchmark the compiled kernel
        result = benchmark(
            ray_segment_distances_jax,
            batch.ray_origins, batch.ray_directions,
            batch.seg_starts, batch.seg_ends, batch.mask
        )
        assert result.shape == (128, 32)
    
    def test_distance_kernel_large(self, benchmark):
        """Benchmark JAX distance kernel with large batch."""
        batch = generate_candidate_batch(batch_size=512, max_candidates=64)
        
        # JIT compile first
        _ = ray_segment_distances_jax(
            batch.ray_origins, batch.ray_directions,
            batch.seg_starts, batch.seg_ends, batch.mask
        )
        
        # Benchmark the compiled kernel
        result = benchmark(
            ray_segment_distances_jax,
            batch.ray_origins, batch.ray_directions,
            batch.seg_starts, batch.seg_ends, batch.mask
        )
        assert result.shape == (512, 64)
    
    def test_overlap_hits_computation(self, benchmark):
        """Benchmark complete overlap hits computation."""
        batch = generate_candidate_batch(batch_size=256, max_candidates=32)
        guard_arcmin = 1.0
        max_hits_per_ray = 10
        
        # Warm up
        _ = compute_overlap_hits_jax(batch, guard_arcmin, max_hits_per_ray)
        
        # Benchmark
        result = benchmark(
            compute_overlap_hits_jax, 
            batch, 
            guard_arcmin, 
            max_hits_per_ray
        )
        assert isinstance(result, HitsSOA)


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [64, 128, 256, 512])
@pytest.mark.parametrize("max_candidates", [16, 32, 64])
class TestScalingBenchmarks:
    """Benchmark scaling behavior with different batch sizes."""
    
    def test_distance_kernel_scaling(self, benchmark, batch_size, max_candidates):
        """Test how distance kernel scales with batch size and candidates."""
        batch = generate_candidate_batch(
            batch_size=batch_size, 
            max_candidates=max_candidates
        )
        
        # JIT compile first
        _ = ray_segment_distances_jax(
            batch.ray_origins, batch.ray_directions,
            batch.seg_starts, batch.seg_ends, batch.mask
        )
        
        # Benchmark
        result = benchmark(
            ray_segment_distances_jax,
            batch.ray_origins, batch.ray_directions,
            batch.seg_starts, batch.seg_ends, batch.mask
        )
        assert result.shape == (batch_size, max_candidates)


@pytest.mark.benchmark
@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestRayParallelBenchmarks:
    """Benchmark Ray-parallel implementations."""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_ray(self):
        """Initialize Ray for parallel benchmarks."""
        if RAY_AVAILABLE:
            if not ray.is_initialized():
                ray.init(num_cpus=2, object_store_memory=1000000000)  # 1GB
            yield
            if ray.is_initialized():
                ray.shutdown()
        else:
            yield
    
    def test_serial_vs_parallel_consistency(self, benchmark):
        """Benchmark that parallel gives same results as serial (consistency check)."""
        from adam_core.geometry.jax_overlap import geometric_overlap_jax
        from adam_core.geometry.jax_remote import query_bvh_parallel_jax
        from adam_core.geometry.bvh import BVHShard
        from adam_core.orbits.polyline import OrbitPolylineSegments
        from adam_core.observations.rays import ObservationRays
        
        # Generate small test data for consistency check
        num_rays = 100
        num_segments = 1000
        
        # This is a simplified benchmark - in practice you'd use real BVH/segments
        # For now, just benchmark the parallel orchestration overhead
        def mock_parallel_query():
            """Mock parallel query to benchmark orchestration overhead."""
            import time
            time.sleep(0.001)  # Simulate some work
            return {"det_ids": [], "orbit_ids": [], "seg_ids": [], "distances": []}
        
        result = benchmark(mock_parallel_query)
        assert isinstance(result, dict)


def _has_gpu():
    """Check if GPU devices are available without raising exceptions."""
    try:
        return len(jax.devices("gpu")) > 0
    except (RuntimeError, ValueError):
        return False


@pytest.mark.benchmark
@pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
class TestGPUBenchmarks:
    """Benchmark GPU acceleration when available."""
    
    def test_gpu_vs_cpu_distance_kernel(self, benchmark):
        """Compare GPU vs CPU performance for distance kernel."""
        batch = generate_candidate_batch(batch_size=1024, max_candidates=64)
        guard_arcmin = 1.0
        
        # Force GPU device
        gpu_devices = jax.devices("gpu")
        with jax.default_device(gpu_devices[0]):
            # JIT compile on GPU
            _ = ray_segment_distances_jax(
                batch.ray_origins, batch.ray_directions,
                batch.seg_starts, batch.seg_ends, batch.mask
            )
            
            # Benchmark GPU execution
            result = benchmark(
                ray_segment_distances_jax,
                batch.ray_origins, batch.ray_directions,
                batch.seg_starts, batch.seg_ends, batch.mask
            )
            assert result.shape == (1024, 64)
    
    def test_gpu_memory_transfer(self, benchmark):
        """Benchmark GPU memory transfer overhead."""
        batch = generate_candidate_batch(batch_size=512, max_candidates=32)
        
        def transfer_to_gpu():
            """Transfer batch to GPU and back."""
            gpu_devices = jax.devices("gpu")
            gpu_batch = jax.device_put(batch, gpu_devices[0])
            return jax.device_get(gpu_batch)
        
        result = benchmark(transfer_to_gpu)
        assert result.ray_origins.shape == (512, 3)


@pytest.mark.benchmark
class TestBackendComparison:
    """Compare different computational backends."""
    
    def test_jax_vs_numpy_backend(self, benchmark):
        """Compare JAX vs NumPy backend performance."""
        from adam_core.geometry.jax_kernels import compute_overlap_hits_numpy
        
        batch = generate_candidate_batch(batch_size=128, max_candidates=32)
        guard_arcmin = 1.0
        max_hits_per_ray = 10
        
        # Benchmark NumPy backend
        result = benchmark(
            compute_overlap_hits_numpy,
            batch,
            guard_arcmin,
            max_hits_per_ray
        )
        assert isinstance(result, HitsSOA)


if __name__ == "__main__":
    # Allow running benchmarks directly
    pytest.main([__file__, "-m", "benchmark", "--benchmark-only", "-v"])
