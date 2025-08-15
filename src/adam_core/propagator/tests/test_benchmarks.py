import numpy as np
import pytest

from adam_assist import ASSISTPropagator

from ...observers.observers import Observers
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits


# Optional Ray-based multi-processing benchmark
RAY_INSTALLED = False
try:
    import ray  # noqa: F401

    RAY_INSTALLED = True
except Exception:
    pass


@pytest.mark.benchmark(group="assist_ephemeris")
def test_assist_generate_ephemeris_single_process_benchmark(benchmark):
    """
    Benchmark the CPU-bound ephemeris generation path implemented by
    ASSISTPropagator using a single process.
    """
    # Keep sizes modest to stay CI-friendly while still exercising the code path
    num_orbits = 20
    num_times = 20

    orbits = make_real_orbits(num_orbits)
    times = Timestamp.from_mjd(np.arange(60000, 60000 + num_times), scale="tdb")
    observers = Observers.from_code("500", times)

    propagator = ASSISTPropagator()

    def run():
        return propagator.generate_ephemeris(
            orbits,
            observers,
            covariance=True,
            num_samples=3,
            max_processes=1,
            chunk_size=1,
            seed=42,
        )

    ephemeris = benchmark(run)
    assert len(ephemeris) == len(orbits) * len(observers)


@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
@pytest.mark.benchmark(group="assist_ephemeris")
def test_assist_generate_ephemeris_multi_process_benchmark(benchmark):
    """
    Benchmark the ephemeris generation with multi-processing enabled (Ray).
    """
    from adam_core.ray_cluster import initialize_use_ray

    num_orbits = 20
    num_times = 20

    orbits = make_real_orbits(num_orbits)
    times = Timestamp.from_mjd(np.arange(60000, 60000 + num_times), scale="tdb")
    observers = Observers.from_code("500", times)

    propagator = ASSISTPropagator()
    initialize_use_ray(num_cpus=4)

    def run():
        return propagator.generate_ephemeris(
            orbits,
            observers,
            covariance=True,
            num_samples=3,
            max_processes=4,
            chunk_size=1,
            seed=42,
        )

    ephemeris = benchmark(run)
    assert len(ephemeris) == len(orbits) * len(observers)


