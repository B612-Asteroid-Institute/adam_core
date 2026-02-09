import numpy as np
import pyarrow as pa
import pytest
from adam_assist import ASSISTPropagator

from adam_core.propagator import propagator as propagator_module

import ray

from adam_core.ray_cluster import initialize_use_ray

from ...observers.observers import Observers
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits


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
    # Anchor benchmark times near the input orbit epochs. Some sample orbits have epochs
    # far from an arbitrary fixed time (e.g., MJD=60000). Long propagations can produce
    # physically unrealistic states (e.g., close encounters/collisions in N-body integration)
    # which then break light-time correction (NaN/inf). Using a per-sample epoch keeps the
    # benchmark stable across integrator/library versions while exercising the same code path.
    orbit_mjd = orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    base_mjd = int(round(float(np.median(orbit_mjd))))
    times = Timestamp.from_mjd(np.arange(base_mjd, base_mjd + num_times), scale="tdb")
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


@pytest.mark.benchmark(group="assist_ephemeris")
def test_assist_generate_ephemeris_multi_process_benchmark(benchmark):
    """
    Benchmark the ephemeris generation with multi-processing enabled (Ray).
    """
    num_orbits = 20
    num_times = 20

    orbits = make_real_orbits(num_orbits)
    orbit_mjd = orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    base_mjd = int(round(float(np.median(orbit_mjd))))
    times = Timestamp.from_mjd(np.arange(base_mjd, base_mjd + num_times), scale="tdb")
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


@pytest.mark.benchmark(group="observer_alignment")
@pytest.mark.parametrize("n", [10_000])
def test_observer_alignment_indices_benchmark(benchmark, n: int):
    """
    Benchmark observer-row alignment used for predicted magnitude attachment.

    We intentionally benchmark the alignment indices computation in isolation so we can
    swap strategies without involving propagation or coordinate transforms.
    """
    rng = np.random.default_rng(123)
    codes = np.asarray([f"C{i:02d}" for i in range(25)], dtype=object)
    obs_code = rng.choice(codes, size=n, replace=True)
    # Make keys unique by construction using unique days.
    obs_days = (60000 + np.arange(n)).astype(np.int64)
    obs_nanos = np.zeros(n, dtype=np.int64)

    perm = rng.permutation(n)
    eph_code = obs_code[perm]
    eph_days = obs_days[perm]
    eph_nanos = obs_nanos[perm]

    def run_struct():
        idx = propagator_module._alignment_indices_struct_index_in(
            obs_code=pa.array(obs_code),
            obs_days=pa.array(obs_days),
            obs_nanos=pa.array(obs_nanos),
            eph_code=pa.array(eph_code),
            eph_days=pa.array(eph_days),
            eph_nanos=pa.array(eph_nanos),
        )
        return idx

    def run_string():
        idx = propagator_module._alignment_indices_string_keys(
            obs_code=obs_code,
            obs_days=obs_days,
            obs_nanos=obs_nanos,
            eph_code=eph_code,
            eph_days=eph_days,
            eph_nanos=eph_nanos,
        )
        return idx

    idx_struct = benchmark(run_struct)
    # Validate correctness once (outside the timed portion) and sanity-check string impl too.
    np.testing.assert_array_equal(obs_days[idx_struct], eph_days)
    idx_string = run_string()
    np.testing.assert_array_equal(obs_days[idx_string], eph_days)


@pytest.mark.benchmark(group="hg_param_mapping")
@pytest.mark.parametrize("n_unique,n_rep", [(20_000, 5)])
def test_hg_param_mapping_benchmark(benchmark, n_unique: int, n_rep: int):
    """
    Benchmark orbit_id -> (H_v, G) mapping aligned to ephemeris rows.

    This mirrors the common case where propagated orbits/ephemeris have repeated orbit_ids
    across many epochs, but physical parameters are constant per orbit_id.
    """
    rng = np.random.default_rng(321)
    unique_ids = np.asarray([f"o{i:06d}" for i in range(n_unique)], dtype=object)
    # Repeated rows as typically produced by propagation to many times.
    orbit_id = np.repeat(unique_ids, n_rep)
    H_v = np.repeat(rng.uniform(10.0, 25.0, size=n_unique).astype(np.float64), n_rep)
    G = np.repeat(rng.uniform(0.0, 0.5, size=n_unique).astype(np.float64), n_rep)

    # Ephemeris orbit_id is a shuffled view of the repeated orbit_id.
    eph_orbit_id = orbit_id[rng.permutation(len(orbit_id))]

    orbit_id_a = pa.array(orbit_id)
    H_v_a = pa.array(H_v)
    G_a = pa.array(G)
    eph_orbit_id_a = pa.array(eph_orbit_id)

    # Smoke-check: output arrays match expected length and have finite values where expected.
    H_ref, G_ref = propagator_module._hg_params_for_ephemeris_rows_arrow(
        orbit_id=orbit_id_a,
        H_v=H_v_a,
        G=G_a,
        ephemeris_orbit_id=eph_orbit_id_a,
    )
    assert len(H_ref) == len(eph_orbit_id)
    assert len(G_ref) == len(eph_orbit_id)
    assert np.all(np.isfinite(H_ref))
    assert np.all(np.isfinite(G_ref))

    def run():
        return propagator_module._hg_params_for_ephemeris_rows_arrow(
            orbit_id=orbit_id_a,
            H_v=H_v_a,
            G=G_a,
            ephemeris_orbit_id=eph_orbit_id_a,
        )

    H_out, G_out = benchmark(run)
    np.testing.assert_allclose(H_out, H_ref, equal_nan=True)
    np.testing.assert_allclose(G_out, G_ref, equal_nan=True)
