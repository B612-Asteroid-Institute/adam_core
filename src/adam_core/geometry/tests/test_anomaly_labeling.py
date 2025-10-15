"""
Benchmarks for anomaly labeling throughput.

Use pytest-benchmark to measure steady-state (warm) throughput of
label_anomalies at different hit counts. These run on CPU; GPU runs can be
added by setting JAX platform if available.
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow.compute as pc
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.keplerian import KeplerianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry.anomaly_labeling import label_anomalies
from adam_core.geometry.bvh.query import OverlapHits
from adam_core.geometry.rays import ObservationRays
from adam_core.orbits.orbits import Orbits
from adam_core.time import Timestamp


def _make_orbits(n_orbits: int) -> Orbits:
    orbit_ids = [f"orbit_{i:05d}" for i in range(n_orbits)]
    # Simple low-e elliptical set
    a = np.linspace(1.0, 3.0, n_orbits)
    e = np.full(n_orbits, 0.1)
    i = np.linspace(0.0, np.deg2rad(20.0), n_orbits)
    raan = np.linspace(0.0, 2 * np.pi, n_orbits, endpoint=False)
    ap = np.zeros(n_orbits)
    M = np.zeros(n_orbits)
    epoch = Timestamp.from_mjd([60000.0] * n_orbits, scale="tdb")

    kep = KeplerianCoordinates.from_kwargs(
        a=a,
        e=e,
        i=np.degrees(i),
        raan=np.degrees(raan),
        ap=np.degrees(ap),
        M=np.degrees(M),
        time=epoch,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n_orbits),
        frame="ecliptic",
    )
    cart = CartesianCoordinates.from_keplerian(kep)
    return Orbits.from_kwargs(orbit_id=orbit_ids, coordinates=cart)


def _make_hits_and_rays(
    n_hits: int, n_orbits: int
) -> tuple[OverlapHits, ObservationRays]:
    rng = np.random.default_rng(42)
    # det_ids
    det_ids = [f"det_{i:07d}" for i in range(n_hits)]
    # Map hits to orbits uniformly
    orbit_ids = [f"orbit_{i % n_orbits:05d}" for i in range(n_hits)]
    seg_ids = rng.integers(low=0, high=64, size=n_hits, dtype=np.int32)
    leaf_ids = rng.integers(low=0, high=256, size=n_hits, dtype=np.int32)
    distances = rng.random(n_hits) * 1e-3

    hits = OverlapHits.from_kwargs(
        det_id=det_ids,
        orbit_id=orbit_ids,
        seg_id=seg_ids,
        leaf_id=leaf_ids,
        distance_au=distances,
    )

    # Rays for the same det_ids
    times = Timestamp.from_mjd(60000.0 + rng.random(n_hits) * 1.0, scale="tdb")
    u_vecs = rng.normal(size=(n_hits, 3))
    u_vecs /= np.linalg.norm(u_vecs, axis=1, keepdims=True)
    observer = CartesianCoordinates.from_kwargs(
        x=np.zeros(n_hits),
        y=np.zeros(n_hits),
        z=np.zeros(n_hits),
        vx=np.zeros(n_hits),
        vy=np.zeros(n_hits),
        vz=np.zeros(n_hits),
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n_hits),
        frame="ecliptic",
    )
    rays = ObservationRays.from_kwargs(
        det_id=det_ids,
        orbit_id=[None] * n_hits,
        observer=observer,
        u_x=u_vecs[:, 0],
        u_y=u_vecs[:, 1],
        u_z=u_vecs[:, 2],
    )
    return hits, rays


@pytest.mark.benchmark
@pytest.mark.parametrize("n_hits", [2000, 20000])
def test_label_anomalies_throughput_cpu(benchmark, n_hits):
    """Benchmark label_anomalies steady-state throughput on CPU.

    Sizes are CI-friendly; larger sizes can be enabled via env variable.
    """
    # Allow enabling a larger run (e.g., 100k) via env var
    large = os.environ.get("ADAM_BENCH_LARGE", "0") == "1"
    if large and n_hits == 20000:
        n_hits = 100000

    n_orbits = max(10, min(1000, n_hits // 20))
    orbits = _make_orbits(n_orbits)
    hits, rays = _make_hits_and_rays(n_hits, n_orbits)

    # Warmup to trigger JIT so the measured time is steady-state
    _ = label_anomalies(hits, rays, orbits)

    def _run():
        return label_anomalies(hits, rays, orbits)

    result = benchmark(_run)
    # Attach simple throughput metric
    mean_time = benchmark.stats.stats.mean
    if mean_time > 0:
        benchmark.extra_info["hits_per_second"] = int(n_hits / mean_time)
        benchmark.extra_info["n_hits"] = n_hits


def test_label_anomalies_frame_origin_enforcement_raises(
    bvh_hits, rays_nbody, orbits_synthetic_stratified_ci
):
    # Build rays not in ecliptic to trigger frame assertion
    rays = rays_nbody[:2]
    # Force frame mismatch by changing observer coordinates frame attribute
    rays_bad = rays.set_column(
        "observer",
        rays.observer.set_column(
            "coordinates",
            CartesianCoordinates.from_kwargs(
                x=rays.observer.coordinates.x,
                y=rays.observer.coordinates.y,
                z=rays.observer.coordinates.z,
                vx=rays.observer.coordinates.vx,
                vy=rays.observer.coordinates.vy,
                vz=rays.observer.coordinates.vz,
                time=rays.observer.coordinates.time,
                origin=rays.observer.coordinates.origin,
                frame="equatorial",
            ),
        ),
    )

    if len(bvh_hits) == 0:
        raise ValueError("No hits; cannot test enforcement")

    keep_orb = pc.is_in(
        orbits_synthetic_stratified_ci.orbit_id, pc.unique(bvh_hits.orbit_id)
    )
    orbits_used = orbits_synthetic_stratified_ci.apply_mask(keep_orb)
    with pytest.raises(ValueError):
        _ = label_anomalies(
            bvh_hits, rays_bad, orbits_used, max_k=1, chunk_size=16, max_processes=0
        )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "n_hits,max_k,eccentricity,inc_deg,filter_thr",
    [
        (2000, 1, 0.05, 0.0, None),
        (2000, 3, 0.05, 0.0, None),
        (20000, 1, 0.05, 0.0, None),
        (20000, 3, 0.05, 0.0, None),
        # Higher eccentricity, moderate inclination
        (2000, 3, 0.4, 20.0, None),
        (20000, 3, 0.4, 20.0, None),
        # With strict filtering
        (2000, 3, 0.1, 5.0, 1e-6),
        (20000, 3, 0.1, 5.0, 1e-6),
    ],
)
def test_multi_anomaly_throughput_cpu(
    benchmark, n_hits, max_k, eccentricity, inc_deg, filter_thr
):
    """
    Benchmark multi-anomaly labeling throughput on CPU.

    Compares K=1 vs K=3 performance to measure multi-candidate overhead.
    """
    # Generate synthetic data (reuse helpers above)
    n_orbits = max(10, min(1000, n_hits // 20))
    # Customize orbit population for this scenario
    orbit_ids = [f"orbit_{i:05d}" for i in range(n_orbits)]
    a = np.linspace(1.0, 3.0, n_orbits)
    e = np.full(n_orbits, eccentricity)
    i = np.full(n_orbits, np.deg2rad(inc_deg))
    raan = np.linspace(0.0, 2 * np.pi, n_orbits, endpoint=False)
    ap = np.zeros(n_orbits)
    M = np.zeros(n_orbits)
    epoch = Timestamp.from_mjd([60000.0] * n_orbits, scale="tdb")
    kep = KeplerianCoordinates.from_kwargs(
        a=a,
        e=e,
        i=np.degrees(i),
        raan=np.degrees(raan),
        ap=np.degrees(ap),
        M=np.degrees(M),
        time=epoch,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n_orbits),
        frame="ecliptic",
    )
    cart = CartesianCoordinates.from_keplerian(kep)
    orbits = Orbits.from_kwargs(orbit_id=orbit_ids, coordinates=cart)
    hits, rays = _make_hits_and_rays(n_hits, n_orbits)

    def _run():
        return label_anomalies(
            hits, rays, orbits, max_k=max_k, snap_error_max_au=filter_thr
        )

    result = benchmark(_run)

    # Attach throughput metrics
    mean_time = benchmark.stats.stats.mean
    if mean_time > 0:
        benchmark.extra_info["hits_per_second"] = int(n_hits / mean_time)
        benchmark.extra_info["candidates_per_second"] = int(len(result) / mean_time)
        benchmark.extra_info["n_hits"] = n_hits
        benchmark.extra_info["max_k"] = max_k
        benchmark.extra_info["e"] = eccentricity
        benchmark.extra_info["inc_deg"] = inc_deg
        benchmark.extra_info["filter_thr"] = filter_thr
        benchmark.extra_info["n_candidates"] = len(result)
