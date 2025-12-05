from __future__ import annotations

import numpy as np
import numpy.testing as npt
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


def _make_orbits(n: int) -> tuple[list[str], Orbits]:
    orbit_ids = [f"o_{i}" for i in range(n)]
    a = np.linspace(1.0, 2.0, n)
    e = np.linspace(0.0, 0.2, n)
    i = np.zeros(n)
    raan = np.zeros(n)
    ap = np.zeros(n)
    M = np.zeros(n)
    epoch = Timestamp.from_mjd([60000.0] * n, scale="tdb")
    kep = KeplerianCoordinates.from_kwargs(
        a=a,
        e=e,
        i=np.degrees(i),
        raan=np.degrees(raan),
        ap=np.degrees(ap),
        M=np.degrees(M),
        time=epoch,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n),
        frame="ecliptic",
    )
    cart = CartesianCoordinates.from_keplerian(kep)
    orbits = Orbits.from_kwargs(orbit_id=orbit_ids, coordinates=cart)
    return orbit_ids, orbits


def _make_hits_rays_aligned(
    n_hits: int, orbit_ids: list[str]
) -> tuple[OverlapHits, ObservationRays]:
    det_ids = [f"d_{i}" for i in range(n_hits)]
    # Map detections to orbits round-robin
    orbits_for_hits = [orbit_ids[i % len(orbit_ids)] for i in range(n_hits)]

    hits = OverlapHits.from_kwargs(
        det_id=det_ids,
        orbit_id=orbits_for_hits,
        seg_id=np.zeros(n_hits, dtype=np.int32),
        leaf_id=np.zeros(n_hits, dtype=np.int32),
        distance_au=np.zeros(n_hits, dtype=float),
    )

    times = Timestamp.from_mjd(60000.0 + np.arange(n_hits) * 0.001, scale="tdb")
    # Simple rays: observer at origin; unit directions along +x
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
    # Build Observers wrapper with a single code value
    from adam_core.observers.observers import Observers

    observers = Observers.from_kwargs(code=["500"] * n_hits, coordinates=observer)
    rays = ObservationRays.from_kwargs(
        det_id=det_ids,
        orbit_id=[None] * n_hits,
        observer=observers,
        u_x=np.ones(n_hits),
        u_y=np.zeros(n_hits),
        u_z=np.zeros(n_hits),
    )
    return hits, rays


def test_label_anomalies_alignment_and_frame_enforcement():
    orbit_ids, orbits = _make_orbits(3)
    hits, rays = _make_hits_rays_aligned(5, orbit_ids)

    # Success path
    labels = label_anomalies(hits, rays, orbits, max_k=1, chunk_size=4, max_processes=0)
    assert len(labels) > 0

    # Missing ray should raise: build a hits table with det_ids not in rays
    hits_bad = OverlapHits.from_kwargs(
        det_id=[f"x{i}" for i in range(len(hits))],
        orbit_id=hits.orbit_id.to_pylist(),
        seg_id=hits.seg_id.to_numpy(zero_copy_only=False).astype(np.int32),
        leaf_id=hits.leaf_id.to_numpy(zero_copy_only=False).astype(np.int32),
        distance_au=hits.distance_au.to_numpy(zero_copy_only=False).astype(float),
    )
    with pytest.raises(ValueError):
        _ = label_anomalies(hits_bad, rays, orbits)

    # Frame enforcement: rays not ecliptic should raise
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
    with pytest.raises((AssertionError, ValueError)):
        _ = label_anomalies(hits, rays_bad, orbits)


def test_label_anomalies_chunking_padding_and_parallel_equivalence():
    orbit_ids, orbits = _make_orbits(8)
    hits, rays = _make_hits_rays_aligned(17, orbit_ids)

    # Choose small chunk_size to force padding (e.g., 4 → 20 padded)
    labels_serial = label_anomalies(
        hits, rays, orbits, max_k=1, chunk_size=4, max_processes=0
    )
    # Parallel path with same chunk size and K; small process count to avoid resource issues
    labels_parallel = label_anomalies(
        hits, rays, orbits, max_k=1, chunk_size=4, max_processes=2
    )

    # Compare sorted tuples (det_id, orbit_id, variant_id)
    ser = list(
        zip(
            labels_serial.det_id.to_pylist(),
            labels_serial.orbit_id.to_pylist(),
            labels_serial.variant_id.to_pylist(),
        )
    )
    par = list(
        zip(
            labels_parallel.det_id.to_pylist(),
            labels_parallel.orbit_id.to_pylist(),
            labels_parallel.variant_id.to_pylist(),
        )
    )
    assert ser == par
