import fcntl
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import quivr as qv
import ray

# ASSIST is available in tests for N-body propagation
from adam_assist import ASSISTPropagator

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.keplerian import KeplerianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.geometry import build_bvh_index_from_segments
from adam_core.geometry.anomaly import AnomalyLabels
from adam_core.geometry.anomaly_labeling import label_anomalies
from adam_core.geometry.bvh import BVHIndex, build_bvh_index_from_segments
from adam_core.geometry.bvh.index import (
    BVHIndex,
    build_bvh_index,
    build_bvh_index_from_segments,
)
from adam_core.geometry.bvh.query import OverlapHits, query_bvh
from adam_core.geometry.rays import ObservationRays, ephemeris_to_rays
from adam_core.observers import Observers
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import OrbitPolylineSegments, sample_ellipse_adaptive
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils.helpers.orbits import make_real_orbits

# Canonical epoch for geometry completeness tests
EPOCH_MJD = 60000.0

# Session-singleton for CI observers (90-day fixed span)
_OBSERVERS_CI_TIMESPAN_SINGLETON: Observers | None = None
_OBSERVERS_SINGLETONS: dict[tuple[int, tuple[str, ...]], Observers] = {}


@pytest.fixture(scope="session")
def _geom_testdata_dir() -> Path:
    return Path(__file__).parent / "testdata"


############################## Synthetic populations (CI) ##############################


def _make_orbits_for_class(
    orbit_class: str, n: int, *, seed: int = 42, epoch_mjd: float = EPOCH_MJD
) -> Orbits:
    """
    Create a synthetic orbit population targeting a specific calc_orbit_class code.

    Supported classes: IMB, MBA, OMB, TJN, CEN, TNO,
    plus additional synthetic categories:
    AMO (Amor), APO (Apollo), ATE (Aten), ATI (Atira),
    MCO (Mars-crossers), HUN (Hungaria), HIL (Hilda),
    JFC (Jupiter-family Comet), HTC (Halley-type Comet), LPC (Long-period Comet),
    UTN (Uranus Trojans), NTN (Neptune Trojans).
    """
    rng = np.random.default_rng(seed)

    if orbit_class == "IMB":  # a < 2.0, q > 1.666
        a = rng.uniform(1.75, 1.99, size=n)
        e_max = 1.0 - (1.666 / a)
        e = rng.uniform(0.0, np.minimum(0.4, e_max))
        i_deg = rng.uniform(0.0, 25.0, size=n)
    elif orbit_class == "MBA":  # 2.0 < a < 3.2, q > 1.666
        a = rng.uniform(2.05, 3.15, size=n)
        e_max = 1.0 - (1.666 / a)
        e = rng.uniform(0.0, np.minimum(0.4, e_max))
        i_deg = rng.uniform(0.0, 25.0, size=n)
    elif orbit_class == "OMB":  # 3.2 < a < 4.6
        a = rng.uniform(3.25, 4.55, size=n)
        e = rng.uniform(0.0, 0.35, size=n)
        i_deg = rng.uniform(0.0, 30.0, size=n)
    elif orbit_class == "TJN":  # 4.6 < a < 5.5 and e < 0.3
        a = rng.normal(5.2, 0.05, size=n)
        a = np.clip(a, 4.7, 5.4)
        e = rng.uniform(0.0, 0.25, size=n)
        i_deg = rng.uniform(0.0, 40.0, size=n)
    elif orbit_class == "CEN":  # 5.5 < a < 30.1
        a = rng.uniform(5.6, 29.9, size=n)
        e = rng.uniform(0.0, 0.6, size=n)
        i_deg = rng.uniform(0.0, 40.0, size=n)
    elif orbit_class == "TNO":  # a > 30.1
        a = rng.uniform(30.2, 100.0, size=n)
        e = rng.uniform(0.0, 0.5, size=n)
        i_deg = rng.uniform(0.0, 40.0, size=n)
    elif orbit_class == "AMO":  # Amor: q just outside Earth (1.017 < q < ~1.3), a > 1
        a = rng.uniform(1.2, 2.5, size=n)
        q_target = rng.uniform(1.05, 1.3, size=n)
        e = 1.0 - (q_target / a)
        e = np.clip(e, 0.0, 0.6)
        i_deg = rng.uniform(0.0, 30.0, size=n)
    elif orbit_class == "APO":  # Apollo: a > 1, q < 1.017
        a = rng.uniform(1.1, 2.5, size=n)
        q_target = rng.uniform(0.7, 1.0, size=n)
        e = 1.0 - (q_target / a)
        e = np.clip(e, 0.05, 0.7)
        i_deg = rng.uniform(0.0, 35.0, size=n)
    elif orbit_class == "ATE":  # Aten: a < 1 AU, Q > 0.983
        a = rng.uniform(0.6, 1.0, size=n)
        # Ensure Q = a(1+e) > ~1, pick modest e
        e = rng.uniform(0.1, 0.5, size=n)
        i_deg = rng.uniform(0.0, 30.0, size=n)
    elif orbit_class == "ATI":  # Atira: entirely interior to Earth, Q < 0.983
        a = rng.uniform(0.4, 0.9, size=n)
        # Choose e so that Q=a(1+e) < 0.983 -> e_max = 0.983/a - 1
        e_max = np.maximum(0.0, 0.983 / a - 1.0)
        e = rng.uniform(0.0, np.minimum(0.4, e_max))
        i_deg = rng.uniform(0.0, 25.0, size=n)
    elif orbit_class == "MCO":  # Mars-crossers: q < 1.666
        a = rng.uniform(1.6, 3.0, size=n)
        q_target = rng.uniform(1.2, 1.65, size=n)
        e = 1.0 - (q_target / a)
        e = np.clip(e, 0.05, 0.5)
        i_deg = rng.uniform(0.0, 30.0, size=n)
    elif orbit_class == "HUN":  # Hungaria: 1.78 < a < 2.0, low e, high i
        a = rng.uniform(1.78, 2.0, size=n)
        e = rng.uniform(0.0, 0.18, size=n)
        i_deg = rng.uniform(16.0, 34.0, size=n)
    elif orbit_class == "HIL":  # Hilda: a ~ 3.9 (3:2), moderate e, low i
        a = rng.uniform(3.7, 4.1, size=n)
        e = rng.uniform(0.0, 0.3, size=n)
        i_deg = rng.uniform(0.0, 20.0, size=n)
    elif (
        orbit_class == "JFC"
    ):  # Jupiter-family comets: 2<T_J<3 (approx: a 3-6, e 0.3-0.7)
        a = rng.uniform(3.0, 6.0, size=n)
        e = rng.uniform(0.3, 0.7, size=n)
        i_deg = rng.uniform(0.0, 30.0, size=n)
    elif orbit_class == "HTC":  # Halley-type comets: high e, large a, broad i
        a = rng.uniform(10.0, 30.0, size=n)
        e = rng.uniform(0.7, 0.95, size=n)
        i_deg = rng.uniform(0.0, 160.0, size=n)
    elif orbit_class == "LPC":  # Long-period comets: very large a, e near 1
        a = rng.uniform(100.0, 1000.0, size=n)
        e = rng.uniform(0.9, 0.999, size=n)
        i_deg = rng.uniform(0.0, 180.0, size=n)
    elif orbit_class == "UTN":  # Uranus Trojans
        a = rng.uniform(19.0, 19.6, size=n)
        e = rng.uniform(0.0, 0.1, size=n)
        i_deg = rng.uniform(0.0, 30.0, size=n)
    elif orbit_class == "NTN":  # Neptune Trojans
        a = rng.uniform(29.7, 30.5, size=n)
        e = rng.uniform(0.0, 0.1, size=n)
        i_deg = rng.uniform(0.0, 30.0, size=n)
    else:
        raise ValueError(f"Unknown orbit_class: {orbit_class}")

    raan = rng.uniform(0.0, 360.0, size=n)
    ap = rng.uniform(0.0, 360.0, size=n)
    M = rng.uniform(0.0, 360.0, size=n)

    times = Timestamp.from_mjd([epoch_mjd] * n, scale="tdb")
    kep = KeplerianCoordinates.from_kwargs(
        a=a,
        e=e,
        i=i_deg,
        raan=raan,
        ap=ap,
        M=M,
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n),
        frame="ecliptic",
    )
    cart = CartesianCoordinates.from_keplerian(kep)
    return Orbits.from_kwargs(
        orbit_id=[f"{orbit_class}_{i:05d}" for i in range(n)],
        coordinates=cart,
    )


############################## Persistent caching helpers ##############################


@pytest.fixture(scope="session")
def _fixture_cache_root(request) -> Path:
    """Path to store fixture caches."""
    return request.config.cache.makedir("adam_core_geometry_bvh_tests")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@contextmanager
def _file_lock(lock_path: Path):
    """Simple POSIX file lock to serialize writers across processes.

    Blocks until the lock is acquired. Releases automatically on exit.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))
        f.flush()
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@pytest.fixture(scope="session")
def observers_ci_timespan(_fixture_cache_root: Path) -> Observers:
    """90-day CI observers, persisted under pytest cache root."""
    root = Path(_fixture_cache_root) / "observers"
    _ensure_dir(root)
    path = root / "ci_90d.parquet"
    lock = path.parent / (path.name + ".lock")
    with _file_lock(lock):
        if path.exists():
            return Observers.from_parquet(path)
        # Build deterministically (identical semantics to previous implementation)
        start = EPOCH_MJD
        days = np.arange(0, 90, dtype=int)
        times = Timestamp.from_mjd(start + days, scale="tdb")
        stations = ["X05", "T08", "I41"]
        codes = [stations[i % len(stations)] for i in range(len(times))]
        observers = Observers.from_codes(times=times, codes=codes)
        observers.to_parquet(path)
        return observers


# ----------------------------- Lightweight fixtures -----------------------------


@pytest.fixture(scope="function")
def small_orbits() -> Orbits:
    n = 5
    times = Timestamp.from_mjd([59000.0] * n, scale="tdb")
    import numpy as np

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


@pytest.fixture(scope="session")
def segments_aabbs(_fixture_cache_root: Path, orbits_synthetic_stratified_ci: Orbits):
    """Segments for tests that used to depend on simple_orbits (1.0 arcmin/256)."""
    root = Path(_fixture_cache_root) / "segments"
    _ensure_dir(root)
    path = root / "segments_aabbs.parquet"
    lock = path.parent / (path.name + ".lock")
    from adam_core.orbits.polyline import OrbitPolylineSegments

    with _file_lock(lock):
        if path.exists():
            return OrbitPolylineSegments.from_parquet(path)
        # Use fine segmentation to preserve query recall at 0.65 arcmin guard
        _, segs = sample_ellipse_adaptive(
            orbits_synthetic_stratified_ci,
            max_chord_arcmin=5.0,
            max_segments_per_orbit=512,
        )
        segs.to_parquet(path)
        return segs


@pytest.fixture(scope="session")
def orbits_synthetic_stratified_ci(_fixture_cache_root: Path) -> Orbits:
    """Synthetic stratified population, persisted under pytest cache root."""
    name = "synthetic_stratified_ci"
    root = Path(_fixture_cache_root) / "orbits"
    _ensure_dir(root)
    path = root / f"{name}.parquet"
    lock = path.parent / (path.name + ".lock")
    with _file_lock(lock):
        if path.exists():
            return Orbits.from_parquet(path)
        # Build identical to previous composition
        synth_strat = qv.concatenate(
            [
                _make_orbits_for_class("IMB", 30, seed=101),
                _make_orbits_for_class("MBA", 25, seed=102),
                _make_orbits_for_class("OMB", 15, seed=103),
                _make_orbits_for_class("TJN", 10, seed=104),
                _make_orbits_for_class("CEN", 10, seed=105),
                _make_orbits_for_class("TNO", 10, seed=106),
                _make_orbits_for_class("AMO", 10, seed=201),
                _make_orbits_for_class("APO", 10, seed=202),
                _make_orbits_for_class("ATE", 10, seed=203),
                _make_orbits_for_class("ATI", 10, seed=204),
                _make_orbits_for_class("MCO", 10, seed=205),
                _make_orbits_for_class("HUN", 10, seed=206),
                _make_orbits_for_class("HIL", 10, seed=207),
                _make_orbits_for_class("JFC", 10, seed=208),
                _make_orbits_for_class("HTC", 10, seed=209),
                _make_orbits_for_class("LPC", 10, seed=210),
                _make_orbits_for_class("UTN", 10, seed=211),
                _make_orbits_for_class("NTN", 10, seed=212),
            ],
            defrag=True,
        )
        synth_strat.to_parquet(path)
        return synth_strat


@pytest.fixture(scope="session")
def ephemeris_2b(
    _fixture_cache_root: Path,
    orbits_synthetic_stratified_ci: Orbits,
    observers_ci_timespan: Observers,
) -> Ephemeris:
    """Two-body ephemeris persisted under pytest cache root (unchanged semantics)."""
    root = Path(_fixture_cache_root) / "ephemeris" / "2b"
    _ensure_dir(root)
    path = root / "synthetic_stratified_ci.parquet"
    lock = path.parent / (path.name + ".lock")
    with _file_lock(lock):
        if path.exists():
            return Ephemeris.from_parquet(path)
        # Generate 2-body ephemeris with same approach as before (ray workers)
        times = observers_ci_timespan.coordinates.time
        initialize_use_ray(num_cpus=None)

        @ray.remote
        def _ephem_worker(orb_slice: Orbits):
            propagated = propagate_2body(orb_slice, times)
            base_times = observers_ci_timespan.coordinates.time
            base_codes = observers_ci_timespan.code.to_pylist()
            reps = len(orb_slice)
            import numpy as _np

            times_all = _np.tile(base_times.mjd().to_numpy(zero_copy_only=False), reps)
            codes_all = base_codes * reps
            obs_tiled = Observers.from_codes(
                times=Timestamp.from_mjd(times_all, scale=base_times.scale),
                codes=codes_all,
            )
            return generate_ephemeris_2body(propagated, obs_tiled)

        futures = []
        chunk_size = 100
        for i in range(0, len(orbits_synthetic_stratified_ci), chunk_size):
            futures.append(
                _ephem_worker.remote(orbits_synthetic_stratified_ci[i : i + chunk_size])
            )
        ephems = ray.get(futures)
        ephem = qv.concatenate(ephems, defrag=True)
        ephem.to_parquet(path)
        return ephem


@pytest.fixture(scope="session")
def rays_2b(
    _fixture_cache_root: Path, ephemeris_2b: Ephemeris, observers_ci_timespan: Observers
) -> ObservationRays:
    """Rays derived from 2b ephemeris; persisted under pytest cache root."""
    root = Path(_fixture_cache_root) / "rays" / "2b"
    _ensure_dir(root)
    path = root / "synthetic_stratified_ci.parquet"
    lock = path.parent / (path.name + ".lock")
    with _file_lock(lock):
        if path.exists():
            return ObservationRays.from_parquet(path)
        codes = observers_ci_timespan.code.to_pylist()
        det_ids = [
            f"{ephemeris_2b.orbit_id[i].as_py()}:{codes[i % len(codes)]}:{i}"
            for i in range(len(ephemeris_2b))
        ]
        rays = ephemeris_to_rays(ephemeris_2b, det_id=det_ids)
        rays.to_parquet(path)
        return rays


@pytest.fixture(scope="session")
def ephemeris_nb(
    _fixture_cache_root: Path,
    orbits_synthetic_stratified_ci: Orbits,
    observers_ci_timespan: Observers,
) -> Ephemeris:
    """N-body ephemeris using ASSIST (end-to-end), persisted under pytest cache root."""
    root = Path(_fixture_cache_root) / "ephemeris" / "nb"
    _ensure_dir(root)
    path = root / "synthetic_stratified_ci.parquet"
    lock = path.parent / (path.name + ".lock")
    with _file_lock(lock):
        if path.exists():
            return Ephemeris.from_parquet(path)
        propagator = ASSISTPropagator()
        ephem = propagator.generate_ephemeris(
            orbits_synthetic_stratified_ci,
            observers_ci_timespan,
            max_processes=1,
            chunk_size=10,
        )
        ephem.to_parquet(path)
        return ephem


@pytest.fixture(scope="session")
def rays_nbody(
    _fixture_cache_root: Path, ephemeris_nb: Ephemeris, observers_ci_timespan: Observers
) -> ObservationRays:
    """Rays derived from N-body ephemeris; persisted under pytest cache root."""
    root = Path(_fixture_cache_root) / "rays" / "nb"
    _ensure_dir(root)
    path = root / "synthetic_stratified_ci.parquet"
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return ObservationRays.from_parquet(path)
        codes = observers_ci_timespan.code.to_pylist()
        det_ids = [
            f"{ephemeris_nb.orbit_id[i].as_py()}:{codes[i % len(codes)]}:{i}"
            for i in range(len(ephemeris_nb))
        ]
        rays = ephemeris_to_rays(ephemeris_nb, det_id=det_ids)
        rays.to_parquet(path)
        return rays


@pytest.fixture(scope="session")
def index_optimal(
    _fixture_cache_root: Path,
    orbits_synthetic_stratified_ci: Orbits,
    segments_aabbs: OrbitPolylineSegments,
) -> BVHIndex:
    """BVH index built from persisted segments with optimal params; atomic write to dir."""

    # Build or load index directory
    idx_root = Path(_fixture_cache_root) / "indices" / "synthetic_stratified_ci" / "l64"
    out_dir = idx_root / "bvh_index"
    _ensure_dir(idx_root)
    lock = idx_root / ".lock"
    with _file_lock(lock):
        if out_dir.exists():
            return BVHIndex.from_parquet(str(out_dir))
        # Build and atomically persist
        index = build_bvh_index(
            orbits_synthetic_stratified_ci,
            max_chord_arcmin=5.0,
            guard_arcmin=1.0,
            max_leaf_size=64,
            max_processes=1,
            max_segments_per_orbit=512,
            epsilon_n_au=1e-9,
            padding_method="baseline",
        )
        index.to_parquet(str(out_dir))
        return index


@pytest.fixture(scope="session", params=[(64, 16), (256, 32)])
def index_small(segments_aabbs, request) -> BVHIndex:
    N, leaf_size = request.param
    idx = build_bvh_index_from_segments(segments_aabbs[:N], max_leaf_size=leaf_size)
    return idx


@pytest.fixture(scope="session")
def bvh_hits(_fixture_cache_root, index_optimal, rays_nbody):
    idx_root = Path(_fixture_cache_root) / "hits" / "nb"
    path = idx_root / "hits.parquet"
    _ensure_dir(idx_root)
    lock = idx_root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return OverlapHits.from_parquet(str(path))
        hits, _ = query_bvh(
            index_optimal,
            rays_nbody,
            guard_arcmin=0.65,
            batch_size=16384,
            max_processes=1,
            window_size=32768,
        )
        hits.to_parquet(str(path))
        return hits


@pytest.fixture(scope="session")
def anomaly_labels(
    _fixture_cache_root, bvh_hits, rays_nbody, orbits_synthetic_stratified_ci
):
    path = (
        Path(_fixture_cache_root) / "anomaly_labels" / "nb" / "anomaly_labels.parquet"
    )
    _ensure_dir(path.parent)
    lock = path.parent / ".lock"
    with _file_lock(lock):
        if path.exists():
            return AnomalyLabels.from_parquet(str(path))
        labels = label_anomalies(
            bvh_hits,
            rays_nbody,
            orbits_synthetic_stratified_ci,
            max_k=1,
            chunk_size=4096,
            max_processes=1,
        )
        labels.to_parquet(str(path))
        return labels


# ---------------------------- Noise-augmented fixtures ----------------------------


@pytest.fixture(scope="session")
def ephemeris_noise(_fixture_cache_root: Path, ephemeris_nb: Ephemeris) -> Ephemeris:
    """Create a noise Ephemeris per unique exposure in the N-body ephemeris.

    For each (observer.code, exact MJD) group in ephemeris_nb:
      - Compute a boresight from signal directions (from ephemeris rays later)
      - Define a spherical cap that encloses signal
      - Sample ~10 per deg^2 uniformly within that cap to create a noise ephemeris row-set
    """
    # Cache under pytest cache root
    root = Path(_fixture_cache_root) / "ephemeris" / "noise"
    _ensure_dir(root)
    path = root / "synthetic_stratified_ci.parquet"
    lock = path.parent / (path.name + ".lock")
    with _file_lock(lock):
        if path.exists():
            return Ephemeris.from_parquet(path)

        if len(ephemeris_nb) == 0:
            return Ephemeris.empty()

        # Build exposure key from observers in ephemeris
        codes = ephemeris_nb.coordinates.origin.code
        mjd = ephemeris_nb.coordinates.time.mjd()
        sep = pa.scalar("||", type=pa.large_string())
        key = pc.binary_join_element_wise(
            pc.cast(codes, pa.large_string()),
            sep,
            pc.cast(pc.cast(mjd, pa.large_string()), pa.large_string()),
        )
        uniq_keys = pc.unique(key)

        noise_per_sqdeg = 10.0
        rng = np.random.default_rng(123)
        out_chunks: list[Ephemeris] = []

        # Derive boresight from ephemeris LOS approximated by direction to object in SSB ecliptic frame
        # Here, we approximate LOS by the unit vector of spherical coordinates converted to Cartesian
        # (rho=1 at given lon/lat), since ephemeris_to_rays will do the proper transform later.
        # We just need a representative direction cluster per exposure.
        lon = ephemeris_nb.coordinates.lon.to_numpy()
        lat = ephemeris_nb.coordinates.lat.to_numpy()
        u_all = np.column_stack(
            [
                np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)),
                np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
                np.sin(np.deg2rad(lat)),
            ]
        )

        for k in uniq_keys.to_pylist():
            mask = pc.equal(key, k)
            idx = np.nonzero(np.asarray(mask))[0]
            if idx.size == 0:
                continue
            u_sig = u_all[idx]
            u_sig /= np.linalg.norm(u_sig, axis=1, keepdims=True) + 1e-30
            u0 = u_sig.mean(axis=0)
            u0 /= np.linalg.norm(u0) + 1e-30
            dot = np.clip(u_sig @ u0, -1.0, 1.0)
            theta_max = float(np.arccos(np.min(dot)))
            A_sr = 2.0 * np.pi * (1.0 - np.cos(theta_max))
            A_deg2 = A_sr * (180.0 / np.pi) ** 2
            n_noise = int(np.ceil(noise_per_sqdeg * max(A_deg2, 1e-6)))
            if n_noise <= 0:
                continue
            # Sample spherical cap around u0
            z = rng.uniform(np.cos(theta_max), 1.0, size=n_noise)
            phi = rng.uniform(0.0, 2.0 * np.pi, size=n_noise)
            theta = np.arccos(z)
            a = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, u0)) > 0.9:
                a = np.array([0.0, 1.0, 0.0])
            v = np.cross(u0, a)
            v /= np.linalg.norm(v) + 1e-30
            w = np.cross(u0, v)
            u = np.cos(theta)[:, None] * u0[None, :] + np.sin(theta)[:, None] * (
                np.cos(phi)[:, None] * v[None, :] + np.sin(phi)[:, None] * w[None, :]
            )
            # Convert unit vectors to spherical lon/lat in degrees
            x, y, zc = u[:, 0], u[:, 1], u[:, 2]
            lon_deg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
            lat_deg = np.degrees(np.arcsin(np.clip(zc, -1.0, 1.0)))
            # Build SphericalCoordinates for noise
            from adam_core.coordinates.origin import Origin
            from adam_core.coordinates.spherical import SphericalCoordinates

            # Build repeated time and origin arrays to length n_noise
            time_rep = ephemeris_nb.coordinates.time.take(
                pa.array([idx[0]] * n_noise, type=pa.int64())
            )
            origin_rep = Origin.from_kwargs(
                code=ephemeris_nb.coordinates.origin.code.take(
                    pa.array([idx[0]] * n_noise, type=pa.int64())
                )
            )
            sph = SphericalCoordinates.from_kwargs(
                rho=np.ones(n_noise),
                lon=lon_deg,
                lat=lat_deg,
                vrho=np.zeros(n_noise),
                vlon=np.zeros(n_noise),
                vlat=np.zeros(n_noise),
                time=time_rep,
                origin=origin_rep,
                frame="equatorial",
            )
            ephem_chunk = Ephemeris.from_kwargs(
                orbit_id=[f"noise_{k}_{i:05d}" for i in range(n_noise)],
                object_id=[None] * n_noise,
                coordinates=sph,
                alpha=[None] * n_noise,
                light_time=[None] * n_noise,
                aberrated_coordinates=None,
            )
            out_chunks.append(ephem_chunk)

        if not out_chunks:
            ephem_out = Ephemeris.empty()
        else:
            ephem_out = qv.concatenate(out_chunks, defrag=True)
        ephem_out.to_parquet(path)
        return ephem_out


@pytest.fixture(scope="session")
def rays_noise(
    _fixture_cache_root: Path, ephemeris_noise: Ephemeris
) -> ObservationRays:
    """Rays derived from noise ephemeris; persisted under pytest cache root."""
    root = Path(_fixture_cache_root) / "rays" / "noise"
    _ensure_dir(root)
    path = root / "synthetic_stratified_ci.parquet"
    lock = path.parent / (path.name + ".lock")
    with _file_lock(lock):
        if path.exists():
            return ObservationRays.from_parquet(path)
        if len(ephemeris_noise) == 0:
            return ObservationRays.empty()
        # Convert noise ephemeris to rays (det_id from orbit_id)
        rays = ephemeris_to_rays(
            ephemeris_noise,
            det_id=ephemeris_noise.orbit_id.to_pylist(),
            max_processes=4,
            chunk_size=10_000,
        )
        rays.to_parquet(path)
        return rays


@pytest.fixture(scope="session")
def rays_nbody_with_noise(
    _fixture_cache_root: Path, rays_nbody: ObservationRays, rays_noise: ObservationRays
) -> ObservationRays:
    """Concatenate signal and noise rays; persisted under pytest cache root."""
    if len(rays_noise) == 0:
        return rays_nbody
    root = Path(_fixture_cache_root) / "rays" / "nb_noise"
    _ensure_dir(root)
    path = root / "synthetic_stratified_ci.parquet"
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return ObservationRays.from_parquet(path)
        rays = qv.concatenate([rays_nbody, rays_noise], defrag=True)
        rays.to_parquet(path)
        return rays


@pytest.fixture(scope="session")
def bvh_hits_with_noise(
    _fixture_cache_root: Path,
    index_optimal: BVHIndex,
    rays_nbody_with_noise: ObservationRays,
):
    if len(rays_nbody_with_noise) == 0:
        return OverlapHits.empty()
    idx_root = Path(_fixture_cache_root) / "hits" / "nb_noise"
    path = idx_root / "hits.parquet"
    _ensure_dir(idx_root)
    lock = idx_root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return OverlapHits.from_parquet(str(path))
        hits, _ = query_bvh(
            index_optimal,
            rays_nbody_with_noise,
            guard_arcmin=0.65,
            batch_size=16384,
            max_processes=1,
            window_size=32768,
        )
        hits.to_parquet(str(path))
        return hits


@pytest.fixture(scope="session")
def anomaly_labels_with_noise(
    _fixture_cache_root: Path,
    bvh_hits_with_noise: OverlapHits,
    rays_nbody_with_noise: ObservationRays,
    orbits_synthetic_stratified_ci: Orbits,
):
    if len(bvh_hits_with_noise) == 0:
        return AnomalyLabels.empty()
    path = (
        Path(_fixture_cache_root)
        / "anomaly_labels"
        / "nb_noise"
        / "anomaly_labels.parquet"
    )
    _ensure_dir(path.parent)
    lock = path.parent / ".lock"
    with _file_lock(lock):
        if path.exists():
            return AnomalyLabels.from_parquet(str(path))
        labels = label_anomalies(
            bvh_hits_with_noise,
            rays_nbody_with_noise,
            orbits_synthetic_stratified_ci,
            max_k=1,
            chunk_size=4096,
            max_processes=1,
        )
        labels.to_parquet(str(path))
        return labels
