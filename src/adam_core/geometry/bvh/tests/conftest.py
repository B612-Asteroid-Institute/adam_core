import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

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
from adam_core.geometry.bvh import BVHIndex, build_bvh_index_from_segments, ObservationRays, ephemeris_to_rays
from adam_core.observers.observers import Observerss
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils.helpers.orbits import make_real_orbits
from adam_core.observers import Observers

# Canonical epoch for geometry completeness tests
EPOCH_MJD = 60000.0

# Session-singleton for CI observers (90-day fixed span)
_OBSERVERS_CI_TIMESPAN_SINGLETON: Observers | None = None
_OBSERVERS_SINGLETONS: dict[tuple[int, tuple[str, ...]], Observers] = {}


@pytest.fixture(scope="session")
def large_orbits():
    """Session-scoped large orbits dataset for reuse."""
    n = 5000
    times = Timestamp.from_mjd([59000.0] * n, scale="tdb")
    import numpy as np

    rng = np.random.default_rng(42)
    coords = CartesianCoordinates.from_kwargs(
        x=rng.uniform(0.5, 2.5, size=n),
        y=rng.uniform(-0.2, 0.2, size=n),
        z=rng.uniform(-0.3, 0.3, size=n),
        vx=rng.uniform(-0.002, 0.002, size=n),
        vy=rng.uniform(0.012, 0.022, size=n),
        vz=rng.uniform(-0.002, 0.002, size=n),
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(
        orbit_id=[f"bench_{i}" for i in range(n)], coordinates=coords
    )


@pytest.fixture(scope="session")
def large_segments_aabb(large_orbits):
    """Session-scoped large segments with AABBs for reuse and slicing."""
    _, segments = sample_ellipse_adaptive(large_orbits, max_chord_arcmin=2.0)
    return compute_segment_aabbs(segments, guard_arcmin=1.0, epsilon_n_au=1e-6)


import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry import build_bvh_index_from_segments
from adam_core.geometry.rays import rays_from_detections
from adam_core.observations.detections import PointSourceDetections
from adam_core.observations.exposures import Exposures
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.time import Timestamp


@pytest.fixture(scope="session")
def _geom_testdata_dir() -> Path:
    return Path(__file__).parent / "testdata"


@pytest.fixture(scope="session")
def simple_orbits(_geom_testdata_dir):
    from adam_core.orbits.orbits import Orbits

    return Orbits.from_parquet(_geom_testdata_dir / "simple_orbits.parquet")


# segments_aabbs fixture moved to bvh/tests/conftest.py


@pytest.fixture(scope="session")
def bvh_index(_geom_testdata_dir):

    return BVHIndex.from_parquet(str(_geom_testdata_dir / "bvh_index"))


# rays fixture moved to bvh/tests/conftest.py


@pytest.fixture(scope="session")
def rays_many(_geom_testdata_dir):
    from adam_core.geometry.rays import ObservationRays

    return ObservationRays.from_parquet(_geom_testdata_dir / "rays_many.parquet")


@pytest.fixture(scope="session")
def bvh_index_loaded(_geom_testdata_dir):
    return BVHIndex.from_parquet(str(_geom_testdata_dir / "bvh_index"))


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


# synthetic_orbits_stratified_ci fixture moved to bvh/tests/conftest.py


@pytest.fixture(scope="session")
def synthetic_orbits_TJN_ci() -> Orbits:
    """Load CI-sized TJN-only synthetic set from cache."""
    name = "synthetic_TJN_ci"
    path = _fixture_cache_root() / "populations" / name / "orbits.parquet"
    return Orbits.from_parquet(path)


@pytest.fixture(scope="session")
def synthetic_orbits_TNO_ci() -> Orbits:
    """Load CI-sized TNO-only synthetic set from cache."""
    name = "synthetic_TNO_ci"
    path = _fixture_cache_root() / "populations" / name / "orbits.parquet"
    return Orbits.from_parquet(path)


@pytest.fixture(scope="session")
def synthetic_orbits_CEN_ci() -> Orbits:
    """Load CI-sized CEN-only synthetic set from cache."""
    name = "synthetic_CEN_ci"
    path = _fixture_cache_root() / "populations" / name / "orbits.parquet"
    return Orbits.from_parquet(path)


############################## Real populations (CI) ##############################


@pytest.fixture(scope="session")
def real_orbits_ci() -> Orbits:
    """Load real orbit sample (~27) from cache (populated from packaged data)."""
    name = "real_ci"
    path = _fixture_cache_root() / "populations" / name / "orbits.parquet"
    return Orbits.from_parquet(path)


@pytest.fixture(scope="session")
def real_orbits_10() -> Orbits:
    """Load real orbit sample (10) from cache."""
    name = "real_10"
    path = _fixture_cache_root() / "populations" / name / "orbits.parquet"
    return Orbits.from_parquet(path)


############################## Persistent caching helpers ##############################


def _fixture_cache_root() -> Path:
    return Path(__file__).parent / "cache"


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


def cache_orbits(orbits: Orbits, name: str) -> Path:
    """Persist orbits to parquet under tests cache and return path."""
    root = _fixture_cache_root() / "populations" / name
    _ensure_dir(root)
    path = root / "orbits.parquet"
    orbits.to_parquet(path)
    return path


def load_cached_orbits(name: str) -> Orbits:
    path = _fixture_cache_root() / "populations" / name / "orbits.parquet"
    return Orbits.from_parquet(path)


def cache_segments(
    orbits: Orbits,
    name: str,
    *,
    max_chord_arcmin: float = 2.0,
    guard_arcmin: float = 1.0,
) -> Path:
    """Compute segments, persist to parquet, and return path."""
    root = _fixture_cache_root() / "segments" / name
    _ensure_dir(root)
    _, segs = sample_ellipse_adaptive(
        orbits, max_chord_arcmin=max_chord_arcmin, max_segments_per_orbit=32768
    )
    path = root / "segments.parquet"
    segs.to_parquet(path)
    return path


def load_cached_segments(name: str):
    from adam_core.orbits.polyline import OrbitPolylineSegments

    path = _fixture_cache_root() / "segments" / name / "segments.parquet"
    return OrbitPolylineSegments.from_parquet(path)


# ------------------------ Tag formatting (shared) ------------------------
def format_segments_tag(
    *,
    max_chord_arcmin: float,
    max_segments_per_orbit: int,
) -> str:
    # Segments depend only on sampling params; guard/epsilon/padding apply at BVH build
    return f"ch{max_chord_arcmin:.2f}_ms{int(max_segments_per_orbit)}"


def format_index_tag(
    *,
    max_leaf_size: int,
    max_chord_arcmin: float,
    guard_arcmin: float,
    padding_method: str,
    epsilon_n_au: float,
    max_segments_per_orbit: int,
) -> str:
    return (
        f"l{int(max_leaf_size)}_ch{max_chord_arcmin:.2f}_g{guard_arcmin:.2f}_"
        f"p{padding_method}_ms{int(max_segments_per_orbit)}_eps{epsilon_n_au:.0e}"
    )


def cache_bvh_index_from_segments(
    segments, name: str, *, max_leaf_size: int = 8
) -> Path:
    """Build BVHIndex from provided segments, persist with to_parquet, and return dir path."""
    root = _fixture_cache_root() / "indices" / name / f"l{max_leaf_size}"
    out_dir = root / "bvh_index"
    _ensure_dir(out_dir)
    idx = build_bvh_index_from_segments(segments, max_leaf_size=max_leaf_size)
    idx.to_parquet(str(out_dir))
    return out_dir


def load_cached_bvh_index(name: str, *, max_leaf_size: int = 8) -> BVHIndex:
    dir_path = (
        _fixture_cache_root() / "indices" / name / f"l{max_leaf_size}" / "bvh_index"
    )
    segs_path = _fixture_cache_root() / "segments" / name / "segments.parquet"
    lock = dir_path.parent / ".lock"
    with _file_lock(lock):
        if not dir_path.exists():
            # Ensure segments exist
            if not segs_path.exists():
                orbits = load_cached_orbits(name)
                cache_segments(orbits, name)
            segments = load_cached_segments(name)
            cache_bvh_index_from_segments(segments, name, max_leaf_size=max_leaf_size)
        return BVHIndex.from_parquet(str(dir_path))


def cache_rays(rays: ObservationRays, name: str) -> Path:
    root = _fixture_cache_root() / "rays" / name
    _ensure_dir(root)
    path = root / "rays.parquet"
    rays.to_parquet(path)
    return path


def load_cached_rays(name: str) -> ObservationRays:
    path = _fixture_cache_root() / "rays" / name / "rays.parquet"
    return ObservationRays.from_parquet(path)


def cache_ephemeris(ephem: Ephemeris, name: str) -> Path:
    root = _fixture_cache_root() / "ephemeris" / name
    _ensure_dir(root)
    path = root / "ephemeris.parquet"
    ephem.to_parquet(path)
    return path


def load_cached_ephemeris(name: str) -> Ephemeris:
    path = _fixture_cache_root() / "ephemeris" / name / "ephemeris.parquet"
    return Ephemeris.from_parquet(path)


############################## Cache build orchestrator ##############################


def _random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def cache_noise_rays_for_observers(
    observers: Observers,
    name: str,
    density: float = 1.0,
    seed: int = 42,
) -> Path:
    """Generate time-aligned random rays (noise) for given observers and persist.

    density=1.0 creates one noise ray per observer entry; density>1 tiles the set and
    adds a fractional subset; density<1 samples a subset.
    """
    root = _fixture_cache_root() / "noise_rays" / name
    _ensure_dir(root)
    tag = f"x{int(density*100):03d}"
    path = root / f"rays_{tag}.parquet"

    import numpy as np

    rng = np.random.default_rng(seed)
    base_n = len(observers)
    base_times_mjd = observers.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    base_codes = observers.code.to_pylist()
    coords = observers.coordinates

    reps = int(np.floor(density))
    frac = float(density - reps)

    times_list = []
    codes_list = []
    x_list = []
    y_list = []
    z_list = []
    vx_list = []
    vy_list = []
    vz_list = []

    for _ in range(max(1, reps)):
        times_list.append(base_times_mjd)
        codes_list.extend(base_codes)
        x_list.append(coords.x.to_numpy(zero_copy_only=False))
        y_list.append(coords.y.to_numpy(zero_copy_only=False))
        z_list.append(coords.z.to_numpy(zero_copy_only=False))
        vx_list.append(coords.vx.to_numpy(zero_copy_only=False))
        vy_list.append(coords.vy.to_numpy(zero_copy_only=False))
        vz_list.append(coords.vz.to_numpy(zero_copy_only=False))

    if reps == 0 or frac > 1e-9:
        k = int(round(frac * base_n)) if reps > 0 else base_n
        if k > 0:
            sel = rng.choice(base_n, size=k, replace=False)
            times_list.append(base_times_mjd[sel])
            codes_list.extend([base_codes[i] for i in sel.tolist()])
            x_list.append(coords.x.to_numpy(zero_copy_only=False)[sel])
            y_list.append(coords.y.to_numpy(zero_copy_only=False)[sel])
            z_list.append(coords.z.to_numpy(zero_copy_only=False)[sel])
            vx_list.append(coords.vx.to_numpy(zero_copy_only=False)[sel])
            vy_list.append(coords.vy.to_numpy(zero_copy_only=False)[sel])
            vz_list.append(coords.vz.to_numpy(zero_copy_only=False)[sel])

    times_all = np.concatenate(times_list) if len(times_list) > 0 else base_times_mjd
    x_all = np.concatenate(x_list)
    y_all = np.concatenate(y_list)
    z_all = np.concatenate(z_list)
    vx_all = np.concatenate(vx_list)
    vy_all = np.concatenate(vy_list)
    vz_all = np.concatenate(vz_list)

    n_total = len(times_all)
    u = _random_unit_vectors(n_total, rng)

    from adam_core.coordinates.origin import Origin, OriginCodes
    from adam_core.time import Timestamp

    # Build an origin column matching length; assume observers are SUN-ecliptic
    origin_code = (
        observers.coordinates.origin.code[0].as_py()
        if len(observers) > 0
        else OriginCodes.SUN.name
    )
    origin_col = Origin.from_kwargs(code=[origin_code] * len(times_all))

    obs_coords = CartesianCoordinates.from_kwargs(
        x=x_all,
        y=y_all,
        z=z_all,
        vx=vx_all,
        vy=vy_all,
        vz=vz_all,
        time=Timestamp.from_mjd(times_all, scale=observers.coordinates.time.scale),
        origin=origin_col,
        frame="ecliptic",
    )

    # Build Observers column matching generated coordinates and codes
    observers_expanded = Observers.from_kwargs(code=codes_list, coordinates=obs_coords)

    rays = ObservationRays.from_kwargs(
        det_id=[f"noise_{i:07d}" for i in range(n_total)],
        orbit_id=[None] * n_total,
        observer=observers_expanded,
        u_x=u[:, 0],
        u_y=u[:, 1],
        u_z=u[:, 2],
    )
    rays.to_parquet(path)
    return path


def load_cached_noise_rays(name: str, density: float = 1.0) -> ObservationRays:
    tag = f"x{int(density*100):03d}"
    path = _fixture_cache_root() / "noise_rays" / name / f"rays_{tag}.parquet"
    if not path.exists():
        # Build observers and generate noise rays cache if missing
        observers = _build_observers_ci_timespan()
        cache_noise_rays_for_observers(observers, name, density=density)
    return ObservationRays.from_parquet(path)


def load_cached_noise_rays_per_sqdeg(
    name: str, per_sqdeg: float = 10.0
) -> ObservationRays:
    """Generate or load noise rays targeting a density of `per_sqdeg` per square degree.

    Approximates density by scaling observers count to desired total = per_sqdeg * 41253.
    """
    observers = _build_observers_ci_timespan()
    base_n = len(observers)
    target = max(1, int(per_sqdeg * 41253.0))
    density = float(target) / float(base_n) if base_n > 0 else 1.0
    # Log when generating
    print(f"[noise] per_sqdeg={per_sqdeg} -> target={target}, density={density:.3f}")
    return load_cached_noise_rays(name, density=density)


@pytest.fixture(scope="session")
def noise_rays_ci_x050(observers_ci_timespan) -> ObservationRays:
    name = "ci_observers"
    tag_path = _fixture_cache_root() / "noise_rays" / name / "rays_x050.parquet"
    if not tag_path.exists():
        cache_noise_rays_for_observers(observers_ci_timespan, name, density=0.5)
    return load_cached_noise_rays(name, density=0.5)


@pytest.fixture(scope="session")
def noise_rays_ci_x100(observers_ci_timespan) -> ObservationRays:
    name = "ci_observers"
    tag_path = _fixture_cache_root() / "noise_rays" / name / "rays_x100.parquet"
    if not tag_path.exists():
        cache_noise_rays_for_observers(observers_ci_timespan, name, density=1.0)
    return load_cached_noise_rays(name, density=1.0)


@pytest.fixture(scope="session")
def noise_rays_ci_x200(observers_ci_timespan) -> ObservationRays:
    name = "ci_observers"
    tag_path = _fixture_cache_root() / "noise_rays" / name / "rays_x200.parquet"
    if not tag_path.exists():
        cache_noise_rays_for_observers(observers_ci_timespan, name, density=2.0)
    return load_cached_noise_rays(name, density=2.0)


def build_population_orbits_cache(populations: list[str]) -> None:
    """Build/populate CI-scale cached datasets if missing.

    - Writes populations: synthetic_stratified_ci, synthetic_TJN_ci, synthetic_TNO_ci, synthetic_CEN_ci,
      real_ci (27), real_10 (10).
    - If build_orbits_only is False: also writes segments+AABBs for stratified and real_ci with default params.
    """
    if "synthetic_stratified_ci" in populations:
        synth_strat = qv.concatenate(
            [
                _make_orbits_for_class("IMB", 30, seed=101),
                _make_orbits_for_class("MBA", 25, seed=102),
                _make_orbits_for_class("OMB", 15, seed=103),
                _make_orbits_for_class("TJN", 10, seed=104),
                _make_orbits_for_class("CEN", 10, seed=105),
                _make_orbits_for_class("TNO", 10, seed=106),
                # Add 10 of each missing category
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
        cache_orbits(synth_strat, "synthetic_stratified_ci")
    if "synthetic_TJN_ci" in populations:
        cache_orbits(_make_orbits_for_class("TJN", 20, seed=110), "synthetic_TJN_ci")
    if "synthetic_TNO_ci" in populations:
        cache_orbits(_make_orbits_for_class("TNO", 20, seed=120), "synthetic_TNO_ci")
    if "synthetic_CEN_ci" in populations:
        cache_orbits(_make_orbits_for_class("CEN", 20, seed=130), "synthetic_CEN_ci")
    if "real_ci" in populations:
        real_all_raw = make_real_orbits()
        # Re-epoch real orbits to canonical epoch using ASSIST
        target_time = Timestamp.from_mjd([EPOCH_MJD], scale="tdb")
        real_all = ASSISTPropagator().propagate_orbits(real_all_raw, target_time)
        cache_orbits(real_all, "real_ci")
    if "real_10" in populations:
        cache_orbits(real_all[:10], "real_10")


def cache_rays_from_ephemeris(
    ephem: Ephemeris, observers: Observers, name: str
) -> Path:
    root = _fixture_cache_root() / "rays_from_ephem" / name
    _ensure_dir(root)
    path = root / "rays.parquet"
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return path
        # Observers repeat per orbit; tile codes by modulo
        codes = observers.code.to_pylist()
        det_ids = [
            f"{ephem.orbit_id[i].as_py()}:{codes[i % len(codes)]}:{i}"
            for i in range(len(ephem))
        ]
        rays = ephemeris_to_rays(ephem, det_id=det_ids)
        rays.to_parquet(path)
        return path


def cache_ephemeris_2body(
    orbits: Orbits, observers: Observers, name: str, *, max_processes: int | None = None
) -> Path:
    root = _fixture_cache_root() / "ephemeris_2body" / name
    _ensure_dir(root)
    path = root / "ephemeris.parquet"
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return path
        # Two-body ephemeris generation requires propagated orbits paired to observers 1:1
        times = observers.coordinates.time
        initialize_use_ray(num_cpus=max_processes)

        @ray.remote
        def _ephem_worker(orb_slice: Orbits):
            propagated_i = propagate_2body(orb_slice, times)
            # Tile observers to pair with each propagated orbit
            base_times = observers.coordinates.time
            base_codes = observers.code.to_pylist()
            reps = len(orb_slice)
            import numpy as np

            times_all = np.tile(base_times.mjd().to_numpy(zero_copy_only=False), reps)
            codes_all = base_codes * reps
            observers_tiled = Observers.from_codes(
                times=Timestamp.from_mjd(times_all, scale=base_times.scale),
                codes=codes_all,
            )
            return generate_ephemeris_2body(propagated_i, observers_tiled)

        futures = []
        chunk_size = 100
        for i in range(0, len(orbits), chunk_size):
            futures.append(_ephem_worker.remote(orbits[i : i + chunk_size]))
        ephems = ray.get(futures)
        ephem = qv.concatenate(ephems, defrag=True)
        ephem.to_parquet(path)
        return path


# ------------------------ N-body ephemeris/rays cache ------------------------
def cache_ephemeris_nbody(
    orbits: Orbits, observers: Observers, name: str, max_processes: int | None = None
) -> Path:
    """Cache N-body ephemeris using ASSISTPropagator."""
    from adam_assist import ASSISTPropagator

    root = _fixture_cache_root() / "ephemeris_nbody" / name
    _ensure_dir(root)
    path = root / "ephemeris.parquet"
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return path
        propagator = ASSISTPropagator()
        ephem = propagator.generate_ephemeris(
            orbits, observers, max_processes=max_processes, chunk_size=10
        )
        ephem.to_parquet(path)
        return path


def load_cached_ephemeris_nbody(name: str) -> Ephemeris:
    path = _fixture_cache_root() / "ephemeris_nbody" / name / "ephemeris.parquet"
    if not path.exists():
        # Use fixed 90-day observers and strip _nb suffix to find base population
        base = name.split("_nb")[0]
        orbits = load_cached_orbits(base)
        observers = _build_observers_ci_timespan()
        cache_ephemeris_nbody(orbits, observers, name)
    return Ephemeris.from_parquet(path)


def cache_rays_from_ephemeris_nbody(
    ephem: Ephemeris, observers: Observers, name: str
) -> Path:
    root = _fixture_cache_root() / "rays_from_ephem_nbody" / name
    _ensure_dir(root)
    path = root / "rays.parquet"
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return path
        codes = observers.code.to_pylist()
        det_ids = [
            f"{ephem.orbit_id[i].as_py()}:{codes[i % len(codes)]}:{i}"
            for i in range(len(ephem))
        ]
        rays = ephemeris_to_rays(ephem, det_id=det_ids)
        rays.to_parquet(path)
        return path


def load_cached_rays_from_ephemeris_nbody(name: str) -> ObservationRays:
    path = _fixture_cache_root() / "rays_from_ephem_nbody" / name / "rays.parquet"
    if not path.exists():
        ephem = load_cached_ephemeris_nbody(name)
        observers = _build_observers_ci_timespan()
        cache_rays_from_ephemeris_nbody(ephem, observers, name)
    return ObservationRays.from_parquet(path)


# ------------------- Parameterized segments/index cache ----------------------
def cache_segments_param(
    orbits: Orbits,
    name: str,
    max_chord_arcmin: float,
    max_segments_per_orbit: int = 32768,
) -> Path:
    from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive

    tag = format_segments_tag(
        max_chord_arcmin=max_chord_arcmin,
        max_segments_per_orbit=max_segments_per_orbit,
    )
    root = _fixture_cache_root() / "segments_param" / name / tag
    _ensure_dir(root)
    path = root / "segments.parquet"
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return path
        params, segs = sample_ellipse_adaptive(
            orbits,
            max_chord_arcmin=max_chord_arcmin,
            max_segments_per_orbit=max_segments_per_orbit,
        )
        segs.to_parquet(path)
        return path


def load_cached_segments_param(
    name: str,
    max_chord_arcmin: float,
    max_segments_per_orbit: int,
):
    from adam_core.orbits.polyline import OrbitPolylineSegments

    tag = format_segments_tag(
        max_chord_arcmin=max_chord_arcmin,
        max_segments_per_orbit=max_segments_per_orbit,
    )
    path = _fixture_cache_root() / "segments_param" / name / tag / "segments.parquet"
    if not path.exists():
        # Auto-build param segments cache
        orbits = load_cached_orbits(name)
        print(
            f"[segments] build: name={name} chord={max_chord_arcmin} ms={max_segments_per_orbit}"
        )
        cache_segments_param(
            orbits,
            name,
            max_chord_arcmin,
            max_segments_per_orbit,
        )
    else:
        # No guard/epsilon/padding validation needed for segments; sampling-only
        pass
    return OrbitPolylineSegments.from_parquet(path)


def cache_bvh_index_param(
    segments,
    name: str,
    *,
    max_leaf_size: int,
    max_chord_arcmin: float,
    guard_arcmin: float,
    padding_method: str,
    epsilon_n_au: float = 1e-6,
    max_segments_per_orbit: int = 32768,
) -> Path:
    tag = format_index_tag(
        max_leaf_size=max_leaf_size,
        max_chord_arcmin=max_chord_arcmin,
        guard_arcmin=guard_arcmin,
        padding_method=padding_method,
        epsilon_n_au=epsilon_n_au,
        max_segments_per_orbit=max_segments_per_orbit,
    )
    root = _fixture_cache_root() / "indices_param" / name / tag
    out_dir = root / "bvh_index"
    _ensure_dir(root)
    index = build_bvh_index_from_segments(segments, max_leaf_size=max_leaf_size)
    # Lock and write directory atomically by rename
    lock = root / ".lock"
    with _file_lock(lock):
        tmp_dir = root / "bvh_index.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        index.to_parquet(str(tmp_dir))
        os.replace(str(tmp_dir), str(out_dir))
    # Write meta.json for validation
    meta = {
        "max_leaf_size": int(max_leaf_size),
        "max_chord_arcmin": float(max_chord_arcmin),
        "guard_arcmin": float(guard_arcmin),
        "padding_method": str(padding_method),
        "epsilon_n_au": float(epsilon_n_au),
        "max_segments_per_orbit": int(max_segments_per_orbit),
        "type": "index_param",
        "name": name,
    }
    with (root / "meta.json").open("w") as f:
        json.dump(meta, f)
    return out_dir


def load_cached_bvh_index_param(
    name: str,
    *,
    max_leaf_size: int,
    max_chord_arcmin: float,
    guard_arcmin: float,
    padding_method: str = "baseline",
    epsilon_n_au: float = 1e-6,
    max_segments_per_orbit: int = 32768,
) -> BVHIndex:
    tag = format_index_tag(
        max_leaf_size=max_leaf_size,
        max_chord_arcmin=max_chord_arcmin,
        guard_arcmin=guard_arcmin,
        padding_method=padding_method,
        epsilon_n_au=epsilon_n_au,
        max_segments_per_orbit=max_segments_per_orbit,
    )
    dir_path = _fixture_cache_root() / "indices_param" / name / tag / "bvh_index"

    expected = {
        "max_leaf_size": int(max_leaf_size),
        "max_chord_arcmin": round(float(max_chord_arcmin), 2),
        "guard_arcmin": round(float(guard_arcmin), 2),
        "padding_method": str(padding_method),
        "epsilon_n_au": float(epsilon_n_au),
        "max_segments_per_orbit": int(max_segments_per_orbit),
    }

    idx = None
    current = None
    exists = dir_path.exists()
    if exists:
        # Validate using self-describing attributes persisted in parquet
        idx_loaded = BVHIndex.from_parquet(str(dir_path))
        current = {
            "max_leaf_size": int(idx_loaded.nodes.build_max_leaf_size),
            "max_chord_arcmin": round(
                float(getattr(idx_loaded.segments, "sample_max_chord_arcmin", -1.0)), 2
            ),
            "guard_arcmin": round(
                float(getattr(idx_loaded.segments, "aabb_guard_arcmin", -1.0)), 2
            ),
            "padding_method": str(
                getattr(idx_loaded.segments, "aabb_padding_method", "")
            ),
            "epsilon_n_au": float(
                getattr(idx_loaded.segments, "aabb_epsilon_n_au", -1.0)
            ),
            "max_segments_per_orbit": int(
                getattr(idx_loaded.segments, "sample_max_segments_per_orbit", -1)
            ),
        }
        idx = idx_loaded

    build_needed = (not exists) or (current != expected)
    if build_needed:
        # Ensure segments exist, then build and persist index
        segments = load_cached_segments_param(
            name,
            max_chord_arcmin=max_chord_arcmin,
            max_segments_per_orbit=max_segments_per_orbit,
        )
        reason = "missing" if not exists else "attributes mismatch"
        print(
            f"[index] {reason} -> build: name={name} leaf={max_leaf_size} chord={max_chord_arcmin} guard={guard_arcmin} pad={padding_method} ms={max_segments_per_orbit} eps={epsilon_n_au}"
        )
        cache_bvh_index_param(
            segments,
            name,
            max_leaf_size=max_leaf_size,
            max_chord_arcmin=max_chord_arcmin,
            guard_arcmin=guard_arcmin,
            padding_method=padding_method,
            epsilon_n_au=epsilon_n_au,
            max_segments_per_orbit=max_segments_per_orbit,
        )
        idx = BVHIndex.from_parquet(str(dir_path))

    # By here, idx is guaranteed
    return idx


def load_prebuilt_bvh_index_param(
    name: str,
    *,
    max_leaf_size: int,
    max_chord_arcmin: float,
    guard_arcmin: float,
    padding_method: str = "baseline",
    epsilon_n_au: float = 1e-6,
    max_segments_per_orbit: int = 32768,
) -> BVHIndex:
    """Load an existing BVH index for the exact params, without building.

    Raises FileNotFoundError if the index directory does not exist.
    Raises RuntimeError if meta.json is missing or does not match expected params.
    """
    tag = format_index_tag(
        max_leaf_size=max_leaf_size,
        max_chord_arcmin=max_chord_arcmin,
        guard_arcmin=guard_arcmin,
        padding_method=padding_method,
        epsilon_n_au=epsilon_n_au,
        max_segments_per_orbit=max_segments_per_orbit,
    )
    dir_path = _fixture_cache_root() / "indices_param" / name / tag / "bvh_index"
    if not dir_path.exists():
        raise FileNotFoundError(
            f"prebuilt index missing: {dir_path}. Run index benchmarks to build caches."
        )
    # Validate using attributes persisted in parquet
    idx = BVHIndex.from_parquet(str(dir_path))
    current = {
        "max_leaf_size": int(idx.nodes.build_max_leaf_size),
        "max_chord_arcmin": round(
            float(getattr(idx.segments, "sample_max_chord_arcmin", -1.0)), 2
        ),
        # Read AABB provenance from nodes (moved from segments)
        "guard_arcmin": round(float(getattr(idx.nodes, "aabb_guard_arcmin", -1.0)), 2),
        "padding_method": str(getattr(idx.nodes, "aabb_padding_method", "")),
        "epsilon_n_au": float(getattr(idx.nodes, "aabb_epsilon_n_au", -1.0)),
        "max_segments_per_orbit": int(
            getattr(idx.segments, "sample_max_segments_per_orbit", -1)
        ),
    }
    expected = {
        "max_leaf_size": int(max_leaf_size),
        "max_chord_arcmin": round(float(max_chord_arcmin), 2),
        "guard_arcmin": round(float(guard_arcmin), 2),
        "padding_method": str(padding_method),
        "epsilon_n_au": float(epsilon_n_au),
        "max_segments_per_orbit": int(max_segments_per_orbit),
    }
    if current != expected:
        raise RuntimeError(
            f"prebuilt index attributes mismatch at {dir_path}; expected {expected}, got {current}"
        )
    return idx


# ------------------- Offline builders for CI parameter grids -----------------
def _build_observers_ci_timespan() -> Observers:
    """Return a cached 90-day CI observers table (singleton per session)."""
    from adam_core.time import Timestamp

    global _OBSERVERS_CI_TIMESPAN_SINGLETON
    if _OBSERVERS_CI_TIMESPAN_SINGLETON is not None:
        return _OBSERVERS_CI_TIMESPAN_SINGLETON

    start = EPOCH_MJD
    days = np.arange(0, 90, dtype=int)
    times = Timestamp.from_mjd(start + days, scale="tdb")
    stations = ["X05", "T08", "I41"]
    codes = [stations[i % len(stations)] for i in range(len(times))]
    _OBSERVERS_CI_TIMESPAN_SINGLETON = Observers.from_codes(times=times, codes=codes)
    return _OBSERVERS_CI_TIMESPAN_SINGLETON


def _compute_observers_key(observers: Observers) -> str:
    times = observers.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    codes = observers.code.to_pylist()
    scale = observers.coordinates.time.scale
    h = hashlib.sha1()
    h.update(times.tobytes(order="C"))
    for c in codes:
        h.update(str(c).encode("utf-8"))
    h.update(str(scale).encode("utf-8"))
    return h.hexdigest()[:16]


def get_or_create_observers(
    span_days: int = 180, stations: tuple[str, ...] = ("X05", "T08", "I41")
) -> Observers:
    """Return session-memoized observers for given span and stations; cache to disk."""
    key = (span_days, tuple(stations))
    if key in _OBSERVERS_SINGLETONS:
        return _OBSERVERS_SINGLETONS[key]

    # Lock per label to avoid concurrent generation
    label = f"d{span_days}_st{'-'.join(stations)}"
    root = _fixture_cache_root() / "observers" / label
    _ensure_dir(root)
    lock = root / ".lock"

    import numpy as _np

    from adam_core.time import Timestamp

    with _file_lock(lock):
        # Generate deterministically
        start = EPOCH_MJD
        days = _np.arange(0, span_days, dtype=int)
        times = Timestamp.from_mjd(start + days, scale="tdb")
        codes = [stations[i % len(stations)] for i in range(len(times))]
        observers = Observers.from_codes(times=times, codes=codes)
        # Compute final path after building to get stable key
        obs_key = _compute_observers_key(observers)
        path = root / f"{obs_key}.parquet"
        if path.exists():
            loaded = Observers.from_parquet(path)
            _OBSERVERS_SINGLETONS[key] = loaded
            return loaded
        print(
            f"[observers] create: key={obs_key} days={span_days} stations={','.join(stations)}"
        )
        observers.to_parquet(path)
        _OBSERVERS_SINGLETONS[key] = observers
        return observers


def get_or_create_orbits(population_name: str) -> Orbits:
    """Load cached population or build and cache if missing (lock-protected)."""
    root_path = _fixture_cache_root() / "populations" / population_name
    lock = root_path / ".lock"
    path = root_path / "orbits.parquet"
    with _file_lock(lock):
        if not path.exists():
            print(f"[orbits] create: population={population_name}")
            build_population_orbits_cache([population_name])
        return Orbits.from_parquet(path)


def _compute_ephemeris_key(ephem: Ephemeris) -> str:
    """Stable key across ephemeris content (distinguishes 2b vs nb).

    Includes orbit_id, time mjd, and spherical coordinates (lon/lat/rho).
    """
    ids = ephem.orbit_id.to_pylist()
    mjd = ephem.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    lon = ephem.coordinates.lon.to_numpy(zero_copy_only=False)
    lat = ephem.coordinates.lat.to_numpy(zero_copy_only=False)
    rho = ephem.coordinates.rho.to_numpy(zero_copy_only=False)
    h = hashlib.sha1()
    for oid in ids:
        h.update(str(oid).encode("utf-8"))
    h.update(mjd.tobytes(order="C"))
    h.update(lon.tobytes(order="C"))
    h.update(lat.tobytes(order="C"))
    h.update(rho.tobytes(order="C"))
    return h.hexdigest()[:16]


def get_or_create_ephemeris(
    population_name: str,
    model: str,
    observers: Observers,
    *,
    max_processes: int | None = None,
) -> Ephemeris:
    """Compute or load ephemeris keyed by (population, model, observers_key)."""
    observers_key = _compute_observers_key(observers)
    root = _fixture_cache_root() / "ephemeris" / population_name / model / observers_key
    _ensure_dir(root)
    path = root / "ephemeris.parquet"

    # Serialize entire build to avoid concurrent generation
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return Ephemeris.from_parquet(path)
        orbits = get_or_create_orbits(population_name)
        print(
            f"[ephemeris] create: pop={population_name} model={model} key={observers_key}"
        )

        if model == "2b":
            # Pair orbits to observers times; tile observers per orbit
            times = observers.coordinates.time
            initialize_use_ray(num_cpus=max_processes)

            @ray.remote
            def _ephem_worker(orb_slice: Orbits):
                propagated = propagate_2body(orb_slice, times)
                base_times = observers.coordinates.time
                base_codes = observers.code.to_pylist()
                reps = len(orb_slice)
                import numpy as _np

                times_all = _np.tile(
                    base_times.mjd().to_numpy(zero_copy_only=False), reps
                )
                codes_all = base_codes * reps
                obs_tiled = Observers.from_codes(
                    times=Timestamp.from_mjd(times_all, scale=base_times.scale),
                    codes=codes_all,
                )
                return generate_ephemeris_2body(propagated, obs_tiled)

            futures = []
            chunk_size = 100
            for i in range(0, len(orbits), chunk_size):
                futures.append(_ephem_worker.remote(orbits[i : i + chunk_size]))
            ephems = ray.get(futures)
            ephem = qv.concatenate(ephems, defrag=True)
        elif model == "nb":
            propagator = ASSISTPropagator()
            ephem = propagator.generate_ephemeris(
                orbits, observers, max_processes=max_processes, chunk_size=10
            )
        else:
            raise ValueError("model must be '2b' or 'nb'")

        ephem.to_parquet(path)
        return ephem


def get_or_create_rays(ephem: Ephemeris, population_name: str) -> ObservationRays:
    """Build or load rays keyed by (ephemeris_key, observers_key)."""
    ephem_key = _compute_ephemeris_key(ephem)
    # Store under population/ephem_key
    root = _fixture_cache_root() / "rays" / population_name / ephem_key
    _ensure_dir(root)
    path = root / "rays.parquet"
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return ObservationRays.from_parquet(path)
        print(f"[rays] create: pop={population_name} key_ephem={ephem_key}")
        codes = ephem.coordinates.origin.code.to_pylist()
        det_ids = [
            f"{ephem.orbit_id[i].as_py()}:{codes[i % len(codes)]}:{i}"
            for i in range(len(ephem))
        ]
        rays = ephemeris_to_rays(ephem, det_id=det_ids)
        rays.to_parquet(path)
        return rays


def get_or_create_noise_rays(
    observers: Observers, *, per_sqdeg: float = 1.0, seed: int = 42
) -> ObservationRays:
    """Noise rays for given observers and density per sq. deg.; cached by observers_key and params."""
    observers_key = _compute_observers_key(observers)
    root = (
        _fixture_cache_root()
        / "noise_rays"
        / observers_key
        / f"per{per_sqdeg}"
        / f"seed{seed}"
    )
    _ensure_dir(root)
    # Use generator's filename convention rays_x{density*100}.parquet
    base_n = len(observers)
    target = max(1, int(per_sqdeg * 41253.0))
    density = float(target) / float(base_n) if base_n > 0 else 1.0
    tag = f"x{int(density*100):03d}"
    path = root / f"rays_{tag}.parquet"

    # Reuse generator impl; ensure it writes to the same directory
    lock = root / ".lock"
    with _file_lock(lock):
        if path.exists():
            return ObservationRays.from_parquet(path)
        # Only log when we actually need to build
        print(
            f"[noise] build: key_obs={observers_key} per_sqdeg={per_sqdeg} -> density={density:.3f} seed={seed}"
        )
        cache_noise_rays_for_observers(
            observers,
            f"{observers_key}/per{per_sqdeg}/seed{seed}",
            density=density,
            seed=seed,
        )
        return ObservationRays.from_parquet(path)


def get_or_create_segments(
    population_name: str,
    *,
    max_chord_arcmin: float,
    index_guard_arcmin: float,
    padding_method: str = "baseline",
    epsilon_n_au: float = 1e-6,
    max_segments_per_orbit: int = 32768,
):
    return load_cached_segments_param(
        population_name,
        max_chord_arcmin=max_chord_arcmin,
        max_segments_per_orbit=max_segments_per_orbit,
    )


def get_or_create_index(
    population_name: str,
    *,
    max_leaf_size: int,
    max_chord_arcmin: float,
    index_guard_arcmin: float,
    padding_method: str = "baseline",
    epsilon_n_au: float = 1e-6,
    max_segments_per_orbit: int = 32768,
    require_prebuilt: bool = False,
):
    if require_prebuilt:
        return load_prebuilt_bvh_index_param(
            population_name,
            max_leaf_size=max_leaf_size,
            max_chord_arcmin=max_chord_arcmin,
            guard_arcmin=index_guard_arcmin,
            padding_method=padding_method,
            epsilon_n_au=epsilon_n_au,
            max_segments_per_orbit=max_segments_per_orbit,
        )
    else:
        return load_cached_bvh_index_param(
            population_name,
            max_leaf_size=max_leaf_size,
            max_chord_arcmin=max_chord_arcmin,
            guard_arcmin=index_guard_arcmin,
            padding_method=padding_method,
            epsilon_n_au=epsilon_n_au,
            max_segments_per_orbit=max_segments_per_orbit,
        )


def load_cached_ephemeris_2body(name: str) -> Ephemeris:
    path = _fixture_cache_root() / "ephemeris_2body" / name / "ephemeris.parquet"
    if not path.exists():
        orbits = load_cached_orbits(name)
        observers = _build_observers_ci_timespan()
        cache_ephemeris_2body(orbits, observers, name)
    return Ephemeris.from_parquet(path)


def load_cached_rays_from_ephemeris(name: str) -> ObservationRays:
    path = _fixture_cache_root() / "rays_from_ephem" / name / "rays.parquet"
    if not path.exists():
        # Ensure ephemeris exists and then build rays
        ephem = load_cached_ephemeris_2body(name)
        observers = _build_observers_ci_timespan()
        cache_rays_from_ephemeris(ephem, observers, name)
    return ObservationRays.from_parquet(path)


@pytest.fixture(scope="session")
def observers_ci_timespan() -> Observers:
    """Session-scoped access to the cached 90-day CI observers."""
    return _build_observers_ci_timespan()


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


# index_small fixture moved to bvh/tests/conftest.py


# BVH-specific fixtures
import pytest

from adam_core.geometry.bvh.index import BVHIndex, build_bvh_index_from_segments
from adam_core.geometry.rays import ObservationRays
from adam_core.orbits.orbits import Orbits


@pytest.fixture(scope="session")
def segments_aabbs(simple_orbits):
    from adam_core.orbits.polyline import sample_ellipse_adaptive

    _, segs = sample_ellipse_adaptive(
        simple_orbits, max_chord_arcmin=1.0, max_segments_per_orbit=256
    )
    return segs


@pytest.fixture(scope="session")
def rays(_geom_testdata_dir):
    return ObservationRays.from_parquet(_geom_testdata_dir / "rays_small.parquet")


@pytest.fixture(scope="session")
def synthetic_orbits_stratified_ci() -> Orbits:
    """Load CI-sized synthetic population (IMB/MBA/OMB/TJN/CEN/TNO) from disk cache.

    Set ADAM_GEOM_CACHE_AUTO_BUILD=1 to auto-generate if missing.
    """
    name = "synthetic_stratified_ci"
    # Use creator that builds if missing
    return get_or_create_orbits(name)


@pytest.fixture(scope="function", params=[(64, 16), (256, 32)])
def index_small(segments_aabbs, request) -> BVHIndex:
    N, leaf_size = request.param
    idx = build_bvh_index_from_segments(segments_aabbs[:N], max_leaf_size=leaf_size)
    return idx
