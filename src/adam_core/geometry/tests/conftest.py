import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import sample_ellipse_adaptive, compute_segment_aabbs
from adam_core.time import Timestamp


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
    return Orbits.from_kwargs(orbit_id=[f"bench_{i}" for i in range(n)], coordinates=coords)


@pytest.fixture(scope="session")
def large_segments_aabb(large_orbits):
    """Session-scoped large segments with AABBs for reuse and slicing."""
    _, segments = sample_ellipse_adaptive(large_orbits, max_chord_arcmin=2.0)
    return compute_segment_aabbs(segments, guard_arcmin=1.0)
import pytest
import numpy as np

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry import build_bvh_index_from_segments
from adam_core.observations.detections import PointSourceDetections
from adam_core.observations.exposures import Exposures
from adam_core.observations.rays import rays_from_detections
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.time import Timestamp


@pytest.fixture(scope="session")
def simple_orbits():
    times = Timestamp.from_mjd([59000.0], scale="tdb")
    coords = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.017202],
        vz=[0.0],
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(orbit_id=["test_orbit"], coordinates=coords)


@pytest.fixture(scope="session")
def segments_aabbs(simple_orbits):
    _, segments = sample_ellipse_adaptive(simple_orbits, max_chord_arcmin=1.0)
    return compute_segment_aabbs(segments, guard_arcmin=1.0)


@pytest.fixture(scope="session")
def bvh_index(segments_aabbs):
    return build_bvh_index_from_segments(segments_aabbs)


@pytest.fixture(scope="session")
def rays():
    times = Timestamp.from_mjd([59000.0, 59000.1], scale="tdb")
    exposures = Exposures.from_kwargs(
        id=["exp_1", "exp_2"],
        start_time=times,
        duration=[300.0, 300.0],
        filter=["r", "g"],
        observatory_code=["500", "500"],
        seeing=[1.2, 1.3],
        depth_5sigma=[22.0, 22.1],
    )
    detections = PointSourceDetections.from_kwargs(
        id=["det_1", "det_2"],
        exposure_id=["exp_1", "exp_2"],
        time=times,
        ra=[0.0, 90.0],
        dec=[0.0, 0.0],
        ra_sigma=[0.1, 0.1],
        dec_sigma=[0.1, 0.1],
        mag=[20.0, 20.1],
        mag_sigma=[0.1, 0.1],
    )
    return rays_from_detections(detections, exposures)


