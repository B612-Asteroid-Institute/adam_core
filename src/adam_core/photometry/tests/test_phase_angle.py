import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.observers.observers import Observers
from adam_core.photometry import calculate_phase_angle
from adam_core.time import Timestamp


def _make_helio_cartesian(
    r_au: np.ndarray,
    *,
    origin: OriginCodes = OriginCodes.SUN,
    frame: str = "ecliptic",
) -> CartesianCoordinates:
    """
    Build a `CartesianCoordinates` table with zero velocities.

    Why: `calculate_phase_angle` only needs positions, but the coordinate type is a 6D state.
    """
    r = np.asarray(r_au, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("r_au must have shape (N, 3)")

    n = int(r.shape[0])
    t = Timestamp.from_mjd([60000.0] * n, scale="tdb")
    origin_col = Origin.from_kwargs(code=[origin.name] * n)

    zeros = np.zeros(n, dtype=np.float64)
    return CartesianCoordinates.from_kwargs(
        x=r[:, 0],
        y=r[:, 1],
        z=r[:, 2],
        vx=zeros,
        vy=zeros,
        vz=zeros,
        time=t,
        origin=origin_col,
        frame=frame,
    )


def _phase_angle_reference_deg(
    object_pos_au: np.ndarray, observer_pos_au: np.ndarray
) -> np.ndarray:
    """
    Reference phase-angle computation via vector angle definition.

    Phase angle is the angle at the object between:
    - Sun direction (object -> Sun), and
    - Observer direction (object -> observer).

    We compute the angle between vectors using atan2(||a×b||, a·b), which is stable near 0/180.
    """
    obj = np.asarray(object_pos_au, dtype=np.float64)
    obs = np.asarray(observer_pos_au, dtype=np.float64)
    a = -obj
    b = obs - obj
    cross = np.cross(a, b)
    dot = np.sum(a * b, axis=1)
    alpha_rad = np.arctan2(np.linalg.norm(cross, axis=1), dot)
    return np.degrees(alpha_rad)


def test_calculate_phase_angle_expected_values() -> None:
    # Three simple heliocentric geometries with known phase angles:
    # - 0 deg (opposition-like): Sun -> observer -> object
    # - 90 deg (quadrature-like)
    # - 180 deg (conjunction-like): Sun -> object -> observer
    t = Timestamp.from_mjd([60000.0, 60000.0, 60000.0], scale="tdb")
    origin_sun = Origin.from_kwargs(code=[OriginCodes.SUN.name] * 3)

    object_coords = CartesianCoordinates.from_kwargs(
        x=[2.0, 1.0, 1.0],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        vx=[0.0, 0.0, 0.0],
        vy=[0.0, 0.0, 0.0],
        vz=[0.0, 0.0, 0.0],
        time=t,
        origin=origin_sun,
        frame="ecliptic",
    )
    observer_coords = CartesianCoordinates.from_kwargs(
        x=[1.0, 1.0, 2.0],
        y=[0.0, 1.0, 0.0],
        z=[0.0, 0.0, 0.0],
        vx=[0.0, 0.0, 0.0],
        vy=[0.0, 0.0, 0.0],
        vz=[0.0, 0.0, 0.0],
        time=t,
        origin=origin_sun,
        frame="ecliptic",
    )
    observers = Observers.from_kwargs(code=["X"] * 3, coordinates=observer_coords)

    alpha = calculate_phase_angle(object_coords, observers)
    np.testing.assert_allclose(alpha, np.array([0.0, 90.0, 180.0]), atol=1e-12)


def test_calculate_phase_angle_length_mismatch_raises() -> None:
    t_obj = Timestamp.from_mjd([60000.0, 60000.0], scale="tdb")
    t_obs = Timestamp.from_mjd([60000.0], scale="tdb")

    origin_obj = Origin.from_kwargs(code=[OriginCodes.SUN.name] * 2)
    origin_obs = Origin.from_kwargs(code=[OriginCodes.SUN.name])

    object_coords = CartesianCoordinates.from_kwargs(
        x=[1.0, 1.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        vx=[0.0, 0.0],
        vy=[0.0, 0.0],
        vz=[0.0, 0.0],
        time=t_obj,
        origin=origin_obj,
        frame="ecliptic",
    )
    observer_coords = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=t_obs,
        origin=origin_obs,
        frame="ecliptic",
    )
    observers = Observers.from_kwargs(code=["X"], coordinates=observer_coords)

    with pytest.raises(ValueError, match="must match object_coords length"):
        calculate_phase_angle(object_coords, observers)


def test_calculate_phase_angle_requires_heliocentric_origin() -> None:
    t = Timestamp.from_mjd([60000.0], scale="tdb")

    obj_ssb = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=t,
        origin=Origin.from_kwargs(code=[OriginCodes.SOLAR_SYSTEM_BARYCENTER.name]),
        frame="ecliptic",
    )
    obs_sun = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=t,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    observers = Observers.from_kwargs(code=["X"], coordinates=obs_sun)

    with pytest.raises(ValueError, match="object_coords must be heliocentric"):
        calculate_phase_angle(obj_ssb, observers)


def test_calculate_phase_angle_requires_heliocentric_observer_origin() -> None:
    t = Timestamp.from_mjd([60000.0], scale="tdb")

    obj_sun = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=t,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    obs_ssb = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=t,
        origin=Origin.from_kwargs(code=[OriginCodes.SOLAR_SYSTEM_BARYCENTER.name]),
        frame="ecliptic",
    )
    observers = Observers.from_kwargs(code=["X"], coordinates=obs_ssb)

    with pytest.raises(
        ValueError, match="observers\\.coordinates must be heliocentric"
    ):
        calculate_phase_angle(obj_sun, observers)


def test_calculate_phase_angle_requires_same_frame() -> None:
    t = Timestamp.from_mjd([60000.0], scale="tdb")
    origin_sun = Origin.from_kwargs(code=[OriginCodes.SUN.name])

    obj = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=t,
        origin=origin_sun,
        frame="ecliptic",
    )
    obs = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=t,
        origin=origin_sun,
        frame="equatorial",
    )
    observers = Observers.from_kwargs(code=["X"], coordinates=obs)

    with pytest.raises(ValueError, match="same frame"):
        calculate_phase_angle(obj, observers)


@pytest.mark.parametrize(
    "object_pos, observer_pos",
    [
        # r == 0 (object at Sun)
        (np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])),
        # delta == 0 (observer colocated with object)
        (np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])),
        # non-finite values
        (np.array([[np.nan, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])),
    ],
)
def test_calculate_phase_angle_rejects_invalid_geometry(
    object_pos: np.ndarray, observer_pos: np.ndarray
) -> None:
    obj = _make_helio_cartesian(object_pos, origin=OriginCodes.SUN, frame="ecliptic")
    obs = _make_helio_cartesian(observer_pos, origin=OriginCodes.SUN, frame="ecliptic")
    observers = Observers.from_kwargs(code=["X"], coordinates=obs)

    with pytest.raises(ValueError, match="Invalid photometry geometry"):
        calculate_phase_angle(obj, observers)


def test_calculate_phase_angle_matches_vector_definition_for_random_cases() -> None:
    rng = np.random.default_rng(0)

    # Generate a deterministic set of non-degenerate geometries.
    obj_list: list[np.ndarray] = []
    obs_list: list[np.ndarray] = []
    while len(obj_list) < 200:
        obj = rng.uniform(-3.0, 3.0, size=3)
        obs = rng.uniform(-3.0, 3.0, size=3)
        if np.linalg.norm(obj) <= 0.2:
            continue
        if np.linalg.norm(obs) <= 0.2:
            continue
        if np.linalg.norm(obj - obs) <= 0.2:
            continue
        obj_list.append(obj)
        obs_list.append(obs)

    object_pos = np.stack(obj_list, axis=0)
    observer_pos = np.stack(obs_list, axis=0)

    obj_coords = _make_helio_cartesian(
        object_pos, origin=OriginCodes.SUN, frame="ecliptic"
    )
    obs_coords = _make_helio_cartesian(
        observer_pos, origin=OriginCodes.SUN, frame="ecliptic"
    )
    observers = Observers.from_kwargs(
        code=["X"] * len(obj_coords), coordinates=obs_coords
    )

    alpha = calculate_phase_angle(obj_coords, observers)
    alpha_ref = _phase_angle_reference_deg(object_pos, observer_pos)

    assert alpha.dtype == np.float64
    assert alpha.shape == (len(obj_coords),)
    assert np.all(np.isfinite(alpha))
    assert np.all(alpha >= 0.0)
    assert np.all(alpha <= 180.0)

    np.testing.assert_allclose(alpha, alpha_ref, rtol=0.0, atol=1e-12)
