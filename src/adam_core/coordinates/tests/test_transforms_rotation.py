import quivr as qv

from ...constants import KM_P_AU, S_P_DAY
from ...time import Timestamp
from ...utils.spice import get_perturber_state, setup_SPICE
from ...utils.spice_backend import get_backend
from ..cartesian import CartesianCoordinates
from ..origin import Origin, OriginCodes
from ..transform import cartesian_to_frame, transform_coordinates
from .test_transforms_translation import assert_coords_equal


def test_cartesian_to_frame(orbital_elements, orbital_elements_equatorial):
    # Test cartesian_to_frame correctly converts between ecliptic and
    # equatorial cartesian coordinates

    cartesian_coordinates_ecliptic = CartesianCoordinates.from_kwargs(
        x=orbital_elements["x"].values,
        y=orbital_elements["y"].values,
        z=orbital_elements["z"].values,
        vx=orbital_elements["vx"].values,
        vy=orbital_elements["vy"].values,
        vz=orbital_elements["vz"].values,
        time=Timestamp.from_mjd(orbital_elements["mjd_tdb"].values, scale="tdb"),
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN" for i in range(len(orbital_elements))]),
    )

    cartesian_coordinates_equatorial = CartesianCoordinates.from_kwargs(
        x=orbital_elements_equatorial["x"].values,
        y=orbital_elements_equatorial["y"].values,
        z=orbital_elements_equatorial["z"].values,
        vx=orbital_elements_equatorial["vx"].values,
        vy=orbital_elements_equatorial["vy"].values,
        vz=orbital_elements_equatorial["vz"].values,
        time=Timestamp.from_mjd(
            orbital_elements_equatorial["mjd_tdb"].values, scale="tdb"
        ),
        frame="equatorial",
        origin=Origin.from_kwargs(
            code=["SUN" for i in range(len(orbital_elements_equatorial))]
        ),
    )

    # Convert equatorial coordinates to ecliptic
    cartesian_coordinates_ecliptic_actual = cartesian_to_frame(
        cartesian_coordinates_equatorial,
        "ecliptic",
    )
    assert cartesian_coordinates_ecliptic_actual.frame == "ecliptic"

    # Test that the two coordinates are equal to 10 mm in position and 10 nm/s in velocity
    assert_coords_equal(
        cartesian_coordinates_ecliptic_actual, cartesian_coordinates_ecliptic
    )

    # Convert ecliptic coordinates to equatorial
    cartesian_coordinates_equatorial_actual = cartesian_to_frame(
        cartesian_coordinates_ecliptic,
        "equatorial",
    )
    assert cartesian_coordinates_equatorial_actual.frame == "equatorial"

    # Test that the two coordinates are equal to 10 mm in position and 10 nm/s in velocity
    assert_coords_equal(
        cartesian_coordinates_equatorial_actual, cartesian_coordinates_equatorial
    )


def test_transform_coordinates_frame(orbital_elements, orbital_elements_equatorial):
    # Test transform_coordinates correctly converts between ecliptic and
    # equatorial cartesian coordinates

    cartesian_coordinates_ecliptic = CartesianCoordinates.from_kwargs(
        x=orbital_elements["x"].values,
        y=orbital_elements["y"].values,
        z=orbital_elements["z"].values,
        vx=orbital_elements["vx"].values,
        vy=orbital_elements["vy"].values,
        vz=orbital_elements["vz"].values,
        time=Timestamp.from_mjd(
            orbital_elements["mjd_tdb"].values,
            scale="tdb",
        ),
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN" for i in range(len(orbital_elements))]),
    )

    cartesian_coordinates_equatorial = CartesianCoordinates.from_kwargs(
        x=orbital_elements_equatorial["x"].values,
        y=orbital_elements_equatorial["y"].values,
        z=orbital_elements_equatorial["z"].values,
        vx=orbital_elements_equatorial["vx"].values,
        vy=orbital_elements_equatorial["vy"].values,
        vz=orbital_elements_equatorial["vz"].values,
        time=Timestamp.from_mjd(
            orbital_elements_equatorial["mjd_tdb"].values,
            scale="tdb",
        ),
        frame="equatorial",
        origin=Origin.from_kwargs(
            code=["SUN" for i in range(len(orbital_elements_equatorial))]
        ),
    )

    # Convert equatorial coordinates to ecliptic
    cartesian_coordinates_ecliptic_actual = transform_coordinates(
        cartesian_coordinates_equatorial,
        frame_out="ecliptic",
    )
    assert cartesian_coordinates_ecliptic_actual.frame == "ecliptic"

    # Test that the two coordinates are equal to 10 mm in position and 10 nm/s in velocity
    assert_coords_equal(
        cartesian_coordinates_ecliptic_actual, cartesian_coordinates_ecliptic
    )

    # Convert ecliptic coordinates to equatorial
    cartesian_coordinates_equatorial_actual = transform_coordinates(
        cartesian_coordinates_ecliptic,
        frame_out="equatorial",
    )
    assert cartesian_coordinates_equatorial_actual.frame == "equatorial"

    # Test that the two coordinates are equal to 10 mm in position and 10 nm/s in velocity
    assert_coords_equal(
        cartesian_coordinates_equatorial_actual, cartesian_coordinates_equatorial
    )


_PERTURBERS = [
    OriginCodes.MOON,
    OriginCodes.VENUS,
    OriginCodes.MARS_BARYCENTER,
    OriginCodes.JUPITER_BARYCENTER,
    OriginCodes.SATURN_BARYCENTER,
    OriginCodes.URANUS_BARYCENTER,
    OriginCodes.NEPTUNE_BARYCENTER,
]


def _direct_backend_states(
    target_ids: list[int], times: Timestamp, frame: str
) -> CartesianCoordinates:
    """Query the SPICE backend directly for each (target, et) pair and
    pack into a CartesianCoordinates. Mirrors the state assembly
    ``transform_coordinates`` reaches via the PCK rotation path, but
    through the SPK reader's own composed frame query — the two paths
    should agree to the backend's rotation-accumulation tolerance.
    """
    backend = get_backend()
    sun = OriginCodes.SUN.value
    n = len(times) * len(target_ids) if False else len(target_ids)
    # target_ids is the flat per-row list; len(target_ids) == len(times).
    ets = times.et().to_numpy(zero_copy_only=False).astype("float64")
    assert len(target_ids) == len(ets)
    out = CartesianCoordinates.empty()
    for target_id, coord_time, et in zip(target_ids, times, ets):
        state = backend.spkez(int(target_id), float(et), frame, sun)
        out = qv.concatenate(
            [
                out,
                CartesianCoordinates.from_kwargs(
                    x=[state[0] / KM_P_AU],
                    y=[state[1] / KM_P_AU],
                    z=[state[2] / KM_P_AU],
                    vx=[state[3] / KM_P_AU * S_P_DAY],
                    vy=[state[4] / KM_P_AU * S_P_DAY],
                    vz=[state[5] / KM_P_AU * S_P_DAY],
                    time=coord_time,
                    frame=frame.lower() if frame == "ITRF93" else "ecliptic",
                    origin=Origin.from_kwargs(code=["SUN"]),
                ),
            ]
        )
    return out


def test_transform_coordinates_to_itrf93():
    """
    ``transform_coordinates(ecliptic → itrf93)`` should match the result
    of querying the SPICE backend directly in the ``ITRF93`` frame. The
    former goes through the ecliptic-state → time-varying PCK rotation
    path; the latter goes through the SPK reader's composed ITRF93
    query — different code paths, same underlying kernel.
    """
    setup_SPICE()

    times = Timestamp.from_mjd([59000, 59500, 60000], scale="tdb")
    states = CartesianCoordinates.empty()
    row_target_ids: list[int] = []
    row_times: list[Timestamp] = []
    for perturber in _PERTURBERS:
        states = qv.concatenate(
            [
                states,
                get_perturber_state(
                    perturber, times, origin=OriginCodes.SUN, frame="ecliptic"
                ),
            ]
        )
        row_target_ids.extend(int(perturber.value) for _ in range(len(times)))
        row_times.extend(list(times))

    states_itrf93 = transform_coordinates(states, frame_out="itrf93")

    flat_times = Timestamp.from_kwargs(
        days=[t.days[0].as_py() for t in row_times],
        nanos=[t.nanos[0].as_py() for t in row_times],
        scale="tdb",
    )
    states_direct_itrf93 = _direct_backend_states(
        row_target_ids, flat_times, "ITRF93"
    )

    # The ecliptic → PCK-rotation path and the direct SPK ITRF93 query
    # agree to ~ULP on the rotation matrix accumulation; at outer-planet
    # distances the ω×r leak is ~1 µm/s, well inside science tolerances.
    assert_coords_equal(
        states_itrf93,
        states_direct_itrf93,
        position_tol_mm=10,
        velocity_tol_nm_s=1000,
    )


def test_transform_coordinates_from_itrf93():
    """Symmetric check: ``transform_coordinates(itrf93 → ecliptic)``
    should match the SPK reader's direct ECLIPJ2000 query."""
    setup_SPICE()

    times = Timestamp.from_mjd([59000, 59500, 60000], scale="tdb")
    states = CartesianCoordinates.empty()
    row_target_ids: list[int] = []
    row_times: list[Timestamp] = []
    for perturber in _PERTURBERS:
        states = qv.concatenate(
            [
                states,
                get_perturber_state(
                    perturber, times, origin=OriginCodes.SUN, frame="itrf93"
                ),
            ]
        )
        row_target_ids.extend(int(perturber.value) for _ in range(len(times)))
        row_times.extend(list(times))

    states_ecliptic = transform_coordinates(states, frame_out="ecliptic")

    flat_times = Timestamp.from_kwargs(
        days=[t.days[0].as_py() for t in row_times],
        nanos=[t.nanos[0].as_py() for t in row_times],
        scale="tdb",
    )
    states_direct_ecliptic = _direct_backend_states(
        row_target_ids, flat_times, "ECLIPJ2000"
    )

    assert_coords_equal(
        states_ecliptic,
        states_direct_ecliptic,
        position_tol_mm=10,
        velocity_tol_nm_s=1000,
    )
