import numpy as np
from adam_assist import ASSISTPropagator

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...constants import KM_P_AU, S_P_DAY
from ...time import Timestamp
from ...utils.spice_backend import get_backend
from ..orbits import Orbits
from ..spice_kernel import orbits_to_spk


def test_orbits_to_spk(tmp_path):
    # Create test orbit
    t0 = Timestamp.from_mjd([60000.0], scale="tdb")
    origin = Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"])

    # Create a simple circular orbit
    orbits = Orbits.from_kwargs(
        orbit_id=["test_orbit"],
        object_id=["test_object"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[1.0],
            z=[0.0],
            vx=[0.0],
            vy=[1.0],
            vz=[0.0],
            time=t0,
            origin=origin,
            frame="equatorial",
        ),
    )

    # Generate SPK file
    spk_file = tmp_path / "test.bsp"
    end_time = t0.add_fractional_days(100.0)

    orbits_to_spk(
        orbits,
        str(spk_file),
        start_time=t0,
        end_time=end_time,
        propagator=ASSISTPropagator(),
        step_days=1.0,
        window_days=10.0,
    )

    # Verify file exists
    assert spk_file.exists()

    # Load SPK and verify contents via the pure-Rust backend
    backend = get_backend()
    backend.furnsh(str(spk_file))
    try:
        et0 = t0.et()[0].as_py()
        state = backend.spkez(1000000, et0, "J2000", 0)

        # Convert backend state (km, km/s) to our units (au, au/day)
        state = np.asarray(state, dtype=np.float64).copy()
        state[:3] /= KM_P_AU
        state[3:] *= S_P_DAY / KM_P_AU

        input_state = orbits.coordinates.values[0]
        assert np.allclose(state, input_state, rtol=1e-10, atol=1e-10)
    finally:
        backend.unload(str(spk_file))
