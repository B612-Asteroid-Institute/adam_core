import numpy as np
import spiceypy as sp
from adam_assist import ASSISTPropagator

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...time import Timestamp
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

    # Load SPK and verify contents
    sp.furnsh(str(spk_file))

    # Check state at t0
    et0 = t0.et()[0].as_py()
    state, lt = sp.spkez(1000000, et0, "J2000", "NONE", 0)

    # Convert SPICE state (km, km/s) to our units (au, au/day)
    state[:3] /= 149597870.7  # km to au
    state[3:] *= 86400.0 / 149597870.7  # km/s to au/day

    # Compare with input state
    input_state = orbits.coordinates.values[0]
    assert np.allclose(state, input_state, rtol=1e-10, atol=1e-10)

    # Clean up
    sp.unload(str(spk_file))


# def test_chebyshev_fit_against_spice_earth():

#     de440_reader = DE440Reader(de440)
#     x_coeffs, y_coeffs, z_coeffs = de440_reader.get_emb_chebyshev(0.0)
#     print(f"x_coeffs: {x_coeffs}")
#     print(f"y_coeffs: {y_coeffs}")
#     print(f"z_coeffs: {z_coeffs}")

#     # Get the summary record for the handle. dafgs does not
#     # take handle as an input

#     # === Use your function to fit Chebyshev polynomials ===
#     # For example, get Earth state at points within this record's time window:
#     num_points = 100
#     times = Timestamp.from_et(np.linspace(record_start_et, record_end_et, num_points))
#     earth_coordinates = get_perturber_state(
#         perturber=OriginCodes.EARTH_MOON_BARYCENTER,
#         times=times,
#         origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
#         frame="equatorial"
#     )

#     # Fit Chebyshev polynomials using your function.
#     # (Assume that fit_chebyshev returns coefficients, a mid time, and a half-interval.)
#     fitted_coeffs, mid_time, half_interval = fit_chebyshev(
#         earth_coordinates,
#         record_start_et,
#         record_end_et,
#         degree=degree
#     )

#     # Compare only the position coefficients (first 3 components)
#     pos_diff = np.abs(fitted_coeffs[:3] - spice_coeffs[:3])
#     print("Position coefficient differences:", pos_diff)

#     # Assert that the position coefficients match within 1 meter (1e-3 km)
#     assert np.all(pos_diff < 1e-3)
