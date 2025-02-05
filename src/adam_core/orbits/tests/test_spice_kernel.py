import numpy as np
import pytest
import spiceypy as sp
from naif_de440 import de440

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin, OriginCodes
from ...time import Timestamp
from ...utils.spice import get_perturber_state, setup_SPICE
from ..orbits import Orbits
from ..spice_kernel import fit_chebyshev, orbits_to_spk


def test_orbits_to_spk(tmp_path):
    # Create test orbit
    t0 = Timestamp.from_mjd([60000.0], scale="tdb")
    origin = Origin.from_kwargs(code=["SUN"])

    # Create a simple circular orbit
    orbits = Orbits.from_kwargs(
        orbit_id=["test_orbit"],
        object_id=["test_object"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[1.0],
            vz=[0.0],
            time=t0,
            origin=origin,
            frame="ecliptic",
        ),
    )

    # Generate SPK file
    spk_file = tmp_path / "test.bsp"
    end_time = t0.add_fractional_days(30.0)

    orbits_to_spk(
        orbits,
        str(spk_file),
        start_time=t0,
        end_time=end_time,
        step_days=1.0,
        window_days=5.0,
        cheby_degree=3,
    )

    # Verify file exists
    assert spk_file.exists()

    # Load SPK and verify contents
    setup_SPICE()
    sp.furnsh(str(spk_file))

    # Check state at t0
    et0 = t0.et()
    state, lt = sp.spkez(1000000, et0, "J2000", "NONE", 10)

    # Convert SPICE state (km, km/s) to our units (au, au/day)
    state[:3] /= 149597870.7  # km to au
    state[3:] *= 86400.0 / 149597870.7  # km/s to au/day

    # Compare with input state
    input_state = orbits.coordinates.values[0]
    assert np.allclose(state, input_state, rtol=1e-10, atol=1e-10)

    # Clean up
    sp.unload(str(spk_file))


def test_chebyshev_fit_against_spice_earth():
    # Get the segment info for Earth from DE440
    handle = sp.spklef(de440)

    # Find Earth Barycenter segment (ID = 3)
    # First find the segment
    dladsc = sp.spksfs(3, 0.0, 100)  # body=3, et=0, idlen=100 for segment ID
    print(f"DLADSC: {dladsc}")
    
    spkuds = sp.spkuds(dladsc[1])
    print(f"SPKUDS: {spkuds}")
    # Unpack the values:
    # body, center, frame, type, first_epoch, last_epoch, begin_addr, end_addr = spkuds
    spkuds = sp.spkuds(dladsc[1])
    body, center, frame, spk_type, start_et, end_et, begin_addr, end_addr = spkuds
    print(f"Body ID: {body}")
    print(f"Center ID: {center}")
    print(f"Frame ID: {frame}")
    print(f"SPK Type: {spk_type}")  # Should be 2 for Chebyshev
    print(f"Time range: {start_et} to {end_et}")
    
    # Read the raw data
    init_data = sp.dafgda(handle, begin_addr, begin_addr + 10)
    print(f"Init data: {init_data}")
    raw_data = sp.dafgda(handle, begin_addr, end_addr)
    
    # First few values in raw_data should be metadata about the polynomials
    print(f"First few values of raw data: {raw_data[:10]}")
    
    
    setup_SPICE(force=True)
    
    print(f"Start ET: {start_et}")
    print(f"End ET: {end_et}")
    # Get Earth states at times matching the DE440 segment
    num_points = 100  # Number of points to sample within window
    times = Timestamp.from_et(np.linspace(start_et, end_et, num_points))

    earth_coordinates = get_perturber_state(
        perturber=OriginCodes.EARTH_MOON_BARYCENTER, times=times, origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER, frame="equatorial"
    )

    # Fit Chebyshev polynomials using our function
    coeffs, mid_time, half_interval = fit_chebyshev(
        earth_coordinates, start_et, end_et, degree=11  # DE440 uses degree 11
    )

    # Get the actual coefficients from DE440
    # Each component has degree+1 coefficients
    spice_coeffs = np.array(raw_data).reshape(-1, 3)  # Reshape into position components

    # Compare position coefficients directly - they should be in the same units (km)
    pos_diff = np.abs(coeffs[:3] - spice_coeffs)

    # Assert position coefficients match within 1 meter
    assert np.all(pos_diff < 1e-3)  # km -> m


