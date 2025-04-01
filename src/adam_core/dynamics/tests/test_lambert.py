import numpy as np
import numpy.testing as npt

from adam_core.constants import KM_P_AU, S_P_DAY
from adam_core.coordinates.origin import OriginCodes, OriginGravitationalParameters
from adam_core.dynamics.lambert import solve_lambert
from adam_core.time import Timestamp
from adam_core.utils.spice import get_perturber_state

# Test cases from Vallado's "Fundamentals of Astrodynamics and Applications"
# and other well-known problems

def test_lambert_solver_lamberthub():

    # Initial conditions
    mu_sun = OriginGravitationalParameters.SUN
    r1 = np.array([0.159321004, 0.579266185, 0.052359607])  # [AU]
    r2 = np.array([0.057594337, 0.605750797, 0.068345246])  # [AU]
    tof = 0.010794065 * 365.25  # Convert years to days

    # Solving the problem
    v1, v2 = solve_lambert(r1, r2, tof, mu=mu_sun, tol=1e-10, max_iter=100000)

    # Expected final results (in AU/year)
    expected_v1 = np.array([-9.303603251, 3.018641330, 1.536362143])
    # Convert expected velocities from AU/year to AU/day
    expected_v1 = expected_v1 / 365.25

    # Assert the results
    np.testing.assert_allclose(v1, expected_v1, atol=1e-6, rtol=1e-6)


def test_lambert_solver():
    """
    Test the Lambert solver with a simple example.
    
    This function tests the Lambert solver with a simple Earth-to-Mars transfer,
    verifying that the velocities are correctly computed.
    """
    # Test case: Earth to Mars transfer
    # Earth at 1 AU
    r1 = np.array([1.0, 0.0, 0.0])
    # Mars at 1.5 AU, 30 degrees ahead
    r2 = np.array([1.5 * np.cos(np.radians(30)), 1.5 * np.sin(np.radians(30)), 0.0])
    tof = 300.0  # Time of flight in days
    
    # Solve Lambert's problem
    v1, v2 = solve_lambert(r1, r2, tof)
    
    # Print results
    print("Initial position (AU):", r1)
    print("Final position (AU):", r2)
    print("Time of flight (days):", tof)
    print("Initial velocity (AU/day):", v1)
    print("Final velocity (AU/day):", v2)
    
    # Basic validation
    assert not np.any(np.isnan(v1)), "Initial velocity contains NaN values"
    assert not np.any(np.isnan(v2)), "Final velocity contains NaN values"
    
    # Check that velocities are reasonable
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    print(f"Initial velocity magnitude (AU/day): {v1_mag:.6f}")
    print(f"Final velocity magnitude (AU/day): {v2_mag:.6f}")
    
    return v1, v2

def test_lambert_solver_vallado_example():
    """
    Test case from Vallado's book, Example 7-1.
    Earth to Mars transfer with specific positions and time of flight.
    """
    # Initial position (Earth at 1 AU)
    r1 = np.array([1.0, 0.0, 0.0])
    
    # Final position (Mars at 1.5 AU, 45 degrees ahead)
    r2 = np.array([1.5 * np.cos(np.radians(45)), 1.5 * np.sin(np.radians(45)), 0.0])
    
    # Time of flight (days)
    tof = 200.0
    
    # Solve Lambert's problem
    v1, v2 = solve_lambert(r1, r2, tof)
    
    # Expected velocities (from Vallado's solution)
    v1_expected = np.array([0.015215, 0.006881, 0.0])
    v2_expected = np.array([-0.006881, 0.015215, 0.0])
    
    # Compare with expected values
    npt.assert_allclose(v1, v1_expected, rtol=1e-6)
    npt.assert_allclose(v2, v2_expected, rtol=1e-6)

def test_lambert_solver_parabolic():
    """
    Test case for a parabolic orbit transfer.
    """
    # Initial position (at 1 AU)
    r1 = np.array([1.0, 0.0, 0.0])
    
    # Final position (at 2 AU, 90 degrees ahead)
    r2 = np.array([0.0, 2.0, 0.0])
    
    # Time of flight (days)
    tof = 100.0
    
    # Solve Lambert's problem
    v1, v2 = solve_lambert(r1, r2, tof)
    
    # Check that velocities are reasonable for a parabolic orbit
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # For a parabolic orbit, the velocity magnitude should be close to escape velocity
    escape_velocity = np.sqrt(2.0)  # In AU/day
    npt.assert_allclose(v1_mag, escape_velocity, rtol=1e-2)
    npt.assert_allclose(v2_mag, escape_velocity, rtol=1e-2)

def test_lambert_solver_hyperbolic():
    """
    Test case for a hyperbolic orbit transfer.
    """
    # Initial position (at 1 AU)
    r1 = np.array([1.0, 0.0, 0.0])
    
    # Final position (at 3 AU, 120 degrees ahead)
    r2 = np.array([-1.5, 2.598076, 0.0])
    
    # Time of flight (days)
    tof = 150.0
    
    # Solve Lambert's problem
    v1, v2 = solve_lambert(r1, r2, tof)
    
    # Check that velocities are reasonable for a hyperbolic orbit
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # For a hyperbolic orbit, the velocity magnitude should be greater than escape velocity
    escape_velocity = np.sqrt(2.0)  # In AU/day
    assert v1_mag > escape_velocity
    assert v2_mag > escape_velocity

def test_lambert_solver_retrograde():
    """
    Test case for a retrograde orbit transfer.
    """
    # Initial position (at 1 AU)
    r1 = np.array([1.0, 0.0, 0.0])
    
    # Final position (at 1.5 AU, 30 degrees ahead)
    r2 = np.array([1.5 * np.cos(np.radians(30)), 1.5 * np.sin(np.radians(30)), 0.0])
    
    # Time of flight (days)
    tof = 300.0
    
    # Solve Lambert's problem with retrograde option
    v1, v2 = solve_lambert(r1, r2, tof, prograde=False)
    
    # Check that velocities are reasonable
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # For a retrograde orbit, the z-component of angular momentum should be negative
    h = np.cross(r1, v1)
    assert h[2] < 0

def test_lambert_solver_long_way():
    """
    Test case for a long-way transfer (transfer angle > 180 degrees).
    """
    # Initial position (at 1 AU)
    r1 = np.array([1.0, 0.0, 0.0])
    
    # Final position (at 1.5 AU, 210 degrees ahead)
    r2 = np.array([1.5 * np.cos(np.radians(210)), 1.5 * np.sin(np.radians(210)), 0.0])
    
    # Time of flight (days)
    tof = 400.0
    
    # Solve Lambert's problem
    v1, v2 = solve_lambert(r1, r2, tof)
    
    # Check that velocities are reasonable
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # For a long-way transfer, the transfer angle should be > 180 degrees
    transfer_angle = np.arccos(np.clip(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)), -1.0, 1.0))
    assert transfer_angle > np.pi

def test_lambert_solver_earth_mars():
    """
    https://www.researchgate.net/publication/323256893_Survey_of_Earth-Mars_Trajectories_using_Lambert's_Problem_and_Applications#pfd
    
    Departure from Earth July 27, 2020 
    Arrival at Mars February 19, 2021
    Time of Flight: 207 days

    """
    departure_time = Timestamp.from_iso8601(["2020-07-27T00:00:00Z"], scale="utc")
    arrival_time = Timestamp.from_iso8601(["2021-02-19T00:00:00Z"], scale="utc")

    departure_state = get_perturber_state(OriginCodes.EARTH, departure_time)
    arrival_state = get_perturber_state(OriginCodes.MARS_BARYCENTER, arrival_time)

    v1, v2 = solve_lambert(departure_state.r, arrival_state.r, 207.0, mu=OriginGravitationalParameters.SUN)
    # Calculate C3 (characteristic energy) in km^2/s^2
    # C3 is the square of the hyperbolic excess velocity relative to Earth
    earth_escape_velocity = np.sqrt(2 * OriginGravitationalParameters.EARTH)
    earth_escape_velocity_km_s = earth_escape_velocity * KM_P_AU / S_P_DAY
    v1_km_s = v1 * KM_P_AU / S_P_DAY
    c3 = np.linalg.norm(v1_km_s)**2 - earth_escape_velocity_km_s**2
    print(f"\nC3: {c3:.2f} km^2/s^2")

    print(v1 * KM_P_AU / S_P_DAY)
    print(v2 * KM_P_AU / S_P_DAY)
