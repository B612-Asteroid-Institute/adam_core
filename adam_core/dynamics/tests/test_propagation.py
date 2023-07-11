import numpy as np
import spiceypy as sp
from astropy import units as u

from ...constants import Constants as c
from ..propagation import _propagate_2body

MU = c.MU


def test__propagate_2body_against_spice_elliptical(orbital_elements):
    # Test propagation against SPICE for elliptical orbits.

    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values

    dts = np.arange(0, 10000, 10)
    for t0_i, cartesian_i in zip(t0, cartesian_elements):

        spice_propagated = np.empty((len(dts), 6))
        cartesian_propagated = np.empty((len(dts), 6))

        for j, dt_i in enumerate(dts):

            spice_propagated[j] = sp.prop2b(MU, np.ascontiguousarray(cartesian_i), dt_i)
            cartesian_propagated[j] = _propagate_2body(cartesian_i, t0_i, t0_i + dt_i)

        # Calculate difference between SPICE and adam_core
        diff = cartesian_propagated - spice_propagated

        # Calculate offset in position in cm
        r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
        # Calculate offset in velocity in mm/s
        v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

        # Assert positions are to within 10 cm
        np.testing.assert_array_less(r_diff, 10)
        # Assert velocities are to within 1 mm/s
        np.testing.assert_array_less(v_diff, 1)


def test__propagate_2body_against_spice_hyperbolic(orbital_elements):
    # Test propagation against SPICE for hyperbolic orbits.

    orbital_elements = orbital_elements[orbital_elements["e"] > 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values

    dts = np.arange(0, 10000, 10)
    for t0_i, cartesian_i in zip(t0, cartesian_elements):

        spice_propagated = np.empty((len(dts), 6))
        cartesian_propagated = np.empty((len(dts), 6))

        for j, dt_i in enumerate(dts):

            spice_propagated[j] = sp.prop2b(MU, np.ascontiguousarray(cartesian_i), dt_i)
            cartesian_propagated[j] = _propagate_2body(cartesian_i, t0_i, t0_i + dt_i)

        # Calculate difference between SPICE and adam_core
        diff = cartesian_propagated - spice_propagated

        # Calculate offset in position in cm
        r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
        # Calculate offset in velocity in mm/s
        v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

        # Assert positions are to within 10 cm
        np.testing.assert_array_less(r_diff, 10)
        # Assert velocities are to within 1 mm/s
        np.testing.assert_array_less(v_diff, 1)
