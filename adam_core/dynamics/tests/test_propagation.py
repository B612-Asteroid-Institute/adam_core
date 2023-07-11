import numpy as np
import spiceypy as sp
from astropy import units as u
from astropy.time import Time

from ...constants import Constants as c
from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...coordinates.times import Times
from ...orbits import Orbits
from ..propagation import _propagate_2body, propagate_2body

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


def test_propagate_2body_against_spice_elliptical(orbital_elements):
    # Test propagation against SPICE for elliptical orbits.

    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values

    # Create orbits object
    orbits = Orbits.from_kwargs(
        orbit_id=orbital_elements["targetname"].values,
        object_id=orbital_elements["targetname"].values,
        coordinates=CartesianCoordinates.from_kwargs(
            x=cartesian_elements[:, 0],
            y=cartesian_elements[:, 1],
            z=cartesian_elements[:, 2],
            vx=cartesian_elements[:, 3],
            vy=cartesian_elements[:, 4],
            vz=cartesian_elements[:, 5],
            time=Times.from_astropy(
                Time(
                    t0,
                    format="mjd",
                    scale="tdb",
                )
            ),
            origin=Origin.from_kwargs(
                code=["SUN" for i in range(len(cartesian_elements))]
            ),
            frame="ecliptic",
        ),
    )

    # Set propagation times (same for all orbits)
    times = Time(
        t0.min() + np.arange(0, 10000, 10),
        format="mjd",
        scale="tdb",
    )

    # Propagate orbits with SPICE and accumulate results
    spice_propagated = []
    for i, cartesian_i in enumerate(cartesian_elements):

        # Calculate dts from t0 for this orbit (each orbit's t0 is different)
        # but the final times we are propagating to are the same for all orbits
        dts = times.tdb.mjd - t0[i]
        spice_propagated_i = np.empty((len(dts), 6))
        for j, dt_i in enumerate(dts):
            spice_propagated_i[j] = sp.prop2b(
                MU, np.ascontiguousarray(cartesian_i), dt_i
            )

        spice_propagated.append(spice_propagated_i)

    spice_propagated = np.vstack(spice_propagated)

    # Propagate orbits with adam_core
    orbits_propagated = propagate_2body(orbits, times)

    # Calculate difference between SPICE and adam_core
    diff = orbits_propagated.coordinates.values - spice_propagated

    # Calculate offset in position in cm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 10 cm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 1 mm/s
    np.testing.assert_array_less(v_diff, 1)


def test_propagate_2body_against_spice_hyperbolic(orbital_elements):
    # Test propagation against SPICE for hyperbolic orbits.

    orbital_elements = orbital_elements[orbital_elements["e"] > 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values

    # Create orbits object
    orbits = Orbits.from_kwargs(
        orbit_id=orbital_elements["targetname"].values,
        object_id=orbital_elements["targetname"].values,
        coordinates=CartesianCoordinates.from_kwargs(
            x=cartesian_elements[:, 0],
            y=cartesian_elements[:, 1],
            z=cartesian_elements[:, 2],
            vx=cartesian_elements[:, 3],
            vy=cartesian_elements[:, 4],
            vz=cartesian_elements[:, 5],
            time=Times.from_astropy(
                Time(
                    t0,
                    format="mjd",
                    scale="tdb",
                )
            ),
            origin=Origin.from_kwargs(
                code=["SUN" for i in range(len(cartesian_elements))]
            ),
            frame="ecliptic",
        ),
    )

    # Set propagation times (same for all orbits)
    times = Time(
        t0.min() + np.arange(0, 10000, 10),
        format="mjd",
        scale="tdb",
    )

    # Propagate orbits with SPICE and accumulate results
    spice_propagated = []
    for i, cartesian_i in enumerate(cartesian_elements):

        # Calculate dts from t0 for this orbit (each orbit's t0 is different)
        # but the final times we are propagating to are the same for all orbits
        dts = times.tdb.mjd - t0[i]
        spice_propagated_i = np.empty((len(dts), 6))
        for j, dt_i in enumerate(dts):
            spice_propagated_i[j] = sp.prop2b(
                MU, np.ascontiguousarray(cartesian_i), dt_i
            )

        spice_propagated.append(spice_propagated_i)

    spice_propagated = np.vstack(spice_propagated)

    # Propagate orbits with adam_core
    orbits_propagated = propagate_2body(orbits, times)

    # Calculate difference between SPICE and adam_core
    diff = orbits_propagated.coordinates.values - spice_propagated

    # Calculate offset in position in cm
    r_diff = np.linalg.norm(diff[:, :3], axis=1) * u.au.to(u.cm)
    # Calculate offset in velocity in mm/s
    v_diff = np.linalg.norm(diff[:, 3:], axis=1) * (u.au / u.d).to(u.mm / u.s)

    # Assert positions are to within 10 cm
    np.testing.assert_array_less(r_diff, 10)
    # Assert velocities are to within 1 mm/s
    np.testing.assert_array_less(v_diff, 1)


def test_benchmark_propagate_2body(benchmark, orbital_elements):
    t0 = orbital_elements["mjd_tdb"].values
    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    cartesian_elements = orbital_elements[["x", "y", "z", "vx", "vy", "vz"]].values
    t0 = orbital_elements["mjd_tdb"].values
    # Create orbits object
    orbits = Orbits.from_kwargs(
        orbit_id=orbital_elements["targetname"].values,
        object_id=orbital_elements["targetname"].values,
        coordinates=CartesianCoordinates.from_kwargs(
            x=cartesian_elements[:, 0],
            y=cartesian_elements[:, 1],
            z=cartesian_elements[:, 2],
            vx=cartesian_elements[:, 3],
            vy=cartesian_elements[:, 4],
            vz=cartesian_elements[:, 5],
            time=Times.from_astropy(
                Time(
                    t0,
                    format="mjd",
                    scale="tdb",
                )
            ),
            origin=Origin.from_kwargs(
                code=["SUN" for i in range(len(cartesian_elements))]
            ),
            frame="ecliptic",
        ),
    )
    times = Time(
        t0.min() + np.arange(0, 100, 10),
        format="mjd",
        scale="tdb",
    )
    benchmark(propagate_2body, orbits, times=times)
