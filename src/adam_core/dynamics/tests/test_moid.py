import numpy as np

from ...coordinates.keplerian import KeplerianCoordinates
from ...coordinates.origin import Origin
from ...orbits import Orbits
from ...time import Timestamp
from ..moid import calculate_moid


def test_calculate_moid_circular_orbits():
    # Test for two coplanar circular orbits that their MOID
    # is equal to the difference of their semi-major axes
    circular_1 = Orbits.from_kwargs(
        orbit_id=["a"],
        coordinates=KeplerianCoordinates.from_kwargs(
            a=[1.0],
            e=[0.0],
            i=[0.0],
            ap=[0.0],
            raan=[0.0],
            M=[0.0],
            time=Timestamp.from_mjd([59000.0], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        ).to_cartesian(),
    )

    circular_3 = Orbits.from_kwargs(
        orbit_id=["b"],
        coordinates=KeplerianCoordinates.from_kwargs(
            a=[3.0],
            e=[0.0],
            i=[0.0],
            ap=[0.0],
            raan=[0.0],
            M=[0.0],
            time=Timestamp.from_mjd([59000.0], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        ).to_cartesian(),
    )

    moid_13, moid_time_13 = calculate_moid(circular_1, circular_3)
    np.testing.assert_allclose(moid_13, 2.0, rtol=0, atol=1e-12)

    moid_31, moid_time_31 = calculate_moid(circular_3, circular_1)
    np.testing.assert_allclose(moid_31, 2.0, rtol=0, atol=1e-12)

    np.testing.assert_allclose(moid_13, moid_31, rtol=0, atol=1e-12)

    # Test that the MOID of two orbits with relative inclination of 90 degrees
    # is equal to the difference of their semi-major axes
    circular_1 = Orbits.from_kwargs(
        orbit_id=["a"],
        coordinates=KeplerianCoordinates.from_kwargs(
            a=[1.0],
            e=[0.0],
            i=[0.0],
            ap=[0.0],
            raan=[0.0],
            M=[0.0],
            time=Timestamp.from_mjd([59000.0], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        ).to_cartesian(),
    )

    inclined_3 = Orbits.from_kwargs(
        orbit_id=["b"],
        coordinates=KeplerianCoordinates.from_kwargs(
            a=[3.0],
            e=[0.0],
            i=[90.0],
            ap=[0.0],
            raan=[0.0],
            M=[0.0],
            time=Timestamp.from_mjd([59000.0], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        ).to_cartesian(),
    )

    moid_13, moid_time_13 = calculate_moid(circular_1, inclined_3)
    np.testing.assert_allclose(moid_13, 2.0, rtol=0, atol=1e-12)

    moid_31, moid_time_31 = calculate_moid(inclined_3, circular_1)
    np.testing.assert_allclose(moid_31, 2.0, rtol=0, atol=1e-12)

    np.testing.assert_allclose(moid_13, moid_31, rtol=0, atol=1e-12)


def test_calculate_moid_noncircular_orbits():

    # Test that the MOID of two coplanar but non-circular orbits located on
    # the x-axis at same time is equal to the difference
    # of their periapse distances
    j2000 = Timestamp.from_jd([2451545.0], scale="tt")

    noncircular_keplerian_1 = KeplerianCoordinates.from_kwargs(
        a=[2.0],
        e=[0.2],
        i=[0.0],
        ap=[0.0],
        raan=[0.0],
        M=[0.0],
        time=j2000,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"]),
    )
    noncircular_1 = Orbits.from_kwargs(
        orbit_id=["a"],
        coordinates=noncircular_keplerian_1.to_cartesian(),
    )

    noncircular_keplerian_2 = KeplerianCoordinates.from_kwargs(
        a=[5.0],
        e=[0.2],
        i=[0.0],
        ap=[0.0],
        raan=[0.0],
        M=[0.0],
        time=j2000,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"]),
    )
    noncircular_2 = Orbits.from_kwargs(
        orbit_id=["b"],
        coordinates=noncircular_keplerian_2.to_cartesian(),
    )

    moid_13, moid_time_13 = calculate_moid(noncircular_1, noncircular_2)
    np.testing.assert_allclose(
        moid_13,
        noncircular_keplerian_2.q[0] - noncircular_keplerian_1.q[0],
        rtol=0,
        atol=1e-12,
    )

    moid_31, moid_time_31 = calculate_moid(noncircular_2, noncircular_1)
    np.testing.assert_allclose(
        moid_31,
        noncircular_keplerian_2.q[0] - noncircular_keplerian_1.q[0],
        rtol=0,
        atol=1e-12,
    )

    # Test again with two orbits that overlap at some point
    noncircular_keplerian_1 = KeplerianCoordinates.from_kwargs(
        a=[2.0],
        e=[0.1],
        i=[0.0],
        ap=[0.0],
        raan=[0.0],
        M=[0.0],
        time=j2000,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"]),
    )
    noncircular_1 = Orbits.from_kwargs(
        orbit_id=["a"],
        coordinates=noncircular_keplerian_1.to_cartesian(),
    )

    noncircular_keplerian_2 = KeplerianCoordinates.from_kwargs(
        a=[3.0],
        e=[0.9],
        i=[0.0],
        ap=[0.0],
        raan=[0.0],
        M=[0.0],
        time=j2000,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"]),
    )
    noncircular_2 = Orbits.from_kwargs(
        orbit_id=["b"],
        coordinates=noncircular_keplerian_2.to_cartesian(),
    )

    moid_13, moid_time_13 = calculate_moid(noncircular_1, noncircular_2)
    np.testing.assert_allclose(moid_13, 0.0, rtol=0, atol=1e-6)

    moid_31, moid_time_31 = calculate_moid(noncircular_2, noncircular_1)
    np.testing.assert_allclose(moid_31, 0.0, rtol=0, atol=1e-6)
