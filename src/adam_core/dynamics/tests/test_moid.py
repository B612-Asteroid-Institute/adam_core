import numpy as np
import quivr as qv

from ...coordinates.keplerian import KeplerianCoordinates
from ...coordinates.origin import Origin, OriginCodes
from ...orbits import Orbits
from ...time import Timestamp
from ...utils.spice import get_perturber_state
from ..moid import calculate_moid, calculate_perturber_moids


def test_calculate_moid_identical_circular_flat_minimum_time_is_witness():
    epoch_mjd = 59000.0
    circular = Orbits.from_kwargs(
        orbit_id=["a"],
        coordinates=KeplerianCoordinates.from_kwargs(
            a=[1.0],
            e=[0.0],
            i=[0.0],
            ap=[0.0],
            raan=[0.0],
            M=[0.0],
            time=Timestamp.from_mjd([epoch_mjd], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        ).to_cartesian(),
    )

    moid, moid_time = calculate_moid(circular, circular)

    np.testing.assert_allclose(moid, 0.0, rtol=0, atol=1e-12)
    dt_at_min = moid_time.mjd().to_numpy(zero_copy_only=False)[0] - epoch_mjd
    period = 2.0 * np.pi * np.sqrt(1.0 / circular.coordinates.origin.mu()[0])
    assert 0.0 <= dt_at_min <= period


def test_calculate_moid_delegates_science_to_shared_rust_kernel(monkeypatch):
    epoch_mjd = 59000.0
    orbit = Orbits.from_kwargs(
        orbit_id=["delegation"],
        coordinates=KeplerianCoordinates.from_kwargs(
            a=[1.0],
            e=[0.1],
            i=[2.0],
            ap=[3.0],
            raan=[4.0],
            M=[5.0],
            time=Timestamp.from_mjd([epoch_mjd], scale="tdb"),
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"]),
        ).to_cartesian(),
    )
    from adam_core._rust import api as rust_api

    calls = []

    def _native(primary, secondary, mu):
        calls.append((primary.copy(), secondary.copy(), mu))
        return 0.25, 1.5

    monkeypatch.setattr(rust_api, "calculate_moid_numpy", _native)
    moid, witness = calculate_moid(orbit, orbit)

    assert moid == 0.25
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], orbit.coordinates.values[0])
    np.testing.assert_array_equal(calls[0][1], orbit.coordinates.values[0])
    assert calls[0][2] == orbit.coordinates.origin.mu()[0]
    assert witness.mjd().to_numpy(zero_copy_only=False)[0] == epoch_mjd + 1.5


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


def test_calculate_perturber_moids():
    pertubers = [OriginCodes.EARTH, OriginCodes.MARS_BARYCENTER, OriginCodes.MERCURY]

    time = Timestamp.from_mjd([59000.0], scale="tdb")

    earth = Orbits.from_kwargs(
        orbit_id=["input_EARTH"],
        coordinates=get_perturber_state(OriginCodes.EARTH, time),
    )
    mars = Orbits.from_kwargs(
        orbit_id=["input_MARS"],
        coordinates=get_perturber_state(OriginCodes.MARS_BARYCENTER, time),
    )
    orbits = qv.concatenate([earth, mars])

    moids = calculate_perturber_moids(orbits, pertubers)

    assert len(moids) == 6
    assert (
        len(moids.select("orbit_id", "input_EARTH").select("perturber.code", "EARTH"))
        == 1
    )
    assert (
        len(
            moids.select("orbit_id", "input_EARTH").select(
                "perturber.code", "MARS_BARYCENTER"
            )
        )
        == 1
    )
    assert (
        len(moids.select("orbit_id", "input_MARS").select("perturber.code", "EARTH"))
        == 1
    )
    assert (
        len(
            moids.select("orbit_id", "input_MARS").select(
                "perturber.code", "MARS_BARYCENTER"
            )
        )
        == 1
    )
    assert (
        len(moids.select("orbit_id", "input_EARTH").select("perturber.code", "MERCURY"))
        == 1
    )
    assert (
        len(moids.select("orbit_id", "input_MARS").select("perturber.code", "MERCURY"))
        == 1
    )


def test_calculate_perturber_moids_uses_record_batch_and_direct_wrap(monkeypatch):
    """The public facade must cross once as a RecordBatch and wrap directly."""
    import numpy as np
    import pyarrow as pa

    from adam_core.coordinates import CartesianCoordinates, Origin, OriginCodes
    from adam_core.dynamics.moid import PerturberMOIDs, calculate_perturber_moids
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    orbits = Orbits.from_kwargs(
        orbit_id=["o1", "o2"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.2, 2.1],
            y=[0.1, -0.4],
            z=[0.02, 0.05],
            vx=[-0.002, 0.001],
            vy=[0.012, 0.009],
            vz=[0.0004, -0.0008],
            time=Timestamp.from_mjd([60000.0, 60001.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )

    from adam_core._rust import api as rust_api
    from adam_core.utils import spice as spice_module

    # Initialization is an explicit caller responsibility, not hidden inside
    # the MOID facade.
    spice_module.setup_SPICE()
    native = rust_api.calculate_perturber_moids_arrow
    called = {"value": False}

    def _checked(orbit_batch, perturber_codes, *args, **kwargs):
        assert isinstance(orbit_batch, pa.RecordBatch)
        called["value"] = True
        return native(orbit_batch, perturber_codes, *args, **kwargs)

    def _forbid_from_kwargs(*args, **kwargs):
        raise AssertionError("PerturberMOIDs.from_kwargs must not rebuild Rust output")

    def _forbid_implicit_spice_setup(*args, **kwargs):
        raise AssertionError("calculate_perturber_moids must not initialize SPICE")

    monkeypatch.setattr(spice_module, "setup_SPICE", _forbid_implicit_spice_setup)
    monkeypatch.setattr(
        "adam_core.dynamics.moid.calculate_perturber_moids_arrow",
        _checked,
        raising=False,
    )
    monkeypatch.setattr(rust_api, "calculate_perturber_moids_arrow", _checked)
    monkeypatch.setattr(PerturberMOIDs, "from_kwargs", _forbid_from_kwargs)

    result = calculate_perturber_moids(
        orbits, [OriginCodes.EARTH, OriginCodes.MARS_BARYCENTER]
    )
    assert called["value"]
    assert len(result) == 4
    assert result.orbit_id.to_pylist() == ["o1", "o2", "o1", "o2"]
    assert result.perturber.code.to_pylist() == [
        "EARTH",
        "EARTH",
        "MARS_BARYCENTER",
        "MARS_BARYCENTER",
    ]
    assert result.time.scale == "tdb"
    assert np.all(np.isfinite(result.moid.to_numpy(zero_copy_only=False)))
