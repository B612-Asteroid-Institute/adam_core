import numpy as np

from mpcq.orbits import MPCOrbits

from ...time import Timestamp


def test_mpcq_orbits_maps_nongrav_parameters_into_adam_core():
    mpc_orbits = MPCOrbits.from_kwargs(
        requested_provid=["test-a", "test-b"],
        primary_designation=["test-a", "test-b"],
        id=[1, 2],
        provid=["test-a", "test-b"],
        epoch=Timestamp.from_mjd([60000.0, 60001.0], scale="tdb"),
        q=[0.8, 1.1],
        e=[0.2, 0.1],
        i=[5.0, 3.0],
        node=[120.0, 130.0],
        argperi=[45.0, 50.0],
        peri_time=[59980.0, 59990.0],
        q_unc=[0.01, 0.02],
        e_unc=[0.001, 0.002],
        i_unc=[0.1, 0.2],
        node_unc=[0.1, 0.2],
        argperi_unc=[0.1, 0.2],
        peri_time_unc=[0.5, 0.6],
        a1=[None, 1.2e-12],
        a2=[-8.7e-14, None],
        a3=[None, 3.4e-14],
        h=[18.0, 19.0],
        g=[0.15, 0.25],
        created_at=Timestamp.from_mjd([60000.0, 60001.0], scale="tdb"),
        updated_at=Timestamp.from_mjd([60000.0, 60001.0], scale="tdb"),
    )

    orbits = mpc_orbits.orbits()

    assert len(orbits) == 2
    assert orbits.non_gravitational_parameters.source.to_pylist() == ["MPCQ", "MPCQ"]
    assert orbits.non_gravitational_parameters.model.to_pylist() == [
        "nongrav",
        "nongrav",
    ]
    assert orbits.non_gravitational_parameters.estimated_parameter_names.to_pylist() == [
        "A2",
        "A1,A3",
    ]
    np.testing.assert_allclose(
        orbits.non_gravitational_parameters.A2.to_numpy(zero_copy_only=False)[0],
        -8.7e-14,
    )
    np.testing.assert_allclose(
        orbits.non_gravitational_parameters.A1.to_numpy(zero_copy_only=False)[1],
        1.2e-12,
    )
    np.testing.assert_allclose(
        orbits.non_gravitational_parameters.A3.to_numpy(zero_copy_only=False)[1],
        3.4e-14,
    )
