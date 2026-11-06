"""Parity and native-timing coverage for the Rust-owned OD utility kernels."""

import numpy as np
import pytest
import quivr as qv

from adam_core._rust.arrow import pa

from ...coordinates.origin import Origin
from ...dynamics.lambert import calculate_c3
from ...time.time import Timestamp


def test_calculate_c3_matches_legacy_expression() -> None:
    rng = np.random.default_rng(42)
    v1 = rng.normal(scale=0.02, size=(64, 3))
    body_v = rng.normal(scale=0.02, size=(64, 3))
    expected = np.linalg.norm(v1 - body_v, axis=1) ** 2
    np.testing.assert_array_equal(calculate_c3(v1, body_v), expected)


def test_calculate_c3_native_timing_returns_samples() -> None:
    from adam_core import _rust_native

    rng = np.random.default_rng(7)
    v1 = rng.normal(size=(32, 3))
    body_v = rng.normal(size=(32, 3))
    trials = _rust_native.benchmark_calculate_c3_numpy(v1, body_v, 3, 2, 1)
    assert len(trials) == 2
    assert all(len(samples) == 3 for samples in trials)
    assert all(sample > 0.0 for samples in trials for sample in samples)


def _legacy_assign_duplicate_observations(orbits, orbit_members):
    from adam_core._rust.arrow import pc

    orbits = orbits.sort_by(
        [
            ("num_obs", "descending"),
            ("arc_length", "descending"),
            ("reduced_chi2", "ascending"),
        ]
    )
    unique_obs_ids = pc.unique(orbit_members.column("obs_id"))
    best_orbit_for_obs = {}
    for obs_id in unique_obs_ids:
        mask = pc.equal(orbit_members.column("obs_id"), obs_id)
        obs_orbit_ids = orbit_members.where(mask).column("orbit_id")
        for sorted_orbit_id in orbits.column("orbit_id"):
            if pc.any(pc.is_in(sorted_orbit_id, value_set=obs_orbit_ids)).as_py():
                best_orbit_for_obs[obs_id.as_py()] = sorted_orbit_id.as_py()
                break
    for obs_id, best_orbit_id in best_orbit_for_obs.items():
        mask_to_remove = pc.and_(
            pc.equal(orbit_members.column("obs_id"), pa.scalar(obs_id)),
            pc.not_equal(orbit_members.column("orbit_id"), pa.scalar(best_orbit_id)),
        )
        orbit_members = orbit_members.apply_mask(pc.invert(mask_to_remove))
    orbits_mask = pc.is_in(
        orbits.column("orbit_id"), value_set=orbit_members.column("orbit_id")
    )
    return orbits.apply_mask(orbits_mask), orbit_members


def test_assign_duplicate_observations_matches_legacy() -> None:
    from ...coordinates.cartesian import CartesianCoordinates
    from ..fitted_orbits import (
        FittedOrbitMembers,
        FittedOrbits,
        assign_duplicate_observations,
    )

    rng = np.random.default_rng(3)
    num_orbits = 6
    orbit_ids = [f"orbit{i:02d}" for i in range(num_orbits)]
    time = Timestamp.from_mjd(np.full(num_orbits, 59000.0), scale="tdb")
    orbits = FittedOrbits.from_kwargs(
        orbit_id=orbit_ids,
        coordinates=CartesianCoordinates.from_kwargs(
            x=rng.normal(size=num_orbits),
            y=rng.normal(size=num_orbits),
            z=rng.normal(size=num_orbits),
            vx=rng.normal(size=num_orbits),
            vy=rng.normal(size=num_orbits),
            vz=rng.normal(size=num_orbits),
            time=time,
            origin=Origin.from_kwargs(code=np.full(num_orbits, "SUN", dtype="object")),
            frame="ecliptic",
        ),
        arc_length=rng.choice([5.0, 10.0, 20.0], size=num_orbits),
        num_obs=rng.integers(3, 8, size=num_orbits),
        chi2=rng.uniform(1.0, 50.0, size=num_orbits),
        reduced_chi2=rng.uniform(0.5, 5.0, size=num_orbits),
    )

    member_orbit_ids = []
    member_obs_ids = []
    for orbit_id in orbit_ids:
        for obs in rng.choice(20, size=6, replace=False):
            member_orbit_ids.append(orbit_id)
            member_obs_ids.append(f"obs{obs:02d}")
    orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=member_orbit_ids,
        obs_id=member_obs_ids,
    )

    expected_orbits, expected_members = _legacy_assign_duplicate_observations(
        orbits, orbit_members
    )
    actual_orbits, actual_members = assign_duplicate_observations(orbits, orbit_members)

    assert actual_orbits.orbit_id.to_pylist() == expected_orbits.orbit_id.to_pylist()
    assert actual_members.orbit_id.to_pylist() == expected_members.orbit_id.to_pylist()
    assert actual_members.obs_id.to_pylist() == expected_members.obs_id.to_pylist()


@pytest.mark.parametrize(
    ("arc_length", "reduced_chi2", "expected_order"),
    [
        (
            [np.nan, 20.0, 10.0, 5.0],
            [1.0, 1.0, 1.0, 1.0],
            ["orbit01", "orbit02", "orbit03", "orbit00"],
        ),
        (
            [10.0, 10.0, 10.0, 10.0],
            [np.nan, 2.0, 1.0, 3.0],
            ["orbit02", "orbit01", "orbit03", "orbit00"],
        ),
        (
            [np.nan, 10.0, np.nan, 10.0],
            [np.nan, np.nan, 1.0, 2.0],
            ["orbit03", "orbit01", "orbit02", "orbit00"],
        ),
        (
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            ["orbit00", "orbit01", "orbit02", "orbit03"],
        ),
    ],
)
def test_assign_duplicate_observations_preserves_arrow_nan_ordering(
    arc_length: list[float],
    reduced_chi2: list[float],
    expected_order: list[str],
) -> None:
    from ...coordinates.cartesian import CartesianCoordinates
    from ..fitted_orbits import (
        FittedOrbitMembers,
        FittedOrbits,
        assign_duplicate_observations,
    )

    num_orbits = 4
    orbit_ids = [f"orbit{i:02d}" for i in range(num_orbits)]
    time = Timestamp.from_mjd(np.full(num_orbits, 59000.0), scale="tdb")
    orbits = FittedOrbits.from_kwargs(
        orbit_id=orbit_ids,
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.arange(num_orbits, dtype=np.float64),
            y=np.zeros(num_orbits),
            z=np.zeros(num_orbits),
            vx=np.zeros(num_orbits),
            vy=np.zeros(num_orbits),
            vz=np.zeros(num_orbits),
            time=time,
            origin=Origin.from_kwargs(code=np.full(num_orbits, "SUN", dtype="object")),
            frame="ecliptic",
        ),
        arc_length=arc_length,
        num_obs=np.full(num_orbits, 3),
        chi2=np.ones(num_orbits),
        reduced_chi2=reduced_chi2,
    )
    orbit_members = FittedOrbitMembers.from_kwargs(
        orbit_id=orbit_ids,
        obs_id=[f"obs{i:02d}" for i in range(num_orbits)],
    )

    expected_orbits, expected_members = _legacy_assign_duplicate_observations(
        orbits, orbit_members
    )
    actual_orbits, actual_members = assign_duplicate_observations(orbits, orbit_members)

    assert expected_orbits.orbit_id.to_pylist() == expected_order
    assert actual_orbits.orbit_id.to_pylist() == expected_order
    assert actual_members.orbit_id.to_pylist() == expected_members.orbit_id.to_pylist()
    assert actual_members.obs_id.to_pylist() == expected_members.obs_id.to_pylist()


class _Linkages(qv.Table):
    cluster_id = qv.LargeStringColumn()


class _Members(qv.Table):
    cluster_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()
