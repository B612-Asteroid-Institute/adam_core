"""Parity and native-timing coverage for the Rust-owned OD utility kernels."""

import numpy as np
import pyarrow as pa
import pytest
import quivr as qv

from ...coordinates.origin import Origin
from ...coordinates.spherical import SphericalCoordinates
from ...dynamics.lambert import calculate_c3
from ...observers.observers import Observers
from ...time.time import Timestamp
from ..differential_correction import OrbitDeterminationObservations
from ..iod import select_observations, sort_by_id_and_time


def _make_observations(times_mjd: np.ndarray) -> OrbitDeterminationObservations:
    time = Timestamp.from_mjd(times_mjd, scale="utc")
    n = len(times_mjd)
    ids = [f"obs{i:02d}" for i in range(n)]
    return OrbitDeterminationObservations.from_kwargs(
        id=ids,
        coordinates=SphericalCoordinates.from_kwargs(
            lon=np.linspace(10.0, 20.0, n),
            lat=np.linspace(-5.0, 5.0, n),
            origin=Origin.from_kwargs(code=np.full(n, "500", dtype="object")),
            time=time,
        ),
        observers=Observers.from_code("500", time),
    )


def _legacy_select_observations(obs_ids, times, method):
    from itertools import combinations

    indexes = np.arange(0, len(obs_ids))
    if method in ("first+middle+last", "thirds"):
        quantiles = (
            [0, 50, 100]
            if method == "first+middle+last"
            else [
                1 / 6 * 100,
                50,
                5 / 6 * 100,
            ]
        )
        selected_times = np.percentile(times, quantiles, method="nearest")
        selected_index = np.intersect1d(times, selected_times, return_indices=True)[1]
        selected_index = np.array([selected_index])
    else:
        selected_index = np.array(
            [np.array(index) for index in combinations(indexes, 3)]
        )
        arc_length = times[selected_index][:, 2] - times[selected_index][:, 0]
        time_from_mid = np.abs(
            (times[selected_index][:, 2] + times[selected_index][:, 0]) / 2
            - times[selected_index][:, 1]
        )
        sort = np.lexsort((time_from_mid, -arc_length))
        selected_index = selected_index[sort]

    keep = [
        i for i, comb in enumerate(times[selected_index]) if len(np.unique(comb)) == 3
    ]
    if len(keep) == 0:
        return np.array([])
    return obs_ids[selected_index[np.array(keep), :]]


def test_calculate_c3_matches_legacy_expression():
    rng = np.random.default_rng(42)
    v1 = rng.normal(scale=0.02, size=(64, 3))
    body_v = rng.normal(scale=0.02, size=(64, 3))
    expected = np.linalg.norm(v1 - body_v, axis=1) ** 2
    np.testing.assert_array_equal(calculate_c3(v1, body_v), expected)


def test_calculate_c3_native_timing_returns_samples():
    from adam_core import _rust_native

    rng = np.random.default_rng(7)
    v1 = rng.normal(size=(32, 3))
    body_v = rng.normal(size=(32, 3))
    trials = _rust_native.benchmark_calculate_c3_numpy(v1, body_v, 3, 2, 1)
    assert len(trials) == 2
    assert all(len(samples) == 3 for samples in trials)
    assert all(sample > 0.0 for samples in trials for sample in samples)


@pytest.mark.parametrize("method", ["combinations", "first+middle+last", "thirds"])
def test_select_observations_matches_legacy(method):
    rng = np.random.default_rng(11)
    times_mjd = np.sort(59000.0 + rng.uniform(0.0, 30.0, size=9))
    # Introduce a duplicated time so the unique-time filter is exercised.
    times_mjd[4] = times_mjd[3]
    observations = _make_observations(times_mjd)

    obs_ids = observations.id.to_numpy(zero_copy_only=False)
    times = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    expected = _legacy_select_observations(obs_ids, times, method)
    actual = select_observations(observations, method=method)
    np.testing.assert_array_equal(actual, expected)


def test_select_observations_rejects_unknown_method():
    observations = _make_observations(np.array([59000.0, 59001.0, 59002.0]))
    with pytest.raises(ValueError, match="method should be one of"):
        select_observations(observations, method="nope")


def test_select_observations_too_few_returns_empty():
    observations = _make_observations(np.array([59000.0, 59001.0]))
    assert len(select_observations(observations)) == 0


def _legacy_assign_duplicate_observations(orbits, orbit_members):
    import pyarrow.compute as pc

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


def test_assign_duplicate_observations_matches_legacy():
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


class _Linkages(qv.Table):
    cluster_id = qv.LargeStringColumn()


class _Members(qv.Table):
    cluster_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()


def test_sort_by_id_and_time_matches_legacy_ordering():
    observations = _make_observations(
        np.array([59003.0, 59001.0, 59002.0, 59000.5, 59004.0, 59000.0])
    )
    linkages = _Linkages.from_kwargs(cluster_id=["c2", "c0", "c1"])
    members = _Members.from_kwargs(
        cluster_id=["c2", "c2", "c0", "c0", "c1", "c1"],
        obs_id=["obs00", "obs05", "obs04", "obs01", "obs02", "obs03"],
    )

    sorted_linkages, sorted_members = sort_by_id_and_time(
        linkages, members, observations, "cluster_id"
    )

    assert sorted_linkages.cluster_id.to_pylist() == ["c0", "c1", "c2"]
    assert sorted_members.cluster_id.to_pylist() == [
        "c0",
        "c0",
        "c1",
        "c1",
        "c2",
        "c2",
    ]
    # Within each linkage, members are ordered by observation time.
    assert sorted_members.obs_id.to_pylist() == [
        "obs01",
        "obs04",
        "obs03",
        "obs02",
        "obs05",
        "obs00",
    ]


def test_sort_by_id_and_time_missing_observation_errors():
    observations = _make_observations(np.array([59000.0, 59001.0, 59002.0]))
    linkages = _Linkages.from_kwargs(cluster_id=["c0"])
    members = _Members.from_kwargs(cluster_id=["c0"], obs_id=["missing"])
    with pytest.raises(ValueError, match="not present in observations"):
        sort_by_id_and_time(linkages, members, observations, "cluster_id")


def test_sort_by_id_and_time_handles_chunked_observations():
    first = _make_observations(np.array([59002.0, 59001.0]))
    second = _make_observations(np.array([59000.0]))
    # Concatenation produces multi-chunk observation columns.
    observations = qv.concatenate([first, second])
    ids = observations.id.to_pylist()
    assert len(ids) == 3
    observations = observations.set_column(
        "id", pa.array(["obs00", "obs01", "obs02"], type=pa.large_string())
    )
    linkages = _Linkages.from_kwargs(cluster_id=["c0"])
    members = _Members.from_kwargs(
        cluster_id=["c0", "c0", "c0"], obs_id=["obs00", "obs01", "obs02"]
    )
    _, sorted_members = sort_by_id_and_time(
        linkages, members, observations, "cluster_id"
    )
    assert sorted_members.obs_id.to_pylist() == ["obs02", "obs01", "obs00"]
