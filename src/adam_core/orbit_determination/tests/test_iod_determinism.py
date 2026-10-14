"""
IOD is deterministic in the caller's row order (PR #217 review round 2,
phase 1, item 4): the Rust driver sorts observations by epoch, station and
original index before searching triplets and maps every member array back
to the input order, so a shuffled copy of the observations yields the same
accepted orbit and the same per-observation flags.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest
import quivr as qv

from adam_core import _rust_native

from ..differential_correction import _native_problem
from ..iod import _placeholder_orbit, iod, run_iod
from .test_differential_correction import TwoBodyPropagator
from .test_run_od_null_covariance import two_body_arc  # noqa: F401


def flags_by_id(members) -> dict[str, tuple[bool, bool]]:
    return dict(
        zip(
            members.obs_id.to_pylist(),
            zip(members.solution.to_pylist(), members.outlier.to_pylist()),
        )
    )


@pytest.mark.parametrize("contamination_percentage", [0.0, 20.0])
def test_iod_is_invariant_to_observation_order(
    two_body_arc, contamination_percentage  # noqa: F811
) -> None:
    settings = dict(
        min_obs=6,
        rchi2_threshold=1e4,
        contamination_percentage=contamination_percentage,
    )
    fitted, members = iod(two_body_arc, TwoBodyPropagator, **settings)
    assert len(fitted) == 1
    assert members.obs_id.to_pylist() == two_body_arc.id.to_pylist()

    permutation = np.random.default_rng(123).permutation(len(two_body_arc))
    shuffled = two_body_arc.take(permutation)
    assert shuffled.id.to_pylist() != two_body_arc.id.to_pylist()
    fitted_shuffled, members_shuffled = iod(shuffled, TwoBodyPropagator, **settings)

    assert len(fitted_shuffled) == 1
    npt.assert_array_equal(
        fitted_shuffled.coordinates.values, fitted.coordinates.values
    )
    assert fitted_shuffled.coordinates.time.equals(fitted.coordinates.time)
    assert fitted_shuffled.num_obs[0].as_py() == fitted.num_obs[0].as_py()
    assert fitted_shuffled.chi2[0].as_py() == fitted.chi2[0].as_py()
    # Members come back in the caller's (shuffled) order with the same flags
    # per observation id.
    assert members_shuffled.obs_id.to_pylist() == shuffled.id.to_pylist()
    assert flags_by_id(members_shuffled) == flags_by_id(members)
    assert sum(members.solution.to_pylist()) == 3
    chi2 = members.residuals.chi2.to_numpy(zero_copy_only=False)
    chi2_shuffled = members_shuffled.residuals.chi2.to_numpy(zero_copy_only=False)
    npt.assert_array_equal(chi2_shuffled, chi2[permutation])


def test_iod_uses_a_duplicated_epoch_at_most_once_per_triplet(
    two_body_arc,  # noqa: F811
) -> None:
    """A second observation at an existing epoch never completes a triplet
    with its twin, and the search still accepts the arc."""
    twin = two_body_arc[-1:].set_column("id", pa.array(["twin"], pa.large_string()))
    with_twin = qv.concatenate([two_body_arc, twin])
    assert with_twin.coordinates.time[-1].equals(with_twin.coordinates.time[-2])

    fitted, members = iod(with_twin, TwoBodyPropagator, min_obs=6, rchi2_threshold=1e4)
    assert len(fitted) == 1
    assert len(members) == len(with_twin)
    flags = flags_by_id(members)
    assert sum(solution for solution, _ in flags.values()) == 3
    last_id = two_body_arc.id[-1].as_py()
    assert not (flags[last_id][0] and flags["twin"][0])
    assert not any(outlier for _, outlier in flags.values())


def test_iod_search_budget_is_a_separate_setting(two_body_arc) -> None:  # noqa: F811
    """``max_triplets`` bounds the triplet search on its own; the
    contamination percentage keeps setting only the outlier allowance."""
    _, observed_ipc, observers_ipc = _native_problem(
        _placeholder_orbit(two_body_arc), two_body_arc
    )

    def run(**overrides):
        settings = dict(min_obs=6, rchi2_threshold=1e-30, contamination_percentage=20.0)
        settings.update(overrides)
        return _rust_native.iod_fit_ipc(
            TwoBodyPropagator(), observed_ipc, observers_ipc, iod_settings=settings
        )

    # Nothing passes a reduced chi2 threshold of 1e-30, so the search runs
    # to its budget: the legacy coupling gives 3 * (max_outliers + 1)
    # triplets, an explicit budget replaces it.
    n = len(two_body_arc)
    max_outliers = min(int(n * 0.2), n - 6)
    legacy = run()
    assert not legacy["found"]
    assert legacy["triplets_tried"] == 3 * (max_outliers + 1)
    assert "none within rchi2_threshold" in legacy["not_found_reason"]

    bounded = run(max_triplets=2)
    assert not bounded["found"]
    assert bounded["triplets_tried"] == 2

    wider = run(max_triplets=3 * (max_outliers + 1) + 2)
    assert wider["triplets_tried"] == 3 * (max_outliers + 1) + 2

    with pytest.raises(ValueError, match="max_triplets"):
        run(max_triplets=0)


def test_iod_epoch_crosses_as_exact_days_and_nanoseconds(
    two_body_arc,  # noqa: F811
) -> None:
    """The accepted orbit's epoch leaves Rust as integer days and
    nanoseconds in the observations' time scale, never as a float MJD, and
    the Python tables carry exactly those values."""
    _, observed_ipc, observers_ipc = _native_problem(
        _placeholder_orbit(two_body_arc), two_body_arc
    )

    def run(**overrides):
        settings = dict(min_obs=6, rchi2_threshold=1e4)
        settings.update(overrides)
        return _rust_native.iod_fit_ipc(
            TwoBodyPropagator(), observed_ipc, observers_ipc, iod_settings=settings
        )

    output = run()
    assert output["found"]
    assert "epoch_mjd" not in output
    assert type(output["epoch_days"]) is int
    assert type(output["epoch_nanos"]) is int
    assert 0 <= output["epoch_nanos"] < 86_400_000_000_000
    assert output["epoch_scale"] == two_body_arc.coordinates.time.scale

    fitted, _ = iod(two_body_arc, TwoBodyPropagator, min_obs=6, rchi2_threshold=1e4)
    assert fitted.coordinates.time.scale == output["epoch_scale"]
    assert fitted.coordinates.time.days[0].as_py() == output["epoch_days"]
    assert fitted.coordinates.time.nanos[0].as_py() == output["epoch_nanos"]

    not_found = run(rchi2_threshold=1e-30, max_triplets=1)
    assert not not_found["found"]
    assert not_found["epoch_days"] is None
    assert not_found["epoch_nanos"] is None
    assert not_found["epoch_scale"] == output["epoch_scale"]


def test_run_iod_reports_every_scored_candidate(two_body_arc) -> None:  # noqa: F811
    """`run_iod` returns the accepted orbit plus the typed candidate set:
    one record per scored candidate with its triplet rows, root and
    statistics, the accepted index, and counters that agree with the list."""
    orbit, members, report = run_iod(
        two_body_arc, TwoBodyPropagator(), min_obs=6, rchi2_threshold=1e4
    )
    assert len(orbit) == 1 and len(members) == len(two_body_arc)
    assert report.found and report.accepted is not None
    assert len(report.candidates) == report.candidates_evaluated
    assert report.corrections_run == 0
    assert report.not_found_reason is None
    accepted = report.accepted_candidate
    assert accepted is not None and accepted.accepted
    assert sum(candidate.accepted for candidate in report.candidates) == 1
    # Exact epoch and state of the accepted candidate are the orbit's.
    assert orbit.coordinates.time.days[0].as_py() == accepted.epoch_days
    assert orbit.coordinates.time.nanos[0].as_py() == accepted.epoch_nanos
    assert orbit.coordinates.time.scale == accepted.epoch_scale
    npt.assert_array_equal(
        np.concatenate([orbit.coordinates.r[0], orbit.coordinates.v[0]]),
        np.asarray(accepted.state),
    )
    # The triplet rows are the solution rows of the members table.
    solution_rows = [i for i, flag in enumerate(members.solution.to_pylist()) if flag]
    assert sorted(accepted.triplet) == solution_rows
    n = len(two_body_arc)
    times = two_body_arc.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    for candidate in report.candidates:
        assert 0 <= candidate.window[0] < n and 0 <= candidate.window[1] < n
        assert times[candidate.window[0]] == times.min()
        assert times[candidate.window[1]] == times.max()
        assert len(set(candidate.triplet)) == 3
        assert len({times[row] for row in candidate.triplet}) == 3
        assert 0 <= candidate.root_index < candidate.root_count
        assert candidate.velocity_method == "gibbs"
        assert candidate.postfit_reduced_chi2 is None and not candidate.converged
        if not candidate.accepted:
            assert candidate.removed == ()
            assert candidate.chi2 == candidate.prefit_chi2

    # A not-found search still reports every candidate and the reason.
    _, _, none = run_iod(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        rchi2_threshold=1e-30,
        max_triplets=2,
    )
    assert not none.found and none.accepted is None
    assert none.triplets_tried == 2
    assert len(none.candidates) == none.candidates_evaluated > 0
    assert all(not candidate.accepted for candidate in none.candidates)
    assert "none within rchi2_threshold" in none.not_found_reason


def test_run_iod_candidate_set_is_invariant_to_observation_order(
    two_body_arc,  # noqa: F811
) -> None:
    n = len(two_body_arc)
    permutation = np.random.default_rng(3).permutation(n)
    shuffled = two_body_arc.take(permutation)
    _, _, report = run_iod(
        two_body_arc, TwoBodyPropagator(), min_obs=6, rchi2_threshold=1e4
    )
    _, _, report_shuffled = run_iod(
        shuffled, TwoBodyPropagator(), min_obs=6, rchi2_threshold=1e4
    )
    # shuffled row j holds input row permutation[j]: map rows back.
    inverse = {int(permutation[j]): j for j in range(n)}
    assert report.accepted == report_shuffled.accepted
    assert len(report.candidates) == len(report_shuffled.candidates)
    for a, b in zip(report.candidates, report_shuffled.candidates):
        assert a.state == b.state
        assert (a.epoch_days, a.epoch_nanos) == (b.epoch_days, b.epoch_nanos)
        assert sorted(inverse[row] for row in a.triplet) == sorted(b.triplet)
        assert sorted(inverse[row] for row in a.removed) == sorted(b.removed)
        assert (inverse[a.window[0]], inverse[a.window[1]]) == b.window
        assert (a.root_index, a.root_count, a.accepted) == (
            b.root_index,
            b.root_count,
            b.accepted,
        )
        assert a.prefit_chi2 == b.prefit_chi2 and a.chi2 == b.chi2
