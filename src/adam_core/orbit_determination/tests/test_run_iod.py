"""
The IOD stage is explicit about its propagator (bead od_experiments_setup-a6w.2):
`run_iod` takes the instance that scores candidates, and adam-core's full-OD
workflow takes an ``iod_propagator`` distinct from the refining one through
explicit native propagation capsules for both stages.
"""

import numpy as np
import numpy.testing as npt
import pytest

from adam_core._rust.arrow import pa

from ..algorithms.iod import (
    IodCandidateFailure,
    RangingSample,
    _report_from_iod_output,
)
from ..evaluate import OrbitDeterminationObservations
from ._focused import (
    explicit_config,
    run_iod_qualified,
    run_prefit_iod_qualified,
)
from .test_differential_correction import TwoBodyPropagator

pytest_plugins = ("adam_core.orbit_determination.tests.test_run_od_null_covariance",)


def test_run_iod_on_empty_observations() -> None:
    orbit, members, report = run_iod_qualified(
        two_body_arc_empty(), TwoBodyPropagator(), min_obs=6
    )
    assert len(orbit) == 0 and len(members) == 0
    assert not report.found and report.candidates == ()
    assert report.not_found_reason == "no observations"


def two_body_arc_empty() -> OrbitDeterminationObservations:
    return OrbitDeterminationObservations.empty()


@pytest.mark.parametrize("velocity_method", ["gibbs", "herrick+gibbs", "auto"])
def test_spans_search_finds_the_arc(
    two_body_arc: OrbitDeterminationObservations, velocity_method: str  # noqa: F811
) -> None:
    orbit, members, report = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        rchi2_threshold=1e4,
        observation_selection_method="spans",
        velocity_method=velocity_method,
    )
    assert report.found, report.not_found_reason
    assert len(orbit) == 1
    methods = {candidate.velocity_method for candidate in report.candidates}
    if velocity_method == "auto":
        assert methods <= {"gibbs", "herrick+gibbs"}
    else:
        assert methods == {velocity_method}


def test_spans_settings_reach_the_search(
    two_body_arc: OrbitDeterminationObservations,
) -> None:  # noqa: F811
    _, _, report = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        rchi2_threshold=1e-30,
        observation_selection_method="spans",
        span_settings={"span_max_triplets": 3},
        # The Gauss search alone (no ranging fallback after it fails).
        ranging_settings={"ranging_enabled": False},
    )
    assert not report.found and report.triplets_tried == 3
    times = two_body_arc.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    spans = [times[c.triplet[2]] - times[c.triplet[0]] for c in report.candidates]
    assert all(abs(span) > 0 for span in spans)
    _, _, explicit_default = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        rchi2_threshold=1e-30,
        observation_selection_method="spans",
        span_settings={"span_max_triplets": 32},
        ranging_settings={"ranging_enabled": False},
    )
    assert explicit_default.triplets_tried <= 32
    _, _, combinations_default = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        contamination_percentage=0.0,
        rchi2_threshold=1e-30,
        observation_selection_method="combinations",
        # Span-only settings must not replace the combinations budget.
        span_settings={"span_max_triplets": 1},
        ranging_settings={"ranging_enabled": False},
    )
    assert combinations_default.triplets_tried == 3
    with pytest.raises(ValueError, match="boolean.*expected usize"):
        run_iod_qualified(two_body_arc, TwoBodyPropagator(), max_triplets=True)
    with pytest.raises(ValueError, match="boolean.*expected usize"):
        run_iod_qualified(
            two_body_arc,
            TwoBodyPropagator(),
            observation_selection_method="spans",
            span_settings={"span_max_triplets": True},
        )
    with pytest.raises(ValueError, match="velocity_method"):
        run_iod_qualified(two_body_arc, TwoBodyPropagator(), velocity_method="laplace")
    with pytest.raises(ValueError, match="observation_selection_method"):
        run_iod_qualified(
            two_body_arc, TwoBodyPropagator(), observation_selection_method="best"
        )


def test_run_iod_defaults_to_corrected_acceptance(
    two_body_arc: OrbitDeterminationObservations,
) -> None:  # noqa: F811
    """The default `run_iod` corrects candidates in stages and returns the
    corrected whole-arc orbit of the best converged one; `iod()` keeps the
    legacy prefit gate."""
    orbit, members, report = run_iod_qualified(
        two_body_arc, TwoBodyPropagator(), min_obs=6
    )
    assert report.found and report.corrections_run >= 1
    assert not report.ambiguous and report.ambiguous_with == ()
    accepted = report.accepted_candidate
    assert accepted is not None
    assert accepted.converged and accepted.stages >= 1
    assert accepted.corrected_state is not None
    assert accepted.postfit_reduced_chi2 is not None
    npt.assert_array_equal(
        np.concatenate([orbit.coordinates.r[0], orbit.coordinates.v[0]]),
        np.asarray(accepted.corrected_state),
    )
    assert orbit.reduced_chi2[0].as_py() == pytest.approx(accepted.postfit_reduced_chi2)
    assert not any(members.outlier.to_pylist()), "IOD rejects nothing"
    # Lowest final reduced chi2, ties (relative 1e-6) to the lowest index.
    for candidate in report.candidates:
        if candidate.converged:
            assert candidate.postfit_reduced_chi2 is not None
            assert candidate.postfit_reduced_chi2 >= accepted.postfit_reduced_chi2 * (
                1 - 1e-6
            )
    prefit, _ = run_prefit_iod_qualified(
        two_body_arc,
        TwoBodyPropagator,
        min_obs=6,
        rchi2_threshold=1e4,
        observation_selection_method="combinations",
    )
    assert len(prefit) == 1


def test_corrected_acceptance_reports_why_nothing_converged(
    two_body_arc: OrbitDeterminationObservations,  # noqa: F811
) -> None:
    _, _, report = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        rchi2_threshold=1e-12,
        correction_settings={"max_corrections": 2},
        ranging_settings={"ranging_enabled": False},
    )
    assert not report.found
    assert report.corrections_run == 2
    assert report.not_found_reason is not None
    assert "all roots failed correction" in report.not_found_reason
    assert "none reached the whole arc" in report.not_found_reason
    assert sum(c.correction_note is not None for c in report.candidates) == 2


def test_stage_loss_toggle(
    two_body_arc: OrbitDeterminationObservations,
) -> None:  # noqa: F811
    """``stage_loss`` selects the partial-stage loss: Huber (default) and
    linear both seed a clean arc; anything else is rejected."""
    for stage_loss in ("huber", "linear"):
        _, _, report = run_iod_qualified(
            two_body_arc,
            TwoBodyPropagator(),
            min_obs=6,
            correction_settings={"stage_loss": stage_loss},
        )
        assert report.found, (stage_loss, report.not_found_reason)
    with pytest.raises(ValueError, match="loss must be one of"):
        run_iod_qualified(
            two_body_arc,
            TwoBodyPropagator(),
            min_obs=6,
            correction_settings={"stage_loss": "cauchy"},
        )


def test_corrected_acceptance_is_invariant_to_observation_order(
    two_body_arc: OrbitDeterminationObservations,  # noqa: F811
) -> None:
    permutation = np.random.default_rng(11).permutation(len(two_body_arc))
    orbit, _, report = run_iod_qualified(two_body_arc, TwoBodyPropagator(), min_obs=6)
    shuffled, _, shuffled_report = run_iod_qualified(
        two_body_arc.take(permutation), TwoBodyPropagator(), min_obs=6
    )
    assert report.accepted == shuffled_report.accepted
    # Search and corrections both run on the time-sorted view: identical.
    npt.assert_array_equal(shuffled.coordinates.values, orbit.coordinates.values)


def test_ranging_for_short_arcs_and_gauss_failures(
    two_body_arc: OrbitDeterminationObservations,
) -> None:  # noqa: F811
    """Corrected IOD routes arcs Gauss cannot handle to systematic ranging:
    a too-short arc returns the ranging sample instead of an orbit, and a
    search with no Gauss candidate accepts a corrected ranging orbit."""
    # The fixture spans 21 days; ranging's 5-day arc limit is raised.
    long_ranging = {"ranging_max_arc_days": 100.0}
    orbit, _, short = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        min_arc_length=100.0,
        ranging_settings=long_ranging,
    )
    assert len(orbit) == 0 and not short.found
    assert isinstance(short.ranging, RangingSample)
    sample = short.ranging
    assert sample.states.shape == (len(sample.weight), 6) and len(sample.weight) > 0
    assert sample.weight.sum() == pytest.approx(1.0)
    assert sample.best == int(np.argmin(sample.chi2))
    variants = sample.to_variant_orbits("arc")
    assert len(variants) == len(sample.weight)
    assert short.not_found_reason is not None
    assert "ranging sample attached" in short.not_found_reason

    no_gauss_settings = {"span_max_days": 1e-3}
    orbit, _, report = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        span_settings=no_gauss_settings,
        ranging_settings=long_ranging,
    )
    assert report.found
    assert report.accepted_candidate is not None
    assert report.accepted_candidate.source == "ranging"
    # The same orbit Gauss finds, compared at the Gauss orbit's epoch.
    from adam_core.dynamics.propagation import propagate_2body

    expected, _, _ = run_iod_qualified(two_body_arc, TwoBodyPropagator(), min_obs=6)
    moved = propagate_2body(orbit.to_orbits(), expected.coordinates.time)
    np.testing.assert_allclose(
        moved.coordinates.r[0], expected.coordinates.r[0], rtol=0, atol=1e-6
    )
    assert orbit.reduced_chi2[0].as_py() == pytest.approx(
        expected.reduced_chi2[0].as_py(), rel=1e-4
    )

    _, _, disabled = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        span_settings=no_gauss_settings,
        ranging_settings={**long_ranging, "ranging_enabled": False},
    )
    assert not disabled.found and disabled.ranging is None
    assert explicit_config().for_iod().search_correction_ranging.ranging.enabled


def test_prefit_acceptance_enforces_minimum_arc(
    two_body_arc: OrbitDeterminationObservations,
) -> None:  # noqa: F811
    orbit, _, report = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        acceptance="prefit",
        observation_selection_method="combinations",
        rchi2_threshold=1e20,
        min_arc_length=100.0,
        max_triplets=3,
    )
    assert len(orbit) == 0 and not report.found
    assert report.triplets_tried == 3
    assert report.candidates
    assert all(candidate.arc_length < 100.0 for candidate in report.candidates)


def test_zero_contamination_still_searches_later_triplets(
    two_body_arc: OrbitDeterminationObservations,  # noqa: F811
) -> None:
    _, _, report = run_iod_qualified(
        two_body_arc,
        TwoBodyPropagator(),
        acceptance="prefit",
        observation_selection_method="combinations",
        contamination_percentage=0.0,
        rchi2_threshold=1e-30,
        max_triplets=3,
    )
    assert not report.found
    assert report.triplets_tried == 3
    assert report.candidates_evaluated >= 3


def test_equal_epoch_station_rows_are_canonicalized_by_content(
    two_body_arc: OrbitDeterminationObservations,  # noqa: F811
) -> None:
    # Make rows 0 and 1 share epoch and station while retaining distinct
    # astrometry and observer states. Their canonical order must therefore be
    # physical-content order, not caller order.
    equal = two_body_arc
    for prefix in ("coordinates.time", "observers.coordinates.time"):
        time = (
            equal.coordinates.time
            if prefix == "coordinates.time"
            else equal.observers.coordinates.time
        )
        days = time.days.to_pylist()
        nanos = time.nanos.to_pylist()
        days[1], nanos[1] = days[0], nanos[0]
        equal = equal.set_column(f"{prefix}.days", pa.array(days, type=pa.int64()))
        equal = equal.set_column(f"{prefix}.nanos", pa.array(nanos, type=pa.int64()))

    kwargs = dict(
        acceptance="prefit",
        observation_selection_method="combinations",
        rchi2_threshold=1e20,
        max_triplets=6,
    )
    _, _, direct = run_iod_qualified(equal, TwoBodyPropagator(), **kwargs)
    permutation = np.array([1, 0, *range(2, len(equal))])
    swapped_input = equal.take(permutation)
    _, _, swapped = run_iod_qualified(swapped_input, TwoBodyPropagator(), **kwargs)

    direct_states = [candidate.state for candidate in direct.candidates]
    swapped_states = [candidate.state for candidate in swapped.candidates]
    npt.assert_array_equal(swapped_states, direct_states)
    direct_ids = [
        tuple(equal.id[row].as_py() for row in candidate.triplet)
        for candidate in direct.candidates
    ]
    swapped_ids = [
        tuple(swapped_input.id[row].as_py() for row in candidate.triplet)
        for candidate in swapped.candidates
    ]
    assert swapped_ids == direct_ids


def test_candidate_failure_diagnostics_are_public_and_typed() -> None:
    report = _report_from_iod_output(
        {
            "candidate_failures": [
                {
                    "triplet": [4, 1, 3],
                    "root_index": 2,
                    "message": "row failure: no convergence",
                }
            ]
        }
    )
    assert report.candidate_failures == (
        IodCandidateFailure((4, 1, 3), 2, "row failure: no convergence"),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_pair_visits": 0}, "max_pair_visits"),
        ({"fit_settings": {"prop_tol": 0.0}}, "propagation_tol"),
    ],
)
def test_run_iod_validates_all_settings_before_empty_exit(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_iod_qualified(two_body_arc_empty(), TwoBodyPropagator(), **kwargs)
