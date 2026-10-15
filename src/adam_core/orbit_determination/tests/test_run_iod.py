"""
The IOD stage is explicit about its propagator (bead od_experiments_setup-a6w.2):
`run_iod` takes the instance that scores candidates, and the fused full-OD work
units take an ``iod_propagator`` distinct from the refining one, on the
callback route as well as on the two-body route.
"""

import numpy as np
import numpy.testing as npt
import pytest

from adam_core import _rust_native

from ..differential_correction import _native_problem
from ..iod import IodReport, _placeholder_orbit, iod, run_iod
from ..native_orbit_fitter import NativeOrbitFitter
from .test_differential_correction import NativeTwoBodyPropagator, TwoBodyPropagator
from .test_run_od_null_covariance import two_body_arc  # noqa: F401


class CountingPropagator(TwoBodyPropagator):
    def __init__(self) -> None:
        self.calls = 0

    def generate_ephemeris(self, orbits, observers, **kwargs):
        self.calls += 1
        return super().generate_ephemeris(orbits, observers, **kwargs)


def test_iod_is_the_accepted_orbit_of_run_iod(two_body_arc) -> None:  # noqa: F811
    legacy_orbit, legacy_members = iod(
        two_body_arc, TwoBodyPropagator, min_obs=6, rchi2_threshold=1e4
    )
    orbit, members, report = run_iod(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        rchi2_threshold=1e4,
        observation_selection_method="combinations",
        acceptance="prefit",
    )
    assert isinstance(report, IodReport) and report.found
    npt.assert_array_equal(legacy_orbit.coordinates.values, orbit.coordinates.values)
    assert legacy_orbit.coordinates.time.equals(orbit.coordinates.time)
    assert legacy_members.solution.to_pylist() == members.solution.to_pylist()
    assert legacy_members.outlier.to_pylist() == members.outlier.to_pylist()
    npt.assert_array_equal(
        legacy_members.residuals.chi2.to_numpy(zero_copy_only=False),
        members.residuals.chi2.to_numpy(zero_copy_only=False),
    )


def test_run_iod_prefers_the_fused_work_unit(two_body_arc) -> None:  # noqa: F811
    callback = CountingPropagator()
    _, _, via_callback = run_iod(two_body_arc, callback, min_obs=6, rchi2_threshold=1e4)
    assert (
        callback.calls
        > via_callback.candidates_evaluated + via_callback.candidates_failed
    ), "scoring calls plus the correction stages"
    assert via_callback.corrections_run >= 1
    _, _, fused = run_iod(
        two_body_arc, NativeTwoBodyPropagator(), min_obs=6, rchi2_threshold=1e4
    )
    assert fused.accepted == via_callback.accepted
    assert [c.state for c in fused.candidates] == [
        c.state for c in via_callback.candidates
    ]
    npt.assert_allclose(
        fused.accepted_candidate.corrected_state,
        via_callback.accepted_candidate.corrected_state,
        rtol=0,
        atol=1e-12,
    )


def test_run_iod_on_empty_observations() -> None:
    orbit, members, report = run_iod(
        two_body_arc_empty(), TwoBodyPropagator(), min_obs=6
    )
    assert len(orbit) == 0 and len(members) == 0
    assert not report.found and report.candidates == ()
    assert report.not_found_reason == "no observations"


def two_body_arc_empty():
    from ..evaluate import OrbitDeterminationObservations

    return OrbitDeterminationObservations.empty()


@pytest.mark.parametrize("entry", ["full_od_ipc", "run_od_ipc"])
def test_callback_route_honours_a_distinct_iod_propagator(
    two_body_arc, entry: str  # noqa: F811
) -> None:
    """Two Python propagators: IOD candidates are scored on ``iod_propagator``
    only, and refinement predictions go to ``propagator`` only."""
    iod_propagator = CountingPropagator()
    refine_propagator = CountingPropagator()
    _, observed_ipc, observers_ipc = _native_problem(
        _placeholder_orbit(two_body_arc), two_body_arc
    )
    iod_settings = {"min_obs": 6, "rchi2_threshold": 1e4}
    if entry == "full_od_ipc":
        output = _rust_native.full_od_ipc(
            refine_propagator,
            observed_ipc,
            observers_ipc,
            iod_settings=iod_settings,
            fit_settings={"validate_covariance": False},
            iod_propagator=iod_propagator,
        )
    else:
        output = _rust_native.run_od_ipc(
            refine_propagator,
            observed_ipc,
            observers_ipc,
            two_body_arc.observers.code.to_pylist(),
            two_body_arc.photometry.band.to_pylist(),
            two_body_arc.astcat.to_pylist(),
            iod_settings=iod_settings,
            fit_settings={"validate_covariance": False},
            iod_propagator=iod_propagator,
        )
    assert output["found"]
    report = output["iod"]
    assert (
        iod_propagator.calls
        > report["candidates_evaluated"] + report["candidates_failed"]
    ), "candidate scoring and correction stages run on the IOD propagator"
    assert report["corrections_run"] >= 1
    assert refine_propagator.calls > 0
    # Same dynamics on both: the same answer as a single propagator.
    single = _rust_native.full_od_ipc(
        CountingPropagator(),
        observed_ipc,
        observers_ipc,
        iod_settings=iod_settings,
        fit_settings={"validate_covariance": False},
    )
    npt.assert_allclose(
        np.asarray(output["fit"]["state"]),
        np.asarray(single["fit"]["state"]),
        rtol=0,
        atol=1e-12,
    )


def test_fitter_initial_fit_runs_iod_on_the_configured_class(
    two_body_arc,  # noqa: F811
) -> None:
    class Configured(CountingPropagator):
        instances: list = []

        def __init__(self) -> None:
            super().__init__()
            type(self).instances.append(self)

    fitter = NativeOrbitFitter(propagator_class=Configured, iod_rchi2_threshold=1e4)
    orbit, members = fitter.initial_fit("x", two_body_arc)
    assert len(orbit) == 1 and len(members) == len(two_body_arc)
    assert len(Configured.instances) == 1 and Configured.instances[0].calls > 0


@pytest.mark.parametrize("velocity_method", ["gibbs", "herrick+gibbs", "auto"])
def test_spans_search_finds_the_arc(
    two_body_arc, velocity_method  # noqa: F811
) -> None:
    orbit, members, report = run_iod(
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


def test_spans_settings_reach_the_search(two_body_arc) -> None:  # noqa: F811
    _, _, report = run_iod(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        rchi2_threshold=1e-30,
        observation_selection_method="spans",
        span_settings={"span_max_triplets": 3},
    )
    assert not report.found and report.triplets_tried == 3
    times = two_body_arc.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    spans = [times[c.triplet[2]] - times[c.triplet[0]] for c in report.candidates]
    assert all(abs(span) > 0 for span in spans)
    with pytest.raises(ValueError, match="velocity_method"):
        run_iod(two_body_arc, TwoBodyPropagator(), velocity_method="laplace")
    with pytest.raises(ValueError, match="observation_selection_method"):
        run_iod(two_body_arc, TwoBodyPropagator(), observation_selection_method="best")


def test_fitter_forwards_the_velocity_method() -> None:
    fitter = NativeOrbitFitter(
        propagator_class=TwoBodyPropagator,
        observation_selection_method="spans",
        velocity_method="auto",
    )
    iod_settings, _, _ = fitter.native_settings()
    assert iod_settings["velocity_method"] == "auto"
    assert iod_settings["observation_selection_method"] == "spans"


def test_run_iod_defaults_to_corrected_acceptance(two_body_arc) -> None:  # noqa: F811
    """The default `run_iod` corrects candidates in stages and returns the
    corrected whole-arc orbit of the best converged one; `iod()` keeps the
    legacy prefit gate."""
    orbit, members, report = run_iod(two_body_arc, TwoBodyPropagator(), min_obs=6)
    assert report.found and report.corrections_run >= 1
    assert not report.ambiguous and report.ambiguous_with == ()
    accepted = report.accepted_candidate
    assert accepted.converged and accepted.stages >= 1
    assert accepted.corrected_state is not None
    npt.assert_array_equal(
        np.concatenate([orbit.coordinates.r[0], orbit.coordinates.v[0]]),
        np.asarray(accepted.corrected_state),
    )
    assert orbit.reduced_chi2[0].as_py() == pytest.approx(accepted.postfit_reduced_chi2)
    assert not any(members.outlier.to_pylist()), "IOD rejects nothing"
    # Lowest final reduced chi2, ties (relative 1e-6) to the lowest index.
    for candidate in report.candidates:
        if candidate.converged:
            assert candidate.postfit_reduced_chi2 >= accepted.postfit_reduced_chi2 * (
                1 - 1e-6
            )
    legacy, _ = iod(two_body_arc, TwoBodyPropagator, min_obs=6, rchi2_threshold=1e4)
    assert len(legacy) == 1


def test_corrected_acceptance_reports_why_nothing_converged(
    two_body_arc,  # noqa: F811
) -> None:
    _, _, report = run_iod(
        two_body_arc,
        TwoBodyPropagator(),
        min_obs=6,
        rchi2_threshold=1e-12,
        correction_settings={"max_corrections": 2},
    )
    assert not report.found
    assert report.corrections_run == 2
    assert "all roots failed correction" in report.not_found_reason
    assert "none reached the whole arc" in report.not_found_reason
    assert sum(c.correction_note is not None for c in report.candidates) == 2


def test_stage_loss_toggle(two_body_arc) -> None:  # noqa: F811
    """``stage_loss`` selects the partial-stage loss: Huber (default) and
    linear both seed a clean arc; anything else is rejected."""
    for stage_loss in ("huber", "linear"):
        _, _, report = run_iod(
            two_body_arc,
            TwoBodyPropagator(),
            min_obs=6,
            correction_settings={"stage_loss": stage_loss},
        )
        assert report.found, (stage_loss, report.not_found_reason)
    with pytest.raises(ValueError, match="loss must be one of"):
        run_iod(
            two_body_arc,
            TwoBodyPropagator(),
            min_obs=6,
            correction_settings={"stage_loss": "cauchy"},
        )


def test_corrected_acceptance_is_invariant_to_observation_order(
    two_body_arc,  # noqa: F811
) -> None:
    permutation = np.random.default_rng(11).permutation(len(two_body_arc))
    orbit, _, report = run_iod(two_body_arc, TwoBodyPropagator(), min_obs=6)
    shuffled, _, shuffled_report = run_iod(
        two_body_arc.take(permutation), TwoBodyPropagator(), min_obs=6
    )
    assert report.accepted == shuffled_report.accepted
    npt.assert_allclose(
        shuffled.coordinates.values, orbit.coordinates.values, rtol=0, atol=1e-12
    )


def test_signature_defaults_are_the_rust_defaults(two_body_arc) -> None:  # noqa: F811
    """`IOD_DEFAULTS` is Rust's ``IodConfig::default()``: the signature
    defaults of `run_iod` and `NativeOrbitFitter` are its values, and passing
    every span / correction key explicitly reproduces the default search."""
    import inspect

    from adam_core.orbit_determination import IOD_DEFAULTS, NativeOrbitFitter

    run_iod_params = inspect.signature(run_iod).parameters
    for key in (
        "min_obs",
        "min_arc_length",
        "contamination_percentage",
        "rchi2_threshold",
        "observation_selection_method",
        "light_time",
        "max_triplets",
        "velocity_method",
        "acceptance",
    ):
        assert run_iod_params[key].default == IOD_DEFAULTS[key], key
    fitter_params = inspect.signature(NativeOrbitFitter).parameters
    for fitter_key, key in (
        ("min_obs", "min_obs"),
        ("min_arc_length", "min_arc_length"),
        ("contamination_percentage", "contamination_percentage"),
        ("iod_rchi2_threshold", "rchi2_threshold"),
        ("observation_selection_method", "observation_selection_method"),
        ("velocity_method", "velocity_method"),
        ("iod_acceptance", "acceptance"),
        ("f_scale", "huber_f_scale"),
    ):
        assert fitter_params[fitter_key].default == IOD_DEFAULTS[key], fitter_key

    span_settings = {k: v for k, v in IOD_DEFAULTS.items() if k.startswith("span_")}
    correction_settings = {
        k: IOD_DEFAULTS[k]
        for k in (
            "max_corrections",
            "max_converged",
            "correction_growth",
            "min_stage_obs",
            "max_stages",
            "stage_loss",
            "prefit_scope",
        )
    }
    _, _, implicit = run_iod(two_body_arc, TwoBodyPropagator())
    _, _, explicit = run_iod(
        two_body_arc,
        TwoBodyPropagator(),
        span_settings=span_settings,
        correction_settings=correction_settings,
    )
    assert explicit == implicit
