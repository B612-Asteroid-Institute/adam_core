import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from ... import _rust_native
from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...coordinates.residuals import Residuals
from ...orbits.orbits import Orbits
from ...orbits.variants import VariantOrbits
from ...propagator import Propagator
from ...time import Timestamp
from .._native_dispatch import native_propagator
from ..evaluate import OrbitDeterminationObservations

if TYPE_CHECKING:
    from ..config import IodRunConfig

from ..fitted_orbits import FittedOrbitMembers, FittedOrbits

logger = logging.getLogger(__name__)

__all__ = ["IodCandidate", "IodCandidateFailure", "IodReport"]


@dataclass(frozen=True)
class IodCandidate:
    """
    One Gauss candidate scored by `run_iod`. Row indices refer to the
    observations as passed to `run_iod`.
    """

    #: Heliocentric ecliptic Cartesian state (au, au/day) at the epoch.
    state: tuple[float, float, float, float, float, float]
    #: Gauss epoch, exact (integer days and nanoseconds in ``epoch_scale``).
    epoch_days: int
    epoch_nanos: int
    epoch_scale: str
    #: Rows of the earliest and latest observation of the search window.
    window: tuple[int, int]
    #: Rows of the three observations the candidate was built from.
    triplet: tuple[int, int, int]
    #: Which of the triplet's ``root_count`` physical Gauss roots this is.
    root_index: int
    root_count: int
    velocity_method: str
    #: Two-body score of the preliminary orbit on the configured prefit
    #: scope: the first correction-stage rows by default (``"local"``), or
    #: the whole arc when ``prefit_scope="whole_arc"``.
    prefit_chi2: float
    prefit_reduced_chi2: float
    #: Rows the outlier trial removed to reach acceptance (empty otherwise).
    removed: tuple[int, ...]
    #: Statistics after the outlier trial (the prefit ones unless rows were
    #: removed).
    chi2: float
    reduced_chi2: float
    num_obs: int
    arc_length: float
    #: Corrected acceptance: reduced chi2 of the corrected orbit on the
    #: observations it kept (None when not corrected or stalled), whether the staged
    #: correction reached the whole arc, the corrected state, the stages
    #: run and why it stopped short, if it did.
    postfit_reduced_chi2: Optional[float]
    converged: bool
    accepted: bool
    corrected_state: Optional[tuple[float, float, float, float, float, float]] = None
    stages: int = 0
    #: Whole-arc score that ranks converged candidates: every observation's
    #: chi2 clipped at CMC2003's rejection threshold, over the whole arc's
    #: degrees of freedom (None when not converged).
    whole_arc_score: Optional[float] = None
    correction_note: Optional[str] = None
    #: ``"gauss"`` (a Gauss root of ``triplet``) or ``"ranging"`` (a
    #: systematic-ranging orbit; ``triplet`` repeats the attributable's
    #: reference row and ``root_index`` is its rank by chi2).
    source: str = "gauss"


@dataclass(frozen=True)
class RangingSample:
    """
    The systematic-ranging sample of an arc Gauss could not handle: the
    attributable (position and rates at the reference observation), and per
    orbit in the admissible region its heliocentric ecliptic state at the
    attributable's epoch, range, range rate, chi2 and normalized physical
    cell probability. Before normalization each node has weight
    ``exp(-chi2 / 2) * rho * delta(log(rho)) * delta(rho_dot)``: the ``rho``
    Jacobian converts the log-spaced range grid to the physical
    ``d(rho) d(rho_dot)`` measure, including the actual admissible range-rate
    cell width.
    """

    attributable: dict[str, Any]
    states: np.ndarray
    rho_au: np.ndarray
    rho_dot_au_per_day: np.ndarray
    chi2: np.ndarray
    weight: np.ndarray
    best: Optional[int]
    grid_points: int
    admissible_points: int
    num_obs: int

    def to_variant_orbits(self, object_id: Optional[str] = None) -> VariantOrbits:
        """The sample as weighted `VariantOrbits` of one orbit
        (``orbit_id`` ``"ranging"``), for propagating or screening it."""
        n = len(self.weight)
        a = self.attributable
        time = Timestamp.from_kwargs(
            days=[int(a["epoch_days"])] * n,
            nanos=[int(a["epoch_nanos"])] * n,
            scale=a["epoch_scale"],
        )
        states = np.asarray(self.states, dtype=np.float64).reshape(n, 6)
        return VariantOrbits.from_kwargs(
            orbit_id=["ranging"] * n,
            object_id=[object_id] * n,
            variant_id=[str(i) for i in range(n)],
            weights=self.weight,
            weights_cov=self.weight,
            coordinates=CartesianCoordinates.from_kwargs(
                x=states[:, 0],
                y=states[:, 1],
                z=states[:, 2],
                vx=states[:, 3],
                vy=states[:, 4],
                vz=states[:, 5],
                time=time,
                origin=Origin.from_kwargs(code=["SUN"] * n),
                frame="ecliptic",
            ),
        )


def _ranging_from_output(ranging: Optional[dict[str, Any]]) -> Optional[RangingSample]:
    if ranging is None:
        return None
    return RangingSample(
        attributable=dict(ranging["attributable"]),
        states=np.asarray(ranging["states"], dtype=np.float64).reshape(-1, 6),
        rho_au=np.asarray(ranging["rho_au"], dtype=np.float64),
        rho_dot_au_per_day=np.asarray(ranging["rho_dot_au_per_day"], dtype=np.float64),
        chi2=np.asarray(ranging["chi2"], dtype=np.float64),
        weight=np.asarray(ranging["weight"], dtype=np.float64),
        best=None if ranging["best"] is None else int(ranging["best"]),
        grid_points=int(ranging["grid_points"]),
        admissible_points=int(ranging["admissible_points"]),
        num_obs=int(ranging["num_obs"]),
    )


@dataclass(frozen=True)
class IodCandidateFailure:
    """Candidate-local numerical failure retained by the bounded IOD search."""

    #: Caller-order rows of the triplet whose root failed.
    triplet: tuple[int, int, int]
    root_index: int
    message: str


@dataclass(frozen=True)
class IodReport:
    """
    What focused IOD searched: every scored candidate (`IodCandidate`), which
    one was accepted, the search counters and the reason when none was.
    A dataclass rather than a table: candidates are few and their rows
    reference the observations by index.
    """

    candidates: tuple[IodCandidate, ...]
    #: Index into ``candidates`` of the accepted orbit; None when none was.
    accepted: Optional[int]
    triplets_tried: int
    candidates_evaluated: int
    candidates_failed: int
    corrections_run: int
    not_found_reason: Optional[str]
    #: Work counters are independent of contamination/outlier policy. The
    #: search stops deterministically when any configured cap is exhausted.
    pair_visits: int = 0
    raw_pair_visits: int = 0
    middle_visits: int = 0
    search_cap_exhausted: bool = False
    #: Candidate-local row failures skipped while later roots/triplets ran.
    candidate_failures: tuple[IodCandidateFailure, ...] = ()
    #: Indices of converged candidates that are distinct solutions (outside
    #: the accepted orbit's 3-sigma region) fitting about as well (chi2
    #: within the same margin); non-empty means the IOD is ambiguous.
    ambiguous_with: tuple[int, ...] = ()
    #: The systematic-ranging sample (`RangingSample`) when ranging ran: the
    #: arc was too short for an orbit, or no Gauss candidate converged.
    ranging: Optional[RangingSample] = None

    @property
    def found(self) -> bool:
        return self.accepted is not None

    @property
    def ambiguous(self) -> bool:
        return bool(self.ambiguous_with)

    @property
    def accepted_candidate(self) -> Optional[IodCandidate]:
        return None if self.accepted is None else self.candidates[self.accepted]


def _state6(values: Any) -> tuple[float, float, float, float, float, float]:
    state = tuple(float(value) for value in values)
    if len(state) != 6:
        raise ValueError(f"native IOD state must contain 6 values, got {len(state)}")
    return state


def _triplet3(values: Any) -> tuple[int, int, int]:
    triplet = tuple(int(value) for value in values)
    if len(triplet) != 3:
        raise ValueError(
            f"native IOD observation triplet must contain 3 rows, got {len(triplet)}"
        )
    return triplet


def _report_from_iod_output(output: dict[str, Any]) -> IodReport:
    """The `IodReport` of a Rust IOD dict (`run_iod_ipc` or the ``iod`` entry
    of an adam-core ``full_od`` / ``run_od`` output)."""
    candidates = tuple(
        IodCandidate(
            state=_state6(candidate["state"]),
            epoch_days=int(candidate["epoch_days"]),
            epoch_nanos=int(candidate["epoch_nanos"]),
            epoch_scale=candidate["epoch_scale"],
            window=(int(candidate["window"][0]), int(candidate["window"][1])),
            triplet=_triplet3(candidate["triplet"]),
            root_index=int(candidate["root_index"]),
            root_count=int(candidate["root_count"]),
            velocity_method=candidate["velocity_method"],
            prefit_chi2=float(candidate["prefit_chi2"]),
            prefit_reduced_chi2=float(candidate["prefit_reduced_chi2"]),
            removed=tuple(int(row) for row in candidate["removed"]),
            chi2=float(candidate["chi2"]),
            reduced_chi2=float(candidate["reduced_chi2"]),
            num_obs=int(candidate["num_obs"]),
            arc_length=float(candidate["arc_length"]),
            postfit_reduced_chi2=candidate.get("postfit_reduced_chi2"),
            whole_arc_score=candidate.get("whole_arc_score"),
            source=candidate.get("source", "gauss"),
            converged=bool(candidate.get("converged", False)),
            accepted=bool(candidate["accepted"]),
            corrected_state=(
                None
                if candidate.get("corrected_state") is None
                else _state6(candidate["corrected_state"])
            ),
            stages=int(candidate.get("stages", 0)),
            correction_note=candidate.get("correction_note"),
        )
        for candidate in output.get("candidates", ())
    )
    accepted = output.get("accepted")
    return IodReport(
        candidates=candidates,
        accepted=None if accepted is None else int(accepted),
        triplets_tried=int(output.get("triplets_tried", 0)),
        candidates_evaluated=int(output.get("candidates_evaluated", 0)),
        candidates_failed=int(output.get("candidates_failed", 0)),
        corrections_run=int(output.get("corrections_run", 0)),
        not_found_reason=output.get("not_found_reason"),
        pair_visits=int(output.get("pair_visits", 0)),
        raw_pair_visits=int(output.get("raw_pair_visits", 0)),
        middle_visits=int(output.get("middle_visits", 0)),
        search_cap_exhausted=bool(output.get("search_cap_exhausted", False)),
        candidate_failures=tuple(
            IodCandidateFailure(
                triplet=_triplet3(failure["triplet"]),
                root_index=int(failure["root_index"]),
                message=str(failure["message"]),
            )
            for failure in output.get("candidate_failures") or ()
        ),
        ambiguous_with=tuple(int(i) for i in output.get("ambiguous_with") or ()),
        ranging=_ranging_from_output(output.get("ranging")),
    )


def _empty_report(reason: str) -> IodReport:
    return IodReport((), None, 0, 0, 0, 0, reason)


def _run_iod_prepared(
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    stage_config: "IodRunConfig",
    *,
    native_backend: object | None = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers, IodReport]:
    """Run the Rust IOD kernel on astrometry prepared by the caller."""
    if len(observations) == 0:
        return (
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            _empty_report("no observations"),
        )
    orbit_id = uuid.uuid4().hex
    from .fitting import _native_problem

    _, observed_ipc, observers_ipc = _native_problem(
        _placeholder_orbit(observations), observations
    )
    from ..config import IodRunConfig

    if not isinstance(stage_config, IodRunConfig):
        raise TypeError("stage_config must be an IodRunConfig")
    output = _rust_native.run_iod_config_ipc(
        native_propagator(propagator) if native_backend is None else native_backend,
        observed_ipc,
        observers_ipc,
        stage_config._native,
    )
    orbit, members = _tables_from_iod_output(orbit_id, observations, output)
    # Canonical provider identity and diagnostics belong to the Rust-owned batch
    # workflow. This focused algorithm returns a typed report and leaves
    # canonical provenance unset.
    return orbit, members, _report_from_iod_output(output)


def _placeholder_orbit(observations: OrbitDeterminationObservations) -> Orbits:
    """A one-row orbit table the IPC bridge needs (IOD takes no seed)."""
    return Orbits.from_kwargs(
        orbit_id=["iod"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.017],
            vz=[0.0],
            time=observations.coordinates.time[0:1],
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


def _tables_from_iod_output(
    orbit_id: str,
    observations: OrbitDeterminationObservations,
    output: dict[str, Any],
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Wrap a Rust typed-IOD decision into `FittedOrbits` / `FittedOrbitMembers`
    in the observations' order; empty tables when no orbit was accepted, in
    which case the search counters and the reason are logged at debug level.
    """
    if not output["found"]:
        reason = output.get("not_found_reason")
        if reason is not None:
            logger.debug(
                "IOD accepted no orbit for %d observations: %s "
                "(triplets tried: %s, candidates evaluated: %s, failed: %s)",
                len(observations),
                reason,
                output.get("triplets_tried"),
                output.get("candidates_evaluated"),
                output.get("candidates_failed"),
            )
        return FittedOrbits.empty(), FittedOrbitMembers.empty()
    state = np.asarray(output["state"], dtype=np.float64)
    orbit = FittedOrbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[None],
        coordinates=CartesianCoordinates.from_kwargs(
            x=state[0:1],
            y=state[1:2],
            z=state[2:3],
            vx=state[3:4],
            vy=state[4:5],
            vz=state[5:6],
            # The IOD epoch crosses exactly (integer days and nanoseconds).
            time=Timestamp.from_kwargs(
                days=[int(output["epoch_days"])],
                nanos=[int(output["epoch_nanos"])],
                scale=output["epoch_scale"],
            ),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
        arc_length=[output["arc_length"]],
        num_obs=[output["num_obs"]],
        chi2=[output["chi2"]],
        reduced_chi2=[output["reduced_chi2"]],
        success=[output["success"]],
        status_code=[output["status_code"]],
    )
    residuals = Residuals.from_kwargs(
        values=np.asarray(output["residual_values"], dtype=np.float64).tolist(),
        chi2=output["residual_chi2"],
        dof=output["residual_dof"],
        probability=output["residual_probability"],
    )
    members = FittedOrbitMembers.from_kwargs(
        orbit_id=[orbit_id] * len(observations),
        obs_id=observations.id,
        residuals=residuals,
        solution=output["solution"],
        outlier=output["outlier"],
    )
    return orbit, members
