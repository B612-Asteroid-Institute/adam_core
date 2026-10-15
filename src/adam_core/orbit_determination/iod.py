import logging
import time
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import quivr as qv

from .. import _rust_native
from ..coordinates import CartesianCoordinates
from ..coordinates.origin import Origin
from ..coordinates.residuals import Residuals
from ..orbits.orbits import Orbits
from ..orbits.variants import VariantOrbits
from ..propagator import Propagator
from ..time import Timestamp
from ..utils.iter import _iterate_chunks
from ._native_dispatch import deprecated, native_work_unit
from .evaluate import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, drop_duplicate_orbits
from .gauss import MU, C
from .observation_uncertainty import ObservationUncertaintyModel
from .provenance import with_iod_provenance

#: The IOD defaults, read once from Rust's ``IodConfig::default()`` (with its
#: span search and correction settings, under the keys of ``run_iod``'s
#: ``span_settings`` / ``correction_settings``, and ``huber_f_scale``). The
#: signature defaults of `run_iod` and `NativeOrbitFitter` are these values;
#: the policy and the source of each number are documented on the Rust types
#: and in the orbit-determination use-case page.
IOD_DEFAULTS = MappingProxyType(_rust_native.iod_defaults())

logger = logging.getLogger(__name__)

__all__ = ["IodCandidate", "IodReport", "initial_orbit_determination", "run_iod"]


def sort_by_id_and_time(
    linkages: qv.AnyTable,
    members: qv.AnyTable,
    observations: OrbitDeterminationObservations,
    linkage_column: str,
) -> Tuple[qv.AnyTable, qv.AnyTable]:
    """
    Sort linkages and linkage members by linkage ID and observation time.

    Parameters
    ----------
    linkages : qv.AnyTable
        Linkages to sort.
    members : qv.AnyTable
        Linkage members to sort.
    observations : Observations
        Observations from which linkage members were generated. Observations
        are used to determine the observation time of each linkage member.
    linkage_column : str
        Column name in the linkage table to use for sorting. For clusters
        this is "cluster_id" and for orbits this is "orbit_id".

    Returns
    -------
    linkages : qv.AnyTable
        Sorted linkages.
    members : qv.AnyTable
        Sorted linkage members.
    """
    from adam_core import _rust_native

    # One Rust crossing owns the observation-time join and the stable
    # (linkage id, observation time) ordering.
    time_table = observations.coordinates.time.table.combine_chunks()
    linkage_order, member_order = _rust_native.sort_linkages_by_id_and_time_numpy(
        linkages.table.column(linkage_column).to_pylist(),
        members.table.column(linkage_column).to_pylist(),
        members.table.column("obs_id").to_pylist(),
        observations.id.to_pylist(),
        time_table.column("days").chunk(0).to_numpy(zero_copy_only=False),
        time_table.column("nanos").chunk(0).to_numpy(zero_copy_only=False),
    )

    linkages = linkages.take(pa.array(linkage_order, type=pa.int64()))
    members = members.take(pa.array(member_order, type=pa.int64()))

    if linkages.fragmented():
        linkages = qv.defragment(linkages)
    if members.fragmented():
        members = qv.defragment(members)
    return linkages, members


def select_observations(
    observations: OrbitDeterminationObservations,
    method: Literal["combinations", "first+middle+last", "thirds"] = "combinations",
) -> npt.NDArray[np.str_]:
    """
    Selects which three observations to use for IOD depending on the method.

    Methods:
        'first+middle+last' : Grab the first, middle and last observations in time.
        'thirds' : Grab the middle observation in the first third, second third, and final third.
        'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
            non-coinciding observation times.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations with at least a column of observation IDs and a column
        of exposure times.
    method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Which method to use to select observations.
        [Default = 'combinations']

    Returns
    -------
    obs_id : `~numpy.ndarray' (N, 3 or 0)
        An array of selected observation IDs. If three unique observations could
        not be selected then returns an empty array.
    """
    obs_ids = observations.id.to_numpy(zero_copy_only=False)
    if len(obs_ids) < 3:
        return np.array([])

    times = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)

    from adam_core import _rust_native

    # One Rust crossing owns percentile/combination selection, arc-length and
    # midpoint ordering, and the unique-time filter; the legacy ValueError
    # message for unknown methods is preserved by the kernel.
    selected_index = np.asarray(
        _rust_native.select_observation_triplets_numpy(
            np.ascontiguousarray(times, dtype=np.float64), method
        ),
        dtype=np.int64,
    )

    # Return an empty array if no observations satisfy the criteria
    if selected_index.shape[0] == 0:
        return np.array([])

    return obs_ids[selected_index]


def _tables_from_native_iod(output: dict) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """Wrap one fused Rust IOD batch product without further computation."""
    orbit_ids = list(output["orbit_ids"])
    if not orbit_ids:
        return FittedOrbits.empty(), FittedOrbitMembers.empty()
    states = np.asarray(output["states"], dtype=np.float64)
    epochs = Timestamp.from_mjd(
        np.asarray(output["epoch_mjd"], dtype=np.float64), scale="utc"
    )
    orbits = FittedOrbits.from_kwargs(
        orbit_id=orbit_ids,
        object_id=[None] * len(orbit_ids),
        coordinates=CartesianCoordinates.from_kwargs(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            vx=states[:, 3],
            vy=states[:, 4],
            vz=states[:, 5],
            time=epochs,
            origin=Origin.from_kwargs(code=["SUN"] * len(orbit_ids)),
            frame="ecliptic",
        ),
        arc_length=output["arc_length"],
        num_obs=output["num_obs"],
        chi2=output["chi2"],
        reduced_chi2=output["reduced_chi2"],
    )
    residuals = Residuals.from_kwargs(
        values=np.asarray(output["residual_values"], dtype=np.float64).tolist(),
        chi2=output["residual_chi2"],
        dof=output["residual_dof"],
        probability=output["residual_probability"],
    )
    members = FittedOrbitMembers.from_kwargs(
        orbit_id=output["member_orbit_ids"],
        obs_id=output["member_obs_ids"],
        residuals=residuals,
        solution=output["solution"],
        outlier=output["outlier"],
    )
    return orbits, members


def _native_initial_orbit_determination(
    prop: Propagator,
    observations: OrbitDeterminationObservations,
    linkage_ids: list[str],
    member_linkage_ids: list[str],
    member_obs_ids: list[str],
    *,
    min_obs: int,
    min_arc_length: float,
    contamination_percentage: float,
    rchi2_threshold: float,
    observation_selection_method: str,
    light_time: bool,
    chunk_size: int,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    output = prop.initial_orbit_determination(
        observations,
        linkage_ids,
        member_linkage_ids,
        member_obs_ids,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        contamination_percentage=contamination_percentage,
        rchi2_threshold=rchi2_threshold,
        observation_selection_method=observation_selection_method,
        light_time=light_time,
        chunk_size=chunk_size,
        mu=MU,
        speed_of_light=C,
    )
    return _tables_from_native_iod(output)


def _iod_worker(
    linkage_ids: npt.NDArray[np.str_],
    observations: OrbitDeterminationObservations,
    linkage_members: FittedOrbitMembers,
    propagator: Type[Propagator],
    min_obs: int = IOD_DEFAULTS["min_obs"],
    min_arc_length: float = IOD_DEFAULTS["min_arc_length"],
    contamination_percentage: float = IOD_DEFAULTS["contamination_percentage"],
    rchi2_threshold: float = IOD_DEFAULTS["rchi2_threshold"],
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds", "spans"
    ] = IOD_DEFAULTS["observation_selection_method"],
    linkage_id_col: str = "cluster_id",
    iterate: bool = False,
    light_time: bool = IOD_DEFAULTS["light_time"],
    propagator_kwargs: dict = {},
    velocity_method: Literal["gauss", "gibbs", "herrick+gibbs", "auto"] = IOD_DEFAULTS[
        "velocity_method"
    ],
    acceptance: Literal["corrected", "prefit"] = IOD_DEFAULTS["acceptance"],
    span_settings: Optional[dict] = None,
    correction_settings: Optional[dict] = None,
    fit_settings: Optional[dict] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """`run_iod` on each linkage in ``linkage_ids`` (one propagator
    instance for the chunk); each accepted orbit takes its linkage id as
    ``orbit_id``. ``iterate`` is accepted for compatibility and ignored."""
    del iterate
    propagator_instance = propagator(**propagator_kwargs)
    # Arrow arrays and chunks yield scalars; the lookups below key on the
    # plain values.
    linkage_ids = [lid.as_py() if hasattr(lid, "as_py") else lid for lid in linkage_ids]

    # Pre-compute linkage_id -> linkage_member rows and obs_id -> observation row
    # so each per-linkage lookup is O(1) rather than O(N_members)/O(N_observations).
    lm_linkage_ids = linkage_members.column(linkage_id_col).to_numpy(
        zero_copy_only=False
    )
    lm_obs_ids = linkage_members.obs_id.to_numpy(zero_copy_only=False)
    lm_rows_by_linkage: dict = {}
    for i, lid in enumerate(lm_linkage_ids):
        lm_rows_by_linkage.setdefault(lid, []).append(i)

    observations_id_arr = observations.id.to_numpy(zero_copy_only=False)
    obs_idx_by_id = {oid: i for i, oid in enumerate(observations_id_arr)}

    iod_orbits_list: list[FittedOrbits] = []
    iod_orbit_members_list: list[FittedOrbitMembers] = []
    for linkage_id in linkage_ids:
        time_start = time.time()
        logger.debug(f"Finding initial orbit for linkage {linkage_id}...")

        lm_rows = lm_rows_by_linkage.get(linkage_id, [])
        obs_row_idx = np.fromiter(
            (obs_idx_by_id[o] for o in lm_obs_ids[lm_rows] if o in obs_idx_by_id),
            dtype=np.int64,
        )
        observations_linkage = observations.take(obs_row_idx)

        # Sort observations by time
        observations_linkage = observations_linkage.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        iod_orbit, iod_orbit_orbit_members, _report = run_iod(
            observations_linkage,
            propagator_instance,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            rchi2_threshold=rchi2_threshold,
            contamination_percentage=contamination_percentage,
            observation_selection_method=observation_selection_method,
            light_time=light_time,
            velocity_method=velocity_method,
            acceptance=acceptance,
            span_settings=span_settings,
            correction_settings=correction_settings,
            fit_settings=fit_settings,
        )
        if len(iod_orbit) > 0:
            iod_orbit = iod_orbit.set_column(
                "orbit_id", pa.array([linkage_id], type=pa.large_string())
            )
            iod_orbit_orbit_members = iod_orbit_orbit_members.set_column(
                "orbit_id",
                pa.array(
                    [linkage_id] * len(iod_orbit_orbit_members), type=pa.large_string()
                ),
            )

        time_end = time.time()
        duration = time_end - time_start
        logger.debug(f"IOD for linkage {linkage_id} completed in {duration:.3f}s.")

        iod_orbits_list.append(iod_orbit)
        iod_orbit_members_list.append(iod_orbit_orbit_members)

    iod_orbits = (
        qv.concatenate(iod_orbits_list) if iod_orbits_list else FittedOrbits.empty()
    )
    iod_orbit_members = (
        qv.concatenate(iod_orbit_members_list)
        if iod_orbit_members_list
        else FittedOrbitMembers.empty()
    )
    return iod_orbits, iod_orbit_members


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
    #: Two-body score of the preliminary orbit against every observation.
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
    attributable's epoch, range, range rate, chi2 and weight
    ``exp(-delta_chi2 / 2)`` (normalized).
    """

    attributable: dict
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


def _ranging_from_output(ranging: Optional[dict]) -> Optional[RangingSample]:
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
class IodReport:
    """
    What `run_iod` searched: every scored candidate (`IodCandidate`), which
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


def report_from_iod_output(output: dict) -> IodReport:
    """The `IodReport` of a Rust IOD dict (`run_iod_ipc` or the ``iod`` entry
    of a fused ``full_od`` / ``run_od`` output)."""
    candidates = tuple(
        IodCandidate(
            state=tuple(float(v) for v in candidate["state"]),
            epoch_days=int(candidate["epoch_days"]),
            epoch_nanos=int(candidate["epoch_nanos"]),
            epoch_scale=candidate["epoch_scale"],
            window=(int(candidate["window"][0]), int(candidate["window"][1])),
            triplet=tuple(int(row) for row in candidate["triplet"]),
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
                else tuple(float(v) for v in candidate["corrected_state"])
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
        ambiguous_with=tuple(int(i) for i in output.get("ambiguous_with") or ()),
        ranging=_ranging_from_output(output.get("ranging")),
    )


def _empty_report(reason: str) -> IodReport:
    return IodReport((), None, 0, 0, 0, 0, reason)


def run_iod(
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    *,
    min_obs: int = IOD_DEFAULTS["min_obs"],
    min_arc_length: float = IOD_DEFAULTS["min_arc_length"],
    contamination_percentage: float = IOD_DEFAULTS["contamination_percentage"],
    rchi2_threshold: float = IOD_DEFAULTS["rchi2_threshold"],
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds", "spans"
    ] = IOD_DEFAULTS["observation_selection_method"],
    light_time: bool = IOD_DEFAULTS["light_time"],
    max_triplets: Optional[int] = IOD_DEFAULTS["max_triplets"],
    velocity_method: Literal["gauss", "gibbs", "herrick+gibbs", "auto"] = IOD_DEFAULTS[
        "velocity_method"
    ],
    span_settings: Optional[dict] = None,
    acceptance: Literal["corrected", "prefit"] = IOD_DEFAULTS["acceptance"],
    correction_settings: Optional[dict] = None,
    ranging_settings: Optional[dict] = None,
    fit_settings: Optional[dict] = None,
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers, IodReport]:
    """
    The IOD stage: Gauss preliminary orbits on observation triplets, scored
    on ``propagator`` and accepted in one of two ways. ``acceptance=
    "corrected"`` (default) treats each Gauss orbit as a seed for
    differential corrections: in increasing prefit reduced chi2, up to
    ``max_corrections`` candidates are corrected on the observations their
    triplet spans, then on spans grown fourfold (jumping apparition gaps)
    up to the whole arc, every stage within ``rchi2_threshold``; the
    converged candidate with the lowest whole-arc reduced chi2 is accepted
    and its orbit is that corrected, unrejected fit. ``"prefit"`` is the
    legacy gate: the first raw Gauss orbit within ``rchi2_threshold`` on
    every observation (after the outlier trial).

    This is the native stage behind `iod`, `NativeOrbitFitter.initial_fit`
    and the IOD half of `run_od`. ``propagator`` is explicit: it is the
    backend that scores candidates and may differ from the one that later
    refines the orbit. A Rust-backed propagator exposing the fused
    ``run_iod`` work unit runs the whole search in one crossing; any other
    propagator runs the same Rust search through its own
    ``generate_ephemeris``.

    Parameters
    ----------
    observations : `OrbitDeterminationObservations`
        Observations believed to belong to one object (any row order; the
        search sorts by epoch, station and row and reports in input order).
    propagator : `~adam_core.propagator.Propagator`
        The propagator instance that scores candidates.
    min_obs, min_arc_length, contamination_percentage, rchi2_threshold,
    observation_selection_method, light_time
        As for `iod`.
    max_triplets : int, optional
        Bound on the triplets searched; None keeps the legacy budget of three
        triplets per allowed outlier (``"spans"``: its own default budget).
    velocity_method : {"gauss", "gibbs", "herrick+gibbs", "auto"}
        Velocity of the Gauss candidates. ``"auto"`` uses Herrick-Gibbs when
        consecutive position vectors are within 5 degrees (Vallado 2013) and
        Gibbs otherwise; candidates record the method used.
    span_settings : dict, optional
        Overrides of the ``observation_selection_method="spans"`` search
        (keys ``span_target_days``, ``span_min_days``, ``span_max_days``,
        ``span_dense_window_days``, ``span_edge_window_days``,
        ``span_max_triplets``); missing keys keep the Rust defaults of
        ``SpanSearchConfig``.
    acceptance : {"corrected", "prefit"}
        See above.
    correction_settings : dict, optional
        Overrides of the corrected acceptance (keys ``max_corrections``,
        ``max_converged``, ``correction_growth``, ``min_stage_obs``,
        ``max_stages``, ``stage_loss``, ``prefit_scope``); missing keys keep
        the Rust ``CorrectionConfig`` defaults. ``stage_loss`` is the loss of
        the stages before the whole arc: ``"huber"`` (default) down-weights
        observations a partial orbit predicts badly and gates each stage on
        its robust cost; ``"linear"`` turns that off. ``prefit_scope`` is
        where each raw Gauss orbit is scored to order the corrections:
        ``"local"`` (default), the observations of its first stage, or
        ``"whole_arc"``, every observation (the legacy order).
    ranging_settings : dict, optional
        Overrides of systematic ranging (keys ``ranging_enabled``,
        ``ranging_rho_min_au``, ``ranging_rho_max_au``, ``ranging_n_rho``,
        ``ranging_n_rho_dot``, ``ranging_max_corrections``,
        ``ranging_chunk_size``, ``ranging_max_arc_days``; Rust
        ``RangingConfig`` defaults otherwise). Ranging runs on arcs of at
        most ``ranging_max_arc_days`` (5).
        Under corrected acceptance an arc shorter than ``min_arc_length``
        returns no orbit but the ranging sample (``IodReport.ranging``), and
        when no Gauss candidate converges the best ranging orbits are
        corrected like candidates (``IodCandidate.source == "ranging"``).
    fit_settings : dict, optional
        Solver settings of the correction stages (as `fit_least_squares`'s;
        the whole-arc stage uses the linear loss, the earlier ones
        ``stage_loss``; all skip covariance validation).
    observatory_bias_model : `ObservationUncertaintyModel`, optional
        Applied to the observations before the search.

    Returns
    -------
    orbit : `FittedOrbits` (0 or 1)
        The accepted preliminary orbit at its exact Gauss epoch.
    members : `FittedOrbitMembers` (0 or N)
        Residuals and solution / outlier flags of every observation against
        the accepted orbit, in input order.
    report : `IodReport`
        Every scored candidate, the accepted index, the search counters and
        the reason when no orbit was accepted.
    """
    if observatory_bias_model is not None:
        observations = observatory_bias_model.apply(observations)
    if len(observations) == 0:
        return (
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            _empty_report("no observations"),
        )
    iod_settings = {
        "min_obs": min_obs,
        "min_arc_length": min_arc_length,
        "contamination_percentage": contamination_percentage,
        "rchi2_threshold": rchi2_threshold,
        "observation_selection_method": observation_selection_method,
        "light_time": light_time,
        "mu": MU,
        "speed_of_light": C,
        "max_triplets": max_triplets,
        "velocity_method": velocity_method,
        "acceptance": acceptance,
        **(span_settings or {}),
        **(correction_settings or {}),
        **(ranging_settings or {}),
    }
    orbit_id = uuid.uuid4().hex
    fused = native_work_unit(propagator, "run_iod")
    if fused is not None:
        output = fused(
            observations, iod_settings=iod_settings, fit_settings=fit_settings
        )
    else:
        from adam_core import _rust_native

        from .differential_correction import _native_problem

        _, observed_ipc, observers_ipc = _native_problem(
            _placeholder_orbit(observations), observations
        )
        output = _rust_native.run_iod_ipc(
            propagator,
            observed_ipc,
            observers_ipc,
            iod_settings=iod_settings,
            fit_settings=fit_settings,
        )
    orbit, members = tables_from_iod_output(orbit_id, observations, output)
    return orbit, members, report_from_iod_output(output)


def _iod(
    observations: OrbitDeterminationObservations,
    propagator: Type[Propagator],
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    rchi2_threshold: float = 200,
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds", "spans"
    ] = "combinations",
    iterate: bool = False,
    light_time: bool = True,
    propagator_kwargs: dict = {},
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
    velocity_method: Literal["gauss", "gibbs", "herrick+gibbs", "auto"] = "gibbs",
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Run initial orbit determination on a set of observations believed to belong to a single
    object.

    Parameters
    ----------
    observations : `OrbitDeterminationObservations`
        Observations with at least the following attributes:
            id : Observation IDs [str],
            coordinates : SkyCoord object containing observation time, RA, and Dec
            observers : Observer object containing observatory positions
    min_obs : int, optional
        Minimum number of observations that must remain in the linkage. For example, if min_obs is set to 6 and
        a linkage has 8 observations, at most the two worst observations will be flagged as outliers if their individual
        chi2 values exceed the chi2 threshold.
    contamination_percentage : float, optional
        Maximum percent of observations that can flagged as outliers.
    rchi2_threshold : float, optional
        Maximum reduced chi2 required for an initial orbit to be accepted.
    observation_selection_method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Selects which three observations to use for IOD depending on the method. The avaliable methods are:
            'first+middle+last' : Grab the first, middle and last observations in time.
            'thirds' : Grab the middle observation in the first third, second third, and final third.
            'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
                non-coinciding observation times.
    iterate : bool, optional
        Accepted for compatibility; the Rust decision loop evaluates every
        candidate the same way and ignores it.
    light_time : bool, optional
        Correct preliminary orbit for light travel time.
    propagator : Type[Propagator], optional
        Which propagator to use for ephemeris generation.
    propagator_kwargs : dict, optional
        Settings and additional parameters to pass to selected
        propagator.
    observatory_bias_model : `~adam_core.orbit_determination.ObservationUncertaintyModel`, optional
        Observation uncertainty model applied to the observations before fitting
        (e.g. inflating per-station sigmas from an observatory bias table). Default
        None leaves the observations unchanged. The model is applied once at this
        entry point and is not forwarded to nested calls.

    Returns
    -------
    iod_orbits : `FittedOrbits`
        Dataframe with orbits found in linkages.
            "orbit_id" : Orbit ID, a uuid [str],
            "epoch" : Epoch at which orbit is defined in MJD TDB [float],
            "x" : Orbit's ecliptic J2000 x-position in au [float],
            "y" : Orbit's ecliptic J2000 y-position in au [float],
            "z" : Orbit's ecliptic J2000 z-position in au [float],
            "vx" : Orbit's ecliptic J2000 x-velocity in au per day [float],
            "vy" : Orbit's ecliptic J2000 y-velocity in au per day [float],
            "vz" : Orbit's ecliptic J2000 z-velocity in au per day [float],
            "arc_length" : Arc length in days [float],
            "num_obs" : Number of observations that were within the chi2 threshold
                of the orbit.
            "chi2" : Total chi2 of the orbit calculated using the predicted location of the orbit
                on the sky compared to the consituent observations.

    iod_orbit_members : `FittedOrbitMembers`
        Dataframe of orbit members with the following columns:
            "orbit_id" : Orbit ID, a uuid [str],
            "obs_id" : Observation IDs [str], one ID per row.
            "residual_ra_arcsec" : Residual (observed - expected) equatorial J2000 Right Ascension in arcseconds [float]
            "residual_dec_arcsec" : Residual (observed - expected) equatorial J2000 Declination in arcseconds [float]
            "chi2" : Observation's chi2 [float]
            "gauss_sol" : Flag to indicate which observations were used to calculate the Gauss soluton [int]
            "outlier" : Flag to indicate which observations are potential outliers (their chi2 is higher than
                the chi2 threshold) [float]
    """
    # Apply the observatory bias model (if any) before anything else, as
    # every entry point does; then the accepted orbit of the `run_iod` stage
    # on a fresh propagator instance (the candidate report is available from
    # `run_iod` itself).
    if observatory_bias_model is not None:
        observations = observatory_bias_model.apply(observations)
    orbit, members, _report = run_iod(
        observations,
        propagator(**propagator_kwargs),
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        contamination_percentage=contamination_percentage,
        rchi2_threshold=rchi2_threshold,
        observation_selection_method=observation_selection_method,
        light_time=light_time,
        velocity_method=velocity_method,
        # `iod` keeps the legacy acceptance; `run_iod` defaults to corrected.
        acceptance="prefit",
    )
    return orbit, members


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


def tables_from_iod_output(
    orbit_id: str,
    observations: OrbitDeterminationObservations,
    output: dict,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Wrap the dict of a Rust IOD decision (`iod_fit_ipc`, or the ``iod`` entry
    of a fused ``full_od`` output) into `FittedOrbits` / `FittedOrbitMembers`
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
    return with_iod_provenance(orbit, output.get("ambiguous_with")), members


def initial_orbit_determination(
    observations: OrbitDeterminationObservations,
    linkage_members: FittedOrbitMembers,
    propagator: Type[Propagator],
    min_obs: int = IOD_DEFAULTS["min_obs"],
    min_arc_length: float = IOD_DEFAULTS["min_arc_length"],
    contamination_percentage: float = IOD_DEFAULTS["contamination_percentage"],
    rchi2_threshold: float = IOD_DEFAULTS["rchi2_threshold"],
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds", "spans"
    ] = IOD_DEFAULTS["observation_selection_method"],
    iterate: bool = False,
    light_time: bool = IOD_DEFAULTS["light_time"],
    linkage_id_col: str = "cluster_id",
    propagator_kwargs: dict = {},
    chunk_size: int = 1,
    max_processes: Optional[int] = 1,
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
    velocity_method: Literal["gauss", "gibbs", "herrick+gibbs", "auto"] = IOD_DEFAULTS[
        "velocity_method"
    ],
    acceptance: Literal["corrected", "prefit"] = IOD_DEFAULTS["acceptance"],
    span_settings: Optional[dict] = None,
    correction_settings: Optional[dict] = None,
    fit_settings: Optional[dict] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Run initial orbit determination on linkages found in observations.

    Each linkage runs the `run_iod` stage with the same policy and defaults
    (``IOD_DEFAULTS``: span-bounded triplets, corrected acceptance); see
    `run_iod` for the settings. The legacy linkage behavior (first raw Gauss
    orbit under the threshold) is ``acceptance="prefit",
    observation_selection_method="combinations", rchi2_threshold=1000``; with
    ``acceptance="prefit"`` a propagator's compiled
    ``initial_orbit_determination`` work unit runs every linkage in one
    crossing.

    If `observatory_bias_model` is provided it is applied to the observations
    once at this entry point (e.g. inflating per-station sigmas from an
    observatory bias table) before any orbits are fit; the default None
    leaves the observations unchanged.
    """
    time_start = time.perf_counter()
    logger.info("Running initial orbit determination...")

    if observatory_bias_model is not None:
        observations = observatory_bias_model.apply(observations)

    # The supported native provider owns the entire multi-linkage workflow in
    # one crossing; chunk_size is consumed by Rust and max_processes is kept
    # only for signature compatibility (native execution replaces Ray).
    native_prop = propagator(**propagator_kwargs)
    # The compiled batch work unit implements the legacy prefit policy only.
    if acceptance == "prefit" and native_work_unit(
        native_prop, "initial_orbit_determination"
    ):
        if len(observations) == 0 or len(linkage_members) == 0:
            return FittedOrbits.empty(), FittedOrbitMembers.empty()
        linkage_ids = linkage_members.column(linkage_id_col).unique().to_pylist()
        return _native_initial_orbit_determination(
            native_prop,
            observations,
            linkage_ids,
            linkage_members.column(linkage_id_col).to_pylist(),
            linkage_members.obs_id.to_pylist(),
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            rchi2_threshold=rchi2_threshold,
            observation_selection_method=observation_selection_method,
            light_time=light_time,
            chunk_size=chunk_size,
        )

    iod_orbits_chunks: list[FittedOrbits] = []
    iod_orbit_members_chunks: list[FittedOrbitMembers] = []
    if len(observations) > 0 and len(linkage_members) > 0:
        # Extract linkage IDs
        linkage_ids = linkage_members.column(linkage_id_col).unique()

        # `max_processes` remains accepted for signature compatibility.
        # Native providers returned above; non-native providers use the
        # deterministic serial fallback instead of Python/Ray distribution.
        del max_processes
        for linkage_id_chunk in _iterate_chunks(linkage_ids, chunk_size):
            iod_orbits_chunk, iod_orbit_members_chunk = _iod_worker(
                linkage_id_chunk,
                observations,
                linkage_members,
                min_obs=min_obs,
                min_arc_length=min_arc_length,
                contamination_percentage=contamination_percentage,
                rchi2_threshold=rchi2_threshold,
                observation_selection_method=observation_selection_method,
                iterate=iterate,
                light_time=light_time,
                linkage_id_col=linkage_id_col,
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
                velocity_method=velocity_method,
                acceptance=acceptance,
                span_settings=span_settings,
                correction_settings=correction_settings,
                fit_settings=fit_settings,
            )
            iod_orbits_chunks.append(iod_orbits_chunk)
            iod_orbit_members_chunks.append(iod_orbit_members_chunk)

        iod_orbits = (
            qv.concatenate(iod_orbits_chunks)
            if iod_orbits_chunks
            else FittedOrbits.empty()
        )
        iod_orbit_members = (
            qv.concatenate(iod_orbit_members_chunks)
            if iod_orbit_members_chunks
            else FittedOrbitMembers.empty()
        )

        time_start_drop = time.time()
        logger.info("Removing duplicate initial orbits...")
        num_orbits = len(iod_orbits)
        iod_orbits, iod_orbit_members = drop_duplicate_orbits(
            iod_orbits,
            iod_orbit_members,
            subset=[
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.x",
                "coordinates.y",
                "coordinates.z",
                "coordinates.vx",
                "coordinates.vy",
                "coordinates.vz",
            ],
            keep="first",
        )
        time_end_drop = time.time()
        logger.info(f"Removed {num_orbits - len(iod_orbits)} duplicate clusters.")
        time_end_drop = time.time()
        logger.info(
            f"Inital orbit deduplication completed in {time_end_drop - time_start_drop:.3f} seconds."
        )

        # Sort initial orbits by orbit ID and observation time
        iod_orbits, iod_orbit_members = sort_by_id_and_time(
            iod_orbits, iod_orbit_members, observations, "orbit_id"
        )

    else:
        iod_orbits = FittedOrbits.empty()
        iod_orbit_members = FittedOrbitMembers.empty()

    time_end = time.perf_counter()
    logger.info(f"Found {len(iod_orbits)} initial orbits.")
    logger.info(
        f"Initial orbit determination completed in {time_end - time_start:.3f} seconds."
    )

    return iod_orbits, iod_orbit_members


@deprecated(
    _iod,
    "run_iod(observations, propagator, acceptance='prefit', "
    "observation_selection_method='combinations', velocity_method='gibbs') "
    "for the legacy policy, or run_iod's defaults",
)
def iod(*args, **kwargs):  # pragma: no cover - replaced by the alias
    """Deprecated alias of `_iod`."""


@deprecated(
    _iod_worker,
    "initial_orbit_determination (or run_iod per linkage)",
)
def iod_worker(*args, **kwargs):  # pragma: no cover - replaced by the alias
    """Deprecated alias of `_iod_worker`."""
