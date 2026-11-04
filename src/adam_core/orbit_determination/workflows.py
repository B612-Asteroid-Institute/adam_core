"""Thin Python marshalling for Rust-owned canonical batch OD workflows.

The four public workflows are ordinary callables exported from this module:

>>> names = ("run_iod", "run_od", "refine", "evaluate_orbits")
>>> all(callable(globals()[name]) for name in names)
True
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
import quivr as qv

from adam_core import _rust_native
from adam_core._rust.arrow import pa
from adam_core.coordinates.residuals import Residuals
from adam_core.observations import Associations, SourceCatalog
from adam_core.observations.arrow_bridge import observations_to_ipc
from adam_core.orbits.arrow_bridge import (
    coordinates_to_ipc,
    observers_to_ipc,
    orbits_to_ipc,
)
from adam_core.propagator.propagator import Propagator

from ._native_dispatch import NativeProvider, native_propagator
from .algorithms.fitting import _fitted_tables_from_native_output
from .algorithms.iod import _tables_from_iod_output
from .config import (
    EvaluationConfig,
    IodRunConfig,
    OdConfig,
    RefinementRunConfig,
    _canonical_json,
)
from .evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry
from .fitted_orbits import (
    FittedOrbitMembers,
    FittedOrbits,
    ObservationAstrometry,
    OdProvenance,
    OrbitDeterminationFailures,
)

_RESULT = tuple[FittedOrbits, FittedOrbitMembers, OrbitDeterminationFailures]
_TableT = TypeVar("_TableT", bound=qv.Table)


def _deferred_native_propagator(
    propagator: Propagator | None,
) -> Callable[[], NativeProvider | None] | None:
    """Resolve and bind a provider only after Rust static preflight.

    The outer callable is retained by PyO3 without invocation. Empty and
    malformed batches return before Rust asks it for the identity descriptor;
    a valid non-empty batch resolves it exactly once immediately before native
    predictor construction.
    """

    if propagator is None:
        return None
    return lambda: native_propagator(propagator)


def _od_observations(source: SourceCatalog) -> OrbitDeterminationObservations:
    """Representation-only SourceCatalog -> OD Arrow-table marshalling."""

    return OrbitDeterminationObservations.from_kwargs(
        id=source.id,
        coordinates=source.coordinates(),
        observers=source.observers(),
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=source.mag,
            rmsmag=source.mag_sigma,
            band=source.filter,
        ),
        astcat=source.astcat,
    )


def _source_inputs(
    source: SourceCatalog,
) -> tuple[OrbitDeterminationObservations, bytes, bytes, bytes]:
    if not isinstance(source, SourceCatalog):
        raise TypeError("observations must be a SourceCatalog")
    # Arrow chunking is representation-only. Normalize it before constructing
    # nested coordinate/time arrays so scientific fingerprints depend only on
    # rows and values, never caller chunk boundaries.
    if source.fragmented():
        source = SourceCatalog.from_pyarrow(source.table.combine_chunks())
    observations = _od_observations(source)
    return (
        observations,
        observations_to_ipc(source),
        coordinates_to_ipc(observations.coordinates, "spherical"),
        observers_to_ipc(observations.observers),
    )


def _native_batch_inputs(
    source: SourceCatalog, associations: Associations
) -> tuple[OrbitDeterminationObservations, bytes, bytes, bytes, bytes]:
    if not isinstance(associations, Associations):
        raise TypeError("associations must be Associations")
    observations, source_ipc, observed_ipc, observers_ipc = _source_inputs(source)
    return (
        observations,
        source_ipc,
        observations_to_ipc(associations),
        observed_ipc,
        observers_ipc,
    )


def _snapshot(value: dict[str, Any]) -> ObservationAstrometry:
    return ObservationAstrometry.from_kwargs(
        lon=value["lon"],
        lat=value["lat"],
        sigma_lon=value["sigma_lon"],
        sigma_lat=value["sigma_lat"],
        variance_lon=value["variance_lon"],
        variance_lat=value["variance_lat"],
        cov_lonlat=value["cov_lonlat"],
    )


def _provenance(value: dict[str, Any]) -> OdProvenance:
    return OdProvenance.from_kwargs(
        input_fingerprint=[value["input_fingerprint"]],
        state_fingerprint=[value["state_fingerprint"]],
        provenance_fingerprint=[value["provenance_fingerprint"]],
        route=[value["route"]],
        fitter=[value["fitter"]],
        iod_backend=[value["iod_backend"]],
        refinement_backend=[value["refinement_backend"]],
        evaluation_backend=[value["evaluation_backend"]],
        iod_provider=[value["iod_provider"]],
        refinement_provider=[value["refinement_provider"]],
        evaluation_provider=[value["evaluation_provider"]],
        iod_diagnostics=[value["iod_diagnostics"]],
        refinement_diagnostics=[value["refinement_diagnostics"]],
        evaluation_diagnostics=[value["evaluation_diagnostics"]],
        settings=[value["settings"]],
        configuration=[value["configuration"]],
        iod_ambiguous=[value["iod_ambiguous"]],
        iod_alternatives=[value["iod_alternatives"]],
        num_rejected=[value["num_rejected"]],
        rejected_fraction=[value["rejected_fraction"]],
    )


def _failure(value: dict[str, Any]) -> OrbitDeterminationFailures:
    return OrbitDeterminationFailures.from_kwargs(
        group_id=[value["group_id"]],
        orbit_id=[value["orbit_id"]],
        input_orbit_id=[value["input_orbit_id"]],
        stage=[value["stage"]],
        code=[value["code"]],
        message=[value["message"]],
        recoverable=[value["recoverable"]],
        num_obs=[value["num_obs"]],
        arc_length=[value["arc_length"]],
        triplets_tried=[value["triplets_tried"]],
        candidates_evaluated=[value["candidates_evaluated"]],
        candidates_failed=[value["candidates_failed"]],
        corrections_run=[value["corrections_run"]],
        passes=[value["passes"]],
        n_rejected=[value["n_rejected"]],
        n_recovered=[value["n_recovered"]],
        status_code=[value["status_code"]],
        best_state_retained=[value["best_state_retained"]],
    )


def _concat(tables: Sequence[_TableT], cls: type[_TableT]) -> _TableT:
    if not tables:
        return cls.empty()
    if len(tables) == 1:
        return tables[0]
    concatenated = qv.concatenate(list(tables))
    if not isinstance(concatenated, cls):
        raise TypeError(
            f"concatenating {cls.__name__} produced {type(concatenated).__name__}"
        )
    return concatenated


def _decorate_group(
    orbit: FittedOrbits,
    members: FittedOrbitMembers,
    group: dict[str, Any],
    observations: OrbitDeterminationObservations,
    *,
    assign_group_object_id: bool = True,
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    if len(orbit) != 1:
        raise RuntimeError("native batch group emitted a non-singleton orbit")
    if len(members) != len(group["row_indices"]):
        raise RuntimeError("native batch group/member cardinality mismatch")
    if assign_group_object_id:
        orbit = orbit.set_column(
            "object_id", pa.array([group["group_id"]], type=pa.large_string())
        )
    orbit = orbit.set_column("provenance", _provenance(group["provenance"]))
    members = (
        members.set_column(
            "original_astrometry", _snapshot(group["original_astrometry"])
        )
        .set_column("used_astrometry", _snapshot(group["used_astrometry"]))
        .set_column("astcat", observations.astcat)
        .set_column(
            "input_fingerprint",
            pa.array(group["member_input_fingerprints"], type=pa.large_string()),
        )
        .set_column(
            "membership_fingerprint",
            pa.array(group["member_membership_fingerprints"], type=pa.large_string()),
        )
    )
    return orbit, members


def _assemble(
    native: dict[str, Any],
    observations: OrbitDeterminationObservations,
) -> _RESULT:
    orbit_tables: list[FittedOrbits] = []
    member_tables: list[FittedOrbitMembers] = []
    failures: list[OrbitDeterminationFailures] = []
    for group in native["groups"]:
        rows = pa.array(group["row_indices"], type=pa.int64())
        grouped = observations.take(rows)
        iod_output = group["iod"]
        iod_orbit = FittedOrbits.empty()
        iod_members = FittedOrbitMembers.empty()
        if iod_output is not None and iod_output["found"]:
            iod_orbit, iod_members = _tables_from_iod_output(
                group["orbit_id"], grouped, iod_output
            )

        if group["output_kind"] == "refined":
            if len(iod_orbit) != 1:
                raise RuntimeError("native refinement has no accepted IOD seed")
            output = group["refined"]
            orbit, members = _fitted_tables_from_native_output(
                iod_orbit.to_orbits(),
                grouped,
                output,
                np.asarray(output["weights"], dtype=np.float64),
            )
            orbit, members = _decorate_group(orbit, members, group, grouped)
            orbit_tables.append(orbit)
            member_tables.append(members)
        elif group["output_kind"] == "iod":
            if len(iod_orbit) != 1:
                raise RuntimeError("native IOD disposition has no accepted state")
            orbit, members = _decorate_group(iod_orbit, iod_members, group, grouped)
            orbit_tables.append(orbit)
            member_tables.append(members)

        if group["failure"] is not None:
            failures.append(_failure(group["failure"]))

    return (
        _concat(orbit_tables, FittedOrbits),
        _concat(member_tables, FittedOrbitMembers),
        _concat(failures, OrbitDeterminationFailures),
    )


def run_iod(
    observations: SourceCatalog,
    associations: Associations,
    propagator: Propagator | None,
    config: IodRunConfig,
) -> _RESULT:
    """Run canonical batch initial orbit determination (IOD).

    Rust validates the complete batch, applies the bound observation pipeline
    once, and executes deterministic IOD per explicit association
    hypothesis. A one-object request uses this same batch route.

    Parameters
    ----------
    observations : SourceCatalog (N)
        Observation rows. The required non-null columns are ``id`` (unique and
        non-blank), ``time``, ``ra``, ``dec``, ``observatory_code``, and
        ``catalog_id``. Astrometric uncertainty columns must be finite and
        positive after ``config.observation_pipeline`` runs. Every present
        covariance has dimension six before any transform is interpreted.
    associations : Associations (M)
        Explicit memberships with non-null ``detection_id`` and ``object_id``.
        Each detection ID must reference ``observations.id``. A detection may
        occur in several hypotheses, but a ``(object_id, detection_id)`` pair
        may occur only once.
    propagator : Propagator or None
        Propagation-only provider. ``None`` explicitly selects Core's built-in
        two-body provider. Any other value must expose the negotiated native
        propagation capsule; it never supplies IOD, fitting, or rejection.
    config : IodRunConfig
        Self-contained IOD configuration containing the complete
        ``IodStageConfig``, ordered bound ``ObservationPipeline``, validated
        resources, and optional immutable profile identity.

    Returns
    -------
    fitted_orbits : FittedOrbits (G_success)
        One accepted state per successful group, in first-appearance group
        order, with complete route/configuration/provider provenance.
    fitted_members : FittedOrbitMembers (K_success)
        Successful-group memberships in source-table order, including original
        and used astrometry plus input and membership fingerprints.
    failures : OrbitDeterminationFailures (G_failure)
        Typed, deterministic per-group scientific failures. No placeholder
        orbit is emitted when an IOD group has no accepted state.

    Raises
    ------
    TypeError
        If a Python table, provider, or ``config`` has the wrong public type.
    ValueError
        If static batch, identity, membership, covariance, or provenance
        validation fails. These errors abort atomically before provider work.
    RuntimeError
        If provider acquisition or execution reports a backend error, panic,
        unsupported capability, malformed response, or protocol violation.
        These errors abort atomically and return no partial products.

    Notes
    -----
    Static configuration/resource/ID/membership checks complete before provider
    acquisition. Empty input returns three typed empty tables and does not
    acquire the provider. Groups follow first appearance in ``associations``;
    members follow source order; overlaps are preserved. Safely isolated
    no-solution outcomes are rows in ``failures`` and do not stop later groups.
    ``orbit_id`` is a mutable relational label; immutable association identity
    is carried by the orbit ``object_id``/native group identity and member
    fingerprints.

    Examples
    --------
    One explicit group with a deliberate no-transform pipeline and the built-in
    provider (represented by ``None``):

    .. code-block:: python

       from adam_core.observations import Associations
       from adam_core.orbit_determination import (
           IodRunConfig,
           IodStageConfig,
           ObservationPipeline,
           ResourceBindings,
           run_iod,
       )

       one_group = Associations.from_kwargs(
           detection_id=source_catalog.id,
           object_id=["hypothesis-1"] * len(source_catalog),
       )
       iod_config = IodRunConfig(
           ObservationPipeline.as_supplied(),
           IodStageConfig.core_v1(),
           ResourceBindings({}),
       )
       fitted_orbits, fitted_members, failures = run_iod(
           source_catalog,
           one_group,
           None,
           iod_config,
       )

    See Also
    --------
    run_od : IOD plus refinement with one preparation.
    refine : Refine the persisted prepared handoff.
    evaluate_orbits : Evaluate explicit orbit/member groups.
    """

    if not isinstance(config, IodRunConfig):
        raise TypeError("config must be an IodRunConfig")
    prepared, source_ipc, associations_ipc, observed_ipc, observers_ipc = (
        _native_batch_inputs(observations, associations)
    )
    provider = _deferred_native_propagator(propagator)
    native = _rust_native.run_iod_batch_ipc(
        provider,
        source_ipc,
        associations_ipc,
        observed_ipc,
        observers_ipc,
        config._native,
    )
    return _assemble(native, prepared)


def run_od(
    observations: SourceCatalog,
    associations: Associations,
    propagator: Propagator | None,
    config: OdConfig,
) -> _RESULT:
    """Run canonical batch IOD followed by refinement.

    This is the complete OD workflow. Rust plans each association group,
    validates and applies the bound pipeline once, runs IOD, and refines each
    accepted seed on the exact same prepared rows.

    Parameters
    ----------
    observations : SourceCatalog (N)
        Observation rows with unique non-blank ``id`` plus required ``time``,
        ``ra``, ``dec``, ``observatory_code``, and ``catalog_id`` columns.
        Uncertainty must be valid after the configured pipeline.
    associations : Associations (M)
        Non-null ``detection_id``/``object_id`` membership pairs. References
        must be complete and unambiguous; duplicate pairs and dangling IDs are
        invocation-fatal. Overlapping hypotheses are allowed.
    propagator : Propagator or None
        Explicit propagation-only provider. ``None`` selects the built-in
        two-body provider; another provider must expose the native capsule.
    config : OdConfig
        Complete bound configuration: ordered pipeline and resources,
        ``IodStageConfig``, ``RefinementRunConfig``, and optional validated
        profile identity. Profiles must be bound before this call.

    Returns
    -------
    fitted_orbits : FittedOrbits (G_output)
        Accepted or retained best states in deterministic group order. Each row
        includes success/status, exact state identity, and complete provenance.
    fitted_members : FittedOrbitMembers (K_output)
        Member residual/selection rows in source order with lossless original
        and used astrometry and Rust-owned fingerprints.
    failures : OrbitDeterminationFailures (G_failure)
        Typed isolated IOD/refinement failures in group order. A refinement
        failure emits a state only when Rust retained an actual best state.

    Raises
    ------
    TypeError
        If an argument does not have the documented public type.
    ValueError
        If static batch, identity, membership, covariance, or provenance
        validation fails. No partial tuple is returned.
    RuntimeError
        If provider acquisition or execution reports a backend error, panic,
        unsupported capability, malformed response, or protocol violation.
        No partial tuple is returned.

    Notes
    -----
    Empty input returns three empty typed tables without provider acquisition.
    Group order is first association appearance and member order is source
    order. Scientific failures continue per group; static and invocation-fatal
    errors abort the whole call.

    Conceptually, ``run_od(config)`` is ``run_iod(config.for_iod())`` followed
    by ``refine(config.for_refinement())`` with prior IOD failures carried
    forward. The fused route performs the observation transforms only once;
    the persisted handoff consumed by standalone ``refine`` prevents a second
    transformation. Matching direct and composed routes have equal scientific
    products and canonical OD provenance. Emitted ``orbit_id`` values are
    mutable relational labels; immutable association identity remains in the
    orbit object/group identity and member fingerprints.

    Examples
    --------
    Bind the public profile explicitly, then run a multi-hypothesis batch:

    .. code-block:: python

       from adam_core.orbit_determination import (
           CORE_OD_V1,
           core_od_v1_resource_bindings,
           run_od,
       )

       config = CORE_OD_V1.bind(core_od_v1_resource_bindings())
       fitted_orbits, fitted_members, failures = run_od(
           source_catalog,
           associations,
           None,
           config,
       )

    The tuple always has exactly these three table types, including empty
    and all-failed batches.

    See Also
    --------
    run_iod : IOD-only stage using ``IodRunConfig``.
    refine : Persisted-handoff refinement.
    evaluate_orbits : Independent residual evaluation.
    """

    if not isinstance(config, OdConfig):
        raise TypeError("config must be an OdConfig")
    prepared, source_ipc, associations_ipc, observed_ipc, observers_ipc = (
        _native_batch_inputs(observations, associations)
    )
    provider = _deferred_native_propagator(propagator)
    native = _rust_native.run_od_batch_ipc(
        provider,
        source_ipc,
        associations_ipc,
        observed_ipc,
        observers_ipc,
        config._native,
    )
    return _assemble(native, prepared)


def _astrometry_columns(
    snapshot: ObservationAstrometry,
) -> tuple[list[float | None], ...]:
    return (
        snapshot.lon.to_pylist(),
        snapshot.lat.to_pylist(),
        snapshot.sigma_lon.to_pylist(),
        snapshot.sigma_lat.to_pylist(),
        snapshot.variance_lon.to_pylist(),
        snapshot.variance_lat.to_pylist(),
        snapshot.cov_lonlat.to_pylist(),
    )


def _orbit_provenance(orbits: FittedOrbits) -> list[str | None]:
    """Marshal every typed provenance field without classifying the route."""

    provenance = orbits.provenance
    columns = {
        name: getattr(provenance, name).to_pylist()
        for name in OdProvenance.schema.names
    }
    encoded: list[str | None] = []
    for row in range(len(orbits)):
        value = {name: column[row] for name, column in columns.items()}
        encoded.append(
            None
            if all(item is None for item in value.values())
            else _canonical_json(value)
        )
    return encoded


def refine(
    orbits: FittedOrbits,
    members: FittedOrbitMembers,
    observations: SourceCatalog,
    propagator: Propagator | None,
    config: RefinementRunConfig,
) -> _RESULT:
    """Refine explicit orbit/member groups from a canonical handoff.

    Parameters
    ----------
    orbits : FittedOrbits (O)
        Input states. ``orbit_id`` must be unique and non-blank, coordinates
        must contain a valid six-dimensional state/covariance, and each row
        must carry complete canonical IOD/OD provenance and state identity.
    members : FittedOrbitMembers (K)
        Explicit memberships. Required non-null columns are ``orbit_id`` and
        ``obs_id``; canonical refinement also requires lossless
        ``original_astrometry``/``used_astrometry``, ``input_fingerprint``, and
        ``membership_fingerprint`` for every row. Every input orbit must have
        at least one member.
    observations : SourceCatalog (N)
        Current source rows. IDs, detection metadata, epochs, observer geometry,
        and original astrometry must match the persisted member fingerprints.
    propagator : Propagator or None
        Explicit propagation-only provider; ``None`` selects Core's built-in
        two-body implementation.
    config : RefinementRunConfig
        Complete fit/Jacobian/covariance/rejection declaration. It contains no
        hidden pipeline or IOD configuration.

    Returns
    -------
    fitted_orbits : FittedOrbits (O_output)
        Refined or explicitly retained best states in deterministic group order.
    fitted_members : FittedOrbitMembers (K_output)
        Updated residual/selection/member products in source order.
    failures : OrbitDeterminationFailures (O_failure)
        Typed per-orbit scientific failures. ``input_orbit_id`` identifies the
        seed even when no output state is emitted.

    Raises
    ------
    TypeError
        If any table or ``config`` has the wrong public type.
    ValueError
        If memberships are incomplete or ambiguous, source rows are stale,
        provenance is missing or inconsistent, or covariance is invalid.
        Invocation-fatal validation errors return no partial products.
    RuntimeError
        If provider acquisition or execution reports a backend error, panic,
        unsupported capability, malformed response, or protocol violation.

    Notes
    -----
    Groups follow first appearance of ``orbit_id`` in ``members``; members are
    restored to ``observations`` order. Empty input returns three typed empty
    tables without acquiring the provider. Safely isolated fit failures
    continue. The observation pipeline is *not* rerun: refinement consumes the
    persisted used astrometry created by ``run_iod``/``run_od``. Static
    membership, covariance, source-identity, and provenance checks complete
    before provider acquisition.

    ``orbit_id`` is mutable relational data. Relabeling is supported only by
    editing the orbit row and every matching member row together before this
    call; there is no rename helper. Fingerprints and immutable scientific
    identity do not depend on that presentation label.

    Examples
    --------
    Persist and later refine the accepted IOD products:

    .. code-block:: python

       iod_orbits, iod_members, iod_failures = run_iod(
           source_catalog,
           associations,
           None,
           config.for_iod(),
       )
       refined_orbits, refined_members, refine_failures = refine(
           iod_orbits,
           iod_members,
           source_catalog,
           None,
           config.for_refinement(),
       )

    ``iod_failures`` remains part of the composed batch result; ``refine``
    returns only outcomes for seeds that reached this stage.

    See Also
    --------
    run_od : Fused one-preparation IOD plus refinement.
    evaluate_orbits : Evaluate without changing input states.
    """

    if not isinstance(orbits, FittedOrbits):
        raise TypeError("orbits must be FittedOrbits")
    if not isinstance(members, FittedOrbitMembers):
        raise TypeError("members must be FittedOrbitMembers")
    if not isinstance(config, RefinementRunConfig):
        raise TypeError("config must be a RefinementRunConfig")
    prepared, source_ipc, observed_ipc, observers_ipc = _source_inputs(observations)
    orbit_ipc = orbits_to_ipc(orbits.to_orbits())
    member_orbit_ids = members.orbit_id.to_pylist()
    member_obs_ids = members.obs_id.to_pylist()
    orbit_provenance = _orbit_provenance(orbits)
    original = _astrometry_columns(members.original_astrometry)
    used = _astrometry_columns(members.used_astrometry)
    provider = _deferred_native_propagator(propagator)
    native = _rust_native.refine_batch_ipc(
        provider,
        orbit_ipc,
        source_ipc,
        observed_ipc,
        observers_ipc,
        member_orbit_ids,
        member_obs_ids,
        orbit_provenance,
        members.input_fingerprint.to_pylist(),
        members.membership_fingerprint.to_pylist(),
        *original,
        *used,
        config._native,
    )
    orbit_tables: list[FittedOrbits] = []
    member_tables: list[FittedOrbitMembers] = []
    failures: list[OrbitDeterminationFailures] = []
    for group in native["groups"]:
        seed = orbits.take(pa.array([group["orbit_index"]], type=pa.int64()))
        grouped = prepared.take(pa.array(group["row_indices"], type=pa.int64()))
        if group["output_kind"] == "refined":
            output = group["refined"]
            orbit, output_members = _fitted_tables_from_native_output(
                seed.to_orbits(),
                grouped,
                output,
                np.asarray(output["weights"], dtype=np.float64),
            )
        else:
            orbit = FittedOrbits.empty()
            output_members = FittedOrbitMembers.empty()
        if len(orbit):
            orbit, output_members = _decorate_group(
                orbit,
                output_members,
                group,
                grouped,
                assign_group_object_id=False,
            )
            orbit_tables.append(orbit)
            member_tables.append(output_members)
        if group["failure"] is not None:
            failures.append(_failure(group["failure"]))
    return (
        _concat(orbit_tables, FittedOrbits),
        _concat(member_tables, FittedOrbitMembers),
        _concat(failures, OrbitDeterminationFailures),
    )


def evaluate_orbits(
    orbits: FittedOrbits,
    members: FittedOrbitMembers,
    observations: SourceCatalog,
    propagator: Propagator | None,
    config: EvaluationConfig,
) -> _RESULT:
    """Evaluate explicit orbit/member groups without changing their states.

    Parameters
    ----------
    orbits : FittedOrbits (O)
        Input states with unique non-blank ``orbit_id``, six-dimensional
        covariance, complete state fingerprint, and validated workflow
        provenance. Returned coordinates are the same states.
    members : FittedOrbitMembers (K)
        Explicit pairs selected for evaluation. Required identity columns are
        ``orbit_id`` and ``obs_id``; canonical original/used astrometry and
        input/membership fingerprints must be present. A validated subset of a
        prior group's members is allowed.
    observations : SourceCatalog (N)
        Source rows referenced by ``members``. Their complete persisted identity
        must still match.
    propagator : Propagator or None
        Explicit propagation-only provider; ``None`` selects the built-in
        two-body provider.
    config : EvaluationConfig
        Independent covariance-dimension and prediction-fit configuration.
        Passing an ``OdConfig`` is an error; use ``config.for_evaluation()`` or
        construct an explicit ``EvaluationConfig``.

    Returns
    -------
    fitted_orbits : FittedOrbits (O_output)
        Unchanged states decorated with evaluation status, statistics, provider
        metadata, and evaluation-route provenance.
    fitted_members : FittedOrbitMembers (K_output)
        Residual, solution, and outlier rows in deterministic source order.
    failures : OrbitDeterminationFailures (O_failure)
        Typed per-orbit scientific evaluation failures.

    Raises
    ------
    TypeError
        If a table or ``config`` has the wrong public type.
    ValueError
        If covariance dimension, memberships, provenance, or source identity
        is invalid.
    RuntimeError
        If provider acquisition or execution reports a backend error, panic,
        unsupported capability, malformed response, or protocol violation.

    Notes
    -----
    Static checks precede provider acquisition. Empty input returns three typed
    empty tables. Groups follow first member appearance and rows follow source
    order. Evaluation does not refit or relabel an orbit, does not reapply an
    observation pipeline, and records only evaluation-stage provider/config
    identity rather than copying prior stage backend fields. ``orbit_id``
    remains a mutable relational label, but evaluation itself never changes it;
    callers must edit the orbit and matching member foreign keys together.

    Examples
    --------
    Evaluate the complete accepted handoff with an explicit stage projection:

    .. code-block:: python

       evaluated_orbits, evaluated_members, failures = evaluate_orbits(
           fitted_orbits,
           fitted_members,
           source_catalog,
           None,
           config.for_evaluation(),
       )

    To evaluate a subset, pass only member rows for the chosen orbit/observation
    pairs while retaining their canonical fingerprint columns.

    See Also
    --------
    run_iod : Produce canonical IOD states and members.
    run_od : Produce canonical refined states and members.
    refine : Change states through configured refinement.
    """

    if not isinstance(orbits, FittedOrbits):
        raise TypeError("orbits must be FittedOrbits")
    if not isinstance(members, FittedOrbitMembers):
        raise TypeError("members must be FittedOrbitMembers")
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    prepared, source_ipc, observed_ipc, observers_ipc = _source_inputs(observations)
    orbit_ipc = orbits_to_ipc(orbits.to_orbits())
    member_orbit_ids = members.orbit_id.to_pylist()
    member_obs_ids = members.obs_id.to_pylist()
    member_outlier = members.outlier.to_pylist()
    orbit_provenance = _orbit_provenance(orbits)
    member_input_fingerprints = members.input_fingerprint.to_pylist()
    member_membership_fingerprints = members.membership_fingerprint.to_pylist()
    original = _astrometry_columns(members.original_astrometry)
    used = _astrometry_columns(members.used_astrometry)
    provider = _deferred_native_propagator(propagator)
    native = _rust_native.evaluate_orbits_batch_ipc(
        provider,
        orbit_ipc,
        source_ipc,
        observed_ipc,
        observers_ipc,
        member_orbit_ids,
        member_obs_ids,
        member_outlier,
        orbit_provenance,
        member_input_fingerprints,
        member_membership_fingerprints,
        *original,
        *used,
        config._native,
    )
    orbit_tables: list[FittedOrbits] = []
    member_tables: list[FittedOrbitMembers] = []
    failures: list[OrbitDeterminationFailures] = []
    for group in native["groups"]:
        seed = orbits.take(pa.array([group["orbit_index"]], type=pa.int64()))
        grouped = prepared.take(pa.array(group["row_indices"], type=pa.int64()))
        output = group["evaluation"]
        if group["output_kind"] == "evaluation":
            if output is None:
                raise RuntimeError("native evaluation disposition has no output")
            residuals = Residuals.from_kwargs(
                values=np.asarray(output["residual_values"], dtype=np.float64).tolist(),
                chi2=output["residual_chi2"],
                dof=output["residual_dof"],
                probability=output["residual_probability"],
            )
            outlier = np.asarray(output["outlier"], dtype=bool)
            orbit = FittedOrbits.from_kwargs(
                orbit_id=seed.orbit_id,
                object_id=seed.object_id,
                coordinates=seed.coordinates,
                physical_parameters=seed.physical_parameters,
                non_gravitational_parameters=seed.non_gravitational_parameters,
                arc_length=[output["arc_length"]],
                num_obs=[output["num_obs"]],
                chi2=[output["chi2"]],
                reduced_chi2=[output["reduced_chi2"]],
                success=[output["success"]],
                status_code=[output["status_code"]],
            )
            output_members = FittedOrbitMembers.from_kwargs(
                orbit_id=[seed.orbit_id[0].as_py()] * len(grouped),
                obs_id=grouped.id,
                residuals=residuals,
                solution=output["solution"],
                outlier=outlier,
            )
        else:
            orbit = FittedOrbits.empty()
            output_members = FittedOrbitMembers.empty()
        if len(orbit):
            orbit, output_members = _decorate_group(
                orbit,
                output_members,
                group,
                grouped,
                assign_group_object_id=False,
            )
            orbit_tables.append(orbit)
            member_tables.append(output_members)
        if group["failure"] is not None:
            failures.append(_failure(group["failure"]))
    return (
        _concat(orbit_tables, FittedOrbits),
        _concat(member_tables, FittedOrbitMembers),
        _concat(failures, OrbitDeterminationFailures),
    )
