"""Thin Python marshalling for Rust-owned canonical batch OD workflows."""

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
        adam_core_version=[value["adam_core_version"]],
        adam_core_build=[value["adam_core_build"]],
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
    """Run canonical batch IOD over explicit association hypotheses.

    ``config`` must be the bound stage projection returned by an explicit
    ``OdConfig.for_iod()``. Group parsing, validation, ordering, transforms,
    IOD, failure classification, identifiers, and provenance are executed by
    Core Rust.
    """

    if not isinstance(config, IodRunConfig):
        raise TypeError("config must be an IodRunConfig")
    if config._workflow_native is None:
        raise ValueError(
            "run_iod requires an IodRunConfig projected from an explicit bound "
            "OdConfig via config.for_iod()"
        )
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
        config._workflow_native,
    )
    return _assemble(native, prepared)


def run_od(
    observations: SourceCatalog,
    associations: Associations,
    propagator: Propagator | None,
    config: OdConfig,
) -> _RESULT:
    """Run canonical batch IOD then refinement on one prepared group batch."""

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
    """Refine explicit fitted-orbit/member groups using their prepared handoff."""

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
    """Evaluate only explicitly declared fitted-orbit/member pairs.

    ``config`` must be the bound stage projection returned by an explicit
    ``OdConfig.for_evaluation()``.
    """

    if not isinstance(orbits, FittedOrbits):
        raise TypeError("orbits must be FittedOrbits")
    if not isinstance(members, FittedOrbitMembers):
        raise TypeError("members must be FittedOrbitMembers")
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    if config._workflow_native is None:
        raise ValueError(
            "evaluate_orbits requires an EvaluationConfig projected from an explicit "
            "bound OdConfig via config.for_evaluation()"
        )
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
        config._workflow_native,
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
