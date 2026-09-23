"""
Fit-time orbit determination orchestration.

`run_od` is the blessed entry point for orbit determination in adam_core. It
takes the ORIGINAL observations together with the observation models to apply,
runs the backend fitter on the transformed ("used") observations, and returns
fitted orbits whose members record BOTH the original and the used astrometry
for every observation.

Design
------
* Observation models (`ObservationUncertaintyModel`: star-catalog debiasing,
  uncertainty inflation, deweighting, ...) are applied here, at fit time, as
  part of the OD step. They are not a separate preprocessing stage producing a
  "modified observations" product: `OrbitDeterminationObservations` remains
  the INPUT type and always holds the originals, and `run_od` returns no
  modified-observation table. The values the fitter actually saw live on
  `FittedOrbitMembers.used_astrometry`, next to ``original_astrometry`` and
  the star catalog ``astcat``, so every fit is self-describing.
* `run_od` is a free function rather than an `OrbitFitter` method so that it
  is backend-agnostic. It drives the backend through `OrbitFitter.full_od`,
  which the built-in `NativeOrbitFitter` inherits (Gauss IOD followed by
  differential correction) and which external plugins such as adam_fo
  (FindOrb) override. Fitters never see the models: they receive
  pre-transformed observations, keeping the `OrbitFitter` interface flag-free.
* ``model.apply(observations) -> observations`` stays the internal primitive,
  shared with the ``observatory_bias_model`` parameter of the lower-level OD
  functions. Prefer `run_od` when the provenance of the fit matters.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pyarrow as pa
import pyarrow.compute as pc

from ..propagator.propagator import Propagator
from .evaluate import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, ObservationAstrometry
from .observation_uncertainty import CompositeModel, ObservationUncertaintyModel
from .orbit_fitter import OrbitFitter

__all__ = [
    "ObservationModels",
    "apply_observation_models",
    "attach_observation_provenance",
    "run_od",
]

# Models accepted by run_od: a single model, an ordered sequence of models
# (applied left to right, equivalent to a CompositeModel), or None.
ObservationModels = (
    ObservationUncertaintyModel | Sequence[ObservationUncertaintyModel] | None
)


def _as_array(column: pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(column, pa.ChunkedArray):
        return column.combine_chunks()
    return column


def apply_observation_models(
    observations: OrbitDeterminationObservations,
    models: ObservationModels,
) -> OrbitDeterminationObservations:
    """
    Apply observation models to the original observations, producing the
    observations a fitter will use.

    Parameters
    ----------
    observations : `OrbitDeterminationObservations` (N)
        Original observations.
    models : `ObservationUncertaintyModel`, sequence of them, or None
        Model(s) to apply. A sequence is applied in order, left to right
        (equivalent to ``CompositeModel(*models)``). None or an empty
        sequence returns the observations unchanged.

    Returns
    -------
    used : `OrbitDeterminationObservations` (N)
        Transformed observations, with the same ids in the same order as the
        input.

    Raises
    ------
    ValueError
        If the models change the number, identity or order of the
        observations. Models transform astrometry and uncertainties; they may
        not add, drop or reorder observations, since fit provenance is keyed
        by observation id.
    """
    if models is None:
        return observations
    if isinstance(models, ObservationUncertaintyModel):
        model: ObservationUncertaintyModel | None = models
    elif len(models) == 0:
        model = None
    else:
        model = CompositeModel(*models)
    if model is None:
        return observations

    used = model.apply(observations)
    if len(used) != len(observations) or not _as_array(used.id).equals(
        _as_array(observations.id)
    ):
        raise ValueError(
            "Observation models must preserve the observations' ids and order: "
            f"got {len(used)} observations from {len(observations)}"
        )
    return used


def attach_observation_provenance(
    members: FittedOrbitMembers,
    original: OrbitDeterminationObservations,
    used: OrbitDeterminationObservations,
) -> FittedOrbitMembers:
    """
    Embed original and used astrometry (and the star catalog) on fitted orbit
    members, joined by observation id.

    Parameters
    ----------
    members : `FittedOrbitMembers` (M)
        Members returned by a fitter; every ``obs_id`` must be an id of
        ``original``.
    original : `OrbitDeterminationObservations` (N)
        Observations as originally supplied to orbit determination.
    used : `OrbitDeterminationObservations` (N)
        Observations as actually used by the fitter (after the observation
        models); must contain the same ids as ``original``.

    Returns
    -------
    members : `FittedOrbitMembers` (M)
        Input members with ``original_astrometry``, ``used_astrometry`` and
        ``astcat`` populated. Existing columns are unchanged.

    Raises
    ------
    ValueError
        If a member references an observation id absent from ``original`` or
        ``used``.
    """
    if len(members) == 0:
        return members

    obs_ids = _as_array(members.obs_id)
    original_indices = pc.index_in(obs_ids, value_set=_as_array(original.id))
    used_indices = pc.index_in(obs_ids, value_set=_as_array(used.id))
    for name, indices in (("original", original_indices), ("used", used_indices)):
        if indices.null_count > 0:
            unknown = pc.unique(pc.filter(obs_ids, pc.is_null(indices))).to_pylist()
            raise ValueError(
                f"Fitted orbit members reference observation ids absent from the "
                f"{name} observations: {unknown}"
            )

    original_rows = original.take(original_indices)
    used_rows = used.take(used_indices)
    return (
        members.set_column(
            "original_astrometry",
            ObservationAstrometry.from_spherical(original_rows.coordinates),
        )
        .set_column(
            "used_astrometry",
            ObservationAstrometry.from_spherical(used_rows.coordinates),
        )
        .set_column("astcat", original_rows.astcat)
    )


def run_od(
    observations: OrbitDeterminationObservations,
    fitter: OrbitFitter,
    models: ObservationModels = None,
    *,
    propagator: Propagator,
    object_id: str | None = None,
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Run orbit determination on the original observations, applying observation
    models at fit time and recording original vs. used astrometry on the
    returned members.

    This is the blessed entry point for orbit determination. It

    1. applies ``models`` to the ORIGINAL ``observations`` to obtain the
       observations the fitter will use (`apply_observation_models`);
    2. runs the backend's full orbit determination on the used observations
       via `OrbitFitter.full_od` (initial fit followed by refinement for
       `NativeOrbitFitter`; backend-specific pipelines for plugins that
       override ``full_od``);
    3. returns the fitted orbit(s) and members carrying residuals and
       solution / outlier flags from the fitter together with the ORIGINAL and
       USED position and uncertainty of each observation and its star catalog
       (`attach_observation_provenance`).

    Parameters
    ----------
    observations : `OrbitDeterminationObservations` (N)
        Original observations of a single object, with unique ids. These are
        never modified; the values the fitter used are recorded on the
        returned members.
    fitter : `OrbitFitter`
        Backend used for the fit, e.g. `NativeOrbitFitter` or an external
        plugin. The fitter receives the transformed observations and never
        the models.
    models : `ObservationUncertaintyModel`, sequence of them, or None, optional
        Observation model(s) to apply before fitting, in order. Default None
        fits the observations as supplied (members then record
        ``used_astrometry`` equal to ``original_astrometry``).
    propagator : `~adam_core.propagator.Propagator`
        Propagator used by the backend during refinement.
    object_id : str, optional
        Object identifier passed to the fitter and stamped on the returned
        orbits when the backend leaves ``object_id`` null. If None, an opaque
        identifier is generated for the backend and the orbits' ``object_id``
        is left as the backend returned it.

    Returns
    -------
    fitted_orbits : `FittedOrbits` (1 or 0)
        Orbit(s) found by the backend (empty if the fit failed).
    fitted_orbit_members : `FittedOrbitMembers` (M)
        Members with residuals and flags from the fitter and with
        ``original_astrometry``, ``used_astrometry`` and ``astcat``
        populated for every member observation.

    Raises
    ------
    ValueError
        If observation ids are not unique, if the models change the set or
        order of observations, or if the fitter returns members referencing
        unknown observation ids.

    Notes
    -----
    Backends that perform their own observation debiasing or weighting
    internally (e.g. FindOrb) will apply them on top of the models supplied
    here; disabling such internal handling when models are supplied is the
    responsibility of the backend plugin.
    """
    if len(observations) == 0:
        return FittedOrbits.empty(), FittedOrbitMembers.empty()

    ids = _as_array(observations.id)
    if ids.null_count > 0 or pc.count_distinct(ids).as_py() != len(ids):
        raise ValueError("Observation ids must be unique and non-null")

    used = apply_observation_models(observations, models)

    backend_object_id = object_id if object_id is not None else uuid.uuid4().hex
    fitted_orbits, fitted_orbit_members = fitter.full_od(
        backend_object_id, used, propagator
    )

    if (
        object_id is not None
        and len(fitted_orbits) > 0
        and fitted_orbits.object_id.null_count == len(fitted_orbits)
    ):
        fitted_orbits = fitted_orbits.set_column(
            "object_id",
            pa.array([object_id] * len(fitted_orbits), type=pa.large_string()),
        )

    fitted_orbit_members = attach_observation_provenance(
        fitted_orbit_members, observations, used
    )
    return fitted_orbits, fitted_orbit_members
