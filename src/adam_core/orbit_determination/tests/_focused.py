"""Test-only builders for complete explicit focused-workflow configuration."""

from dataclasses import fields, replace
from typing import Any

import numpy as np
import pyarrow as pa

from adam_core import _rust_native

from .._native_dispatch import native_propagator
from ..algorithms.fitting import (
    _emit_native_warnings,
    _fitted_tables_from_native_output,
    _native_problem,
)
from ..algorithms.iod import _run_iod_prepared
from ..config import (
    EvaluationConfig,
    IodRunConfig,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
    WorstResidualRefinement,
)
from ..observation_pipeline import ObservationPipeline


def explicit_config(**settings: Any) -> OdConfig:
    """Build a complete config while varying fields under numerical test."""

    iod = IodRunConfig.core_v1()
    search = iod.search_correction_ranging
    correction_settings = dict(settings.pop("correction_settings", {}) or {})
    ranging_settings = dict(settings.pop("ranging_settings", {}) or {})
    span_settings = dict(settings.pop("span_settings", {}) or {})
    fit_settings = dict(settings.pop("fit_settings", {}) or {})
    refinement_method = settings.pop("refinement_method", "cmc2003")

    # Historical test spellings are translated only inside the test suite.
    ranging_settings = {
        (key.removeprefix("ranging_")): value for key, value in ranging_settings.items()
    }
    span_aliases = {
        "span_max_triplets": "max_triplets",
        "span_max_days": "max_span_days",
        "span_min_days": "min_span_days",
        "span_target_days": "target_span_days",
    }
    span_settings = {
        span_aliases.get(key, key.removeprefix("span_")): value
        for key, value in span_settings.items()
    }
    search_names = {item.name for item in fields(search)}
    direct = {key: settings.pop(key) for key in list(settings) if key in search_names}
    if settings:
        raise TypeError(f"unknown test IOD settings: {sorted(settings)}")
    if correction_settings:
        search = replace(
            search,
            correction=replace(search.correction, **correction_settings),
        )
    if ranging_settings:
        search = replace(search, ranging=replace(search.ranging, **ranging_settings))
    if span_settings:
        search = replace(search, spans=replace(search.spans, **span_settings))
    if direct:
        search = replace(search, **direct)
    two_body_aliases = {
        "prop_tol": "propagation_tol",
        "prop_max_iter": "propagation_max_iter",
        "eph_tol": "ephemeris_tol",
        "eph_max_iter": "ephemeris_max_iter",
        "lt_tol": "lt_tol",
        "max_lt_iter": "max_lt_iter",
    }
    two_body_values = {
        two_body_aliases.pop(key): fit_settings.pop(key)
        for key in list(fit_settings)
        if key in two_body_aliases
    }
    fit = iod.candidate_correction_fit
    if two_body_values:
        fit = replace(fit, two_body=replace(fit.two_body, **two_body_values))
    if fit_settings:
        fit = replace(fit, **fit_settings)
    iod = IodRunConfig(search, fit)
    refinement = RefinementRunConfig.core_v1()
    if refinement_method == "worst_residual":
        refinement = RefinementRunConfig(
            WorstResidualRefinement(
                method="worst_residual",
                rchi2_threshold=10.0,
                min_obs=6,
                min_arc_length=1.0,
                contamination_percentage=20.0,
                fit=refinement.refinement.fit,
            )
        )
    elif refinement_method != "cmc2003":
        raise ValueError(f"unknown test refinement method {refinement_method!r}")
    return OdConfig.explicit(
        observation_pipeline=ObservationPipeline.as_supplied(),
        iod=iod,
        refinement=refinement,
        evaluation=EvaluationConfig.six_dimensional(),
        resources=ResourceBindings({}),
    )


def _instance(propagator):
    return propagator() if isinstance(propagator, type) else propagator


def run_iod_qualified(observations, propagator, **settings):
    """Test-only numerical bridge; not a production configured workflow."""
    propagator = _instance(propagator)
    config = explicit_config(**settings)
    return _run_iod_prepared(observations, propagator, config.for_iod())


def run_prefit_iod_qualified(observations, propagator, **settings):
    settings.setdefault("acceptance", "prefit")
    settings.setdefault("velocity_method", "gibbs")
    orbit, members, _ = run_iod_qualified(observations, propagator, **settings)
    return orbit, members


def refine_qualified(
    orbit, observations, propagator, *, _config: OdConfig | None = None, **settings
):
    """Test-only native refinement bridge without workflow provenance.

    ``_config`` lets the composition helper pass the exact configuration used
    for IOD. Direct callers instead provide only refinement settings.
    """
    propagator = _instance(propagator)
    if _config is not None and settings:
        raise TypeError("settings cannot be combined with the prepared test config")
    config = _config if _config is not None else explicit_config(**settings)
    seed = orbit.to_orbits() if hasattr(orbit, "to_orbits") else orbit
    output = _rust_native.refine_config_ipc(
        native_propagator(propagator),
        *_native_problem(seed, observations),
        config.for_refinement()._native,
    )
    _emit_native_warnings(output)
    return _fitted_tables_from_native_output(
        seed,
        observations,
        output,
        np.asarray(output["weights"], dtype=np.float64),
    )


def run_od_qualified(observations, propagator, *, object_id=None, **settings):
    """Test-only IOD then refinement composition over as-supplied rows."""
    propagator = _instance(propagator)
    config = explicit_config(**settings)
    seed, _, _ = _run_iod_prepared(observations, propagator, config.for_iod())
    if len(seed) == 0:
        from ..fitted_orbits import FittedOrbitMembers, FittedOrbits

        return FittedOrbits.empty(), FittedOrbitMembers.empty()
    fitted, members = refine_qualified(
        seed,
        observations,
        propagator,
        _config=config,
    )
    if object_id is not None and len(fitted):
        fitted = fitted.set_column(
            "object_id",
            pa.array([object_id] * len(fitted), type=pa.large_string()),
        )
    return fitted, members
