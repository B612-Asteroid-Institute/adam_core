import logging
import uuid
import warnings
from typing import Any, Literal, Optional, Tuple, Type, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from ..time import Timestamp
from ._native_dispatch import (
    _has_compatible_native_spec,
    _warn_if_analytic_jacobian_mismatches,
    native_propagator,
)
from .differential_correction import (
    _FUSED_LOOP_KWARGS,
    HUBER_F_SCALE_DEFAULT,
    OD_DEFAULTS,
    LossType,
    _emit_native_warnings,
    _fit_settings,
    _fitted_tables_from_native_output,
    _iterative_fit,
)
from .evaluate import OrbitDeterminationObservations, evaluate_orbits
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, OdProvenance
from .iod import IOD_DEFAULTS, _placeholder_orbit, run_iod
from .orbit_fitter import OrbitFitter
from .rejection import _cmc2003_fit

logger = logging.getLogger(__name__)


def tables_from_native_full_od(
    object_id: str | pa.LargeStringScalar,
    observations: OrbitDeterminationObservations,
    output: dict,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Wrap the dict from adam-core's ``full_od`` driver into `FittedOrbits` /
    `FittedOrbitMembers`: empty tables when IOD found no orbit, otherwise the
    refined fit with Python-owned orbit ids and the given object id.
    """
    if not output["found"] or output.get("fit") is None:
        reason = (output.get("iod") or {}).get("not_found_reason")
        if reason:
            logger.debug("IOD found no orbit for %s: %s", object_id, reason)
        return FittedOrbits.empty(), FittedOrbitMembers.empty()
    iod = output["iod"]
    state = np.asarray(iod["state"], dtype=np.float64)
    seed = Orbits.from_kwargs(
        orbit_id=[uuid.uuid4().hex],
        object_id=[str(object_id)],
        coordinates=CartesianCoordinates.from_kwargs(
            x=state[0:1],
            y=state[1:2],
            z=state[2:3],
            vx=state[3:4],
            vy=state[4:5],
            vz=state[5:6],
            # The IOD epoch crosses exactly (integer days and nanoseconds).
            time=Timestamp.from_kwargs(
                days=[int(iod["epoch_days"])],
                nanos=[int(iod["epoch_nanos"])],
                scale=iod["epoch_scale"],
            ),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    fit = output["fit"]
    _emit_native_warnings(fit)
    fitted_orbits, members = _fitted_tables_from_native_output(
        seed, observations, fit, np.asarray(fit["weights"], dtype=np.float64)
    )
    alternatives = len(iod.get("ambiguous_with") or ())
    import json

    iod_diagnostics = {
        key: iod.get(key)
        for key in (
            "triplets_tried",
            "candidates_evaluated",
            "candidates_failed",
            "candidate_failures",
            "pair_visits",
            "raw_pair_visits",
            "middle_visits",
            "search_cap_exhausted",
            "corrections_run",
            "accepted",
            "ambiguous_with",
            "not_found_reason",
        )
    }
    refinement_diagnostics = {
        key: fit.get(key) for key in ("passes", "n_rejected", "n_recovered", "flags")
    }
    fitted_orbits = fitted_orbits.set_column(
        "provenance",
        OdProvenance.from_kwargs(
            iod_ambiguous=[alternatives > 0] * len(fitted_orbits),
            iod_alternatives=[alternatives] * len(fitted_orbits),
            iod_provider=[
                json.dumps(
                    output["iod_provider"], sort_keys=True, separators=(",", ":")
                )
            ]
            * len(fitted_orbits),
            refinement_provider=[
                json.dumps(
                    output["refinement_provider"], sort_keys=True, separators=(",", ":")
                )
            ]
            * len(fitted_orbits),
            iod_diagnostics=[
                json.dumps(iod_diagnostics, sort_keys=True, separators=(",", ":"))
            ]
            * len(fitted_orbits),
            refinement_diagnostics=[
                json.dumps(
                    refinement_diagnostics, sort_keys=True, separators=(",", ":")
                )
            ]
            * len(fitted_orbits),
        ),
    )
    return fitted_orbits, members


class NativeOrbitFitter(OrbitFitter):
    """
    Orbit fitter using adam_core's native Gauss IOD and iterative least-squares DC.

    The `OrbitFitter` over adam_core's native stages: `run_iod` (Gauss
    candidates accepted by staged correction; see the IOD policy) on the IOD
    propagator, then `refine` (CMC2003 or worst-residual rejection) on the
    propagator `full_od` / `run_od` receive. Every loop belongs to adam-core
    and runs in Rust in one workflow crossing. Each stage propagator supplies
    only a native propagation capsule; providers do not implement OD methods.

    Parameters
    ----------
    iod_propagator : Propagator
        The propagator *instance* that scores and corrects the IOD candidates
        (the refinement propagator is the one `full_od` / `run_od` receive).
        Passing a propagator class, or ``propagator_class=`` /
        ``propagator_kwargs=``, is deprecated: the class is instantiated with
        the kwargs for each IOD run, as before.
    propagator_kwargs : dict, optional
        Deprecated; only with a propagator class.
    min_obs : int, optional
        Minimum number of observations required for a valid fit.  Default 6.
    min_arc_length : float, optional
        Minimum arc length in days required to retain a fit.  Default 1.0.
    contamination_percentage : float, optional
        Maximum percentage of observations that may be rejected as outliers
        across the full OD pipeline.  Default 20.0.
    rchi2_threshold : float, optional
        Reduced chi2 convergence threshold for differential correction.
        Default 10.0.
    iod_rchi2_threshold : float, optional
        Reduced chi2 threshold used during IOD to filter candidate orbits.
        Default 200.0.
    observation_selection_method : str, optional
        Strategy for selecting observation triplets in IOD.  One of
        ``"combinations"``, ``"first+middle+last"``, ``"thirds"``,
        ``"spans"`` (span-bounded triplets swept across search windows).
        Default ``"spans"``.
    velocity_method : {"gauss", "gibbs", "herrick+gibbs", "auto"}, optional
        Velocity of the Gauss IOD candidates; ``"auto"`` chooses
        Herrick-Gibbs or Gibbs per root by the separation of its position
        vectors. Default ``"auto"``.
    iod_acceptance : {"corrected", "prefit"}, optional
        ``"corrected"`` (default): each Gauss orbit seeds staged differential
        corrections from its triplet's span to the whole arc, and the
        converged candidate with the lowest reduced chi2 is accepted
        (``iod_rchi2_threshold`` bounds every stage). ``"prefit"``: the
        legacy gate on the raw Gauss orbit. See `run_iod`.
    loss : {"linear", "huber"}, optional
        Loss used during differential correction (`iterative_fit`).
        ``"huber"`` downweights large residuals instead of letting them pull
        the solution; combine with ``contamination_percentage=0.0`` to use
        Huber M-estimation as the sole outlier treatment. Default
        ``"linear"``.
    f_scale : float, optional
        Huber transition point in units of whitened (1-sigma) residual
        components. Default 1.345. Ignored for ``loss="linear"``.
    outlier_rejection : {"cmc2003", "worst_residual"}, optional
        Outlier treatment during differential correction.

        - ``"cmc2003"`` (default): `cmc2003_fit`, Carpino-Milani-Chesley
          (2003) rejection with re-inclusion against the expected post-fit
          residual covariance, with OrbFit's ``reject.def`` constants.
          Covariance and rejection share central differences through the full
          predictor; ``jacobian`` in ``rejection_kwargs`` controls optimizer
          steps only. ``rchi2_threshold``, ``min_obs``, ``min_arc_length`` and
          ``contamination_percentage`` then apply to IOD only; tune the
          scheme through ``rejection_kwargs``.
        - ``"worst_residual"``: `iterative_fit`, which removes the
          worst-residual observation and refits while the reduced chi2
          exceeds ``rchi2_threshold``, bounded by ``contamination_percentage``,
          ``min_obs`` and ``min_arc_length`` (the pre-2026-09 behaviour).

        Both accept ``loss="huber"``; its combination with CMC2003 is
        heuristic, without calibrated rejection statistics. See the Notes
        of `run_od` for the full default configuration.
    rejection_kwargs : dict, optional
        Extra keyword arguments for the rejection function (e.g.
        ``chi2_reject``, ``chi2_recover`` for ``"cmc2003"``; ``jacobian``,
        ``max_nfev`` for either).
    """

    def __init__(
        self,
        iod_propagator: Propagator | Type[Propagator] | None = None,
        propagator_kwargs: Optional[dict[str, Any]] = None,
        min_obs: int = IOD_DEFAULTS["min_obs"],
        min_arc_length: float = IOD_DEFAULTS["min_arc_length"],
        contamination_percentage: float = IOD_DEFAULTS["contamination_percentage"],
        rchi2_threshold: float = OD_DEFAULTS["worst_residual"]["rchi2_threshold"],
        iod_rchi2_threshold: float = IOD_DEFAULTS["rchi2_threshold"],
        observation_selection_method: Literal[
            "combinations", "first+middle+last", "thirds", "spans"
        ] = IOD_DEFAULTS["observation_selection_method"],
        velocity_method: Literal[
            "gauss", "gibbs", "herrick+gibbs", "auto"
        ] = IOD_DEFAULTS["velocity_method"],
        iod_acceptance: Literal["corrected", "prefit"] = IOD_DEFAULTS["acceptance"],
        loss: LossType = OD_DEFAULTS["fit"]["loss"],
        f_scale: float = HUBER_F_SCALE_DEFAULT,
        outlier_rejection: Literal["worst_residual", "cmc2003"] = cast(
            Literal["worst_residual", "cmc2003"], OD_DEFAULTS["refinement"]
        ),
        rejection_kwargs: dict[str, Any] | None = None,
        *,
        propagator_class: Optional[Type[Propagator]] = None,
    ) -> None:
        if propagator_class is not None:
            if iod_propagator is not None:
                raise TypeError("pass either iod_propagator or propagator_class")
            iod_propagator = propagator_class
        if iod_propagator is None:
            raise TypeError("NativeOrbitFitter needs an IOD propagator instance")
        if isinstance(iod_propagator, type):
            warnings.warn(
                "NativeOrbitFitter(propagator_class, propagator_kwargs) is "
                "deprecated: pass a propagator instance, "
                "NativeOrbitFitter(propagator_class(**propagator_kwargs)).",
                DeprecationWarning,
                stacklevel=2,
            )
            self._iod_instance: Optional[Propagator] = None
            self.propagator_class = iod_propagator
            self.propagator_kwargs = dict(propagator_kwargs or {})
        else:
            if propagator_kwargs:
                raise TypeError(
                    "propagator_kwargs only applies to a propagator class; "
                    "configure the instance instead"
                )
            self._iod_instance = iod_propagator
            self.propagator_class = type(iod_propagator)
            self.propagator_kwargs = {}
        if outlier_rejection not in ("worst_residual", "cmc2003"):
            raise ValueError(
                "outlier_rejection must be 'worst_residual' or 'cmc2003'; "
                f"got {outlier_rejection!r}"
            )
        self.min_obs = min_obs
        self.min_arc_length = min_arc_length
        self.contamination_percentage = contamination_percentage
        self.rchi2_threshold = rchi2_threshold
        self.iod_rchi2_threshold = iod_rchi2_threshold
        self.observation_selection_method = observation_selection_method
        self.velocity_method = velocity_method
        self.iod_acceptance = iod_acceptance
        self.loss = loss
        self.f_scale = f_scale
        self.outlier_rejection = outlier_rejection
        self.rejection_kwargs = dict(rejection_kwargs or {})

    def native_settings(self) -> tuple[dict, dict, dict]:
        """
        ``(iod_settings, refinement, fit_settings)`` dicts describing this
        fitter's configuration for adam-core's native ``full_od`` / ``run_od``
        drivers: the IOD decision-loop settings,
        the outlier-rejection loop and its constants, and the whitened-fit
        options (`fit_least_squares` defaults unless overridden through
        ``rejection_kwargs``).

        A subclass overriding ``full_od``, ``initial_fit`` or ``refine_fit``
        uses the stage composition by default. Defining ``native_settings``
        on that subclass explicitly opts into the one-crossing core driver and
        promises these dicts describe its complete fitting behavior, including
        the overridden hooks.
        """
        from .gauss import MU, C

        iod_settings = {
            "min_obs": self.min_obs,
            "min_arc_length": self.min_arc_length,
            "contamination_percentage": self.contamination_percentage,
            "rchi2_threshold": self.iod_rchi2_threshold,
            "observation_selection_method": self.observation_selection_method,
            "velocity_method": self.velocity_method,
            "acceptance": self.iod_acceptance,
            "light_time": True,
            "mu": MU,
            "speed_of_light": C,
        }
        loop_keys = {
            "cmc2003": {
                "chi2_reject",
                "chi2_recover",
                "chi2_frac",
                "max_iterations",
                "max_rejected_fraction",
                "apparition_gap_days",
                "psd_floor_frac",
            },
            "worst_residual": set(),
        }[self.outlier_rejection]
        refinement = {
            "method": self.outlier_rejection,
            **{k: v for k, v in self.rejection_kwargs.items() if k in loop_keys},
        }
        if self.outlier_rejection == "worst_residual":
            refinement.update(
                rchi2_threshold=self.rchi2_threshold,
                min_obs=self.min_obs,
                min_arc_length=self.min_arc_length,
                contamination_percentage=self.contamination_percentage,
            )
        fit_kwargs = {
            k: v for k, v in self.rejection_kwargs.items() if k not in loop_keys
        }
        fit_settings = _fit_settings(
            self.loss,
            self.f_scale,
            fit_kwargs.get("jacobian", OD_DEFAULTS["fit"]["jacobian"]),
            fit_kwargs.get("validate_covariance", True),
            fit_kwargs,
        )
        return iod_settings, refinement, fit_settings

    def _supports_native_full_od_snapshot(self) -> bool:
        """Whether hooks can use one already-frozen settings snapshot."""
        if not _has_compatible_native_spec(
            self, "native_settings", ("full_od", "initial_fit", "refine_fit")
        ):
            return False
        loop_keys = {
            "chi2_reject",
            "chi2_recover",
            "chi2_frac",
            "max_iterations",
            "max_rejected_fraction",
            "apparition_gap_days",
            "psd_floor_frac",
        }
        return set(self.rejection_kwargs) <= loop_keys | _FUSED_LOOP_KWARGS

    def supports_native_full_od(self) -> bool:
        """Whether the fitting hooks fit adam-core's one-crossing driver.

        Inherited settings cannot represent overridden fitting hooks. Check
        that compatibility before parsing settings for the built-in drivers.
        """
        if not self._supports_native_full_od_snapshot():
            return False
        _, _, _ = self.native_settings()
        return True

    @property
    def iod_propagator(self) -> Propagator:
        """The IOD propagator: the configured instance, or (deprecated
        class form) a fresh ``propagator_class(**propagator_kwargs)``."""
        if self._iod_instance is not None:
            return self._iod_instance
        return self.propagator_class(**self.propagator_kwargs)

    def fused_propagator_matches(self, propagator: Propagator) -> bool:
        """
        Whether ``propagator`` is the IOD propagator this fitter is
        configured for: the configured instance itself, or (deprecated class
        form) an instance of ``propagator_class`` built without extra keyword
        arguments. When it is, the core driver scores IOD candidates on
        ``propagator`` itself; otherwise `iod_propagator_for` supplies the
        configured IOD propagation capsule explicitly.
        """
        if self._iod_instance is not None:
            return propagator is self._iod_instance
        return (
            isinstance(propagator, self.propagator_class) and not self.propagator_kwargs
        )

    def iod_propagator_for(self, propagator: Propagator) -> Optional[Propagator]:
        """
        The explicit IOD propagator for a core driver refining on
        ``propagator``: None when ``propagator`` already is the configured
        IOD propagator (`fused_propagator_matches`), else a fresh
        ``propagator_class(**propagator_kwargs)``.
        """
        if self.fused_propagator_matches(propagator):
            return None
        return self.iod_propagator

    def full_od(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
        propagator: Propagator,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """
        IOD followed by refinement. The adam-core native driver owns the whole
        pipeline (Gauss IOD decision loop, differential correction and
        outlier rejection) in one crossing when ``supports_native_full_od``
        confirms that this fitter's hooks retain the canonical behavior;
        the IOD stage runs on the configured IOD propagator
        (`iod_propagator_for`), which the driver receives explicitly when it is
        not ``propagator`` itself. Otherwise `initial_fit` and `refine_fit` are chained,
        honoring subclass overrides.
        """
        if not self.supports_native_full_od():
            return super().full_od(object_id, observations, propagator)
        from adam_core import _rust_native

        from .differential_correction import _native_problem

        iod_settings, refinement, fit_settings = self.native_settings()
        _warn_if_analytic_jacobian_mismatches(propagator, fit_settings["jacobian"])
        _, observed_ipc, observers_ipc = _native_problem(
            _placeholder_orbit(observations), observations
        )
        output = _rust_native.full_od_ipc(
            native_propagator(propagator),
            observed_ipc,
            observers_ipc,
            iod_settings=iod_settings,
            refinement=refinement,
            fit_settings=fit_settings,
            iod_propagator=native_propagator(self.iod_propagator_for(propagator)),
        )
        return tables_from_native_full_od(object_id, observations, output)

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        state.setdefault("_iod_instance", None)
        self.__dict__.update(state)

    def initial_fit(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
        reference_orbit: Optional[Orbits] = None,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Run Gauss IOD on the observations, or warm-start from a reference orbit.

        Parameters
        ----------
        object_id : str | pa.LargeStringScalar
            Object identifier for output tables.
        observations : OrbitDeterminationObservations
            Observations to fit, assumed to belong to a single object.
        reference_orbit : Orbits, optional
            Warm-start seed (length-1 Orbits). When provided, Gauss IOD is
            skipped and the seed is evaluated against the observations to
            produce the initial `FittedOrbits`; `refine_fit` then starts the
            differential correction from it.

        Returns
        -------
        fitted_orbit : FittedOrbits
            Best IOD orbit(s) found (may be empty if IOD fails), or the
            evaluated reference orbit.
        fitted_orbit_members : FittedOrbitMembers
            Observations with solution/outlier flags from IOD.
        """
        if reference_orbit is not None:
            assert len(reference_orbit) == 1, "reference_orbit must contain one orbit"
            fitted_orbits, fitted_orbit_members = evaluate_orbits(
                reference_orbit,
                observations,
                self.iod_propagator,
            )
            if fitted_orbits.object_id.null_count == len(fitted_orbits):
                fitted_orbits = fitted_orbits.set_column(
                    "object_id",
                    pa.array(
                        [str(object_id)] * len(fitted_orbits), type=pa.large_string()
                    ),
                )
            fitted_orbit_members = fitted_orbit_members.set_column(
                "solution", pc.invert(fitted_orbit_members.outlier)
            )
            return fitted_orbits, fitted_orbit_members

        fitted_orbits, fitted_orbit_members, _report = run_iod(
            observations,
            self.iod_propagator,
            min_obs=self.min_obs,
            min_arc_length=self.min_arc_length,
            contamination_percentage=self.contamination_percentage,
            rchi2_threshold=self.iod_rchi2_threshold,
            observation_selection_method=self.observation_selection_method,
            velocity_method=self.velocity_method,
            acceptance=self.iod_acceptance,
            fit_settings=self.native_settings()[2],
        )
        return fitted_orbits, fitted_orbit_members

    def refine_fit(
        self,
        fitted_orbit: FittedOrbits,
        observations: OrbitDeterminationObservations,
        propagator: Propagator,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Refine an IOD orbit via differential correction with outlier treatment.

        Parameters
        ----------
        fitted_orbit : FittedOrbits (1)
            Orbit to refine, typically from `initial_fit`.
        observations : OrbitDeterminationObservations
            Observations to fit against.
        propagator : Propagator
            Propagator instance used during DC ephemeris evaluation.

        Returns
        -------
        fitted_orbit : FittedOrbits (1)
            DC-refined orbit with covariance and quality statistics.
        fitted_orbit_members : FittedOrbitMembers (N)
            Observations with residuals and outlier/solution flags.
        """
        assert len(fitted_orbit) == 1, "refine_fit expects exactly one orbit"
        orbit = fitted_orbit.to_orbits()
        if self.outlier_rejection == "cmc2003":
            return _cmc2003_fit(
                orbit,
                observations,
                propagator,
                loss=self.loss,
                f_scale=self.f_scale,
                **self.rejection_kwargs,
            )
        return _iterative_fit(
            orbit,
            observations,
            propagator,
            rchi2_threshold=self.rchi2_threshold,
            min_obs=self.min_obs,
            min_arc_length=self.min_arc_length,
            contamination_percentage=self.contamination_percentage,
            loss=self.loss,
            f_scale=self.f_scale,
            **self.rejection_kwargs,
        )
