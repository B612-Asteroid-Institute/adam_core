import logging
import uuid
from typing import Any, Literal, Optional, Tuple, Type

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from ..time import Timestamp
from .differential_correction import (
    _FUSED_LOOP_KWARGS,
    HUBER_F_SCALE_DEFAULT,
    LossType,
    _emit_native_warnings,
    _fit_settings,
    _fitted_tables_from_native_output,
    iterative_fit,
)
from .evaluate import OrbitDeterminationObservations, evaluate_orbits
from .fitted_orbits import FittedOrbitMembers, FittedOrbits
from .iod import iod
from .orbit_fitter import OrbitFitter
from .rejection import cmc2003_fit

logger = logging.getLogger(__name__)


def tables_from_native_full_od(
    object_id: str | pa.LargeStringScalar,
    observations: OrbitDeterminationObservations,
    output: dict,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Wrap the dict of a fused ``full_od`` work unit into `FittedOrbits` /
    `FittedOrbitMembers`: empty tables when IOD found no orbit, otherwise the
    refined fit with Python-owned orbit ids and the given object id.
    """
    if not output["found"] or output.get("fit") is None:
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
            time=Timestamp.from_mjd(
                [float(iod["epoch_mjd"])], scale=iod["epoch_scale"]
            ),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )
    fit = output["fit"]
    _emit_native_warnings(fit)
    return _fitted_tables_from_native_output(
        seed, observations, fit, np.asarray(fit["weights"], dtype=np.float64)
    )


class NativeOrbitFitter(OrbitFitter):
    """
    Orbit fitter using adam_core's native Gauss IOD and iterative least-squares DC.

    This fitter is a thin wrapper that chains:
    1. `iod()` — Gauss initial orbit determination (Milani 2008)
    2. `iterative_fit()` — scipy least-squares differential correction with
       outlier rejection

    Parameters
    ----------
    propagator_class : Type[Propagator]
        Propagator *class* (not instance) used during IOD ephemeris evaluation.
    propagator_kwargs : dict, optional
        Keyword arguments forwarded to the propagator constructor / IOD call.
        Copied; default None means no arguments.
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
        ``"combinations"``, ``"first+middle+last"``, ``"thirds"``.
        Default ``"combinations"``.
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
          ``rchi2_threshold``, ``min_obs``, ``min_arc_length`` and
          ``contamination_percentage`` then apply to IOD only; tune the
          scheme through ``rejection_kwargs``.
        - ``"worst_residual"``: `iterative_fit`, which removes the
          worst-residual observation and refits while the reduced chi2
          exceeds ``rchi2_threshold``, bounded by ``contamination_percentage``,
          ``min_obs`` and ``min_arc_length`` (the pre-2026-09 behaviour).

        Both compose with ``loss="huber"``. See the Notes of `run_od` for
        the full default configuration.
    rejection_kwargs : dict, optional
        Extra keyword arguments for the rejection function (e.g.
        ``chi2_reject``, ``chi2_recover`` for ``"cmc2003"``; ``jacobian``,
        ``max_nfev`` for either).
    """

    def __init__(
        self,
        propagator_class: Type[Propagator],
        propagator_kwargs: Optional[dict[str, Any]] = None,
        min_obs: int = 6,
        min_arc_length: float = 1.0,
        contamination_percentage: float = 20.0,
        rchi2_threshold: float = 10.0,
        iod_rchi2_threshold: float = 200.0,
        observation_selection_method: Literal[
            "combinations", "first+middle+last", "thirds"
        ] = "combinations",
        loss: LossType = "linear",
        f_scale: float = HUBER_F_SCALE_DEFAULT,
        outlier_rejection: Literal["worst_residual", "cmc2003"] = "cmc2003",
        rejection_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if outlier_rejection not in ("worst_residual", "cmc2003"):
            raise ValueError(
                "outlier_rejection must be 'worst_residual' or 'cmc2003'; "
                f"got {outlier_rejection!r}"
            )
        self.propagator_class = propagator_class
        self.propagator_kwargs = dict(propagator_kwargs or {})
        self.min_obs = min_obs
        self.min_arc_length = min_arc_length
        self.contamination_percentage = contamination_percentage
        self.rchi2_threshold = rchi2_threshold
        self.iod_rchi2_threshold = iod_rchi2_threshold
        self.observation_selection_method = observation_selection_method
        self.loss = loss
        self.f_scale = f_scale
        self.outlier_rejection = outlier_rejection
        self.rejection_kwargs = dict(rejection_kwargs or {})

    def native_settings(self) -> tuple[dict, dict, dict]:
        """
        ``(iod_settings, refinement, fit_settings)`` dicts describing this
        fitter's configuration for a Rust-backed propagator's fused
        ``full_od`` / ``run_od`` work units: the IOD decision-loop settings,
        the outlier-rejection loop and its constants, and the whitened-fit
        options (`fit_least_squares` defaults unless overridden through
        ``rejection_kwargs``).
        """
        from .gauss import MU, C

        iod_settings = {
            "min_obs": self.min_obs,
            "min_arc_length": self.min_arc_length,
            "contamination_percentage": self.contamination_percentage,
            "rchi2_threshold": self.iod_rchi2_threshold,
            "observation_selection_method": self.observation_selection_method,
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
            fit_kwargs.get("jacobian", "analytic"),
            fit_kwargs.get("validate_covariance", True),
            fit_kwargs,
        )
        return iod_settings, refinement, fit_settings

    def supports_native_full_od(self) -> bool:
        """Whether ``rejection_kwargs`` are all understood by the fused work units."""
        _, _, _ = self.native_settings()
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

    def full_od(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
        propagator: Propagator,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """
        IOD followed by refinement. On a Rust-backed propagator exposing the
        fused ``full_od`` work unit the whole pipeline (Gauss IOD decision
        loop, differential correction, outlier rejection) runs natively in
        one crossing; otherwise `initial_fit` and `refine_fit` are chained.
        """
        fused = getattr(propagator, "full_od", None)
        if fused is not None and self.supports_native_full_od():
            iod_settings, refinement, fit_settings = self.native_settings()
            output = fused(
                observations,
                iod_settings=iod_settings,
                refinement=refinement,
                fit_settings=fit_settings,
            )
            return tables_from_native_full_od(object_id, observations, output)
        return super().full_od(object_id, observations, propagator)

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
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
                self.propagator_class(**self.propagator_kwargs),
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

        fitted_orbits, fitted_orbit_members = iod(
            observations,
            self.propagator_class,
            min_obs=self.min_obs,
            min_arc_length=self.min_arc_length,
            contamination_percentage=self.contamination_percentage,
            rchi2_threshold=self.iod_rchi2_threshold,
            observation_selection_method=self.observation_selection_method,
            propagator_kwargs=self.propagator_kwargs,
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
            return cmc2003_fit(
                orbit,
                observations,
                propagator,
                loss=self.loss,
                f_scale=self.f_scale,
                **self.rejection_kwargs,
            )
        return iterative_fit(
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
