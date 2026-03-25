"""
Native adam_core orbit-determination backend for
:func:`~adam_core.orbit_determination.fit_orbit.fit_orbit`.

This backend chains the existing adam_core IOD pipeline with differential
correction:

1. :func:`~adam_core.orbit_determination.iod.iod` — runs Gauss IOD on
   observation triplets selected from the input set and evaluates candidate
   orbits against all observations.
2. :func:`~adam_core.orbit_determination.od.differential_correction` — refines
   the IOD candidates.

A :class:`~adam_core.propagator.Propagator` class **must** be supplied via
``BackendConfig.backend_kwargs["propagator"]``.  The canonical choice is
``ASSISTPropagator`` from the ``adam-assist`` package.

Notes
-----
The adam backend is *always* available (no optional dependency).  It uses only
packages that are already required by adam_core itself.

Because :func:`~adam_core.orbit_determination.iod.iod` may return multiple
candidate orbits (one per Gauss triplet that converges), this backend returns
*all* candidates after differential correction.  Callers that want a single
best orbit should select the row with the lowest ``reduced_chi2``.

The :attr:`~adam_core.orbit_determination.config.WeightingPolicy` setting has
no effect on this backend because residuals are always computed by
:func:`~adam_core.orbit_determination.evaluate.evaluate_orbits`.
"""

import logging
import os
from typing import Literal, Optional, Tuple, Type

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..config import BackendConfig
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..iod import iod
from ..od import differential_correction
from .base import BackendWrapper

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))

_BACKEND_VERSION = "adam_core"  # versioned at package level; see importlib.metadata


class AdamBackend(BackendWrapper):
    """Native adam_core IOD + differential-correction backend.

    This backend is always available and requires no external binaries.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from adam_core.orbit_determination.backends import AdamBackend
    >>> from adam_core.orbit_determination.config import BackendConfig
    >>> from adam_assist import ASSISTPropagator
    >>> backend = AdamBackend()
    >>> config = BackendConfig(backend_kwargs={"propagator": ASSISTPropagator})
    >>> fitted, members = backend.fit(observations, config)
    """

    AVAILABLE: bool = True
    BACKEND_NAME: str = "adam"

    def fit(
        self,
        observations: OrbitDeterminationObservations,
        config: BackendConfig,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        """Fit an orbit using adam_core's IOD + differential-correction pipeline.

        Parameters
        ----------
        observations : OrbitDeterminationObservations
            Observations for a single object.
        config : BackendConfig
            Runtime configuration.  Relevant ``backend_kwargs`` keys:

            * ``"propagator"`` *(Type[Propagator], required)* — propagator
              class used for ephemeris generation.
            * ``"min_obs"`` *(int, default 6)* — minimum number of
              observations required for a valid solution.
            * ``"min_arc_length"`` *(float, default 1.0)* — minimum arc
              length in days.
            * ``"contamination_percentage"`` *(float, default 20.0)* —
              maximum fraction of observations that can be flagged as
              outliers.
            * ``"rchi2_threshold"`` *(float, default 200)* — maximum reduced
              chi2 for an IOD candidate to be accepted.
            * ``"observation_selection_method"``
              *({'combinations', 'first+middle+last', 'thirds'}, default
              'combinations')* — how to pick triplets for Gauss IOD.
            * ``"max_processes"`` *(int or None, default 1)* — number of
              parallel processes to use.

        Returns
        -------
        fitted_orbits : FittedOrbits
            Differentially-corrected orbit(s).  Multiple rows indicate
            multiple IOD candidates that all converged.
        fitted_orbit_members : FittedOrbitMembers
            Per-observation residuals and outlier flags.

        Raises
        ------
        ValueError
            If ``propagator`` is not provided in ``backend_kwargs``, or if
            no valid IOD candidate is found and differential correction
            fails for all of them.
        RuntimeError
            If differential correction produces no orbits (all candidates
            rejected after outlier removal).
        """
        self._check_available()

        # --- Extract configuration ------------------------------------------
        propagator_cls = config.backend_kwargs.get("propagator")
        if propagator_cls is None:
            raise ValueError(
                "The adam backend requires a 'propagator' key in "
                "BackendConfig.backend_kwargs.  "
                "Example:\n"
                "  from adam_assist import ASSISTPropagator\n"
                "  config = BackendConfig(\n"
                "      backend_kwargs={'propagator': ASSISTPropagator}\n"
                "  )"
            )

        kwargs = config.backend_kwargs
        min_obs: int = kwargs.get("min_obs", 6)
        min_arc_length: float = kwargs.get("min_arc_length", 1.0)
        contamination_percentage: float = kwargs.get("contamination_percentage", 20.0)
        rchi2_threshold: float = kwargs.get("rchi2_threshold", 200.0)
        observation_selection_method: Literal[
            "combinations", "first+middle+last", "thirds"
        ] = kwargs.get("observation_selection_method", "combinations")
        max_processes: Optional[int] = kwargs.get("max_processes", 1)
        propagator_kwargs: dict = kwargs.get("propagator_kwargs", {})

        logger.info(
            "AdamBackend: running IOD on %d observations (min_obs=%d, "
            "rchi2_threshold=%.1f).",
            len(observations),
            min_obs,
            rchi2_threshold,
        )

        # --- Step 1: Initial orbit determination (Gauss IOD) -----------------
        iod_orbits, iod_members = iod(
            observations=observations,
            propagator=propagator_cls,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            rchi2_threshold=rchi2_threshold,
            observation_selection_method=observation_selection_method,
            propagator_kwargs=propagator_kwargs,
        )

        if len(iod_orbits) == 0:
            raise ValueError(
                f"adam backend: IOD found no valid orbit candidates for "
                f"{len(observations)} observations.  "
                f"Try lowering 'rchi2_threshold' (currently {rchi2_threshold}) "
                f"or 'min_obs' (currently {min_obs}), or check that observer "
                f"states are populated and covariances are set."
            )

        logger.info(
            "AdamBackend: IOD produced %d candidate(s); running differential correction.",
            len(iod_orbits),
        )

        # --- Step 2: Differential correction ---------------------------------
        dc_orbits, dc_members = differential_correction(
            orbits=iod_orbits,
            orbit_members=iod_members,
            observations=observations,
            propagator=propagator_cls,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            rchi2_threshold=rchi2_threshold,
            max_processes=max_processes,
            propagator_kwargs=propagator_kwargs,
        )

        if len(dc_orbits) == 0:
            raise RuntimeError(
                "adam backend: differential correction produced no orbits.  "
                "All IOD candidates were rejected (reduced chi2 too high or "
                "arc length too short after outlier removal).  "
                f"Started with {len(iod_orbits)} IOD candidate(s), "
                f"{len(observations)} observations."
            )

        logger.info(
            "AdamBackend: differential correction converged on %d orbit(s).",
            len(dc_orbits),
        )

        # --- Stamp provenance ------------------------------------------------
        try:
            from importlib.metadata import version as _version

            pkg_version = _version("adam-core")
        except Exception:
            pkg_version = _BACKEND_VERSION

        dc_orbits = dc_orbits.set_column(
            "backend", pa.repeat(self.BACKEND_NAME, len(dc_orbits))
        )
        dc_orbits = dc_orbits.set_column(
            "backend_version", pa.repeat(pkg_version, len(dc_orbits))
        )

        return dc_orbits, dc_members
