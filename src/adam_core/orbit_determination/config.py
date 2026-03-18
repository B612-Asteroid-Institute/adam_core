"""
Configuration dataclasses and enumerations for the orbit determination module.
"""

import enum
from dataclasses import dataclass, field
from typing import Any


class WeightingPolicy(enum.Enum):
    """Controls how residual weighting is handled during orbit fitting.

    Parameters
    ----------
    DELEGATE :
        Delegate residual weighting entirely to the backend.  The backend
        is responsible for computing and applying its own weights.  This is
        the default and is appropriate when you want to use the backend
        author's recommended settings without post-processing.
    ADAM :
        After the backend returns an orbit, override the per-observation
        residuals by calling :func:`~adam_core.orbit_determination.evaluate.evaluate_orbits`
        with an adam-managed propagator.  This requires a ``propagator``
        key to be present in :attr:`BackendConfig.backend_kwargs`.

    Notes
    -----
    Per-observatory uncertainty overrides are a planned future extension;
    see TODO(od-module): per-observatory uncertainty overrides.
    """

    DELEGATE = "delegate"
    ADAM = "adam"


@dataclass
class BackendConfig:
    """Runtime configuration passed to a :func:`~adam_core.orbit_determination.fit_orbit.fit_orbit`
    backend.

    Parameters
    ----------
    weighting_policy : WeightingPolicy, optional
        How to handle residual weighting.  Defaults to
        :attr:`WeightingPolicy.DELEGATE`.
    backend_kwargs : dict, optional
        Arbitrary keyword arguments forwarded transparently to the selected
        backend.  Common keys:

        * ``"propagator"`` – a :class:`~adam_core.propagator.Propagator`
          *class* (not instance) required by the
          :attr:`WeightingPolicy.ADAM` policy and the adam backend.
        * ``"min_obs"`` – minimum number of observations for the adam backend.
        * ``"rchi2_threshold"`` – reduced-chi2 convergence threshold.
        * ``"contamination_percentage"`` – maximum fraction of outliers.
        * ``"out_dir"`` – output directory forwarded to the find_orb backend.

    Examples
    --------
    >>> from adam_core.orbit_determination.config import BackendConfig, WeightingPolicy
    >>> cfg = BackendConfig(
    ...     weighting_policy=WeightingPolicy.ADAM,
    ...     backend_kwargs={"propagator": MyPropagator},
    ... )
    """

    weighting_policy: WeightingPolicy = WeightingPolicy.DELEGATE
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
