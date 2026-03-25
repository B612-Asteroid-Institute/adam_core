"""
Public entrypoint for orbit determination.

The single public symbol exported from this module is :func:`fit_orbit`.  All
backend-specific implementation details live under
:mod:`adam_core.orbit_determination.backends`.

Usage
-----
::

    from adam_core.orbit_determination import fit_orbit
    from adam_core.orbit_determination.config import BackendConfig
    from adam_assist import ASSISTPropagator

    fitted_orbits, fitted_members = fit_orbit(
        observations,
        backend="adam",
        config=BackendConfig(backend_kwargs={"propagator": ASSISTPropagator}),
    )

Backend availability
--------------------
The :func:`fit_orbit` function checks whether the requested backend is
installed before dispatching.  If the backend package is absent it raises an
:exc:`ImportError` with an install command in the message.
"""

import logging
import os
from typing import Literal, Optional, Tuple

import pyarrow.compute as pc

from .backends import (
    FIND_ORB_AVAILABLE,
    LAYUP_AVAILABLE,
    ORBFIT_AVAILABLE,
    AdamBackend,
    FindOrbBackend,
    LayupBackend,
    OrbFitBackend,
)
from .config import BackendConfig
from .evaluate import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers, FittedOrbits

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_CLASSES = {
    "adam": AdamBackend,
    "find_orb": FindOrbBackend,
    "orbfit": OrbFitBackend,
    "layup": LayupBackend,
}

_BACKEND_INSTALL_COMMANDS: dict[str, str] = {
    "find_orb": "pip install adam-fo  # then run: build-fo",
    "orbfit": "pip install orbfit",
    "layup": "pip install layup",
}

_BACKEND_AVAILABLE: dict[str, bool] = {
    "adam": True,
    "find_orb": FIND_ORB_AVAILABLE,
    "orbfit": ORBFIT_AVAILABLE,
    "layup": LAYUP_AVAILABLE,
}


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_observations(observations: OrbitDeterminationObservations) -> None:
    """Raise descriptive errors for invalid observation inputs.

    Parameters
    ----------
    observations : OrbitDeterminationObservations
        Observations to validate.

    Raises
    ------
    ValueError
        * If *observations* is empty.
        * If any observation is missing covariance information (all-NaN
          diagonal on RA/Dec dimensions).
        * If observer states (Cartesian coordinates) are all-NaN.
    """
    n = len(observations)
    if n == 0:
        raise ValueError(
            "fit_orbit received 0 observations.  "
            "At least 3 observations are required for orbit determination."
        )

    # --- Check covariances ---------------------------------------------------
    # We require non-NaN sigma on the RA (index 1) and Dec (index 2) dimensions
    # at minimum.  A fully NaN covariance matrix means the backend cannot weight
    # residuals correctly.
    sigmas = observations.coordinates.covariance.sigmas  # (N, 6) array-like
    ra_sigma = sigmas[:, 1].to_numpy(zero_copy_only=False)
    dec_sigma = sigmas[:, 2].to_numpy(zero_copy_only=False)

    import numpy as np

    missing_cov_mask = np.isnan(ra_sigma) & np.isnan(dec_sigma)
    if missing_cov_mask.all():
        raise ValueError(
            "fit_orbit: all observations are missing RA/Dec covariance information.  "
            "Populate SphericalCoordinates.covariance with at least RA and Dec sigmas "
            "before calling fit_orbit."
        )
    if missing_cov_mask.any():
        bad_ids = observations.id.filter(
            pc.equal(
                observations.id,
                observations.id,
            )
        ).to_pylist()
        # Report up to the first 5 bad IDs
        import numpy as np

        bad_idx = list(np.where(missing_cov_mask)[0][:5])
        bad_ids = [observations.id[i].as_py() for i in bad_idx]
        logger.warning(
            "fit_orbit: %d observation(s) are missing RA/Dec covariance "
            "(e.g. ids: %s).  These will be treated as unconstrained by "
            "backends that respect covariances.",
            int(missing_cov_mask.sum()),
            bad_ids,
        )

    # --- Check observer states -----------------------------------------------
    # Observer Cartesian coordinates must be populated (not all NaN).
    obs_x = observations.observers.coordinates.x.to_numpy(zero_copy_only=False)
    if np.isnan(obs_x).all():
        raise ValueError(
            "fit_orbit: observer states (Observers.coordinates) are all NaN.  "
            "Populate observer positions before calling fit_orbit, e.g. via "
            "Observers.from_codes(codes, times)."
        )

    missing_state_mask = np.isnan(obs_x)
    if missing_state_mask.any():
        bad_idx = list(np.where(missing_state_mask)[0][:5])
        bad_ids = [observations.id[i].as_py() for i in bad_idx]
        raise ValueError(
            f"fit_orbit: {int(missing_state_mask.sum())} observation(s) have "
            f"missing observer states (e.g. obs ids: {bad_ids}).  "
            f"Ensure every observation has a corresponding observer position."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_orbit(
    observations: OrbitDeterminationObservations,
    backend: Literal["find_orb", "orbfit", "layup", "adam"] = "adam",
    config: Optional[BackendConfig] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """Fit an orbit to a set of observations using the specified backend.

    This is the primary public entrypoint for orbit determination in
    adam_core.  It validates the input observations, dispatches to the
    requested backend wrapper, and returns standardised
    :class:`~adam_core.orbit_determination.fitted_orbits.FittedOrbits` and
    :class:`~adam_core.orbit_determination.fitted_orbits.FittedOrbitMembers`
    tables regardless of which backend was used.

    Parameters
    ----------
    observations : OrbitDeterminationObservations
        All observations believed to belong to a **single** object.  The
        table must contain:

        * ``id`` — unique observation identifiers.
        * ``coordinates`` — :class:`~adam_core.coordinates.SphericalCoordinates`
          in the equatorial frame with at least RA/Dec sigmas populated.
        * ``observers`` — :class:`~adam_core.observers.Observers` with
          Cartesian states computed (e.g. via
          :meth:`~adam_core.observers.Observers.from_codes`).

    backend : {'adam', 'find_orb', 'orbfit', 'layup'}, optional
        Which backend to use.  Defaults to ``'adam'`` (the native
        adam_core IOD + differential-correction pipeline).  External backends
        require the corresponding optional package to be installed:

        * ``'find_orb'`` → ``pip install adam-fo`` then ``build-fo``
        * ``'orbfit'`` → ``pip install orbfit`` *(not yet implemented)*
        * ``'layup'`` → ``pip install layup`` *(not yet implemented)*

    config : BackendConfig, optional
        Runtime configuration controlling weighting policy and
        backend-specific parameters.  If ``None``, a default
        :class:`~adam_core.orbit_determination.config.BackendConfig` is used
        (``WeightingPolicy.DELEGATE``, no extra kwargs).

    Returns
    -------
    fitted_orbits : FittedOrbits
        Best-fit orbit(s).  The ``backend`` and ``backend_version`` columns
        record which backend produced each orbit.
    fitted_orbit_members : FittedOrbitMembers
        Per-observation table linking each observation to its orbit via
        ``orbit_id``, with residuals and outlier flags where available.

    Raises
    ------
    ValueError
        If *observations* is empty, missing covariances, or missing observer
        states; or if the backend requires a propagator that was not supplied.
    ImportError
        If the requested backend package is not installed, with an install
        command in the error message.
    RuntimeError
        If the backend binary returns an error or produces no orbit.

    Examples
    --------
    Using the native adam backend with ASSIST:

    >>> from adam_core.orbit_determination import fit_orbit
    >>> from adam_core.orbit_determination.config import BackendConfig
    >>> from adam_assist import ASSISTPropagator
    >>> fitted, members = fit_orbit(
    ...     observations,
    ...     backend="adam",
    ...     config=BackendConfig(backend_kwargs={"propagator": ASSISTPropagator}),
    ... )

    Using Find_Orb with default DELEGATE weighting:

    >>> fitted, members = fit_orbit(observations, backend="find_orb")

    Using Find_Orb with adam-managed residuals:

    >>> from adam_core.orbit_determination.config import WeightingPolicy
    >>> fitted, members = fit_orbit(
    ...     observations,
    ...     backend="find_orb",
    ...     config=BackendConfig(
    ...         weighting_policy=WeightingPolicy.ADAM,
    ...         backend_kwargs={"propagator": ASSISTPropagator},
    ...     ),
    ... )

    Notes
    -----
    ``fit_orbit`` is designed for **single-object** orbit determination.  For
    batch processing of many objects, call ``fit_orbit`` in a loop or use
    the parallelism options available in the backend's own pipeline
    (e.g. ``differential_correction`` with Ray).
    """
    if config is None:
        config = BackendConfig()

    # --- Validate backend name -----------------------------------------------
    if backend not in _BACKEND_CLASSES:
        raise ValueError(
            f"Unknown backend {backend!r}.  "
            f"Choose from: {sorted(_BACKEND_CLASSES.keys())}."
        )

    # --- Check availability --------------------------------------------------
    if not _BACKEND_AVAILABLE[backend]:
        install_cmd = _BACKEND_INSTALL_COMMANDS.get(backend, f"pip install {backend}")
        raise ImportError(
            f"The '{backend}' backend is not available in this environment.  "
            f"Install it with:\n\n    {install_cmd}\n"
        )

    # --- Validate observations ------------------------------------------------
    _validate_observations(observations)

    logger.info(
        "fit_orbit: dispatching %d observations to backend '%s'.",
        len(observations),
        backend,
    )

    # --- Dispatch ------------------------------------------------------------
    backend_instance = _BACKEND_CLASSES[backend]()
    fitted_orbits, fitted_members = backend_instance.fit(observations, config)

    logger.info(
        "fit_orbit: backend '%s' returned %d orbit(s) covering %d observation memberships.",
        backend,
        len(fitted_orbits),
        len(fitted_members),
    )

    return fitted_orbits, fitted_members
