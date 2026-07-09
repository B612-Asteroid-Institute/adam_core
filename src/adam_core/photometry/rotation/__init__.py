"""Rotation-period estimation with a measured confidence verdict.

Recovers an asteroid's rotation period from sparse, multi-band photometry and
returns a categorical confidence verdict rather than a bare number. See
:func:`estimate_rotation_period` and the detection-level wrappers.

The public API is re-exported through ``adam_core.photometry``; the imports below
are the organizational entry point for this subpackage.
"""

# ruff: noqa: F401  (organizational re-exports; public surface is adam_core.photometry)
from .core import (
    GroupedRotationPeriodResults,
    RotationPeriodObservations,
    RotationPeriodResult,
)
from .estimator import estimate_rotation_period
from .wrappers import (
    build_rotation_period_observations_from_detections,
    estimate_rotation_period_best_apparition,
    estimate_rotation_period_from_detections,
    estimate_rotation_period_from_detections_grouped,
)
