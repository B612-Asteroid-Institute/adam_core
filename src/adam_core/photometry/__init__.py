# flake8: noqa: F401
from .absolute_magnitude import (
    GroupedPhysicalParameters,
    estimate_absolute_magnitude_v_from_detections,
    estimate_absolute_magnitude_v_from_detections_grouped,
)
from .lightcurve import reduced_magnitude
from .magnitude import (
    calculate_apparent_magnitude_v,
    calculate_apparent_magnitude_v_and_phase_angle,
    calculate_phase_angle,
    convert_magnitude,
    predict_magnitudes,
)
from .magnitude_common import hg_phase_correction
from .rotation_period_fourier import estimate_rotation_period
from .rotation_period_types import (
    GroupedRotationPeriodResults,
    RotationPeriodObservations,
    RotationPeriodResult,
)
from .rotation_period_wrappers import (
    build_rotation_period_observations_from_detections,
    estimate_rotation_period_from_detections,
    estimate_rotation_period_from_detections_grouped,
)

__all__ = [
    # Simple magnitude system
    "calculate_apparent_magnitude_v",
    "calculate_apparent_magnitude_v_and_phase_angle",
    "calculate_phase_angle",
    "convert_magnitude",
    "predict_magnitudes",
    # Shared lightcurve photometric reductions
    "hg_phase_correction",
    "reduced_magnitude",
    # Inverse magnitude system
    "estimate_absolute_magnitude_v_from_detections",
    "estimate_absolute_magnitude_v_from_detections_grouped",
    "GroupedPhysicalParameters",
    # Rotation-period analysis
    "build_rotation_period_observations_from_detections",
    "estimate_rotation_period",
    "estimate_rotation_period_from_detections",
    "estimate_rotation_period_from_detections_grouped",
    "RotationPeriodObservations",
    "RotationPeriodResult",
    "GroupedRotationPeriodResults",
]
