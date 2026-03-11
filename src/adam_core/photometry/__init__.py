# flake8: noqa: F401
from .absolute_magnitude import (
    GroupedPhysicalParameters,
    estimate_absolute_magnitude_v_from_detections,
    estimate_absolute_magnitude_v_from_detections_grouped,
)
from .magnitude import (
    calculate_apparent_magnitude_v,
    calculate_apparent_magnitude_v_and_phase_angle,
    calculate_phase_angle,
    convert_magnitude,
    predict_magnitudes,
)

__all__ = [
    # Simple magnitude system
    "calculate_apparent_magnitude_v",
    "calculate_apparent_magnitude_v_and_phase_angle",
    "calculate_phase_angle",
    "convert_magnitude",
    "predict_magnitudes",
    # Inverse magnitude system
    "estimate_absolute_magnitude_v_from_detections",
    "estimate_absolute_magnitude_v_from_detections_grouped",
    "GroupedPhysicalParameters",
]
