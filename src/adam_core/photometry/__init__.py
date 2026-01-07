# flake8: noqa: F401
from .simple_magnitude import (
    InstrumentFilters,
    StandardFilters,
    calculate_apparent_magnitude_v,
    calculate_apparent_magnitude_v_jax,
    convert_magnitude,
    convert_magnitude_jax,
    find_conversion_path,
    predict_magnitudes,
    predict_magnitudes_jax,
)

__all__ = [
    # Simple magnitude system
    "StandardFilters",
    "InstrumentFilters", 
    "calculate_apparent_magnitude_v",
    "calculate_apparent_magnitude_v_jax",
    "convert_magnitude",
    "convert_magnitude_jax",
    "find_conversion_path",
    "predict_magnitudes",
    "predict_magnitudes_jax",
]
