# flake8: noqa: F401
from .simple_magnitude import (
    InstrumentFilters,
    StandardFilters,
    calculate_apparent_magnitude,
    convert_magnitude,
    find_conversion_path,
    predict_magnitudes,
)

__all__ = [
    # Simple magnitude system
    "StandardFilters",
    "InstrumentFilters", 
    "calculate_apparent_magnitude",
    "convert_magnitude",
    "find_conversion_path",
    "predict_magnitudes",
]
