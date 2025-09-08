"""
Anomaly labeling data structures and functions.

This module provides quivr.Table definitions for anomaly labels and functions
to convert detection-orbit hits into orbital anomaly assignments.
"""

from __future__ import annotations

import quivr as qv

__all__ = [
    "AnomalyLabels",
]


class AnomalyLabels(qv.Table):
    """
    Anomaly labels for detection-orbit hits.
    
    Each row represents one anomaly variant for a detection-orbit pairing.
    Multiple variants can exist for the same hit (ambiguous cases near nodes).
    
    Sorted by: (det_id, orbit_id, variant_id, snap_error)
    """
    
    # Hit identification
    det_id = qv.StringColumn()
    orbit_id = qv.StringColumn() 
    seg_id = qv.Int32Column()
    variant_id = qv.Int32Column()  # 0, 1, 2, ... for multiple variants per hit
    
    # Anomaly values at epoch
    f_rad = qv.Float64Column()  # True anomaly (radians)
    E_rad = qv.Float64Column()  # Eccentric anomaly (radians) 
    M_rad = qv.Float64Column()  # Mean anomaly (radians)
    n_rad_day = qv.Float64Column()  # Mean motion (radians/day)
    r_au = qv.Float64Column()   # Heliocentric distance (AU)
    
    # Quality metrics
    snap_error = qv.Float64Column()      # Residual from ellipse fit
    plane_distance_au = qv.Float64Column()  # Distance from orbital plane (AU)
    
    @classmethod
    def empty(cls) -> AnomalyLabels:
        """Create an empty AnomalyLabels table."""
        return cls.from_kwargs(
            det_id=[],
            orbit_id=[],
            seg_id=[],
            variant_id=[],
            f_rad=[],
            E_rad=[],
            M_rad=[],
            n_rad_day=[],
            r_au=[],
            snap_error=[],
            plane_distance_au=[],
        )
