"""Shared per-observation photometric reductions for lightcurve analysis.

Small, dependency-light helpers used to turn raw apparent magnitudes into the
distance-/phase-reduced quantities that downstream lightcurve work (rotation-period
estimation, color estimation) operates on. Kept separate from the H-G magnitude
machinery in ``magnitude.py`` so callers that only need the reduction do not pull in
the full apparent-magnitude stack.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def reduced_magnitude(
    mag: npt.NDArray[np.float64],
    r_au: npt.NDArray[np.float64],
    delta_au: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Distance-reduced magnitude ``mag - 5 * log10(r_au * delta_au)``.

    Removes the heliocentric (``r_au``) and observer (``delta_au``) distance
    dependence so that the residual variation reflects the object's intrinsic
    brightness (rotation, color), not its changing geometry. Phase-angle dependence
    is handled separately by the H-G phase correction.
    """
    mag = np.asarray(mag, dtype=np.float64)
    r_au = np.asarray(r_au, dtype=np.float64)
    delta_au = np.asarray(delta_au, dtype=np.float64)
    return np.asarray(mag - 5.0 * np.log10(r_au * delta_au), dtype=np.float64)
