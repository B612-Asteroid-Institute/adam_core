"""
Python side of the Rust orbit-determination callback route.

When a propagator has no native Rust work units, the Rust drivers (fit,
rejection loops, IOD, full OD) still run the orchestration: every time they
need predicted positions they call `predict_spherical` here, which asks the
propagator's own ``generate_ephemeris`` for them. This module only converts
tables and reorders rows; it holds no orbit-determination logic.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..coordinates import SphericalCoordinates, transform_coordinates
from ..orbits.arrow_bridge import observers_from_ipc, orbits_from_ipc

__all__ = ["predict_spherical"]


def predict_spherical(
    propagator: object,
    orbits_ipc: bytes,
    observers_ipc: bytes,
    n_candidates: int,
    n_observers: int,
) -> npt.NDArray[np.float64]:
    """
    Predicted topocentric spherical coordinates of every candidate orbit at
    every observer, ``(n_candidates * n_observers, 6)`` in candidate order
    then observer order, from ``propagator.generate_ephemeris``.

    Candidates are handed to ``generate_ephemeris`` in one call; a propagator
    that only handles one orbit per call (a duck-typed test double) is called
    once per candidate instead.
    """
    orbits = orbits_from_ipc(orbits_ipc)
    observers = observers_from_ipc(observers_ipc)
    if len(orbits) != n_candidates or len(observers) != n_observers:
        raise ValueError(
            f"callback received {len(orbits)} candidates and {len(observers)} "
            f"observers, expected {n_candidates} and {n_observers}"
        )
    ids = orbits.orbit_id.to_pylist()
    out = np.empty((n_candidates * n_observers, 6), dtype=np.float64)

    ephemeris = propagator.generate_ephemeris(orbits, observers, max_processes=1)
    if len(ephemeris) == n_candidates * n_observers:
        blocks = _blocks_by_candidate(ephemeris, ids, n_observers)
    else:
        blocks = [
            _single_block(
                propagator.generate_ephemeris(
                    orbits.take([k]), observers, max_processes=1
                ),
                n_observers,
            )
            for k in range(n_candidates)
        ]
    for k, block in enumerate(blocks):
        out[k * n_observers : (k + 1) * n_observers] = block
    return np.ascontiguousarray(out)


def _equatorial_values(ephemeris) -> npt.NDArray[np.float64]:
    coordinates = ephemeris.coordinates
    if coordinates.frame != "equatorial":
        coordinates = transform_coordinates(
            coordinates, SphericalCoordinates, frame_out="equatorial"
        )
    return np.asarray(coordinates.values, dtype=np.float64)


def _single_block(ephemeris, n_observers: int) -> npt.NDArray[np.float64]:
    if len(ephemeris) != n_observers:
        raise ValueError(
            f"generate_ephemeris returned {len(ephemeris)} rows for one orbit, "
            f"expected {n_observers}"
        )
    return _equatorial_values(ephemeris)


def _blocks_by_candidate(
    ephemeris, ids: list[str], n_observers: int
) -> list[npt.NDArray[np.float64]]:
    values = _equatorial_values(ephemeris)
    row_ids = np.asarray(ephemeris.orbit_id.to_numpy(zero_copy_only=False))
    blocks = []
    for orbit_id in ids:
        rows = values[row_ids == orbit_id]
        if len(rows) != n_observers:
            raise ValueError(
                f"generate_ephemeris returned {len(rows)} rows for orbit "
                f"{orbit_id!r}, expected {n_observers}"
            )
        blocks.append(rows)
    return blocks
