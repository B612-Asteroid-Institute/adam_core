"""
Python side of the Rust orbit-determination callback route.

When a propagator has no native Rust work units, the Rust drivers (fit,
rejection loops, IOD, full OD) still run the orchestration: every time they
need predicted positions they call `predict_spherical` here, which asks the
propagator's own ``generate_ephemeris`` for them. This module only converts
tables and reorders rows; it holds no orbit-determination logic.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import numpy.typing as npt

from ..coordinates import SphericalCoordinates, transform_coordinates
from ..observers import Observers
from ..orbits.arrow_bridge import observers_from_ipc, orbits_from_ipc
from ..time import Timestamp

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
    once per candidate instead. Predictions are matched by orbit id, epoch
    and observer code, independently of the backend's row ordering. Exact
    epochs are preferred; unambiguous float-MJD round trips are also accepted
    for backends that store epochs that way. Missing or ambiguous observer
    matches raise ``ValueError``.
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
        blocks = _blocks_by_candidate(ephemeris, ids, observers)
    else:
        blocks = [
            _blocks_by_candidate(
                propagator.generate_ephemeris(
                    orbits.take([k]), observers, max_processes=1
                ),
                [ids[k]],
                observers,
            )[0]
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


def _observer_keys(times: Timestamp, codes: list[str]) -> list[tuple[int, int, str]]:
    """Use integer epochs: float MJD would merge distinct observation times."""
    keys = list(zip(times.days.to_pylist(), times.nanos.to_pylist(), codes))
    if any(day is None or nano is None or code is None for day, nano, code in keys):
        raise ValueError("Cannot match an observer with a null epoch or station code")
    return keys


def _blocks_by_candidate(
    ephemeris, ids: list[str], observers: Observers
) -> list[npt.NDArray[np.float64]]:
    """Restore candidate-major, input-observer order using ephemeris metadata.

    Duplicate requests for the same station and epoch are interchangeable
    only when their observer states agree and the backend predictions agree.
    Ephemeris carries no observer row id to distinguish conflicting duplicates.
    """
    n_observers = len(observers)
    if len(ephemeris) != len(ids) * n_observers:
        raise ValueError(
            f"generate_ephemeris returned {len(ephemeris)} rows for {len(ids)} "
            f"orbits and {n_observers} observers, expected {len(ids) * n_observers}"
        )
    # Rescale the requests to the backend's output scale, just as
    # Ephemeris.link_to_observers does. Prefer exact integer epochs.
    expected_times = observers.coordinates.time.rescale(
        ephemeris.coordinates.time.scale
    )
    codes = observers.code.to_pylist()
    expected_keys = _observer_keys(expected_times, codes)
    states = observers.coordinates.values
    origins = observers.coordinates.origin.code.to_pylist()
    first_observer: dict[tuple[int, int, str], int] = {}
    for row, key in enumerate(expected_keys):
        first = first_observer.setdefault(key, row)
        if first != row and (
            origins[first] != origins[row]
            or not np.array_equal(states[first], states[row], equal_nan=True)
        ):
            raise ValueError(
                f"Cannot match ambiguous observer {key!r}: "
                "the same station and epoch have different observer states"
            )

    values = _equatorial_values(ephemeris)
    actual_keys = _observer_keys(
        ephemeris.coordinates.time, ephemeris.coordinates.origin.code.to_pylist()
    )
    if set(actual_keys) != set(expected_keys):
        # Legacy backends such as PYOORB store a single float MJD, losing
        # sub-microsecond precision. Accept only that specific round trip,
        # rather than rounding all matches or using a nearest-time tolerance.
        rounded_times = Timestamp.from_mjd(
            expected_times.mjd(), scale=expected_times.scale
        )
        rounded_keys = _observer_keys(rounded_times, codes)
        if set(actual_keys) == set(rounded_keys):
            original_by_rounded: dict[tuple[int, int, str], tuple[int, int, str]] = {}
            for original, rounded in zip(expected_keys, rounded_keys):
                if original_by_rounded.setdefault(rounded, original) != original:
                    raise ValueError(
                        f"Cannot match ambiguous observer {rounded!r}: "
                        "distinct requested epochs collapse to the same float MJD"
                    )
            expected_keys = rounded_keys

    row_indices: dict[tuple[str, int, int, str], deque[int]] = defaultdict(deque)
    for row, (orbit_id, key) in enumerate(
        zip(ephemeris.orbit_id.to_pylist(), actual_keys)
    ):
        indices = row_indices[(orbit_id, *key)]
        if indices and not np.array_equal(
            values[indices[0]], values[row], equal_nan=True
        ):
            raise ValueError(
                f"generate_ephemeris returned ambiguous predictions for observer "
                f"{key!r} of orbit {orbit_id!r}"
            )
        indices.append(row)

    blocks = []
    for orbit_id in ids:
        order = []
        for key in expected_keys:
            indices = row_indices.get((orbit_id, *key))
            if not indices:
                raise ValueError(
                    f"generate_ephemeris has no matching row for observer "
                    f"{key!r} of orbit {orbit_id!r}; expected the same "
                    "station codes, epochs and duplicate counts as the request"
                )
            order.append(indices.popleft())
        blocks.append(values[order])
    return blocks
