"""Runtime-neutral builders for parity cases that exercise public table facades.

These helpers import ``adam_core`` only inside each call, so the current runner
and pinned-legacy subprocess execute the same facade construction against their
own active runtime without duplicating orchestration in the benchmark harness.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_EPHEMERIS_INPUT_CACHE_KEY: tuple[Any, ...] | None = None
_EPHEMERIS_INPUT_CACHE_VALUE: tuple[Any, Any] | None = None


def build_propagate_2body_inputs(
    *,
    coords: Any,
    epoch_mjd: Any,
    target_mjd: Any,
    origin: str,
    frame: str,
) -> tuple[Any, Any]:
    """Build runtime-local Orbits and target Timestamp facade inputs."""
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    values = np.asarray(coords, dtype=np.float64)
    rows = values.shape[0]
    orbits = Orbits.from_kwargs(
        orbit_id=[str(index) for index in range(rows)],
        object_id=[str(index) for index in range(rows)],
        coordinates=CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=Timestamp.from_mjd(np.asarray(epoch_mjd), scale="tdb"),
            origin=Origin.from_kwargs(code=np.full(rows, str(origin), dtype="object")),
            frame=str(frame),
        ),
    )
    targets = Timestamp.from_mjd(np.asarray(target_mjd), scale="tdb")
    return orbits, targets


def propagate_2body(
    *,
    max_iter: int,
    tol: float,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Build Orbits, call public ``propagate_2body``, and return state values."""
    from adam_core.dynamics import propagate_2body as public_propagate_2body

    orbits, targets = build_propagate_2body_inputs(**kwargs)
    propagated = public_propagate_2body(
        orbits,
        targets,
        max_iter=max_iter,
        tol=tol,
        max_processes=1,
    )
    return {"out": np.asarray(propagated.coordinates.values, dtype=np.float64)}


def build_generate_ephemeris_inputs(
    *,
    orbits: Any,
    observer_states: Any,
    epoch_mjd: Any,
    covariance: Any | None = None,
    covariances: Any | None = None,
    **_unused: Any,
) -> tuple[Any, Any]:
    """Build or reuse runtime-local paired Orbits and Observers tables.

    Timing loops repeatedly invoke the public API with the same logical input
    tables. Keep those tables outside timed calls, matching ordinary callers
    that already own ``Orbits``/``Observers`` rather than charging table setup
    to each API invocation.
    """
    global _EPHEMERIS_INPUT_CACHE_KEY, _EPHEMERIS_INPUT_CACHE_VALUE
    cache_key = (orbits, observer_states, epoch_mjd, covariance, covariances)
    if (
        _EPHEMERIS_INPUT_CACHE_KEY is not None
        and all(
            current is cached
            for current, cached in zip(cache_key, _EPHEMERIS_INPUT_CACHE_KEY)
        )
        and _EPHEMERIS_INPUT_CACHE_VALUE is not None
    ):
        return _EPHEMERIS_INPUT_CACHE_VALUE

    from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances
    from adam_core.coordinates.origin import Origin
    from adam_core.observers import Observers
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    orbit_values = np.asarray(orbits, dtype=np.float64)
    observer_values = np.asarray(observer_states, dtype=np.float64)
    rows = orbit_values.shape[0]
    times = Timestamp.from_mjd(np.asarray(epoch_mjd, dtype=np.float64), scale="tdb")
    origins = Origin.from_kwargs(code=np.full(rows, "SUN", dtype="object"))
    covariance_values = covariance if covariance is not None else covariances
    orbit_covariance = None
    if covariance_values is not None:
        orbit_covariance = CoordinateCovariances.from_matrix(
            np.asarray(covariance_values, dtype=np.float64).reshape(rows, 6, 6)
        )
    orbit_table = Orbits.from_kwargs(
        orbit_id=[str(index) for index in range(rows)],
        object_id=[str(index) for index in range(rows)],
        coordinates=CartesianCoordinates.from_kwargs(
            x=orbit_values[:, 0],
            y=orbit_values[:, 1],
            z=orbit_values[:, 2],
            vx=orbit_values[:, 3],
            vy=orbit_values[:, 4],
            vz=orbit_values[:, 5],
            time=times,
            covariance=orbit_covariance,
            origin=origins,
            frame="ecliptic",
        ),
    )
    observer_table = Observers.from_kwargs(
        code=["500"] * rows,
        coordinates=CartesianCoordinates.from_kwargs(
            x=observer_values[:, 0],
            y=observer_values[:, 1],
            z=observer_values[:, 2],
            vx=observer_values[:, 3],
            vy=observer_values[:, 4],
            vz=observer_values[:, 5],
            time=times,
            origin=origins,
            frame="ecliptic",
        ),
    )
    _EPHEMERIS_INPUT_CACHE_KEY = cache_key
    _EPHEMERIS_INPUT_CACHE_VALUE = (orbit_table, observer_table)
    return orbit_table, observer_table


def generate_ephemeris_2body(
    *,
    covariance: Any | None = None,
    covariances: Any | None = None,
    lt_tol: float,
    max_iter: int,
    tol: float,
    stellar_aberration: bool,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Build paired tables and call the public two-body ephemeris facade."""
    from adam_core.dynamics import generate_ephemeris_2body as public_ephemeris

    orbit_table, observer_table = build_generate_ephemeris_inputs(
        covariance=covariance,
        covariances=covariances,
        **kwargs,
    )
    ephemeris = public_ephemeris(
        orbit_table,
        observer_table,
        lt_tol=lt_tol,
        max_iter=max_iter,
        tol=tol,
        stellar_aberration=stellar_aberration,
        predict_magnitudes=False,
        max_processes=1,
    )
    result = {
        "spherical": np.asarray(ephemeris.coordinates.values, dtype=np.float64),
        "light_time": np.asarray(ephemeris.light_time, dtype=np.float64),
        "aberrated_state": np.asarray(
            ephemeris.aberrated_coordinates.values, dtype=np.float64
        ),
    }
    if covariance is not None or covariances is not None:
        result["covariance"] = np.asarray(
            ephemeris.coordinates.covariance.to_matrix(), dtype=np.float64
        )
    return result
