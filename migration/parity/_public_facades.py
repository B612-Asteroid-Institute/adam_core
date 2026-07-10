"""Runtime-neutral builders for parity cases that exercise public table facades.

These helpers import ``adam_core`` only inside each call, so the current runner
and pinned-legacy subprocess execute the same facade construction against their
own active runtime without duplicating orchestration in the benchmark harness.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
