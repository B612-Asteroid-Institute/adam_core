"""Shared public porkchop orchestration adapter for parity runners."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np


@contextmanager
def _plotly_import_scope() -> Iterator[None]:
    """Let the baseline non-plotting porkchop path import without plot extras."""
    try:
        import plotly.graph_objects  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "plotly":
            raise
    else:
        yield
        return

    import sys
    import types

    had_plotly = "plotly" in sys.modules
    had_graph_objects = "plotly.graph_objects" in sys.modules
    previous_plotly = sys.modules.get("plotly")
    previous_graph_objects = sys.modules.get("plotly.graph_objects")

    plotly = types.ModuleType("plotly")
    graph_objects = types.ModuleType("plotly.graph_objects")
    plotly.graph_objects = graph_objects
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = graph_objects
    try:
        yield
    finally:
        if had_plotly:
            sys.modules["plotly"] = previous_plotly
        else:
            sys.modules.pop("plotly", None)
        if had_graph_objects:
            sys.modules["plotly.graph_objects"] = previous_graph_objects
        else:
            sys.modules.pop("plotly.graph_objects", None)


def _index_by_id(orbit_ids: np.ndarray, label: str) -> dict[str, int]:
    index_by_id: dict[str, int] = {}
    for index, orbit_id in enumerate(orbit_ids):
        key = str(orbit_id)
        if key in index_by_id:
            raise ValueError(f"duplicate {label} orbit_id {key!r}")
        index_by_id[key] = index
    return index_by_id


def _indices_for_ids(
    orbit_ids: np.ndarray,
    index_by_id: Mapping[str, int],
    label: str,
) -> np.ndarray:
    try:
        return np.asarray(
            [index_by_id[str(value)] for value in orbit_ids], dtype=np.float64
        )
    except KeyError as exc:
        raise ValueError(f"unknown {label} orbit_id {exc.args[0]!r}") from exc


def _pack_porkchop_solutions(
    solutions: Any,
    departure_index_by_id: Mapping[str, int],
    arrival_index_by_id: Mapping[str, int],
) -> dict[str, np.ndarray]:
    departure_ids = np.asarray(solutions.departure_body_id.to_pylist(), dtype=object)
    arrival_ids = np.asarray(solutions.arrival_body_id.to_pylist(), dtype=object)
    departure_index = _indices_for_ids(
        departure_ids, departure_index_by_id, "departure"
    )
    arrival_index = _indices_for_ids(arrival_ids, arrival_index_by_id, "arrival")
    order = np.lexsort((arrival_index, departure_index))
    solution_departure_velocity = np.asarray(
        solutions.table.select(
            [
                "solution_departure_vx",
                "solution_departure_vy",
                "solution_departure_vz",
            ]
        ),
        dtype=np.float64,
    )
    solution_arrival_velocity = np.asarray(
        solutions.table.select(
            ["solution_arrival_vx", "solution_arrival_vy", "solution_arrival_vz"]
        ),
        dtype=np.float64,
    )
    return {
        "departure_index": departure_index[order],
        "arrival_index": arrival_index[order],
        "departure_time_mjd": np.asarray(
            solutions.departure_time.mjd().to_numpy(False), dtype=np.float64
        )[order],
        "arrival_time_mjd": np.asarray(
            solutions.arrival_time.mjd().to_numpy(False), dtype=np.float64
        )[order],
        "solution_departure_velocity": solution_departure_velocity[order],
        "solution_arrival_velocity": solution_arrival_velocity[order],
        "c3_departure": np.asarray(solutions.c3_departure(), dtype=np.float64)[order],
        "vinf_arrival": np.asarray(solutions.vinf_arrival(), dtype=np.float64)[order],
        "time_of_flight": np.asarray(solutions.time_of_flight(), dtype=np.float64)[
            order
        ],
    }


def run_generate_porkchop_data(
    departure_coords: np.ndarray,
    arrival_coords: np.ndarray,
    departure_time_mjd: np.ndarray,
    arrival_time_mjd: np.ndarray,
    departure_orbit_ids: np.ndarray,
    arrival_orbit_ids: np.ndarray,
    propagation_origin: str,
    frame: str,
    prograde: bool,
    max_iter: int,
    tol: float,
    max_processes: int,
) -> dict[str, np.ndarray]:
    departure_index_by_id = _index_by_id(departure_orbit_ids, "departure")
    arrival_index_by_id = _index_by_id(arrival_orbit_ids, "arrival")

    with _plotly_import_scope():
        from adam_core.coordinates.cartesian import CartesianCoordinates
        from adam_core.coordinates.origin import Origin, OriginCodes
        from adam_core.missions.porkchop import generate_porkchop_data
        from adam_core.orbits import Orbits
        from adam_core.time import Timestamp

        n_departures = departure_coords.shape[0]
        n_arrivals = arrival_coords.shape[0]
        departure_coordinates = CartesianCoordinates.from_kwargs(
            x=departure_coords[:, 0],
            y=departure_coords[:, 1],
            z=departure_coords[:, 2],
            vx=departure_coords[:, 3],
            vy=departure_coords[:, 4],
            vz=departure_coords[:, 5],
            time=Timestamp.from_mjd(departure_time_mjd, scale="tdb"),
            origin=Origin.from_kwargs(
                code=np.full(n_departures, propagation_origin, dtype=object)
            ),
            frame=frame,
        )
        arrival_coordinates = CartesianCoordinates.from_kwargs(
            x=arrival_coords[:, 0],
            y=arrival_coords[:, 1],
            z=arrival_coords[:, 2],
            vx=arrival_coords[:, 3],
            vy=arrival_coords[:, 4],
            vz=arrival_coords[:, 5],
            time=Timestamp.from_mjd(arrival_time_mjd, scale="tdb"),
            origin=Origin.from_kwargs(
                code=np.full(n_arrivals, propagation_origin, dtype=object)
            ),
            frame=frame,
        )
        departure_orbits = Orbits.from_kwargs(
            orbit_id=departure_orbit_ids, coordinates=departure_coordinates
        )
        arrival_orbits = Orbits.from_kwargs(
            orbit_id=arrival_orbit_ids, coordinates=arrival_coordinates
        )
        solutions = generate_porkchop_data(
            departure_orbits,
            arrival_orbits,
            propagation_origin=OriginCodes[str(propagation_origin)],
            prograde=prograde,
            max_iter=max_iter,
            tol=tol,
            max_processes=max_processes,
        )
    return _pack_porkchop_solutions(
        solutions, departure_index_by_id, arrival_index_by_id
    )
