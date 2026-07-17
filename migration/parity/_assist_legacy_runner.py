"""Legacy adam_assist oracle runner -- runs inside ``.legacy-assist-venv``.

Invoked by :mod:`migration.parity._assist_oracle` as a subprocess. It reads a
pickled request carrying Arrow-IPC quivr payloads plus kwargs, exercises the
downstream, composition-based ``adam_assist.ASSISTPropagator`` (legacy adam_core
provides the Ray/Python ``Propagator`` composition), and writes a pickled
Arrow-IPC response.

The subprocess isolation is the whole point: the migration repo's adam_core has
deleted the base ``Propagator`` composition, so the legacy reference cannot run
in-process. ``.legacy-assist-venv`` pins legacy adam_core (``4c1fbc4c``) +
downstream ``adam_assist==0.3.9`` + ``assist==1.2.3`` so the legacy behavior is
reproducible and independent of the main runtime -- exactly like the adam_core
parity oracle's ``.legacy-venv``.
"""

from __future__ import annotations

import io
import pickle
import sys
import time
import traceback
from typing import Any

from migration.parity._assist_serde import table_from_ipc, table_to_ipc


def _orbits_cls(name: str) -> Any:
    from adam_core.orbits import Orbits
    from adam_core.orbits.variants import VariantOrbits

    return {"Orbits": Orbits, "VariantOrbits": VariantOrbits}[name]


def _build_propagator() -> Any:
    from adam_assist import ASSISTPropagator

    return ASSISTPropagator()


def _load_orbits(req: dict[str, Any]) -> Any:
    return table_from_ipc(_orbits_cls(req["orbits_cls"]), req["orbits"])


def _run_propagate_orbits(req: dict[str, Any]) -> dict[str, Any]:
    from adam_core.time import Timestamp

    orbits = _load_orbits(req)
    times = table_from_ipc(Timestamp, req["times"])
    out = _build_propagator().propagate_orbits(orbits, times, **req["kwargs"])
    return {"result": table_to_ipc(out), "result_cls": type(out).__name__}


def _run_generate_ephemeris(req: dict[str, Any]) -> dict[str, Any]:
    from adam_core.observers import Observers

    orbits = _load_orbits(req)
    observers = table_from_ipc(Observers, req["observers"])
    out = _build_propagator().generate_ephemeris(orbits, observers, **req["kwargs"])
    return {"result": table_to_ipc(out), "result_cls": type(out).__name__}


def _run_detect(req: dict[str, Any], *, private: bool) -> dict[str, Any]:
    from adam_core.dynamics.impacts import CollisionConditions

    orbits = _load_orbits(req)
    conditions = (
        table_from_ipc(CollisionConditions, req["conditions"])
        if req.get("conditions") is not None
        else None
    )
    prop = _build_propagator()
    if private:
        results, events = prop._detect_collisions(orbits, req["num_days"], conditions)
    else:
        results, events = prop.detect_collisions(
            orbits, req["num_days"], conditions=conditions, **req.get("kwargs", {})
        )
    return {
        "results": table_to_ipc(results),
        "results_cls": type(results).__name__,
        "events": table_to_ipc(events),
    }


_DISPATCH = {
    "propagate_orbits": lambda req: _run_propagate_orbits(req),
    "generate_ephemeris": lambda req: _run_generate_ephemeris(req),
    "detect_collisions": lambda req: _run_detect(req, private=False),
    "_detect_collisions": lambda req: _run_detect(req, private=True),
}


def _run_one(req: dict[str, Any]) -> dict[str, Any]:
    method = req["method"]
    if method not in _DISPATCH:
        raise NotImplementedError(f"Legacy adam_assist oracle has no method {method!r}")
    return _DISPATCH[method](req)


def _handle(req: dict[str, Any]) -> dict[str, Any]:
    mode = req.get("mode", "parity")
    if mode == "parity":
        return {"ok": True, **_run_one(req)}
    if mode == "time":
        warmup = int(req.get("warmup", 1))
        reps = int(req.get("reps", 5))
        # Match the local Rust timing helper: one unmeasured priming call, then
        # explicit warmups, then measured reps.
        _run_one(req)
        for _ in range(warmup):
            _run_one(req)
        elapsed: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _run_one(req)
            elapsed.append(time.perf_counter() - t0)
        return {"ok": True, "elapsed": elapsed}
    raise ValueError(f"Unknown mode {mode!r}")


def main() -> int:
    try:
        request = pickle.load(sys.stdin.buffer)
        response = _handle(request)
    except Exception as exc:  # noqa: BLE001 - report to the parent process
        response = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    buf = io.BytesIO()
    pickle.dump(response, buf, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(buf.getvalue())
    sys.stdout.buffer.flush()
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
