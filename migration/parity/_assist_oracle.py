"""Subprocess oracle for legacy adam_assist parity -- main-venv side.

This mirrors :mod:`migration.parity._oracle` (the adam_core legacy oracle) but
targets a second isolated runtime, ``.legacy-assist-venv``, which pins legacy
adam_core (composition) + downstream ``adam_assist``. It exposes
:class:`LegacyAssistPropagator`, a drop-in proxy with the same
``propagate_orbits`` / ``generate_ephemeris`` / ``detect_collisions`` /
``_detect_collisions`` surface as ``adam_assist.ASSISTPropagator``. Each call
serializes its quivr inputs to Arrow IPC, runs the legacy propagator in the
isolated runtime, and reconstructs the result under the main runtime's
adam_core -- so parity tests compare ``adam_assist_rust.ASSISTPropagator``
against the legacy reference across two separate runtimes, exactly like the
adam_core parity gate.

Results are cached on disk (keyed by a stable hash of the request) so the
expensive legacy ASSIST integrations are serialized once and reused; set
``ADAM_CORE_ASSIST_PARITY_REFRESH=1`` to force re-running the legacy runtime.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

from ._assist_serde import table_from_ipc, table_to_ipc

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ASSIST_VENV_PYTHON = Path(
    os.environ.get(
        "ADAM_CORE_LEGACY_ASSIST_VENV_PYTHON",
        str(REPO_ROOT / ".legacy-assist-venv" / "bin" / "python"),
    )
)
CACHE_DIR = Path(
    os.environ.get(
        "ADAM_CORE_ASSIST_PARITY_CACHE_DIR",
        str(REPO_ROOT / "migration" / "artifacts" / "assist_parity_cache"),
    )
)

_BOOTSTRAP = (
    "import pickle, sys; "
    "sys.path.insert(0, %r); "
    "from migration.parity._assist_legacy_runner import _handle; "
    "req = pickle.load(sys.stdin.buffer); "
    "resp = _handle(req); "
    "sys.stdout.buffer.write(pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL))"
)


class LegacyAssistOracleError(RuntimeError):
    """Raised when the legacy adam_assist subprocess fails or its venv is missing."""


def _ensure_venv() -> None:
    if not LEGACY_ASSIST_VENV_PYTHON.exists():
        raise LegacyAssistOracleError(
            f"Legacy adam_assist oracle Python not found at "
            f"{LEGACY_ASSIST_VENV_PYTHON}. Build it per migration/parity/README:\n"
            f"  python3.13 -m venv .legacy-assist-venv\n"
            f"  .legacy-assist-venv/bin/pip install assist==1.2.3 rebound\n"
            f"  .legacy-assist-venv/bin/pip install -e "
            f"/Users/aleck/Code/adam-core-legacy-main\n"
            f"  .legacy-assist-venv/bin/pip install adam-assist==0.3.9 --no-deps"
        )


def _cache_key(request: dict[str, Any]) -> str:
    payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def _run(request: dict[str, Any]) -> dict[str, Any]:
    refresh = os.environ.get("ADAM_CORE_ASSIST_PARITY_REFRESH") == "1"
    key = _cache_key(request)
    cache_path = CACHE_DIR / f"{key}.pkl"
    if not refresh and cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    _ensure_venv()
    payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [str(LEGACY_ASSIST_VENV_PYTHON), "-c", _BOOTSTRAP % str(REPO_ROOT)],
        input=payload,
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise LegacyAssistOracleError(
            f"Legacy adam_assist subprocess exited {proc.returncode}.\n"
            f"stderr:\n{proc.stderr.decode(errors='replace')}"
        )
    try:
        response = pickle.load(io.BytesIO(proc.stdout))
    except Exception as exc:  # noqa: BLE001
        raise LegacyAssistOracleError(
            f"Could not deserialize legacy response: {exc}\n"
            f"stderr:\n{proc.stderr.decode(errors='replace')}"
        ) from exc
    if not response.get("ok"):
        raise LegacyAssistOracleError(
            f"Legacy adam_assist oracle raised: {response.get('error')}\n"
            f"{response.get('traceback', '')}"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(response, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    return response


def _result_table(response: dict[str, Any]) -> Any:
    from adam_core.orbits import Orbits
    from adam_core.orbits.ephemeris import Ephemeris
    from adam_core.orbits.variants import VariantEphemeris, VariantOrbits

    registry = {
        "Orbits": Orbits,
        "VariantOrbits": VariantOrbits,
        "Ephemeris": Ephemeris,
        "VariantEphemeris": VariantEphemeris,
    }
    return table_from_ipc(registry[response["result_cls"]], response["result"])


def _detect_result(response: dict[str, Any]) -> tuple[Any, Any]:
    from adam_core.dynamics.impacts import CollisionEvent
    from adam_core.orbits import Orbits
    from adam_core.orbits.variants import VariantOrbits

    results_cls = {"Orbits": Orbits, "VariantOrbits": VariantOrbits}[
        response["results_cls"]
    ]
    results = table_from_ipc(results_cls, response["results"])
    events = table_from_ipc(CollisionEvent, response["events"])
    return results, events


class LegacyAssistPropagator:
    """Drop-in proxy for downstream ``adam_assist.ASSISTPropagator`` executed in
    the isolated ``.legacy-assist-venv`` legacy runtime."""

    def propagate_orbits(self, orbits: Any, times: Any, **kwargs: Any) -> Any:
        response = _run(
            {
                "method": "propagate_orbits",
                "orbits": table_to_ipc(orbits),
                "orbits_cls": type(orbits).__name__,
                "times": table_to_ipc(times),
                "kwargs": kwargs,
            }
        )
        return _result_table(response)

    def generate_ephemeris(self, orbits: Any, observers: Any, **kwargs: Any) -> Any:
        response = _run(
            {
                "method": "generate_ephemeris",
                "orbits": table_to_ipc(orbits),
                "orbits_cls": type(orbits).__name__,
                "observers": table_to_ipc(observers),
                "kwargs": kwargs,
            }
        )
        return _result_table(response)

    def detect_collisions(
        self, orbits: Any, num_days: Any, conditions: Any = None, **kwargs: Any
    ) -> tuple[Any, Any]:
        response = _run(
            {
                "method": "detect_collisions",
                "orbits": table_to_ipc(orbits),
                "orbits_cls": type(orbits).__name__,
                "num_days": num_days,
                "conditions": None if conditions is None else table_to_ipc(conditions),
                "kwargs": kwargs,
            }
        )
        return _detect_result(response)

    def _detect_collisions(
        self, orbits: Any, num_days: Any, conditions: Any
    ) -> tuple[Any, Any]:
        response = _run(
            {
                "method": "_detect_collisions",
                "orbits": table_to_ipc(orbits),
                "orbits_cls": type(orbits).__name__,
                "num_days": num_days,
                "conditions": table_to_ipc(conditions),
            }
        )
        return _detect_result(response)
