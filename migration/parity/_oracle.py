"""Subprocess oracle helper — main-venv side.

This module is imported by the parity gate scripts. It invokes
``.legacy-venv/bin/python -m migration.parity._legacy_runner`` as a
subprocess, sends a pickled request on stdin, and reads a pickled
response from stdout.

We do NOT batch invocations across calls — each call to ``parity()`` /
``time_legacy()`` spawns its own subprocess. Subprocess startup cost
(~150 ms for Python+JAX import) is amortized two ways:
1. The fuzz parity gate samples once per (api_id, seed) — startup is
   absorbed into the per-seed budget.
2. The speed gate runs ``reps`` timing loops INSIDE the subprocess, so
   subprocess overhead is excluded from latency measurements.
"""

from __future__ import annotations

import io
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import _threading

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_VENV_PYTHON = Path(
    os.environ.get(
        "ADAM_CORE_LEGACY_VENV_PYTHON",
        str(REPO_ROOT / ".legacy-venv" / "bin" / "python"),
    )
)


class LegacyOracleError(RuntimeError):
    """Raised when the legacy subprocess fails or the legacy venv is missing."""


def _ensure_legacy_venv() -> None:
    if not LEGACY_VENV_PYTHON.exists():
        raise LegacyOracleError(
            f"Legacy oracle Python not found at {LEGACY_VENV_PYTHON}. "
            f"Set up the legacy venv per migration/parity/README — typically:\n"
            f"  python3.13 -m venv .legacy-venv\n"
            f"  .legacy-venv/bin/pip install -e /Users/aleck/Code/adam-core"
        )


def _subprocess_env(thread_mode: str | None = None) -> dict[str, str]:
    env = _threading.env_for_thread_mode(thread_mode, os.environ)
    # Make sure the legacy venv's site-packages take precedence; clear any
    # PYTHONPATH that might leak the migration repo's adam_core.
    env.pop("PYTHONPATH", None)
    return env


def _run_subprocess(
    request: dict[str, Any], *, thread_mode: str | None = None
) -> dict[str, Any]:
    _ensure_legacy_venv()

    payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
    env = _subprocess_env(thread_mode)

    proc = subprocess.run(
        [
            str(LEGACY_VENV_PYTHON),
            "-c",
            "import pickle, sys; "
            "sys.path.insert(0, %r); "
            "from migration.parity._legacy_runner import _handle; "
            "req = pickle.load(sys.stdin.buffer); "
            "resp = _handle(req); "
            "sys.stdout.buffer.write(pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL))"
            % str(REPO_ROOT),
        ],
        input=payload,
        capture_output=True,
        env=env,
        check=False,
    )

    if proc.returncode != 0:
        raise LegacyOracleError(
            f"Legacy oracle subprocess exited {proc.returncode}.\n"
            f"stderr:\n{proc.stderr.decode(errors='replace')}"
        )

    try:
        response = pickle.load(io.BytesIO(proc.stdout))
    except Exception as e:
        raise LegacyOracleError(
            f"Could not deserialize legacy response: {e}\n"
            f"stderr:\n{proc.stderr.decode(errors='replace')}"
        ) from e

    if not response.get("ok"):
        raise LegacyOracleError(
            f"Legacy oracle raised: {response.get('error')}\n"
            f"{response.get('traceback', '')}"
        )

    return response


def parity(api_id: str, **kwargs: Any) -> dict[str, np.ndarray]:
    """Invoke the legacy implementation once and return its outputs."""
    response = _run_subprocess({"api": api_id, "mode": "parity", "kwargs": kwargs})
    return response["outputs"]


def time_legacy(
    api_id: str,
    *,
    reps: int = 7,
    warmup: int = 1,
    thread_mode: str | None = None,
    **kwargs: Any,
) -> list[float]:
    """Time the legacy implementation (warm). Returns per-rep elapsed seconds."""
    response = _run_subprocess(
        {
            "api": api_id,
            "mode": "time",
            "warmup": warmup,
            "reps": reps,
            "kwargs": kwargs,
        },
        thread_mode=thread_mode,
    )
    return response["elapsed"]


def time_legacy_cold(
    api_id: str, *, thread_mode: str | None = None, **kwargs: Any
) -> float:
    """End-to-end cold latency for legacy: wall-clock from fresh Python
    subprocess spawn → JAX import → JIT compile → first call → result.

    The subprocess invocation overhead (~50–100 ms for Python startup) is
    INCLUDED on purpose — that is what users pay in a one-shot CLI invocation.
    """
    import time as _time

    _ensure_legacy_venv()
    payload = pickle.dumps(
        {"api": api_id, "mode": "parity", "kwargs": kwargs},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    env = _subprocess_env(thread_mode)
    t0 = _time.perf_counter()
    proc = subprocess.run(
        [
            str(LEGACY_VENV_PYTHON),
            "-c",
            "import pickle, sys; "
            "sys.path.insert(0, %r); "
            "from migration.parity._legacy_runner import _handle; "
            "req = pickle.load(sys.stdin.buffer); "
            "resp = _handle(req); "
            "sys.stdout.buffer.write(pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL))"
            % str(REPO_ROOT),
        ],
        input=payload,
        capture_output=True,
        env=env,
        check=False,
    )
    elapsed = _time.perf_counter() - t0
    if proc.returncode != 0:
        raise LegacyOracleError(
            f"cold-time legacy subprocess failed: {proc.stderr.decode(errors='replace')}"
        )
    return elapsed


def time_rust_cold(
    api_id: str, *, thread_mode: str | None = None, **kwargs: Any
) -> float:
    """End-to-end cold latency for rust: wall-clock from fresh Python
    subprocess spawn → adam_core._rust import → first call → result.

    Same wall-clock accounting as ``time_legacy_cold`` so the two are
    directly comparable.
    """
    import time as _time

    payload = pickle.dumps(
        {"api": api_id, "kwargs": kwargs},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    main_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not main_python.exists():
        raise LegacyOracleError(f"rust venv python not found at {main_python}")
    env = _subprocess_env(thread_mode)
    t0 = _time.perf_counter()
    proc = subprocess.run(
        [
            str(main_python),
            "-c",
            "import pickle, sys; "
            "sys.path.insert(0, %r); "
            "from migration.parity._rust_runner import run; "
            "req = pickle.load(sys.stdin.buffer); "
            "out = run(req['api'], **req['kwargs']); "
            "sys.stdout.buffer.write(pickle.dumps({'ok': True, 'outputs': out}, protocol=pickle.HIGHEST_PROTOCOL))"
            % str(REPO_ROOT),
        ],
        input=payload,
        capture_output=True,
        env=env,
        check=False,
    )
    elapsed = _time.perf_counter() - t0
    if proc.returncode != 0:
        raise LegacyOracleError(
            f"cold-time rust subprocess failed: {proc.stderr.decode(errors='replace')}"
        )
    return elapsed


def smoke_test() -> None:
    """Verify the legacy oracle is reachable and dispatches correctly."""
    coords = np.array([[1.0, 0.0, 0.0, 0.0, 0.017, 0.0]], dtype=np.float64)
    out = parity("coordinates.cartesian_to_spherical", coords=coords)["out"]
    assert out.shape == (1, 6), f"unexpected smoke output shape: {out.shape}"
    print(f"[oracle smoke] cart→sph OK, out[0]={out[0].tolist()}", file=sys.stderr)


if __name__ == "__main__":
    smoke_test()
