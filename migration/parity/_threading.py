"""Thread-pool policy helpers for parity/performance governance.

The Rust-vs-baseline warm speed gate measures production-realistic
best-effort throughput by default: both sides run with their thread pools
uncapped (Rust Rayon and legacy NumPy/JAX/XLA/BLAS) so a real-world caller
sees the same comparison the gate enforces.

Thread-mode terminology:

- ``multi-thread`` (default) removes harness-imposed thread caps so both
  Rust Rayon and the legacy NumPy/JAX/XLA/BLAS pools scale across available
  cores like a real production process.
- ``single`` caps every known thread-pool knob to 1 (Rayon, OpenMP, BLAS,
  JAX/XLA, NumExpr, vecLib) and forwards those env vars to the legacy
  subprocess. On macOS Apple Silicon, JAX/XLA's CPU thread pool cannot be
  reliably constrained from those env vars (verified 2026-05-07: legacy
  JAX/XLA photometry kernels show cpu/wall ~3.6 cores even with all known
  caps applied), so ``single`` there is an asymmetric Rust-1-core vs
  legacy-uncapped diagnostic rather than a clean apples-to-apples gate. On
  Linux it is closer to symmetric but still depends on whether libomp/BLAS
  honors the env caps in a particular wheel build.

``native`` is accepted as a deprecated backward-compatibility alias for
``multi-thread`` and is normalized on entry to the canonical name.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Literal, cast

ThreadMode = Literal["single", "multi-thread"]

# Backward-compat aliases accepted at the CLI / API surface but normalized
# to the canonical names above before any cache key, artifact, or env
# decision is taken.
_THREAD_MODE_ALIASES: dict[str, ThreadMode] = {
    "native": "multi-thread",
    "multithread": "multi-thread",
    "multi_thread": "multi-thread",
}

# Keep this tuple stable so artifact snapshots and tests print env vars in a
# predictable order.
THREAD_ENV_KEYS: tuple[str, ...] = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "JAX_NUM_THREADS",
    "XLA_FLAGS",
)

# Load-bearing env vars by backend class:
# - RAYON_NUM_THREADS controls adam-core's Rust parallel iterators.
# - XLA_FLAGS/JAX_NUM_THREADS control the legacy JAX/XLA oracle.
# - OPENBLAS/MKL/VECLIB/OMP cover NumPy/SciPy/BLAS/OpenMP code paths such as
#   residual/covariance helpers.
# - NUMEXPR_NUM_THREADS is hygiene; adam-core does not currently use numexpr.
#
# JAX's CPU backend accepts the intra-op setting without a leading "--" on the
# second token for the pinned legacy environment. The double-dash variant is an
# unknown XLA flag there.
SINGLE_THREAD_ENV: dict[str, str] = {
    "RAYON_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "JAX_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


def validate_thread_mode(mode: str) -> ThreadMode:
    canonical = _THREAD_MODE_ALIASES.get(mode, mode)
    if canonical not in {"single", "multi-thread"}:
        raise ValueError(
            f"thread mode must be 'single' or 'multi-thread' (got {mode!r}); "
            "'native' is accepted as a deprecated alias for 'multi-thread'"
        )
    return cast(ThreadMode, canonical)


def env_for_thread_mode(
    mode: str | None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with the requested thread policy applied.

    ``single`` overwrites all known thread-pool knobs with deterministic caps
    for both Rust (Rayon) and the legacy baseline (NumPy/JAX/XLA/BLAS).
    ``multi-thread`` removes caps that match this harness's single-thread
    values so a child process can represent ordinary production behavior
    where both Rust Rayon and the legacy NumPy/JAX/BLAS pools scale across
    available cores. Non-default external values (for example
    ``RAYON_NUM_THREADS=4``) are preserved as authored.
    """
    env = dict(os.environ if base_env is None else base_env)
    if mode is None:
        return env

    thread_mode = validate_thread_mode(mode)
    if thread_mode == "single":
        env.update(SINGLE_THREAD_ENV)
        return env

    for key, value in SINGLE_THREAD_ENV.items():
        if env.get(key) == value:
            env.pop(key, None)
    return env


def apply_thread_mode(
    mode: str,
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Apply a thread mode to ``env`` in-place and return a snapshot.

    Call this before importing ``adam_core._rust``, JAX, or BLAS-backed modules
    in benchmark entrypoints. Thread-pool libraries typically read these values
    when their runtime initializes, so changing them after a Rust/JAX import is
    not a reliable reconfiguration mechanism.
    """
    target = os.environ if env is None else env
    updated = env_for_thread_mode(mode, target)
    for key in THREAD_ENV_KEYS:
        if key in updated:
            target[key] = updated[key]
        else:
            target.pop(key, None)
    return snapshot_thread_env(target)


def snapshot_thread_env(
    env: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    source = os.environ if env is None else env
    return {key: source.get(key) for key in THREAD_ENV_KEYS}
