"""SPICE access for adam-core.

``RustBackend`` is a thin Python owner for the process-local Rust backend
implemented in ``adam_core_rs_spice`` and exposed through
``adam_core._rust_native.AdamCoreSpiceBackend``. Kernel registration,
SPK/PCK reader dispatch, text-kernel name bindings, and last-loaded-wins
semantics live in Rust so standalone ``adam-core-rs`` can use the same
behavior without importing Python ``spicekit``.

Parity against CSPICE remains owned upstream by the public ``spicekit``
crate's ``spicekit-bench`` suite. adam-core tests validate adam-core's
wiring and process-local backend semantics.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

from .._rust import adam_core_spice_backend


class NotCovered(Exception):
    """Raised when a SPICE request falls outside loaded kernel coverage."""


class RustBackend:
    """adam-core's SPICE implementation backed by direct Rust ``spicekit``.

    The Python object deliberately owns only locking and exception mapping.
    All kernel state and dispatch semantics are held by the Rust backend.
    """

    def __init__(self) -> None:
        # Every handle delegates to the process-global Rust backend, which is
        # the single source of truth for loaded kernels. Kernel-registration
        # bookkeeping lives in Rust (the backend's kernel list), so it can
        # never desync from what is actually loaded, and multiple handles
        # (e.g. after a ``get_backend`` rebuild) all see the same kernels.
        self._inner = adam_core_spice_backend()
        self._lock = threading.Lock()

    def furnsh(self, path: str) -> None:
        with self._lock:
            self._inner.furnsh(path)

    def unload(self, path: str) -> None:
        with self._lock:
            self._inner.unload(path)

    def registered_kernels(self) -> list[str]:
        """Paths of every kernel loaded in the process-global backend."""
        with self._lock:
            return list(self._inner.registered_kernels())

    def is_registered(self, path: str) -> bool:
        with self._lock:
            return bool(self._inner.is_registered(path))

    def clear(self) -> None:
        """Unload every kernel and clear custom names (test isolation)."""
        with self._lock:
            self._inner.clear()

    def spkez(
        self,
        target: int,
        et: float,
        frame: str,
        observer: int,
    ) -> np.ndarray:
        ets = np.array([et], dtype=np.float64)
        return self.spkez_batch(target, observer, frame, ets)[0]

    def spkez_batch(
        self,
        target: int,
        observer: int,
        frame: str,
        ets: np.ndarray,
    ) -> np.ndarray:
        ets = np.ascontiguousarray(ets, dtype=np.float64)
        with self._lock:
            try:
                return np.asarray(
                    self._inner.spkez_batch(int(target), int(observer), frame, ets),
                    dtype=np.float64,
                )
            except (RuntimeError, ValueError) as exc:
                raise NotCovered(
                    f"rust SPK readers do not cover target={target} "
                    f"observer={observer} frame={frame} across {len(ets)} epochs"
                ) from exc

    def pxform_batch(
        self,
        frame_from: str,
        frame_to: str,
        ets: np.ndarray,
    ) -> np.ndarray:
        ets = np.ascontiguousarray(ets, dtype=np.float64)
        with self._lock:
            try:
                return np.asarray(
                    self._inner.pxform_batch(frame_from, frame_to, ets),
                    dtype=np.float64,
                )
            except (RuntimeError, ValueError) as exc:
                raise NotCovered(
                    f"rust PCK readers do not cover pxform({frame_from},{frame_to})"
                ) from exc

    def sxform_batch(
        self,
        frame_from: str,
        frame_to: str,
        ets: np.ndarray,
    ) -> np.ndarray:
        ets = np.ascontiguousarray(ets, dtype=np.float64)
        with self._lock:
            try:
                return np.asarray(
                    self._inner.sxform_batch(frame_from, frame_to, ets),
                    dtype=np.float64,
                )
            except (RuntimeError, ValueError) as exc:
                raise NotCovered(
                    f"rust PCK readers do not cover sxform({frame_from},{frame_to})"
                ) from exc

    def load_mpc_obscodes(self, json_text: str) -> int:
        """Parse + cache MPC observatory parallax coefficients Rust-side."""
        with self._lock:
            return int(self._inner.load_mpc_obscodes(json_text))

    def mpc_obscodes_loaded(self) -> int:
        """Number of MPC observatory ground sites loaded in the backend."""
        with self._lock:
            return int(self._inner.mpc_obscodes_loaded())

    def observer_states_from_codes(
        self,
        unique_codes: list,
        code_indices: np.ndarray,
        time_scale: str,
        time_days: np.ndarray,
        time_nanos: np.ndarray,
        frame: str,
        origin_code: str,
    ) -> np.ndarray:
        """Single-crossing per-row MPC-code ground-observer states
        (dictionary-encoded codes + numpy epoch buffers)."""
        with self._lock:
            return np.asarray(
                self._inner.observer_states_from_codes(
                    unique_codes,
                    np.ascontiguousarray(code_indices, dtype=np.int64),
                    time_scale,
                    np.ascontiguousarray(time_days, dtype=np.int64),
                    np.ascontiguousarray(time_nanos, dtype=np.int64),
                    frame,
                    origin_code,
                ),
                dtype=np.float64,
            )

    def observer_states_from_codes_arrow(self, batch, time_scale: str):
        """Arrow-native single-crossing ``Observers.from_codes``.

        Takes one pyarrow ``RecordBatch`` (``code``/``days``/``nanos`` columns)
        plus the input ``time_scale`` and returns one nested ``Observers``
        pyarrow ``RecordBatch`` (code + heliocentric-ecliptic Cartesian
        coordinates). No numpy marshaling on either side; grouping/dedup and
        table assembly happen in Rust.
        """
        with self._lock:
            return self._inner.observer_states_from_codes_arrow(batch, time_scale)

    def bodn2c(self, name: str) -> int:
        with self._lock:
            try:
                return int(self._inner.bodn2c(name))
            except (RuntimeError, ValueError) as exc:
                raise NotCovered(
                    f"NAIF body name '{name}' is not in the built-in table "
                    f"and was not declared by any loaded text kernel"
                ) from exc


_BACKEND_LOCK = threading.Lock()
_BACKEND: Optional[RustBackend] = None
_BACKEND_PID: Optional[int] = None


def get_backend() -> RustBackend:
    """Return the process-local SPICE backend.

    Ray workers inherit module state across fork/spawn boundaries, so the
    backend is rebuilt whenever the PID changes.
    """
    global _BACKEND, _BACKEND_PID
    pid = os.getpid()
    with _BACKEND_LOCK:
        if _BACKEND is None or _BACKEND_PID != pid:
            _BACKEND = RustBackend()
            _BACKEND_PID = pid
        return _BACKEND
