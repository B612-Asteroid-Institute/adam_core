"""SPICE access for adam-core.

:class:`RustBackend` wraps the pure-Rust ``NaifSpk``, ``NaifPck``,
built-in body-name table, and text-kernel parser from the ``spicekit``
crate. Requests that fall outside the loaded kernel coverage raise
:class:`NotCovered`. Parity against CSPICE is maintained upstream in
the ``spicekit`` library.

Process safety
--------------
* ``furnsh`` / ``unload`` apply to the current process only; Ray
  workers spawn with their own pools and re-initialize independently
  on first use in that PID.
* Default-kernel load is guarded by a per-PID env-var sentinel so
  concurrent calls in the same process don't double-furnsh.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .._rust import (
    SPICEKIT_AVAILABLE,
    naif_bodn2c as _rust_builtin_bodn2c,
    naif_parse_text_kernel_bindings,
    naif_pck_open,
    naif_spk_open,
)


class NotCovered(Exception):
    """Raised when a SPICE request falls outside the loaded kernel
    coverage — e.g. an ITRF93 query with no binary PCK loaded, or a
    body name not in the built-in table or any loaded text kernel.
    """


_INERTIAL_FRAMES = ("J2000", "ECLIPJ2000")


def _normalize_body_name(name: str) -> str:
    """Mirror the Rust ``normalize_name``: uppercase, collapse whitespace
    and underscores into single spaces. Kept in sync with
    ``adam_core_rs_naif::naif_ids::normalize_name``."""
    out: list[str] = []
    prev_space = True
    for ch in name:
        if ch.isspace() or ch == "_":
            if not prev_space:
                out.append(" ")
                prev_space = True
        else:
            out.append(ch.upper())
            prev_space = False
    s = "".join(out)
    if s.endswith(" "):
        s = s[:-1]
    return s


@dataclass
class _RustKernel:
    """Record of a kernel the Rust backend knows about. Binary SPKs and
    PCKs carry a live reader; text kernels (``.tk``/``.tf``/``.tpc``)
    carry the list of body bindings extracted at furnsh time; other
    kernels are tracked only to keep ``unload`` idempotent."""

    path: str
    kind: str  # "spk" | "pck" | "text" | "ignored"
    reader: object = None
    bindings: List[Tuple[str, int]] = field(default_factory=list)


class RustBackend:
    """adam-core's SPICE implementation, backed by spicekit's pure-Rust
    readers.

    Handles the full adam-core SPICE surface:

    * Binary SPK (state queries in J2000 / ECLIPJ2000 / ITRF93).
    * Binary PCK (ITRF93 rotations via pxform/sxform).
    * Built-in NAIF body-name table (692 entries mirroring CSpice's
      ``zzidmap.c``).
    * Text-kernel body bindings (``NAIF_BODY_NAME += ...`` /
      ``NAIF_BODY_CODE += ...``) parsed at ``furnsh`` time. Custom
      bindings take precedence over the built-in table and follow
      CSpice's last-loaded-wins semantics.

    Raises :class:`NotCovered` when a request falls outside the loaded
    kernel coverage.
    """

    def __init__(self):
        if not SPICEKIT_AVAILABLE:
            raise RuntimeError(
                "RustBackend requires the spicekit Python package, which "
                "is not available in this environment."
            )
        self._kernels: List[_RustKernel] = []
        self._pck_readers: List[object] = []
        # Name → code, last-loaded-wins via dict insertion order.
        self._custom_names: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ---- kernel registration ----

    @staticmethod
    def _peek_daf_idword(path: str) -> Optional[bytes]:
        """Return the first 8 bytes of a DAF file, or None if the file
        can't be read or isn't a DAF kernel."""
        try:
            with open(path, "rb") as fh:
                idword = fh.read(8)
        except OSError:
            return None
        if not idword.startswith(b"DAF/"):
            return None
        return idword

    def _rebuild_name_index(self) -> None:
        """Rebuild the name→code dict from the ordered kernel list.

        Text kernels are replayed in load order so last-loaded-wins
        semantics apply (dict insertion keeps the latest assignment as
        the effective value). This is cheap — custom kernels typically
        carry a handful of bindings — and avoids subtle bugs when a
        kernel is unloaded partway through the list.
        """
        self._custom_names.clear()
        for k in self._kernels:
            if k.kind != "text":
                continue
            for raw_name, code in k.bindings:
                self._custom_names[_normalize_body_name(raw_name)] = int(code)

    def furnsh(self, path: str) -> None:
        with self._lock:
            # Idempotent: silently skip duplicates.
            for k in self._kernels:
                if k.path == path:
                    return

            idword = self._peek_daf_idword(path)
            if idword is not None and idword.startswith(b"DAF/SPK"):
                try:
                    reader = naif_spk_open(path)
                except (RuntimeError, ValueError) as exc:
                    raise RuntimeError(
                        f"RustBackend cannot open SPK {path}: {exc}"
                    ) from exc
                self._kernels.append(_RustKernel(path, "spk", reader))
                return
            if idword is not None and idword.startswith(b"DAF/PCK"):
                try:
                    reader = naif_pck_open(path)
                except (RuntimeError, ValueError) as exc:
                    raise RuntimeError(
                        f"RustBackend cannot open PCK {path}: {exc}"
                    ) from exc
                self._kernels.append(_RustKernel(path, "pck", reader))
                self._pck_readers.append(reader)
                return

            # Not a binary DAF — try parsing as a text kernel. If it
            # declares body bindings we register them; otherwise it's
            # tracked as "ignored" so unload() stays idempotent.
            try:
                bindings = naif_parse_text_kernel_bindings(path) or []
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"RustBackend cannot parse text kernel {path}: {exc}"
                ) from exc
            if bindings:
                self._kernels.append(
                    _RustKernel(path, "text", bindings=list(bindings))
                )
                self._rebuild_name_index()
            else:
                self._kernels.append(_RustKernel(path, "ignored"))

    def unload(self, path: str) -> None:
        with self._lock:
            remaining: List[_RustKernel] = []
            had_text = False
            for k in self._kernels:
                if k.path == path:
                    if k.kind == "pck":
                        self._pck_readers.remove(k.reader)
                    elif k.kind == "text":
                        had_text = True
                    continue
                remaining.append(k)
            self._kernels = remaining
            if had_text:
                self._rebuild_name_index()

    # ---- state queries ----

    def _spk_readers(self):
        # Newest-first for last-loaded-wins resolution.
        return [k.reader for k in reversed(self._kernels) if k.kind == "spk"]

    def _try_spk_state_batch(
        self, target: int, center: int, frame: str, ets: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return a (N, 6) array or None if no loaded SPK covers the pair.

        Supports the three frames adam-core actually queries:

        * ``J2000`` — directly from the SPK reader.
        * ``ECLIPJ2000`` — delegated to the SPK reader's
          ``state_batch_in_frame`` (analytic ecliptic obliquity).
        * ``ITRF93`` — compose the J2000 state with the PCK-driven
          ``sxform(J2000, ITRF93)``. Needs at least one binary PCK
          loaded; otherwise returns None so the caller raises
          NotCovered cleanly.
        """
        readers = self._spk_readers()
        if not readers:
            return None

        if frame in _INERTIAL_FRAMES:
            for reader in readers:
                try:
                    if frame == "J2000":
                        return np.asarray(reader.state_batch(target, center, ets))
                    return np.asarray(
                        reader.state_batch_in_frame(target, center, ets, "ECLIPJ2000")
                    )
                except (RuntimeError, ValueError):
                    continue
            return None

        if frame == "ITRF93":
            sxform = self._try_sxform_batch("J2000", "ITRF93", ets)
            if sxform is None:
                return None
            j2000 = None
            for reader in readers:
                try:
                    j2000 = np.asarray(reader.state_batch(target, center, ets))
                    break
                except (RuntimeError, ValueError):
                    continue
            if j2000 is None:
                return None
            # sxform is (N, 6, 6); multiply row-wise by j2000 (N, 6).
            return np.einsum("nij,nj->ni", sxform, j2000)

        return None

    def spkez(
        self,
        target: int,
        et: float,
        frame: str,
        observer: int,
    ) -> np.ndarray:
        ets = np.array([et], dtype=np.float64)
        out = self._try_spk_state_batch(int(target), int(observer), frame, ets)
        if out is None:
            raise NotCovered(
                f"rust SPK readers do not cover target={target} observer={observer} "
                f"frame={frame} at et={et}"
            )
        return out[0]

    def spkez_batch(
        self,
        target: int,
        observer: int,
        frame: str,
        ets: np.ndarray,
    ) -> np.ndarray:
        ets = np.ascontiguousarray(ets, dtype=np.float64)
        out = self._try_spk_state_batch(int(target), int(observer), frame, ets)
        if out is None:
            raise NotCovered(
                f"rust SPK readers do not cover target={target} observer={observer} "
                f"frame={frame} across {len(ets)} epochs"
            )
        return out

    # ---- rotations ----

    def _try_pxform_batch(
        self, frame_from: str, frame_to: str, ets: np.ndarray
    ) -> Optional[np.ndarray]:
        if "ITRF93" not in (frame_from, frame_to):
            return None
        for reader in reversed(self._pck_readers):
            try:
                return np.asarray(reader.pxform_batch(frame_from, frame_to, ets))
            except (RuntimeError, ValueError):
                continue
        return None

    def _try_sxform_batch(
        self, frame_from: str, frame_to: str, ets: np.ndarray
    ) -> Optional[np.ndarray]:
        if "ITRF93" not in (frame_from, frame_to):
            return None
        for reader in reversed(self._pck_readers):
            try:
                return np.asarray(reader.sxform_batch(frame_from, frame_to, ets))
            except (RuntimeError, ValueError):
                continue
        return None

    def pxform_batch(
        self,
        frame_from: str,
        frame_to: str,
        ets: np.ndarray,
    ) -> np.ndarray:
        ets = np.ascontiguousarray(ets, dtype=np.float64)
        out = self._try_pxform_batch(frame_from, frame_to, ets)
        if out is None:
            raise NotCovered(
                f"rust PCK readers do not cover pxform({frame_from},{frame_to})"
            )
        return out

    def sxform_batch(
        self,
        frame_from: str,
        frame_to: str,
        ets: np.ndarray,
    ) -> np.ndarray:
        ets = np.ascontiguousarray(ets, dtype=np.float64)
        out = self._try_sxform_batch(frame_from, frame_to, ets)
        if out is None:
            raise NotCovered(
                f"rust PCK readers do not cover sxform({frame_from},{frame_to})"
            )
        return out

    # ---- name lookup ----

    def bodn2c(self, name: str) -> int:
        """Resolve a NAIF body name: custom text-kernel bindings first
        (last-loaded-wins), then the built-in ``zzidmap`` table. Raises
        :class:`NotCovered` when the name is unknown in both."""
        if self._custom_names:
            key = _normalize_body_name(name)
            if key in self._custom_names:
                return self._custom_names[key]
        rc = _rust_builtin_bodn2c(name)
        if rc is None:
            raise NotCovered(
                f"NAIF body name '{name}' is not in the built-in table and was "
                f"not declared by any loaded text kernel"
            )
        return rc


# ----------------------------------------------------------------------
# Module-level accessor + process-safe init
# ----------------------------------------------------------------------


_BACKEND_LOCK = threading.Lock()
_BACKEND: Optional[RustBackend] = None
_BACKEND_PID: Optional[int] = None


def get_backend() -> RustBackend:
    """Return the process-local SPICE backend, constructing it on first
    use. Re-builds if the PID has changed (Ray workers inherit the
    parent's module state but need their own backend instance)."""
    global _BACKEND, _BACKEND_PID
    pid = os.getpid()
    with _BACKEND_LOCK:
        if _BACKEND is None or _BACKEND_PID != pid:
            if not SPICEKIT_AVAILABLE:
                raise RuntimeError(
                    "adam-core's SPICE backend requires the spicekit "
                    "Python package. Install spicekit to restore SPICE "
                    "support."
                )
            _BACKEND = RustBackend()
            _BACKEND_PID = pid
        return _BACKEND
