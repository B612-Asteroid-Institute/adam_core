"""Centralized parallel-execution backend for adam-core.

Four adam-core surfaces (the ``Propagator`` ABC plus ``dynamics.propagation``,
``dynamics.ephemeris``, ``dynamics.impacts``, ``orbit_determination.iod``, and
``orbit_determination.od``) historically duplicated the same pattern: pick
between a serial loop and a Ray cluster based on ``max_processes``, then
hand-roll ``ray.put`` / ``ray.remote`` / ``ray.wait`` / ``ray.get`` plumbing
with a ``len(futures) >= max_processes * 1.5`` throttle.

This module collapses that pattern into a small ``ParallelBackend`` protocol
plus two implementations:

- :class:`SequentialBackend` runs tasks in the calling process. ``put`` and
  ``get`` are identity, so a sequential pipeline can transparently treat
  shared inputs the same way as a Ray pipeline.
- :class:`RayBackend` wraps the existing :func:`adam_core.ray_cluster.initialize_use_ray`
  init flow and handles ``put`` / ``get`` / ``is_ref`` / ``free`` against
  Ray's object store. Worker functions are decorated with ``@ray.remote``
  lazily (cached per function + options) so call sites can keep their
  workers as plain Python.

Use :func:`get_backend` to pick a backend by ``max_processes`` policy and
:meth:`ParallelBackend.map_unordered` to run a batched, throttled fan-out
over an iterable of arg tuples.

The abstraction preserves the historical Ray semantics exactly when
``max_processes > 1``; this module is the seam where future work can
replace per-call Ray with a single Python loop calling Rayon-parallel Rust
kernels for surfaces where that is faster.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Optional, Protocol, Tuple

from .ray_cluster import initialize_use_ray

__all__ = [
    "ParallelBackend",
    "SequentialBackend",
    "RayBackend",
    "get_backend",
    "resolve_max_processes",
]


def resolve_max_processes(max_processes: Optional[int]) -> int:
    """Return the effective worker count for a ``max_processes`` argument.

    Mirrors the historical ``if max_processes is None: max_processes =
    mp.cpu_count()`` block used at every Ray callsite so the default-None
    semantics are preserved.
    """
    if max_processes is None:
        return mp.cpu_count()
    return max(1, int(max_processes))


class ParallelBackend(Protocol):
    """Protocol implemented by every parallel-execution backend.

    Backends are short-lived helpers, not long-lived pools: a callsite asks
    :func:`get_backend` for one, drives it, and lets it go out of scope. Ray
    cluster lifetime is managed inside :class:`RayBackend` and is independent
    of any single backend instance.
    """

    name: str

    def put(self, obj: Any) -> Any:
        """Place ``obj`` in the backend's shared store and return a handle.

        ``SequentialBackend`` is identity: the handle is the object itself.
        ``RayBackend`` returns an ``ObjectRef``. The handle is opaque: pass
        it back through :meth:`get` to dereference, or feed it to
        :meth:`map_unordered` arg tuples and let the backend dereference
        it for the worker.
        """

    def get(self, ref: Any) -> Any:
        """Resolve a handle returned by :meth:`put` back to its object."""

    def is_ref(self, obj: Any) -> bool:
        """Return ``True`` when ``obj`` is a handle from this backend."""

    def free(self, refs: list) -> None:
        """Release shared-store handles. No-op for sequential."""

    def map_unordered(
        self,
        fn: Callable[..., Any],
        args_iter: Iterable[Tuple[Any, ...]],
        *,
        max_outstanding: Optional[int] = None,
        worker_options: Optional[dict] = None,
    ) -> Iterator[Any]:
        """Run ``fn(*args)`` for each tuple in ``args_iter``.

        Results are yielded as they complete; the order is unspecified. The
        ``max_outstanding`` bound lets the caller cap pending tasks. The
        ``worker_options`` dict is forwarded to ``ray.remote`` (only the Ray
        backend reads it).
        """


class SequentialBackend:
    """In-process serial execution. ``put`` / ``get`` are identity."""

    name = "sequential"

    def put(self, obj: Any) -> Any:
        return obj

    def get(self, ref: Any) -> Any:
        return ref

    def is_ref(self, obj: Any) -> bool:
        return False

    def free(self, refs: list) -> None:
        return None

    def map_unordered(
        self,
        fn: Callable[..., Any],
        args_iter: Iterable[Tuple[Any, ...]],
        *,
        max_outstanding: Optional[int] = None,
        worker_options: Optional[dict] = None,
    ) -> Iterator[Any]:
        for args in args_iter:
            yield fn(*args)


class RayBackend:
    """Ray-cluster execution preserving the historical adam-core semantics.

    The constructor calls :func:`adam_core.ray_cluster.initialize_use_ray`
    with the requested ``num_cpus``; that helper attaches to an existing
    cluster if one is reachable and otherwise starts a fresh local cluster.
    Cluster lifetime is intentionally independent of the backend instance.
    """

    name = "ray"

    def __init__(self, num_cpus: int, **init_kwargs: Any) -> None:
        import ray

        self._ray = ray
        self._num_cpus = max(1, int(num_cpus))
        # Honour the historical try-attach-existing / fall-back-to-fresh
        # behaviour and JAX fork-warning silencing.
        if not initialize_use_ray(num_cpus=self._num_cpus, **init_kwargs):
            raise RuntimeError(
                f"Failed to initialize Ray cluster (num_cpus={self._num_cpus})."
            )
        self._remote_cache: dict[Tuple[int, Tuple[Tuple[str, Any], ...]], Any] = {}

    def _remote(self, fn: Callable[..., Any], options: Optional[dict]) -> Any:
        opts_key = tuple(sorted((options or {}).items()))
        cache_key = (id(fn), opts_key)
        cached = self._remote_cache.get(cache_key)
        if cached is not None:
            return cached
        if options:
            remote_fn = self._ray.remote(**options)(fn)
        else:
            remote_fn = self._ray.remote(fn)
        self._remote_cache[cache_key] = remote_fn
        return remote_fn

    def put(self, obj: Any) -> Any:
        if isinstance(obj, self._ray.ObjectRef):
            return obj
        return self._ray.put(obj)

    def get(self, ref: Any) -> Any:
        if isinstance(ref, self._ray.ObjectRef):
            return self._ray.get(ref)
        return ref

    def is_ref(self, obj: Any) -> bool:
        return isinstance(obj, self._ray.ObjectRef)

    def free(self, refs: list) -> None:
        if not refs:
            return
        self._ray.internal.free(refs)

    def map_unordered(
        self,
        fn: Callable[..., Any],
        args_iter: Iterable[Tuple[Any, ...]],
        *,
        max_outstanding: Optional[int] = None,
        worker_options: Optional[dict] = None,
    ) -> Iterator[Any]:
        ray = self._ray
        remote_fn = self._remote(fn, worker_options)
        cap = (
            max(1, int(max_outstanding))
            if max_outstanding is not None
            else max(1, int(self._num_cpus * 1.5))
        )
        pending: list = []
        for args in args_iter:
            pending.append(remote_fn.remote(*args))
            if len(pending) >= cap:
                finished, pending = ray.wait(pending, num_returns=1)
                yield ray.get(finished[0])
        while pending:
            finished, pending = ray.wait(pending, num_returns=1)
            yield ray.get(finished[0])


def get_backend(
    max_processes: Optional[int],
    *,
    object_store_bytes: Optional[int] = None,
    **ray_init_kwargs: Any,
) -> ParallelBackend:
    """Return a backend appropriate for ``max_processes``.

    ``max_processes <= 1`` always returns :class:`SequentialBackend`; any
    higher value returns :class:`RayBackend`. ``max_processes is None``
    expands to ``mp.cpu_count()`` so the historical default is preserved.
    """
    n = resolve_max_processes(max_processes)
    if n <= 1:
        return SequentialBackend()
    return RayBackend(
        num_cpus=n,
        object_store_bytes=object_store_bytes,
        **ray_init_kwargs,
    )
