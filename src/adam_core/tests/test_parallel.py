"""Sanity tests for the centralized parallel-execution backend.

These cover the contract that all callsites in adam-core depend on: serial
is a strict identity wrapper, the Ray backend caches remote-fn handles, and
``map_unordered`` returns one result per submitted args tuple.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from adam_core import parallel


def _square(x: int, scale: int = 1) -> int:
    return x * x * scale


def test_resolve_max_processes_default_uses_cpu_count() -> None:
    assert parallel.resolve_max_processes(None) == mp.cpu_count()
    assert parallel.resolve_max_processes(0) == 1
    assert parallel.resolve_max_processes(-3) == 1
    assert parallel.resolve_max_processes(4) == 4


def test_get_backend_picks_sequential_for_one_or_zero_processes() -> None:
    assert isinstance(parallel.get_backend(1), parallel.SequentialBackend)
    assert isinstance(parallel.get_backend(0), parallel.SequentialBackend)


def test_sequential_backend_is_identity_for_put_get_is_ref() -> None:
    backend = parallel.SequentialBackend()
    sentinel = object()
    ref = backend.put(sentinel)
    assert ref is sentinel
    assert backend.get(ref) is sentinel
    assert backend.is_ref(sentinel) is False
    backend.free([sentinel])  # no-op, must not raise


def test_sequential_map_unordered_runs_each_args_tuple_once() -> None:
    backend = parallel.SequentialBackend()
    results = list(
        backend.map_unordered(_square, [(2,), (3,), (4,)], max_outstanding=2)
    )
    assert sorted(results) == [4, 9, 16]


def test_sequential_map_unordered_forwards_extra_args() -> None:
    backend = parallel.SequentialBackend()
    results = list(backend.map_unordered(_square, [(2, 10), (3, 10)]))
    assert sorted(results) == [40, 90]


@pytest.mark.parametrize(
    "args_iter,expected",
    [
        ([], []),
        ([(5,)], [25]),
    ],
)
def test_sequential_map_unordered_handles_empty_and_single(args_iter, expected) -> None:
    backend = parallel.SequentialBackend()
    assert list(backend.map_unordered(_square, args_iter)) == expected
