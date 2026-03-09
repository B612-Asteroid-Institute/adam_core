from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from typing import TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def bounded_lru_get(cache: "OrderedDict[K, V]", key: K, *, maxsize: int) -> V | None:
    """
    Get a value from a bounded LRU cache.

    If present, the entry is marked as most-recently used.
    If maxsize <= 0, the cache is treated as disabled.
    """
    if int(maxsize) <= 0:
        return None
    v = cache.get(key)
    if v is None:
        return None
    cache.move_to_end(key)
    return v


def bounded_lru_put(
    cache: "OrderedDict[K, V]", key: K, value: V, *, maxsize: int
) -> None:
    """
    Put a value into a bounded LRU cache.

    If maxsize <= 0, the cache is treated as disabled.
    """
    if int(maxsize) <= 0:
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > int(maxsize):
        cache.popitem(last=False)
