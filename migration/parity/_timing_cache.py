"""Semantic-result cache policy for canonical performance measurements.

Performance samples measure computation, not repeated-identical-input memoization.
Before every warmup and timed public-Python sample, the harness clears adam_core's
semantic result caches outside the measured interval. Imports, loaded SPICE
kernels/readers, JIT state, and thread pools remain warm.

The same function is imported under both runtimes: it resolves ``adam_core`` to
the current package in the main process and to pinned legacy adam_core in the
isolated legacy process.
"""

from __future__ import annotations

SEMANTIC_CACHE_POLICY = "semantic-result-caches-cleared-before-each-sample"
SEMANTIC_CACHES_CLEARED = (
    "observer-state",
    "origin-translation",
    "spkez-state",
)


def clear_semantic_result_caches() -> None:
    """Clear memoized results without unloading kernels or cold-starting runtimes."""
    from adam_core.coordinates.transform import clear_translation_cache
    from adam_core.observers.state import clear_observer_state_cache
    from adam_core.utils.spice import clear_spkez_cache

    clear_observer_state_cache()
    clear_translation_cache()
    clear_spkez_cache()
