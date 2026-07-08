"""Shared fixtures for the adam_assist_rust parity suite.

Parity is measured across two isolated runtimes -- exactly like the adam_core
parity gate. The legacy, composition-based ``adam_assist.ASSISTPropagator`` runs
in a dedicated ``.legacy-assist-venv`` (legacy adam_core + downstream
adam_assist) via a subprocess oracle, and the Rust ``adam_assist_rust``
propagator runs in this (composition-deleted) main runtime. The legacy oracle
serializes/caches its outputs so the expensive ASSIST integrations run once.

This replaces the previous in-process ``adam_assist.ASSISTPropagator()``
reference, which is no longer importable-and-instantiable here now that
adam_core's base ``Propagator`` composition has been deleted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def python_reference_propagator():
    """Legacy adam_assist reference propagator via the isolated legacy runtime.

    Returns a :class:`LegacyAssistPropagator` drop-in proxy that runs downstream
    ``adam_assist.ASSISTPropagator`` in ``.legacy-assist-venv`` and returns
    results reconstructed under this runtime's adam_core. Skips when the legacy
    runtime has not been built.
    """
    from migration.parity._assist_oracle import (
        LEGACY_ASSIST_VENV_PYTHON,
        LegacyAssistPropagator,
    )

    if not LEGACY_ASSIST_VENV_PYTHON.exists():
        pytest.skip(
            "legacy adam_assist runtime (.legacy-assist-venv) not built; "
            "see migration/parity/README"
        )
    return LegacyAssistPropagator()
