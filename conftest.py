from __future__ import annotations

import os

import pytest


def _require_rust_backend() -> bool:
    value = os.environ.get("ADAM_CORE_REQUIRE_RUST_BACKEND", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pytest_sessionstart(session: pytest.Session) -> None:
    if not _require_rust_backend():
        return

    from adam_core._rust import RUST_BACKEND_AVAILABLE

    if not RUST_BACKEND_AVAILABLE:
        pytest.exit(
            "ADAM_CORE_REQUIRE_RUST_BACKEND is set, but adam_core._rust_native is unavailable. "
            "Run `pdm run rust-develop` and retry.",
            returncode=2,
        )
