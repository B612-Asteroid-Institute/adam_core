"""Rust surface name-parity enforcement (personal-cmy.34).

Core migration requirement: the canonical Rust surface must expose functions
named IDENTICALLY to the legacy public API names (the Python wrappers already
keep the legacy names). Implementation-detail additions are allowed but are
not the canonical surface:

* ``_flat6``  -- flattened-6 row layout helper (6 f64 per row, row-major)
* ``_numpy``  -- PyO3 / NumPy boundary wrapper
* ``_batch`` / ``_row`` -- batched / single-row helpers

This test enumerates the migrated legacy public API names and asserts that an
identically-named canonical Rust symbol exists in the Rust crate sources --
either a ``pub fn <name>`` definition (free function or inherent method) or a
``pub use ... as <name>`` / ``pub use mod::{<name>}`` re-export alias.

It is deliberately a source-introspection test: Rust symbols are not
importable from Python, and this keeps the convention enforced as new surface
is migrated without a hand-maintained parallel registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# repo root: .../src/adam_core/tests/<this file>
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Rust crates that host the canonical numeric / composed coordinate surface.
_RUST_CRATE_SRC_DIRS = (
    _REPO_ROOT / "rust" / "adam_core_rs_coords" / "src",
    _REPO_ROOT / "rust" / "adam_core_rs_spice" / "src",
)

# Legacy public API names (adam_core.coordinates / adam_core.dynamics) whose
# computation has been migrated to Rust. Each must have an identically-named
# canonical Rust counterpart. `_flat6`/`_numpy`/`_batch`/`_row` suffixed helpers
# do NOT satisfy the requirement on their own.
LEGACY_PUBLIC_API_NAMES = (
    # Representation conversions.
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "cartesian_to_keplerian",
    "keplerian_to_cartesian",
    "cartesian_to_cometary",
    "cometary_to_cartesian",
    "cartesian_to_geodetic",
    # Composed top-level surface (SPICE-backed; single-crossing orchestrator).
    "transform_coordinates",
    # Two-body / Keplerian dynamics helpers.
    "calc_mean_motion",
    "calc_mean_anomaly",
    "calc_period",
    "calc_periapsis_distance",
    "calc_apoapsis_distance",
    "calc_semi_major_axis",
    "calc_semi_latus_rectum",
    # MOID.
    "calculate_moid",
)

# Suffixes that mark implementation-detail helpers, not the canonical surface.
_HELPER_SUFFIXES = ("_flat6", "_numpy", "_batch", "_row")


def _rust_source_text() -> str:
    chunks: list[str] = []
    for src_dir in _RUST_CRATE_SRC_DIRS:
        assert src_dir.is_dir(), f"missing Rust crate source dir: {src_dir}"
        for rs_file in sorted(src_dir.rglob("*.rs")):
            chunks.append(rs_file.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _has_canonical_symbol(name: str, source: str) -> bool:
    # A canonical symbol is either a public definition (free fn or inherent
    # method) or a public re-export/alias under exactly this name.
    patterns = (
        rf"pub fn {re.escape(name)}\s*[(<]",  # pub fn name( / pub fn name<
        rf"\bas {re.escape(name)}\b",  # pub use ... as name
        rf"pub use[^;]*\b{re.escape(name)}\b",  # pub use mod::{{name, ...}}
    )
    return any(re.search(pattern, source) for pattern in patterns)


@pytest.fixture(scope="module")
def rust_source() -> str:
    return _rust_source_text()


@pytest.mark.parametrize("name", LEGACY_PUBLIC_API_NAMES)
def test_canonical_rust_name_matches_legacy_public_api(name: str, rust_source: str):
    assert _has_canonical_symbol(name, rust_source), (
        f"legacy public API '{name}' has no identically-named canonical Rust "
        f"symbol; a suffixed helper ({', '.join(_HELPER_SUFFIXES)}) does not "
        f"satisfy the name-parity requirement (personal-cmy.34)."
    )


def test_legacy_api_names_are_not_helper_suffixed() -> None:
    # Guard the enumeration itself: the canonical names must be bare, never the
    # implementation-suffixed helper names.
    offenders = [
        name for name in LEGACY_PUBLIC_API_NAMES if name.endswith(_HELPER_SUFFIXES)
    ]
    assert not offenders, f"canonical API names must be bare, got helpers: {offenders}"
