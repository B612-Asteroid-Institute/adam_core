"""Comparison-mode metadata for parity and speed artifacts.

The migration gates intentionally compare different layers depending on the API:
public Python facades, thin PyO3/NumPy bindings, raw helper kernels, or private
backend candidates. This module centralizes the labels so JSON artifacts and
markdown reports make that distinction explicit.
"""

from __future__ import annotations

from typing import Any

from adam_core._rust.status import API_MIGRATIONS_BY_ID

from . import backend_candidates

PUBLIC_PYTHON_FACADE = "public_python_facade_vs_legacy_public_python"
THIN_PYTHON_NUMPY_WRAPPER = "thin_python_numpy_wrapper_vs_legacy_public_python"
RAW_RUST_PYO3_KERNEL = "raw_rust_pyo3_kernel_vs_legacy_python_oracle"
BACKEND_CANDIDATE = "backend_candidate_vs_legacy_public_python"
RUST_NATIVE_ONLY = "rust_native_only_diagnostic"
UNKNOWN = "unknown_comparison_mode"


def for_api(api_id: str) -> dict[str, Any]:
    """Return machine-readable comparison metadata for a parity/speed API id."""
    candidate = backend_candidates.get(api_id)
    if candidate is not None:
        return {
            "comparison_mode": BACKEND_CANDIDATE,
            "comparison_mode_short": "impl candidate",
            "comparison_mode_label": "backend candidate vs legacy public Python",
            "current_entrypoint_kind": "python_adapter_backend_candidate",
            "current_boundary": candidate.boundary,
            "legacy_entrypoint_kind": "legacy_public_python",
            "legacy_entrypoint": candidate.legacy_comparator,
            "registry_status": "backend-candidate",
            "rust_module": candidate.rust_module,
            "rust_native_top_level": False,
            "speed_gate_scope": "diagnostic_backend_candidate",
            "backend_candidate": candidate.to_json(),
        }

    migration = API_MIGRATIONS_BY_ID.get(api_id)
    if migration is None:
        return {
            "comparison_mode": UNKNOWN,
            "comparison_mode_short": "unknown",
            "comparison_mode_label": "unknown comparison mode",
            "current_entrypoint_kind": "unknown",
            "current_boundary": "unknown",
            "legacy_entrypoint_kind": "legacy_oracle",
            "legacy_entrypoint": api_id,
            "registry_status": "unknown",
            "rust_module": "",
            "rust_native_top_level": False,
            "speed_gate_scope": "unknown",
        }

    base: dict[str, Any] = {
        "current_boundary": migration.boundary,
        "legacy_entrypoint_kind": "legacy_public_python_or_oracle",
        "legacy_entrypoint": api_id,
        "registry_status": migration.status,
        "parity_coverage": migration.parity_coverage,
        "coverage_note": migration.coverage_note,
        "covered_subcases": migration.covered_subcases,
        "excluded_subcases": migration.excluded_subcases,
        "rust_module": migration.rust_module,
        # This tracks full Rust-port completion, not whether the current Python
        # facade calls Rust internally. Most current rows still enter through
        # Python/PyO3 even when the computational kernel is Rust.
        "rust_native_top_level": migration.status == "rust-only",
    }

    if migration.status == "raw-kernel-only":
        base.update(
            {
                "comparison_mode": RAW_RUST_PYO3_KERNEL,
                "comparison_mode_short": "raw kernel",
                "comparison_mode_label": "raw Rust/PyO3 kernel vs legacy Python oracle",
                "current_entrypoint_kind": "raw_rust_pyo3_numpy_binding",
                "speed_gate_scope": "diagnostic_raw_kernel",
            }
        )
        return base

    if migration.status == "orchestration-rust-default":
        base.update(
            {
                "comparison_mode": PUBLIC_PYTHON_FACADE,
                "comparison_mode_short": "public facade",
                "comparison_mode_label": "current public Python facade vs legacy public Python",
                "current_entrypoint_kind": "public_python_facade_rust_backed",
                "speed_gate_scope": "public_facade_enforced",
            }
        )
        return base

    if migration.status == "public-rust-default":
        if "python" in migration.boundary or "quivr" in migration.boundary:
            base.update(
                {
                    "comparison_mode": PUBLIC_PYTHON_FACADE,
                    "comparison_mode_short": "public facade",
                    "comparison_mode_label": "current public Python facade vs legacy public Python",
                    "current_entrypoint_kind": "public_python_facade_rust_backed",
                    "speed_gate_scope": "public_facade_enforced",
                }
            )
        else:
            base.update(
                {
                    "comparison_mode": THIN_PYTHON_NUMPY_WRAPPER,
                    "comparison_mode_short": "thin wrapper",
                    "comparison_mode_label": "thin Python/NumPy Rust binding vs legacy public Python",
                    "current_entrypoint_kind": "thin_python_numpy_wrapper_to_rust_pyo3",
                    "speed_gate_scope": "thin_wrapper_enforced",
                }
            )
        return base

    if migration.status == "rust-only":
        base.update(
            {
                "comparison_mode": RUST_NATIVE_ONLY,
                "comparison_mode_short": "rust native",
                "comparison_mode_label": "Rust-native diagnostic/baseline",
                "current_entrypoint_kind": "rust_native",
                "speed_gate_scope": "rust_native_diagnostic",
            }
        )
        return base

    base.update(
        {
            "comparison_mode": UNKNOWN,
            "comparison_mode_short": "unknown",
            "comparison_mode_label": "unknown comparison mode",
            "current_entrypoint_kind": "unknown",
            "speed_gate_scope": "unknown",
        }
    )
    return base
