from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("adam_core-photometry")
    group.addoption(
        "--photometry-fixtures-verbose",
        action="store_true",
        default=False,
        help="Print per-filter residual summary tables for photometry regression fixtures.",
    )
    group.addoption(
        "--photometry-fixtures-h-source",
        action="store",
        default="both",
        choices=("both", "mpc"),
        help="For MPC fixtures, print residuals for 'mpc' H only, or 'both' (mpc + jpl).",
    )
    group.addoption(
        "--run-rotation-period-real-data",
        action="store_true",
        default=False,
        help=(
            "Run optional rotation-period regression tests built from real MPC mirror observations. "
            "These fixtures are curated and slower than the synthetic unit tests."
        ),
    )
    group.addoption(
        "--run-rotation-period-pds",
        action="store_true",
        default=False,
        help=(
            "Run optional rotation-period regression tests built from PDS LCDB/ALCDEF fixtures. "
            "These are offline, curated, and slower than the synthetic unit tests."
        ),
    )
