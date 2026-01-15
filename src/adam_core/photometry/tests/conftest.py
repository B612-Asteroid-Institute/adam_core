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
