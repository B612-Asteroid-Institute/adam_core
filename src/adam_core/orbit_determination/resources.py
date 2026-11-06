"""Installed public resource paths for explicit OD configuration binding.

This module performs no acquisition. Python consumers may use the immutable
public data package path below; standalone Rust callers continue to supply the
same semantic resource path directly.
"""

from __future__ import annotations

from pathlib import Path

from .config import ResourceBindings


def core_od_v1_resource_bindings() -> ResourceBindings:
    """Return the installed public resources required by ``CORE_OD_V1``.

    The exact package requirement is declared by adam-core. Rust still checks
    the file's product identity, byte size, SHA-256, and schema when the profile
    is bound; this helper only locates the installed immutable file.
    """

    import jpl_debias_2018

    return ResourceBindings(
        {"efcc18_catalog_bias_table": Path(jpl_debias_2018.bias_dat)}
    )


__all__ = ["core_od_v1_resource_bindings"]
