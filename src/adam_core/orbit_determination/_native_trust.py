"""Import-time trust anchors for native OD provenance and execution.

The package module and the extension remain mutable Python objects.  Keep the
objects and entry points accepted when this module is imported, then compare
module dictionaries by identity before any exact-build claim or native OD
call.  Never call a replacement while deciding whether it is trusted.
"""

from __future__ import annotations

from typing import Any

import adam_core as _adam_core_package
from adam_core import _rust_native as _TRUSTED_NATIVE_MODULE

_TRUSTED_ENTRYPOINTS = (
    ("build_identity", _TRUSTED_NATIVE_MODULE.build_identity),
    ("validate_full_od_settings", _TRUSTED_NATIVE_MODULE.validate_full_od_settings),
    ("run_od_ipc", _TRUSTED_NATIVE_MODULE.run_od_ipc),
    (
        "apply_observation_models_ipc",
        _TRUSTED_NATIVE_MODULE.apply_observation_models_ipc,
    ),
)


def validate_native_trust() -> tuple[Any, Any, Any, Any]:
    """Return only the import-time entry points after validating their owners.

    Callers retain only this validator. They do not expose separate mutable
    module globals for the trusted execution functions.
    """
    if vars(_adam_core_package).get("_rust_native") is not _TRUSTED_NATIVE_MODULE:
        raise RuntimeError("adam_core native extension identity changed after import")
    namespace: dict[str, Any] = vars(_TRUSTED_NATIVE_MODULE)
    if any(
        namespace.get(name) is not expected for name, expected in _TRUSTED_ENTRYPOINTS
    ):
        raise RuntimeError("adam_core native OD entry point changed after import")
    return tuple(expected for _, expected in _TRUSTED_ENTRYPOINTS)
