"""Dependency-free PEP 517 backend for this tiny documentation example."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

NAME = "acme_orbit_models"
VERSION = "3.2.0"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"
METADATA = f"""Metadata-Version: 2.1
Name: acme-orbit-models
Version: {VERSION}
Summary: Negative live-Python provenance fixture for adam-core
Requires-Python: >=3.11
Requires-Dist: adam-core>=0.7.0
""".encode()
WHEEL = b"""Wheel-Version: 1.0
Generator: acme-example-backend
Root-Is-Purelib: true
Tag: py3-none-any
"""


def get_requires_for_build_wheel(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    destination = Path(metadata_directory, DIST_INFO)
    destination.mkdir(parents=True)
    destination.joinpath("METADATA").write_bytes(METADATA)
    destination.joinpath("WHEEL").write_bytes(WHEEL)
    return DIST_INFO


def _record(path: str, payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"{path},sha256={digest.decode()},{len(payload)}"


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    root = Path(__file__).parent
    files = {
        f"{NAME}/__init__.py": root.joinpath("src", NAME, "__init__.py").read_bytes(),
        f"{NAME}/models.py": root.joinpath("src", NAME, "models.py").read_bytes(),
        f"{DIST_INFO}/METADATA": METADATA,
        f"{DIST_INFO}/WHEEL": WHEEL,
    }
    records = [_record(path, payload) for path, payload in files.items()]
    records.append(f"{DIST_INFO}/RECORD,,")
    files[f"{DIST_INFO}/RECORD"] = ("\n".join(records) + "\n").encode()
    wheel = f"{NAME}-{VERSION}-py3-none-any.whl"
    with ZipFile(Path(wheel_directory, wheel), "w", ZIP_DEFLATED) as archive:
        for path, payload in files.items():
            archive.writestr(path, payload)
    return wheel
