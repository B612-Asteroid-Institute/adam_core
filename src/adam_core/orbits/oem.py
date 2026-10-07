"""CCSDS OEM writer with explicit metadata.

:func:`adam_core.orbits.oem_io.orbit_to_oem` writes OEM 2.0 with REF_FRAME
EME2000. :class:`OrbitEphemerisMessage` writes OEM 3.0 through the same Rust
KVN renderer with the labels spelled out: ICRF for adam_core's equatorial
frame (the J2000 axes SPICE and DE440 deliver, which NAIF aligns with the
ICRF), the origin as CENTER_NAME and the Timestamp scale as TIME_SYSTEM.
``orbit_from_oem`` reads the result back. A covariance block is written on
request, in REF_FRAME or a local orbital frame. CCSDS 502.0-B-3 table 5-4
cites RSW, RTN and TNW (3.2.4.11) for COV_REF_FRAME and its normative annex B5
admits the SANA orbit-relative frames such as VNC_ROTATING, so labels outside
the 3.2.4.11 set get a COMMENT line and are refused under ``strict``.
"""

from __future__ import annotations

import datetime
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from ..coordinates.local_orbital_frames import LocalFrameCovariances
from ..coordinates.units import (
    convert_cartesian_covariance_au_to_km,
    convert_cartesian_values_au_to_km,
)
from .orbits import Orbits

_REF_FRAME = {"equatorial": "ICRF", "itrf93": "ITRF-93"}
_OEM_COVARIANCE_FRAMES = ("RSW", "RTN", "TNW")


@dataclass(frozen=True, eq=False)
class OrbitEphemerisMessage:
    """
    One object's state history (an Orbits table in adam_core units) and the
    CCSDS labels it is written under, for example ``center_name="SUN"``,
    ``ref_frame="ICRF"``, ``time_system="TDB"``. ``creation_date`` is UTC and
    defaults to the write time. ``comments`` follow META_START.
    """

    states: Orbits
    originator: str
    object_name: str
    object_id: str
    center_name: str
    ref_frame: str
    time_system: str
    creation_date: Optional[str] = None
    ccsds_oem_vers: str = "3.0"
    comments: tuple[str, ...] = ()

    @classmethod
    def from_orbits(
        cls,
        orbits: Orbits,
        originator: str,
        *,
        object_name: Optional[str] = None,
        object_id: Optional[str] = None,
        creation_date: Optional[str] = None,
        comments: Sequence[str] = (),
    ) -> "OrbitEphemerisMessage":
        """
        Derive the labels from an Orbits table in the ``"equatorial"`` (written
        as ICRF) or ``"itrf93"`` frame. CENTER_NAME follows the origin and
        TIME_SYSTEM the Timestamp scale, so rescale first for a different one.
        """
        from .oem_io import _adam_to_oem_center

        states = _single_object_sorted(_on_millisecond_grid(orbits))
        coords = states.coordinates
        if coords.frame not in _REF_FRAME:
            raise ValueError(
                f"Frame {coords.frame!r} cannot be written to an OEM, supported "
                f"frames are {list(_REF_FRAME)}. Transform to equatorial first."
            )
        default_id = states.object_id[0].as_py()
        return cls(
            states=states,
            originator=originator,
            object_name=object_name or default_id,
            object_id=object_id or default_id,
            center_name=_adam_to_oem_center(coords.origin.code[0].as_py()),
            ref_frame=_REF_FRAME[coords.frame],
            time_system=coords.time.scale.upper(),
            creation_date=creation_date,
            comments=tuple(comments),
        )

    def write(
        self,
        path: Union[str, os.PathLike],
        covariance_frame: Optional[str] = None,
        strict: bool = False,
    ) -> str:
        """
        Write the KVN file and return its path. ``covariance_frame`` None writes
        no covariance block, the REF_FRAME label writes the state covariance, a
        local orbital frame name or alias writes the rotated covariance under
        that label. ``strict`` refuses labels outside the 3.2.4.11 set.
        """
        from adam_core import _rust_native as _rn

        coords = self.states.coordinates
        epochs = coords.time.to_iso8601().to_pylist()
        comments = list(self.comments)
        records: list = []
        if covariance_frame is not None:
            records, note = self._covariance_records(covariance_frame, strict)
            comments += [note] if note else []
        header = {
            "CCSDS_OEM_VERS": self.ccsds_oem_vers,
            "CREATION_DATE": self.creation_date
            or datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "ORIGINATOR": self.originator,
        }
        metadata = {
            "OBJECT_NAME": self.object_name,
            "OBJECT_ID": self.object_id,
            "CENTER_NAME": self.center_name,
            "REF_FRAME": self.ref_frame,
            "TIME_SYSTEM": self.time_system,
            "START_TIME": epochs[0],
            "STOP_TIME": epochs[-1],
        }
        _rn.oem_write_kvn(
            str(path),
            json.dumps(header),
            json.dumps(metadata),
            coords.time.scale,
            np.ascontiguousarray(coords.time.days.to_numpy(zero_copy_only=False)),
            np.ascontiguousarray(coords.time.nanos.to_numpy(zero_copy_only=False)),
            np.ascontiguousarray(
                convert_cartesian_values_au_to_km(coords.values).ravel()
            ),
            records,
        )
        if comments:
            lines = Path(path).read_text().split("\n")
            at = lines.index("META_START") + 1
            lines[at:at] = [f"COMMENT {comment}" for comment in comments]
            Path(path).write_text("\n".join(lines))
        return str(path)

    def _covariance_records(self, label: str, strict: bool):
        """Per epoch (days, nanos, frame, lower triangle in km) plus a COMMENT or None."""
        coords = self.states.coordinates
        if coords.covariance.is_all_nan():
            raise ValueError("The states carry no covariance.")
        note = None
        label = label.upper()  # normative values are single case (7.5.3)
        if label == self.ref_frame.upper():
            label, matrices = self.ref_frame, coords.covariance.to_matrix()
        else:
            if label not in _OEM_COVARIANCE_FRAMES:
                note = (
                    f"COV_REF_FRAME {label} is a SANA orbit-relative reference frame "
                    "admitted by CCSDS 502.0-B-3 annex B5, outside the "
                    f"{', '.join(_OEM_COVARIANCE_FRAMES)} set of 3.2.4.11 that "
                    "table 5-4 cites."
                )
                if strict:
                    raise ValueError(note + " Pass strict=False to write it anyway.")
            matrices = LocalFrameCovariances.from_orbits(
                self.states, label
            ).covariance.to_matrix()
        matrices_km = convert_cartesian_covariance_au_to_km(matrices)
        days = coords.time.days.to_pylist()
        nanos = coords.time.nanos.to_pylist()
        lower = np.tril_indices(6)
        return [
            (days[i], nanos[i], label, matrices_km[i][lower].tolist())
            for i in range(len(days))
            if not np.isnan(matrices_km[i]).all()
        ], note


def _on_millisecond_grid(orbits: Orbits) -> Orbits:
    time = orbits.coordinates.time
    rounded = time.rounded("ms")
    shift = np.abs(
        rounded.nanos.to_numpy(zero_copy_only=False)
        - time.nanos.to_numpy(zero_copy_only=False)
    )
    if shift.any():
        warnings.warn(
            f"{int((shift > 0).sum())} of {len(time)} epochs were moved onto the "
            f"millisecond grid OEM epochs are written with, the largest by "
            f"{shift.max() / 1e3:.1f} microseconds."
        )
        return orbits.set_column("coordinates.time", rounded)
    return orbits


def _single_object_sorted(orbits: Orbits) -> Orbits:
    """One object with one origin at unique epochs, sorted by time."""
    if len(orbits) == 0:
        raise ValueError("An OEM needs at least one state.")
    if orbits.object_id.null_count:
        raise ValueError("Every state needs an object_id for the OEM metadata.")
    object_ids = orbits.object_id.unique().to_pylist()
    origins = orbits.coordinates.origin.code.unique().to_pylist()
    if len(object_ids) != 1 or len(origins) != 1:
        raise ValueError(
            "An OEM carries one object about one center per file, got object_ids "
            f"{object_ids} and origins {origins}."
        )
    if len(orbits.coordinates.time.unique()) != len(orbits):
        raise ValueError("Epochs must be unique within an OEM.")
    return orbits.sort_by(["coordinates.time.days", "coordinates.time.nanos"])
