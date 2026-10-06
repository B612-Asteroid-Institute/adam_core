"""CCSDS Orbit Ephemeris Message (OEM) with explicit metadata.

The helpers in :mod:`adam_core.orbits.oem_io` write one fixed shape of file.
This module adds a typed message whose CCSDS labels (center, reference frame,
time system, version) are explicit, and a reader that keeps every header and
metadata key it finds. Inside the message, states stay in adam_core units and
frames. Only the rendered file is in kilometres and kilometres per second.

A state covariance in the OEM reference frame can be written into the
covariance block. A covariance in a local orbital frame (see
:mod:`adam_core.coordinates.local_orbital_frames`) can also be written there
with its own ``COV_REF_FRAME`` label. The CCSDS orbit data message standard
(502.0-B-3, section 3.2.4.11 via table 5-4) lists only ``RSW``, ``RTN`` and
``TNW`` as local covariance frames for an OEM, so a ``VNC`` label is a
registered SANA frame name outside that set. The writer records that with a
comment line and can refuse it when ``strict`` is set.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.local_orbital_frames import (
    LocalFrameCovariances,
    resolve_local_orbital_frame,
)
from ..coordinates.origin import Origin
from ..coordinates.units import (
    convert_cartesian_covariance_au_to_km,
    convert_cartesian_covariance_km_to_au,
    convert_cartesian_values_au_to_km,
    convert_cartesian_values_km_to_au,
)
from ..time import Timestamp
from .orbits import Orbits

__all__ = [
    "EPOCH_PRECISION_NANOS",
    "OEM_COVARIANCE_FRAMES",
    "OEM_DEFAULT_REF_FRAME",
    "OEM_REF_FRAME_LABELS",
    "OEM_TIME_SYSTEMS",
    "OemHeader",
    "OemSegment",
    "OemSegmentMetadata",
    "OrbitEphemerisMessage",
]

#: OEM reference frame labels accepted for each adam_core frame on read, and
#: allowed as overrides on write. Every label in a row names the same axes
#: within the precision adam_core models (the ICRF to J2000 frame bias of
#: about 23 milliarcseconds is not applied anywhere in adam_core).
OEM_REF_FRAME_LABELS: dict[str, tuple[str, ...]] = {
    "equatorial": ("ICRF", "EME2000", "J2000", "GCRF"),
    "itrf93": ("ITRF-93",),
}

#: Label written for each adam_core frame when the caller gives none.
OEM_DEFAULT_REF_FRAME: dict[str, str] = {
    "equatorial": "ICRF",
    "itrf93": "ITRF-93",
}

#: OEM TIME_SYSTEM values mapped to adam_core Timestamp scales.
OEM_TIME_SYSTEMS: dict[str, str] = {
    "TDB": "tdb",
    "TT": "tt",
    "TAI": "tai",
    "UTC": "utc",
}

#: Local orbital frames the OEM standard lists for COV_REF_FRAME (3.2.4.11).
OEM_COVARIANCE_FRAMES: tuple[str, ...] = ("RSW", "RTN", "TNW")

#: Epoch precision of the KVN renderer, in nanoseconds. Epochs are written
#: with three decimal places of seconds.
EPOCH_PRECISION_NANOS = 1_000_000

_METADATA_ORDER = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "REF_FRAME_EPOCH",
    "TIME_SYSTEM",
    "START_TIME",
    "USEABLE_START_TIME",
    "USEABLE_STOP_TIME",
    "STOP_TIME",
    "INTERPOLATION",
    "INTERPOLATION_DEGREE",
)


@dataclass(frozen=True)
class OemHeader:
    """
    OEM header block.

    Parameters
    ----------
    originator : str
        ORIGINATOR, the organisation producing the file.
    creation_date : str, optional
        CREATION_DATE in UTC as ``YYYY-MM-DDTHH:MM:SS``. When None the write
        time in UTC is used.
    ccsds_oem_vers : str, optional
        CCSDS_OEM_VERS. Default ``"3.0"``, the current standard.
    message_id : str, optional
        MESSAGE_ID, optional in OEM 3.0.
    comments : tuple of str, optional
        COMMENT lines written after the version line.
    """

    originator: str
    creation_date: Optional[str] = None
    ccsds_oem_vers: str = "3.0"
    message_id: Optional[str] = None
    comments: tuple[str, ...] = ()


@dataclass(frozen=True)
class OemSegmentMetadata:
    """
    OEM metadata block for one segment.

    Every field is the CCSDS keyword of the same name in lower case. Labels are
    CCSDS labels, for example ``center_name="SUN"``, ``ref_frame="ICRF"``,
    ``time_system="TDB"``. Times are rendered epoch strings in ``time_system``.
    ``extra`` keeps keywords this module does not model, in file order, so a
    read file can be written back without losing them.
    """

    object_name: str
    object_id: str
    center_name: str
    ref_frame: str
    time_system: str
    start_time: Optional[str] = None
    stop_time: Optional[str] = None
    useable_start_time: Optional[str] = None
    useable_stop_time: Optional[str] = None
    ref_frame_epoch: Optional[str] = None
    interpolation: Optional[str] = None
    interpolation_degree: Optional[int] = None
    comments: tuple[str, ...] = ()
    extra: tuple[tuple[str, str], ...] = ()

    def adam_frame(self) -> str:
        """adam_core frame matching ``ref_frame``."""
        return _oem_frame_to_adam(self.ref_frame)

    def adam_origin_code(self) -> str:
        """adam_core origin code matching ``center_name``."""
        # Imported here because oem_io imports the orbits package, which
        # imports this module.
        from .oem_io import _oem_to_adam_center

        return _oem_to_adam_center(self.center_name)

    def adam_time_scale(self) -> str:
        """adam_core Timestamp scale matching ``time_system``."""
        return _oem_time_system_to_scale(self.time_system)

    def as_ordered_items(self) -> list[tuple[str, str]]:
        """Metadata as ``(keyword, value)`` pairs in CCSDS table 5-3 order."""
        values = {
            "OBJECT_NAME": self.object_name,
            "OBJECT_ID": self.object_id,
            "CENTER_NAME": self.center_name,
            "REF_FRAME": self.ref_frame,
            "REF_FRAME_EPOCH": self.ref_frame_epoch,
            "TIME_SYSTEM": self.time_system,
            "START_TIME": self.start_time,
            "USEABLE_START_TIME": self.useable_start_time,
            "USEABLE_STOP_TIME": self.useable_stop_time,
            "STOP_TIME": self.stop_time,
            "INTERPOLATION": self.interpolation,
            "INTERPOLATION_DEGREE": self.interpolation_degree,
        }
        items = [
            (key, str(values[key]))
            for key in _METADATA_ORDER
            if values[key] is not None
        ]
        items.extend((key, str(value)) for key, value in self.extra)
        return items


@dataclass(frozen=True, eq=False)
class OemSegment:
    """
    One OEM segment: a metadata block and the states it describes.

    ``states`` is an :class:`~adam_core.orbits.orbits.Orbits` table in
    adam_core units with the frame, origin and time scale implied by the
    metadata. A covariance block whose frame equals ``REF_FRAME`` is carried
    on ``states.coordinates.covariance``. Covariance records in another frame
    are kept in ``local_covariances`` with the file's label as ``frame``.
    ``local_covariances`` is only populated by :meth:`OrbitEphemerisMessage.from_kvn`.
    """

    metadata: OemSegmentMetadata
    states: Orbits
    local_covariances: Optional[LocalFrameCovariances] = None


@dataclass(frozen=True, eq=False)
class OrbitEphemerisMessage:
    """
    A CCSDS OEM: one header and one or more segments.

    Build one from propagated states with :meth:`from_orbits`, render it with
    :meth:`to_kvn` or :meth:`write`, and read one back with :meth:`from_kvn`.
    Writing supports one segment, which is one object per file. Reading keeps
    every segment.
    """

    header: OemHeader
    segments: tuple[OemSegment, ...] = field(default_factory=tuple)

    @classmethod
    def from_orbits(
        cls,
        orbits: Orbits,
        originator: str,
        *,
        object_name: Optional[str] = None,
        object_id: Optional[str] = None,
        ref_frame: Optional[str] = None,
        center_name: Optional[str] = None,
        time_system: Optional[str] = None,
        creation_date: Optional[str] = None,
        ccsds_oem_vers: str = "3.0",
        message_id: Optional[str] = None,
        comments: Sequence[str] = (),
        metadata_comments: Sequence[str] = (),
        interpolation: Optional[str] = None,
        interpolation_degree: Optional[int] = None,
    ) -> "OrbitEphemerisMessage":
        """
        Build a single-segment message from a multi-epoch state history.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            States of one object at N epochs, for example the output of
            ``propagate_orbits``. ``object_id`` must be set and single valued.
            The frame must be ``"equatorial"`` or ``"itrf93"``. Ecliptic
            states raise, transform them first. Rows are sorted by time and
            epochs must be unique.
        originator : str
            ORIGINATOR header value.
        object_name, object_id : str, optional
            OBJECT_NAME and OBJECT_ID. Default to the orbits' ``object_id``.
        ref_frame : str, optional
            REF_FRAME label. Default ``"ICRF"`` for equatorial states and
            ``"ITRF-93"`` for ITRF93 states. An override must name the same
            axes, see :data:`OEM_REF_FRAME_LABELS`.
        center_name : str, optional
            CENTER_NAME. Default derived from the orbits' origin, for example
            ``"SUN"``.
        time_system : str, optional
            TIME_SYSTEM. Default is the Timestamp scale of the orbits in upper
            case. A different value rescales every epoch before writing.
        creation_date, ccsds_oem_vers, message_id, comments :
            Header fields, see :class:`OemHeader`.
        metadata_comments : sequence of str, optional
            COMMENT lines written after META_START.
        interpolation, interpolation_degree : optional
            INTERPOLATION and INTERPOLATION_DEGREE metadata.

        Returns
        -------
        message : `OrbitEphemerisMessage`
        """
        states = _single_object_sorted(orbits)
        coords = states.coordinates

        from .oem_io import _adam_to_oem_center

        frame_label = _adam_frame_to_oem(coords.frame, ref_frame)
        center = center_name or _adam_to_oem_center(
            coords.origin.code.unique()[0].as_py()
        )
        scale = coords.time.scale
        system = (time_system or scale).upper()
        target_scale = _oem_time_system_to_scale(system)
        if target_scale != scale:
            states = states.set_column(
                "coordinates.time", coords.time.rescale(target_scale)
            )
            states = _single_object_sorted(states)
            coords = states.coordinates

        default_id = states.object_id[0].as_py()
        epochs = coords.time.to_iso8601().to_pylist()
        metadata = OemSegmentMetadata(
            object_name=object_name or default_id,
            object_id=object_id or default_id,
            center_name=center,
            ref_frame=frame_label,
            time_system=system,
            start_time=epochs[0],
            stop_time=epochs[-1],
            interpolation=interpolation,
            interpolation_degree=interpolation_degree,
            comments=tuple(metadata_comments),
        )
        header = OemHeader(
            originator=originator,
            creation_date=creation_date,
            ccsds_oem_vers=ccsds_oem_vers,
            message_id=message_id,
            comments=tuple(comments),
        )
        return cls(header=header, segments=(OemSegment(metadata, states),))

    def to_kvn(
        self,
        *,
        covariance_frame: Optional[str] = None,
        strict: bool = False,
        mu: Optional[Union[float, np.ndarray]] = None,
        allow_epoch_rounding: bool = False,
    ) -> str:
        """
        Render the message as KVN text.

        Parameters
        ----------
        covariance_frame : str, optional
            Frame of the covariance block. None writes no covariance block.
            The segment's REF_FRAME label writes ``states.coordinates.covariance``
            as is. A local orbital frame name or alias (``"TNW"``,
            ``"VNC_ROTATING"``, ...) rotates each epoch's covariance into that
            frame and writes it with a per record COV_REF_FRAME. The label is
            written exactly as given. When the segment was read from a file
            and carries ``local_covariances`` under that same label, that
            block is written back unchanged instead of being recomputed.
        strict : bool, optional
            When True, refuse a covariance frame outside the set the OEM
            standard lists (the REF_FRAME set plus RSW, RTN, TNW). When False,
            the default, such a frame is written and a COMMENT line in the
            metadata block records that it is a SANA orbit-relative frame
            outside that set.
        mu : float or array, optional
            Gravitational parameter for the frame rate of ``_ROTATING``
            covariance frames. Defaults to the origin's value.
        allow_epoch_rounding : bool, optional
            The renderer writes epochs with millisecond precision. By default
            an epoch that is not on a millisecond boundary raises so a state
            is never silently placed at a shifted time. Set True to round.

        Returns
        -------
        str
            The KVN text.
        """
        if len(self.segments) != 1:
            raise NotImplementedError(
                "Writing is supported for messages with exactly one segment "
                f"(one object per file), got {len(self.segments)}."
            )
        segment = self.segments[0]
        metadata = segment.metadata
        states = _single_object_sorted(segment.states)
        coords = states.coordinates
        _check_frame_matches(coords.frame, metadata.ref_frame)
        _check_center_matches(coords, metadata.center_name)
        scale = _oem_time_system_to_scale(metadata.time_system)
        if coords.time.scale != scale:
            raise ValueError(
                f"States are in time scale {coords.time.scale!r} but the metadata "
                f"TIME_SYSTEM is {metadata.time_system!r}."
            )
        _check_epoch_precision(coords.time, allow_epoch_rounding)

        epochs = coords.time.to_iso8601().to_pylist()
        metadata = replace(
            metadata,
            start_time=metadata.start_time or epochs[0],
            stop_time=metadata.stop_time or epochs[-1],
        )

        records: list[tuple[int, int, str, list[float]]] = []
        metadata_comments = list(metadata.comments)
        if covariance_frame is not None:
            records, note = _covariance_records(
                states,
                metadata.ref_frame,
                covariance_frame,
                strict,
                mu,
                segment.local_covariances,
            )
            if note is not None:
                metadata_comments.append(note)

        header_items = [("CCSDS_OEM_VERS", self.header.ccsds_oem_vers)]
        header_items.append(
            (
                "CREATION_DATE",
                self.header.creation_date or _utc_now_iso(),
            )
        )
        header_items.append(("ORIGINATOR", self.header.originator))
        if self.header.message_id is not None:
            header_items.append(("MESSAGE_ID", self.header.message_id))

        text = _render_kvn(
            header_items,
            metadata.as_ordered_items(),
            coords,
            records,
        )
        return _insert_comments(text, self.header.comments, metadata_comments)

    def write(self, path: Union[str, os.PathLike], **kwargs) -> str:
        """
        Write the message to ``path`` as KVN. Keyword arguments are those of
        :meth:`to_kvn`. Returns the path as a string.
        """
        text = self.to_kvn(**kwargs)
        Path(path).write_text(text)
        return str(path)

    @classmethod
    def from_kvn(cls, path: Union[str, os.PathLike]) -> "OrbitEphemerisMessage":
        """
        Read a KVN OEM file.

        Every header and metadata keyword is kept. COMMENT lines are not,
        the parser drops them. States are converted to AU and AU/day in the
        adam_core frame matching REF_FRAME. Covariance records in REF_FRAME
        attach to ``states.coordinates.covariance``. Records in a different
        frame become a :class:`~adam_core.coordinates.local_orbital_frames.LocalFrameCovariances`
        on the segment with the file's label as ``frame``.

        Raises
        ------
        ValueError
            For a REF_FRAME, CENTER_NAME or TIME_SYSTEM this module does not
            map, or for covariance records in more than one non REF_FRAME
            frame within a segment.
        """
        from adam_core import _rust_native as _rn

        payload = json.loads(_rn.oem_parse_kvn(str(path)))
        raw_header = payload.get("header", {})
        header = OemHeader(
            originator=str(raw_header.get("ORIGINATOR", "")),
            creation_date=_optional_str(raw_header.get("CREATION_DATE")),
            ccsds_oem_vers=str(raw_header.get("CCSDS_OEM_VERS", "")),
            message_id=_optional_str(raw_header.get("MESSAGE_ID")),
        )
        segments = tuple(_segment_from_payload(raw) for raw in payload["segments"])
        return cls(header=header, segments=segments)

    def to_orbits(self) -> Orbits:
        """
        States of every segment as one Orbits table.

        Raises
        ------
        ValueError
            If segments disagree on frame or time scale, because the
            concatenated table carries one of each.
        """
        if not self.segments:
            return Orbits.empty()
        tables = [segment.states for segment in self.segments]
        frames = {table.coordinates.frame for table in tables}
        scales = {table.coordinates.time.scale for table in tables}
        if len(frames) > 1 or len(scales) > 1:
            raise ValueError(
                "Segments have mixed frames or time scales "
                f"(frames {sorted(frames)}, scales {sorted(scales)}), "
                "convert them before concatenating."
            )
        if len(tables) == 1:
            return tables[0]
        return qv.concatenate(tables)


# --- label maps -----------------------------------------------------------------


def _adam_frame_to_oem(frame: str, label: Optional[str]) -> str:
    if frame not in OEM_REF_FRAME_LABELS:
        hint = (
            " Transform to the equatorial frame first." if frame == "ecliptic" else ""
        )
        raise ValueError(
            f"Frame {frame!r} cannot be written to an OEM. Supported adam_core "
            f"frames are {list(OEM_REF_FRAME_LABELS)}.{hint}"
        )
    if label is None:
        return OEM_DEFAULT_REF_FRAME[frame]
    allowed = OEM_REF_FRAME_LABELS[frame]
    if label.upper() not in allowed:
        raise ValueError(
            f"REF_FRAME {label!r} does not name the axes of adam_core frame "
            f"{frame!r}. Allowed labels are {list(allowed)}."
        )
    return label.upper()


def _oem_frame_to_adam(label: str) -> str:
    upper = label.strip().upper()
    for frame, labels in OEM_REF_FRAME_LABELS.items():
        if upper in labels:
            return frame
    supported = [label for labels in OEM_REF_FRAME_LABELS.values() for label in labels]
    raise ValueError(
        f"Unsupported OEM REF_FRAME: {label!r}. Supported labels are {supported}."
    )


def _check_frame_matches(frame: str, label: str) -> None:
    if _oem_frame_to_adam(label) != frame:
        raise ValueError(
            f"States are in frame {frame!r} but the metadata REF_FRAME is {label!r}."
        )


def _oem_time_system_to_scale(system: str) -> str:
    upper = system.strip().upper()
    if upper not in OEM_TIME_SYSTEMS:
        raise ValueError(
            f"Unsupported OEM TIME_SYSTEM: {system!r}. Supported values are "
            f"{list(OEM_TIME_SYSTEMS)}."
        )
    return OEM_TIME_SYSTEMS[upper]


def _optional_str(value) -> Optional[str]:
    return None if value is None else str(value)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# --- state checks -----------------------------------------------------------------


def _single_object_sorted(orbits: Orbits) -> Orbits:
    if len(orbits) == 0:
        raise ValueError("An OEM needs at least one state.")
    if not pc.all(pc.is_valid(orbits.object_id)).as_py():
        raise ValueError("Every state needs an object_id for the OEM metadata.")
    object_ids = orbits.object_id.unique()
    if len(object_ids) != 1:
        raise ValueError(
            "An OEM carries one object per file, got object_ids "
            f"{object_ids.to_pylist()}."
        )
    origins = orbits.coordinates.origin.code.unique()
    if len(origins) != 1:
        raise ValueError(
            "An OEM segment has one CENTER_NAME, so every state needs the same "
            f"origin, got {origins.to_pylist()}."
        )
    time = orbits.coordinates.time
    days = time.days.to_numpy(zero_copy_only=False)
    nanos = time.nanos.to_numpy(zero_copy_only=False)
    order = np.lexsort((nanos, days))
    sorted_days = days[order]
    sorted_nanos = nanos[order]
    duplicate = (sorted_days[1:] == sorted_days[:-1]) & (
        sorted_nanos[1:] == sorted_nanos[:-1]
    )
    if duplicate.any():
        raise ValueError("Epochs must be unique within an OEM segment.")
    if np.array_equal(order, np.arange(len(orbits))):
        return orbits
    return orbits.take(pa.array(order, type=pa.int64()))


def _check_epoch_precision(time: Timestamp, allow_rounding: bool) -> None:
    days = time.days.to_numpy(zero_copy_only=False)
    nanos = time.nanos.to_numpy(zero_copy_only=False)
    off_grid = nanos % EPOCH_PRECISION_NANOS != 0
    if off_grid.any():
        if not allow_rounding:
            row = int(np.flatnonzero(off_grid)[0])
            raise ValueError(
                f"Epoch at row {row} is not on a millisecond boundary and the OEM "
                "renderer writes three decimal places of seconds. Choose epochs on "
                "millisecond boundaries or pass allow_epoch_rounding=True."
            )
        rounded = np.round(nanos / EPOCH_PRECISION_NANOS).astype(np.int64)
        keys = days * (86_400_000 + 1) + rounded
        if len(np.unique(keys)) != len(keys):
            raise ValueError(
                "Two epochs fall in the same millisecond, so the rendered OEM "
                "would carry duplicate epochs. Thin the states before writing."
            )


def _check_center_matches(coords: CartesianCoordinates, center_name: str) -> None:
    from .oem_io import _adam_to_oem_center

    expected = _adam_to_oem_center(coords.origin.code.unique()[0].as_py())
    if center_name.strip().upper() != expected.upper():
        raise ValueError(
            f"States have origin {coords.origin.code.unique().to_pylist()} "
            f"(CENTER_NAME {expected}) but the metadata CENTER_NAME is "
            f"{center_name!r}."
        )


# --- covariance block -----------------------------------------------------------


def _is_standard_covariance_frame(label: str, ref_frame: str) -> bool:
    upper = label.strip().upper()
    return upper == ref_frame.upper() or upper in OEM_COVARIANCE_FRAMES


def _lower_triangle(matrix: np.ndarray) -> list[float]:
    rows, cols = np.tril_indices(6)
    return [float(value) for value in matrix[rows, cols]]


def _covariance_records(
    states: Orbits,
    ref_frame: str,
    covariance_frame: str,
    strict: bool,
    mu: Optional[Union[float, np.ndarray]],
    stored: Optional[LocalFrameCovariances] = None,
) -> tuple[list[tuple[int, int, str, list[float]]], Optional[str]]:
    label = covariance_frame.strip()
    note: Optional[str] = None
    coords = states.coordinates
    if label.upper() == ref_frame.upper():
        if coords.covariance.is_all_nan():
            raise ValueError(
                f"covariance_frame={covariance_frame!r} was requested but the "
                "states carry no covariance."
            )
        matrices = coords.covariance.to_matrix()
        label = ref_frame
    else:
        if not _is_standard_covariance_frame(label, ref_frame):
            message = (
                f"COV_REF_FRAME {label} is a SANA orbit-relative reference frame "
                "outside the OEM covariance frame set of CCSDS 502.0-B-3 "
                f"section 3.2.4.11 ({', '.join(OEM_COVARIANCE_FRAMES)})."
            )
            if strict:
                raise ValueError(message + " Pass strict=False to write it anyway.")
            note = message
        if stored is not None and stored.frame.upper() == label.upper():
            # A segment read from a file keeps its local-frame block here.
            # Writing the same label writes that block back unchanged.
            if len(stored) != len(states):
                raise ValueError(
                    "local_covariances must be row aligned with the states."
                )
            matrices = stored.to_matrix()
        else:
            canonical = resolve_local_orbital_frame(label)
            if coords.covariance.is_all_nan():
                raise ValueError(
                    f"covariance_frame={covariance_frame!r} was requested but the "
                    "states carry no covariance."
                )
            matrices = LocalFrameCovariances.from_orbits(
                states, frame=canonical, mu=mu
            ).to_matrix()

    matrices_km = convert_cartesian_covariance_au_to_km(matrices)
    days = coords.time.days.to_numpy(zero_copy_only=False)
    nanos = coords.time.nanos.to_numpy(zero_copy_only=False)
    records = []
    for i in range(len(states)):
        if np.isnan(matrices_km[i]).all():
            continue
        records.append(
            (int(days[i]), int(nanos[i]), label, _lower_triangle(matrices_km[i]))
        )
    return records, note


# --- rendering --------------------------------------------------------------------


def _render_kvn(
    header_items: list[tuple[str, str]],
    metadata_items: list[tuple[str, str]],
    coords: CartesianCoordinates,
    records: list[tuple[int, int, str, list[float]]],
) -> str:
    """Render through the Rust KVN renderer, which owns epoch and float formatting."""
    from adam_core import _rust_native as _rn

    header_json = json.dumps(dict(header_items))
    metadata_json = json.dumps(dict(metadata_items))
    days = np.ascontiguousarray(coords.time.days.to_numpy(zero_copy_only=False))
    nanos = np.ascontiguousarray(coords.time.nanos.to_numpy(zero_copy_only=False))
    states_km = np.ascontiguousarray(
        convert_cartesian_values_au_to_km(coords.values).reshape(-1), dtype=np.float64
    )
    handle, path = tempfile.mkstemp(suffix=".oem")
    os.close(handle)
    try:
        _rn.oem_write_kvn(
            path,
            header_json,
            metadata_json,
            coords.time.scale,
            days,
            nanos,
            states_km,
            records,
        )
        return Path(path).read_text()
    finally:
        os.unlink(path)


def _insert_comments(
    text: str, header_comments: Sequence[str], metadata_comments: Sequence[str]
) -> str:
    if not header_comments and not metadata_comments:
        return text
    lines = text.split("\n")
    out: list[str] = []
    for index, line in enumerate(lines):
        out.append(line)
        if index == 0 and header_comments and line.startswith("CCSDS_OEM_VERS"):
            out.extend(f"COMMENT {comment}" for comment in header_comments)
        elif line == "META_START" and metadata_comments:
            out.extend(f"COMMENT {comment}" for comment in metadata_comments)
    return "\n".join(out)


# --- parsing ----------------------------------------------------------------------


def _segment_from_payload(raw: dict) -> OemSegment:
    meta = {str(key): value for key, value in raw["metadata"].items()}
    known = {key: meta.pop(key, None) for key in _METADATA_ORDER}
    for required in ("OBJECT_ID", "CENTER_NAME", "REF_FRAME", "TIME_SYSTEM"):
        if known[required] is None:
            raise ValueError(f"OEM segment is missing {required}.")
    degree = known["INTERPOLATION_DEGREE"]
    metadata = OemSegmentMetadata(
        object_name=str(known["OBJECT_NAME"] or known["OBJECT_ID"]),
        object_id=str(known["OBJECT_ID"]),
        center_name=str(known["CENTER_NAME"]),
        ref_frame=str(known["REF_FRAME"]),
        time_system=str(known["TIME_SYSTEM"]),
        start_time=_optional_str(known["START_TIME"]),
        stop_time=_optional_str(known["STOP_TIME"]),
        useable_start_time=_optional_str(known["USEABLE_START_TIME"]),
        useable_stop_time=_optional_str(known["USEABLE_STOP_TIME"]),
        ref_frame_epoch=_optional_str(known["REF_FRAME_EPOCH"]),
        interpolation=_optional_str(known["INTERPOLATION"]),
        interpolation_degree=None if degree is None else int(degree),
        extra=tuple((key, str(value)) for key, value in meta.items()),
    )

    frame = metadata.adam_frame()
    origin_code = metadata.adam_origin_code()
    scale = metadata.adam_time_scale()

    states = raw["states"]
    days = np.asarray(states["days"], dtype=np.int64)
    nanos = np.asarray(states["nanos"], dtype=np.int64)
    n = len(days)
    values_km = np.asarray(states["values_km"], dtype=np.float64).reshape(n, 6)
    values_au = convert_cartesian_values_km_to_au(values_km)
    time = Timestamp.from_kwargs(days=days, nanos=nanos, scale=scale)

    # Covariance records keyed by epoch. Records in REF_FRAME attach to the
    # states. Records in another frame become a local-frame product.
    epoch_index = {(int(d), int(ns)): i for i, (d, ns) in enumerate(zip(days, nanos))}
    state_cov = np.full((n, 6, 6), np.nan)
    local_cov = np.full((n, 6, 6), np.nan)
    local_label: Optional[str] = None
    has_state_cov = False
    for record in raw.get("covariances", []):
        key = (int(record["days"]), int(record["nanos"]))
        if key not in epoch_index:
            continue
        label = record.get("frame") or metadata.ref_frame
        matrix_km = np.asarray(record["matrix"], dtype=np.float64).reshape(1, 6, 6)
        matrix_au = convert_cartesian_covariance_km_to_au(matrix_km)[0]
        if label.upper() == metadata.ref_frame.upper():
            state_cov[epoch_index[key]] = matrix_au
            has_state_cov = True
        else:
            if local_label is not None and label != local_label:
                raise ValueError(
                    "OEM segment has covariance records in more than one "
                    f"non REF_FRAME frame: {local_label!r} and {label!r}."
                )
            local_label = label
            local_cov[epoch_index[key]] = matrix_au

    covariance = (
        CoordinateCovariances.from_matrix(state_cov)
        if has_state_cov
        else CoordinateCovariances.nulls(n)
    )
    origin = Origin.from_kwargs(code=[origin_code] * n)
    coordinates = CartesianCoordinates.from_kwargs(
        x=values_au[:, 0],
        y=values_au[:, 1],
        z=values_au[:, 2],
        vx=values_au[:, 3],
        vy=values_au[:, 4],
        vz=values_au[:, 5],
        time=time,
        covariance=covariance,
        origin=origin,
        frame=frame,
    )
    orbits = Orbits.from_kwargs(
        orbit_id=[metadata.object_id] * n,
        object_id=[metadata.object_id] * n,
        coordinates=coordinates,
    )

    local_covariances = None
    if local_label is not None:
        local_covariances = LocalFrameCovariances.from_kwargs(
            orbit_id=orbits.orbit_id,
            object_id=orbits.object_id,
            time=time,
            covariance=CoordinateCovariances.from_matrix(local_cov),
            origin=origin,
            frame=local_label,
            reference_frame=frame,
        )
    return OemSegment(metadata, orbits, local_covariances)
