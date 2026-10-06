"""CCSDS Orbit Ephemeris Message (OEM) with explicit metadata.

:func:`adam_core.orbits.oem_io.orbit_to_oem` writes one fixed shape of file.
:class:`OrbitEphemerisMessage` makes the CCSDS labels (center, reference
frame, time system, version) explicit, renders through the same Rust KVN
renderer, and reads every header and metadata keyword back. Inside the
message states stay in adam_core units and frames. Only the file is in km.

A covariance block is written on request, either in the OEM reference frame
or rotated into a local orbital frame
(:mod:`adam_core.coordinates.local_orbital_frames`) with its own
``COV_REF_FRAME``. CCSDS 502.0-B-3 lists only RSW, RTN and TNW for that
keyword (table 5-4 via 3.2.4.11), so a VNC label is recorded with a COMMENT
line and refused under ``strict``.
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
from ..coordinates.local_orbital_frames import LocalFrameCovariances
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
    "OEM_COVARIANCE_FRAMES",
    "OEM_REF_FRAME_LABELS",
    "OEM_TIME_SYSTEMS",
    "OemHeader",
    "OemSegment",
    "OemSegmentMetadata",
    "OrbitEphemerisMessage",
]

#: REF_FRAME labels accepted for each adam_core frame. The first is written
#: by default. Every label in a row names the same axes within the precision
#: adam_core models (the ICRF to J2000 frame bias of about 23 mas is not
#: applied anywhere in adam_core).
OEM_REF_FRAME_LABELS = {
    "equatorial": ("ICRF", "EME2000", "J2000", "GCRF"),
    "itrf93": ("ITRF-93",),
}

#: TIME_SYSTEM values mapped to adam_core Timestamp scales.
OEM_TIME_SYSTEMS = {"TDB": "tdb", "TT": "tt", "TAI": "tai", "UTC": "utc"}

#: Local orbital frames the OEM standard lists for COV_REF_FRAME (3.2.4.11).
OEM_COVARIANCE_FRAMES = ("RSW", "RTN", "TNW")

# The Rust KVN renderer writes epochs with three decimal places of seconds.
_EPOCH_PRECISION_NANOS = 1_000_000

# Metadata keywords in CCSDS table 5-3 order. Each is a field of
# OemSegmentMetadata in lower case.
_METADATA_KEYS = (
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
    """OEM header. ``creation_date`` is UTC and defaults to the write time."""

    originator: str
    creation_date: Optional[str] = None
    ccsds_oem_vers: str = "3.0"
    message_id: Optional[str] = None


@dataclass(frozen=True)
class OemSegmentMetadata:
    """
    OEM metadata block for one segment.

    Fields are the CCSDS keywords in lower case and hold CCSDS labels, for
    example ``center_name="SUN"``, ``ref_frame="ICRF"``, ``time_system="TDB"``.
    ``comments`` are written after META_START. ``extra`` keeps keywords this
    module does not model, in file order.
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

    def as_ordered_items(self) -> list[tuple[str, str]]:
        """Metadata as ``(keyword, value)`` pairs in CCSDS table 5-3 order."""
        items = [
            (key, str(getattr(self, key.lower())))
            for key in _METADATA_KEYS
            if getattr(self, key.lower()) is not None
        ]
        return items + [(key, str(value)) for key, value in self.extra]


@dataclass(frozen=True, eq=False)
class OemSegment:
    """
    One OEM segment: a metadata block and the states it describes.

    ``states`` is in adam_core units with the frame, origin and time scale the
    metadata implies. A covariance block in ``REF_FRAME`` is carried on
    ``states.coordinates.covariance``. Covariance records in another frame are
    kept in ``local_covariances`` with the file's label as ``frame``, which
    only :meth:`OrbitEphemerisMessage.from_kvn` populates.
    """

    metadata: OemSegmentMetadata
    states: Orbits
    local_covariances: Optional[LocalFrameCovariances] = None


@dataclass(frozen=True, eq=False)
class OrbitEphemerisMessage:
    """
    A CCSDS OEM: one header and one or more segments.

    Build one with :meth:`from_orbits`, render it with :meth:`to_kvn` or
    :meth:`write`, read one back with :meth:`from_kvn`. Writing supports one
    segment, one object per file. Reading keeps every segment.
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
        time_system: Optional[str] = None,
        creation_date: Optional[str] = None,
        ccsds_oem_vers: str = "3.0",
        message_id: Optional[str] = None,
        metadata_comments: Sequence[str] = (),
    ) -> "OrbitEphemerisMessage":
        """
        Build a single-segment message from a multi-epoch state history.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits` (N)
            States of one object at N unique epochs, for example the output
            of ``propagate_orbits``, in the ``"equatorial"`` or ``"itrf93"``
            frame with a single non-null ``object_id`` and a single origin.
            Rows are sorted by time.
        originator : str
            ORIGINATOR header value.
        object_name, object_id : str, optional
            OBJECT_NAME and OBJECT_ID. Default to the orbits' ``object_id``.
        ref_frame : str, optional
            REF_FRAME label. Defaults to ``"ICRF"`` for equatorial states. An
            override must name the same axes, see :data:`OEM_REF_FRAME_LABELS`.
        time_system : str, optional
            TIME_SYSTEM. Defaults to the Timestamp scale in upper case. A
            different value rescales every epoch.
        creation_date, ccsds_oem_vers, message_id : optional
            Header fields, see :class:`OemHeader`.
        metadata_comments : sequence of str, optional
            COMMENT lines written after META_START.
        """
        states = _single_object_sorted(orbits)
        coords = states.coordinates
        ref_frame = _ref_frame_label(coords.frame, ref_frame)
        time_system = (time_system or coords.time.scale).upper()
        scale = _time_scale(time_system)
        if scale != coords.time.scale:
            states = states.set_column("coordinates.time", coords.time.rescale(scale))
            coords = states.coordinates
        epochs = coords.time.to_iso8601().to_pylist()
        default_id = states.object_id[0].as_py()
        metadata = OemSegmentMetadata(
            object_name=object_name or default_id,
            object_id=object_id or default_id,
            center_name=_center_name(coords),
            ref_frame=ref_frame,
            time_system=time_system,
            start_time=epochs[0],
            stop_time=epochs[-1],
            comments=tuple(metadata_comments),
        )
        header = OemHeader(originator, creation_date, ccsds_oem_vers, message_id)
        return cls(header, (OemSegment(metadata, states),))

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
            Frame of the covariance block. None writes no block. The REF_FRAME
            label writes ``states.coordinates.covariance``. A local orbital
            frame name or alias (``"TNW"``, ``"VNC_ROTATING"``, ...) rotates
            each epoch's covariance into that frame and writes a per record
            COV_REF_FRAME with the label as given. A segment read from a file
            with ``local_covariances`` under that label writes them back
            unchanged.
        strict : bool, optional
            Refuse a covariance frame outside the REF_FRAME set and
            :data:`OEM_COVARIANCE_FRAMES`. Off by default, in which case such
            a frame is written and a COMMENT line records that it is a SANA
            orbit-relative frame outside the OEM set.
        mu : float or array, optional
            Gravitational parameter for the frame rate of ``_ROTATING``
            covariance frames. Defaults to the origin's value.
        allow_epoch_rounding : bool, optional
            Epochs are written with millisecond precision. By default an epoch
            off that grid raises so a state is never silently shifted in time.
        """
        if len(self.segments) != 1:
            raise NotImplementedError(
                "Writing is supported for messages with exactly one segment "
                f"(one object per file), got {len(self.segments)}."
            )
        segment = self.segments[0]
        states = _single_object_sorted(segment.states)
        coords = states.coordinates
        metadata = _checked_metadata(segment.metadata, coords, allow_epoch_rounding)

        records: list = []
        comments = list(metadata.comments)
        if covariance_frame is not None:
            records, note = _covariance_records(
                states,
                metadata.ref_frame,
                covariance_frame,
                strict,
                mu,
                segment.local_covariances,
            )
            comments += [note] if note else []

        header = [
            ("CCSDS_OEM_VERS", self.header.ccsds_oem_vers),
            ("CREATION_DATE", self.header.creation_date or _utc_now()),
            ("ORIGINATOR", self.header.originator),
        ]
        if self.header.message_id is not None:
            header.append(("MESSAGE_ID", self.header.message_id))
        text = _render_kvn(header, metadata.as_ordered_items(), coords, records)
        if comments:
            lines = text.split("\n")
            at = lines.index("META_START") + 1
            lines[at:at] = [f"COMMENT {comment}" for comment in comments]
            text = "\n".join(lines)
        return text

    def write(self, path: Union[str, os.PathLike], **kwargs) -> str:
        """Write KVN to ``path``. Keyword arguments are those of :meth:`to_kvn`."""
        Path(path).write_text(self.to_kvn(**kwargs))
        return str(path)

    @classmethod
    def from_kvn(cls, path: Union[str, os.PathLike]) -> "OrbitEphemerisMessage":
        """
        Read a KVN OEM file.

        Every header and metadata keyword is kept (the parser drops COMMENT
        lines). States are converted to AU and AU/day in the adam_core frame
        matching REF_FRAME. Covariance records at a state epoch attach to
        ``states.coordinates.covariance`` when in REF_FRAME, otherwise they
        become the segment's ``local_covariances`` with the file's label.
        Raises ``ValueError`` for labels this module does not map or for more
        than one non REF_FRAME covariance frame in a segment.
        """
        from adam_core import _rust_native as _rn

        payload = json.loads(_rn.oem_parse_kvn(str(path)))
        raw = payload.get("header", {})
        header = OemHeader(
            originator=str(raw.get("ORIGINATOR", "")),
            creation_date=_optional_str(raw.get("CREATION_DATE")),
            ccsds_oem_vers=str(raw.get("CCSDS_OEM_VERS", "")),
            message_id=_optional_str(raw.get("MESSAGE_ID")),
        )
        return cls(header, tuple(_segment_from_payload(s) for s in payload["segments"]))

    def to_orbits(self) -> Orbits:
        """States of every segment as one Orbits table. Segments must share a frame and time scale."""
        tables = [segment.states for segment in self.segments]
        if not tables:
            return Orbits.empty()
        frames = {table.coordinates.frame for table in tables}
        scales = {table.coordinates.time.scale for table in tables}
        if len(frames) > 1 or len(scales) > 1:
            raise ValueError(
                f"Segments have mixed frames or time scales ({sorted(frames)}, "
                f"{sorted(scales)}), convert them before concatenating."
            )
        return tables[0] if len(tables) == 1 else qv.concatenate(tables)


# --- labels -----------------------------------------------------------------------


def _ref_frame_label(frame: str, label: Optional[str]) -> str:
    if frame not in OEM_REF_FRAME_LABELS:
        hint = (
            " Transform to the equatorial frame first." if frame == "ecliptic" else ""
        )
        raise ValueError(
            f"Frame {frame!r} cannot be written to an OEM. Supported frames are "
            f"{list(OEM_REF_FRAME_LABELS)}.{hint}"
        )
    allowed = OEM_REF_FRAME_LABELS[frame]
    if label is None:
        return allowed[0]
    if label.upper() not in allowed:
        raise ValueError(
            f"REF_FRAME {label!r} does not name the axes of adam_core frame "
            f"{frame!r}. Allowed labels are {list(allowed)}."
        )
    return label.upper()


def _adam_frame(label: str) -> str:
    for frame, labels in OEM_REF_FRAME_LABELS.items():
        if label.strip().upper() in labels:
            return frame
    raise ValueError(
        f"Unsupported OEM REF_FRAME: {label!r}. Supported labels are "
        f"{[x for labels in OEM_REF_FRAME_LABELS.values() for x in labels]}."
    )


def _time_scale(system: str) -> str:
    try:
        return OEM_TIME_SYSTEMS[system.strip().upper()]
    except KeyError:
        raise ValueError(
            f"Unsupported OEM TIME_SYSTEM: {system!r}. Supported values are "
            f"{list(OEM_TIME_SYSTEMS)}."
        ) from None


def _center_name(coords: CartesianCoordinates) -> str:
    from .oem_io import _adam_to_oem_center

    return _adam_to_oem_center(coords.origin.code[0].as_py())


def _optional_str(value) -> Optional[str]:
    return None if value is None else str(value)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# --- state checks -----------------------------------------------------------------


def _single_object_sorted(orbits: Orbits) -> Orbits:
    """One object, one origin, unique epochs, sorted by time."""
    if len(orbits) == 0:
        raise ValueError("An OEM needs at least one state.")
    if not pc.all(pc.is_valid(orbits.object_id)).as_py():
        raise ValueError("Every state needs an object_id for the OEM metadata.")
    object_ids = orbits.object_id.unique().to_pylist()
    if len(object_ids) != 1:
        raise ValueError(f"An OEM carries one object per file, got {object_ids}.")
    origins = orbits.coordinates.origin.code.unique().to_pylist()
    if len(origins) != 1:
        raise ValueError(
            f"An OEM segment has one CENTER_NAME, so every state needs the same "
            f"origin, got {origins}."
        )
    days = orbits.coordinates.time.days.to_numpy(zero_copy_only=False)
    nanos = orbits.coordinates.time.nanos.to_numpy(zero_copy_only=False)
    order = np.lexsort((nanos, days))
    epochs = np.stack([days[order], nanos[order]], axis=1)
    if (epochs[1:] == epochs[:-1]).all(axis=1).any():
        raise ValueError("Epochs must be unique within an OEM segment.")
    if np.array_equal(order, np.arange(len(orbits))):
        return orbits
    return orbits.take(pa.array(order))


def _checked_metadata(
    metadata: OemSegmentMetadata, coords: CartesianCoordinates, allow_rounding: bool
) -> OemSegmentMetadata:
    """Check the metadata against the states and fill START_TIME and STOP_TIME."""
    for name, expected, actual in (
        ("REF_FRAME", coords.frame, _adam_frame(metadata.ref_frame)),
        ("CENTER_NAME", _center_name(coords).upper(), metadata.center_name.upper()),
        ("TIME_SYSTEM", coords.time.scale, _time_scale(metadata.time_system)),
    ):
        if expected != actual:
            raise ValueError(
                f"Metadata {name} {getattr(metadata, name.lower())!r} does not "
                "match the states."
            )
    days = coords.time.days.to_numpy(zero_copy_only=False)
    nanos = coords.time.nanos.to_numpy(zero_copy_only=False)
    off_grid = nanos % _EPOCH_PRECISION_NANOS != 0
    if off_grid.any():
        if not allow_rounding:
            raise ValueError(
                f"Epoch at row {int(np.flatnonzero(off_grid)[0])} is not on a "
                "millisecond boundary and OEM epochs are written with three "
                "decimal places of seconds. Pass allow_epoch_rounding=True to round."
            )
        rounded = np.stack([days, np.round(nanos / _EPOCH_PRECISION_NANOS)], axis=1)
        if len(np.unique(rounded, axis=0)) != len(rounded):
            raise ValueError("Two epochs fall in the same millisecond.")
    epochs = coords.time.to_iso8601().to_pylist()
    return replace(
        metadata,
        start_time=metadata.start_time or epochs[0],
        stop_time=metadata.stop_time or epochs[-1],
    )


# --- covariance block -----------------------------------------------------------


def _covariance_records(
    states: Orbits,
    ref_frame: str,
    label: str,
    strict: bool,
    mu,
    stored: Optional[LocalFrameCovariances],
) -> tuple[list[tuple[int, int, str, list[float]]], Optional[str]]:
    """Per epoch (days, nanos, frame, lower triangle in km) plus an optional COMMENT."""
    label = label.strip()
    coords = states.coordinates
    days = coords.time.days.to_pylist()
    nanos = coords.time.nanos.to_pylist()
    note = None
    if label.upper() == ref_frame.upper():
        if coords.covariance.is_all_nan():
            raise ValueError("The states carry no covariance.")
        label, matrices = ref_frame, coords.covariance.to_matrix()
    else:
        if label.upper() not in OEM_COVARIANCE_FRAMES:
            note = (
                f"COV_REF_FRAME {label} is a SANA orbit-relative reference frame "
                "outside the OEM covariance frame set of CCSDS 502.0-B-3 "
                f"section 3.2.4.11 ({', '.join(OEM_COVARIANCE_FRAMES)})."
            )
            if strict:
                raise ValueError(note + " Pass strict=False to write it anyway.")
        if stored is not None and stored.frame.upper() == label.upper():
            by_epoch = dict(
                zip(
                    zip(stored.time.days.to_pylist(), stored.time.nanos.to_pylist()),
                    stored.covariance.to_matrix(),
                )
            )
            missing = np.full((6, 6), np.nan)
            matrices = np.array(
                [by_epoch.get(epoch, missing) for epoch in zip(days, nanos)]
            )
        else:
            matrices = LocalFrameCovariances.from_orbits(
                states, label, mu
            ).covariance.to_matrix()

    matrices_km = convert_cartesian_covariance_au_to_km(matrices)
    lower = np.tril_indices(6)
    return [
        (days[i], nanos[i], label, matrices_km[i][lower].tolist())
        for i in range(len(states))
        if not np.isnan(matrices_km[i]).all()
    ], note


# --- rendering and parsing ----------------------------------------------------


def _render_kvn(header, metadata, coords: CartesianCoordinates, records) -> str:
    """Render through the Rust KVN renderer, which owns epoch and float formatting."""
    from adam_core import _rust_native as _rn

    handle, path = tempfile.mkstemp(suffix=".oem")
    os.close(handle)
    try:
        _rn.oem_write_kvn(
            path,
            json.dumps(dict(header)),
            json.dumps(dict(metadata)),
            coords.time.scale,
            np.ascontiguousarray(coords.time.days.to_numpy(zero_copy_only=False)),
            np.ascontiguousarray(coords.time.nanos.to_numpy(zero_copy_only=False)),
            np.ascontiguousarray(
                convert_cartesian_values_au_to_km(coords.values).ravel()
            ),
            records,
        )
        return Path(path).read_text()
    finally:
        os.unlink(path)


def _segment_from_payload(raw: dict) -> OemSegment:
    meta = dict(raw["metadata"])
    known = {key.lower(): meta.pop(key, None) for key in _METADATA_KEYS}
    for required in ("object_id", "center_name", "ref_frame", "time_system"):
        if known[required] is None:
            raise ValueError(f"OEM segment is missing {required.upper()}.")
    degree = known.pop("interpolation_degree")
    metadata = OemSegmentMetadata(
        **{key: _optional_str(value) for key, value in known.items()},
        interpolation_degree=None if degree is None else int(degree),
        extra=tuple((key, str(value)) for key, value in meta.items()),
    )
    if metadata.object_name is None:
        metadata = replace(metadata, object_name=metadata.object_id)

    from .oem_io import _oem_to_adam_center

    frame = _adam_frame(metadata.ref_frame)
    origin_code = _oem_to_adam_center(metadata.center_name)
    scale = _time_scale(metadata.time_system)

    days = np.asarray(raw["states"]["days"], dtype=np.int64)
    nanos = np.asarray(raw["states"]["nanos"], dtype=np.int64)
    n = len(days)
    values = convert_cartesian_values_km_to_au(
        np.asarray(raw["states"]["values_km"], dtype=np.float64).reshape(n, 6)
    )
    time = Timestamp.from_kwargs(days=days, nanos=nanos, scale=scale)

    # Covariance records at a state epoch: REF_FRAME ones attach to the states,
    # the others become a local frame product.
    epoch_index = {(int(d), int(ns)): i for i, (d, ns) in enumerate(zip(days, nanos))}
    state_cov = np.full((n, 6, 6), np.nan)
    local_cov = np.full((n, 6, 6), np.nan)
    local_label = None
    for record in raw.get("covariances", []):
        index = epoch_index.get((int(record["days"]), int(record["nanos"])))
        if index is None:
            continue
        label = record.get("frame") or metadata.ref_frame
        matrix = convert_cartesian_covariance_km_to_au(
            np.asarray(record["matrix"], dtype=np.float64).reshape(1, 6, 6)
        )[0]
        if label.upper() == metadata.ref_frame.upper():
            state_cov[index] = matrix
        elif local_label in (None, label):
            local_label = label
            local_cov[index] = matrix
        else:
            raise ValueError(
                "OEM segment has covariance records in more than one non "
                f"REF_FRAME frame: {local_label!r} and {label!r}."
            )

    origin = Origin.from_kwargs(code=[origin_code] * n)
    states = Orbits.from_kwargs(
        orbit_id=[metadata.object_id] * n,
        object_id=[metadata.object_id] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=time,
            covariance=CoordinateCovariances.from_matrix(state_cov),
            origin=origin,
            frame=frame,
        ),
    )
    local_covariances = None
    if local_label is not None:
        local_covariances = LocalFrameCovariances.from_kwargs(
            orbit_id=states.orbit_id,
            object_id=states.object_id,
            time=time,
            covariance=CoordinateCovariances.from_matrix(local_cov),
            origin=origin,
            frame=local_label,
            reference_frame=frame,
        )
    return OemSegment(metadata, states, local_covariances)
