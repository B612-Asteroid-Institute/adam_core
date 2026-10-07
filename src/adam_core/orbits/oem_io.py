"""CCSDS OEM reading and writing.

Two writers share the Rust KVN renderer. :func:`orbit_to_oem` is the legacy
one shot writer, OEM 2.0 with REF_FRAME EME2000 and the state covariance when
present. :class:`OrbitEphemerisMessage` writes OEM 3.0 with the labels spelled
out: ICRF for adam_core's equatorial frame (the J2000 axes SPICE and DE440
deliver, which NAIF aligns with the ICRF), the origin as CENTER_NAME and the
Timestamp scale as TIME_SYSTEM, plus an optional covariance block in REF_FRAME
or a local orbital frame. CCSDS 502.0-B-3 table 5-4 cites RSW, RTN and TNW
(3.2.4.11) for COV_REF_FRAME and annex B5 admits SANA frames such as
VNC_ROTATING, so other labels get a COMMENT line and are refused under
``strict``. :func:`orbit_from_oem` reads either back.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Type, Union

import numpy as np
import pyarrow.compute as pc
import quivr as qv

from adam_core.coordinates.transform import transform_coordinates

from ..coordinates import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.local_orbital_frames import LocalFrameCovariances
from ..coordinates.origin import Origin
from ..coordinates.units import (
    convert_cartesian_covariance_au_to_km,
    convert_cartesian_covariance_km_to_au,
    convert_cartesian_values_au_to_km,
    km_per_s_to_au_per_day,
    km_to_au,
)
from ..time import Timestamp
from .orbits import Orbits

if TYPE_CHECKING:
    from ..propagator import Propagator

logger = logging.getLogger(__name__)

# CCSDS OEM version written by this module. Matches the Python `oem`
# package's CURRENT_VERSION ('2.0'), which this module's Rust KVN engine
# replaced (bead personal-cmy.28).
OEM_VERSION = "2.0"

REF_FRAME_VALUES = (
    "EME2000",  # Earth Mean Equator and Equinox of J2000
    "GCRF",  # Geocentric Celestial Reference Frame
    "GRC",  # Geocentric Reference Frame
    "ICRF",  # International Celestial Reference Frame
    "ITRF2000",  # International Terrestrial Reference Frame
    "ITRF-93",  # International Terrestrial Reference Frame
    "ITRF-97",  # International Terrestrial Reference Frame
    "MCI",  # Mean Celestial Intermediate
    "TDR",  # True of Date
    "TEME",  # True Equator Mean Equinox
    "TOD",  # True of Date
)

CCSDS_CENTER_NAME_VALUES = (
    "101955 BENNU",
    "103P/HARTLEY 2",
    "11351 LEUCUS",
    "132524 APL",
    "15094 POLYMELE",
    "162173 RYUGU",
    "16 PSYCHE",
    "19P/BORRELLY",
    "1 CERES",
    "1P/HALLEY",
    "21900 ORUS",
    "21 LUTETIA",
    "21P/GIACOBINI-ZINNER",
    "243 IDA",
    "25143 ITOKAWA",
    "253 MATHILDE",
    "26P/GRIGG-SKJELLRUP",
    "2867 STEINS",
    "3200 PHAETHON",
    "3548 EURYBATES",
    "4179 TOUTATIS",
    "433 EROS",
    "4 VESTA",
    "52246 DONALDJOHANSON",
    "5525 ANNEFRANK",
    "617 PATROCLUS",
    "65803 DIDYMOS/DIMORPHOS",
    "67P/CHURYUMOV-GERASIMENKO",
    "81P/WILD 2",
    "951 GASPRA",
    "9969 BRAILLE",
    "9P/TEMPEL 1",
    "AMALTHEA",
    "ARIEL",
    "ARROKOTH",
    "ATLAS",
    "CALLISTO",
    "CALYPSO",
    "CHARON",
    "DEIMOS",
    "DIONE",
    "EARTH",
    "EARTH BARYCENTER",
    "EARTH-MOON L1",
    "EARTH-MOON L2",
    "ENCELADUS",
    "EPIMETHEUS",
    "EUROPA",
    "GANYMEDE",
    "HELENE",
    "HYPERION",
    "IAPETUS",
    "IO",
    "JANUS",
    "JUPITER",
    "JUPITER BARYCENTER",
    "LARISSA",
    "MARS",
    "MARS BARYCENTER",
    "MERCURY",
    "MERCURY BARYCENTER",
    "MIMAS",
    "MIRANDA",
    "MOON",
    "NEPTUNE",
    "NEPTUNE BARYCENTER",
    "OBERON",
    "PANDORA",
    "PHOBOS",
    "PHOEBE",
    "PLUTO",
    "PLUTO BARYCENTER",
    "PROTEUS",
    "RHEA",
    "SATURN",
    "SATURN BARYCENTER",
    "SOLAR SYSTEM BARYCENTER",
    "SUN",
    "SUN-EARTH L1",
    "SUN-EARTH L2",
    "TELESTO",
    "TETHYS",
    "TITAN",
    "TITANIA",
    "TRITON",
    "UMBRIEL",
    "URANUS",
    "URANUS BARYCENTER",
    "VENUS",
    "VENUS BARYCENTER",
)


def _adam_to_oem_frame(frame: str) -> str:
    """
    Convert ADAM Core frame to OEM frame.

    Parameters
    ----------
    frame : str
        The ADAM Core frame ('equatorial', 'ecliptic', 'itrf93')

    Returns
    -------
    str
        The corresponding OEM frame

    Raises
    ------
    ValueError
        If the frame is not supported
    """
    frame_map = {
        "equatorial": "EME2000",  # Earth Mean Equator and Equinox of J2000
        "itrf93": "ITRF-93",  # International Terrestrial Reference Frame
    }

    if frame in frame_map:
        return frame_map[frame]
    else:
        raise ValueError(
            f"Unsupported frame for OEM conversion: {frame}. Only 'equatorial' and 'itrf93' are supported."
        )


def _oem_to_adam_frame(frame: str) -> str:
    """
    Convert OEM frame to ADAM Core frame.

    Parameters
    ----------
    frame : str
        The OEM frame

    Returns
    -------
    str
        The corresponding ADAM Core frame

    Raises
    ------
    ValueError
        If the frame is not supported
    """
    frame_map = {
        "EME2000": "equatorial",  # Earth Mean Equator and Equinox of J2000
        "ICRF": "equatorial",  # the J2000 axes within the frame bias adam_core ignores
        "J2000": "equatorial",
        "GCRF": "equatorial",
        "ITRF-93": "itrf93",  # International Terrestrial Reference Frame
    }

    if frame in frame_map:
        return frame_map[frame]
    else:
        raise ValueError(
            f"Unsupported OEM frame: {frame}. Supported frames are {list(frame_map.keys())}."
        )


def _adam_to_oem_center(code: str) -> str:
    """
    Convert ADAM Core origin code to OEM center name.

    Parameters
    ----------
    code : str
        The ADAM Core origin code

    Returns
    -------
    str
        The corresponding OEM center name

    Raises
    ------
    ValueError
        If the origin code is not supported
    """
    center_map = {
        "SOLAR_SYSTEM_BARYCENTER": "SOLAR SYSTEM BARYCENTER",
        "MERCURY_BARYCENTER": "MERCURY BARYCENTER",
        "VENUS_BARYCENTER": "VENUS BARYCENTER",
        "EARTH_MOON_BARYCENTER": "EARTH BARYCENTER",
        "MARS_BARYCENTER": "MARS BARYCENTER",
        "JUPITER_BARYCENTER": "JUPITER BARYCENTER",
        "SATURN_BARYCENTER": "SATURN BARYCENTER",
        "URANUS_BARYCENTER": "URANUS BARYCENTER",
        "NEPTUNE_BARYCENTER": "NEPTUNE BARYCENTER",
        "SUN": "SUN",
        "MERCURY": "MERCURY",
        "VENUS": "VENUS",
        "EARTH": "EARTH",
        "MOON": "MOON",
        "MARS": "MARS",
        "JUPITER": "JUPITER",
        "SATURN": "SATURN",
        "URANUS": "URANUS",
        "NEPTUNE": "NEPTUNE",
    }

    if code in center_map:
        return center_map[code]
    else:
        raise ValueError(f"Unsupported origin code for OEM conversion: {code}")


def _oem_to_adam_center(center: str) -> str:
    """
    Convert OEM center name to ADAM Core origin code.

    Parameters
    ----------
    center : str
        The OEM center name

    Returns
    -------
    str
        The corresponding ADAM Core origin code

    Raises
    ------
    ValueError
        If the center name is not supported
    """
    center_map = {
        "SOLAR SYSTEM BARYCENTER": "SOLAR_SYSTEM_BARYCENTER",
        "MERCURY BARYCENTER": "MERCURY_BARYCENTER",
        "VENUS BARYCENTER": "VENUS_BARYCENTER",
        "EARTH BARYCENTER": "EARTH_MOON_BARYCENTER",  # Note: OEM uses EARTH BARYCENTER for what SPICE calls EMB
        "MARS BARYCENTER": "MARS_BARYCENTER",
        "JUPITER BARYCENTER": "JUPITER_BARYCENTER",
        "SATURN BARYCENTER": "SATURN_BARYCENTER",
        "URANUS BARYCENTER": "URANUS_BARYCENTER",
        "NEPTUNE BARYCENTER": "NEPTUNE_BARYCENTER",
        "SUN": "SUN",
        "MERCURY": "MERCURY",
        "VENUS": "VENUS",
        "EARTH": "EARTH",
        "MOON": "MOON",
        "MARS": "MARS",
        "JUPITER": "JUPITER",
        "SATURN": "SATURN",
        "URANUS": "URANUS",
        "NEPTUNE": "NEPTUNE",
    }
    center_upper = center.upper()

    if center_upper in center_map:
        return center_map[center_upper]
    else:
        raise ValueError(
            f"Unsupported OEM center name: {center}. Supported centers are {list(center_map.keys())}."
        )


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
                    f"COV_REF_FRAME {label} follows the SANA orbit-relative reference "
                    "frames registry (CCSDS 502.0-B-3 annex B5)."
                )
                if strict:
                    raise ValueError(
                        f"{note} It is outside the {', '.join(_OEM_COVARIANCE_FRAMES)} "
                        "set of table 5-4. Pass strict=False to write it anyway."
                    )
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
    """One object about one center at unique epochs, sorted by time (OEM 5.1.3)."""
    ids = orbits.object_id.unique().to_pylist()
    origins = orbits.coordinates.origin.code.unique().to_pylist()
    if len(ids) != 1 or ids[0] is None or len(origins) != 1:
        raise ValueError(
            "An OEM needs an object_id and carries one object about one center per "
            f"file, got object_ids {ids} and origins {origins}."
        )
    if len(orbits.coordinates.time.unique()) != len(orbits):
        raise ValueError("Epochs must be unique within an OEM.")
    return orbits.sort_by(["coordinates.time.days", "coordinates.time.nanos"])


def orbit_to_oem(
    orbits: Orbits,
    output_file: str,
    originator: str = "ADAM CORE USER",
) -> str:
    """
    Convert Orbit object to an OEM file.

    This function converts the state vectors and epoch from an Orbit object into the OEM format.

    Parameters
    ----------
    orbit : Orbit
        The Orbit object to convert, must be pre-propagated to the desired times.
    output_file : str
        Path to the output OEM file

    Returns
    -------
    str
        Path to the output OEM file
    """
    # Check that we have a single object_id
    assert (
        len(orbits.object_id.unique()) == 1
    ), "Only one object_id is supported for OEM conversion."

    assert pc.all(
        pc.invert(pc.is_null(orbits.object_id))
    ).as_py(), "Orbits must specify object_id for oem metadata."

    # If there is only one time, throw a warning
    if len(orbits.coordinates.time.unique()) == 1:
        logger.warning(
            "WARNING: Orbit has only one time, you probably wanted to use orbit_to_oem_propagated instead."
        )

    _write_oem_fused(orbits, output_file, originator)

    return output_file


def _write_oem_fused(orbits: Orbits, output_file: str, originator: str) -> None:
    """One fused Rust crossing owns the equatorial rotation (ecliptic input),
    stable time sort, metadata assembly, AU->km conversion, covariance
    extraction, KVN rendering, and file write (bead personal-cmy.37.4.4).
    The SPICE/time-dependent ITRF93 transform stays on the Rust-owned
    ``transform_coordinates`` crossing; the nondeterministic CREATION_DATE
    stays a Python input like other nondeterministic inputs."""
    from adam_core import _rust_native as _rn

    from .arrow_bridge import orbits_to_ipc

    frame = orbits.coordinates.frame
    if frame not in ("ecliptic", "equatorial", "itrf93"):
        # Exact legacy error from transform_coordinates(frame_out="equatorial").
        raise ValueError("frame should be one of {'ecliptic', 'equatorial', 'itrf93'}")
    if frame == "itrf93":
        orbits = orbits.set_column(
            "coordinates",
            transform_coordinates(orbits.coordinates, frame_out="equatorial"),
        )

    _rn.oem_write_orbits_kvn(
        str(output_file),
        orbits_to_ipc(orbits),
        originator,
        datetime.datetime.now().isoformat(),
    )


def orbit_to_oem_propagated(
    orbits: Orbits,
    output_file: str,
    times: Timestamp,
    propagator_klass: Type[Propagator],
    originator: str = "ADAM CORE USER",
) -> str:
    """
    Convert Orbit object to an OEM file.

    This function converts the state vectors and epoch from an Orbit object into the OEM format.

    Parameters
    ----------
    orbit : Orbit
        The Orbit object to convert
    output_file : str
        Path to the output OEM file

    Returns
    -------
    str
        Path to the output OEM file
    """
    # Check that we have a single object_id
    assert (
        len(orbits.object_id.unique()) == 1
    ), "Only one object_id is supported for OEM conversion."

    assert pc.all(
        pc.invert(pc.is_null(orbits.object_id))
    ).as_py(), "Orbits must specify object_id for oem metadata."

    # Assert that output times are unique
    assert len(times) == len(times.unique()), "Times must be unique for each state"

    propagator = propagator_klass()

    object_states = propagator.propagate_orbits(orbits, times, covariance=True)

    _write_oem_fused(object_states, output_file, originator)

    return output_file


def orbit_from_oem(
    input_file: str,
) -> Orbits:
    """
    Convert an OEM file to an Orbit object.

    This function reads an OEM file and converts the state vectors and epoch into an Orbit object.
    Each state in the oem file is converted to an Orbit row. Covariances are only supported
    if the covariance epoch matches the state epoch.

    Parameters
    ----------
    input_file : str
        Path to the input OEM file

    Returns
    -------
    Orbit
        The Orbit object
    """
    from adam_core import _rust_native as _rn

    from .arrow_bridge import orbits_from_ipc

    # One fused Rust crossing owns parsing, frame/center mapping (exact
    # legacy errors), km->AU conversion, covariance matching, legacy per
    # state orbit ids, and nested Orbits assembly. Files whose segments
    # disagree on frame or time system fall back to the legacy per-state
    # composition so quivr surfaces its own concatenation behavior.
    try:
        raw = _rn.oem_read_orbits_ipc(str(input_file))
    except ValueError as exc:
        # The Python composer also reads the ICRF labels the message writes.
        if "mixed reference frames or time systems" in str(exc) or (
            "Unsupported OEM frame" in str(exc)
        ):
            return _orbit_from_oem_legacy(input_file)
        raise
    if raw is None:
        return Orbits.empty()
    return orbits_from_ipc(raw)


def _orbit_from_oem_legacy(
    input_file: str,
) -> Orbits:
    """Legacy per-state composition, retained only for multi-segment files
    with mixed frames/time systems (rare; preserves exact legacy quivr
    concatenation behavior for that edge)."""
    from adam_core import _rust_native as _rn

    payload = json.loads(_rn.oem_parse_kvn(str(input_file)))

    orbits_list: list[Orbits] = []

    for i, segment in enumerate(payload["segments"]):
        metadata = segment["metadata"]
        object_id = metadata["OBJECT_ID"]

        # Convert OEM frame and center to ADAM Core format (constant per
        # segment, matching the legacy per-state values).
        frame = _oem_to_adam_frame(metadata["REF_FRAME"])
        origin = _oem_to_adam_center(metadata["CENTER_NAME"])
        scale = metadata["TIME_SYSTEM"].lower()

        states = segment["states"]
        for j, (state_days, state_nanos) in enumerate(
            zip(states["days"], states["nanos"])
        ):
            time = Timestamp.from_kwargs(
                days=[state_days], nanos=[state_nanos], scale=scale
            )
            values_km = np.asarray(states["values_km"][j], dtype=np.float64)

            # Convert position and velocity from km/km-s (OEM units) to AU/AU-day (ADAM Core units)
            position_au = km_to_au(values_km[:3])
            velocity_au_day = km_per_s_to_au_per_day(values_km[3:6])

            # We only join covariances that match the epoch and the frame of states
            # TODO: In the future, we should consider alternative modes where we read in entire segments
            # as orbits and solve for the covariance given epochs available.
            adam_cov = CoordinateCovariances.nulls(1)
            for covariance in segment["covariances"]:
                if (
                    covariance["days"] == state_days
                    and covariance["nanos"] == state_nanos
                ):
                    if covariance["frame"] == metadata["REF_FRAME"]:
                        # Reshape the covariance matrix to include batch dimension (N, 6, 6)
                        cov_matrix_km = np.asarray(
                            covariance["matrix"], dtype=np.float64
                        ).reshape(1, 6, 6)
                        # Convert covariance from km units to AU units
                        cov_matrix_au = convert_cartesian_covariance_km_to_au(
                            cov_matrix_km
                        )
                        adam_cov = CoordinateCovariances.from_matrix(cov_matrix_au)

            coordinates = CartesianCoordinates.from_kwargs(
                time=time,
                x=[position_au[0]],
                y=[position_au[1]],
                z=[position_au[2]],
                vx=[velocity_au_day[0]],
                vy=[velocity_au_day[1]],
                vz=[velocity_au_day[2]],
                frame=frame,
                origin=Origin.from_kwargs(code=[origin]),
                covariance=adam_cov,
            )

            orbit_id = f"{object_id}_seg_{i}_{time.to_iso8601()[0].as_py()}"

            orbits_list.append(
                Orbits.from_kwargs(
                    object_id=[object_id],
                    orbit_id=[orbit_id],
                    coordinates=coordinates,
                )
            )

    return qv.concatenate(orbits_list) if orbits_list else Orbits.empty()
