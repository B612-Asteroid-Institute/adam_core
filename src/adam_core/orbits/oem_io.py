try:
    from oem import CURRENT_VERSION as OEM_VERSION
    from oem import OrbitEphemerisMessage
    from oem.components import EphemerisSegment, HeaderSection, MetaDataSection
except ImportError:
    raise ImportError(
        "oem is not installed. This package requires oem to run. Please install it using 'pip install oem'."
    )

import datetime
import logging
from typing import Type

import numpy as np
import pyarrow.compute as pc
import quivr as qv

from adam_core.coordinates.transform import transform_coordinates

from ..coordinates import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.origin import Origin
from ..coordinates.units import (
    convert_cartesian_covariance_km_to_au,
    km_per_s_to_au_per_day,
    km_to_au,
)
from ..propagator import Propagator
from ..time import Timestamp
from . import Orbits

logger = logging.getLogger(__name__)

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

    object_id = orbits.object_id[0].as_py()

    # Of the default OEM frames, we only support EME2000 (equatorial).
    # So let's transform to that frame.
    object_states = orbits.set_column(
        "coordinates",
        transform_coordinates(orbits.coordinates, frame_out="equatorial"),
    )
    object_states = object_states.sort_by("coordinates.time")

    oem_header = {
        "CCSDS_OEM_VERS": OEM_VERSION,
        "CREATION_DATE": datetime.datetime.now().isoformat(),
        "ORIGINATOR": originator,
    }

    oem_frame = _adam_to_oem_frame(object_states.coordinates.frame)

    # Convert origin from ADAM Core format to OEM format
    oem_center = _adam_to_oem_center(object_states.coordinates.origin.code[0].as_py())

    segment_metadata = {
        "OBJECT_NAME": object_id,
        "OBJECT_ID": object_id,
        "CENTER_NAME": oem_center,
        "REF_FRAME": oem_frame,
        "TIME_SYSTEM": object_states.coordinates.time.scale.upper(),
        "START_TIME": object_states.coordinates.time.min().to_iso8601()[0].as_py(),
        "STOP_TIME": object_states.coordinates.time.max().to_iso8601()[0].as_py(),
    }
    metadata_section = MetaDataSection(segment_metadata)

    states = []
    covariances = []
    for orbit_state in object_states:
        # Get values in km and km/s for OEM export
        values_km = orbit_state.coordinates.values_km[0]  # Get first (and only) row

        state = [
            orbit_state.coordinates.time.to_astropy()[0],
            *values_km,  # Already in km and km/s
        ]
        states.append(state)
        if not orbit_state.coordinates.covariance[0].is_all_nan():
            # Get covariance matrix in km units
            matrix_km = orbit_state.coordinates.covariance_km()[0]
            matrix_lt = matrix_km[np.tril_indices(6)].tolist()
            covariance = [
                orbit_state.coordinates.time.to_astropy()[0],
                oem_frame,
                *matrix_lt,
            ]
            covariances.append(covariance)

    states = list(zip(*states))
    covariances = list(zip(*covariances))
    segment = EphemerisSegment(metadata_section, states, covariance_data=covariances)

    header = HeaderSection(oem_header)

    oem_file = OrbitEphemerisMessage(
        header=header,
        segments=[segment],
    )

    oem_file.save_as(output_file)

    return output_file


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

    object_id = orbits.object_id[0].as_py()

    # Assert that output times are unique
    assert len(times) == len(times.unique()), "Times must be unique for each state"

    propagator = propagator_klass()

    object_states = propagator.propagate_orbits(orbits, times, covariance=True)

    # Of the default OEM frames, we only support EME2000 (equatorial).
    # So let's transform to that frame.
    object_states = object_states.set_column(
        "coordinates",
        transform_coordinates(object_states.coordinates, frame_out="equatorial"),
    )
    object_states = object_states.sort_by("coordinates.time")

    oem_header = {
        "CCSDS_OEM_VERS": OEM_VERSION,
        "CREATION_DATE": datetime.datetime.now().isoformat(),
        "ORIGINATOR": originator,
    }

    oem_frame = _adam_to_oem_frame(object_states.coordinates.frame)

    # Convert origin from ADAM Core format to OEM format
    oem_center = _adam_to_oem_center(object_states.coordinates.origin.code[0].as_py())

    segment_metadata = {
        "OBJECT_NAME": object_id,
        "OBJECT_ID": object_id,
        "CENTER_NAME": oem_center,
        "REF_FRAME": oem_frame,
        "TIME_SYSTEM": object_states.coordinates.time.scale.upper(),
        "START_TIME": object_states.coordinates.time.min().to_iso8601()[0].as_py(),
        "STOP_TIME": object_states.coordinates.time.max().to_iso8601()[0].as_py(),
    }
    metadata_section = MetaDataSection(segment_metadata)

    states = []
    covariances = []
    for orbit_state in object_states:
        # Get values in km and km/s for OEM export
        values_km = orbit_state.coordinates.values_km[0]  # Get first (and only) row

        state = [
            orbit_state.coordinates.time.to_astropy()[0],
            *values_km,  # Already in km and km/s
        ]
        states.append(state)
        if not orbit_state.coordinates.covariance[0].is_all_nan():
            # Get covariance matrix in km units
            matrix_km = orbit_state.coordinates.covariance_km()[0]
            matrix_lt = matrix_km[np.tril_indices(6)].tolist()
            covariance = [
                orbit_state.coordinates.time.to_astropy()[0],
                oem_frame,
                *matrix_lt,
            ]
            covariances.append(covariance)

    states = list(zip(*states))
    covariances = list(zip(*covariances))
    segment = EphemerisSegment(metadata_section, states, covariance_data=covariances)

    header = HeaderSection(oem_header)

    oem_file = OrbitEphemerisMessage(
        header=header,
        segments=[segment],
    )

    oem_file.save_as(output_file)

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
    oem_file = OrbitEphemerisMessage.open(input_file)

    orbits = Orbits.empty()

    for i, segment in enumerate(oem_file.segments):
        object_id = segment.metadata["OBJECT_ID"]

        for state in segment:
            # Convert OEM frame and center to ADAM Core format
            frame = _oem_to_adam_frame(state.frame)
            origin = _oem_to_adam_center(state.center)
            time = Timestamp.from_astropy(state.epoch)

            # Convert position and velocity from km/km-s (OEM units) to AU/AU-day (ADAM Core units)
            position_au = km_to_au(np.array(state.position))
            velocity_au_day = km_per_s_to_au_per_day(np.array(state.velocity))

            # We only join covariances that match the epoch and the frame of states
            # TODO: In the future, we should consider alternative modes where we read in entire segments
            # as orbits and solve for the covariance given epochs available.
            adam_cov = CoordinateCovariances.nulls(1)
            for covariance in segment.covariances:
                if covariance.epoch == state.epoch:
                    if covariance.frame == state.frame:
                        # Reshape the covariance matrix to include batch dimension (N, 6, 6)
                        cov_matrix_km = covariance.matrix.reshape(1, 6, 6)
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

            orbit = Orbits.from_kwargs(
                object_id=[object_id],
                orbit_id=[orbit_id],
                coordinates=coordinates,
            )

            orbits = qv.concatenate([orbits, orbit])

    return orbits
