try:
    from oem import OrbitEphemerisMessage
    from oem.components import EphemerisSegment, HeaderSection, MetaDataSection
except ImportError:
    raise ImportError(
        "oem is not installed. This package requires oem to run. Please install it using 'pip install oem'."
    )

import datetime
from typing import Optional, Type

import quivr as qv

from ..coordinates import CartesianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..propagator import Propagator
from ..time import Timestamp
from . import Orbits

frame_ref_table = {
    "ECLIPJ2000": "ecliptic",
    "J2000": "equatorial",
    "EME2000": "equatorial",
    "equatorial": "equatorial",
    "ecliptic": "ecliptic",
}


def orbit_to_oem(
    orbits: Orbits,
    output_file: str,
    times: Timestamp,
    propagator_klass: Type[Propagator],
    originator: str = "adam_core",
) -> str:
    """
    Convert Orbit object to an OEM file.

    This function converts the state vectors and epoch from an Orbit object into the OEM format.
    The OEM file will contain the spherical coordinates (right ascension, declination, range) and
    their rates, along with any available covariance information.

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
    propagator = propagator_klass()

    results = propagator.propagate_orbits(orbits, times)

    unique_orbit_ids = results.orbit_id.unique()

    oem_header = {
        "CCSDS_OEM_VERS": "1.0",
        "CREATION_DATE": datetime.datetime.now().isoformat(),
        "ORIGINATOR": originator,
    }

    segments = []

    for orbit_id in unique_orbit_ids:
        orbit_states = results.select("orbit_id", orbit_id)

        metadata = {
            "OBJECT_NAME": orbit_id,
            "OBJECT_ID": orbit_id,
            "CENTER_NAME": orbit_states.coordinates.origin.code[0].as_py(),
            # This should be more specific - likely expecting ex J2000
            "REF_FRAME": orbit_states.coordinates.frame,
            "TIME_SYSTEM": orbit_states.coordinates.time.scale.upper(),
            "START_TIME": orbit_states.coordinates.time.min().to_iso8601()[0].as_py(),
            "STOP_TIME": orbit_states.coordinates.time.max().to_iso8601()[0].as_py(),
        }

        metadata_section = MetaDataSection(metadata)

        states = []

        for orbit_state in orbit_states:
            state = [
                orbit_state.coordinates.time.to_astropy()[0],
                *orbit_state.coordinates.values[0],
            ]
            states.append(state)

        states = list(zip(*states))
        segment = EphemerisSegment(metadata_section, states)

        segments.append(segment)

    header = HeaderSection(oem_header)

    oem_file = OrbitEphemerisMessage(
        header=header,
        segments=segments,
    )

    oem_file.save_as(output_file)

    return output_file


def orbit_from_oem(
    input_file: str,
) -> Orbits:
    """
    Convert an OEM file to an Orbit object.

    This function reads an OEM file and converts the state vectors and epoch into an Orbit object.
    The Orbit object will contain the spherical coordinates (right ascension, declination, range) and
    their rates, along with any available covariance information.

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
            frame = state.frame
            origin = state.center
            time = Timestamp.from_astropy(state.epoch)
            position = state.position
            velocity = state.velocity

            coordinates = CartesianCoordinates.from_kwargs(
                time=time,
                x=[position[0]],
                y=[position[1]],
                z=[position[2]],
                vx=[velocity[0]],
                vy=[velocity[1]],
                vz=[velocity[2]],
                frame=frame_ref_table[frame],
                origin=Origin.from_kwargs(code=[origin.upper()]),
            )

            orbit_id = f"{object_id}_seg_{i}_{time.to_iso8601()[0].as_py()}"

            orbit = Orbits.from_kwargs(
                object_id=[object_id],
                orbit_id=[orbit_id],
                coordinates=coordinates,
            )

            orbits = qv.concatenate([orbits, orbit])

    return orbits
