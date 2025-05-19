try:
    from oem import OrbitEphemerisMessage, Ephemeris, OrbitEphemerisHeader, State, EphemerisSegment, HeaderSection
except ImportError:
    raise ImportError("oem is not installed. This package requires oem to run. Please install it using 'pip install oem'.")

from typing import Optional, Type
from ..coordinates import Timestamp, CartesianCoordinates
from ..dynamics import Propagator
from ..orbits import Orbits
from ..orbits.ephemeris import Ephemeris
import quivr as qv
import datetime

def orbit_to_oem(
    orbits: Orbits,
    output_file: str,
    times: Timestamp,
    propagator_klass: Type[Propagator],
    comment: str = "OEM file generated from adam_core Orbit",
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
    comment : str, optional
        Comment to include in the OEM file header

    Returns
    -------
    str
        Path to the output OEM file
    """
    propagator = propagator_klass()

    results = propagator.propagate_orbits(orbits, times)

    unique_orbit_ids = results.orbit_id.unique()

    oem_header = {
        "CCSDS_OEM_VERS": "2.0",
        "CREATION_DATE": datetime.now().isoformat(),
        "ORIGINATOR": originator,
        "COMMENT": comment,
    }

    segments = []

    for orbit_id in unique_orbit_ids:
        orbit_states = results.select("orbit_id", orbit_id)

        metadata = {
            "OBJECT_NAME": orbit_id,
            "OBJECT_ID": orbit_id,
            "CENTER_NAME": orbit_states.coordinates.origin.code,
            "REF_FRAME": orbit_states.coordinates.frame,
            "TIME_SYSTEM": orbit_states.coordinates.time.scale,
            "START_TIME": orbit_states.coordinates.time.min().to_iso8601()[0].as_py(),
            "STOP_TIME": orbit_states.coordinates.time.max().to_iso8601()[0].as_py(),
        }

        states = []

        for orbit_state in orbit_states:
            state = [orbit_state.coordinates.time.to_astropy(),*orbit_state.coordinates.values]
            states.append(state)
        
        segment = EphemerisSegment(
            states=states,
            header=metadata,
        )

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

    oem_file = OrbitEphemerisMessage()
    oem_file.open(input_file)

    orbits = Orbits.empty()

    for segment in oem_file.segments:
        for state in segment:
            object_id = state.object_id
            frame = state.frame
            origin = state.origin
            time = Timestamp.from_astropy(state.epoch)
            position = state.position
            velocity = state.velocity


            coordinates = CartesianCoordinates.from_kwargs(
                time=time,
                x = position[0],
                y = position[1],
                z = position[2],
                vx = velocity[0],
                vy = velocity[1],
                vz = velocity[2],
                frame=frame,
                origin=origin,
            )

            orbit_id = object_id + "_" + time.to_iso()

            orbit = Orbits.from_kwargs(
                object_id=object_id,
                orbit_id=orbit_id,
                coordinates=coordinates,
            )

            orbits = qv.concatenate([orbits, orbit])

    return orbits

