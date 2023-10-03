from typing import List, Union

import numpy.typing as npt
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.cometary import CometaryCoordinates
from ...coordinates.keplerian import KeplerianCoordinates
from ...coordinates.origin import Origin
from ...observers import Observers
from ...time import Timestamp
from ..orbits import Orbits


def _get_horizons_vectors(
    object_ids: Union[List, npt.ArrayLike],
    times: Time,
    location: str = "@sun",
    id_type: str = "smallbody",
    aberrations: str = "geometric",
    refplane: str = "ecliptic",
) -> pd.DataFrame:
    """
    Query JPL Horizons (through astroquery) for an object's
    state vectors at the given times.

    Parameters
    ----------
    object_ids : Union[List, `~numpy.ndarray`] (N)
        Object IDs / designations recognizable by HORIZONS.
    times : `~astropy.core.time.Time` (M)
        Astropy time object at which to gather state vectors.
    location : str, optional
        Location of the origin typically a NAIF code.
        ('0' or '@ssb' for solar system barycenter, '10' or '@sun' for heliocenter)
        [Default = '@sun']
    id_type : {'majorbody', 'smallbody', 'designation',
               'name', 'asteroid_name', 'comet_name', 'id'}
        ID type, Horizons will find closest match under any given type.
    aberrations : {'geometric', 'astrometric', 'apparent'}
        Adjust state for one of three different aberrations.
    refplane : {'ecliptic', 'earth'}
        Reference plane for state vectors.

    Returns
    -------
    vectors : `~pandas.DataFrame`
        Dataframe containing the full cartesian state, r, r_rate, delta, delta_rate and light time
        of the object at each time.
    """
    dfs = []
    for i, obj_id in enumerate(object_ids):
        obj = Horizons(
            id=obj_id,
            epochs=times.tdb.mjd,
            location=location,
            id_type=id_type,
        )
        vectors = obj.vectors(
            refplane=refplane, aberrations=aberrations, cache=False
        ).to_pandas()
        vectors.insert(0, "orbit_id", f"{i:05d}")
        dfs.append(vectors)

    vectors = pd.concat(dfs, ignore_index=True)
    vectors.sort_values(
        by=["orbit_id", "datetime_jd"],
        inplace=True,
        ignore_index=True,
    )
    return vectors


def _get_horizons_elements(
    object_ids: Union[List, npt.ArrayLike],
    times: Time,
    location: str = "@sun",
    id_type: str = "smallbody",
    refplane: str = "ecliptic",
) -> pd.DataFrame:
    """
    Query JPL Horizons (through astroquery) for an object's
    elements at the given times.

    Parameters
    ----------
    object_ids : Union[List, `~numpy.ndarray`] (N)
        Object IDs / designations recognizable by HORIZONS.
    times : `~astropy.core.time.Time`
        Astropy time object at which to gather state vectors.
    location : str, optional
        Location of the origin typically a NAIF code.
        ('0' or '@ssb' for solar system barycenter, '10' or '@sun' for heliocenter)
        [Default = '@sun']
    id_type : {'majorbody', 'smallbody', 'designation',
               'name', 'asteroid_name', 'comet_name', 'id'}
        ID type, Horizons will find closest match under any given type.
    refplane : {'ecliptic', 'earth'}
        Reference plane for orbital elements.

    Returns
    -------
    elements : `~pandas.DataFrame`
        Dataframe containing the full cartesian state, r, r_rate, delta, delta_rate and light time
        of the object at each time.
    """
    dfs = []
    for i, obj_id in enumerate(object_ids):
        obj = Horizons(
            id=obj_id,
            epochs=times.tdb.mjd,
            location=location,
            id_type=id_type,
        )
        elements = obj.elements(
            refsystem="J2000", refplane=refplane, tp_type="absolute", cache=False
        ).to_pandas()
        elements.insert(0, "orbit_id", f"{i:05d}")
        dfs.append(elements)

    elements = pd.concat(dfs, ignore_index=True)
    elements.sort_values(
        by=["orbit_id", "datetime_jd"],
        inplace=True,
        ignore_index=True,
    )
    return elements


def _get_horizons_ephemeris(
    object_ids: Union[List, npt.ArrayLike],
    times: Time,
    location: str,
    id_type: str = "smallbody",
) -> pd.DataFrame:
    """
    Query JPL Horizons (through astroquery) for an object's
    predicted ephemeris as seen from a given location at the given times.

    Parameters
    ----------
    object_ids : Union[List, `~numpy.ndarray`] (N)
        Object IDs / designations recognizable by HORIZONS.
    times : `~astropy.core.time.Time`
        Astropy time object at which to gather state vectors.
    location : str, optional
        Location of the origin typically a NAIF code or MPC observatory code
    id_type : {'majorbody', 'smallbody', 'designation',
               'name', 'asteroid_name', 'comet_name', 'id'}
        ID type, Horizons will find closest match under any given type.

    Returns
    -------
    ephemeris : `~pandas.DataFrame`
        Dataframe containing the predicted ephemerides of the given objects
        as seen from the observer location at the given times.
    """
    dfs = []
    for i, obj_id in enumerate(object_ids):
        obj = Horizons(
            id=obj_id,
            epochs=times.utc.mjd,
            location=location,
            id_type=id_type,
        )
        ephemeris = obj.ephemerides(
            # RA, DEC, r, r_rate, delta, delta_rate, lighttime
            # quantities="1, 2, 19, 20, 21",
            extra_precision=True,
            cache=False,
        ).to_pandas()
        ephemeris.insert(0, "orbit_id", f"{i:05d}")
        ephemeris.insert(2, "mjd_utc", times.utc.mjd)
        ephemeris.insert(3, "observatory_code", location)

        dfs.append(ephemeris)

    ephemeris = pd.concat(dfs)
    ephemeris.sort_values(
        by=["orbit_id", "datetime_jd", "observatory_code"],
        inplace=True,
        ignore_index=True,
    )
    return ephemeris


def query_horizons_ephemeris(
    object_ids: Union[List, npt.ArrayLike], observers: Observers
) -> pd.DataFrame:
    """
    Query JPL Horizons (through astroquery) for an object's predicted ephemeris
    as seen from a given location at the given times.

    Parameters
    ----------
    object_ids : Union[List, `~numpy.ndarray`] (N)
        Object IDs / designations recognizable by HORIZONS.
    observers : `~adam_core.observers.observers.Observers`
        Observers object containing the location and times
        of the observers.

    Returns
    -------
    ephemeris : `~pandas.DataFrame`
        Dataframe containing the predicted ephemerides of the given objects
        as seen from the observer location at the given times.
    """
    dfs = []
    for observatory_code, observers_i in observers.iterate_codes():
        ephemeris = _get_horizons_ephemeris(
            object_ids,
            observers_i.coordinates.time.to_astropy(),
            observatory_code,
        )
        dfs.append(ephemeris)

    ephemeris = pd.concat(dfs, ignore_index=True)
    ephemeris.sort_values(
        by=["orbit_id", "datetime_jd", "observatory_code"],
        inplace=True,
        ignore_index=True,
    )
    return ephemeris


def query_horizons(
    object_ids: Union[List, npt.ArrayLike],
    times: Time,
    coordinate_type: str = "cartesian",
    location: str = "@sun",
    id_type: str = "smallbody",
    aberrations: str = "geometric",
) -> Orbits:
    """
    Query JPL Horizons (through astroquery) for an object's state vectors or elements at the given times.

    Parameters
    ----------
    object_ids : npt.ArrayLike (N)
        Object IDs / designations recognizable by HORIZONS.
    times : `~astropy.core.time.Time` (M)
        Astropy time object at which to gather state vectors.
    coordinate_type : {'cartesian', 'keplerian', 'cometary'}
        Type of orbital elements to return.
    location : str, optional
        Location of the origin typically a NAIF code.
        ('0' or '@ssb' for solar system barycenter, '10' or '@sun' for heliocenter)
        [Default = '@sun']
    id_type : {'majorbody', 'smallbody', 'designation',
                'name', 'asteroid_name', 'comet_name', 'id'}
        ID type, Horizons will find closest match under any given type.
    aberrations : {'geometric', 'astrometric', 'apparent'}
        Adjust state for one of three different aberrations.

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits`
        Orbits object containing the state vectors or elements of the object at each time.
    """
    if coordinate_type == "cartesian":
        vectors = _get_horizons_vectors(
            object_ids,
            times,
            location=location,
            id_type=id_type,
            aberrations=aberrations,
        )

        times = Timestamp.from_jd(vectors["datetime_jd"].values, scale="tdb")
        origin = Origin.from_kwargs(code=["SUN" for i in range(len(times))])
        frame = "ecliptic"
        coordinates = CartesianCoordinates.from_kwargs(
            time=times,
            x=vectors["x"].values,
            y=vectors["y"].values,
            z=vectors["z"].values,
            vx=vectors["vx"].values,
            vy=vectors["vy"].values,
            vz=vectors["vz"].values,
            origin=origin,
            frame=frame,
        )
        orbit_id = vectors["orbit_id"].values
        object_id = vectors["targetname"].values

        return Orbits.from_kwargs(
            orbit_id=orbit_id, object_id=object_id, coordinates=coordinates
        )

    elif coordinate_type == "keplerian":
        elements = _get_horizons_elements(
            object_ids,
            times,
            location=location,
            id_type=id_type,
        )

        times = Timestamp.from_jd(
            elements["datetime_jd"].values,
            scale="tdb",
        )
        origin = Origin.from_kwargs(code=["SUN" for i in range(len(times))])
        frame = "ecliptic"
        coordinates = KeplerianCoordinates(
            time=times,
            a=elements["a"].values,
            e=elements["e"].values,
            i=elements["incl"].values,
            raan=elements["Omega"].values,
            ap=elements["w"].values,
            M=elements["M"].values,
            origin=origin,
            frame=frame,
        )
        orbit_id = elements["orbit_id"].values
        object_id = elements["targetname"].values

        return Orbits.from_kwargs(
            orbit_id=orbit_id,
            object_id=object_id,
            coordinates=coordinates.to_cartesian(),
        )

    elif coordinate_type == "cometary":
        elements = _get_horizons_elements(
            object_ids,
            times,
            location=location,
            id_type=id_type,
        )

        tp = Timestamp.from_jd(elements["Tp_jd"].values, scale="tdb")
        times = Timestamp.from_jd(elements["datetime_jd"].values, scale="tdb")
        origin = Origin.from_kwargs(code=["SUN" for i in range(len(times))])
        frame = "ecliptic"
        coordinates = CometaryCoordinates.from_kwargs(
            time=times,
            q=elements["q"].values,
            e=elements["e"].values,
            i=elements["incl"].values,
            raan=elements["Omega"].values,
            ap=elements["w"].values,
            tp=tp.mjd(),
            origin=origin,
            frame=frame,
        )
        orbit_id = elements["orbit_id"].values
        object_id = elements["targetname"].values

        return Orbits.from_kwargs(
            orbit_id=orbit_id,
            object_id=object_id,
            coordinates=coordinates.to_cartesian(),
        )

    else:
        err = "coordinate_type should be one of {'cartesian', 'keplerian', 'cometary'}"
        raise ValueError(err)
