from typing import List, Union

import numpy.typing as npt
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.cometary import CometaryCoordinates
from ...coordinates.coordinates import Coordinates
from ...coordinates.keplerian import KeplerianCoordinates
from ...utils.astropy import _check_times
from ..orbits import Orbits


def _get_horizons_vectors(
    object_ids: Union[List, npt.ArrayLike],
    times: Time,
    location: str = "@sun",
    id_type: str = "smallbody",
    aberrations: str = "geometric",
) -> pd.DataFrame:
    """
    Query JPL Horizons (through astroquery) for an object's
    state vectors at the given times.

    Parameters
    ----------
    object_ids : `~numpy.ndarray` (N)
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

    Returns
    -------
    vectors : `~pandas.DataFrame`
        Dataframe containing the full cartesian state, r, r_rate, delta, delta_rate and light time
        of the object at each time.
    """
    _check_times(times, "times")
    dfs = []
    for obj_id in object_ids:
        obj = Horizons(
            id=obj_id,
            epochs=times.tdb.mjd,
            location=location,
            id_type=id_type,
        )
        vectors = obj.vectors(
            refplane="ecliptic", aberrations=aberrations, cache=False
        ).to_pandas()
        dfs.append(vectors)

    vectors = pd.concat(dfs, ignore_index=True)
    return vectors


def _get_horizons_elements(
    object_ids: Union[List, npt.ArrayLike],
    times: Time,
    location: str = "@sun",
    id_type: str = "smallbody",
) -> pd.DataFrame:
    """
    Query JPL Horizons (through astroquery) for an object's
    elements at the given times.

    Parameters
    ----------
    object_ids : `~numpy.ndarray` (N)
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

    Returns
    -------
    elements : `~pandas.DataFrame`
        Dataframe containing the full cartesian state, r, r_rate, delta, delta_rate and light time
        of the object at each time.
    """
    _check_times(times, "times")
    dfs = []
    for obj_id in object_ids:
        obj = Horizons(
            id=obj_id,
            epochs=times.tdb.mjd,
            location=location,
            id_type=id_type,
        )
        elements = obj.elements(
            refsystem="J2000", refplane="ecliptic", tp_type="absolute", cache=False
        ).to_pandas()
        dfs.append(elements)

    elements = pd.concat(dfs, ignore_index=True)
    return elements


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
    coordinates: Coordinates
    if coordinate_type == "cartesian":
        vectors = _get_horizons_vectors(
            object_ids,
            times,
            location=location,
            id_type=id_type,
            aberrations=aberrations,
        )

        coordinates = CartesianCoordinates(
            times=Time(vectors["datetime_jd"].values, scale="tdb", format="jd"),
            x=vectors["x"].values,
            y=vectors["y"].values,
            z=vectors["z"].values,
            vx=vectors["vx"].values,
            vy=vectors["vy"].values,
            vz=vectors["vz"].values,
            origin="heliocenter",
            frame="ecliptic",
        )
        object_ids = vectors["targetname"].values

        return Orbits(coordinates, object_ids=object_ids)

    elif coordinate_type == "keplerian":
        elements = _get_horizons_elements(
            object_ids,
            times,
            location=location,
            id_type=id_type,
        )

        coordinates = KeplerianCoordinates(
            times=Time(elements["datetime_jd"].values, scale="tdb", format="jd"),
            a=elements["a"].values,
            e=elements["e"].values,
            i=elements["incl"].values,
            raan=elements["Omega"].values,
            ap=elements["w"].values,
            M=elements["M"].values,
            origin="heliocenter",
            frame="ecliptic",
        )
        object_ids = elements["targetname"].values

        return Orbits(coordinates, object_ids=object_ids)

    elif coordinate_type == "cometary":
        elements = _get_horizons_elements(
            object_ids,
            times,
            location=location,
            id_type=id_type,
        )

        tp = Time(elements["Tp_jd"].values, scale="tdb", format="jd")
        coordinates = CometaryCoordinates(
            times=Time(elements["datetime_jd"].values, scale="tdb", format="jd"),
            q=elements["q"].values,
            e=elements["e"].values,
            i=elements["incl"].values,
            raan=elements["Omega"].values,
            ap=elements["w"].values,
            tp=tp.tdb.mjd,
            origin="heliocenter",
            frame="ecliptic",
        )
        object_ids = elements["targetname"].values
        return Orbits(coordinates, object_ids=object_ids)

    else:
        err = "coordinate_type should be one of {'cartesian', 'keplerian', 'cometary'}"
        raise ValueError(err)
