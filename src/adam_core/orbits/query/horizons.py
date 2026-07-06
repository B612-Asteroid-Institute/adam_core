import json
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import quivr as qv
from astroquery.jplhorizons import Horizons

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.cometary import CometaryCoordinates
from ...coordinates.keplerian import KeplerianCoordinates
from ...coordinates.origin import Origin
from ...coordinates.spherical import SphericalCoordinates
from ...observers import Observers
from ...time import Timestamp
from ...utils.iter import _iterate_chunks
from ..ephemeris import Ephemeris
from ..orbits import Orbits


def _get_horizons_vectors(
    object_ids: Union[List, npt.ArrayLike],
    times: Timestamp,
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
    times : Timestamp (M)
        Time at which to gather state vectors.
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
            epochs=times.rescale("tdb").jd().to_numpy(zero_copy_only=False),
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
    times: Timestamp,
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
    times : Timestamp
        Time at which to gather state vectors.
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
            epochs=times.rescale("tdb").mjd().to_numpy(zero_copy_only=False),
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
    times: Timestamp,
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
    times : Timestamp
        Time at which to gather state vectors.
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
    jd_utc = times.rescale("utc").jd().to_numpy(zero_copy_only=False)
    for i, obj_id in enumerate(object_ids):
        obj = Horizons(
            id=obj_id,
            epochs=jd_utc,
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
        ephemeris.insert(2, "jd_utc", jd_utc)
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
) -> Ephemeris:
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
        _ephemeris = _get_horizons_ephemeris(
            object_ids,
            observers_i.coordinates.time,
            observatory_code,
        )
        dfs.append(_ephemeris)

    dfs = pd.concat(dfs, ignore_index=True)
    dfs.sort_values(
        by=["orbit_id", "datetime_jd", "observatory_code"],
        inplace=True,
        ignore_index=True,
    )

    from adam_core import _rust_native as _rn

    normalized = json.loads(
        _rn.query_horizons_ephemeris_normalize(dfs.to_json(orient="records"))
    )
    ephemeris = Ephemeris.from_kwargs(
        orbit_id=normalized["orbit_id"],
        object_id=normalized["object_id"],
        light_time=normalized["light_time"],
        alpha=normalized["alpha"],
        coordinates=SphericalCoordinates.from_kwargs(
            time=Timestamp.from_jd(
                pa.array(normalized["times_jd"], type=pa.float64()), scale="utc"
            ),
            lon=normalized["lon"],
            lat=normalized["lat"],
            origin=Origin.from_kwargs(code=normalized["observatory_code"]),
            frame="ecliptic",
        ),
    )

    return ephemeris


def query_horizons(
    object_ids: Union[List, npt.ArrayLike],
    times: Timestamp,
    coordinate_type: str = "cartesian",
    location: str = "@sun",
    aberrations: str = "geometric",
    id_type: Optional[str] = None,
) -> Orbits:
    """
    Query JPL Horizons (through astroquery) for an object's state vectors or elements at the given times.

    Parameters
    ----------
    object_ids : npt.ArrayLike (N)
        Object IDs / designations recognizable by HORIZONS.
    times : Timestamp (M)
        Time at which to gather state vectors.
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
    chunk_size = (
        50  # This is based on the Horizon's limit of 50-75 times before it fails
    )
    total_orbits_list: list[Orbits] = []
    assert len(times) > 0, "Must have at least one time"

    # Sort times to make sure they are in order
    times = times.sort_by(["days", "nanos"])

    for times_i in _iterate_chunks(times, chunk_size):

        if coordinate_type == "cartesian":
            vectors = _get_horizons_vectors(
                object_ids,
                times_i,
                location=location,
                id_type=id_type,
                aberrations=aberrations,
            )

            from adam_core import _rust_native as _rn

            normalized = json.loads(
                _rn.query_horizons_vectors_normalize(vectors.to_json(orient="records"))
            )
            coords = np.asarray(normalized["coords_cartesian"], dtype=np.float64)
            times = Timestamp.from_jd(
                pa.array(normalized["times_jd"], type=pa.float64()), scale="tdb"
            )
            origin = Origin.from_kwargs(code=["SUN" for _ in range(len(times))])
            frame = "ecliptic"
            coordinates = CartesianCoordinates.from_kwargs(
                time=times,
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                vx=coords[:, 3],
                vy=coords[:, 4],
                vz=coords[:, 5],
                origin=origin,
                frame=frame,
            )

            orbits = Orbits.from_kwargs(
                orbit_id=normalized["orbit_id"],
                object_id=normalized["object_id"],
                coordinates=coordinates,
            )

            total_orbits_list.append(orbits)

        elif coordinate_type == "keplerian":
            elements = _get_horizons_elements(
                object_ids,
                times_i,
                location=location,
                id_type=id_type,
            )

            from adam_core import _rust_native as _rn

            normalized = json.loads(
                _rn.query_horizons_elements_normalize(
                    elements.to_json(orient="records"), "keplerian"
                )
            )
            coords = np.asarray(normalized["coords"], dtype=np.float64)
            times = Timestamp.from_jd(
                pa.array(normalized["times_jd"], type=pa.float64()), scale="tdb"
            )
            origin = Origin.from_kwargs(code=["SUN" for _ in range(len(times))])
            frame = "ecliptic"
            coordinates = KeplerianCoordinates.from_kwargs(
                time=times,
                a=coords[:, 0],
                e=coords[:, 1],
                i=coords[:, 2],
                raan=coords[:, 3],
                ap=coords[:, 4],
                M=coords[:, 5],
                origin=origin,
                frame=frame,
            )

            orbits = Orbits.from_kwargs(
                orbit_id=normalized["orbit_id"],
                object_id=normalized["object_id"],
                coordinates=coordinates.to_cartesian(),
            )

            total_orbits_list.append(orbits)

        elif coordinate_type == "cometary":
            elements = _get_horizons_elements(
                object_ids,
                times_i,
                location=location,
                id_type=id_type,
            )

            from adam_core import _rust_native as _rn

            normalized = json.loads(
                _rn.query_horizons_elements_normalize(
                    elements.to_json(orient="records"), "cometary"
                )
            )
            coords = np.asarray(normalized["coords"], dtype=np.float64)
            times = Timestamp.from_jd(
                pa.array(normalized["times_jd"], type=pa.float64()), scale="tdb"
            )
            origin = Origin.from_kwargs(code=["SUN" for _ in range(len(times))])
            frame = "ecliptic"
            coordinates = CometaryCoordinates.from_kwargs(
                time=times,
                q=coords[:, 0],
                e=coords[:, 1],
                i=coords[:, 2],
                raan=coords[:, 3],
                ap=coords[:, 4],
                tp=coords[:, 5],
                origin=origin,
                frame=frame,
            )

            orbits = Orbits.from_kwargs(
                orbit_id=normalized["orbit_id"],
                object_id=normalized["object_id"],
                coordinates=coordinates.to_cartesian(),
            )

            total_orbits_list.append(orbits)

        else:
            err = "coordinate_type should be one of {'cartesian', 'keplerian', 'cometary'}"
            raise ValueError(err)

    total_orbits = (
        qv.concatenate(total_orbits_list) if total_orbits_list else Orbits.empty()
    )
    # Sort orbits by time
    total_orbits = total_orbits.sort_by(
        ["coordinates.time.days", "coordinates.time.nanos"]
    )

    return total_orbits
