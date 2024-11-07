import logging

import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import requests

from ...coordinates.cometary import CometaryCoordinates
from ...coordinates.origin import Origin
from ...time import Timestamp
from ..variants import VariantOrbits

logger = logging.getLogger(__name__)


class ScoutObjectSummary(qv.Table):
    """
    Table the represents results of the Scout summary query (returns all objects)

    Those objects look like this
    {
      "unc": "59",
      "lastRun": "2024-10-18 15:31",
      "dec": "-19",
      "H": "18.7",
      "moid": "0.1",
      "geocentricScore": 0,
      "ra": "16:54",
      "rating": null,
      "tisserandScore": 15,
      "uncP1": "76",
      "ieoScore": 47,
      "rate": "2.2",
      "rmsN": "0.00",
      "Vmag": "20.5",
      "neoScore": 100,
      "nObs": 2,
      "objectName": "v17oct4",
      "phaScore": 20,
      "tEphem": "2024-10-22 20:15",
      "arc": "0.06",
      "caDist": null,
      "elong": "45",
      "vInf": null,
      "neo1kmScore": 5
    }
    """

    unc = qv.LargeStringColumn()
    lastRun = qv.LargeStringColumn()
    dec = qv.LargeStringColumn()
    H = qv.LargeStringColumn()
    moid = qv.LargeStringColumn()
    geocentricScore = qv.Int64Column()
    ra = qv.LargeStringColumn()
    rating = qv.Int8Column(nullable=True)
    tisserandScore = qv.Int64Column()
    uncP1 = qv.LargeStringColumn()
    ieoScore = qv.Int64Column()
    rate = qv.LargeStringColumn()
    rmsN = qv.LargeStringColumn()
    Vmag = qv.LargeStringColumn()
    neoScore = qv.Int64Column()
    nObs = qv.Int64Column()
    objectName = qv.LargeStringColumn()
    phaScore = qv.Int64Column()
    tEphem = qv.LargeStringColumn()
    arc = qv.LargeStringColumn()
    caDist = qv.LargeStringColumn(nullable=True)
    elong = qv.LargeStringColumn()
    vInf = qv.LargeStringColumn(nullable=True)
    neo1kmScore = qv.Int64Column()


def get_scout_objects() -> ScoutObjectSummary:
    """
    Query the Scout API for a summary of all objects.

    Returns
    -------
    scout_objects : `~adam_core.orbits.query.scout.ScoutObjectSummary`
        Table containing the summary of all objects.
    """
    url = "https://ssd-api.jpl.nasa.gov/scout.api"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    data = data["data"]
    table = pa.Table.from_pylist(data, schema=ScoutObjectSummary.schema)
    return ScoutObjectSummary.from_pyarrow(table)


class ScoutOrbit(qv.Table):
    """
    Represents a sample orbit from scout

    The fields are as follows:
    "fields":["idx","epoch","ec","qr","tp","om","w","inc","H","dca","tca","moid","vinf","geoEcc","impFlag"]
    Sample data:
          [
        0,
        "2457581.871499164",
        "3.357123709445450E-01",
        "9.083681207232809E-01",
        "2457636.871738402",
        "1.1141193497296813E+02",
        "2.4446138666195648E+02",
        "1.6610875455065550E+01",
        "24.694617",
        null,
        null,
        "0.321364995E-01",
        null,
        1.000000000E+99,
        0
      ],
    """

    idx = qv.Int64Column()
    epoch = qv.LargeStringColumn()
    ec = qv.LargeStringColumn()
    qr = qv.LargeStringColumn()
    tp = qv.LargeStringColumn()
    om = qv.LargeStringColumn()
    w = qv.LargeStringColumn()
    inc = qv.LargeStringColumn()
    H = qv.LargeStringColumn()
    dca = qv.LargeStringColumn(nullable=True)
    tca = qv.LargeStringColumn(nullable=True)
    moid = qv.LargeStringColumn()
    vinf = qv.LargeStringColumn(nullable=True)
    geoEcc = qv.LargeStringColumn()
    impFlag = qv.Int64Column()


def scout_orbits_to_variant_orbits(
    object_id: str, scout_orbits: ScoutOrbit
) -> VariantOrbits:
    """
    Convert a table of scout orbits to a table of variant orbits

    Parameters
    ----------
    scout_orbits : `~adam_core.orbits.query.scout.ScoutOrbit`
        Table containing the scout orbits

    Returns
    -------
    variant_orbits : `~adam_core.orbits.VariantOrbits`
        Table containing the variant orbits
    """
    cometary_coords = CometaryCoordinates.from_kwargs(
        q=pc.cast(scout_orbits.qr, pa.float64()),
        e=pc.cast(scout_orbits.ec, pa.float64()),
        i=pc.cast(scout_orbits.inc, pa.float64()),
        raan=pc.cast(scout_orbits.om, pa.float64()),
        ap=pc.cast(scout_orbits.w, pa.float64()),
        tp=pc.subtract(pc.cast(scout_orbits.tp, pa.float64()), 2400000.5),
        time=Timestamp.from_jd(pc.cast(scout_orbits.epoch, pa.float64())),
        origin=Origin.from_kwargs(code=pa.repeat("SUN", len(scout_orbits))),
    )

    cartesian_coords = cometary_coords.to_cartesian()

    variants = VariantOrbits.from_kwargs(
        coordinates=cartesian_coords,
        orbit_id=pc.cast(scout_orbits.idx, pa.large_string()),
        object_id=pa.repeat(object_id, len(scout_orbits)),
    )

    return variants


def query_scout(ids: npt.ArrayLike) -> VariantOrbits:
    """
    Query the Scout API for a list of objects by id

    We return VariantOrbits for each orbit sample
    as the Scout API does not provide covariance information.

    These can be further reconstructed into an Orbit with
    covariance information using the `adam_core.orbits.Orbits` class.

    Parameters
    ----------
    ids : array-like
        List of object ids to query

    Returns
    -------
    orbits : `~adam_core.orbits.VariantOrbits`
        Table containing the orbits of the objects
    """
    url = "https://ssd-api.jpl.nasa.gov/scout.api?tdes={}&orbits=1"

    variant_orbits = VariantOrbits.empty()
    for object_id in ids:
        response = requests.get(url.format(object_id))
        response.raise_for_status()
        data = response.json()
        data = data["orbits"]["data"]

        # Convert from list of rows to list of columns
        data = list(map(list, zip(*data)))
        table = pa.Table.from_arrays(data, schema=ScoutOrbit.schema)
        scout_orbits = ScoutOrbit.from_pyarrow(table)
        object_variants = scout_orbits_to_variant_orbits(object_id, scout_orbits)
        variant_orbits = qv.concatenate([variant_orbits, object_variants])

    return variant_orbits
