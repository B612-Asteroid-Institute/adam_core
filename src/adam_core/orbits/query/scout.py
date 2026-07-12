import numpy.typing as npt
import quivr as qv

from ...utils.http import _raise_compatible_http_error
from ..variants import VariantOrbits


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
    from adam_core import _rust_native

    from ..._rust.arrow import table_from_record_batch

    try:
        batch = _rust_native.get_scout_objects_arrow()
    except RuntimeError as error:
        _raise_compatible_http_error(error)
    return table_from_record_batch(ScoutObjectSummary, batch)


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
    if len(scout_orbits) == 0:
        return VariantOrbits.empty()

    from adam_core import _rust_native

    from ..._rust.arrow import table_from_record_batch

    batch = _rust_native.scout_orbits_to_variants_arrow(
        str(object_id), scout_orbits.table.to_batches()[0]
    )
    return table_from_record_batch(VariantOrbits, batch)


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
    from adam_core import _rust_native

    from ..._rust.arrow import table_from_record_batch

    try:
        batch = _rust_native.query_scout_arrow([str(value) for value in ids])
    except RuntimeError as error:
        _raise_compatible_http_error(error)
    return table_from_record_batch(VariantOrbits, batch)
