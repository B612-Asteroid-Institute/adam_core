import logging
from typing import List, OrderedDict

import numpy as np
import numpy.typing as npt
from astroquery.jplsbdb import SBDB

from ...coordinates.cometary import CometaryCoordinates
from ...coordinates.covariances import CoordinateCovariances, sigmas_to_covariances
from ...coordinates.origin import Origin
from ...time import Timestamp
from ..orbits import Orbits

logger = logging.getLogger(__name__)


def _convert_SBDB_covariances(
    sbdb_covariances: npt.ArrayLike,
) -> npt.ArrayLike:
    """
    Convert SBDB covariance matrices to Cometary covariance matrices.

    Parameters
    ----------
    sbdb_covariances : `~numpy.ndarray` (N, 6, 6)
        Covariance matrices pulled from JPL's Small Body Database Browser.

    Returns
    -------
    covariances : `~numpy.ndarray` (N, 6, 6)
        Cometary covariance matrices.
    """
    covariances = np.zeros_like(sbdb_covariances)
    # sigma_q{x}
    covariances[:, 0, 0] = sbdb_covariances[:, 1, 1]  # sigma_qq
    covariances[:, 1, 0] = covariances[:, 0, 1] = sbdb_covariances[:, 0, 1]  # sigma_qe
    covariances[:, 2, 0] = covariances[:, 0, 2] = sbdb_covariances[:, 5, 1]  # sigma_qi
    covariances[:, 3, 0] = covariances[:, 0, 3] = sbdb_covariances[
        :, 3, 1
    ]  # sigma_qraan
    covariances[:, 4, 0] = covariances[:, 0, 4] = sbdb_covariances[:, 4, 1]  # sigma_qap
    covariances[:, 5, 0] = covariances[:, 0, 5] = sbdb_covariances[:, 2, 1]  # sigma_qtp

    # sigma_e{x}
    covariances[:, 1, 1] = sbdb_covariances[:, 0, 0]  # sigma_ee
    covariances[:, 2, 1] = covariances[:, 1, 2] = sbdb_covariances[:, 5, 0]  # sigma_ei
    covariances[:, 3, 1] = covariances[:, 1, 3] = sbdb_covariances[
        :, 3, 0
    ]  # sigma_eraan
    covariances[:, 4, 1] = covariances[:, 1, 4] = sbdb_covariances[:, 4, 0]  # sigma_eap
    covariances[:, 5, 1] = covariances[:, 1, 5] = sbdb_covariances[:, 2, 0]  # sigma_etp

    # sigma_i{x}
    covariances[:, 2, 2] = sbdb_covariances[:, 5, 5]  # sigma_ii
    covariances[:, 3, 2] = covariances[:, 2, 3] = sbdb_covariances[
        :, 3, 5
    ]  # sigma_iraan
    covariances[:, 4, 2] = covariances[:, 2, 4] = sbdb_covariances[:, 4, 5]  # sigma_iap
    covariances[:, 5, 2] = covariances[:, 2, 5] = sbdb_covariances[:, 2, 5]  # sigma_itp

    # sigma_raan{x}
    covariances[:, 3, 3] = sbdb_covariances[:, 3, 3]  # sigma_raanraan
    covariances[:, 4, 3] = covariances[:, 3, 4] = sbdb_covariances[
        :, 4, 3
    ]  # sigma_raanap
    covariances[:, 5, 3] = covariances[:, 3, 5] = sbdb_covariances[
        :, 2, 3
    ]  # sigma_raantp

    # sigma_ap{x}
    covariances[:, 4, 4] = sbdb_covariances[:, 4, 4]  # sigma_apap
    covariances[:, 5, 4] = covariances[:, 4, 5] = sbdb_covariances[
        :, 2, 4
    ]  # sigma_aptp

    # sigma_tp{x}
    covariances[:, 5, 5] = sbdb_covariances[:, 2, 2]  # sigma_tptp

    return covariances


def _get_sbdb_elements(obj_ids: List[str]) -> List[OrderedDict]:
    """
    Get orbital elements and other object properties
    from JPL's Small Body Database Browser.

    Parameters
    ----------
    obj_ids : List
        Object IDs to query.

    Returns
    -------
    results : List
        List of dictionaries containing orbital elements and other object properties.
    """
    results = []
    for obj_id in obj_ids:
        result = SBDB.query(
            obj_id,
            covariance="mat",
            id_type="search",
            full_precision=True,
            solution_epoch=False,
        )
        results.append(result)

    return results


def query_sbdb(ids: npt.ArrayLike) -> Orbits:
    """
    Query JPL's Small-Body Database (SBDB) for orbits. The epoch at
    which the orbits are returned are near the epoch as published by the
    Minor Planet Center.

    By default, the orbit's covariance matrices are also queried for. If they
    are not available, then the 1-sigma uncertainties are used to construct
    the covariance matrices.

    Parameters
    ----------
    ids : list
        List of object IDs to query.

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits`
        Orbits object containing the queried orbits.

    Raises
    ------
    NotFoundError: If any of the queries object IDs are not found.
    """
    results = _get_sbdb_elements(ids)

    orbit_ids = []
    object_ids = []
    classes = []
    coords_cometary = np.zeros((len(results), 6), dtype=np.float64)
    covariances_sbdb = np.zeros((len(results), 6, 6), dtype=np.float64)
    times = np.zeros((len(results)), dtype=np.float64)

    for i, result in enumerate(results):
        if "object" not in result:
            raise NotFoundError("object {} was not found", ids[i])

        orbit_ids.append(f"{i:05d}")
        object_ids.append(result["object"]["fullname"])
        classes.append(result["object"]["orbit_class"]["code"])

        orbit = result["orbit"]
        elements = orbit["elements"]
        epoch = orbit["epoch"]
        if "covariance" in orbit:
            labels = orbit["covariance"]["labels"]
            if len(labels) > 6:
                logger.debug(
                    "Covariance matrix has more parameters than just orbital elements. "
                    "Ignoring non-orbital elements in covariance matrix."
                )
                labels = labels[:6]

            expected_labels = ["e", "q", "tp", "node", "peri", "i"]
            if labels != expected_labels:
                raise ValueError(
                    "Expected covariance matrix labels to be {expected_labels} instead got {labels}."
                )

            # Limit covariances to just the orbital elements
            # The SBDB API documentation states that physical parameter covariances
            # are appended to the rows and columns of the covariance matrix with the
            # orbital elements remaining in the first 6 rows and columns.
            # See: Orbit Subsection: covariance in https://ssd-api.jpl.nasa.gov/doc/sbdb.html
            covariances_sbdb[i, :, :] = orbit["covariance"]["data"][:6, :6]

            if "elements" in orbit["covariance"]:
                # If elements is provided inside covariance, then it's
                # the elements at the epoch which was used to
                # calculate covariance, so we should prefer it.
                elements = orbit["covariance"]["elements"]
                epoch = orbit["covariance"]["epoch"]

        else:
            sigmas = np.array(
                [
                    [
                        elements["e_sig"],
                        elements["q_sig"].value,
                        elements["tp_sig"].value,
                        elements["om_sig"].value,
                        elements["w_sig"].value,
                        elements["i_sig"].value,
                    ]
                ]
            )
            covariances_sbdb[i, :, :] = sigmas_to_covariances(sigmas).filled()[0]

        times[i] = epoch.value
        coords_cometary[i, 0] = elements["q"].value
        coords_cometary[i, 1] = elements["e"]
        coords_cometary[i, 2] = elements["i"].value
        coords_cometary[i, 3] = elements["om"].value
        coords_cometary[i, 4] = elements["w"].value
        coords_cometary[i, 5] = (
            Timestamp.from_jd([elements["tp"].value], scale="tdb").mjd()[0].as_py()
        )

    covariances_cometary = _convert_SBDB_covariances(covariances_sbdb)
    times = Timestamp.from_jd(times, scale="tdb")
    origin = Origin.from_kwargs(code=["SUN" for i in range(len(times))])
    frame = "ecliptic"
    coordinates = CometaryCoordinates.from_kwargs(
        time=times,
        q=coords_cometary[:, 0],
        e=coords_cometary[:, 1],
        i=coords_cometary[:, 2],
        raan=coords_cometary[:, 3],
        ap=coords_cometary[:, 4],
        tp=coords_cometary[:, 5],
        covariance=CoordinateCovariances.from_matrix(covariances_cometary),
        origin=origin,
        frame=frame,
    )

    orbit_ids = np.array(orbit_ids, dtype="object")
    object_ids = np.array(object_ids, dtype="object")
    classes = np.array(classes)

    return Orbits.from_kwargs(
        orbit_id=orbit_ids, object_id=object_ids, coordinates=coordinates.to_cartesian()
    )


class NotFoundError(Exception):
    def __init__(self, message, object_id):
        self.message = message
        self.object_id = object_id

    def __str__(self):
        return self.message
