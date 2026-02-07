from __future__ import annotations

import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Any, List

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import requests
from astroquery.jplsbdb import SBDB

from ...coordinates.cometary import CometaryCoordinates
from ...coordinates.covariances import CoordinateCovariances, sigmas_to_covariances
from ...coordinates.origin import Origin
from ...time import Timestamp
from ..orbits import Orbits

logger = logging.getLogger(__name__)

_SBDB_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
_SBDB_API_FAIR_USE_MAX_CONCURRENT_REQUESTS = 1

_thread_local = threading.local()


def _get_requests_session() -> requests.Session:
    """
    Return a per-thread `requests.Session`.

    Why: using a session enables connection pooling, which reduces overhead when querying many IDs.
    A per-thread session is a safe default if callers choose to enable limited concurrency.
    """
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


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
    SBDB.clear_cache()  # Yikes!
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


def _orbits_from_sbdb_results(ids: npt.ArrayLike, results: List[OrderedDict]) -> Orbits:
    """
    Convert SBDB query results into an `Orbits` table.

    What: shared implementation for both the legacy astroquery-based query and the new
    direct-HTTP query.
    Why: keeping a single conversion path ensures both entrypoints return identical results.
    """
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
            covariances_sbdb[i, :, :] = sigmas_to_covariances(sigmas)[0]

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
    return _orbits_from_sbdb_results(ids, results)


def _sbdb_api_get_json(
    object_id: str,
    *,
    timeout_s: float,
    max_attempts: int,
) -> dict[str, Any]:
    """
    Query JPL's public SBDB JSON API for a single object, with retries.

    Why: we want explicit timeout/retry behavior, and we want to avoid global cache clearing.

    Notes:
    - Per the JPL SSD/CNEOS API fair use policy, clients should not send concurrent requests.
      This helper does not enforce that policy; the public entrypoint defaults to sequential
      requests (max_concurrent_requests=1).
    """
    obj = str(object_id).strip()
    if not obj:
        raise ValueError("object_id must be non-empty")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")

    params = {
        "sstr": obj,
        "cov": "mat",
        "full-prec": "true",
    }

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = _get_requests_session().get(
                _SBDB_API_URL, params=params, timeout=timeout_s
            )
            resp.raise_for_status()
            return resp.json()
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as err:
            last_err = err
        except requests.exceptions.HTTPError as err:
            # Retry on transient server errors and explicit throttling.
            status = err.response.status_code if err.response is not None else None
            if status is not None and (status >= 500 or status == 429):
                last_err = err
            else:
                raise
        except Exception:
            # Non-retryable (JSON decode, unexpected failure, etc.)
            raise

        # Exponential backoff with a small cap.
        sleep_s = min(8.0, 0.5 * (2**attempt))
        time.sleep(sleep_s)

    raise RuntimeError(f"SBDB query failed after {max_attempts} attempts: {last_err}")


def _sbdb_float(value: Any) -> float:
    """
    Convert a JSON scalar (string/number) into a float.

    SBDB returns many numeric fields as strings (including scientific notation). We normalize
    those to floats for orbit construction.
    """
    if value is None:
        raise ValueError("Expected a numeric value, got None.")
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected a numeric JSON scalar, got {type(value)!r}.")


def _sbdb_elements_map(elements: Any) -> dict[str, dict[str, Any]]:
    """
    Convert SBDB's `orbit.elements` list into a dict keyed by element short-name.
    """
    if not isinstance(elements, list):
        raise ValueError("Expected SBDB orbit elements to be a list.")
    out: dict[str, dict[str, Any]] = {}
    for el in elements:
        if not isinstance(el, dict):
            continue
        name = el.get("name")
        if name is None:
            continue
        out[str(name)] = el
    return out


def _sbdb_element_value(
    elements_by_name: dict[str, dict[str, Any]], name: str
) -> float:
    el = elements_by_name.get(name)
    if el is None:
        raise ValueError(f"SBDB orbit elements missing {name!r}.")
    if "value" not in el or el["value"] is None:
        raise ValueError(f"SBDB orbit element {name!r} is missing a value.")
    return _sbdb_float(el["value"])


def _sbdb_element_sigma(
    elements_by_name: dict[str, dict[str, Any]], name: str
) -> float:
    el = elements_by_name.get(name)
    if el is None:
        raise ValueError(f"SBDB orbit elements missing {name!r}.")
    sigma = el.get("sigma")
    if sigma is None:
        # Some SBDB payloads omit per-element uncertainties when no covariance matrix is provided.
        # We treat this as "unknown uncertainty" and propagate NaNs through the diagonal fallback.
        return float("nan")
    try:
        return _sbdb_float(sigma)
    except Exception:
        return float("nan")


def _orbits_from_sbdb_payloads(
    ids: list[str],
    payloads: list[dict[str, Any]],
) -> Orbits:
    """
    Convert raw SBDB JSON payloads into an `Orbits` table.

    This mirrors the behavior of the legacy `query_sbdb` implementation:
    - Prefer covariance-provided elements/epoch when present.
    - Use covariance matrix when available; otherwise build a diagonal covariance from sigmas.
    """
    if len(ids) != len(payloads):
        raise ValueError("ids and payloads must have the same length.")

    expected_labels = ["e", "q", "tp", "node", "peri", "i"]

    orbit_ids: list[str] = []
    object_ids: list[str] = []

    coords_cometary = np.zeros((len(payloads), 6), dtype=np.float64)
    covariances_sbdb = np.zeros((len(payloads), 6, 6), dtype=np.float64)
    times_jd = np.zeros((len(payloads)), dtype=np.float64)

    for i, (obj_id, payload) in enumerate(zip(ids, payloads)):
        if "object" not in payload:
            raise NotFoundError("object {} was not found", obj_id)
        if "orbit" not in payload:
            raise ValueError(f"SBDB payload for {obj_id!r} missing 'orbit'.")

        obj = payload["object"] or {}
        orbit_ids.append(f"{i:05d}")
        object_ids.append(str(obj.get("fullname")))

        orbit = payload["orbit"] or {}
        elements_list = orbit.get("elements")
        epoch_jd = _sbdb_float(orbit.get("epoch"))

        cov = orbit.get("covariance")
        cov_matrix: np.ndarray | None = None
        if isinstance(cov, dict) and cov.get("data") is not None:
            labels = cov.get("labels")
            if isinstance(labels, list):
                labels6 = [str(x) for x in labels[:6]]
                if labels6 != expected_labels:
                    raise ValueError(
                        f"Expected covariance matrix labels to be {expected_labels} "
                        f"in the first 6 entries, got {labels6}."
                    )

            data = np.asarray(cov["data"], dtype=np.float64)
            if data.ndim != 2 or data.shape[0] < 6 or data.shape[1] < 6:
                raise ValueError("Expected SBDB covariance matrix to be at least 6x6.")
            cov_matrix = data[:6, :6]

            # If covariance provides elements, prefer them (and the covariance epoch).
            if "elements" in cov and cov["elements"] is not None:
                elements_list = cov["elements"]
                if cov.get("epoch") is not None:
                    epoch_jd = _sbdb_float(cov.get("epoch"))

        if elements_list is None:
            raise ValueError(f"SBDB payload for {obj_id!r} missing orbit elements.")

        elements_by_name = _sbdb_elements_map(elements_list)

        if cov_matrix is None:
            # Fallback: build a diagonal covariance from per-element sigmas.
            sigmas = np.array(
                [
                    [
                        _sbdb_element_sigma(elements_by_name, "e"),
                        _sbdb_element_sigma(elements_by_name, "q"),
                        _sbdb_element_sigma(elements_by_name, "tp"),
                        _sbdb_element_sigma(elements_by_name, "om"),
                        _sbdb_element_sigma(elements_by_name, "w"),
                        _sbdb_element_sigma(elements_by_name, "i"),
                    ]
                ],
                dtype=np.float64,
            )
            cov_matrix = sigmas_to_covariances(sigmas)[0]

        covariances_sbdb[i, :, :] = cov_matrix

        times_jd[i] = epoch_jd

        q = _sbdb_element_value(elements_by_name, "q")
        e = _sbdb_element_value(elements_by_name, "e")
        inc = _sbdb_element_value(elements_by_name, "i")
        om = _sbdb_element_value(elements_by_name, "om")
        w = _sbdb_element_value(elements_by_name, "w")
        tp_jd = _sbdb_element_value(elements_by_name, "tp")
        tp_mjd = Timestamp.from_jd([tp_jd], scale="tdb").mjd()[0].as_py()

        coords_cometary[i, 0] = q
        coords_cometary[i, 1] = e
        coords_cometary[i, 2] = inc
        coords_cometary[i, 3] = om
        coords_cometary[i, 4] = w
        coords_cometary[i, 5] = tp_mjd

    covariances_cometary = _convert_SBDB_covariances(covariances_sbdb)
    times = Timestamp.from_jd(times_jd, scale="tdb")
    origin = Origin.from_kwargs(code=["SUN" for _ in range(len(times))])

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
        frame="ecliptic",
    )

    return Orbits.from_kwargs(
        orbit_id=np.array(orbit_ids, dtype="object"),
        object_id=np.array(object_ids, dtype="object"),
        coordinates=coordinates.to_cartesian(),
    )


def _get_sbdb_payloads_new(
    obj_ids: List[str],
    *,
    max_concurrent_requests: int,
    timeout_s: float,
    max_attempts: int,
) -> list[dict[str, Any]]:
    """
    Fetch SBDB JSON payloads via direct HTTP, optionally with limited concurrency.

    Important: JPL's SSD/CNEOS API fair use policy requests only one in-flight request at a time.
    The public entrypoint defaults to `max_concurrent_requests=1` to comply.
    """
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be > 0")

    n = len(obj_ids)
    if n == 0:
        return []

    max_workers = min(int(max_concurrent_requests), n)
    if max_workers > 1:
        logger.warning(
            "query_sbdb_new is configured with max_concurrent_requests=%s. "
            "JPL's SSD/CNEOS API fair use policy requests only one in-flight request at a time; "
            "concurrent requests may be rejected.",
            max_workers,
        )

    results: list[dict[str, Any] | None] = [None] * n

    def fetch_one(i: int, object_id: str) -> tuple[int, dict[str, Any]]:
        payload = _sbdb_api_get_json(
            object_id, timeout_s=timeout_s, max_attempts=max_attempts
        )
        return i, payload

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_one, i, obj_id) for i, obj_id in enumerate(obj_ids)]
        for fut in as_completed(futures):
            i, res = fut.result()
            results[i] = res

    out: list[dict[str, Any]] = []
    for r in results:
        if r is None:
            raise RuntimeError("SBDB payload missing from concurrent fetch.")
        out.append(r)
    return out


def query_sbdb_new(
    ids: npt.ArrayLike,
    *,
    max_concurrent_requests: int = 1,
    timeout_s: float = 60.0,
    max_attempts: int = 5,
    allow_missing: bool = False,
    orbit_id_from_input: bool = False,
) -> Orbits:
    """
    Query JPL SBDB for orbits using direct HTTP requests (new implementation).

    This is intended to be a drop-in alternative to `query_sbdb` that:
    - avoids `SBDB.clear_cache()`,
    - provides explicit timeout/retry controls, and
    - can optionally fetch multiple objects concurrently.

    Notes
    -----
    JPL's SSD/CNEOS API fair use policy requests only one in-flight API request at a time.
    Therefore, the default `max_concurrent_requests=1` is the recommended setting.

    Parameters
    ----------
    allow_missing : bool, optional
        If True, do not raise when an ID is not found in SBDB. Instead, return an `Orbits` table
        containing only the successfully resolved IDs (potentially empty).
    orbit_id_from_input : bool, optional
        If True, set the returned `Orbits.orbit_id` values to the input IDs (after any missing
        filtering). This is useful when callers need to map rows back to the requested identifiers.
    """
    # Normalize ids into a list of strings while preserving the caller's order.
    if isinstance(ids, (str, bytes)):
        obj_ids = [str(ids)]
    else:
        obj_ids = [str(x) for x in ids]

    payloads = _get_sbdb_payloads_new(
        obj_ids,
        max_concurrent_requests=max_concurrent_requests,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )
    if allow_missing:
        kept_ids: list[str] = []
        kept_payloads: list[dict[str, Any]] = []
        for obj_id, payload in zip(obj_ids, payloads):
            if "object" not in payload:
                continue
            kept_ids.append(obj_id)
            kept_payloads.append(payload)

        if not kept_ids:
            return Orbits.empty()

        orbits = _orbits_from_sbdb_payloads(kept_ids, kept_payloads)
        if orbit_id_from_input:
            orbits = orbits.set_column(
                "orbit_id", pa.array(kept_ids, type=pa.large_string())
            )
        return orbits

    return _orbits_from_sbdb_payloads(obj_ids, payloads)


class NotFoundError(Exception):
    def __init__(self, message, object_id):
        self.message = message
        self.object_id = object_id

    def __str__(self):
        return self.message
