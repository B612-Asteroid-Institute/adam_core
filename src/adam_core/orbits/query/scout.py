import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import requests

from ...coordinates.cometary import CometaryCoordinates
from ...coordinates.origin import Origin
from ...observations.obs80 import (
    Obs80ParseError,
    ScoutObservations,
    parse_optical_obs80_file,
)
from ...time import Timestamp
from ..variants import VariantOrbits

logger = logging.getLogger(__name__)
_SCOUT_API_URL = "https://ssd-api.jpl.nasa.gov/scout.api"


class ScoutQueryError(RuntimeError):
    """A structured failure while querying JPL Scout."""

    error_type = "query_error"
    http_status = 502
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        object_id: str | None = None,
        upstream_status: int | None = None,
    ) -> None:
        self.object_id = object_id
        self.upstream_status = upstream_status
        super().__init__(message)


class ScoutObjectNotFoundError(ScoutQueryError):
    """The requested live Scout object no longer exists."""

    error_type = "not_found"
    http_status = 404


class ScoutServiceUnavailableError(ScoutQueryError):
    """Scout remained unavailable after bounded retries."""

    error_type = "service_unavailable"
    http_status = 503
    retryable = True


class ScoutResponseError(ScoutQueryError):
    """Scout returned an unsupported or malformed successful response."""

    error_type = "invalid_response"


def _request_scout_json(
    *,
    params: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
    max_attempts: int = 3,
    retry_delay_s: float = 0.5,
    http_get: Callable[..., Any] = requests.get,
) -> Any:
    """Fetch Scout JSON with bounded retries for transient HTTP failures."""
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            response = http_get(_SCOUT_API_URL, params=params, timeout=timeout_s)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if not retryable or attempt + 1 >= attempts:
                raise
            time.sleep(retry_delay_s * (2**attempt))
    raise RuntimeError("unreachable Scout retry state")


def _request_scout_object_json(
    object_id: str,
    *,
    params: dict[str, Any],
    timeout_s: float,
    max_attempts: int,
    retry_delay_s: float,
    http_get: Callable[..., Any],
) -> dict[str, Any]:
    """Fetch one object product and map lifecycle/transport failures."""
    try:
        payload = _request_scout_json(
            params=params,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            retry_delay_s=retry_delay_s,
            http_get=http_get,
        )
    except requests.RequestException as exc:
        upstream_status = getattr(getattr(exc, "response", None), "status_code", None)
        if upstream_status == 404:
            raise ScoutObjectNotFoundError(
                f"Scout object {object_id!r} was not found",
                object_id=object_id,
                upstream_status=upstream_status,
            ) from exc
        if upstream_status is None or upstream_status == 429 or upstream_status >= 500:
            raise ScoutServiceUnavailableError(
                f"Scout is temporarily unavailable for {object_id!r}; retry later",
                object_id=object_id,
                upstream_status=upstream_status,
            ) from exc
        raise ScoutQueryError(
            f"Scout request failed for {object_id!r} with HTTP {upstream_status}",
            object_id=object_id,
            upstream_status=upstream_status,
        ) from exc

    if not isinstance(payload, dict):
        raise ScoutResponseError(
            f"Scout returned a non-object response for {object_id!r}",
            object_id=object_id,
        )
    if payload.get("error"):
        detail = str(payload["error"])
        normalized = detail.casefold()
        if any(
            token in normalized
            for token in (
                "not found",
                "no object",
                "does not exist",
                "unknown object",
                "invalid designation",
            )
        ):
            raise ScoutObjectNotFoundError(
                f"Scout object {object_id!r} is unavailable: {detail}",
                object_id=object_id,
            )
        if any(
            token in normalized
            for token in (
                "temporar",
                "service unavailable",
                "timeout",
                "try again",
                "rate limit",
                "internal server",
            )
        ):
            raise ScoutServiceUnavailableError(
                f"Scout is temporarily unavailable for {object_id!r}: {detail}",
                object_id=object_id,
            )
        raise ScoutResponseError(
            f"Scout returned an error for {object_id!r}: {detail}",
            object_id=object_id,
        )
    return payload


def _require_scout_signature(
    payload: dict[str, Any], *, object_id: str
) -> dict[str, Any]:
    signature = payload.get("signature")
    if not isinstance(signature, dict) or str(signature.get("version")) != "1.3":
        raise ScoutResponseError(
            "Unsupported or missing Scout API signature for "
            f"{object_id}: {signature!r}",
            object_id=object_id,
        )
    return signature


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
    data = _request_scout_json()
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
        frame="ecliptic",
    )

    cartesian_coords = cometary_coords.to_cartesian()

    unique_orbit_ids = pc.cast(scout_orbits.idx, pa.large_string())

    variants = VariantOrbits.from_kwargs(
        coordinates=cartesian_coords,
        orbit_id=unique_orbit_ids,
        variant_id=unique_orbit_ids,
        object_id=pa.repeat(object_id, len(scout_orbits)),
    )

    return variants


def query_scout_observations(
    ids: npt.ArrayLike,
    *,
    timeout_s: float = 30.0,
    max_attempts: int = 3,
    retry_delay_s: float = 0.5,
    http_get: Callable[..., Any] = requests.get,
) -> ScoutObservations:
    """Query authoritative fitted observations from JPL Scout.

    Scout's ``file=mpc`` payload exclusively defines snapshot membership.
    ``nObs`` is retained as metadata but is not used to add/remove records: in
    live Scout data it can differ from the file row count.
    """
    object_ids = [ids] if isinstance(ids, str) else list(ids)
    snapshots: list[ScoutObservations] = []
    for object_id_value in object_ids:
        object_id = str(object_id_value)
        payload = _request_scout_object_json(
            object_id,
            params={"tdes": object_id, "file": "mpc"},
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            retry_delay_s=retry_delay_s,
            http_get=http_get,
        )
        signature = _require_scout_signature(payload, object_id=object_id)

        raw_file = payload.get("fileMPC")
        if not isinstance(raw_file, str) or not raw_file.strip():
            raise ScoutResponseError(
                f"Scout returned no file=mpc snapshot for {object_id}",
                object_id=object_id,
            )
        try:
            observations = parse_optical_obs80_file(raw_file)
        except Obs80ParseError as exc:
            raise ScoutResponseError(
                f"Scout returned an invalid file=mpc snapshot for {object_id}: {exc}",
                object_id=object_id,
            ) from exc
        if len(observations) == 0:
            raise ScoutResponseError(
                f"Scout returned an empty file=mpc snapshot for {object_id}",
                object_id=object_id,
            )
        if any(value != object_id for value in observations.designation.to_pylist()):
            raise ScoutResponseError(
                f"Scout file=mpc designation mismatch for requested object {object_id}",
                object_id=object_id,
            )

        declared_n_obs_raw = payload.get("nObs")
        try:
            declared_n_obs = (
                int(declared_n_obs_raw) if declared_n_obs_raw is not None else None
            )
        except (TypeError, ValueError):
            declared_n_obs = None
        if declared_n_obs is not None and declared_n_obs != len(observations):
            logger.warning(
                "Scout nObs differs from file=mpc membership for %s: "
                "nObs=%d file=%d; file=mpc remains authoritative",
                object_id,
                declared_n_obs,
                len(observations),
            )

        n = len(observations)
        snapshots.append(
            ScoutObservations.from_kwargs(
                object_id=pa.repeat(object_id, n),
                solution_date_utc=pa.array(
                    [payload.get("lastRun")] * n, type=pa.large_string()
                ),
                declared_n_obs=pa.array([declared_n_obs] * n, type=pa.int64()),
                snapshot_sha256=pa.repeat(
                    hashlib.sha256(raw_file.encode("utf-8")).hexdigest(), n
                ),
                snapshot_observation_count=pa.repeat(n, n),
                signature_version=pa.array(
                    [signature.get("version")] * n, type=pa.large_string()
                ),
                signature_source=pa.array(
                    [signature.get("source")] * n, type=pa.large_string()
                ),
                observation_index=pa.array(range(n), type=pa.int64()),
                observation=observations,
            )
        )
    if not snapshots:
        return ScoutObservations.empty()
    return qv.concatenate(snapshots)


def query_scout(
    ids: npt.ArrayLike,
    *,
    timeout_s: float = 30.0,
    max_attempts: int = 3,
    retry_delay_s: float = 0.5,
    http_get: Callable[..., Any] = requests.get,
) -> VariantOrbits:
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
    object_ids = [ids] if isinstance(ids, str) else list(ids)
    variant_orbits = VariantOrbits.empty()
    for object_id_value in object_ids:
        object_id = str(object_id_value)
        payload = _request_scout_object_json(
            object_id,
            params={"tdes": object_id, "orbits": "1"},
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            retry_delay_s=retry_delay_s,
            http_get=http_get,
        )
        _require_scout_signature(payload, object_id=object_id)
        orbits_payload = payload.get("orbits")
        rows = orbits_payload.get("data") if isinstance(orbits_payload, dict) else None
        if not isinstance(rows, list):
            raise ScoutResponseError(
                f"Scout response for {object_id!r} is missing orbits.data",
                object_id=object_id,
            )
        if not rows:
            raise ScoutObjectNotFoundError(
                f"Scout returned no orbit samples for {object_id!r}",
                object_id=object_id,
            )

        try:
            # Convert from list of rows to list of columns.
            columns = list(map(list, zip(*rows)))
            table = pa.Table.from_arrays(columns, schema=ScoutOrbit.schema)
            scout_orbits = ScoutOrbit.from_pyarrow(table)
        except (TypeError, ValueError, pa.ArrowException) as exc:
            raise ScoutResponseError(
                f"Scout returned invalid orbit samples for {object_id!r}: {exc}",
                object_id=object_id,
            ) from exc
        object_variants = scout_orbits_to_variant_orbits(object_id, scout_orbits)
        variant_orbits = qv.concatenate([variant_orbits, object_variants])

    return variant_orbits
