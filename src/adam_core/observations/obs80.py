"""Typed MPC 80-column optical observations.

The parser intentionally supports only self-contained optical astrometry rows.
Radar, satellite, and roving-observer records require companion lines and are
rejected rather than partially interpreted.
"""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_EVEN

import pyarrow as pa
import quivr as qv
from quivr.validators import and_, ge, le

from ..time import Timestamp

NANOSECONDS_PER_DAY = 86_400_000_000_000
_MJD_EPOCH = datetime.date(1858, 11, 17)
_UNSUPPORTED_TWO_LINE_TYPES = frozenset({"R", "S", "V", "r", "s", "v"})


class Obs80ParseError(ValueError):
    """An MPC 80-column record is malformed or unsupported."""


class OpticalObs80(qv.Table):
    """Self-contained optical observations parsed from MPC 80-column rows."""

    raw_line = qv.LargeStringColumn()
    designation = qv.LargeStringColumn()
    discovery = qv.BooleanColumn()
    note1 = qv.LargeStringColumn(nullable=True)
    note2 = qv.LargeStringColumn(nullable=True)
    observatory_code = qv.LargeStringColumn()
    time = Timestamp.as_column()
    ra_deg = qv.Float64Column(validator=and_(ge(0), le(360)))
    dec_deg = qv.Float64Column(validator=and_(ge(-90), le(90)))
    mag = qv.Float64Column(nullable=True)
    band = qv.LargeStringColumn(nullable=True)
    astrometric_catalog = qv.LargeStringColumn(nullable=True)
    reference = qv.LargeStringColumn(nullable=True)


class ScoutObservations(qv.Table):
    """One or more authoritative JPL Scout ``file=mpc`` snapshots.

    Snapshot metadata is repeated per observation so filtered/sliced tables
    retain their source hash, API signature, and ordering without side data.
    ``declared_n_obs`` is Scout's summary metadata; membership is exclusively
    defined by ``snapshot_observation_count`` and the nested observations.
    """

    object_id = qv.LargeStringColumn()
    solution_date_utc = qv.LargeStringColumn(nullable=True)
    declared_n_obs = qv.Int64Column(nullable=True)
    snapshot_sha256 = qv.LargeStringColumn()
    snapshot_observation_count = qv.Int64Column()
    signature_version = qv.LargeStringColumn(nullable=True)
    signature_source = qv.LargeStringColumn(nullable=True)
    observation_index = qv.Int64Column()
    observation = OpticalObs80.as_column()


def _parse_decimal(raw: str, *, field: str) -> Decimal:
    try:
        return Decimal(raw.strip())
    except (InvalidOperation, ValueError) as exc:
        raise Obs80ParseError(f"invalid {field}") from exc


def _parse_obstime(raw: str) -> tuple[int, int]:
    parts = raw.strip().split()
    if len(parts) != 3:
        raise Obs80ParseError("invalid observation date")
    try:
        year = int(parts[0])
        month = int(parts[1])
    except ValueError as exc:
        raise Obs80ParseError("invalid observation date") from exc

    day_decimal = _parse_decimal(parts[2], field="observation day")
    day = int(day_decimal.to_integral_value(rounding=ROUND_FLOOR))
    fraction = day_decimal - Decimal(day)
    if fraction < 0 or fraction >= 1:
        raise Obs80ParseError("invalid observation day fraction")
    try:
        calendar_day = datetime.date(year, month, day)
    except ValueError as exc:
        raise Obs80ParseError("invalid observation date") from exc

    nanos = int(
        (fraction * Decimal(NANOSECONDS_PER_DAY)).to_integral_value(
            rounding=ROUND_HALF_EVEN
        )
    )
    if nanos == NANOSECONDS_PER_DAY:
        calendar_day += datetime.timedelta(days=1)
        nanos = 0
    return (calendar_day - _MJD_EPOCH).days, nanos


def _parse_ra(raw: str) -> float:
    parts = raw.strip().split()
    if len(parts) != 3:
        raise Obs80ParseError("invalid right ascension")
    try:
        hours, minutes, seconds = (float(value) for value in parts)
    except ValueError as exc:
        raise Obs80ParseError("invalid right ascension") from exc
    if not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60):
        raise Obs80ParseError("right ascension outside valid range")
    return 15.0 * (hours + minutes / 60.0 + seconds / 3600.0)


def _parse_dec(raw: str) -> float:
    parts = raw.strip().split()
    if len(parts) != 3 or not parts[0] or parts[0][0] not in "+-":
        raise Obs80ParseError("invalid declination")
    sign = -1.0 if parts[0][0] == "-" else 1.0
    try:
        degrees = float(parts[0][1:])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except ValueError as exc:
        raise Obs80ParseError("invalid declination") from exc
    if not (0 <= degrees <= 90 and 0 <= minutes < 60 and 0 <= seconds < 60):
        raise Obs80ParseError("declination outside valid range")
    value = sign * (degrees + minutes / 60.0 + seconds / 3600.0)
    if not -90.0 <= value <= 90.0:
        raise Obs80ParseError("declination outside valid range")
    return value


def parse_optical_obs80(raw: str) -> OpticalObs80:
    """Parse one self-contained optical MPC 80-column record."""
    if not isinstance(raw, str) or len(raw) < 80:
        raise Obs80ParseError("record is shorter than 80 columns")
    line = raw[:80]
    note2 = line[14].strip() or None
    if note2 in _UNSUPPORTED_TWO_LINE_TYPES:
        raise Obs80ParseError(f"unsupported two-line record type {note2}")

    designation = line[5:12].strip()
    observatory_code = line[77:80].strip()
    if not designation:
        raise Obs80ParseError("missing designation")
    if len(observatory_code) != 3:
        raise Obs80ParseError("invalid observatory code")

    magnitude_raw = line[65:70].strip()
    try:
        magnitude = float(magnitude_raw) if magnitude_raw else None
    except ValueError as exc:
        raise Obs80ParseError("invalid magnitude") from exc

    obstime_days, obstime_nanos = _parse_obstime(line[15:32])
    return OpticalObs80.from_kwargs(
        raw_line=[line],
        designation=[designation],
        discovery=[line[12] == "*"],
        note1=[line[13].strip() or None],
        note2=[note2],
        observatory_code=[observatory_code],
        time=Timestamp.from_kwargs(
            days=[obstime_days], nanos=[obstime_nanos], scale="utc"
        ),
        ra_deg=[_parse_ra(line[32:44])],
        dec_deg=[_parse_dec(line[44:56])],
        mag=pa.array([magnitude], type=pa.float64()),
        band=[line[70].strip() or None],
        astrometric_catalog=[line[71].strip() or None],
        reference=[line[72:77].strip() or None],
    )


def parse_optical_obs80_file(raw: str, *, strict: bool = True) -> OpticalObs80:
    """Parse every nonblank optical row in an MPC-format text file.

    Parameters
    ----------
    raw
        MPC-format file contents.
    strict
        Raise on the first malformed/unsupported row. If false, omit invalid
        rows. Scout snapshot ingestion uses the strict default so it can never
        expose a partial fitted observation set.
    """
    parsed: list[OpticalObs80] = []
    for line_number, line in enumerate(str(raw or "").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed.append(parse_optical_obs80(line))
        except Obs80ParseError as exc:
            if strict:
                raise Obs80ParseError(f"line {line_number}: {exc}") from exc
    if not parsed:
        return OpticalObs80.empty()
    return qv.concatenate(parsed)
