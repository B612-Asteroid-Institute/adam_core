"""Parsing helpers for MPC's legacy 80-column observation format.

This module intentionally parses only self-contained optical astrometry rows.
Radar, satellite, and roving-observer records require companion lines and are
rejected rather than partially interpreted.
"""

from __future__ import annotations

import dataclasses
import datetime
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_EVEN

NANOSECONDS_PER_DAY = 86_400_000_000_000
_MJD_EPOCH = datetime.date(1858, 11, 17)
_UNSUPPORTED_TWO_LINE_TYPES = frozenset({"R", "S", "V", "r", "s", "v"})


class Obs80ParseError(ValueError):
    """An MPC 80-column record is malformed or unsupported."""


@dataclasses.dataclass(frozen=True)
class OpticalObs80:
    """One parsed, self-contained optical MPC 80-column observation."""

    raw_line: str
    designation: str
    discovery: bool
    note1: str | None
    note2: str | None
    observatory_code: str
    obstime_days_utc: int
    obstime_nanos_utc: int
    ra_deg: float
    dec_deg: float
    mag: float | None
    band: str | None
    astrometric_catalog: str | None
    reference: str | None

    @property
    def obstime_mjd_utc(self) -> float:
        return self.obstime_days_utc + self.obstime_nanos_utc / NANOSECONDS_PER_DAY


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
    """Parse one self-contained optical MPC 80-column record.

    Parameters
    ----------
    raw
        A line containing at least the standard 80 columns. Trailing content
        is ignored.

    Raises
    ------
    Obs80ParseError
        If the record is malformed or is a two-line radar, satellite, or
        roving-observer record.
    """
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
    return OpticalObs80(
        raw_line=line,
        designation=designation,
        discovery=line[12] == "*",
        note1=line[13].strip() or None,
        note2=note2,
        observatory_code=observatory_code,
        obstime_days_utc=obstime_days,
        obstime_nanos_utc=obstime_nanos,
        ra_deg=_parse_ra(line[32:44]),
        dec_deg=_parse_dec(line[44:56]),
        mag=magnitude,
        band=line[70].strip() or None,
        astrometric_catalog=line[71].strip() or None,
        reference=line[72:77].strip() or None,
    )


def parse_optical_obs80_file(raw: str) -> list[OpticalObs80]:
    """Parse every nonblank optical row in an MPC-format text file.

    Failure is row-local: unsupported or malformed records are omitted. Callers
    that require strict completeness should compare the returned row count to
    the source's declared observation count.
    """
    parsed: list[OpticalObs80] = []
    for line in str(raw or "").splitlines():
        if not line.strip():
            continue
        try:
            parsed.append(parse_optical_obs80(line))
        except Obs80ParseError:
            continue
    return parsed
