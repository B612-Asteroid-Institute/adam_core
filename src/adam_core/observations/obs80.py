"""Typed MPC 80-column optical observations.

Parsing and Arrow table assembly are Rust-owned. The Python layer only defines
quivr schemas, preserves the public exception, and wraps one native crossing.
"""

from __future__ import annotations

import quivr as qv
from quivr.validators import and_, ge, le

from ..time import Timestamp

NANOSECONDS_PER_DAY = 86_400_000_000_000


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


def _parse_native(raw: str, *, strict: bool, file: bool) -> OpticalObs80:
    from adam_core import _rust_native  # type: ignore[attr-defined]

    from .._rust.arrow import table_from_record_batch

    try:
        batch = _rust_native.parse_optical_obs80_arrow(raw, strict, file)
    except ValueError as exc:
        raise Obs80ParseError(str(exc)) from exc
    return table_from_record_batch(OpticalObs80, batch)


def parse_optical_obs80(raw: str) -> OpticalObs80:
    """Parse one self-contained optical MPC 80-column record."""
    if not isinstance(raw, str) or len(raw) < 80:
        raise Obs80ParseError("record is shorter than 80 columns")
    return _parse_native(raw, strict=True, file=False)


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
    return _parse_native(str(raw or ""), strict=bool(strict), file=True)
