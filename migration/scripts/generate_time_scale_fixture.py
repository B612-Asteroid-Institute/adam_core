"""Generate the RM-STANDALONE-004 time-scale parity fixture.

The fixture is intentionally generated from the current Python Timestamp contract.
Rust time implementations must match this fixture unless a future science-policy
change explicitly updates the fixture and its documentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import astropy.time
import numpy as np

from adam_core.time import Timestamp

SCALES: tuple[str, ...] = ("utc", "tai", "tt", "tdb")
RESCALE_CORRECTNESS_SCALES: tuple[str, ...] = ("tai", "utc", "tdb", "tt", "ut1")
RESCALE_CORRECTNESS_DAYS: tuple[int, ...] = (
    -57032,
    -36525,
    -2,
    -1,
    0,
    51544,
    103088,
    164178,
    68000,
    68000,
    68010,
    68020,
)
RESCALE_CORRECTNESS_NANOS: tuple[int, ...] = (
    50_000,
    0,
    123,
    100_000_000,
    200_000_000,
    300_000_000,
    400_000_000,
    500_000_000,
    1,
    2,
    3,
    4,
)
POST_1999_LEAP_SECOND_DAYS: tuple[str, ...] = (
    "2005-12-31",
    "2008-12-31",
    "2012-06-30",
    "2015-06-30",
    "2016-12-31",
)
BASELINE_UTC_ISO: tuple[str, ...] = (
    "1985-02-01T00:00:00.000000000",
    "1999-12-31T23:59:59.000000000",
    "2000-01-01T12:00:00.000000000",
    "2023-02-25T12:34:56.789000000",
)


class TimePayload(TypedDict):
    days: list[int]
    nanos: list[int]


class RescaleCase(TypedDict):
    from_scale: str
    to_scale: str
    input: TimePayload
    output: TimePayload


class TdbEtCases(TypedDict):
    tdb: TimePayload
    mjd_tdb: list[float]
    et_seconds: list[float]


class Fixture(TypedDict):
    schema: str
    version: int
    generated_by: str
    source_contract: str
    scales: list[str]
    post_1999_leap_second_days: list[str]
    utc_iso: list[str]
    cases: list[RescaleCase]
    tdb_et_cases: TdbEtCases
    rescale_correctness_scales: list[str]
    rescale_correctness_input: TimePayload
    rescale_correctness_cases: list[RescaleCase]


def build_utc_iso() -> list[str]:
    values = list(BASELINE_UTC_ISO)
    for day in POST_1999_LEAP_SECOND_DAYS:
        next_day = np.datetime64(day) + np.timedelta64(1, "D")
        values.extend(
            [
                f"{day}T23:59:59.000000000",
                f"{day}T23:59:60.000000000",
                f"{str(next_day)}T00:00:00.000000000",
            ]
        )
    return values


def payload(timestamp: Timestamp) -> TimePayload:
    return {
        "days": [int(value) for value in timestamp.days.to_pylist()],
        "nanos": [int(value) for value in timestamp.nanos.to_pylist()],
    }


def float_list(values: list[float | None]) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None:
            raise ValueError("time fixture values must not contain nulls")
        output.append(float(value))
    return output


def build_rescale_cases() -> list[RescaleCase]:
    cases: list[RescaleCase] = []
    for from_scale in RESCALE_CORRECTNESS_SCALES:
        source = Timestamp.from_kwargs(
            days=list(RESCALE_CORRECTNESS_DAYS),
            nanos=list(RESCALE_CORRECTNESS_NANOS),
            scale=from_scale,
        )
        for to_scale in RESCALE_CORRECTNESS_SCALES:
            cases.append(
                {
                    "from_scale": from_scale,
                    "to_scale": to_scale,
                    "input": payload(source),
                    "output": payload(source.rescale(to_scale)),
                }
            )
    return cases


def build_fixture() -> Fixture:
    utc_iso = build_utc_iso()
    utc_times = Timestamp.from_astropy(
        astropy.time.Time(utc_iso, format="isot", scale="utc")
    )
    by_scale = {scale: utc_times.rescale(scale) for scale in SCALES}

    cases: list[RescaleCase] = []
    for from_scale, source in by_scale.items():
        for to_scale in SCALES:
            cases.append(
                {
                    "from_scale": from_scale,
                    "to_scale": to_scale,
                    "input": payload(source),
                    "output": payload(source.rescale(to_scale)),
                }
            )

    tdb_times = by_scale["tdb"]
    return {
        "schema": "adam_core.time_scale_rescale_fixture",
        "version": 1,
        "generated_by": "migration/scripts/generate_time_scale_fixture.py",
        "source_contract": (
            "Current Python Timestamp.rescale contract: ERFA UTC/TAI leap-second "
            "conversion, constant TAI/TT offset, project-local TT/TDB approximation, "
            "Astropy/IERS-owned UT1 conversions, and pure arithmetic MJD_TDB to ET seconds."
        ),
        "scales": list(SCALES),
        "post_1999_leap_second_days": list(POST_1999_LEAP_SECOND_DAYS),
        "utc_iso": utc_iso,
        "cases": cases,
        "tdb_et_cases": {
            "tdb": payload(tdb_times),
            "mjd_tdb": float_list(tdb_times.mjd().to_pylist()),
            "et_seconds": float_list(tdb_times.et().to_pylist()),
        },
        "rescale_correctness_scales": list(RESCALE_CORRECTNESS_SCALES),
        "rescale_correctness_input": {
            "days": list(RESCALE_CORRECTNESS_DAYS),
            "nanos": list(RESCALE_CORRECTNESS_NANOS),
        },
        "rescale_correctness_cases": build_rescale_cases(),
    }


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "migration"
        / "artifacts"
        / "time_scale_rescale_fixture_2026-05-15.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output JSON fixture path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
