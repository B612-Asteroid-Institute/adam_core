"""Generate the Timestamp-arithmetic parity fixture (bead personal-cmy.19).

The fixture freezes the legacy-baseline Timestamp arithmetic contract:
add_days / add_nanos / add_fractional_days, difference / difference_scalar,
mjd / jd projection, and the from_mjd / from_jd float-quantization round-trip,
over a fixed epoch matrix that includes leap-second-adjacent UTC days and
nanos values chosen to exercise day-boundary overflow/underflow.

Generate it with the legacy baseline interpreter so the migration checkout is
gated against the legacy contract, not against itself:

    .legacy-venv/bin/python migration/scripts/generate_timestamp_arithmetic_fixture.py

The migration test
``src/adam_core/time/tests/test_time.py::test_timestamp_arithmetic_fixture_matches_legacy_contract``
asserts bit-exact reproduction (integer days/nanos equality, exact float
equality for mjd/jd).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa

from adam_core.time import Timestamp

SCALES: tuple[str, ...] = ("utc", "tai", "tt", "tdb")

# Epoch matrix: negative MJD, epoch zero, J2000, leap-second-adjacent UTC days
# (2005-12-31/2006-01-01 -> 53735/53736, 2015-06-30/07-01 -> 57203/57204,
# 2016-12-31/2017-01-01 -> 57753/57754), modern and far-future days. Nanos are
# chosen to sit at day boundaries so the arithmetic over/underflow branches
# are exercised.
EPOCH_DAYS: tuple[int, ...] = (
    -36525,
    0,
    51544,
    51544,
    53735,
    53736,
    57203,
    57204,
    57753,
    57754,
    59000,
    60000,
    62502,
    68000,
    40000,
    51179,
)
EPOCH_NANOS: tuple[int, ...] = (
    0,
    1,
    43_200_000_000_000,
    86_399_999_999_999,
    86_399_000_000_000,
    0,
    86_340_000_000_000,
    500,
    86_399_999_999_999,
    1,
    123_456_789,
    86_399_500_000_000,
    43_200_000_000_001,
    999_999_999,
    100,
    86_399_999_999_000,
)

ADD_DAYS_ARG: tuple[int, ...] = (
    1,
    -1,
    365,
    -365,
    10_000,
    -10_000,
    0,
    7,
    -7,
    30,
    -30,
    100,
    -100,
    1,
    -1,
    365,
)
ADD_NANOS_ARG: tuple[int, ...] = (
    1,
    -1,
    86_399_999_999_999,
    -86_399_999_999_999,
    43_200_000_000_000,
    -43_200_000_000_000,
    999,
    -999,
    1_000_000_000,
    -1_000_000_000,
    0,
    86_399_999_999_999,
    -1,
    1,
    -86_399_999_999_999,
    500,
)
ADD_FRACTIONAL_DAYS_ARG: tuple[float, ...] = (
    0.5,
    -0.5,
    1.25,
    -1.25,
    2.000000001,
    -2.000000001,
    0.0,
    10.75,
    -10.75,
    0.123456789,
    -0.123456789,
    365.5,
    -365.5,
    0.999999999,
    -0.999999999,
    1e-9,
)
DIFFERENCE_SCALAR_ARG: dict[str, int] = {"days": 51544, "nanos": 43_200_000_000_000}


def payload(timestamp: Timestamp) -> dict[str, list[int]]:
    return {
        "days": [int(value) for value in timestamp.days.to_pylist()],
        "nanos": [int(value) for value in timestamp.nanos.to_pylist()],
    }


def int_lists(days: pa.Int64Array, nanos: pa.Int64Array) -> dict[str, list[int]]:
    return {
        "days": [int(value) for value in days.to_pylist()],
        "nanos": [int(value) for value in nanos.to_pylist()],
    }


def float_list(array: pa.lib.DoubleArray) -> list[float]:
    values = array.to_pylist()
    if any(value is None for value in values):
        raise ValueError("time fixture values must not contain nulls")
    return [float(value) for value in values]


def build_case(scale: str) -> dict:
    epochs = Timestamp.from_kwargs(
        days=list(EPOCH_DAYS), nanos=list(EPOCH_NANOS), scale=scale
    )
    other = epochs.add_days(pa.array(ADD_DAYS_ARG, pa.int64())).add_nanos(
        pa.array(ADD_NANOS_ARG, pa.int64())
    )
    diff_days, diff_nanos = epochs.difference(other)
    scalar_days, scalar_nanos = epochs.difference_scalar(
        DIFFERENCE_SCALAR_ARG["days"], DIFFERENCE_SCALAR_ARG["nanos"]
    )
    mjd = epochs.mjd()
    jd = epochs.jd()
    return {
        "scale": scale,
        "input": payload(epochs),
        "ops": {
            "add_days": payload(epochs.add_days(pa.array(ADD_DAYS_ARG, pa.int64()))),
            "add_nanos": payload(epochs.add_nanos(pa.array(ADD_NANOS_ARG, pa.int64()))),
            "add_fractional_days": payload(
                epochs.add_fractional_days(
                    pa.array(ADD_FRACTIONAL_DAYS_ARG, pa.float64())
                )
            ),
            "difference_other": payload(other),
            "difference": int_lists(diff_days, diff_nanos),
            "difference_scalar": int_lists(scalar_days, scalar_nanos),
            "mjd": float_list(mjd),
            "jd": float_list(jd),
            "from_mjd_roundtrip": payload(Timestamp.from_mjd(mjd, scale=scale)),
            "from_jd_roundtrip": payload(Timestamp.from_jd(jd, scale=scale)),
        },
    }


def build_fixture() -> dict:
    return {
        "schema": "adam_core.timestamp_arithmetic_fixture",
        "version": 1,
        "generated_by": "migration/scripts/generate_timestamp_arithmetic_fixture.py",
        "source_contract": (
            "Legacy-baseline Python Timestamp arithmetic contract: pyarrow "
            "integer day/nano arithmetic with day-boundary carry, float "
            "fractional-day decomposition (floor + truncating nano cast), "
            "difference normalization to nanos in [0, 86400e9), and the "
            "from_mjd/from_jd rounding quantization."
        ),
        "scales": list(SCALES),
        "epoch_days": list(EPOCH_DAYS),
        "epoch_nanos": list(EPOCH_NANOS),
        "add_days_arg": list(ADD_DAYS_ARG),
        "add_nanos_arg": list(ADD_NANOS_ARG),
        "add_fractional_days_arg": list(ADD_FRACTIONAL_DAYS_ARG),
        "difference_scalar_arg": dict(DIFFERENCE_SCALAR_ARG),
        "cases": [build_case(scale) for scale in SCALES],
    }


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "migration"
        / "artifacts"
        / "timestamp_arithmetic_fixture_2026-07-05.json"
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
