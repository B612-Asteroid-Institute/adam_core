"""Generate the ADES writer/parser parity fixture (bead personal-cmy.20).

Run this with the LEGACY baseline interpreter (the untouched adam-core
checkout, which has had no Rust work), so the migration checkout is gated
against the true legacy contract rather than an in-repo holdout:

    .legacy-venv/bin/python migration/scripts/generate_ades_parity_fixture.py

The fixture freezes, for seeded randomized observation panels (mixed nulls,
NaN values, pipes/quotes/unicode/padding in remarks, duplicated rows,
non-utc input scales) and several writer option sets:

* the exact ``ADES_to_string`` output string;
* the exact ``ADES_string_to_tables`` parse of that string (flat observation
  columns + ObsContext dicts);
* hand-crafted raw ADES strings (unknown columns, whitespace padding,
  missing optional columns) and their legacy parses;
* the legacy error messages for missing contexts / missing IDs.

The migration test
``src/adam_core/observations/tests/test_ades_rust_parity.py`` asserts the
Rust-dispatched public functions reproduce all of it byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from adam_core.observations.ades import (
    ADES_string_to_tables,
    ADES_to_string,
    ADESObservations,
    ObsContext,
    ObservatoryObsContext,
    SoftwareObsContext,
    SubmitterObsContext,
    TelescopeObsContext,
)
from adam_core.time import Timestamp

CODES = ["695", "W84", "X05", "V00"]

CONTEXT_SPEC = {
    code: {
        "observatory": {"mpcCode": code, "name": f"Observatory {code}"},
        "submitter": {"name": "J. Doe", "institution": "B612"},
        "observers": ["J. Doe", "A. Smith"],
        "measurers": ["J. Doe"],
        "telescope": {
            "design": "Reflector",
            "aperture": 4.0,
            "detector": "CCD",
            "name": f"Telescope {code}",
        },
        "software": {"objectDetection": "ADAM::THOR"},
        "fundingSource": "Test, Funding|Source",
        "comments": ["THIS IS A TEST FILE", "second comment line"],
    }
    for code in CODES
}

REMARKS_PANEL = [
    "plain remark",
    "remark|with|pipes",
    'remark with "quotes"',
    "remark with \u00fcnicode",
    "  padded remark  ",
    None,
]

RAW_STRINGS = {
    "minimal_with_unknowns": (
        "# version=2022\n"
        "# observatory\n"
        "! mpcCode X05\n"
        "# submitter\n"
        "! name J. Doe\n"
        "# observers\n"
        "! name J. Doe\n"
        "# measurers\n"
        "! name J. Doe\n"
        "# telescope\n"
        "! design Reflector\n"
        "! aperture 1.0\n"
        "! detector CCD\n"
        "permID |trkSub |obsTime                |ra        |dec      |rmsRA |rmsDec |stn |mode |astCat |obsCenter |remarks\n"
        "12345  |t1     |2024-05-04T00:00:00.00Z|240.0     |-15.0    |0.5   |0.6    |X05 |CCD  |Gaia2  |whatever  | keep pad \n"
        "       |t2     |2024-05-04T02:24:00.00Z|240.05    |-15.05   |      |       |X05 |CCD  |Gaia2  |whatever  |\n"
    ),
    "multiple_blocks": (
        "# version=2022\n"
        "# observatory\n"
        "! mpcCode 695\n"
        "# submitter\n"
        "! name J. Doe\n"
        "# observers\n"
        "! name J. Doe\n"
        "# measurers\n"
        "! name J. Doe\n"
        "# telescope\n"
        "! design Reflector\n"
        "! aperture 1.0\n"
        "! detector CCD\n"
        "trkSub|obsTime|ra|dec|stn|mode|astCat\n"
        "a1|2024-05-04T00:00:00.000Z|240.000000001|-15.5|695|CCD|Gaia2\n"
        "# observatory\n"
        "! mpcCode W84\n"
        "# submitter\n"
        "! name J. Doe\n"
        "# observers\n"
        "! name J. Doe\n"
        "# measurers\n"
        "! name J. Doe\n"
        "# telescope\n"
        "! design Reflector\n"
        "! aperture 2.0\n"
        "! detector CCD\n"
        "permID|obsTime|ra|dec|mag|stn|mode|astCat\n"
        "99942|2024-06-01T12:00:00.500Z|1.25|-0.5|19.75|W84|CCD|Gaia3\n"
    ),
}


def build_contexts(spec: dict) -> dict[str, ObsContext]:
    contexts = {}
    for code, entry in spec.items():
        contexts[code] = ObsContext(
            observatory=ObservatoryObsContext(**entry["observatory"]),
            submitter=SubmitterObsContext(**entry["submitter"]),
            observers=list(entry["observers"]),
            measurers=list(entry["measurers"]),
            telescope=TelescopeObsContext(**entry["telescope"]),
            software=(
                SoftwareObsContext(**entry["software"])
                if entry.get("software")
                else None
            ),
            fundingSource=entry.get("fundingSource"),
            comments=list(entry["comments"]) if entry.get("comments") else None,
        )
    return contexts


def random_observations(seed: int, n: int, scale: str) -> ADESObservations:
    rng = np.random.default_rng(seed)

    def opt_str(prefix, present_p):
        return [
            f"{prefix}{rng.integers(0, 5000):04d}" if rng.random() < present_p else None
            for _ in range(n)
        ]

    def opt_float(low, high, present_p, nan_p=0.0):
        out = []
        for _ in range(n):
            draw = rng.random()
            if draw < present_p:
                out.append(float(rng.uniform(low, high)))
            elif draw < present_p + nan_p:
                out.append(float("nan"))
            else:
                out.append(None)
        return out

    perm_id = opt_str("3", 0.6)
    prov_id = opt_str("2024 A", 0.5)
    trk_sub = opt_str("trk", 0.7)
    trk_sub = [
        trk if (perm is not None or prov is not None or trk is not None) else "trk9999"
        for perm, prov, trk in zip(perm_id, prov_id, trk_sub)
    ]

    observations = ADESObservations.from_kwargs(
        permID=perm_id,
        provID=prov_id,
        trkSub=trk_sub,
        obsSubID=opt_str("obs", 0.8),
        obsTime=Timestamp.from_mjd(rng.uniform(59000, 61000, n), scale=scale),
        rmsTime=opt_float(0.001, 10.0, 0.5, nan_p=0.1),
        ra=rng.uniform(0, 360, n),
        dec=rng.uniform(-89, 89, n),
        rmsRACosDec=opt_float(0.01, 2.0, 0.6),
        rmsDec=opt_float(0.01, 2.0, 0.6),
        rmsCorr=opt_float(-0.9, 0.9, 0.4),
        mag=opt_float(15.0, 25.0, 0.6, nan_p=0.1),
        rmsMag=opt_float(0.01, 0.5, 0.5),
        band=[rng.choice(["g", "r", "i", None]) for _ in range(n)],
        stn=[str(rng.choice(CODES)) for _ in range(n)],
        mode=["CCD"] * n,
        astCat=[str(rng.choice(["Gaia2", "Gaia3"])) for _ in range(n)],
        photCat=[rng.choice(["Gaia2", None]) for _ in range(n)],
        logSNR=opt_float(0.1, 3.0, 0.4),
        seeing=opt_float(0.5, 3.0, 0.4),
        exp=opt_float(10.0, 120.0, 0.6),
        remarks=[
            REMARKS_PANEL[int(rng.integers(0, len(REMARKS_PANEL)))] for _ in range(n)
        ],
    )
    # Duplicate a few complete rows so stable sorting sees full ties.
    return ADESObservations.from_pyarrow(
        observations.table.take(list(range(n)) + [0, 1, 2, 0])
    )


def observations_to_flat(observations: ADESObservations) -> dict:
    flat = {}
    for column in observations.table.column_names:
        if column.startswith("obsTime"):
            continue
        flat[column] = observations.table.column(column).to_pylist()
    flat["obsTime"] = {
        "days": observations.obsTime.days.to_pylist(),
        "nanos": observations.obsTime.nanos.to_pylist(),
        "scale": observations.obsTime.scale,
    }
    return flat


def parse_to_payload(ades_string: str) -> dict:
    # The legacy writer emits the string "nan" for missing values in
    # precision-formatted columns, and the legacy parser then fails quivr
    # validation on those files (NaN in rmsRACosDec/rmsDec). That crash is
    # part of the legacy contract, so freeze the error too.
    try:
        contexts, observations = ADES_string_to_tables(ades_string)
    except Exception as exc:  # noqa: BLE001 - freezing legacy behavior
        return {
            "parse_error_type": type(exc).__name__,
            "parse_error_message": str(exc),
        }
    return {
        "observations": observations_to_flat(observations),
        "contexts": {code: asdict(context) for code, context in contexts.items()},
    }


PANELS = [
    {"name": "seed0_default", "seed": 0, "n": 96, "scale": "utc", "options": None},
    {"name": "seed1_default", "seed": 1, "n": 96, "scale": "utc", "options": None},
    {
        "name": "seed3_custom_low",
        "seed": 3,
        "n": 64,
        "scale": "utc",
        "options": {
            "seconds_precision": 1,
            "columns_precision": {"ra": 6, "dec": 6, "mag": 1},
        },
    },
    {
        "name": "seed4_custom_high",
        "seed": 4,
        "n": 64,
        "scale": "utc",
        "options": {
            "seconds_precision": 6,
            "columns_precision": {
                "ra": 11,
                "dec": 11,
                "rmsRACosDec": 4,
                "rmsDec": 4,
            },
        },
    },
    {"name": "seed7_tdb_input", "seed": 7, "n": 48, "scale": "tdb", "options": None},
]


def build_fixture() -> dict:
    contexts = build_contexts(CONTEXT_SPEC)

    panels = []
    for spec in PANELS:
        observations = random_observations(spec["seed"], spec["n"], spec["scale"])
        kwargs = {}
        if spec["options"]:
            kwargs = dict(spec["options"])
        ades_string = ADES_to_string(observations, contexts, **kwargs)
        panels.append(
            {
                "name": spec["name"],
                "options": spec["options"],
                "observations": observations_to_flat(observations),
                "ades_string": ades_string,
                "parsed": parse_to_payload(ades_string),
            }
        )

    raw_strings = [
        {
            "name": name,
            "ades_string": ades_string,
            "parsed": parse_to_payload(ades_string),
        }
        for name, ades_string in RAW_STRINGS.items()
    ]

    errors = {}
    missing_context = build_contexts(
        {code: CONTEXT_SPEC[code] for code in ["695", "W84", "X05"]}
    )
    observations = random_observations(11, 32, "utc")
    try:
        ADES_to_string(observations, missing_context)
    except ValueError as exc:
        errors["missing_context"] = str(exc)
    no_ids = ADESObservations.from_kwargs(
        obsTime=Timestamp.from_mjd([60434.0], scale="utc"),
        ra=[10.0],
        dec=[-5.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
    )
    try:
        ADES_to_string(no_ids, build_contexts(CONTEXT_SPEC))
    except ValueError as exc:
        errors["missing_ids"] = str(exc)

    return {
        "schema": "adam_core.ades_parity_fixture",
        "version": 1,
        "generated_by": "migration/scripts/generate_ades_parity_fixture.py",
        "source_contract": (
            "Legacy adam-core ADES writer/parser (pandas to_csv + astropy isot "
            "formatting), executed in the untouched legacy checkout."
        ),
        "context_spec": CONTEXT_SPEC,
        "panels": panels,
        "raw_strings": raw_strings,
        "errors": errors,
    }


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "migration"
        / "artifacts"
        / "ades_parity_fixture_2026-07-05.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output_path())
    args = parser.parse_args()
    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
