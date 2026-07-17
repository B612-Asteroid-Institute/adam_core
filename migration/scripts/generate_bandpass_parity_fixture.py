"""Generate the bandpass photometry parity fixture (bead personal-cmy.24).

Run with the LEGACY baseline interpreter (untouched adam-core checkout):

    .legacy-venv/bin/python migration/scripts/generate_bandpass_parity_fixture.py

Freezes the legacy bandpasses runtime behavior: filter-id order, the full
observatory band-map replay through map_to_canonical_filter_bands (plus X05
normalization variants, SDSS/PS1 fallbacks, and error cases), template and
mix integrals across all vendored filters, V-relative delta tables, delta
mags/color terms, custom-template registration, and error messages. The
migration test replays everything against the Rust-backed functions: strings
and errors exact, integral-derived floats at rtol <= 1e-12 (numpy pairwise
vs sequential trapezoid summation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adam_core.photometry.bandpasses.api import (
    assert_filter_ids_have_curves,
    bandpass_color_terms,
    bandpass_delta_mag,
    compute_mix_integrals,
    get_integrals,
    load_bandpass_curves,
    load_observatory_band_map,
    map_to_canonical_filter_bands,
    register_custom_template,
)
from adam_core.photometry.magnitude_common import (
    bandpass_delta_table_for_composition,
)

import adam_core.photometry.bandpasses.api as _legacy_api

# The vendored parquet data files have drifted between the legacy and
# migration checkouts (curves/band-map/integrals). The fixture freezes legacy
# CODE behavior on THIS repo's data, so point the legacy loaders at the
# migration checkout's data directory before any lazy (lru_cache) loads.
_MIGRATION_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "adam_core"
    / "photometry"
    / "bandpasses"
    / "data"
)
_legacy_api._DATA_DIR = _MIGRATION_DATA_DIR

TEMPLATES = ["C", "S", "NEO", "MBA"]
MIXES = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.7, 0.3), (0.25, 1.5)]
COMPOSITIONS = ["C", "S", "NEO", "MBA", (0.5, 0.5), (2.0, 1.0)]

X05_BAND_VARIANTS = [
    "LSST_g",
    "LSST_y",
    "Lg",
    "Lr",
    "Li",
    "Lz",
    "Ly",
    "LU",
    "LY",
    "Y",
    " g ",
    "u",
]
FALLBACK_PANEL = [
    ("999", "u"),
    ("999", "g"),
    ("999", "r"),
    ("999", "i"),
    ("999", "z"),
    ("999", "y"),
    ("999", "Z"),
    ("999", "Y"),
]
ERROR_PANEL = [
    ("999", "q"),
    ("X05", "nonsense"),
]


def result_of(fn, *args, **kwargs):
    try:
        return {"output": fn(*args, **kwargs)}
    except Exception as exc:  # noqa: BLE001 - freezing legacy behavior
        return {"error_type": type(exc).__name__, "error_message": str(exc)}


def composition_payload(composition):
    if isinstance(composition, str):
        return {"template_id": composition}
    return {"mix": list(composition)}


def build_fixture() -> dict:
    curves = load_bandpass_curves()
    filter_ids = [str(x) for x in curves.filter_id.to_pylist()]

    mapping = load_observatory_band_map()
    map_codes = [str(x) for x in mapping.observatory_code.to_pylist()]
    map_bands = [str(x) for x in mapping.reported_band.to_pylist()]

    # Full band-map replay + passthrough of canonical ids + X05 variants.
    replay_codes = (
        map_codes
        + ["695"] * len(filter_ids)
        + ["X05"] * len(X05_BAND_VARIANTS)
        + [pair[0] for pair in FALLBACK_PANEL]
    )
    replay_bands = (
        map_bands
        + filter_ids
        + X05_BAND_VARIANTS
        + [pair[1] for pair in FALLBACK_PANEL]
    )
    mapped = map_to_canonical_filter_bands(replay_codes, replay_bands)

    integrals = {
        template: get_integrals(template, np.asarray(filter_ids, dtype=object)).tolist()
        for template in TEMPLATES
    }
    mixes = {
        json.dumps(list(mix)): compute_mix_integrals(
            mix[0], mix[1], np.asarray(filter_ids, dtype=object)
        ).tolist()
        for mix in MIXES
    }
    delta_tables = [
        {
            "composition": composition_payload(composition),
            "delta": bandpass_delta_table_for_composition(composition).tolist(),
        }
        for composition in COMPOSITIONS
    ]
    delta_mags = [
        {
            "composition": composition_payload(composition),
            "source": source,
            "target": target,
            "result": result_of(bandpass_delta_mag, composition, source, target),
        }
        for composition in ["C", (0.5, 0.5)]
        for (source, target) in [
            ("V", "SDSS_r"),
            ("SDSS_g", "PS1_y"),
            ("V", "V"),
            ("V", "not_a_filter"),
        ]
    ]
    color_terms = [
        {
            "composition": composition_payload("C"),
            "source": "V",
            "targets": None,
            "result": result_of(bandpass_color_terms, "C", source_filter_id="V"),
        },
        {
            "composition": composition_payload((0.5, 0.5)),
            "source": "SDSS_r",
            "targets": ["V", "SDSS_g", "PS1_y"],
            "result": result_of(
                bandpass_color_terms,
                (0.5, 0.5),
                source_filter_id="SDSS_r",
                target_filter_ids=["V", "SDSS_g", "PS1_y"],
            ),
        },
    ]

    errors = {
        "map_unknown": result_of(
            map_to_canonical_filter_bands,
            [pair[0] for pair in ERROR_PANEL],
            [pair[1] for pair in ERROR_PANEL],
        ),
        "map_fallback_disallowed": result_of(
            map_to_canonical_filter_bands,
            ["999", "695"],
            ["g", "V"],
            allow_fallback_filters=False,
        ),
        "assert_unknown": result_of(
            assert_filter_ids_have_curves, ["V", "bogus_b", "bogus_a", "bogus_b"]
        ),
        "integrals_missing_template": result_of(
            get_integrals, "NOPE", np.asarray(filter_ids[:2], dtype=object)
        ),
        "mix_negative": result_of(
            compute_mix_integrals, -1.0, 0.5, np.asarray(filter_ids[:1], dtype=object)
        ),
        "mix_zero": result_of(
            compute_mix_integrals, 0.0, 0.0, np.asarray(filter_ids[:1], dtype=object)
        ),
    }

    # Custom template: a mild synthetic reflectance slope over the visible.
    wl = np.linspace(300.0, 1100.0, 161)
    refl = 0.8 + 0.0004 * (wl - 550.0)
    register_custom_template("FIXTURE_X", wl, refl)
    custom_integrals = get_integrals(
        "FIXTURE_X", np.asarray(filter_ids, dtype=object)
    ).tolist()
    custom_errors = {
        "empty_id": result_of(register_custom_template, "", wl, refl),
        "short": result_of(
            register_custom_template, "T", np.asarray([1.0]), np.asarray([0.5])
        ),
        "not_increasing": result_of(
            register_custom_template,
            "T",
            np.asarray([1.0, 1.0]),
            np.asarray([0.5, 0.6]),
        ),
        "not_finite": result_of(
            register_custom_template,
            "T",
            np.asarray([1.0, np.nan]),
            np.asarray([0.5, 0.6]),
        ),
        "length_mismatch": result_of(
            register_custom_template,
            "T",
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([0.5, 0.6]),
        ),
    }

    return {
        "schema": "adam_core.bandpass_parity_fixture",
        "version": 1,
        "generated_by": "migration/scripts/generate_bandpass_parity_fixture.py",
        "source_contract": (
            "Legacy adam-core bandpasses runtime (numpy interp/trapezoid + "
            "pyarrow mapping), executed in the untouched legacy checkout."
        ),
        "filter_ids": filter_ids,
        "map_replay": {
            "codes": replay_codes,
            "bands": replay_bands,
            "mapped": [str(x) for x in mapped.tolist()],
        },
        "integrals": integrals,
        "mix_integrals": mixes,
        "delta_tables": delta_tables,
        "delta_mags": delta_mags,
        "color_terms": color_terms,
        "errors": errors,
        "custom_template": {
            "wavelength_nm": wl.tolist(),
            "reflectance": refl.tolist(),
            "integrals": custom_integrals,
            "errors": custom_errors,
        },
    }


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "migration"
        / "artifacts"
        / "bandpass_parity_fixture_2026-07-06.json"
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
