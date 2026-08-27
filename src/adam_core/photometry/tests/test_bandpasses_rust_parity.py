"""Parity gate for the Rust bandpasses runtime (bead personal-cmy.24) against
the frozen legacy fixture generated in the untouched legacy checkout via
``.legacy-venv/bin/python migration/scripts/generate_bandpass_parity_fixture.py``.

Strings, mappings, and error messages must match exactly; integral-derived
floats are gated at rtol <= 1e-12 (numpy pairwise vs Rust sequential
trapezoid summation -- documented deviation)."""

import json
from pathlib import Path

import numpy as np
import pytest

from adam_core.photometry.bandpasses.api import (
    assert_filter_ids_have_curves,
    bandpass_color_terms,
    bandpass_delta_mag,
    compute_mix_integrals,
    get_integrals,
    map_to_canonical_filter_bands,
    register_custom_template,
)
from adam_core.photometry.magnitude_common import (
    bandpass_delta_table_for_composition,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration"
    / "artifacts"
    / "bandpass_parity_fixture_2026-07-06.json"
)

RTOL = 1e-12


@pytest.fixture(scope="module")
def fixture():
    assert FIXTURE_PATH.exists(), (
        "bandpass parity fixture missing; generate with the legacy interpreter: "
        ".legacy-venv/bin/python migration/scripts/generate_bandpass_parity_fixture.py"
    )
    return json.loads(FIXTURE_PATH.read_text())


def composition_from(payload):
    if "template_id" in payload:
        return payload["template_id"]
    return tuple(payload["mix"])


def assert_error(result_fn, expected, label):
    with pytest.raises(Exception) as exc_info:
        result_fn()
    assert type(exc_info.value).__name__ == expected["error_type"], label
    assert str(exc_info.value) == expected["error_message"], label


def test_filter_ids_match(fixture):
    from adam_core import _rust_native as _rn
    from adam_core.photometry.bandpasses.api import _data_dir_str

    assert _rn.bandpasses_filter_ids(_data_dir_str()) == fixture["filter_ids"]


def test_map_replay_matches_legacy(fixture):
    replay = fixture["map_replay"]
    mapped = map_to_canonical_filter_bands(replay["codes"], replay["bands"])
    assert [str(x) for x in mapped.tolist()] == replay["mapped"]


def test_template_and_mix_integrals_match_legacy(fixture):
    filter_ids = np.asarray(fixture["filter_ids"], dtype=object)
    for template, expected in fixture["integrals"].items():
        actual = get_integrals(template, filter_ids)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0)
    for mix_key, expected in fixture["mix_integrals"].items():
        w_c, w_s = json.loads(mix_key)
        actual = compute_mix_integrals(w_c, w_s, filter_ids)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0)


def test_delta_tables_match_legacy(fixture):
    for case in fixture["delta_tables"]:
        composition = composition_from(case["composition"])
        actual = bandpass_delta_table_for_composition(composition)
        # Deltas are differences of logs; keep the same rtol with a small atol
        # floor for entries that are exactly ~0 (V relative to itself).
        np.testing.assert_allclose(actual, case["delta"], rtol=RTOL, atol=1e-13)


def test_delta_mags_match_legacy(fixture):
    for case in fixture["delta_mags"]:
        composition = composition_from(case["composition"])
        label = f"delta_mag {case['composition']} {case['source']}->{case['target']}"
        expected = case["result"]
        if "output" in expected:
            actual = bandpass_delta_mag(composition, case["source"], case["target"])
            np.testing.assert_allclose(
                actual, expected["output"], rtol=RTOL, atol=1e-13
            ), label
        else:
            assert_error(
                lambda: bandpass_delta_mag(composition, case["source"], case["target"]),
                expected,
                label,
            )


def test_color_terms_match_legacy(fixture):
    for case in fixture["color_terms"]:
        composition = composition_from(case["composition"])
        kwargs = {"source_filter_id": case["source"]}
        if case["targets"] is not None:
            kwargs["target_filter_ids"] = case["targets"]
        expected = case["result"]["output"]
        actual = bandpass_color_terms(composition, **kwargs)
        assert set(actual) == set(expected)
        for key, value in expected.items():
            np.testing.assert_allclose(actual[key], value, rtol=RTOL, atol=1e-13)


def test_error_messages_match_legacy(fixture):
    errors = fixture["errors"]
    assert_error(
        lambda: map_to_canonical_filter_bands(["999", "X05"], ["q", "nonsense"]),
        errors["map_unknown"],
        "map_unknown",
    )
    assert_error(
        lambda: map_to_canonical_filter_bands(
            ["999", "695"], ["g", "V"], allow_fallback_filters=False
        ),
        errors["map_fallback_disallowed"],
        "map_fallback_disallowed",
    )
    assert_error(
        lambda: assert_filter_ids_have_curves(["V", "bogus_b", "bogus_a", "bogus_b"]),
        errors["assert_unknown"],
        "assert_unknown",
    )
    filter_ids = fixture["filter_ids"]
    assert_error(
        lambda: get_integrals("NOPE", np.asarray(filter_ids[:2], dtype=object)),
        errors["integrals_missing_template"],
        "integrals_missing_template",
    )
    assert_error(
        lambda: compute_mix_integrals(
            -1.0, 0.5, np.asarray(filter_ids[:1], dtype=object)
        ),
        errors["mix_negative"],
        "mix_negative",
    )
    assert_error(
        lambda: compute_mix_integrals(
            0.0, 0.0, np.asarray(filter_ids[:1], dtype=object)
        ),
        errors["mix_zero"],
        "mix_zero",
    )


def test_custom_template_matches_legacy(fixture):
    from adam_core import _rust_native as _rn

    custom = fixture["custom_template"]
    wl = np.asarray(custom["wavelength_nm"], dtype=np.float64)
    refl = np.asarray(custom["reflectance"], dtype=np.float64)
    _rn.bandpasses_clear_custom_templates()
    try:
        register_custom_template("FIXTURE_X", wl, refl)
        actual = get_integrals(
            "FIXTURE_X", np.asarray(fixture["filter_ids"], dtype=object)
        )
        np.testing.assert_allclose(actual, custom["integrals"], rtol=RTOL, atol=0.0)

        for label, expected in custom["errors"].items():
            if label == "empty_id":
                args = ("", wl, refl)
            elif label == "short":
                args = ("T", np.asarray([1.0]), np.asarray([0.5]))
            elif label == "not_increasing":
                args = ("T", np.asarray([1.0, 1.0]), np.asarray([0.5, 0.6]))
            elif label == "not_finite":
                args = ("T", np.asarray([1.0, np.nan]), np.asarray([0.5, 0.6]))
            else:
                args = ("T", np.asarray([1.0, 2.0, 3.0]), np.asarray([0.5, 0.6]))
            assert_error(
                lambda args=args: register_custom_template(*args), expected, label
            )
    finally:
        _rn.bandpasses_clear_custom_templates()
