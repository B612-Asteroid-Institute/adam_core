import numpy as np
import pyarrow.compute as pc
import pytest

from adam_core.photometry.bandpasses import (
    bandpass_color_terms,
    bandpass_delta_mag,
    compute_mix_integrals,
    get_integrals,
    load_asteroid_templates,
    load_bandpass_curves,
    load_observatory_band_map,
    map_to_canonical_filter_bands,
    native_band_for,
)


def test_observatory_band_map_covers_required_pairs():
    mapping = load_observatory_band_map()
    assert len(mapping) > 0

    # Minimal coverage checks plus explicit dataset coverage needed by our MPC data.
    required = [
        ("695", "z"),
        ("I41", "i"),
        ("I41", "r"),
        ("I41", "g"),
        ("M22", "o"),
        ("M22", "c"),
        ("T05", "c"),
        ("T05", "o"),
        ("T08", "c"),
        ("T08", "o"),
        ("V00", "g"),
        ("V00", "r"),
        ("W68", "c"),
        ("W68", "o"),
        ("W84", "r"),
        ("W84", "i"),
        ("W84", "g"),
        ("W84", "u"),
        ("W84", "VR"),
        ("W84", "z"),
        ("W84", "Y"),
        # Existing mappings we already rely on elsewhere.
        ("Q55", "v"),
        ("X05", "u"),
        ("X05", "g"),
        ("X05", "r"),
        ("X05", "i"),
        ("X05", "z"),
        ("X05", "y"),
        # Pan-STARRS1 wide filter (PS1.w), MPC codes F51 / F52.
        ("F51", "w"),
        ("F52", "w"),
        # MPC-prefixed ATLAS bands ("Ao"/"Ac") at every ATLAS site. The
        # native equivalents ("o"/"c") are tested above; both forms must
        # resolve so vendor exposure-index lookups work for both native
        # ATLAS rows and MPC obs80 ingestions of the same observations.
        ("T05", "Ao"),
        ("T05", "Ac"),
        ("T08", "Ao"),
        ("T08", "Ac"),
        ("M22", "Ao"),
        ("M22", "Ac"),
        ("W68", "Ao"),
        ("W68", "Ac"),
    ]
    for code, band in required:
        key = f"{code}|{band}"
        assert bool(pc.any(pc.equal(mapping.key, key)).as_py())


def test_native_band_for_native_passthrough():
    """For already-native band strings, ``native_band_for`` returns the same
    string unchanged."""
    assert native_band_for("T05", "o") == "o"
    assert native_band_for("T08", "c") == "c"
    assert native_band_for("I41", "g") == "g"
    assert native_band_for("I41", "r") == "r"
    assert native_band_for("X05", "y") == "y"
    assert native_band_for("W84", "VR") == "VR"


def test_native_band_for_mpc_prefixed_atlas():
    """MPC obs80 ingests ATLAS observations with a leading "A" prefix
    ("Ao"/"Ac"); the native form is "o"/"c". ``native_band_for`` must
    resolve both equivalently at every ATLAS site.
    """
    for code in ("T05", "T08", "M22", "W68"):
        assert native_band_for(code, "Ao") == "o"
        assert native_band_for(code, "Ac") == "c"


def test_native_band_for_unknown_returns_none():
    """Unmapped (observatory, band) pairs return None rather than raising.
    Callers (e.g. cutouts engine) can fall back to passing the original
    reported band, omitting the filter from a search, or surfacing a
    warning.
    """
    assert native_band_for("XXX", "g") is None
    assert native_band_for("T05", "bogus") is None
    assert native_band_for("", "o") is None
    assert native_band_for("T05", "") is None


def test_map_to_canonical_filter_bands_strict_happy_path():
    resolved = map_to_canonical_filter_bands(
        ["W84", "I41", "X05"],
        ["g", "r", "y"],
        allow_fallback_filters=False,
    )
    assert resolved.tolist() == ["DECam_g", "ZTF_r", "LSST_y"]


def test_x05_normalizes_mpc_l_band_encodings():
    # MPC/ADES encodings for X05 can use 'L*' or 'LSST_*' variants.
    out = map_to_canonical_filter_bands(
        ["X05", "X05", "X05", "X05", "X05", "X05"],
        ["Lg", "Lr", "Li", "LSST_g", "Y", "LY"],
        allow_fallback_filters=False,
    )
    assert out.tolist() == ["LSST_g", "LSST_r", "LSST_i", "LSST_g", "LSST_y", "LSST_y"]


def test_find_suggested_filter_bands_passes_through_canonical_ids():
    # If already canonical, observatory code should not matter.
    out = map_to_canonical_filter_bands(["XXX", "W84"], ["LSST_g", "DECam_r"])
    assert out.tolist() == ["LSST_g", "DECam_r"]


def test_find_suggested_filter_bands_uses_mapping_table():
    out = map_to_canonical_filter_bands(["W84", "T08", "V00"], ["VR", "c", "g"])
    assert out.tolist() == ["DECam_VR", "ATLAS_c", "BASS_g"]


def test_find_suggested_filter_bands_fallback_default_is_non_strict():
    # Unknown observatory codes fall back for generic bands.
    out = map_to_canonical_filter_bands(["XXX", "YYY", "ZZZ"], ["g", "z", "y"])
    assert out.tolist() == ["SDSS_g", "SDSS_z", "PS1_y"]


def test_find_suggested_filter_bands_strict_disallows_fallback():
    with pytest.raises(ValueError, match="No non-fallback mapping found"):
        map_to_canonical_filter_bands(["XXX"], ["g"], allow_fallback_filters=False)


def test_find_suggested_filter_bands_raises_for_unknown_band_even_non_strict():
    with pytest.raises(ValueError, match="Unable to suggest canonical filter_id"):
        map_to_canonical_filter_bands(
            ["XXX"], ["not_a_band"], allow_fallback_filters=True
        )


def test_map_ps1_w_resolves_for_f51_and_f52():
    out = map_to_canonical_filter_bands(
        ["F51", "F52"], ["w", "w"], allow_fallback_filters=False
    )
    assert out.tolist() == ["PS1_w", "PS1_w"]


def test_map_w_does_not_resolve_for_non_ps1_observatories():
    # Amateur / non-PS1 stations that report band "w" must NOT be silently
    # mapped to PS1_w; their reported band is observatory-specific clear-glass
    # response with no canonical mapping.
    with pytest.raises(ValueError, match="Unable to suggest canonical filter_id"):
        map_to_canonical_filter_bands(
            ["C41", "C94", "J84"], ["w", "w", "w"], allow_fallback_filters=True
        )


def test_on_unknown_skip_returns_none_for_unmappable_rows():
    out = map_to_canonical_filter_bands(
        ["C41", "844", "W84"],
        ["w", None, "g"],
        allow_fallback_filters=True,
        on_unknown="skip",
    )
    # C41|w and 844|None are not in the table or fallback set; W84|g resolves.
    assert out.tolist() == [None, None, "DECam_g"]


def test_on_unknown_skip_does_not_raise_with_all_unmappable():
    out = map_to_canonical_filter_bands(
        ["C41", "C94"],
        ["w", "w"],
        on_unknown="skip",
    )
    assert out.tolist() == [None, None]


def test_on_unknown_skip_cooperates_with_allow_fallback_filters():
    # Generic-band fallback still applies under on_unknown="skip".
    out = map_to_canonical_filter_bands(
        ["XXX", "YYY", "C41"],
        ["g", "y", "w"],
        allow_fallback_filters=True,
        on_unknown="skip",
    )
    assert out.tolist() == ["SDSS_g", "PS1_y", None]


def test_on_unknown_skip_with_no_fallbacks_leaves_generic_bands_unmapped():
    # When fallbacks are off and on_unknown="skip", generic bands without an
    # explicit observatory mapping become None rather than raising or
    # falling back.
    out = map_to_canonical_filter_bands(
        ["XXX"], ["g"],
        allow_fallback_filters=False,
        on_unknown="skip",
    )
    assert out.tolist() == [None]


def test_on_unknown_raise_is_default_behavior():
    # Default (no on_unknown kwarg) must keep raising on unknown tuples, to
    # preserve backward compatibility for existing callers.
    with pytest.raises(ValueError, match="Unable to suggest canonical filter_id"):
        map_to_canonical_filter_bands(["C41"], ["w"])


def test_on_unknown_skip_keeps_canonical_pass_through():
    # Rows whose band is already a known canonical filter_id are passed through
    # regardless of on_unknown; this guards against accidental nulling of
    # explicitly-canonical input.
    out = map_to_canonical_filter_bands(
        ["WHATEVER", "C41"],
        ["LSST_g", "w"],
        on_unknown="skip",
    )
    assert out.tolist() == ["LSST_g", None]


def test_on_unknown_invalid_value_raises_immediately():
    with pytest.raises(ValueError, match="on_unknown must be 'raise' or 'skip'"):
        map_to_canonical_filter_bands(["W84"], ["g"], on_unknown="warn")  # type: ignore[arg-type]


def test_bandpass_curves_are_sane():
    curves = load_bandpass_curves()
    assert len(curves) > 0

    filter_ids = set(curves.filter_id.to_pylist())
    assert {"Bessell_U", "Bessell_B", "Bessell_R", "Bessell_I"}.issubset(filter_ids)
    assert {"SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"}.issubset(filter_ids)
    assert {"PS1_g", "PS1_r", "PS1_i", "PS1_z", "PS1_y", "PS1_w"}.issubset(filter_ids)
    assert {"DECam_VR", "BASS_g", "BASS_r"}.issubset(filter_ids)

    for wl_list, thr_list in zip(
        curves.wavelength_nm.to_pylist(), curves.throughput.to_pylist()
    ):
        wl = np.asarray(wl_list, dtype=float)
        thr = np.asarray(thr_list, dtype=float)
        assert wl.ndim == 1
        assert thr.ndim == 1
        assert len(wl) == len(thr)
        assert len(wl) >= 2
        assert np.all(np.isfinite(wl))
        assert np.all(np.isfinite(thr))
        assert np.all(np.diff(wl) > 0)
        assert np.min(thr) >= 0.0
        assert np.max(thr) <= 1.0 + 1e-12


def test_templates_exist_and_weights_sum_to_one():
    templates = load_asteroid_templates()
    ids = set(templates.template_id.to_pylist())
    assert {"C", "S", "NEO", "MBA"}.issubset(ids)

    w_c = np.asarray(templates.weight_C.to_numpy(zero_copy_only=False), dtype=float)
    w_s = np.asarray(templates.weight_S.to_numpy(zero_copy_only=False), dtype=float)
    assert np.allclose(w_c + w_s, 1.0)


def test_mix_integrals_match_linear_combination():
    filter_ids = np.asarray(
        ["V", "LSST_g", "DECam_r", "ZTF_r", "ATLAS_c"], dtype=object
    )
    ints_neo = get_integrals("NEO", filter_ids)
    ints_mba = get_integrals("MBA", filter_ids)

    ints_neo_mix = compute_mix_integrals(0.5, 0.5, filter_ids)
    ints_mba_mix = compute_mix_integrals(0.7, 0.3, filter_ids)

    assert np.allclose(ints_neo, ints_neo_mix, rtol=0, atol=1e-9)
    assert np.allclose(ints_mba, ints_mba_mix, rtol=0, atol=1e-9)


def test_bandpass_delta_mag_matches_integral_ratio():
    dm = bandpass_delta_mag("C", "V", "LSST_r")
    ints = get_integrals("C", np.asarray(["V", "LSST_r"], dtype=object))
    expected = float(-2.5 * np.log10(float(ints[1]) / float(ints[0])))
    assert dm == pytest.approx(expected, abs=1e-12)


def test_bandpass_color_terms_includes_expected_key():
    terms = bandpass_color_terms(
        "S", source_filter_id="V", target_filter_ids=["LSST_r"]
    )
    assert "LSST_r" in terms
