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
    ]
    for code, band in required:
        key = f"{code}|{band}"
        assert bool(pc.any(pc.equal(mapping.key, key)).as_py())


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


def test_bandpass_curves_are_sane():
    curves = load_bandpass_curves()
    assert len(curves) > 0

    filter_ids = set(curves.filter_id.to_pylist())
    assert {"Bessell_U", "Bessell_B", "Bessell_R", "Bessell_I"}.issubset(filter_ids)
    assert {"SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"}.issubset(filter_ids)
    assert {"PS1_g", "PS1_r", "PS1_i", "PS1_z", "PS1_y"}.issubset(filter_ids)
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
