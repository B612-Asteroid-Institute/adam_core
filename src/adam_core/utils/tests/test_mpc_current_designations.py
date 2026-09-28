"""Standards-complete MPC designation and ADES identity regressions."""

import pytest

from ..mpc import (
    ADESDesignationParts,
    pack_comet_designation,
    pack_mpc_designation,
    pack_numbered_comet_designation,
    pack_numbered_designation,
    pack_permanent_satellite_designation,
    pack_provisional_comet_designation,
    pack_provisional_designation,
    pack_provisional_satellite_designation,
    pack_satellite_designation,
    pack_survey_designation,
    parse_ades_designation,
    unpack_comet_designation,
    unpack_mpc_designation,
    unpack_numbered_comet_designation,
    unpack_numbered_designation,
    unpack_permanent_satellite_designation,
    unpack_provisional_comet_designation,
    unpack_provisional_designation,
    unpack_provisional_satellite_designation,
    unpack_satellite_designation,
    unpack_survey_designation,
)

NUMBERED_MINOR_PLANETS = {
    "1": "00001",
    "99999": "99999",
    "100000": "A0000",
    "359999": "Z9999",
    "360000": "a0000",
    "619999": "z9999",
    "620000": "~0000",
    "15396335": "~zzzz",
}

PROVISIONAL_MINOR_PLANETS = {
    "A801 AA": "I01A00A",
    "A908 CJ": "J08C00J",
    "A924 YA": "J24Y00A",
    "1925 AA": "J25A00A",
    "1995 XA": "J95X00A",
    "2025 OY619": "K25Oz9Y",
    "2000 AA620": "_0A0000",
    "2015 BX634": "_FB0060",
    "2015 BZ631": "_FB004p",
    "2024 AB631": "_OA004S",
    "2025 OY625": "_PO002O",
    "2025 OT677": "_PO00NH",
    "2028 EA339749": "_SEZZZZ",
    "2029 FL591673": "_TFzzzz",
    "2061 AZ620": "_zA000O",
}

SURVEYS = {
    "2040 P-L": "PLS2040",
    "3138 T-1": "T1S3138",
    "1010 T-2": "T2S1010",
    "4101 T-3": "T3S4101",
}

NUMBERED_COMETS = {
    "1P": "0001P",
    "1I": "0001I",
    "73P-A": "0073Pa",
    "73P-AA": "0073Paa",
    "354P": "0354P",
    "1D": "0001D",
}

PROVISIONAL_COMETS = {
    "C/1995 O1": "CJ95O010",
    "D/1993 F2-B": "DJ93F02b",
    "A/2017 U1": "AK17U010",
    "I/2017 U1": "IK17U010",
    "P/2010 TO20": "PK10T20O",
    "1P/1986 F1": "0001PJ86F010",
    "29P/1993 F1": "0029PJ93F010",
    "C/240 V1": "C240V010",
    "C/837 F1": "C837F010",
    "C/-43 K1": "C/56K010",
    "C/-146 P1": "C.53P010",
    "C/-240 V1": "C-59V010",
    # MPC documents no packed policy for two-letter provisional fragments.
    "P/1930 J1-AA": "P/1930 J1-AA",
}

PROVISIONAL_SATELLITES = {
    "S/1877 M 1": "SI77M010",
    "S/2003 J 2": "SK03J020",
    "S/2019 S 22": "SK19S220",
    "S/2018 U 1": "SK18U010",
    "S/2020 N 1": "SK20N010",
    "S/2005 P 1": "SK05P010",
}

PERMANENT_SATELLITES = {
    "Mars I": "M001S",
    "Jupiter XIII": "J013S",
    "Saturn X": "S010S",
    "Uranus I": "U001S",
    "Neptune XIV": "N014S",
    "Pluto I": "P001S",
}


@pytest.mark.parametrize("unpacked,packed", NUMBERED_MINOR_PLANETS.items())
def test_numbered_minor_planet_round_trip(unpacked, packed):
    assert pack_numbered_designation(unpacked) == packed
    assert unpack_numbered_designation(packed) == unpacked


@pytest.mark.parametrize("unpacked,packed", PROVISIONAL_MINOR_PLANETS.items())
def test_provisional_minor_planet_round_trip(unpacked, packed):
    assert pack_provisional_designation(unpacked) == packed
    assert unpack_provisional_designation(packed) == unpacked


@pytest.mark.parametrize("unpacked,packed", SURVEYS.items())
def test_survey_round_trip(unpacked, packed):
    assert pack_survey_designation(unpacked) == packed
    assert unpack_survey_designation(packed) == unpacked


@pytest.mark.parametrize("unpacked,packed", NUMBERED_COMETS.items())
def test_numbered_comet_round_trip(unpacked, packed):
    assert pack_numbered_comet_designation(unpacked) == packed
    assert unpack_numbered_comet_designation(packed) == unpacked
    assert pack_comet_designation(unpacked) == packed
    assert unpack_comet_designation(packed) == unpacked


@pytest.mark.parametrize("unpacked,packed", PROVISIONAL_COMETS.items())
def test_provisional_comet_round_trip(unpacked, packed):
    assert pack_provisional_comet_designation(unpacked) == packed
    assert unpack_provisional_comet_designation(packed) == unpacked
    assert pack_comet_designation(unpacked) == packed
    assert unpack_comet_designation(packed) == unpacked


@pytest.mark.parametrize("unpacked,packed", PROVISIONAL_SATELLITES.items())
def test_provisional_satellite_round_trip(unpacked, packed):
    assert pack_provisional_satellite_designation(unpacked) == packed
    assert unpack_provisional_satellite_designation(packed) == unpacked
    assert pack_satellite_designation(unpacked) == packed
    assert unpack_satellite_designation(packed) == unpacked


@pytest.mark.parametrize("unpacked,packed", PERMANENT_SATELLITES.items())
def test_permanent_satellite_round_trip(unpacked, packed):
    assert pack_permanent_satellite_designation(unpacked) == packed
    assert unpack_permanent_satellite_designation(packed) == unpacked
    assert pack_satellite_designation(unpacked) == packed
    assert unpack_satellite_designation(packed) == unpacked


@pytest.mark.parametrize(
    "unpacked,packed",
    list(NUMBERED_MINOR_PLANETS.items())
    + list(PROVISIONAL_MINOR_PLANETS.items())
    + list(SURVEYS.items())
    + list(NUMBERED_COMETS.items())
    + list(PROVISIONAL_COMETS.items())
    + list(PROVISIONAL_SATELLITES.items())
    + list(PERMANENT_SATELLITES.items()),
)
def test_generic_dispatch_round_trip(unpacked, packed):
    assert pack_mpc_designation(unpacked) == packed
    assert unpack_mpc_designation(packed) == unpacked


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "0001",
        "15396336",
        "1893 AP",
        "1908 CJ",
        "1995 XA0",
        "A908 CJ0",
        "A925 AA",
        "2025 IA620",
        "2025 AI620",
        "1999 AA620",
        "2062 AA620",
        "2025 AA591674",
        "0000 P-L",
        "10000 T-1",
        "C/1995 I1",
        "C/1995 O0",
        "C/1995 O1-I",
        "C/1995 O1-ABC",
        "1P/1995 O1-AA",
        "P/2010 TO0",
        "C/-300 A1",
        "0P",
        "10000P",
        "73P-I",
        "S/2019 S 0",
        "S/2019 E 1",
        "Jupiter 13",
        "Jupiter IIII",
    ],
)
def test_packers_reject_noncanonical_or_out_of_range_values(value):
    with pytest.raises(ValueError):
        pack_mpc_designation(value)


@pytest.mark.parametrize(
    "value",
    [
        "00000",
        "~zzzzx",
        "_!A0000",
        "_zI0000",
        "PLS0000",
        "0000P",
        "0073Pi",
        "CJ95I010",
        "CJ95O000",
        "C000A010",
        "SK19S000",
        "J000S",
        "JIIII",
    ],
)
def test_unpackers_reject_noncanonical_or_out_of_range_values(value):
    with pytest.raises(ValueError):
        unpack_mpc_designation(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("241001", ADESDesignationParts(perm_id="241001")),
        ("2015 BZ631", ADESDesignationParts(prov_id="2015 BZ631")),
        ("A904 OA", ADESDesignationParts(prov_id="A904 OA")),
        ("2040 P-L", ADESDesignationParts(prov_id="2040 P-L")),
        ("(2010 AB1)", ADESDesignationParts(prov_id="2010 AB1")),
        (
            "17032 Edlu (1999 FM9)",
            ADESDesignationParts(perm_id="17032", prov_id="1999 FM9"),
        ),
        ("1036 Ganymed", ADESDesignationParts(perm_id="1036")),
        ("1P/Halley", ADESDesignationParts(perm_id="1P")),
        ("1I/ʻOumuamua", ADESDesignationParts(perm_id="1I")),
        ("73P-BB/Schwassmann-Wachmann", ADESDesignationParts(perm_id="73P-BB")),
        ("C/2013 A1", ADESDesignationParts(prov_id="C/2013 A1")),
        (
            "1P/1986 F1",
            ADESDesignationParts(perm_id="1P", prov_id="P/1986 F1"),
        ),
        ("S/2000 J 1", ADESDesignationParts(prov_id="S/2000 J 1")),
        ("S/1877 M 1", ADESDesignationParts(prov_id="S/1877 M 1")),
        ("S/2000 (65803) 1", ADESDesignationParts(prov_id="S/2000 (65803) 1")),
        (
            "S/2000 (1998 WW31) 1",
            ADESDesignationParts(prov_id="S/2000 (1998 WW31) 1"),
        ),
        ("Jupiter XIII", ADESDesignationParts(perm_id="Jupiter XIII")),
        ("Pluto I", ADESDesignationParts(perm_id="Pluto I")),
        ("(134340) I", ADESDesignationParts(perm_id="(134340) I")),
        ("ZTF-123_", ADESDesignationParts(trk_sub="ZTF-123_")),
        ("Apophis", ADESDesignationParts(trk_sub="Apophis")),
    ],
)
def test_parse_ades_designation_official_and_legacy_forms(value, expected):
    assert parse_ades_designation(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        " 2015 BZ631",
        "2015 BZ631 ",
        "2015  BZ631",
        "1991 V",
        "2015 BX634 junk",
        "2015  BX634",
        "2015 bx634",
        "12 T-4",
        "0040 P-L",
        "00433",
        "03202",
        "2015 BZ0631",
        "99999999",
        "C/2013 A1-ABC",
        "S/2000 E 1",
        "Jupiter 13",
        "(65803) 1",
        "(134340) IIII",
        "MIRA_2025",
        "tracking-id-too-long",
        "name with spaces",
        "1036  Ganymed",
        "1036 Ganymed (1999 FM9",
        "1036 Ganymed (not a designation)",
        "A0345",
        "_FB004p",
        "K24A00A",
        "PLS2040",
        "0001P",
        "CJ95O010",
        "C240V010",
        "C/56K010",
        "0001PJ86F010",
        "SK19S220",
        "J013S",
        "P001S",
    ],
)
def test_parse_ades_designation_rejects_malformed_ambiguous_and_packed(value):
    with pytest.raises(ValueError):
        parse_ades_designation(value)


@pytest.mark.parametrize(
    "pack_name,unpack_name,cases",
    [
        (
            "pack_numbered_designation",
            "unpack_numbered_designation",
            NUMBERED_MINOR_PLANETS,
        ),
        (
            "pack_provisional_designation",
            "unpack_provisional_designation",
            PROVISIONAL_MINOR_PLANETS,
        ),
        ("pack_survey_designation", "unpack_survey_designation", SURVEYS),
        (
            "pack_numbered_comet_designation",
            "unpack_numbered_comet_designation",
            NUMBERED_COMETS,
        ),
        (
            "pack_provisional_comet_designation",
            "unpack_provisional_comet_designation",
            PROVISIONAL_COMETS,
        ),
        (
            "pack_provisional_satellite_designation",
            "unpack_provisional_satellite_designation",
            PROVISIONAL_SATELLITES,
        ),
        (
            "pack_permanent_satellite_designation",
            "unpack_permanent_satellite_designation",
            PERMANENT_SATELLITES,
        ),
    ],
)
def test_python_wrappers_match_direct_rust_surface(pack_name, unpack_name, cases):
    from adam_core import _rust_native
    from adam_core.utils import mpc

    python_pack = getattr(mpc, pack_name)
    python_unpack = getattr(mpc, unpack_name)
    rust_pack = getattr(_rust_native, pack_name)
    rust_unpack = getattr(_rust_native, unpack_name)
    for unpacked, packed in cases.items():
        assert python_pack(unpacked) == rust_pack(unpacked) == packed
        assert python_unpack(packed) == rust_unpack(packed) == unpacked


def test_python_ades_parser_matches_direct_rust_surface():
    from adam_core import _rust_native

    for value in (
        "241001",
        "2015 BZ631",
        "1P/Halley",
        "C/2013 A1",
        "S/2000 J 1",
        "Jupiter XIII",
        "ZTF-123_",
    ):
        parsed = parse_ades_designation(value)
        assert (parsed.perm_id, parsed.prov_id, parsed.trk_sub) == (
            _rust_native.parse_ades_designation(value)
        )


_INVALID_UNPACKABLE_PROVISIONALS = (
    "1999 AA620",
    "2062 AA620",
    "2025 AA591674",
)
_INVALID_UNPACKABLE_ADES_FORMS = tuple(
    value
    for designation in _INVALID_UNPACKABLE_PROVISIONALS
    for value in (
        designation,
        f"({designation})",
        f"17032 Edlu ({designation})",
        f"S/2000 ({designation}) 1",
    )
)


@pytest.mark.parametrize("value", _INVALID_UNPACKABLE_ADES_FORMS)
def test_ades_parser_rejects_unpacked_but_unencodable_provisionals_in_every_context(
    value,
):
    from adam_core import _rust_native

    with pytest.raises(ValueError):
        parse_ades_designation(value)
    with pytest.raises(ValueError):
        _rust_native.parse_ades_designation(value)


@pytest.mark.parametrize("designation", ["2061 AZ620", "2029 FL591673"])
def test_ades_parser_accepts_packable_extended_boundaries_in_every_context(designation):
    from adam_core import _rust_native

    values = [
        (designation, ADESDesignationParts(prov_id=designation)),
        (f"({designation})", ADESDesignationParts(prov_id=designation)),
        (
            f"17032 Edlu ({designation})",
            ADESDesignationParts(perm_id="17032", prov_id=designation),
        ),
        (
            f"S/2000 ({designation}) 1",
            ADESDesignationParts(prov_id=f"S/2000 ({designation}) 1"),
        ),
    ]
    for value, expected in values:
        parsed = parse_ades_designation(value)
        assert parsed == expected
        assert _rust_native.parse_ades_designation(value) == (
            expected.perm_id,
            expected.prov_id,
            expected.trk_sub,
        )


@pytest.mark.parametrize(
    "designation",
    ["S/2019 S 620", "S/2019 S 999999", "S/2019 S 0619"],
)
def test_ades_parser_rejects_unencodable_planetary_satellite_numbers(designation):
    from adam_core import _rust_native

    with pytest.raises(ValueError):
        parse_ades_designation(designation)
    with pytest.raises(ValueError):
        _rust_native.parse_ades_designation(designation)


@pytest.mark.parametrize(
    ("designation", "expected"),
    [
        ("S/2019 S 619", ADESDesignationParts(prov_id="S/2019 S 619")),
        (
            "S/2019 (134340) 620",
            ADESDesignationParts(prov_id="S/2019 (134340) 620"),
        ),
    ],
)
def test_ades_parser_preserves_valid_satellite_number_boundaries(designation, expected):
    from adam_core import _rust_native

    assert parse_ades_designation(designation) == expected
    assert _rust_native.parse_ades_designation(designation) == (
        expected.perm_id,
        expected.prov_id,
        expected.trk_sub,
    )
