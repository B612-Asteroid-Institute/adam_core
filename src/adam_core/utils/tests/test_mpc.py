import numpy as np
import pytest

from ..mpc import (
    ADESDesignationParts,
    convert_mpc_packed_dates,
    pack_mpc_designation,
    pack_numbered_designation,
    pack_provisional_designation,
    pack_survey_designation,
    parse_ades_designation,
    unpack_mpc_designation,
    unpack_numbered_designation,
    unpack_provisional_designation,
    unpack_survey_designation,
)

# --- Tests last updated: 22-03-2023
# Packed : Unpacked
# Taken from: https://minorplanetcenter.net/iau/info/PackedDes.html
# Provisional Minor Planet Designations
PROVISIONAL_DESIGNATIONS_UP2P = {
    "1995 XA": "J95X00A",
    "1995 XL1": "J95X01L",
    "1995 FB13": "J95F13B",
    "1998 SQ108": "J98SA8Q",
    "1998 SV127": "J98SC7V",
    "1998 SS162": "J98SG2S",
    "2099 AZ193": "K99AJ3Z",
    "2008 AA360": "K08Aa0A",
    "2007 TA418": "K07Tf8A",
}
EXTENDED_PROVISIONAL_DESIGNATIONS_UP2P = {
    # MPC extended packed provisional specification examples and retained
    # production identities.
    "2024 AB631": "_OA004S",
    "2015 BX634": "_FB0060",
    "2015 BZ631": "_FB004p",
    "2025 OY625": "_PO002O",
    "2025 OT677": "_PO00NH",
    "2025 DA620": "_PD0000",
    "2026 DY620": "_QD000N",
    "2027 DZ6190": "_RD0aEM",
    "2028 EA339749": "_SEZZZZ",
    "2029 FL591673": "_TFzzzz",
}
# Provisional Minor Planet Designations (Surveys)
SURVEY_DESIGNATIONS_UP2P = {
    "2040 P-L": "PLS2040",
    "3138 T-1": "T1S3138",
    "1010 T-2": "T2S1010",
    "4101 T-3": "T3S4101",
}
# Permanent Minor Planet Designations
NUMBERED_DESIGNATION_UP2P = {
    "3202": "03202",
    "50000": "50000",
    "100345": "A0345",
    "360017": "a0017",
    "203289": "K3289",
    "620000": "~0000",
    "620061": "~000z",
    "3140113": "~AZaz",
    "15396335": "~zzzz",
}
DESIGNATIONS_UP2P = {
    **PROVISIONAL_DESIGNATIONS_UP2P,
    **EXTENDED_PROVISIONAL_DESIGNATIONS_UP2P,
    **SURVEY_DESIGNATIONS_UP2P,
    **NUMBERED_DESIGNATION_UP2P,
}


def test_convert_mpc_packed_dates():
    Time = pytest.importorskip("astropy.time").Time
    # Use a few modified examples from https://minorplanetcenter.net/iau/info/PackedDates.html
    # and test conversion from packed form to MJDs
    isot_tt = np.array(
        [
            "1996-01-01",
            "1996-01-10",
            "1996-09-30",
            "1996-10-01",
            "2001-10-22",
            "2001-10-22T00:00:00.0000",
            "2001-10-22T12:00:00.0000",
            "1996-09-30T18:00:00.0000",
            "1996-09-30T18:45:00.0000",
        ]
    )

    pf_tt = np.array(
        [
            "J9611",
            "J961A",
            "J969U",
            "J96A1",
            "K01AM",
            "K01AM",
            "K01AM5",
            "J969U75",
            "J969U78125",
        ]
    )

    mjd_actual = convert_mpc_packed_dates(pf_tt)
    mjd_desired = Time(isot_tt, format="isot", scale="tt")

    np.testing.assert_equal(mjd_actual.tt.mjd, mjd_desired.tt.mjd)

    from adam_core import _rust_native

    samples = _rust_native.benchmark_unpack_mpc_dates_isot(pf_tt.tolist(), 2, 2, 1)
    assert all(sample > 0.0 for trial in samples for sample in trial)


def test_unpack_numbered_designation():
    # Test unpacking of numbered designations
    for designation, designation_pf in NUMBERED_DESIGNATION_UP2P.items():
        assert unpack_numbered_designation(designation_pf) == designation


def test_unpack_numbered_designation_raises():
    # Test invalid unpacking of provisional designations as numbered designations
    for designation_pf in PROVISIONAL_DESIGNATIONS_UP2P:
        with pytest.raises(ValueError):
            unpack_numbered_designation(designation_pf)

    # Test invalid unpacking of survey designations as numbered designations
    for designation_pf in SURVEY_DESIGNATIONS_UP2P:
        with pytest.raises(ValueError):
            unpack_numbered_designation(designation_pf)


def test_unpack_provisional_designation():
    # Test unpacking of provisional designations
    for designation, designation_pf in PROVISIONAL_DESIGNATIONS_UP2P.items():
        assert unpack_provisional_designation(designation_pf) == designation


def test_unpack_provisional_designation_raises():
    # Test invalid unpacking of numbered designations as provisional designations
    for designation_pf in NUMBERED_DESIGNATION_UP2P:
        with pytest.raises(ValueError):
            unpack_provisional_designation(designation_pf)

    # Test invalid unpacking of survey designations as provisional designations
    for designation_pf in SURVEY_DESIGNATIONS_UP2P:
        with pytest.raises(ValueError):
            unpack_provisional_designation(designation_pf)


def test_unpack_survey_designation():
    # Test unpacking of survey designations
    for designation, designation_pf in SURVEY_DESIGNATIONS_UP2P.items():
        assert unpack_survey_designation(designation_pf) == designation


def test_unpack_survey_designation_raises():
    # Test invalid unpacking of numbered designations as survey designations
    for designation_pf in NUMBERED_DESIGNATION_UP2P:
        with pytest.raises(ValueError):
            unpack_survey_designation(designation_pf)

    # Test invalid unpacking of provisional designations as survey designations
    for designation_pf in PROVISIONAL_DESIGNATIONS_UP2P:
        with pytest.raises(ValueError):
            unpack_survey_designation(designation_pf)


def test_pack_numbered_designation():
    # Test packing of numbered designations
    for designation, designation_pf in NUMBERED_DESIGNATION_UP2P.items():
        assert pack_numbered_designation(designation) == designation_pf


def test_pack_numbered_designation_raises():
    # Test invalid packing of provisional designations as numbered designations
    for designation in PROVISIONAL_DESIGNATIONS_UP2P:
        with pytest.raises(ValueError):
            pack_numbered_designation(designation)

    # Test invalid packing of survey designations as numbered designations
    for designation in SURVEY_DESIGNATIONS_UP2P:
        with pytest.raises(ValueError):
            pack_numbered_designation(designation)


def test_pack_provisional_designation():
    # Test packing of provisional designations
    for designation, designation_pf in PROVISIONAL_DESIGNATIONS_UP2P.items():
        assert pack_provisional_designation(designation) == designation_pf


def test_pack_provisional_designation_raises():
    # Test invalid packing of numbered designations as provisional designations
    for designation in NUMBERED_DESIGNATION_UP2P:
        with pytest.raises(ValueError):
            pack_provisional_designation(designation)

    # Test invalid packing of survey designations as provisional designations
    for designation in SURVEY_DESIGNATIONS_UP2P:
        with pytest.raises(ValueError):
            pack_provisional_designation(designation)


def test_pack_survey_designation():
    # Test packing of survey designations
    for designation, designation_pf in SURVEY_DESIGNATIONS_UP2P.items():
        assert pack_survey_designation(designation) == designation_pf


def test_pack_survey_designation_raises():
    # Test invalid packing of numbered designations as survey designations
    for designation in NUMBERED_DESIGNATION_UP2P:
        with pytest.raises(ValueError):
            pack_survey_designation(designation)

    # Test invalid packing of provisional designations as survey designations
    for designation in PROVISIONAL_DESIGNATIONS_UP2P:
        with pytest.raises(ValueError):
            pack_survey_designation(designation)


@pytest.mark.parametrize(
    ("unpacked", "packed"),
    list(EXTENDED_PROVISIONAL_DESIGNATIONS_UP2P.items())
    + [("2025 OY619", "K25Oz9Y"), ("2025 OA620", "_PO0000")],
)
def test_extended_provisional_designation_round_trip(unpacked, packed):
    assert pack_provisional_designation(unpacked) == packed
    assert unpack_provisional_designation(packed) == unpacked
    assert pack_mpc_designation(unpacked) == packed
    assert unpack_mpc_designation(packed) == unpacked


@pytest.mark.parametrize(
    "value",
    [
        "2015 IA620",
        "2015 ZA620",
        "2015 BZ0620",
        "1999 BA620",
        "2062 BA620",
        "2025 BA591674",
        "_FB006!",
        "_?B0060",
        "_FZ0060",
    ],
)
def test_provisional_designation_rejects_malformed_or_out_of_range_values(value):
    function = (
        unpack_provisional_designation
        if value.startswith("_")
        else pack_provisional_designation
    )
    with pytest.raises(ValueError):
        function(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2015 BZ631", ADESDesignationParts(prov_id="2015 BZ631")),
        ("2040 P-L", ADESDesignationParts(prov_id="2040 P-L")),
        ("1991 V", ADESDesignationParts(perm_id="1991")),
        ("A904 OA", ADESDesignationParts(prov_id="A904 OA")),
        ("C/2013 A1", ADESDesignationParts(prov_id="C/2013 A1")),
        ("I/2017 U1", ADESDesignationParts(prov_id="I/2017 U1")),
        ("S/2000 J 1", ADESDesignationParts(prov_id="S/2000 J 1")),
        (
            "17032 Edlu (1999 FM9)",
            ADESDesignationParts(perm_id="17032", prov_id="1999 FM9"),
        ),
        ("1P/Halley", ADESDesignationParts(perm_id="1P")),
        ("1I/ʻOumuamua", ADESDesignationParts(perm_id="1I")),
        ("73P-BB/Schwassmann-Wachmann", ADESDesignationParts(perm_id="73P-BB")),
        ("Jupiter XIII", ADESDesignationParts(perm_id="Jupiter XIII")),
        ("(134340) I", ADESDesignationParts(perm_id="(134340) I")),
        ("ZTF-123_", ADESDesignationParts(trk_sub="ZTF-123_")),
    ],
)
def test_parse_ades_designation_official_forms(value, expected):
    assert parse_ades_designation(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "2015 BZ0631",
        "99999999",
        "C/2013 A1-ABC",
        "S/2000 E 1",
        "Jupiter 13",
        "(65803) 1",
        "(134340) IIII",
        "tracking-id-too-long",
        "name with spaces",
    ],
)
def test_parse_ades_designation_rejects_malformed_or_unsupported_text(value):
    with pytest.raises(ValueError):
        parse_ades_designation(value)


def test_unpack_mpc_designation():
    # Test unpacking of packed form designations
    for designation_pf, designation in DESIGNATIONS_UP2P.items():
        assert unpack_mpc_designation(designation) == designation_pf


def test_pack_mpc_designation():
    # Test packing of unpacked designations
    for designation, designation_pf in DESIGNATIONS_UP2P.items():
        assert pack_mpc_designation(designation) == designation_pf
