from decimal import Decimal

import pytest

from adam_core.observations.obs80 import (
    NANOSECONDS_PER_DAY,
    Obs80ParseError,
    parse_optical_obs80,
    parse_optical_obs80_file,
)


def test_parse_standard_scout_neocp_optical_record() -> None:
    parsed = parse_optical_obs80(
        "     A11EpSe*0C2026 07 08.17725719 41 24.185-30 19 19.42         19.35oVNEOCPW68"
    )

    assert parsed.designation == "A11EpSe"
    assert parsed.discovery is True
    assert parsed.note1 == "0"
    assert parsed.note2 == "C"
    assert parsed.observatory_code == "W68"
    assert parsed.obstime_days_utc == 61229
    assert parsed.obstime_nanos_utc == int(
        (Decimal("0.177257") * Decimal(NANOSECONDS_PER_DAY)).to_integral_value()
    )
    assert parsed.ra_deg == pytest.approx(295.3507708333333)
    assert parsed.dec_deg == pytest.approx(-30.32206111111111)
    assert parsed.mag == pytest.approx(19.35)
    assert parsed.band == "o"
    assert parsed.astrometric_catalog == "V"
    assert parsed.reference == "NEOCP"
    assert parsed.obstime_mjd_utc == pytest.approx(61229.177257)


def test_parse_record_with_space_padded_precision() -> None:
    parsed = parse_optical_obs80(
        "     A11EpSe KC2026 07 14.53636 19 37 22.30 -29 16 44.5          19.0 GVNEOCPE23"
    )

    assert parsed.designation == "A11EpSe"
    assert parsed.note1 == "K"
    assert parsed.note2 == "C"
    assert parsed.observatory_code == "E23"
    assert parsed.mag == pytest.approx(19.0)


def test_file_parser_omits_unsupported_two_line_rows() -> None:
    optical = "     ST26G06  C2026 07 08.33794520 25 33.638-00 47 36.82         19.62GVNEOCPU68"
    satellite = optical[:14] + "S" + optical[15:]

    parsed = parse_optical_obs80_file(f"{optical}\n{satellite}\n")

    assert len(parsed) == 1
    assert parsed[0].designation == "ST26G06"
    with pytest.raises(Obs80ParseError, match="two-line"):
        parse_optical_obs80(satellite)


def test_parser_rejects_malformed_rows() -> None:
    optical = "     ST26G06  C2026 07 08.33794520 25 33.638-00 47 36.82         19.62GVNEOCPU68"
    with pytest.raises(Obs80ParseError, match="shorter"):
        parse_optical_obs80("too short")
    with pytest.raises(Obs80ParseError, match="observation date"):
        parse_optical_obs80(optical[:15] + "not a valid date!" + optical[32:])
