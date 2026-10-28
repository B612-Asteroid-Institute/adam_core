"""Accepted d21 numerical constant pins for the Rust configuration."""

from ._focused import explicit_config


def test_core_config_pins_accepted_d21_two_body_constants() -> None:
    two_body = explicit_config().for_iod().candidate_correction_fit.two_body
    assert two_body.lt_tol == 1.0e-12
    assert two_body.ephemeris_max_iter == 1000
    assert two_body.max_lt_iter == 10
