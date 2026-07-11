from migration.scripts.audit_public_surface import DEFAULT_OUTPUT, collect


def test_public_surface_manifest_is_complete_and_current() -> None:
    expected = collect()
    assert not expected["parse_errors"]
    assert not expected["duplicate_ids"]
    assert DEFAULT_OUTPUT.exists()

    import json

    committed = json.loads(DEFAULT_OUTPUT.read_text())
    assert committed == expected


def test_every_non_plotting_public_symbol_has_an_explicit_review_slot() -> None:
    manifest = collect()
    for symbol in manifest["symbols"]:
        if symbol["plotting_exemption_candidate"]:
            continue
        assert "review_status" in symbol, symbol["id"]
        assert "implementation_class" in symbol, symbol["id"]
        assert "parity_coverage" in symbol, symbol["id"]
        assert "native_timing" in symbol, symbol["id"]
        assert symbol["tracking_issue"], symbol["id"]
