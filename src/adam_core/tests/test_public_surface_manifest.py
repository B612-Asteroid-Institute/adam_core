from migration.scripts.audit_public_surface import DEFAULT_OUTPUT, collect


def test_public_surface_manifest_is_complete_and_current() -> None:
    expected = collect()
    assert not expected["parse_errors"]
    assert not expected["duplicate_ids"]
    assert DEFAULT_OUTPUT.exists()

    import json

    committed = json.loads(DEFAULT_OUTPUT.read_text())
    assert committed == expected


def test_frozen_updated_upstream_retirements_are_classified() -> None:
    reconciliation = collect()["frozen_updated_upstream_reconciliation"]
    assert reconciliation["commit"] == "757c09fca86adf9e3d5899952db3d379e09413f6"
    assert reconciliation["upstream_symbols"] == 600
    assert reconciliation["upstream_only_symbols"] == 22
    assert reconciliation["all_upstream_only_symbols_classified"] is True
    for retirement in reconciliation["retirements"]:
        assert retirement["classification"].startswith("retired_")
        audit_document = DEFAULT_OUTPUT.parents[2] / retirement["audit_document"]
        assert audit_document.exists(), retirement["id"]


def test_every_non_plotting_public_symbol_is_classified_and_tracked() -> None:
    manifest = collect()
    assert manifest["summary"]["unreviewed"] == 0
    for symbol in manifest["symbols"]:
        if symbol["plotting_exemption_candidate"]:
            assert symbol["implementation_class"] == "plotting_exemption_candidate"
            continue
        assert symbol["review_status"] == "domain_audited", symbol["id"]
        assert symbol["implementation_class"] != "unreviewed", symbol["id"]
        assert symbol["parity_coverage"], symbol["id"]
        assert symbol["native_timing"], symbol["id"]
        assert symbol["tracking_issue"] != "personal-cmy.37", symbol["id"]
        audit_document = DEFAULT_OUTPUT.parents[2] / symbol["audit_document"]
        assert audit_document.exists(), symbol["id"]
