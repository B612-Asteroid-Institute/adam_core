"""Fail if rendered ``adam_core::od`` exposes an undocumented ordinary member.

The umbrella module inlines public items re-exported from domain crates, so
source-only checks of ``rust/adam_core/src/od.rs`` are insufficient. This guard
parses the rendered Rustdoc tree and checks item docs, inherent methods,
associated constants, struct fields, enum variants, and enum-variant fields
while excluding trait and blanket implementations.

The pinned count is a rendered-coverage sentinel (pages plus inherent members,
fields, variants, and variant fields), not an API-size or stability promise.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

EXPECTED_MEMBER_COUNT = 378


def rustdoc_coverage(root: Path) -> tuple[int, list[str]]:
    """Return the checked-member count and missing-doc descriptions."""

    missing: list[str] = []
    total = 0
    for page in sorted(root.rglob("*.html")):
        if page.name in {"index.html", "sidebar-items.js"}:
            continue
        text = page.read_text(encoding="utf-8")
        item = str(page.relative_to(root.parent))
        total += 1
        before_sections = text.split('<h2 id="', 1)[0]
        if (
            'class="toggle top-doc"' not in text
            and 'class="docblock"' not in before_sections
        ):
            missing.append(f"{item}: <item doc>")

        inherent = ""
        match = re.search(
            r'<div id="implementations-list">(.*?)(<h2 id="'
            r"(trait-implementations|synthetic-implementations|blanket-implementations)|$)",
            text,
            re.S,
        )
        if match:
            inherent = match.group(1)
        for member in re.finditer(
            r'(<details class="toggle method-toggle" open><summary>)?'
            r'<section id="((?:method|associatedconstant)\.[^"]+)"',
            inherent,
        ):
            total += 1
            if not member.group(1):
                missing.append(f"{item}: {member.group(2)}")
        for field in re.finditer(
            r'<span id="(structfield\.[^"]+)" class="structfield[^"]*">.*?</span>'
            r'(<div class="docblock">)?',
            text,
            re.S,
        ):
            total += 1
            if not field.group(2):
                missing.append(f"{item}: {field.group(1)}")
        for variant in re.finditer(
            r'<section id="(variant\.[^"]+)" class="variant">.*?</section>'
            r'(<div class="docblock">)?',
            text,
            re.S,
        ):
            total += 1
            if not variant.group(2):
                missing.append(f"{item}: {variant.group(1)}")
        for variant_field in re.finditer(
            r'<span id="(variant\.[^"]+\.field\.[^"]+)"[^>]*>.*?</span>'
            r'(<div class="docblock">)?',
            text,
            re.S,
        ):
            total += 1
            if not variant_field.group(2):
                missing.append(f"{item}: {variant_field.group(1)}")
    return total, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-total", type=int, default=EXPECTED_MEMBER_COUNT)
    args = parser.parse_args()
    if not args.root.is_dir():
        raise SystemExit(
            f"rendered adam_core::od Rustdoc tree does not exist: {args.root}"
        )
    total, missing = rustdoc_coverage(args.root)
    for entry in missing:
        print(f"MISSING {entry}")
    print(f"checked={total} missing={len(missing)}")
    if total != args.expected_total:
        raise SystemExit(
            f"rendered adam_core::od member count changed: "
            f"expected {args.expected_total}, got {total}"
        )
    if missing:
        raise SystemExit(
            f"rendered adam_core::od has {len(missing)} undocumented members"
        )


if __name__ == "__main__":
    main()
