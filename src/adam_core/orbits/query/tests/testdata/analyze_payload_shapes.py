"""
Analyze shapes of fetched SBDB and NEOCC payloads (no network).

Run from repo root after fetch_real_payloads.py:

    python src/adam_core/orbits/query/tests/testdata/analyze_payload_shapes.py

Prints a summary of top-level keys, phys_par name variety, and parse outcomes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
_SBDB_DIR = _BASE / "sbdb"
_NEOCC_DIR = _BASE / "neocc"


def _analyze_sbdb() -> None:
    print("=== SBDB payload shapes ===\n")
    files = sorted(_SBDB_DIR.glob("*.json"))
    if not files:
        print("  (no JSON files)")
        return

    all_top_keys: set[str] = set()
    kind_counts: dict[str, int] = {}
    phys_par_names: list[list[str]] = []
    has_H = 0
    has_H_mag = 0
    has_G = 0
    has_H_no_G = 0
    orbit_keys: set[str] = set()
    element_counts: list[int] = []

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  {path.name}: load failed ({e})")
            continue
        all_top_keys.update(data.keys())
        if "object" not in data:
            continue
        obj = data.get("object") or {}
        kind = obj.get("kind", "?")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if "orbit" in data:
            orbit_keys.update(data["orbit"].keys())
            el = data["orbit"].get("elements")
            element_counts.append(len(el) if isinstance(el, list) else 0)
        phys = data.get("phys_par") or []
        names = []
        has_this_H = False
        has_this_G = False
        for p in phys:
            if isinstance(p, dict) and "name" in p:
                names.append(str(p["name"]))
                if p.get("name") == "H":
                    has_H += 1
                    has_this_H = True
                if p.get("name") == "H_mag":
                    has_H_mag += 1
                    has_this_H = True
                if p.get("name") == "G":
                    has_G += 1
                    has_this_G = True
        if has_this_H and not has_this_G:
            has_H_no_G += 1
        phys_par_names.append(names)

    print("Top-level keys:", sorted(all_top_keys))
    print("Object kinds seen:", dict(sorted(kind_counts.items())))
    print("Orbit keys (union):", sorted(orbit_keys))
    print("Elements count per payload:", min(element_counts) if element_counts else 0, "..", max(element_counts) if element_counts else 0)
    print("phys_par: payloads with H:", has_H, "H_mag:", has_H_mag, "G:", has_G, "H but no G:", has_H_no_G)
    all_phys_names: set[str] = set()
    for names in phys_par_names:
        all_phys_names.update(names)
    print("All phys_par entry names:", sorted(all_phys_names))
    print()
    print("Per-file phys_par names (first 20):")
    for path in files[:20]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        phys = data.get("phys_par") or []
        names = [p.get("name") for p in phys if isinstance(p, dict)]
        obj = data.get("object") or {}
        kind = obj.get("kind", "?")
        full = obj.get("fullname", path.stem)
        print(f"  {path.name}: kind={kind} phys_par names={names[:12]}{'...' if len(names) > 12 else ''}")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more files")
    print()


def _analyze_neocc() -> None:
    print("=== NEOCC OEF parse outcomes ===\n")
    # Import here so we can run script without full package if needed
    try:
        from adam_core.orbits.query.neocc import _parse_oef
    except ImportError:
        print("  (cannot import _parse_oef; run from repo with env)")
        return

    files = []
    for ext in (".ke0", ".ke1"):
        files.extend(sorted(_NEOCC_DIR.glob(f"*{ext}")))
    files = [f for f in files if f.name[0].isdigit() or f.name.startswith("20")]
    if not files:
        print("  (no .ke0/.ke1 files)")
        return

    ok = 0
    fail = 0
    errors: dict[str, str] = {}
    fail_cov_count: dict[str, int] = {}
    has_mag = 0
    cov_lens: list[int] = []

    for path in sorted(files):
        data = path.read_text(encoding="utf-8")
        try:
            result = _parse_oef(data)
        except Exception as e:
            fail += 1
            errors[path.name] = str(e)[:80]
            # Count COV elements in raw file for failed parses (we expect 21 = 6x6 upper tri)
            n = sum(
                len(line.split()) - 1
                for line in data.splitlines()
                if line.strip().startswith("COV")
            )
            if n:
                fail_cov_count[path.name] = n
            continue
        ok += 1
        if result.get("magnitude"):
            has_mag += 1
        cov = result.get("covariance")
        if cov is not None and hasattr(cov, "shape"):
            cov_lens.append(cov.size)
        elif "covariance" in result:
            cov_lens.append(-1)

    print("Parsed OK:", ok, "Failed:", fail)
    print("With magnitude (H,G):", has_mag)
    if cov_lens:
        print("Covariance sizes (element count):", min(cov_lens), "..", max(cov_lens))
    if fail_cov_count:
        print("Failed files COV element count (parser expects 21):", dict(sorted(fail_cov_count.items())))
    if errors:
        print("\nParse errors by file:")
        for name, err in sorted(errors.items()):
            print(f"  {name}: {err}")
    print()


def main() -> None:
    _analyze_sbdb()
    _analyze_neocc()
    print("=== Summary ===")
    print("SBDB: Use H or H_mag for absolute magnitude; G often missing for unnumbered NEOs.")
    print("SBDB: Comets (cn/cu) have M1/K1/M2/K2/PC, not H/G.")
    print("NEOCC: Parser expects 21-element upper-triangular COV; some objects use 28+ (different format).")


if __name__ == "__main__":
    main()
    sys.exit(0)
