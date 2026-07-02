# Third-party data attribution — rotation-period validation fixtures

The `rotation_period_validation_fixture_*.npz` files in this directory are a small,
curated, derived subset of published asteroid photometry used solely to test the
rotation-period estimator. The full source catalogues are **not** vendored here; each
fixture embeds its own provenance (`source_title`, `source_url`, `session_submitter`,
`session_bibcode`).

## Period ground truth — Asteroid Lightcurve Database (LCDB)
Reference rotation periods and quality (`U`) codes are from the LCDB.
- Warner, B. D., Harris, A. W., & Pravec, P. (2009). *The asteroid lightcurve database.*
  Icarus, 202, 134–146 (updated v4.0).
- PDS Small Bodies Node: DOI **10.26033/j3xc-3359**.
- Free to use with attribution. Only a handful of objects' summary rows are used.

## Photometry — Asteroid Lightcurve Data Exchange Format (ALCDEF)
The per-observation magnitudes derive from ALCDEF-contributed lightcurves.
- PDS dataset `gbo.ast.alcdef-database_V1_0`, DOI **10.26033/b8cw-s522**.
- Public domain; the project requests acknowledgment of NASA grant **80NSSC18K0851**.
- Contributing observers are credited per fixture in `session_submitter` (and where
  available `session_bibcode`), and include B. A. Skiff, Brian D. Warner, F. Pilcher,
  J. Garlitz, M. S. Alkema, R. A. Koff, Robert D. Stephens, and T. A. Polakis.

## Observing geometry
Per-observation `r_au`, `delta_au`, and `phase_angle_deg` were computed from
**JPL Horizons** ephemerides at fixture-build time and frozen into the `.npz`.

---
These fixtures are provided for testing only and do not redistribute the LCDB, ALCDEF,
or any other catalogue in whole. See the repository `LICENSE.md` for the adam_core
license, which does not extend to the third-party data credited above.
