# Recorded JPL Horizons responses

Captured 2026-07-12 from `https://ssd.jpl.nasa.gov/api/horizons.api` via the pinned legacy `astroquery.jplhorizons.Horizons` protocol for object `101955` (Bennu), epoch JD 2460310.5 / MJD 60310, with:

- `vectors_bennu_20240101.txt`: vectors, `@sun`, ecliptic, geometric, AU/day.
- `elements_bennu_20240101.txt`: elements, `@sun`, J2000 ecliptic, absolute periapsis time.
- `ephemerides_bennu_20240101.txt`: observer ephemerides, MPC `500`, extra precision.

These verbatim service responses gate deterministic Rust parsing and typed Arrow assembly without network access. Live service behavior remains covered by the opt-in Horizons integration test.
