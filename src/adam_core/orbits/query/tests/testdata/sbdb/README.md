# testdata/sbdb

Sample JSON responses from the JPL SBDB lookup API. Existing files were fetched without `phys-par=1`, so they do not contain `phys_par` (physical parameters H, G).

## Getting payloads with physical parameters

Include `phys-par=1` to exercise H/G parsing:

- `https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=OBJECT_ID&cov=mat&full-prec=true&phys-par=1`

Save with a descriptive name, e.g. `433_Eros_phys.json`.

## Object IDs that exercise different responses

| Object ID   | Use case |
|-------------|----------|
| 433, Eros   | Numbered NEA; typically has H, G (and often sigmas) in `phys_par`. |
| 2001VB      | Unnumbered NEO; orbit + optional phys_par. |
| Ceres, 1    | Main-belt; good for covariance and phys_par. |
| 1P          | Comet; `phys_par` has M1/K1/M2/K2 (comet magnitudes), not H/G. |
| (any)       | Omit `phys-par` or use unknown ID to get no phys_par / 404-style. |

To refresh or add samples, run from repo root:

    python src/adam_core/orbits/query/tests/testdata/fetch_real_payloads.py

## Original fixture URLs (no phys-par)

- https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=2001VB&cov=mat&full-prec=true
- https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=443104&cov=mat&full-prec=true
- https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=54509&cov=mat&full-prec=true
- https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=missingno&cov=mat&full-prec=true
