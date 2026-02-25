# testdata/neocc

Sample OEF files from ESA NEOCC (per-object orbit download). Each file is the response for one object and one epoch type.

## Getting more samples

Per-object OEF URL (designator without spaces, orbit type, epoch 0=middle/1=present-day):

- `https://neo.ssa.esa.int/PSDB-portlet/download?file=DESIG.ke0`  (Keplerian, middle of arc)
- `https://neo.ssa.esa.int/PSDB-portlet/download?file=DESIG.ke1`  (Keplerian, present-day)

Examples: `433.ke1`, `2024YR4.ke1`, `99942.ke1` (Apophis). Not all objects are in NEOCC; use NEOs from their risk list or catalog.

## Object IDs that exercise different OEF content

| Object   | Use case |
|----------|----------|
| 2024YR4  | Recent NEO; MAG line present (H, G); 21-element COV. |
| 2022OB5  | NEO; standard OEF with MAG. |
| 433      | Eros; numbered NEO. |
| 99942, 101955 | Apophis, Bennu; PHA. Some NEOCC OEF use 28-element COV (not 21 upper-tri); current parser skips those. |

To refresh or add samples, run from repo root:

    python src/adam_core/orbits/query/tests/testdata/fetch_real_payloads.py
