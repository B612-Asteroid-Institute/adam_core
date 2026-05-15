# RM-STANDALONE-004 Time-Scale Strategy Spike (2026-05-15)

Status: strategy + fixture spike for the standalone Rust time model.

Related artifacts:

- Fixture: `migration/artifacts/time_scale_rescale_fixture_2026-05-15.json`
- Generator: `migration/scripts/generate_time_scale_fixture.py`
- Data-model RFC: `migration/rust_native_data_model_rfc_2026-05-15.md`
- Roadmap: `migration/standalone_rust_surface_roadmap_2026-05-15.md`

## Baseline contract

Current Python `Timestamp` behavior is the oracle for the first standalone Rust time work.
The implementation is not a pure Astropy clone:

- UTC ↔ TAI leap-second lookup uses ERFA (`erfa.utctai`, `erfa.taiutc`).
- TAI ↔ TT is the exact `32.184 s` offset.
- TT ↔ TDB uses the project-local sinusoidal approximation in `Timestamp._tt_tdb_correction`, not Astropy's location-dependent TDB model.
- Any conversion involving UT1 delegates to Astropy/IERS because ERFA requires caller-supplied UT1-UTC data.
- TDB → ET seconds is pure arithmetic: `(MJD_TDB - 51544.5) * 86400`.

That means the first Rust implementation must preserve the current project contract, not silently switch to a different SOFA/Astropy TDB convention.

## Fixture scope

The checked-in fixture covers the supported non-UT1 conversion matrix:

```text
utc, tai, tt, tdb  ×  utc, tai, tt, tdb
```

Rows are generated from UTC instants that include:

- representative pre-J2000, J2000, and current-era timestamps;
- every post-1999 leap-second insertion day (`2005-12-31`, `2008-12-31`, `2012-06-30`, `2015-06-30`, `2016-12-31`);
- the second before each insertion, the leap second itself (`23:59:60`), and the first second after it.

The fixture also stores TDB days/nanos plus ET seconds for the same physical instants. Rust unit tests should treat those ET values as the pure-arithmetic identity fixture.

UT1 is intentionally not part of the first fixed fixture because it depends on IERS table selection and interpolation. A Rust UT1 implementation needs a separate explicit IERS data-provider contract.

## Strategy decision

Use an ERFA C FFI path first, but only for the behavior ERFA currently owns in Python:

1. Bind the ERFA UTC/TAI conversion routines needed to reproduce leap-second behavior (`eraUtctai`, `eraTaiutc`).
2. Implement TAI/TT as the existing exact constant offset in Rust.
3. Port the existing project-local TT/TDB approximation literally and compare against the fixture.
4. Keep TDB→ET as pure Rust arithmetic over `TimeArray`.
5. Reject UT1 in the first Rust service unless an explicit IERS provider is supplied.

Prefer ERFA/liberfa over raw SOFA for the first FFI because Python already depends on PyERFA/Astropy behavior and ERFA has the redistribution posture expected by PyERFA. Do not link against the Python wheel's private shared objects; either use a Rust crate with an audited ERFA build or vendor/build the needed ERFA C sources through Cargo with license text preserved.

## Rust-native replacement evaluation path

A Rust-native candidate can replace the ERFA FFI only if it matches the checked-in fixture exactly for integer days/nanos outputs, or if a future science-policy change deliberately updates the fixture. Candidate evaluation must include:

- all fixture rows across leap seconds;
- UTC↔TAI round trips at leap-second boundaries;
- TAI↔TT exactness;
- TT↔TDB parity with the current project approximation;
- TDB→ET arithmetic parity;
- explicit UT1 behavior, either rejected or backed by a fixture-pinned IERS data source.

Libraries such as `hifitime` may still be useful for parsing or internal time representations, but they are not acceptable as a semantic replacement unless they pass this matrix.

## Immediate implementation posture

The RM-STANDALONE-003 `TimeArray` stays as the canonical Rust batch type. RM-STANDALONE-004 adds the fixture and starts with a TDB-only Rust arithmetic helper; the ERFA service should be added as a small module inside `adam_core_rs_coords::types` first. Split to an `adam_core_rs_time` crate only after the API is stable and reused by SPICE, propagation, and OD workflows.

Unsupported conversions must fail loudly. Python can remain the adapter for full `Timestamp.rescale` until the Rust service passes the fixture.
