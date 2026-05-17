# RM-STANDALONE-004 Time-Scale Strategy Spike (2026-05-15)

Status: strategy + fixture spike, first ERFA-backed Rust implementation for UTC↔TAI, and RM-STANDALONE-004B parity saturation against the existing Python time-rescale test matrix.

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

## RM-STANDALONE-004B parity saturation requirement

Before `TimeArray` is treated as sufficient for standalone propagation or used to replace Python `Timestamp.rescale`, the Rust implementation must pass the existing Python rescale test contract. That means porting or mirroring:

- `test_Timestamp_rescale_roundtrip`, including the no-`86400e9` nanos regression case;
- `test_time_scale_fixture_matches_current_timestamp_contract`;
- the full `test_Timestamp_rescale_correctness` scale-pair matrix over `tai`, `utc`, `tdb`, `tt`, and `ut1`.

Current Rust covers the non-UT1 fixture-backed `utc`/`tai`/`tt`/`tdb` contract. UT1 remains adapter-owned through Astropy/IERS until a Rust provider contract exists. RM-STANDALONE-004B makes that boundary explicit in tests: the fixture includes the full Python `tai`/`utc`/`tdb`/`tt`/`ut1` matrix, Rust verifies it through an explicit `TimeScaleProvider` boundary, and provider-less UT1/GPS continue to fail loudly.

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

## RM-STANDALONE-004A implementation

The first ERFA-backed Rust service lives in `adam_core_rs_coords::types::time` and uses the `erfars` crate, which builds/bundles ERFA C sources through Cargo rather than linking against PyERFA's private wheel artifacts.

Implemented Rust behavior:

- `TimeArray::utc_to_tai_erfa()` and `TimeArray::tai_to_utc_erfa()` call ERFA UTC/TAI routines and require the matching source scale.
- `TimeArray::rescale()` now supports the fixture matrix (`utc`, `tai`, `tt`, `tdb`) using ERFA for UTC↔TAI, the exact `32.184 s` TAI↔TT offset, and a literal port of the existing project-local TT↔TDB approximation.
- `TimeArray::tdb_et_seconds()` remains pure arithmetic.
- UT1 and GPS rescaling fail loudly until an explicit provider contract exists.

The fixture-backed Rust test asserts exact day/nanosecond parity for all UTC/TAI/TT/TDB cases in `time_scale_rescale_fixture_2026-05-15.json`.
