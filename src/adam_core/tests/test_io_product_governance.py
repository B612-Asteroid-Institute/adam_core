"""Non-registry governance for migrated IO/query/product surfaces
(bead personal-cmy.37.4.10).

These gates intentionally live outside the 44-API parity registry. They pin:

- the recorded external-service fixture inventory that makes query products
  deterministic offline oracles;
- the opt-in environment flags for live HTTP integration gates;
- the atomic file-product/byte-fixture test modules for ADES/OEM/OpenSpace/SPK;
- the Rust-owned deterministic timing entrypoints for every migrated
  query/serialization/product family, and that they emit positive samples.
"""

from pathlib import Path

import numpy as np

from adam_core import _rust_native

SRC = Path(__file__).parents[1]

RECORDED_QUERY_FIXTURES = [
    "orbits/query/tests/data/horizons/vectors_bennu_20240101.txt",
    "orbits/query/tests/data/horizons/elements_bennu_20240101.txt",
    "orbits/query/tests/data/horizons/ephemerides_bennu_20240101.txt",
    "orbits/query/tests/testdata/neocc/2024YR4.ke0",
    "orbits/query/tests/testdata/neocc/2024YR4.ke1",
    "orbits/query/tests/testdata/neocc/2022OB5.ke1",
    "orbits/query/tests/testdata/scout/summary.json",
    "orbits/query/tests/testdata/scout/orbits.json",
    "orbits/query/tests/testdata/sbdb/Ceres.json",
    "orbits/query/tests/testdata/sbdb/2001VB.json",
    "orbits/query/tests/testdata/sbdb/54509.json",
    "orbits/query/tests/testdata/sbdb/missing.json",
    "photometry/tests/data/vendor/svo_bessell_v.xml",
    "photometry/tests/data/vendor/solar_spec.fits",
]

LIVE_GATES = [
    ("orbits/query/tests/test_horizons.py", "ADAM_CORE_LIVE_HORIZONS"),
    (
        "orbits/query/tests/test_query_clients_live.py",
        "ADAM_CORE_LIVE_QUERY_CLIENTS",
    ),
]

PRODUCT_ROUND_TRIP_MODULES = [
    "observations/tests/test_ades_rust_parity.py",
    "orbits/tests/test_oem.py",
    "orbits/tests/test_oem_rust_parity.py",
    "orbits/openspace/tests/test_openspace_rust_parity.py",
    "orbits/tests/test_spice_kernel.py",
]

NATIVE_TIMING_ENTRYPOINTS = [
    # Query clients
    "benchmark_horizons_response_processing",
    "benchmark_query_client_processing",
    # ADES
    "benchmark_ades_string_to_tables_fused",
    "benchmark_ades_to_string_fused_ipc",
    # OEM
    "benchmark_oem_read_orbits",
    "benchmark_oem_write_orbits_kvn",
    # OpenSpace products
    "benchmark_openspace_write_sbdb_csv",
    "benchmark_openspace_create_orbital_kepler_product",
    "benchmark_openspace_create_trail_orbit_product",
    # SPK products
    "benchmark_spk_write_orbits_product",
    # MPC packed dates
    "benchmark_unpack_mpc_dates_isot",
    # Bandpass vendor products
    "benchmark_bandpass_vendor_products",
]


def test_recorded_query_fixture_inventory_is_present():
    missing = [
        fixture for fixture in RECORDED_QUERY_FIXTURES if not (SRC / fixture).is_file()
    ]
    assert missing == [], f"missing recorded service fixtures: {missing}"


def test_live_http_gates_are_opt_in_via_documented_env_vars():
    for module, env_var in LIVE_GATES:
        source = (SRC / module).read_text()
        assert env_var in source, f"{module} must gate live HTTP on {env_var}"
        assert "skipif" in source, f"{module} live tests must be opt-in"


def test_file_product_round_trip_modules_exist():
    missing = [
        module for module in PRODUCT_ROUND_TRIP_MODULES if not (SRC / module).is_file()
    ]
    assert missing == [], f"missing product round-trip gates: {missing}"


def test_migrated_io_surfaces_expose_rust_owned_timing():
    missing = [
        name for name in NATIVE_TIMING_ENTRYPOINTS if not hasattr(_rust_native, name)
    ]
    assert missing == [], f"missing Rust timing entrypoints: {missing}"


def test_query_timing_entrypoints_emit_positive_samples():
    horizons = np.asarray(
        _rust_native.benchmark_horizons_response_processing(
            "vectors",
            [
                (
                    SRC / "orbits/query/tests/data/horizons/vectors_bennu_20240101.txt"
                ).read_text()
            ],
            2,
            2,
            1,
        )
    )
    neocc = np.asarray(
        _rust_native.benchmark_query_client_processing(
            "neocc",
            [(SRC / "orbits/query/tests/testdata/neocc/2024YR4.ke1").read_text()],
            2,
            2,
            1,
        )
    )
    for samples in (horizons, neocc):
        assert samples.shape == (2, 2)
        assert np.all(samples > 0.0)
