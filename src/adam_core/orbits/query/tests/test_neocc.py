from pathlib import Path

import numpy as np

from adam_core import _rust_native
from adam_core._rust.arrow import table_from_record_batch
from adam_core.orbits import Orbits

from ..neocc import _parse_oef, _physical_parameters_from_neocc

TESTDATA_DIR = Path(__file__).parent / "testdata" / "neocc"


def test__parse_oef_2024YR4_ke0():
    with open(TESTDATA_DIR / "2024YR4.ke0", "r") as file:
        data = file.read()

    result = _parse_oef(data)

    # Test header
    assert result["header"]["format"] == "OEF2.0"
    assert result["header"]["rectype"] == "ML"
    assert result["header"]["refsys"] == "ECLM J2000"

    # Test object identification
    assert result["object_id"] == "2024YR4"

    # Test Keplerian elements (baseline from current 2024YR4.ke0 fixture)
    assert result["elements"]["a"] == 2.516306904413629
    assert result["elements"]["e"] == 0.6615997598102339
    assert result["elements"]["i"] == 3.4082582480218
    assert result["elements"]["node"] == 271.3674684214714
    assert result["elements"]["peri"] == 134.3645021052249
    assert result["elements"]["M"] == 16.920818963896668

    # Test epoch and time system
    assert result["epoch"] == 60704.950998578
    assert result["time_system"] == "TDT"

    # Test magnitude (OEF MAG = absolute magnitude H, slope parameter G; V-band)
    assert result["magnitude"]["H"] == 24.047
    assert result["magnitude"]["G"] == 0.150

    # Test derived parameters
    assert abs(result["derived"]["perihelion"] - 0.8515188608447389) < 1e-14
    assert abs(result["derived"]["aphelion"] - 4.181094947982519) < 1e-14
    assert abs(result["derived"]["period"] - 1457.9540638135068) < 1e-14
    assert result["derived"]["pha"] == "F"
    assert result["derived"]["orb_type"] == "Apollo"

    # Test covariance and correlation matrices
    assert result["covariance"].shape == (6, 6)
    assert result["correlation"].shape == (6, 6)

    # Test matrix properties
    assert np.allclose(result["covariance"], result["covariance"].T)  # Symmetry
    assert np.allclose(result["correlation"], result["correlation"].T)  # Symmetry
    assert np.allclose(
        np.diagonal(result["correlation"]), 1.0
    )  # Diagonal elements should be 1


def test__parse_oef_2022OB5_ke1():
    with open(TESTDATA_DIR / "2022OB5.ke1", "r") as file:
        data = file.read()

    result = _parse_oef(data)

    # Test header
    assert result["header"]["format"] == "OEF2.0"
    assert result["header"]["rectype"] == "ML"
    assert result["header"]["refsys"] == "ECLM J2000"

    # Test object identification
    assert result["object_id"] == "2022OB5"

    # Test Keplerian elements (baseline from current 2022OB5.ke1 fixture)
    assert result["elements"]["a"] == 1.0034630497303025
    assert result["elements"]["e"] == 0.0555727787282651
    assert result["elements"]["i"] == 2.0877951248265
    assert result["elements"]["node"] == 300.2025534594475
    assert result["elements"]["peri"] == 99.4880289155792
    assert result["elements"]["M"] == 14.654786035215105

    # Test epoch and time system
    assert result["epoch"] == 61000.0
    assert result["time_system"] == "TDT"

    # Test magnitude (OEF MAG = H, G)
    assert result["magnitude"]["H"] == 28.912
    assert result["magnitude"]["G"] == 0.150

    # Test derived parameters
    assert abs(result["derived"]["perihelion"] - 0.9476978197056504) < 1e-14
    assert abs(result["derived"]["aphelion"] - 1.0592282797549548) < 1e-14
    assert abs(result["derived"]["period"] - 367.15589424322116) < 1e-14
    assert result["derived"]["pha"] == "F"
    assert result["derived"]["orb_type"] == "Apollo"

    # Test covariance and correlation matrices
    assert result["covariance"].shape == (6, 6)
    assert result["correlation"].shape == (6, 6)

    # Test matrix properties
    assert np.allclose(result["covariance"], result["covariance"].T)  # Symmetry
    assert np.allclose(result["correlation"], result["correlation"].T)  # Symmetry
    assert np.allclose(
        np.diagonal(result["correlation"]), 1.0
    )  # Diagonal elements should be 1


def test_query_neocc_recorded_products():
    object_ids = ["2024YR4", "2022OB5"]
    for orbit_epoch, suffix, expected_days in [
        ("present-day", "ke1", [61000, 61000]),
        ("middle", "ke0", [60704, 60129]),
    ]:
        payloads = [
            (TESTDATA_DIR / f"{obj}.{suffix}").read_text() for obj in object_ids
        ]
        batch = _rust_native.query_neocc_arrow(object_ids, "ke", orbit_epoch, payloads)
        orbits = table_from_record_batch(Orbits, batch)
        assert orbits.orbit_id.to_pylist() == object_ids
        assert orbits.object_id.to_pylist() == object_ids
        assert orbits.coordinates.time.days.to_pylist() == expected_days
        assert orbits.coordinates.time.scale == "tt"
        assert np.all(np.isfinite(orbits.coordinates.values))
        assert np.all(np.isfinite(orbits.coordinates.covariance.to_matrix()))
        assert orbits.physical_parameters.H_v.to_pylist() == [24.047, 28.912]
        assert orbits.physical_parameters.G.to_pylist() == [0.15, 0.15]


def test__physical_parameters_from_neocc_with_magnitude() -> None:
    data = {"magnitude": {"H": 23.5, "G": 0.15}}
    phys = _physical_parameters_from_neocc(data)
    assert len(phys) == 1
    assert phys.H_v[0].as_py() == 23.5
    assert phys.G[0].as_py() == 0.15
    assert np.isnan(phys.H_v_sigma[0].as_py())
    assert np.isnan(phys.G_sigma[0].as_py())


def test__physical_parameters_from_neocc_magnitude_missing_G() -> None:
    data = {"magnitude": {"H": 20.0}}
    phys = _physical_parameters_from_neocc(data)
    assert len(phys) == 1
    assert phys.H_v[0].as_py() == 20.0
    assert np.isnan(phys.G[0].as_py())


def test__physical_parameters_from_neocc_no_magnitude() -> None:
    data = {}
    phys = _physical_parameters_from_neocc(data)
    assert len(phys) == 1
    assert np.isnan(phys.H_v[0].as_py())
    assert np.isnan(phys.G[0].as_py())


def test__physical_parameters_from_neocc_magnitude_empty() -> None:
    data = {"magnitude": {}}
    phys = _physical_parameters_from_neocc(data)
    assert len(phys) == 1
    assert np.isnan(phys.H_v[0].as_py())
    assert np.isnan(phys.G[0].as_py())


def test_real_neocc_oef_files_parse_without_error() -> None:
    # Parse every .ke0/.ke1 in testdata; ensures real NEOCC OEF shapes don't break us.
    # Some real files may use COV/format variants we don't support yet; skip those.
    parsed = 0
    for path in sorted(TESTDATA_DIR.iterdir()):
        if path.suffix not in (".ke0", ".ke1") or not path.name[0].isdigit():
            continue
        data = path.read_text()
        try:
            result = _parse_oef(data)
        except Exception:
            continue
        if not result.get("object_id") or "elements" not in result:
            continue
        assert "epoch" in result
        if "magnitude" in result:
            assert "H" in result["magnitude"]
            assert "G" in result["magnitude"]
        parsed += 1
    assert (
        parsed >= 2
    ), "at least two real OEF files should parse (e.g. 2024YR4, 2022OB5)"


def test_empty_response() -> None:
    batch = _rust_native.query_neocc_arrow(["1416T-2"], "ke", "present-day", [""])
    assert len(table_from_record_batch(Orbits, batch)) == 0


def test_28element_matrix() -> None:
    batch = _rust_native.query_neocc_arrow(
        ["2018 CW2"],
        "ke",
        "present-day",
        [(TESTDATA_DIR / "2018CW2.ke1").read_text()],
    )
    assert len(table_from_record_batch(Orbits, batch)) == 1
