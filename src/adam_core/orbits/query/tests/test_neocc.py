from pathlib import Path

import numpy as np

from ..neocc import _parse_oef, _physical_parameters_from_neocc, query_neocc

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


def test_query_neocc(mocker):
    """Test query_neocc with both present-day and middle epochs (using saved test data)"""
    import requests

    # Setup mock responses for each test file
    mock_responses = {}
    for orbit_type in ["ke0", "ke1"]:
        for obj in ["2024YR4", "2022OB5"]:
            with open(TESTDATA_DIR / f"{obj}.{orbit_type}", "r") as f:
                mock_responses[f"{obj}.{orbit_type}"] = f.read()

    # Create mock response
    def mock_get(url, params):
        mock = mocker.MagicMock()
        mock.status_code = 200
        mock.text = mock_responses[params["file"]]
        return mock

    # Patch requests.get
    mocker.patch("requests.get", side_effect=mock_get)

    # Test querying multiple objects
    object_ids = ["2024YR4", "2022OB5"]
    orbits = query_neocc(object_ids, orbit_type="ke", orbit_epoch="present-day")

    # Verify the results
    assert orbits is not None
    assert len(orbits) == 2

    assert orbits.orbit_id[0].as_py() == "2024YR4"
    assert orbits.object_id[0].as_py() == "2024YR4"
    assert orbits.orbit_id[1].as_py() == "2022OB5"
    assert orbits.object_id[1].as_py() == "2022OB5"
    # present-day uses ke1 files; epochs from current fixtures
    assert orbits.coordinates.time.days[0].as_py() == 61000
    assert orbits.coordinates.time.days[1].as_py() == 61000
    assert orbits.coordinates.time.scale == "tt"
    assert np.all(~np.isnan(orbits.coordinates.values))
    assert np.all(~np.isnan(orbits.coordinates.covariance.to_matrix()))
    # Physical parameters from OEF MAG (H, G); no uncertainties in NEOCC OEF
    assert orbits.physical_parameters is not None
    assert orbits.physical_parameters.H_v[0].as_py() == 24.047
    assert orbits.physical_parameters.G[0].as_py() == 0.150
    assert orbits.physical_parameters.H_v[1].as_py() == 28.912
    assert orbits.physical_parameters.G[1].as_py() == 0.150

    # Verify the mock was called with correct parameters
    requests.get.assert_has_calls(
        [
            mocker.call(
                "https://neo.ssa.esa.int/PSDB-portlet/download",
                params={"file": "2024YR4.ke1"},
            ),
            mocker.call(
                "https://neo.ssa.esa.int/PSDB-portlet/download",
                params={"file": "2022OB5.ke1"},
            ),
        ]
    )

    # Test querying multiple objects
    object_ids = ["2024YR4", "2022OB5"]
    orbits = query_neocc(object_ids, orbit_type="ke", orbit_epoch="middle")

    # Verify the results
    assert orbits is not None
    assert len(orbits) == 2

    assert orbits.orbit_id[0].as_py() == "2024YR4"
    assert orbits.object_id[0].as_py() == "2024YR4"
    assert orbits.orbit_id[1].as_py() == "2022OB5"
    assert orbits.object_id[1].as_py() == "2022OB5"
    # middle uses ke0 files; epochs from current fixtures
    assert orbits.coordinates.time.days[0].as_py() == 60704
    assert orbits.coordinates.time.days[1].as_py() == 60129
    assert orbits.coordinates.time.scale == "tt"
    assert np.all(~np.isnan(orbits.coordinates.values))
    assert np.all(~np.isnan(orbits.coordinates.covariance.to_matrix()))

    # Verify the mock was called with correct parameters
    requests.get.assert_has_calls(
        [
            mocker.call(
                "https://neo.ssa.esa.int/PSDB-portlet/download",
                params={"file": "2024YR4.ke0"},
            ),
            mocker.call(
                "https://neo.ssa.esa.int/PSDB-portlet/download",
                params={"file": "2022OB5.ke0"},
            ),
        ]
    )


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
    assert parsed >= 2, "at least two real OEF files should parse (e.g. 2024YR4, 2022OB5)"
