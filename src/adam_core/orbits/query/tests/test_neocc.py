from pathlib import Path

import numpy as np

from ..neocc import _parse_oef, _upper_triangular_to_full, query_neocc

TESTDATA_DIR = Path(__file__).parent / "testdata" / "neocc"


def test__upper_triangular_to_full():
    # Test that we can reconstruct a full covariance matrix from an upper triangular matrix.
    expected_array = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 2, 3, 4, 5, 6],
            [3, 3, 3, 4, 5, 6],
            [4, 4, 4, 4, 5, 6],
            [5, 5, 5, 5, 5, 6],
            [6, 6, 6, 6, 6, 6],
        ],
        dtype=np.float64,
    )

    triangular_array = np.triu(expected_array)[np.triu_indices(6)].flatten()

    actual_array = _upper_triangular_to_full(triangular_array)
    np.testing.assert_array_equal(actual_array, expected_array)


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

    # Test Keplerian elements
    assert result["elements"]["a"] == 2.5164332860625849
    assert result["elements"]["e"] == 0.66161845913486128
    assert result["elements"]["i"] == 3.4083351351535
    assert result["elements"]["node"] == 271.3684549331765
    assert result["elements"]["peri"] == 134.3639170929742
    assert result["elements"]["M"] == 12.947786309204405

    # Test epoch and time system
    assert result["epoch"] == 60688.865642464
    assert result["time_system"] == "TDT"

    # Test magnitude
    assert result["magnitude"]["value"] == 23.876
    assert result["magnitude"]["uncertainty"] == 0.150

    # Test derived parameters
    assert abs(result["derived"]["perihelion"] - 0.85151457282218190) < 1e-14
    assert abs(result["derived"]["aphelion"] - 4.1813519993029882) < 1e-14
    assert abs(result["derived"]["period"] - 1458.0639039239943) < 1e-14
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

    # Test Keplerian elements
    assert result["elements"]["a"] == 1.0088199810054714
    assert result["elements"]["e"] == 0.058593341916141808
    assert result["elements"]["i"] == 2.0599781391032
    assert result["elements"]["node"] == 302.6856493105631
    assert result["elements"]["peri"] == 105.7080782884594
    assert result["elements"]["M"] == 334.06138612130985

    # Test epoch and time system
    assert result["epoch"] == 60600.0
    assert result["time_system"] == "TDT"

    # Test magnitude
    assert result["magnitude"]["value"] == 28.813
    assert result["magnitude"]["uncertainty"] == 0.150

    # Test derived parameters
    assert abs(result["derived"]["perihelion"] - 0.94970984692658189) < 1e-14
    assert abs(result["derived"]["aphelion"] - 1.0679301150843603) < 1e-14
    assert abs(result["derived"]["period"] - 370.09987635676049) < 1e-14
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
    assert orbits.coordinates.time.days[0].as_py() == 60800
    assert orbits.coordinates.time.days[1].as_py() == 60600
    assert orbits.coordinates.time.scale == "tt"
    assert np.all(~np.isnan(orbits.coordinates.values))
    assert np.all(~np.isnan(orbits.coordinates.covariance.to_matrix()))

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
    assert orbits.coordinates.time.days[0].as_py() == 60688
    assert orbits.coordinates.time.days[1].as_py() == 59792
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
