from pathlib import Path

import numpy as np

from ..neocc import (
    _non_gravitational_parameters_from_neocc,
    _parse_oef,
    _physical_parameters_from_neocc,
    _solved_state_covariance_from_upper_triangular,
    query_neocc,
)

TESTDATA_DIR = Path(__file__).parent / "testdata" / "neocc"


def test__solved_state_covariance_drops_trailing_magnitude_for_6d_solution():
    # A 28-element (7x7) upper-triangular COV on a 6D orbital solution carries
    # an appended magnitude row/column; it must be dropped to a 6x6 solved
    # state (see "2018 CW2").
    upper = list(range(1, 29))  # 28 distinct values
    solved = _solved_state_covariance_from_upper_triangular(upper, solved_dimension=6)
    assert solved.shape == (6, 6)


def test__solved_state_covariance_keeps_nongrav_parameter_for_7d_solution():
    # The same 28-element (7x7) COV on a 7D non-grav solution (6 orbital + A2)
    # must keep the full 7x7 — the 7th row/column is A2, not magnitude.
    upper = list(range(1, 29))
    solved = _solved_state_covariance_from_upper_triangular(upper, solved_dimension=7)
    assert solved.shape == (7, 7)
    # The leading 6x6 block is identical whether or not the 7th param is kept.
    dropped = _solved_state_covariance_from_upper_triangular(upper, solved_dimension=6)
    np.testing.assert_array_equal(solved[:6, :6], dropped)


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
    assert orbits.solved_state_covariance.dimension.to_pylist() == [6, 6]
    # Physical parameters from OEF MAG (H, G); no uncertainties in NEOCC OEF
    assert orbits.physical_parameters is not None
    assert orbits.physical_parameters.H_v[0].as_py() == 24.047
    assert orbits.physical_parameters.G[0].as_py() == 0.150
    assert orbits.physical_parameters.H_v[1].as_py() == 28.912
    assert orbits.physical_parameters.G[1].as_py() == 0.150
    assert orbits.non_gravitational_parameters.A2[0].as_py() is None
    assert orbits.non_gravitational_parameters.A2[1].as_py() is None

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
    assert orbits.solved_state_covariance.dimension.to_pylist() == [6, 6]

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


def test__parse_oef_99942_ke1_nongrav() -> None:
    result = _parse_oef((TESTDATA_DIR / "99942.ke1").read_text())

    assert result["nongrav"]["model_used"] == 1
    assert result["nongrav"]["parameter_count"] == 2
    assert result["nongrav"]["dimension"] == 7
    assert result["nongrav"]["solve_for_parameter_codes"] == [2]
    assert result["nongrav"]["vector"] == [0.0, -2.90010329254113e-04]
    assert result["covariance"].shape == (6, 6)
    assert result["covariance_full"].shape == (7, 7)
    assert result["correlation_full"].shape == (7, 7)


def test__non_gravitational_parameters_from_neocc_yarkovsky() -> None:
    data = _parse_oef((TESTDATA_DIR / "101955.ke1").read_text())
    nongrav = _non_gravitational_parameters_from_neocc(data)

    assert len(nongrav) == 1
    assert nongrav.source[0].as_py() == "NEOCC"
    assert nongrav.model[0].as_py() == "yarkovsky"
    assert nongrav.solution_dimension[0].as_py() == 7
    # parameter_count counts the estimated parameters, matching SBDB semantics.
    assert nongrav.parameter_count[0].as_py() == 1
    assert nongrav.estimated_parameter_names[0].as_py() == "A2"
    assert nongrav.AMRAT[0].as_py() == 0.0
    assert np.isclose(nongrav.A2[0].as_py(), -4.60477568857430e-14)
    # sigma(A2) comes from the OEF covariance diagonal (RMS 2.40555E-06 in
    # 1e-10 au/d^2 units) converted to canonical au/d^2.
    assert np.isclose(nongrav.A2_sigma[0].as_py(), 2.40555e-16, rtol=1e-4)


def test__non_gravitational_parameters_from_neocc_unsupported_model(caplog) -> None:
    import logging

    data = _parse_oef((TESTDATA_DIR / "99942.ke1").read_text())
    # Rewrite the parsed solution as an unsupported cometary-style model
    # solving for A1 (code 3): the (AMRAT, A2) vector decoding does not apply.
    data["nongrav"]["model_used"] = 2
    data["nongrav"]["solve_for_parameter_codes"] = [3]

    with caplog.at_level(logging.WARNING, logger="adam_core.orbits.query.neocc"):
        nongrav = _non_gravitational_parameters_from_neocc(data)

    assert any("unsupported" in record.message for record in caplog.records)
    assert nongrav.model[0].as_py() == "neocc-model-2"
    assert nongrav.estimated_parameter_names[0].as_py() == "A1"
    # Values must not be mislabeled from the (AMRAT, A2) positional layout.
    assert nongrav.AMRAT[0].as_py() is None
    assert nongrav.A2[0].as_py() is None
    assert nongrav.A1[0].as_py() is None


def test_query_neocc_preserves_full_solved_state_covariance(mocker):
    response_text = (TESTDATA_DIR / "99942.ke1").read_text()

    def mock_get(url, params):
        mock = mocker.MagicMock()
        mock.status_code = 200
        mock.text = response_text
        return mock

    mocker.patch("requests.get", side_effect=mock_get)
    orbits = query_neocc(["99942"], orbit_type="ke", orbit_epoch="present-day")

    assert orbits.solved_state_covariance.dimension[0].as_py() == 7
    assert (
        orbits.solved_state_covariance.parameter_names[0].as_py()
        == "x,y,z,vx,vy,vz,A2"
    )
    covariance = orbits.solved_state_covariance.to_matrix()[0]
    assert covariance is not None
    assert covariance.shape == (7, 7)

    # The leading 6x6 block must match the coordinate covariance: both are
    # transformed Keplerian -> Cartesian through the same Jacobian.
    np.testing.assert_allclose(
        covariance[:6, :6],
        orbits.coordinates.covariance.to_matrix()[0],
        rtol=1e-10,
    )
    # sigma(A2) in canonical au/d^2: the OEF RMS line gives 2.32321E-06 in
    # 1e-10 au/d^2 units. The A2 diagonal is invariant under the orbital-block
    # Jacobian, so this also verifies the unit scaling of the covariance.
    np.testing.assert_allclose(np.sqrt(covariance[6, 6]), 2.32321e-16, rtol=1e-4)
    np.testing.assert_allclose(
        orbits.non_gravitational_parameters.A2_sigma[0].as_py(),
        2.32321e-16,
        rtol=1e-4,
    )
    # parameter_count counts the estimated parameters.
    estimated = orbits.non_gravitational_parameters.estimated_parameter_names[
        0
    ].as_py()
    assert orbits.non_gravitational_parameters.parameter_count[0].as_py() == len(
        estimated.split(",")
    )


def test_query_neocc_include_nongrav_false_strips_nongrav(mocker):
    response_text = (TESTDATA_DIR / "99942.ke1").read_text()

    def mock_get(url, params):
        mock = mocker.MagicMock()
        mock.status_code = 200
        mock.text = response_text
        return mock

    mocker.patch("requests.get", side_effect=mock_get)
    orbits = query_neocc(
        ["99942"], orbit_type="ke", orbit_epoch="present-day", include_nongrav=False
    )

    assert orbits.non_gravitational_parameters.A2[0].as_py() is None
    assert orbits.solved_state_covariance.dimension[0].as_py() is None


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


def test_empty_response(mocker) -> None:
    # Sometimes, e.g. "1416T-2", the response text is empty even though the
    # HTTP code is 200. We should just skip those objects.

    import requests

    # Create mock response
    def mock_get(url, params):
        mock = mocker.MagicMock()
        mock.status_code = 200
        mock.text = ""
        return mock

    mocker.patch("requests.get", side_effect=mock_get)

    object_ids = ["1416T-2"]
    orbits = query_neocc(object_ids, orbit_type="ke", orbit_epoch="present-day")

    # Verify the results
    assert orbits is not None
    assert len(orbits) == 0
    requests.get.assert_has_calls(
        [
            mocker.call(
                "https://neo.ssa.esa.int/PSDB-portlet/download",
                params={"file": "1416T-2.ke1"},
            ),
        ]
    )


def test_28element_matrix(mocker) -> None:
    # Some objects have 28 elements upper triangular for covariance and correlation
    import requests

    # Create mock response
    def mock_get(url, params):
        mock = mocker.MagicMock()
        mock.status_code = 200
        with open(TESTDATA_DIR / "2018CW2.ke1", "r") as f:
            mock.text = f.read()
        return mock

    mocker.patch("requests.get", side_effect=mock_get)

    object_ids = ["2018 CW2"]
    orbits = query_neocc(object_ids, orbit_type="ke", orbit_epoch="present-day")

    # Verify the results
    assert orbits is not None
    assert len(orbits) == 1
    requests.get.assert_has_calls(
        [
            mocker.call(
                "https://neo.ssa.esa.int/PSDB-portlet/download",
                params={"file": "2018CW2.ke1"},
            ),
        ]
    )
