import os

import numpy as np
import pyarrow.compute as pc
import pytest

# Check if required modules are available
try:
    from adam_assist import ASSISTPropagator

    from ...orbits.oem_io import orbit_from_oem, orbit_to_oem, orbit_to_oem_propagated

    _OEM_AVAILABLE = True
except ImportError:
    _OEM_AVAILABLE = False

from ...coordinates import CartesianCoordinates
from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.origin import Origin
from ...time import Timestamp
from ..orbits import Orbits

astroforge_optical_path = (
    f"{os.path.dirname(__file__)}/testdata/AstroForgeSC_Optical.oem"
)

covariance_example_path = f"{os.path.dirname(__file__)}/testdata/CovarianceExample.oem"


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_from_oem():
    optical_orbits = orbit_from_oem(astroforge_optical_path)

    first_segment = optical_orbits.apply_mask(
        pc.match_substring(optical_orbits.orbit_id, "seg_0")
    )
    second_segment = optical_orbits.apply_mask(
        pc.match_substring(optical_orbits.orbit_id, "seg_1")
    )

    assert len(first_segment) == 10
    assert len(second_segment) == 5
    assert len(optical_orbits) == 15

    assert optical_orbits.coordinates.frame == "equatorial"  # J2000
    assert optical_orbits.coordinates.origin.code[0].as_py() == "EARTH"


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_from_oem_covariance():
    orbits = orbit_from_oem(covariance_example_path)

    assert len(orbits) == 4

    assert not orbits.coordinates.covariance[0].is_all_nan()
    assert orbits.coordinates.covariance[1:].is_all_nan()


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_propagated(tmp_path):
    mjds = [60000.0, 60001.0, 60002.0, 60003.0, 60004.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")
    test_orbit = Orbits.from_kwargs(
        object_id=["object_id1"],
        orbit_id=["test_orbit"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times[0],
            x=[1.0],
            y=[1.1],
            z=[1.2],
            vx=[0.01],
            vy=[0.01],
            vz=[0.01],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["SUN"]),
            covariance=CoordinateCovariances.from_sigmas(
                np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
            ),
        ),
    )

    oem_path = f"{tmp_path}/test.oem"
    orbit_to_oem_propagated(test_orbit, oem_path, times, ASSISTPropagator)

    # test load
    orbits_rt = orbit_from_oem(oem_path)

    orbits_rt_df = orbits_rt.to_dataframe()

    assert orbits_rt.coordinates.frame == "equatorial"
    assert orbits_rt.coordinates.origin.code[0].as_py() == "SUN"
    assert len(orbits_rt_df) == 5
    assert orbits_rt.coordinates.time.scale == "tdb"
    assert not np.any([cov.is_all_nan() for cov in orbits_rt.coordinates.covariance])


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_basic(tmp_path):
    """Test basic functionality of orbit_to_oem with multiple time points."""
    mjds = [60000.0, 60001.0, 60002.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")

    # Create pre-propagated orbits with multiple time points
    test_orbits = Orbits.from_kwargs(
        object_id=["test_object", "test_object", "test_object"],
        orbit_id=["orbit_1", "orbit_2", "orbit_3"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times,
            x=[1.0, 1.1, 1.2],
            y=[2.0, 2.1, 2.2],
            z=[3.0, 3.1, 3.2],
            vx=[0.01, 0.011, 0.012],
            vy=[0.02, 0.021, 0.022],
            vz=[0.03, 0.031, 0.032],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
            covariance=CoordinateCovariances.nulls(3),
        ),
    )

    oem_path = f"{tmp_path}/test_basic.oem"
    result_path = orbit_to_oem(test_orbits, oem_path)

    assert result_path == oem_path
    assert os.path.exists(oem_path)

    # Verify we can read it back
    orbits_rt = orbit_from_oem(oem_path)
    assert len(orbits_rt) == 3
    assert orbits_rt.coordinates.frame == "equatorial"
    assert orbits_rt.coordinates.origin.code[0].as_py() == "SUN"


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_with_covariance(tmp_path):
    """Test orbit_to_oem with covariance data."""
    mjds = [60000.0, 60001.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")

    # Create covariance matrices
    cov_matrices = np.array(
        [
            [
                [1.0, 0.1, 0.2, 0.01, 0.02, 0.03],
                [0.1, 2.0, 0.3, 0.04, 0.05, 0.06],
                [0.2, 0.3, 3.0, 0.07, 0.08, 0.09],
                [0.01, 0.04, 0.07, 0.001, 0.002, 0.003],
                [0.02, 0.05, 0.08, 0.002, 0.002, 0.004],
                [0.03, 0.06, 0.09, 0.003, 0.004, 0.003],
            ],
            [
                [1.1, 0.11, 0.21, 0.011, 0.021, 0.031],
                [0.11, 2.1, 0.31, 0.041, 0.051, 0.061],
                [0.21, 0.31, 3.1, 0.071, 0.081, 0.091],
                [0.011, 0.041, 0.071, 0.0011, 0.0021, 0.0031],
                [0.021, 0.051, 0.081, 0.0021, 0.0021, 0.0041],
                [0.031, 0.061, 0.091, 0.0031, 0.0041, 0.0031],
            ],
        ]
    )

    test_orbits = Orbits.from_kwargs(
        object_id=["test_object", "test_object"],
        orbit_id=["orbit_1", "orbit_2"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times,
            x=[1.0, 1.1],
            y=[2.0, 2.1],
            z=[3.0, 3.1],
            vx=[0.01, 0.011],
            vy=[0.02, 0.021],
            vz=[0.03, 0.031],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["EARTH", "EARTH"]),
            covariance=CoordinateCovariances.from_matrix(cov_matrices),
        ),
    )

    oem_path = f"{tmp_path}/test_covariance.oem"
    orbit_to_oem(test_orbits, oem_path)

    # Verify covariance is preserved
    orbits_rt = orbit_from_oem(oem_path)
    assert len(orbits_rt) == 2
    assert not orbits_rt.coordinates.covariance[0].is_all_nan()
    assert not orbits_rt.coordinates.covariance[1].is_all_nan()


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_custom_originator(tmp_path):
    """Test orbit_to_oem with custom originator."""
    mjds = [60000.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")

    test_orbits = Orbits.from_kwargs(
        object_id=["test_object"],
        orbit_id=["orbit_1"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times,
            x=[1.0],
            y=[2.0],
            z=[3.0],
            vx=[0.01],
            vy=[0.02],
            vz=[0.03],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["SUN"]),
            covariance=CoordinateCovariances.nulls(1),
        ),
    )

    oem_path = f"{tmp_path}/test_originator.oem"
    custom_originator = "CUSTOM TEST ORIGINATOR"
    orbit_to_oem(test_orbits, oem_path, originator=custom_originator)

    assert os.path.exists(oem_path)

    # Read the file and check originator is set correctly
    # Note: We can't easily verify the originator without parsing the OEM file directly
    # but we can at least verify the file was created successfully
    orbits_rt = orbit_from_oem(oem_path)
    assert len(orbits_rt) == 1


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_different_origins(tmp_path):
    """Test orbit_to_oem with different supported origins."""
    mjds = [60000.0, 60001.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")

    # Test different origins
    origins = ["SUN", "EARTH", "MARS", "JUPITER"]

    for origin in origins:
        test_orbits = Orbits.from_kwargs(
            object_id=["test_object", "test_object"],
            orbit_id=["orbit_1", "orbit_2"],
            coordinates=CartesianCoordinates.from_kwargs(
                time=times,
                x=[1.0, 1.1],
                y=[2.0, 2.1],
                z=[3.0, 3.1],
                vx=[0.01, 0.011],
                vy=[0.02, 0.021],
                vz=[0.03, 0.031],
                frame="equatorial",
                origin=Origin.from_kwargs(code=[origin, origin]),
                covariance=CoordinateCovariances.nulls(2),
            ),
        )

        oem_path = f"{tmp_path}/test_{origin.lower()}.oem"
        orbit_to_oem(test_orbits, oem_path)

        # Verify origin is preserved
        orbits_rt = orbit_from_oem(oem_path)
        assert orbits_rt.coordinates.origin.code[0].as_py() == origin


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_single_time_warning(tmp_path, caplog):
    """Test that orbit_to_oem issues a warning for single time point."""
    mjds = [60000.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")

    test_orbits = Orbits.from_kwargs(
        object_id=["test_object"],
        orbit_id=["orbit_1"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times,
            x=[1.0],
            y=[2.0],
            z=[3.0],
            vx=[0.01],
            vy=[0.02],
            vz=[0.03],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["SUN"]),
            covariance=CoordinateCovariances.nulls(1),
        ),
    )

    oem_path = f"{tmp_path}/test_single_time.oem"

    with caplog.at_level("WARNING"):
        orbit_to_oem(test_orbits, oem_path)

    assert "only one time" in caplog.text
    assert "orbit_to_oem_propagated" in caplog.text


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_multiple_object_ids_error(tmp_path):
    """Test that orbit_to_oem raises error for multiple object_ids."""
    mjds = [60000.0, 60001.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")

    test_orbits = Orbits.from_kwargs(
        object_id=["object_1", "object_2"],  # Different object IDs
        orbit_id=["orbit_1", "orbit_2"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times,
            x=[1.0, 1.1],
            y=[2.0, 2.1],
            z=[3.0, 3.1],
            vx=[0.01, 0.011],
            vy=[0.02, 0.021],
            vz=[0.03, 0.031],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            covariance=CoordinateCovariances.nulls(2),
        ),
    )

    oem_path = f"{tmp_path}/test_error.oem"

    with pytest.raises(AssertionError, match="Only one object_id is supported"):
        orbit_to_oem(test_orbits, oem_path)


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_null_object_id_error(tmp_path):
    """Test that orbit_to_oem raises error for null object_id."""
    mjds = [60000.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")

    test_orbits = Orbits.from_kwargs(
        object_id=[None],  # Null object ID
        orbit_id=["orbit_1"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times,
            x=[1.0],
            y=[2.0],
            z=[3.0],
            vx=[0.01],
            vy=[0.02],
            vz=[0.03],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["SUN"]),
            covariance=CoordinateCovariances.nulls(1),
        ),
    )

    oem_path = f"{tmp_path}/test_null_error.oem"

    with pytest.raises(AssertionError, match="Orbits must specify object_id"):
        orbit_to_oem(test_orbits, oem_path)


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_unsupported_origin_error(tmp_path):
    """Test that orbit_to_oem raises error for unsupported origin."""
    mjds = [60000.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")

    test_orbits = Orbits.from_kwargs(
        object_id=["test_object"],
        orbit_id=["orbit_1"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times,
            x=[1.0],
            y=[2.0],
            z=[3.0],
            vx=[0.01],
            vy=[0.02],
            vz=[0.03],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["UNSUPPORTED_ORIGIN"]),
            covariance=CoordinateCovariances.nulls(1),
        ),
    )

    oem_path = f"{tmp_path}/test_unsupported_origin.oem"

    with pytest.raises(ValueError, match="Unsupported origin code"):
        orbit_to_oem(test_orbits, oem_path)


@pytest.mark.skipif(not _OEM_AVAILABLE, reason="OEM dependencies not installed")
def test_orbit_to_oem_time_sorting(tmp_path):
    """Test that orbit_to_oem properly sorts orbits by time."""
    # Create times out of order
    mjds = [60002.0, 60000.0, 60001.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")

    test_orbits = Orbits.from_kwargs(
        object_id=["test_object", "test_object", "test_object"],
        orbit_id=["orbit_1", "orbit_2", "orbit_3"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times,
            x=[3.0, 1.0, 2.0],  # Values corresponding to unsorted times
            y=[3.0, 1.0, 2.0],
            z=[3.0, 1.0, 2.0],
            vx=[0.03, 0.01, 0.02],
            vy=[0.03, 0.01, 0.02],
            vz=[0.03, 0.01, 0.02],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
            covariance=CoordinateCovariances.nulls(3),
        ),
    )

    oem_path = f"{tmp_path}/test_sorting.oem"
    orbit_to_oem(test_orbits, oem_path)

    # Read back and verify times are sorted
    orbits_rt = orbit_from_oem(oem_path)
    times_rt = orbits_rt.coordinates.time.mjd()

    # Should be sorted: 60000.0, 60001.0, 60002.0
    assert times_rt[0].as_py() == 60000.0
    assert times_rt[1].as_py() == 60001.0
    assert times_rt[2].as_py() == 60002.0

    # Verify corresponding x values are also sorted correctly
    x_values = orbits_rt.coordinates.x
    assert x_values[0].as_py() == pytest.approx(1.0)  # x value for 60000.0
    assert x_values[1].as_py() == pytest.approx(2.0)  # x value for 60001.0
    assert x_values[2].as_py() == pytest.approx(3.0)  # x value for 60002.0
