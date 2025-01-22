import numpy as np
import pytest

from ...time import Timestamp
from ..ades import (
    ADES_string_to_tables,
    ADES_to_string,
    ADESObservations,
    ObsContext,
    ObservatoryObsContext,
    SoftwareObsContext,
    SubmitterObsContext,
    TelescopeObsContext,
)


@pytest.fixture
def ades_obscontext():
    # Nearly real metadata used for ADAM::THOR observations
    measurers = [
        "J. Moeyens",
        "M. Juric",
        "S. Nelson",
        "A. Koumjian",
        "K. Kiker",
        "N. Tellis",
        "D. Veronese-Milin",
        "A. Posner",
        "E. Lu",
        "C. Fiaschetti",
        "D. Remy",
    ]

    software_context = SoftwareObsContext(
        objectDetection="Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)",
    )

    submitter_context = SubmitterObsContext(
        name="J. Moeyens", institution="B612 Asteroid Institute"
    )

    fundingSource = "WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google"
    comments = [
        "THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS",
        "Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,",
        "using the institute's ADAM::THOR discovery service running on Google Cloud.",
    ]

    obs_contexts = {
        "W84": ObsContext(
            observatory=ObservatoryObsContext(
                mpcCode="W84", name="Cerro Tololo - Blanco + DECam"
            ),
            submitter=submitter_context,
            observers=["D. E. Survey"],
            measurers=measurers,
            telescope=TelescopeObsContext(
                name="Blanco 4m", design="Reflector", detector="CCD", aperture=4.0
            ),
            software=software_context,
            fundingSource=fundingSource,
            comments=comments,
        ),
        "695": ObsContext(
            observatory=ObservatoryObsContext(
                mpcCode="695", name="Kitt Peak National Observatory - Mayall + Mosaic3"
            ),
            submitter=submitter_context,
            observers=["M. L. Survey"],
            measurers=measurers,
            telescope=TelescopeObsContext(
                name="Mayall 4m",
                design="Reflector",
                detector="CCD",
                aperture=4.0,
            ),
            software=software_context,
            fundingSource=fundingSource,
            comments=comments,
        ),
        "V00": ObsContext(
            observatory=ObservatoryObsContext(
                mpcCode="V00", name="Kitt Peak National Observatory - Bok + 90Prime"
            ),
            submitter=submitter_context,
            observers=["B. A. S. Survey"],
            measurers=measurers,
            telescope=TelescopeObsContext(
                name="Bok 2.3m",
                design="Reflector",
                detector="CCD",
                aperture=2.3,
            ),
            software=software_context,
            fundingSource=fundingSource,
            comments=comments,
        ),
    }

    return obs_contexts


@pytest.fixture
def ades_observations():

    observations = ADESObservations.from_kwargs(
        permID=["3000", "3000", "3001", "3001"],
        trkSub=["a1234b", "a1234b", "a2345b", "a2345b"],
        obsSubID=["obs01", "obs02", "obs03", "obs04"],
        obsTime=Timestamp.from_mjd(
            [60434.0, 60434.1, 60435.0, 60435.2],
            scale="utc",
        ),
        ra=[240.00, 240.05, 15.00, 15.05],
        dec=[-15.00, -15.05, 10.00, 10.05],
        rmsRACosDec=[0.9659, 0.9657, None, None],
        rmsDec=[1.0, 1.0, None, None],
        mag=[20.0, 20.3, None, 21.4],
        band=["r", "g", None, "r"],
        stn=["W84", "W84", "V00", "695"],
        mode=["CCD", "CCD", "CCD", "CCD"],
        astCat=["Gaia2", "Gaia2", "Gaia2", "Gaia2"],
        remarks=[
            "This is a dummy observation",
            "This is another dummy observation",
            None,
            "This is the fourth dummy observation",
        ],
    )
    return observations


def test_ObsContext_to_string(ades_obscontext):
    # Test that we can convert an ObsContext to a string representation
    W84 = ades_obscontext["W84"]
    string = W84.to_string()

    assert (
        string
        == """# observatory
! mpcCode W84
! name Cerro Tololo - Blanco + DECam
# submitter
! name J. Moeyens
! institution B612 Asteroid Institute
# observers
! name D. E. Survey
# measurers
! name J. Moeyens
! name M. Juric
! name S. Nelson
! name A. Koumjian
! name K. Kiker
! name N. Tellis
! name D. Veronese-Milin
! name A. Posner
! name E. Lu
! name C. Fiaschetti
! name D. Remy
# telescope
! name Blanco 4m
! design Reflector
! aperture 4.0
! detector CCD
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
"""  # noqa: E501
    )

    V00 = ades_obscontext["V00"]
    string = V00.to_string()

    assert (
        string
        == """# observatory
! mpcCode V00
! name Kitt Peak National Observatory - Bok + 90Prime
# submitter
! name J. Moeyens
! institution B612 Asteroid Institute
# observers
! name B. A. S. Survey
# measurers
! name J. Moeyens
! name M. Juric
! name S. Nelson
! name A. Koumjian
! name K. Kiker
! name N. Tellis
! name D. Veronese-Milin
! name A. Posner
! name E. Lu
! name C. Fiaschetti
! name D. Remy
# telescope
! name Bok 2.3m
! design Reflector
! aperture 2.3
! detector CCD
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
"""  # noqa: E501
    )


def test_ADES_to_string(ades_observations, ades_obscontext):

    desired = """# version=2022
# observatory
! mpcCode 695
! name Kitt Peak National Observatory - Mayall + Mosaic3
# submitter
! name J. Moeyens
! institution B612 Asteroid Institute
# observers
! name M. L. Survey
# measurers
! name J. Moeyens
! name M. Juric
! name S. Nelson
! name A. Koumjian
! name K. Kiker
! name N. Tellis
! name D. Veronese-Milin
! name A. Posner
! name E. Lu
! name C. Fiaschetti
! name D. Remy
# telescope
! name Mayall 4m
! design Reflector
! aperture 4.0
! detector CCD
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
permID|trkSub|obsSubID|obsTime|ra|dec|mag|band|stn|mode|astCat|remarks
3001|a2345b|obs04|2024-05-05T04:48:00.000Z|15.05000000|10.05000000|21.40|r|695|CCD|Gaia2|This is the fourth dummy observation
# observatory
! mpcCode V00
! name Kitt Peak National Observatory - Bok + 90Prime
# submitter
! name J. Moeyens
! institution B612 Asteroid Institute
# observers
! name B. A. S. Survey
# measurers
! name J. Moeyens
! name M. Juric
! name S. Nelson
! name A. Koumjian
! name K. Kiker
! name N. Tellis
! name D. Veronese-Milin
! name A. Posner
! name E. Lu
! name C. Fiaschetti
! name D. Remy
# telescope
! name Bok 2.3m
! design Reflector
! aperture 2.3
! detector CCD
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
permID|trkSub|obsSubID|obsTime|ra|dec|stn|mode|astCat
3001|a2345b|obs03|2024-05-05T00:00:00.000Z|15.00000000|10.00000000|V00|CCD|Gaia2
# observatory
! mpcCode W84
! name Cerro Tololo - Blanco + DECam
# submitter
! name J. Moeyens
! institution B612 Asteroid Institute
# observers
! name D. E. Survey
# measurers
! name J. Moeyens
! name M. Juric
! name S. Nelson
! name A. Koumjian
! name K. Kiker
! name N. Tellis
! name D. Veronese-Milin
! name A. Posner
! name E. Lu
! name C. Fiaschetti
! name D. Remy
# telescope
! name Blanco 4m
! design Reflector
! aperture 4.0
! detector CCD
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
permID|trkSub|obsSubID|obsTime|ra|dec|rmsRA|rmsDec|mag|band|stn|mode|astCat|remarks
3000|a1234b|obs01|2024-05-04T00:00:00.000Z|240.00000000|-15.00000000|0.9659|1.0000|20.00|r|W84|CCD|Gaia2|This is a dummy observation
3000|a1234b|obs02|2024-05-04T02:24:00.000Z|240.05000000|-15.05000000|0.9657|1.0000|20.30|g|W84|CCD|Gaia2|This is another dummy observation
"""  # noqa: E501

    actual = ADES_to_string(ades_observations, ades_obscontext)
    assert desired == actual

    desired = """# version=2022
# observatory
! mpcCode 695
! name Kitt Peak National Observatory - Mayall + Mosaic3
# submitter
! name J. Moeyens
! institution B612 Asteroid Institute
# observers
! name M. L. Survey
# measurers
! name J. Moeyens
! name M. Juric
! name S. Nelson
! name A. Koumjian
! name K. Kiker
! name N. Tellis
! name D. Veronese-Milin
! name A. Posner
! name E. Lu
! name C. Fiaschetti
! name D. Remy
# telescope
! name Mayall 4m
! design Reflector
! aperture 4.0
! detector CCD
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
permID|trkSub|obsSubID|obsTime|ra|dec|mag|band|stn|mode|astCat|remarks
3001|a2345b|obs04|2024-05-05T04:48:00.0Z|15.050000|10.050000|21.4|r|695|CCD|Gaia2|This is the fourth dummy observation
# observatory
! mpcCode V00
! name Kitt Peak National Observatory - Bok + 90Prime
# submitter
! name J. Moeyens
! institution B612 Asteroid Institute
# observers
! name B. A. S. Survey
# measurers
! name J. Moeyens
! name M. Juric
! name S. Nelson
! name A. Koumjian
! name K. Kiker
! name N. Tellis
! name D. Veronese-Milin
! name A. Posner
! name E. Lu
! name C. Fiaschetti
! name D. Remy
# telescope
! name Bok 2.3m
! design Reflector
! aperture 2.3
! detector CCD
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
permID|trkSub|obsSubID|obsTime|ra|dec|stn|mode|astCat
3001|a2345b|obs03|2024-05-05T00:00:00.0Z|15.000000|10.000000|V00|CCD|Gaia2
# observatory
! mpcCode W84
! name Cerro Tololo - Blanco + DECam
# submitter
! name J. Moeyens
! institution B612 Asteroid Institute
# observers
! name D. E. Survey
# measurers
! name J. Moeyens
! name M. Juric
! name S. Nelson
! name A. Koumjian
! name K. Kiker
! name N. Tellis
! name D. Veronese-Milin
! name A. Posner
! name E. Lu
! name C. Fiaschetti
! name D. Remy
# telescope
! name Blanco 4m
! design Reflector
! aperture 4.0
! detector CCD
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
permID|trkSub|obsSubID|obsTime|ra|dec|rmsRA|rmsDec|mag|band|stn|mode|astCat|remarks
3000|a1234b|obs01|2024-05-04T00:00:00.0Z|240.000000|-15.000000|0.9659|1.0000|20.0|r|W84|CCD|Gaia2|This is a dummy observation
3000|a1234b|obs02|2024-05-04T02:24:00.0Z|240.050000|-15.050000|0.9657|1.0000|20.3|g|W84|CCD|Gaia2|This is another dummy observation
"""  # noqa: E501

    actual = ADES_to_string(
        ades_observations,
        ades_obscontext,
        seconds_precision=1,
        columns_precision={
            "ra": 6,
            "dec": 6,
            "rmsRACosDec": 4,
            "rmsDec": 4,
            "mag": 1,
            "rmsMag": 1,
        },
    )
    assert desired == actual


def test_ADES_string_to_tables(ades_observations, ades_obscontext):
    # Create the ADES string that we'll parse
    ades_string = ADES_to_string(ades_observations, ades_obscontext)
    # Parse the string back into objects
    parsed_contexts, parsed_observations = ADES_string_to_tables(ades_string)

    # Test that we got the same observatory contexts back
    assert set(parsed_contexts.keys()) == set(ades_obscontext.keys())
    for code in ades_obscontext:
        expected = ades_obscontext[code]
        actual = parsed_contexts[code]

        # Test each field of the ObsContext
        assert actual.observatory.mpcCode == expected.observatory.mpcCode
        assert actual.observatory.name == expected.observatory.name

        assert actual.submitter.name == expected.submitter.name
        assert actual.submitter.institution == expected.submitter.institution

        assert actual.observers == expected.observers
        assert actual.measurers == expected.measurers

        assert actual.telescope.name == expected.telescope.name
        assert actual.telescope.design == expected.telescope.design
        assert actual.telescope.aperture == expected.telescope.aperture
        assert actual.telescope.detector == expected.telescope.detector

        if expected.software is not None:
            assert actual.software.objectDetection == expected.software.objectDetection
            assert actual.software.astrometry == expected.software.astrometry
            assert actual.software.fitOrder == expected.software.fitOrder
            assert actual.software.photometry == expected.software.photometry
        else:
            assert actual.software is None

        assert actual.fundingSource == expected.fundingSource
        assert actual.comments == expected.comments

    # Order the observations by mpc code and obsTime
    ades_observations = ades_observations.sort_by(
        ["stn", "obsTime.days", "obsTime.nanos"]
    )
    parsed_observations = parsed_observations.sort_by(
        ["stn", "obsTime.days", "obsTime.nanos"]
    )

    # Test that we got the same observations back
    # First convert timestamps to MJD for comparison since that's what we use internally
    expected_mjd = ades_observations.obsTime.mjd().to_numpy(zero_copy_only=False)
    parsed_mjd = parsed_observations.obsTime.mjd().to_numpy(zero_copy_only=False)
    np.testing.assert_array_almost_equal(parsed_mjd, expected_mjd)

    # Test all other columns
    for col in [
        "permID",
        "provID",
        "trkSub",
        "obsSubID",
        "ra",
        "dec",
        "rmsRACosDec",
        "rmsDec",
        "mag",
        "band",
        "stn",
        "mode",
        "astCat",
        "remarks",
    ]:
        if hasattr(ades_observations, col):
            expected = getattr(ades_observations, col).to_numpy(zero_copy_only=False)
            actual = getattr(parsed_observations, col).to_numpy(zero_copy_only=False)
            np.testing.assert_array_equal(actual, expected, err_msg=f"{col} not equal")


def test_ADES_string_to_tables_minimal():
    """Test parsing a minimal ADES string with just required fields."""
    minimal_string = """# version=2022
# observatory
! mpcCode 695
# submitter
! name J. Moeyens
# observers
! name Observer1
# measurers
! name Measurer1
# telescope
! name Telescope1
! design Reflector
permID|obsTime|ra|dec|stn|mode|astCat
1234|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2
"""

    parsed_contexts, parsed_observations = ADES_string_to_tables(minimal_string)

    # Test observatory context
    assert len(parsed_contexts) == 1
    assert "695" in parsed_contexts
    context = parsed_contexts["695"]

    assert context.observatory.mpcCode == "695"
    assert context.observatory.name is None
    assert context.submitter.name == "J. Moeyens"
    assert context.submitter.institution is None
    assert context.observers == ["Observer1"]
    assert context.measurers == ["Measurer1"]
    assert context.telescope.name == "Telescope1"
    assert context.telescope.design == "Reflector"
    assert context.software is None
    assert context.fundingSource is None
    assert context.comments == []

    # Test observations
    assert len(parsed_observations) == 1
    assert parsed_observations.permID[0].as_py() == "1234"
    assert parsed_observations.ra[0].as_py() == 180.0
    assert parsed_observations.dec[0].as_py() == 0.0
    assert parsed_observations.stn[0].as_py() == "695"
    assert parsed_observations.mode[0].as_py() == "CCD"
    assert parsed_observations.astCat[0].as_py() == "Gaia2"


def test_ADES_string_to_tables_empty_observations():
    """Test parsing an ADES string with metadata but no observations."""
    empty_string = """# version=2022
# observatory
! mpcCode 695
# submitter
! name J. Moeyens
# observers
! name Observer1
# measurers
! name Measurer1
# telescope
! name Telescope1
! design Reflector
permID|obsTime|ra|dec|stn|mode|astCat
"""

    parsed_contexts, parsed_observations = ADES_string_to_tables(empty_string)

    # Test that we got the context but no observations
    assert len(parsed_contexts) == 1
    assert "695" in parsed_contexts
    assert len(parsed_observations) == 0


def test_ADES_string_to_tables_multiple_observatories():
    """Test parsing an ADES string with multiple observatory contexts."""
    multi_obs_string = """# version=2022
# observatory
! mpcCode 695
# submitter
! name J. Moeyens
# observers
! name Observer1
# measurers
! name Measurer1
# telescope
! name Telescope1
! design Reflector
# observatory
! mpcCode W84
# submitter
! name J. Moeyens
# observers
! name Observer2
# measurers
! name Measurer2
# telescope
! name Telescope2
! design Reflector
permID|obsTime|ra|dec|stn|mode|astCat
1234|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2
5678|2024-01-01T00:00:00.000Z|190.0|10.0|W84|CCD|Gaia2
"""

    parsed_contexts, parsed_observations = ADES_string_to_tables(multi_obs_string)

    # Test that we got both observatory contexts
    assert len(parsed_contexts) == 2
    assert set(parsed_contexts.keys()) == {"695", "W84"}

    # Test that we got observations from both observatories
    assert len(parsed_observations) == 2
    assert set(parsed_observations.stn.to_numpy(zero_copy_only=False)) == {"695", "W84"}


def test_ADES_string_to_tables_unknown_columns():
    """Test parsing an ADES string with columns we don't currently support."""
    ades_string = """# version=2022
# observatory
! mpcCode 695
# submitter
! name J. Moeyens
# observers
! name Observer1
# measurers
! name Measurer1
# telescope
! name Telescope1
! design Reflector
permID|obsTime|ra|dec|raStar|decStar|stn|mode|astCat
1234|2024-01-01T00:00:00.000Z|180.0|0.0|180.1|0.1|695|CCD|Gaia2
5678|2024-01-01T00:00:00.000Z|190.0|10.0|190.1|10.1|695|CCD|Gaia2
"""

    parsed_contexts, parsed_observations = ADES_string_to_tables(ades_string)

    # Test that we got the basic data correctly
    assert len(parsed_observations) == 2
    assert parsed_observations.permID[0].as_py() == "1234"
    assert parsed_observations.ra[0].as_py() == 180.0
    assert parsed_observations.dec[0].as_py() == 0.0
    assert parsed_observations.stn[0].as_py() == "695"
    assert parsed_observations.mode[0].as_py() == "CCD"
    assert parsed_observations.astCat[0].as_py() == "Gaia2"

    # Test that we got the second row correctly too
    assert parsed_observations.permID[1].as_py() == "5678"
    assert parsed_observations.ra[1].as_py() == 190.0
    assert parsed_observations.dec[1].as_py() == 10.0


def test_ADES_string_to_tables_null_handling():
    """Test parsing an ADES string with null fields represented as empty or whitespace."""
    ades_string = """# version=2022
# observatory
! mpcCode 695
# submitter
! name J. Moeyens
# observers
! name Observer1
# measurers
! name Measurer1
# telescope
! name Telescope1
! design Reflector
permID|obsTime|ra|dec|mag|band|stn|mode|astCat|remarks
1234|2024-01-01T00:00:00.000Z|180.0|0.0||r|695|CCD|Gaia2|First observation
5678|2024-01-01T00:00:00.000Z|190.0|10.0| |g|695|CCD|Gaia2|Second observation
9012|2024-01-01T00:00:00.000Z|200.0|20.0|   ||695|CCD|Gaia2|Third observation
"""

    parsed_contexts, parsed_observations = ADES_string_to_tables(ades_string)

    # Test that we got the observations
    assert len(parsed_observations) == 3

    # Test that empty and whitespace fields are converted to None
    assert parsed_observations.mag[0].as_py() is None  # Empty field
    assert parsed_observations.mag[1].as_py() is None  # Single space
    assert parsed_observations.mag[2].as_py() is None  # Multiple spaces
    assert parsed_observations.band[2].as_py() is None  # Empty field

    # Test that non-null fields are preserved
    assert parsed_observations.ra[0].as_py() == 180.0
    assert parsed_observations.band[0].as_py() == "r"
    assert parsed_observations.remarks[0].as_py() == "First observation"
