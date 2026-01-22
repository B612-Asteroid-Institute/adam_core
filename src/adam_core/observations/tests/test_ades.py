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
! design Reflector
! aperture 4.0
! detector CCD
! name Blanco 4m
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
! design Reflector
! aperture 2.3
! detector CCD
! name Bok 2.3m
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
! design Reflector
! aperture 4.0
! detector CCD
! name Mayall 4m
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
permID|trkSub|obsSubID|obsTime|ra|dec|mag|band|stn|mode|astCat|remarks
3001|a2345b|obs04|2024-05-05T04:48:00.000Z|15.050000000|10.050000000|21.4000|r|695|CCD|Gaia2|This is the fourth dummy observation
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
! design Reflector
! aperture 2.3
! detector CCD
! name Bok 2.3m
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
permID|trkSub|obsSubID|obsTime|ra|dec|stn|mode|astCat
3001|a2345b|obs03|2024-05-05T00:00:00.000Z|15.000000000|10.000000000|V00|CCD|Gaia2
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
! design Reflector
! aperture 4.0
! detector CCD
! name Blanco 4m
# software
! objectDetection Asteroid Discovery Analysis and Mapping + Tracklet-less Heliocentric Orbit Recovery (ADAM::THOR)
# fundingSource WilliamBowes, McGregorGirand, Tito's Vodka, PRawls, SKraus, Yishan/KWong, SGalitsky, Google
# comment
! line THIS IS A TEST FILE CONTAINING FAKE OBSERVATIONS
! line Discovery candidates found by members of the Asteroid Institute, a program of B612 Foundation,
! line using the institute's ADAM::THOR discovery service running on Google Cloud.
permID|trkSub|obsSubID|obsTime|ra|dec|rmsRA|rmsDec|mag|band|stn|mode|astCat|remarks
3000|a1234b|obs01|2024-05-04T00:00:00.000Z|240.000000000|-15.000000000|0.96590|1.00000|20.0000|r|W84|CCD|Gaia2|This is a dummy observation
3000|a1234b|obs02|2024-05-04T02:24:00.000Z|240.050000000|-15.050000000|0.96570|1.00000|20.3000|g|W84|CCD|Gaia2|This is another dummy observation
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
! design Reflector
! aperture 4.0
! detector CCD
! name Mayall 4m
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
! design Reflector
! aperture 2.3
! detector CCD
! name Bok 2.3m
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
! design Reflector
! aperture 4.0
! detector CCD
! name Blanco 4m
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
! aperture 1.0
! design Reflector
! detector CCD
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
    assert context.telescope.aperture == 1.0
    assert context.telescope.design == "Reflector"
    assert context.telescope.detector == "CCD"
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
! aperture 1.0
! design Reflector
! detector CCD
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
! aperture 1.0
! design Reflector
! detector CCD
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
! aperture 1.0
! design Reflector
! detector CCD
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
! aperture 1.0
! design Reflector
! detector CCD
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
! aperture 1.0
! design Reflector
! detector CCD
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


def test_ADES_roundtrip_optional_fields():
    """Round-trip a variety of optional ADES fields (optical, radar, pos/cov)."""
    observations = ADESObservations.from_kwargs(
        permID=["X1"],
        obsSubID=["o1"],
        obsTime=Timestamp.from_mjd([60434.0], scale="utc"),
        ra=[123.456789],
        dec=[-20.123456],
        # Optical extras
        raStar=[123.4561],
        decStar=[-20.1231],
        deltaRA=[0.1234],
        deltaDec=[-0.2345],
        dist=[1.2345],
        pa=[45.67],
        rmsDist=[0.0123],
        rmsPA=[0.0456],
        # Photometry/exposure
        mag=[20.12],
        rmsMag=[0.03],
        band=["r"],
        fltr=["r"],
        photAp=["3.0"],
        rmsFit=[0.01],
        nucMag=[19.9],
        # Generic coordinates/covariance
        ctr=[399],
        pos1=[1.0],
        pos2=[2.0],
        pos3=[3.0],
        poscov11=[0.1],
        poscov12=[0.01],
        poscov13=[0.001],
        poscov22=[0.2],
        poscov23=[0.02],
        poscov33=[0.3],
        vel1=[0.1],
        vel2=[0.2],
        vel3=[0.3],
        # Radar
        delay=[123.456789],
        rmsDelay=[0.123456],
        doppler=[-12.3456],
        rmsDoppler=[0.01234],
        frq=[8560.0],
        trx=["TX"],
        rcv=["RX"],
        sys=["X"],
        # Time precision/uncertainty
        uncTime=["~0.1s"],
        precTime=[3],
        precRA=["mas"],
        precDec=["mas"],
        # Context
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
        obsCenter=["TelescopeCenter"],
        remarks=["opt+radar+cov"],
    )

    # Write and parse without contexts for simplicity
    s = ADES_to_string(observations, None)
    _, parsed = ADES_string_to_tables(s)

    # Compare numerics and strings one by one
    for col in [
        # Optical extras
        "raStar",
        "decStar",
        "deltaRA",
        "deltaDec",
        "dist",
        "pa",
        "rmsDist",
        "rmsPA",
        # Photometry/exposure
        "mag",
        "rmsMag",
        "rmsFit",
        "nucMag",
        # Generic coords/covariance
        "pos1",
        "pos2",
        "pos3",
        "poscov11",
        "poscov12",
        "poscov13",
        "poscov22",
        "poscov23",
        "poscov33",
        "vel1",
        "vel2",
        "vel3",
        # Radar
        "delay",
        "rmsDelay",
        "doppler",
        "rmsDoppler",
        "frq",
    ]:
        assert getattr(parsed, col)[0].as_py() == getattr(observations, col)[0].as_py()

    for col in [
        "fltr",
        "photAp",
        "trx",
        "rcv",
        "sys",
        "uncTime",
        "precRA",
        "precDec",
        "obsCenter",
        "remarks",
    ]:
        assert getattr(parsed, col)[0].as_py() == getattr(observations, col)[0].as_py()

    # Core requireds remain intact
    assert parsed.permID[0].as_py() == "X1"
    assert parsed.stn[0].as_py() == "695"
    assert parsed.mode[0].as_py() == "CCD"
    assert parsed.astCat[0].as_py() == "Gaia2"


def test_ADES_writer_omits_all_null_optional_headers():
    """Verify optional columns do not appear when entirely null."""
    observations = ADESObservations.from_kwargs(
        permID=["X2"],
        obsTime=Timestamp.from_mjd([60434.5], scale="utc"),
        ra=[10.0],
        dec=[-5.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
    )

    s = ADES_to_string(observations, None)
    # Find the header line (first non-comment line)
    header_line = next(
        (line for line in s.split("\n") if line and not line.startswith("#")),
        "",
    )
    # Ensure some optional names are not present
    assert "raStar" not in header_line
    assert "deltaRA" not in header_line
    assert "pos1" not in header_line
    assert "delay" not in header_line
    assert "photAp" not in header_line


def test_ADES_writer_enforces_group_orders():
    """Ensure key ADES groups are ordered in header when present."""
    observations = ADESObservations.from_kwargs(
        permID=["X3"],
        obsSubID=["o1"],
        obsTime=Timestamp.from_mjd([60436.0], scale="utc"),
        ra=[200.0],
        dec=[10.0],
        rmsRACosDec=[0.5],
        rmsDec=[0.6],
        rmsCorr=[0.01],
        # Optical groups
        raStar=[200.1],
        decStar=[10.1],
        deltaRA=[0.12],
        deltaDec=[-0.34],
        dist=[1.23],
        pa=[33.3],
        rmsDist=[0.02],
        rmsPA=[0.04],
        # Generic coords/covariance
        sys=["J2000"],
        ctr=[399],
        pos1=[1.0],
        pos2=[2.0],
        pos3=[3.0],
        vel1=[0.1],
        vel2=[0.2],
        vel3=[0.3],
        poscov11=[0.1],
        poscov12=[0.01],
        poscov13=[0.001],
        poscov22=[0.2],
        poscov23=[0.02],
        poscov33=[0.3],
        # Radar
        trx=["TX"],
        rcv=["RX"],
        frq=[8560.0],
        delay=[100.0],
        rmsDelay=[0.1],
        doppler=[-10.0],
        rmsDoppler=[0.01],
        # Photometry
        mag=[21.0],
        rmsMag=[0.1],
        band=["r"],
        fltr=["r"],
        photAp=["3.0"],
        photCat=["PS1"],
        nucMag=[20.0],
        rmsFit=[0.02],
        logSNR=[1.5],
        seeing=[1.2],
        exp=[30.0],
        # Station/context
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
        obsCenter=["Center"],
    )

    s = ADES_to_string(observations, None)
    header_line = next(
        (line for line in s.split("\n") if line and not line.startswith("#")), ""
    )
    cols = header_line.split("|")

    def assert_increasing(sequence):
        idx = [cols.index(c) for c in sequence if c in cols]
        assert idx == sorted(idx), f"Order incorrect for group {sequence}: {idx}"

    # Time group
    assert_increasing(["obsTime", "rmsTime", "precTime", "uncTime"])
    # RA/Dec group
    assert_increasing(["ra", "dec", "rmsRA", "rmsDec", "rmsCorr"])
    # Optical groups
    assert_increasing(["raStar", "decStar"])
    assert_increasing(["deltaRA", "deltaDec"])
    assert_increasing(["dist", "pa", "rmsDist", "rmsPA"])
    # Generic coords/covariance
    assert_increasing(
        [
            "sys",
            "ctr",
            "pos1",
            "pos2",
            "pos3",
            "vel1",
            "vel2",
            "vel3",
            "poscov11",
            "poscov12",
            "poscov13",
            "poscov22",
            "poscov23",
            "poscov33",
        ]
    )
    # Radar
    assert_increasing(
        ["trx", "rcv", "frq", "delay", "rmsDelay", "doppler", "rmsDoppler"]
    )
    # Photometry
    assert_increasing(
        ["mag", "rmsMag", "band", "fltr", "photAp", "photCat", "nucMag", "rmsFit"]
    )
    # Meta
    assert_increasing(["stn", "mode", "astCat", "obsCenter"])


def test_ADES_writer_nulls_emit_empty_cells_not_nan():
    """Ensure missing numeric values serialize as empty fields, not 'nan'."""
    observations = ADESObservations.from_kwargs(
        permID=["A", "A"],
        obsSubID=["1", "2"],
        obsTime=Timestamp.from_mjd([60430.0, 60430.1], scale="utc"),
        ra=[10.0, 10.1],
        dec=[-5.0, -5.1],
        mag=[21.0, None],  # second row missing
        band=["r", "r"],
        stn=["695", "695"],
        mode=["CCD", "CCD"],
        astCat=["Gaia2", "Gaia2"],
    )

    s = ADES_to_string(observations, None)
    lines = [ln for ln in s.split("\n") if ln and not ln.startswith("#")]
    header = lines[0].split("|")
    row1 = lines[1].split("|")
    row2 = lines[2].split("|")

    mag_idx = header.index("mag") if "mag" in header else -1
    assert mag_idx != -1, "mag should be present in header"
    assert row1[mag_idx] == "21.0000"
    assert (
        row2[mag_idx] == ""
    ), f"Expected empty cell for null mag, got {row2[mag_idx]!r}"


def test_ADES_string_to_tables_artSat_keyword_record_detection():
    """Ensure PSV parsing detects artSat-only Keyword Records (ADES Table 4 alternative C)."""
    ades_string = """# version=2022
# observatory
! mpcCode 695
# submitter
! name Submitter
# observers
! name Observer1
# measurers
! name Measurer1
# telescope
! name Telescope1
! aperture 1.0
! design Reflector
! detector CCD
artSat|obsTime|ra|dec|stn|mode|astCat
1998-067A|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2
"""

    _, observations = ADES_string_to_tables(ades_string)
    assert len(observations) == 1
    assert observations.artSat[0].as_py() == "1998-067A"


def test_ADES_to_string_validation_seconds_precision_cap():
    observations = ADESObservations.from_kwargs(
        permID=["A"],
        obsTime=Timestamp.from_mjd([60430.0], scale="utc"),
        ra=[10.0],
        dec=[-5.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
    )

    with pytest.raises(ValueError):
        ADES_to_string(
            observations,
            None,
            seconds_precision=7,
            goal="spec_compliance",
            enforcement="error",
        )

    s = ADES_to_string(
        observations,
        None,
        seconds_precision=7,
        goal="spec_compliance",
        enforcement="autofix",
    )
    lines = [ln for ln in s.split("\n") if ln and not ln.startswith("#")]
    header = lines[0].split("|")
    row = lines[1].split("|")
    obs_time = row[header.index("obsTime")]
    frac = obs_time.split(".")[1].rstrip("Z") if "." in obs_time else ""
    assert len(frac) <= 6


def test_ADES_to_string_validation_ctr_and_mag():
    observations = ADESObservations.from_kwargs(
        permID=["A"],
        obsTime=Timestamp.from_mjd([60430.0], scale="utc"),
        ra=[10.0],
        dec=[-5.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
        ctr=[1],
        mag=[40.0],
        band=["r"],
    )

    with pytest.raises(ValueError):
        ADES_to_string(observations, None, goal="spec_compliance", enforcement="error")


def test_find_ades_psv_problems_detects_non_lowercase_keyword_tokens():
    # "Band" appears in the PDF table, but PSV element names must be lower-case (band).
    ades_string = """# version=2022
permID|obsTime|ra|dec|stn|mode|astCat|Band
1234|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2|r
"""
    from ..ades import find_ades_psv_problems

    problems = find_ades_psv_problems(ades_string)
    codes = {p.code for p in problems}
    assert "keyword_record_tokens_not_lowercase" in codes
