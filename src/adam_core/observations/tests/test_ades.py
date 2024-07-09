import pytest

from ...time import Timestamp
from ..ades import (
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
        rmsRA=[1 / 3600, 1 / 3600, None, None],
        rmsDec=[1 / 3600, 1 / 3600, None, None],
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
            "rmsRA": 4,
            "rmsDec": 4,
            "mag": 1,
            "rmsMag": 1,
        },
    )
    assert desired == actual
