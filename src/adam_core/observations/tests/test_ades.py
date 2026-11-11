import numpy as np
import pyarrow as pa
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

    assert string == """# observatory
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

    V00 = ades_obscontext["V00"]
    string = V00.to_string()

    assert string == """# observatory
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


def test_ADES_to_string_without_context(ades_observations):
    ades_string = ADES_to_string(ades_observations, None)

    assert ades_string.startswith("# version=2022\n")
    assert "# observatory" not in ades_string
    contexts, parsed_observations = ADES_string_to_tables(ades_string)
    assert contexts == {}
    assert len(parsed_observations) == len(ades_observations)

    with pytest.raises(ValueError, match="not found in obs_contexts"):
        ADES_to_string(ades_observations, {})


def test_ADES_to_string_round_trips_mixed_null_numeric_cells():
    observations = ADESObservations.from_kwargs(
        permID=["3000", "3000"],
        obsSubID=["obs01", "obs02"],
        obsTime=Timestamp.from_mjd([60434.0, 60434.1], scale="utc"),
        rmsTime=[0.01, None],
        ra=[240.00, 240.05],
        dec=[-15.00, -15.05],
        rmsRACosDec=[0.9659, None],
        rmsDec=[1.0, None],
        rmsCorr=[0.1, None],
        mag=[20.0, None],
        rmsMag=[0.2, None],
        stn=["W84", "W84"],
        mode=["CCD", "CCD"],
        astCat=["Gaia2", "Gaia2"],
        logSNR=[5.0, None],
        seeing=[1.2, None],
        exp=[30.0, None],
    )

    ades_string = ADES_to_string(observations, None, sort=False)
    lines = ades_string.splitlines()
    header = lines[1].split("|")
    second_row = dict(zip(header, lines[3].split("|")))
    optional_numeric_columns = {
        "rmsTime",
        "rmsRA",
        "rmsDec",
        "rmsCorr",
        "mag",
        "rmsMag",
        "logSNR",
        "seeing",
        "exp",
    }
    assert optional_numeric_columns <= second_row.keys()
    assert all(second_row[column] == "" for column in optional_numeric_columns)
    assert "nan" not in ades_string.lower()

    contexts, parsed_observations = ADES_string_to_tables(ades_string)

    assert contexts == {}
    expected_numeric_values = {
        "rmsTime": 0.01,
        "rmsRACosDec": 0.9659,
        "rmsDec": 1.0,
        "rmsCorr": 0.1,
        "mag": 20.0,
        "rmsMag": 0.2,
        "logSNR": 5.0,
        "seeing": 1.2,
        "exp": 30.0,
    }
    for column, expected in expected_numeric_values.items():
        actual = parsed_observations.table.column(column).to_pylist()
        assert actual[0] == pytest.approx(expected)
        assert actual[1] is None


def test_ADES_writer_reader_round_trip_preserves_quote_characters():
    remarks = 'unmatched " and "quoted-looking" text'
    observations = ADESObservations.from_kwargs(
        provID=["2024 AB"],
        obsTime=Timestamp.from_mjd([60434.0], scale="utc"),
        ra=[180.0],
        dec=[0.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
        remarks=[remarks],
    )

    rendered = ADES_to_string(observations, None)
    assert f"|{remarks}\n" in rendered
    assert '""' not in rendered

    _contexts, parsed = ADES_string_to_tables(rendered)
    assert parsed.remarks.to_pylist() == [remarks]


@pytest.mark.parametrize(
    ("remarks", "message"),
    [
        ("unrepresentable | separator", r"unrepresentable `\|` delimiter"),
        ("unrepresentable\nnewline", "unrepresentable physical newline"),
    ],
)
def test_ADES_writer_rejects_unrepresentable_observation_fields(remarks, message):
    observations = ADESObservations.from_kwargs(
        provID=["2024 AB"],
        obsTime=Timestamp.from_mjd([60434.0], scale="utc"),
        ra=[180.0],
        dec=[0.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
        remarks=[remarks],
    )

    with pytest.raises(
        ValueError,
        match=rf"field remarks at observation row 1.*{message}",
    ):
        ADES_to_string(observations, None)


@pytest.mark.parametrize(
    "prov_id",
    ["#reserved", "!reserved", "  #reserved", "\t!reserved", " \t #reserved"],
)
def test_ADES_writer_reader_path_rejects_reserved_context_markers(prov_id):
    observations = ADESObservations.from_kwargs(
        provID=[prov_id],
        obsTime=Timestamp.from_mjd([60434.0], scale="utc"),
        ra=[180.0],
        dec=[0.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
    )

    def writer_reader_round_trip():
        return ADES_string_to_tables(ADES_to_string(observations, None))

    with pytest.raises(
        ValueError,
        match=r"field provID at observation row 1: first field begins with reserved context marker",
    ):
        writer_reader_round_trip()


def test_ADES_writer_reader_normalizes_safe_leading_identity_whitespace():
    observations = ADESObservations.from_kwargs(
        provID=["  2024 AB"],
        obsTime=Timestamp.from_mjd([60434.0], scale="utc"),
        ra=[180.0],
        dec=[0.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
    )

    _contexts, parsed = ADES_string_to_tables(ADES_to_string(observations, None))

    assert parsed.provID.to_pylist() == ["2024 AB"]


def test_ADES_to_string_publishes_large_schema_valid_uncertainties():
    observations = ADESObservations.from_kwargs(
        provID=["2024 YR4", "2024 YR4"],
        obsSubID=["large-rms", "ordinary-rms"],
        obsTime=Timestamp.from_mjd([60434.0, 60434.1], scale="utc"),
        ra=[240.00, 240.05],
        dec=[-15.00, -15.05],
        rmsRACosDec=[584.7052976482118, 3.6],
        rmsDec=[584.7052976482118, 7.2],
        stn=["W84", "W84"],
        mode=["CCD", "CCD"],
        astCat=["Gaia2", "Gaia2"],
    )

    ades_string = ADES_to_string(observations, None, sort=False)
    lines = ades_string.splitlines()
    header = lines[1].split("|")
    first = dict(zip(header, lines[2].split("|")))
    second = dict(zip(header, lines[3].split("|")))

    assert first["rmsRA"] == "584.705"
    assert first["rmsDec"] == "584.705"
    assert second["rmsRA"] == "3.60000"
    assert second["rmsDec"] == "7.20000"
    assert all(
        len(row[column]) <= 7
        for row in (first, second)
        for column in ("rmsRA", "rmsDec")
    )

    _contexts, parsed = ADES_string_to_tables(ades_string)
    assert parsed.rmsRACosDec.to_pylist() == pytest.approx([584.705, 3.6])
    assert parsed.rmsDec.to_pylist() == pytest.approx([584.705, 7.2])


def test_ADES_to_string_preserves_block_order_when_sort_disabled(ades_observations):
    observations = ades_observations.select("stn", "W84").take(pa.array([1, 0]))

    ades_string = ADES_to_string(observations, None, sort=False)
    _contexts, parsed_observations = ADES_string_to_tables(ades_string)

    assert parsed_observations.obsSubID.to_pylist() == observations.obsSubID.to_pylist()


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


def test_ADES_parser_preserves_literal_quotes_and_identity_names_per_line():
    ades_string = """# version=2022
provID|obsTime|ra|dec|stn|mode|astCat|remarks
2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2|plain provID review
2024 AC|2024-01-01T00:01:00.000Z|181.0|1.0|695|CCD|Gaia2|"unmatched provID review
2024 AD|2024-01-01T00:02:00.000Z|182.0|2.0|695|CCD|Gaia2|"quoted-looking provID text"
"""

    _contexts, observations = ADES_string_to_tables(ades_string)

    assert len(observations) == 3
    assert observations.provID.to_pylist() == ["2024 AB", "2024 AC", "2024 AD"]
    assert observations.remarks.to_pylist() == [
        "plain provID review",
        '"unmatched provID review',
        '"quoted-looking provID text"',
    ]


def test_ADES_parser_treats_pipes_inside_quotes_as_field_boundaries():
    ades_string = """# version=2022
provID|obsTime|ra|dec|stn|mode|astCat|remarks
2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2|"follow-up | provID review"
2024 AC|2024-01-01T00:01:00.000Z|181.0|1.0|695|CCD|Gaia2|not swallowed
"""

    with pytest.raises(
        ValueError,
        match=r"row at line 3 has 9 fields; expected 8.*line 2",
    ):
        ADES_string_to_tables(ades_string)


def test_ADES_parser_allows_unknown_header_tokens_with_warning(caplog):
    ades_string = """# version=2022
provID|obsTime|ra|dec|future_field|stn|mode|astCat
2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|future value|695|CCD|Gaia2
"""

    _contexts, observations = ADES_string_to_tables(ades_string)

    assert len(observations) == 1
    assert "future_field" in caplog.text
    assert "unknown ADES columns" in caplog.text


def test_ADES_parser_rejects_identity_header_missing_obs_time_explicitly():
    ades_string = """# version=2022
provID|ra|dec|stn|mode|astCat
2024 AB|180.0|0.0|695|CCD|Gaia2
"""

    with pytest.raises(
        ValueError,
        match=r"invalid ADES keyword record at line 2: missing required obsTime column",
    ) as exc_info:
        ADES_string_to_tables(ades_string)
    assert "invalid ISO time" not in str(exc_info.value)


def test_ADES_parser_rejects_non_first_identity_as_header_error():
    ades_string = """# version=2022
obsTime|provID|ra|dec|stn|mode|astCat
2024-01-01T00:00:00.000Z|2024 AB|180.0|0.0|695|CCD|Gaia2
"""

    with pytest.raises(
        ValueError,
        match=r"invalid ADES keyword record at line 2: target identity column provID must be the first field",
    ) as exc_info:
        ADES_string_to_tables(ades_string)
    assert "invalid ISO time" not in str(exc_info.value)


def test_ADES_context_writer_rejects_physical_newlines():
    observations = ADESObservations.from_kwargs(
        provID=["2024 AB"],
        obsTime=Timestamp.from_mjd([60434.0], scale="utc"),
        ra=[180.0],
        dec=[0.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
    )
    context = ObsContext(
        observatory=ObservatoryObsContext(mpcCode="695", name="bad\nname"),
        submitter=SubmitterObsContext(name="Submitter"),
        observers=["Observer"],
        measurers=["Measurer"],
        telescope=TelescopeObsContext(design="Reflector", aperture=1.0, detector="CCD"),
    )

    with pytest.raises(
        ValueError,
        match=r"context field observatory.name.*unrepresentable physical newline",
    ):
        ADES_to_string(observations, {"695": context})


def test_ADES_context_round_trip_preserves_literal_quotes_and_pipes():
    observations = ADESObservations.from_kwargs(
        provID=["2024 AB"],
        obsTime=Timestamp.from_mjd([60434.0], scale="utc"),
        ra=[180.0],
        dec=[0.0],
        stn=["695"],
        mode=["CCD"],
        astCat=["Gaia2"],
    )
    context = ObsContext(
        observatory=ObservatoryObsContext(
            mpcCode="695", name='unmatched " | observatory'
        ),
        submitter=SubmitterObsContext(name='"quoted | submitter"'),
        observers=['Observer " | literal'],
        measurers=['"Measurer | literal"'],
        telescope=TelescopeObsContext(design="Reflector", aperture=1.0, detector="CCD"),
        comments=['Comment with unmatched " and | separator'],
    )

    rendered = ADES_to_string(observations, {"695": context})
    parsed_contexts, parsed_observations = ADES_string_to_tables(rendered)

    assert len(parsed_observations) == 1
    assert parsed_contexts["695"] == context


def test_ADES_parser_accepts_first_line_bom_and_preserves_line_numbers():
    valid = """\ufeff# version=2022

provID|obsTime|ra|dec|stn|mode|astCat
2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2
"""
    _contexts, observations = ADES_string_to_tables(valid)
    assert len(observations) == 1

    short = """\ufeff# version=2022

provID|obsTime|ra|dec|stn|mode|astCat|band

2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2
"""
    with pytest.raises(
        ValueError,
        match=r"row at line 5 has 7 fields; expected 8.*line 3",
    ):
        ADES_string_to_tables(short)


def test_ADES_parser_writer_round_trip_accepts_crlf_records():
    crlf = (
        "# version=2022\r\n"
        "provID|obsTime|ra|dec|stn|mode|astCat|remarks\r\n"
        '2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2|literal " quote\r\n'
    )

    _contexts, parsed = ADES_string_to_tables(crlf)
    rendered = ADES_to_string(parsed, None, sort=False).replace("\n", "\r\n")
    _contexts, reparsed = ADES_string_to_tables(rendered)

    assert parsed.table.equals(reparsed.table)


def test_ADES_parser_reports_physical_lines_across_blanks():
    ades_string = """
# version=2022

provID|obsTime|ra|dec|stn|mode|astCat|band

2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2
"""

    with pytest.raises(
        ValueError,
        match=r"row at line 6 has 7 fields; expected 8.*line 4",
    ):
        ADES_string_to_tables(ades_string)


@pytest.mark.parametrize(
    ("row", "actual"),
    [
        (
            "2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2",
            7,
        ),
        (
            "2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2|r|extra",
            9,
        ),
    ],
)
def test_ADES_parser_rejects_wrong_field_counts(row, actual):
    ades_string = f"""# version=2022
provID|obsTime|ra|dec|stn|mode|astCat|band
{row}
"""

    with pytest.raises(
        ValueError,
        match=rf"row at line 3 has {actual} fields; expected 8.*line 2",
    ):
        ADES_string_to_tables(ades_string)


@pytest.mark.parametrize(
    "ades_string, line_number",
    [
        (
            """# version=2022
artSat|obsTime|ra|dec|future_field|stn|mode|astCat
1998-067A|2024-01-01T00:00:00.000Z|180.0|0.0|future value|695|CCD|Gaia2
""",
            2,
        ),
        (
            """# version=2022
provID|obsTime|ra|dec|stn|mode|astCat
2024 AB|2024-01-01T00:00:00.000Z|180.0|0.0|695|CCD|Gaia2
artSat|obsTime|ra|dec|stn|mode|astCat
1998-067A|2024-01-01T00:01:00.000Z|181.0|1.0|695|CCD|Gaia2
""",
            4,
        ),
        (
            """# version=2022
artSat|ra|dec|stn|mode|astCat
""",
            2,
        ),
    ],
)
def test_ADES_parser_rejects_unsupported_artsat_identity(ades_string, line_number):
    with pytest.raises(
        ValueError,
        match=rf"unsupported ADES identity column artSat.*line {line_number}",
    ):
        ADES_string_to_tables(ades_string)
