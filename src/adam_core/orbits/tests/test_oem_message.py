import re

import numpy as np
import pyarrow as pa
import pytest

from ...coordinates import (
    CartesianCoordinates,
    CoordinateCovariances,
    LocalFrameCovariances,
    Origin,
    transform_coordinates,
)
from ...dynamics.propagation import propagate_2body
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from .. import OemHeader, OemSegment, OemSegmentMetadata, OrbitEphemerisMessage, Orbits
from ..oem_io import orbit_from_oem, orbit_to_oem

ORIGINATOR = "TEST ORIGINATOR"
EPOCHS = [
    "2027-03-01T00:00:00",
    "2027-03-15T12:00:00",
    "2027-04-01T00:00:00",
    "2028-03-01T00:00:00",
    "2028-03-15T12:00:00",
    "2028-04-01T00:00:00",
]


@pytest.fixture
def heliocentric_history() -> Orbits:
    """One object, six heliocentric equatorial TDB epochs with covariances."""
    seed = make_real_orbits(1)
    seed = seed.set_column(
        "coordinates", transform_coordinates(seed.coordinates, frame_out="equatorial")
    )
    seed = seed.set_column(
        "object_id", pa.array(["TEST OBJECT"], type=pa.large_string())
    )
    propagated = propagate_2body(seed, Timestamp.from_iso8601(EPOCHS, scale="tdb"))
    rng = np.random.default_rng(1)
    scales = np.array([1e-7, 1e-7, 1e-7, 1e-9, 1e-9, 1e-9])
    factors = rng.normal(size=(len(propagated), 6, 6)) * scales[None, None, :]
    covariances = np.einsum("nij,nkj->nik", factors, factors)
    return propagated.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(covariances)
    )


def _with_metadata(message: OrbitEphemerisMessage, **changes) -> OrbitEphemerisMessage:
    segment = message.segments[0]
    metadata = OemSegmentMetadata(**{**segment.metadata.__dict__, **changes})
    return OrbitEphemerisMessage(
        message.header, (OemSegment(metadata, segment.states),)
    )


def _assert_states_equal(actual: Orbits, expected: Orbits, rtol: float = 1e-12):
    assert actual.coordinates.frame == expected.coordinates.frame
    assert actual.coordinates.time.scale == expected.coordinates.time.scale
    assert actual.coordinates.time.days.equals(expected.coordinates.time.days)
    assert actual.coordinates.time.nanos.equals(expected.coordinates.time.nanos)
    assert (
        actual.coordinates.origin.code.to_pylist()
        == expected.coordinates.origin.code.to_pylist()
    )
    np.testing.assert_allclose(
        actual.coordinates.values, expected.coordinates.values, rtol=rtol
    )


def test_round_trip_heliocentric_icrf(heliocentric_history, tmp_path):
    message = OrbitEphemerisMessage.from_orbits(
        heliocentric_history,
        ORIGINATOR,
        creation_date="2026-10-06T00:00:00",
        message_id="MSG-1",
        comments=["header comment"],
        metadata_comments=["metadata comment"],
        interpolation="LAGRANGE",
        interpolation_degree=7,
    )
    metadata = message.segments[0].metadata
    assert metadata == OemSegmentMetadata(
        object_name="TEST OBJECT",
        object_id="TEST OBJECT",
        center_name="SUN",
        ref_frame="ICRF",
        time_system="TDB",
        start_time="2027-03-01T00:00:00.000",
        stop_time="2028-04-01T00:00:00.000",
        interpolation="LAGRANGE",
        interpolation_degree=7,
        comments=("metadata comment",),
    )

    path = message.write(tmp_path / "heliocentric.oem")
    text = (tmp_path / "heliocentric.oem").read_text()
    lines = text.split("\n")
    assert lines[:5] == [
        "CCSDS_OEM_VERS = 3.0",
        "COMMENT header comment",
        "CREATION_DATE = 2026-10-06T00:00:00",
        f"ORIGINATOR = {ORIGINATOR}",
        "MESSAGE_ID = MSG-1",
    ]
    for line in (
        "COMMENT metadata comment",
        "CENTER_NAME = SUN",
        "REF_FRAME = ICRF",
        "TIME_SYSTEM = TDB",
        "START_TIME = 2027-03-01T00:00:00.000",
        "STOP_TIME = 2028-04-01T00:00:00.000",
        "INTERPOLATION_DEGREE = 7",
    ):
        assert line in lines
    assert len([line for line in lines if re.match(r"^\d{4}-\d{2}-\d{2}T", line)]) == 6
    assert "COVARIANCE_START" not in text

    loaded = OrbitEphemerisMessage.from_kvn(path)
    assert loaded.header == OemHeader(
        originator=ORIGINATOR,
        creation_date="2026-10-06T00:00:00",
        ccsds_oem_vers="3.0",
        message_id="MSG-1",
    )
    segment = loaded.segments[0]
    # The parser drops COMMENT lines, everything else survives.
    assert segment.metadata == OemSegmentMetadata(
        **{**metadata.__dict__, "comments": ()}
    )
    _assert_states_equal(segment.states, heliocentric_history)
    assert segment.states.orbit_id.to_pylist() == ["TEST OBJECT"] * 6
    assert segment.states.coordinates.covariance.is_all_nan()
    assert segment.local_covariances is None
    _assert_states_equal(loaded.to_orbits(), heliocentric_history)


def test_covariance_blocks(heliocentric_history, tmp_path):
    message = OrbitEphemerisMessage.from_orbits(
        heliocentric_history, ORIGINATOR, creation_date="2026-10-06T00:00:00"
    )
    expected = heliocentric_history.coordinates.covariance.to_matrix()

    # In REF_FRAME the label is omitted and the block attaches to the states.
    icrf = message.to_kvn(covariance_frame="ICRF")
    assert "COVARIANCE_START" in icrf and "COV_REF_FRAME" not in icrf
    path = tmp_path / "icrf.oem"
    path.write_text(icrf)
    segment = OrbitEphemerisMessage.from_kvn(path).segments[0]
    np.testing.assert_allclose(
        segment.states.coordinates.covariance.to_matrix(), expected, rtol=1e-12
    )
    assert segment.local_covariances is None

    # TNW is in the OEM covariance frame set, VNC is not.
    tnw = message.to_kvn(covariance_frame="TNW", strict=True)
    assert tnw.count("COV_REF_FRAME = TNW") == 6 and "SANA" not in tnw
    with pytest.raises(ValueError, match="outside the OEM covariance frame set"):
        message.to_kvn(covariance_frame="VNC_ROTATING", strict=True)
    vnc = message.to_kvn(covariance_frame="VNC_ROTATING")
    assert vnc.count("COV_REF_FRAME = VNC_ROTATING") == 6
    assert "COMMENT COV_REF_FRAME VNC_ROTATING is a SANA orbit-relative" in vnc
    with pytest.raises(ValueError, match="Unknown local orbital frame"):
        message.to_kvn(covariance_frame="LVLH")

    # A local frame block reads back as a product, not as a state covariance,
    # and is written back unchanged under the same label.
    path.write_text(vnc)
    segment = OrbitEphemerisMessage.from_kvn(path).segments[0]
    assert segment.states.coordinates.covariance.is_all_nan()
    product = segment.local_covariances
    assert product.frame == "VNC_ROTATING"
    assert product.reference_frame == "equatorial"
    assert product.origin.code.to_pylist() == ["SUN"] * 6
    np.testing.assert_allclose(
        product.to_matrix(),
        LocalFrameCovariances.from_orbits(
            heliocentric_history, "VNC_ROTATING"
        ).to_matrix(),
        rtol=1e-12,
    )
    loaded = OrbitEphemerisMessage.from_kvn(path)
    assert loaded.to_kvn(covariance_frame="VNC_ROTATING") == vnc
    with pytest.raises(ValueError, match="carry no covariance"):
        loaded.to_kvn(covariance_frame="TNW")


def test_label_validation(heliocentric_history, tmp_path):
    ecliptic = heliocentric_history.set_column(
        "coordinates",
        transform_coordinates(heliocentric_history.coordinates, frame_out="ecliptic"),
    )
    with pytest.raises(ValueError, match="Transform to the equatorial frame first"):
        OrbitEphemerisMessage.from_orbits(ecliptic, ORIGINATOR)
    with pytest.raises(ValueError, match="does not name the axes"):
        OrbitEphemerisMessage.from_orbits(
            heliocentric_history, ORIGINATOR, ref_frame="ITRF-93"
        )
    with pytest.raises(ValueError, match="Unsupported OEM TIME_SYSTEM"):
        OrbitEphemerisMessage.from_orbits(
            heliocentric_history, ORIGINATOR, time_system="GPS"
        )
    mixed = heliocentric_history.set_column(
        "coordinates.origin",
        Origin.from_kwargs(
            code=["SUN", "SUN", "SUN", "SOLAR_SYSTEM_BARYCENTER", "SUN", "SUN"]
        ),
    )
    with pytest.raises(ValueError, match="same origin"):
        OrbitEphemerisMessage.from_orbits(mixed, ORIGINATOR)

    # Metadata edited after construction is checked against the states.
    message = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR)
    with pytest.raises(ValueError, match="REF_FRAME"):
        _with_metadata(message, ref_frame="ITRF-93").to_kvn()
    with pytest.raises(ValueError, match="TIME_SYSTEM"):
        _with_metadata(message, time_system="UTC").to_kvn()
    with pytest.raises(ValueError, match="CENTER_NAME"):
        _with_metadata(message, center_name="SOLAR SYSTEM BARYCENTER").to_kvn()

    # Read side: EME2000 maps to equatorial, unmapped labels raise.
    text = OrbitEphemerisMessage.from_orbits(
        heliocentric_history, ORIGINATOR, ref_frame="eme2000"
    ).to_kvn()
    path = tmp_path / "labels.oem"
    path.write_text(text)
    loaded = OrbitEphemerisMessage.from_kvn(path).segments[0]
    assert loaded.metadata.ref_frame == "EME2000"
    assert loaded.states.coordinates.frame == "equatorial"
    path.write_text(text.replace("REF_FRAME = EME2000", "REF_FRAME = TEME"))
    with pytest.raises(ValueError, match="Unsupported OEM REF_FRAME"):
        OrbitEphemerisMessage.from_kvn(path)
    path.write_text(text.replace("TIME_SYSTEM = TDB", "TIME_SYSTEM = GPS"))
    with pytest.raises(ValueError, match="Unsupported OEM TIME_SYSTEM"):
        OrbitEphemerisMessage.from_kvn(path)


def test_time_system_utc(heliocentric_history, tmp_path):
    # TDB epochs on whole seconds do not land on UTC millisecond boundaries.
    message = OrbitEphemerisMessage.from_orbits(
        heliocentric_history, ORIGINATOR, time_system="UTC"
    )
    assert message.segments[0].metadata.time_system == "UTC"
    assert message.segments[0].states.coordinates.time.scale == "utc"
    with pytest.raises(ValueError, match="millisecond boundary"):
        message.to_kvn()

    # States given in UTC on whole seconds write and read as UTC.
    coords = heliocentric_history.coordinates
    utc_history = heliocentric_history.set_column(
        "coordinates",
        CartesianCoordinates.from_kwargs(
            x=coords.x,
            y=coords.y,
            z=coords.z,
            vx=coords.vx,
            vy=coords.vy,
            vz=coords.vz,
            time=Timestamp.from_iso8601(EPOCHS, scale="utc"),
            covariance=coords.covariance,
            origin=coords.origin,
            frame=coords.frame,
        ),
    )
    path = OrbitEphemerisMessage.from_orbits(utc_history, ORIGINATOR).write(
        tmp_path / "utc.oem"
    )
    assert "TIME_SYSTEM = UTC" in (tmp_path / "utc.oem").read_text()
    _assert_states_equal(OrbitEphemerisMessage.from_kvn(path).to_orbits(), utc_history)


def test_state_table_rules(heliocentric_history):
    with pytest.raises(ValueError, match="one object per file"):
        OrbitEphemerisMessage.from_orbits(
            heliocentric_history.set_column(
                "object_id",
                pa.array(["A", "A", "A", "B", "B", "B"], type=pa.large_string()),
            ),
            ORIGINATOR,
        )
    with pytest.raises(ValueError, match="needs an object_id"):
        OrbitEphemerisMessage.from_orbits(
            heliocentric_history.set_column(
                "object_id", pa.nulls(6, type=pa.large_string())
            ),
            ORIGINATOR,
        )
    with pytest.raises(ValueError, match="at least one state"):
        OrbitEphemerisMessage.from_orbits(Orbits.empty(), ORIGINATOR)

    reversed_history = heliocentric_history.take(pa.array(list(range(5, -1, -1))))
    message = OrbitEphemerisMessage.from_orbits(reversed_history, ORIGINATOR)
    _assert_states_equal(message.segments[0].states, heliocentric_history)
    with pytest.raises(ValueError, match="unique"):
        OrbitEphemerisMessage.from_orbits(
            heliocentric_history.take(pa.array([0, 0, 1, 2, 3, 4])), ORIGINATOR
        )

    # Epochs off the millisecond grid raise unless rounding is allowed, and
    # rounding may not merge two epochs.
    time = heliocentric_history.coordinates.time
    days = time.days.to_numpy(zero_copy_only=False).copy()
    nanos = time.nanos.to_numpy(zero_copy_only=False).copy()
    nanos[1] += 123_456
    shifted = heliocentric_history.set_column(
        "coordinates.time", Timestamp.from_kwargs(days=days, nanos=nanos, scale="tdb")
    )
    message = OrbitEphemerisMessage.from_orbits(shifted, ORIGINATOR)
    with pytest.raises(ValueError, match="millisecond boundary"):
        message.to_kvn()
    assert "2027-03-15T12:00:00.000" in message.to_kvn(allow_epoch_rounding=True)
    days[1] = days[0]
    nanos[1] = nanos[0] + 400_000
    colliding = heliocentric_history.set_column(
        "coordinates.time", Timestamp.from_kwargs(days=days, nanos=nanos, scale="tdb")
    )
    with pytest.raises(ValueError, match="same millisecond"):
        OrbitEphemerisMessage.from_orbits(colliding, ORIGINATOR).to_kvn(
            allow_epoch_rounding=True
        )


def test_matches_legacy_writer_byte_for_byte(heliocentric_history, tmp_path):
    legacy_path = str(tmp_path / "legacy.oem")
    orbit_to_oem(heliocentric_history, legacy_path, originator=ORIGINATOR)
    legacy_text = open(legacy_path).read()
    creation_date = re.search(r"^CREATION_DATE = (.*)$", legacy_text, re.M).group(1)
    message = OrbitEphemerisMessage.from_orbits(
        heliocentric_history,
        ORIGINATOR,
        ref_frame="EME2000",
        ccsds_oem_vers="2.0",
        creation_date=creation_date,
    )
    assert message.to_kvn(covariance_frame="EME2000") == legacy_text


def test_legacy_reader_fallback_for_icrf_files(heliocentric_history, tmp_path):
    path = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR).write(
        tmp_path / "icrf.oem", covariance_frame="ICRF"
    )
    orbits = orbit_from_oem(path)
    _assert_states_equal(orbits, heliocentric_history)
    np.testing.assert_allclose(
        orbits.coordinates.covariance.to_matrix(),
        heliocentric_history.coordinates.covariance.to_matrix(),
        rtol=1e-12,
    )
    ids = orbits.orbit_id.to_pylist()
    assert len(set(ids)) == 6
    assert ids[0] == "TEST OBJECT_seg_0_2027-03-01T00:00:00.000"


def test_from_kvn_keeps_unknown_metadata_and_multiple_segments(
    heliocentric_history, tmp_path
):
    first = OrbitEphemerisMessage.from_orbits(
        heliocentric_history[:3], ORIGINATOR, creation_date="2026-10-06T00:00:00"
    ).to_kvn()
    second = OrbitEphemerisMessage.from_orbits(
        heliocentric_history[3:], ORIGINATOR
    ).to_kvn()
    second_segment = (
        second[second.index("META_START") :]
        .replace(
            "TIME_SYSTEM = TDB\n",
            "TIME_SYSTEM = TDB\nUSEABLE_START_TIME = 2028-03-01T00:00:00.000\n",
        )
        .replace(
            "STOP_TIME = 2028-04-01T00:00:00.000\n",
            "STOP_TIME = 2028-04-01T00:00:00.000\nUSER_KEY = user value\n",
        )
    )
    path = tmp_path / "two_segments.oem"
    path.write_text(first + second_segment)

    loaded = OrbitEphemerisMessage.from_kvn(path)
    assert len(loaded.segments) == 2
    metadata = loaded.segments[1].metadata
    assert metadata.useable_start_time == "2028-03-01T00:00:00.000"
    assert metadata.extra == (("USER_KEY", "user value"),)
    assert metadata.as_ordered_items()[-1] == ("USER_KEY", "user value")
    _assert_states_equal(loaded.to_orbits(), heliocentric_history)
    with pytest.raises(NotImplementedError, match="exactly one segment"):
        loaded.to_kvn()
