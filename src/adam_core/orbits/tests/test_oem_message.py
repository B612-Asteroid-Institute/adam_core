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
from ..oem import OEM_COVARIANCE_FRAMES
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


def _synthetic_covariances(n: int, seed: int = 1) -> CoordinateCovariances:
    rng = np.random.default_rng(seed)
    scales = np.array([1e-7, 1e-7, 1e-7, 1e-9, 1e-9, 1e-9])
    matrices = []
    for _ in range(n):
        factor = rng.normal(size=(6, 6)) * scales[None, :]
        matrices.append(factor @ factor.T)
    return CoordinateCovariances.from_matrix(np.array(matrices))


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
    times = Timestamp.from_iso8601(EPOCHS, scale="tdb")
    propagated = propagate_2body(seed, times)
    return propagated.set_column(
        "coordinates.covariance", _synthetic_covariances(len(propagated))
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


def test_from_orbits_defaults_and_rendered_metadata(heliocentric_history):
    message = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR)
    assert len(message.segments) == 1
    metadata = message.segments[0].metadata
    assert metadata.object_name == "TEST OBJECT"
    assert metadata.object_id == "TEST OBJECT"
    assert metadata.center_name == "SUN"
    assert metadata.ref_frame == "ICRF"
    assert metadata.time_system == "TDB"
    assert metadata.start_time == "2027-03-01T00:00:00.000"
    assert metadata.stop_time == "2028-04-01T00:00:00.000"
    assert message.header.ccsds_oem_vers == "3.0"

    text = message.to_kvn()
    lines = text.split("\n")
    assert lines[0] == "CCSDS_OEM_VERS = 3.0"
    assert "CENTER_NAME = SUN" in lines
    assert "REF_FRAME = ICRF" in lines
    assert "TIME_SYSTEM = TDB" in lines
    assert "START_TIME = 2027-03-01T00:00:00.000" in lines
    assert "STOP_TIME = 2028-04-01T00:00:00.000" in lines
    assert f"ORIGINATOR = {ORIGINATOR}" in lines
    assert re.search(
        r"^CREATION_DATE = \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", text, re.M
    )
    state_lines = [line for line in lines if re.match(r"^\d{4}-\d{2}-\d{2}T", line)]
    assert len(state_lines) == len(EPOCHS)
    assert "COVARIANCE_START" not in text


def test_round_trip_multi_epoch_heliocentric_icrf(heliocentric_history, tmp_path):
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
    path = message.write(tmp_path / "heliocentric.oem")
    text = (tmp_path / "heliocentric.oem").read_text()
    assert "COMMENT header comment" in text
    assert "COMMENT metadata comment" in text
    assert "INTERPOLATION_DEGREE = 7" in text

    loaded = OrbitEphemerisMessage.from_kvn(path)
    assert loaded.header == OemHeader(
        originator=ORIGINATOR,
        creation_date="2026-10-06T00:00:00",
        ccsds_oem_vers="3.0",
        message_id="MSG-1",
    )
    segment = loaded.segments[0]
    expected_metadata = message.segments[0].metadata
    # Comments are dropped by the parser, everything else survives.
    assert segment.metadata == OemSegmentMetadata(
        **{**expected_metadata.__dict__, "comments": ()}
    )
    _assert_states_equal(segment.states, heliocentric_history)
    assert segment.states.coordinates.covariance.is_all_nan()
    assert segment.local_covariances is None
    assert segment.states.orbit_id.to_pylist() == ["TEST OBJECT"] * len(EPOCHS)
    _assert_states_equal(loaded.to_orbits(), heliocentric_history)


def test_ref_frame_covariance_round_trip(heliocentric_history, tmp_path):
    message = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR)
    text = message.to_kvn(covariance_frame="ICRF")
    assert "COVARIANCE_START" in text
    assert "COV_REF_FRAME" not in text  # same as REF_FRAME, so omitted
    path = tmp_path / "cov.oem"
    path.write_text(text)
    segment = OrbitEphemerisMessage.from_kvn(path).segments[0]
    np.testing.assert_allclose(
        segment.states.coordinates.covariance.to_matrix(),
        heliocentric_history.coordinates.covariance.to_matrix(),
        rtol=1e-12,
    )
    assert segment.local_covariances is None


def test_local_frame_covariance_in_the_file(heliocentric_history, tmp_path):
    message = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR)

    # TNW is in the OEM covariance frame set, so strict accepts it.
    tnw_text = message.to_kvn(covariance_frame="TNW", strict=True)
    assert tnw_text.count("COV_REF_FRAME = TNW") == len(EPOCHS)
    assert "SANA" not in tnw_text

    # VNC is not, so strict refuses and the default records a comment.
    with pytest.raises(ValueError, match="outside the OEM covariance frame set"):
        message.to_kvn(covariance_frame="VNC_ROTATING", strict=True)
    vnc_text = message.to_kvn(covariance_frame="VNC_ROTATING")
    assert vnc_text.count("COV_REF_FRAME = VNC_ROTATING") == len(EPOCHS)
    assert "COMMENT COV_REF_FRAME VNC_ROTATING is a SANA orbit-relative" in vnc_text
    for frame in OEM_COVARIANCE_FRAMES:
        assert frame in vnc_text

    path = tmp_path / "vnc.oem"
    path.write_text(vnc_text)
    segment = OrbitEphemerisMessage.from_kvn(path).segments[0]
    assert segment.states.coordinates.covariance.is_all_nan()
    product = segment.local_covariances
    assert product is not None
    assert product.frame == "VNC_ROTATING"
    assert product.reference_frame == "equatorial"
    assert product.origin.code.to_pylist() == ["SUN"] * len(EPOCHS)
    expected = LocalFrameCovariances.from_orbits(heliocentric_history, "VNC_ROTATING")
    np.testing.assert_allclose(product.to_matrix(), expected.to_matrix(), rtol=1e-12)
    assert product.time.days.equals(expected.time.days)

    with pytest.raises(ValueError, match="Unknown local orbital frame"):
        message.to_kvn(covariance_frame="LVLH")


def test_covariance_requested_without_covariance_raises(heliocentric_history):
    orbits = heliocentric_history.set_column(
        "coordinates.covariance", CoordinateCovariances.nulls(len(heliocentric_history))
    )
    message = OrbitEphemerisMessage.from_orbits(orbits, ORIGINATOR)
    with pytest.raises(ValueError, match="carry no covariance"):
        message.to_kvn(covariance_frame="ICRF")
    assert "COVARIANCE_START" not in message.to_kvn()


def test_ref_frame_override(heliocentric_history, tmp_path):
    message = OrbitEphemerisMessage.from_orbits(
        heliocentric_history, ORIGINATOR, ref_frame="eme2000"
    )
    assert message.segments[0].metadata.ref_frame == "EME2000"
    path = message.write(tmp_path / "eme.oem")
    loaded = OrbitEphemerisMessage.from_kvn(path)
    assert loaded.segments[0].metadata.ref_frame == "EME2000"
    assert loaded.segments[0].states.coordinates.frame == "equatorial"

    with pytest.raises(ValueError, match="does not name the axes"):
        OrbitEphemerisMessage.from_orbits(
            heliocentric_history, ORIGINATOR, ref_frame="ITRF-93"
        )


def test_ecliptic_states_raise(heliocentric_history):
    ecliptic = heliocentric_history.set_column(
        "coordinates",
        transform_coordinates(heliocentric_history.coordinates, frame_out="ecliptic"),
    )
    with pytest.raises(ValueError, match="Transform to the equatorial frame first"):
        OrbitEphemerisMessage.from_orbits(ecliptic, ORIGINATOR)


def test_sub_millisecond_epochs_raise_unless_rounding_is_allowed(
    heliocentric_history,
):
    time = heliocentric_history.coordinates.time
    nanos = time.nanos.to_numpy(zero_copy_only=False).copy()
    nanos[1] += 123_456
    shifted = heliocentric_history.set_column(
        "coordinates.time",
        Timestamp.from_kwargs(days=time.days, nanos=nanos, scale=time.scale),
    )
    message = OrbitEphemerisMessage.from_orbits(shifted, ORIGINATOR)
    with pytest.raises(ValueError, match="millisecond boundary"):
        message.to_kvn()
    text = message.to_kvn(allow_epoch_rounding=True)
    assert "2027-03-15T12:00:00.000" in text


def test_time_system_override(heliocentric_history, tmp_path):
    # TDB epochs on whole seconds do not land on UTC millisecond boundaries.
    message = OrbitEphemerisMessage.from_orbits(
        heliocentric_history, ORIGINATOR, time_system="UTC"
    )
    assert message.segments[0].metadata.time_system == "UTC"
    assert message.segments[0].states.coordinates.time.scale == "utc"
    with pytest.raises(ValueError, match="millisecond boundary"):
        message.to_kvn()

    # States given in UTC on whole seconds write and read as UTC.
    utc_times = Timestamp.from_iso8601(EPOCHS, scale="utc")
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
            time=utc_times,
            covariance=coords.covariance,
            origin=coords.origin,
            frame=coords.frame,
        ),
    )
    utc_message = OrbitEphemerisMessage.from_orbits(utc_history, ORIGINATOR)
    path = utc_message.write(tmp_path / "utc.oem")
    assert "TIME_SYSTEM = UTC" in (tmp_path / "utc.oem").read_text()
    loaded = OrbitEphemerisMessage.from_kvn(path)
    _assert_states_equal(loaded.segments[0].states, utc_history)

    with pytest.raises(ValueError, match="Unsupported OEM TIME_SYSTEM"):
        OrbitEphemerisMessage.from_orbits(
            heliocentric_history, ORIGINATOR, time_system="GPS"
        )


def test_object_id_rules(heliocentric_history):
    two_objects = heliocentric_history.set_column(
        "object_id",
        pa.array(["A", "A", "A", "B", "B", "B"], type=pa.large_string()),
    )
    with pytest.raises(ValueError, match="one object per file"):
        OrbitEphemerisMessage.from_orbits(two_objects, ORIGINATOR)

    missing = heliocentric_history.set_column(
        "object_id", pa.nulls(len(heliocentric_history), type=pa.large_string())
    )
    with pytest.raises(ValueError, match="needs an object_id"):
        OrbitEphemerisMessage.from_orbits(missing, ORIGINATOR)

    with pytest.raises(ValueError, match="at least one state"):
        OrbitEphemerisMessage.from_orbits(Orbits.empty(), ORIGINATOR)


def test_epochs_are_sorted_and_must_be_unique(heliocentric_history):
    reversed_history = heliocentric_history.take(
        pa.array(list(reversed(range(len(heliocentric_history)))))
    )
    message = OrbitEphemerisMessage.from_orbits(reversed_history, ORIGINATOR)
    _assert_states_equal(message.segments[0].states, heliocentric_history)

    duplicated = heliocentric_history.take(pa.array([0, 0, 1, 2, 3, 4]))
    with pytest.raises(ValueError, match="unique"):
        OrbitEphemerisMessage.from_orbits(duplicated, ORIGINATOR)


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


def test_legacy_reader_reads_icrf_files(heliocentric_history, tmp_path):
    message = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR)
    path = message.write(tmp_path / "icrf.oem", covariance_frame="ICRF")
    orbits = orbit_from_oem(path)
    _assert_states_equal(orbits, heliocentric_history)
    np.testing.assert_allclose(
        orbits.coordinates.covariance.to_matrix(),
        heliocentric_history.coordinates.covariance.to_matrix(),
        rtol=1e-12,
    )


def test_from_kvn_keeps_unknown_metadata_and_multiple_segments(
    heliocentric_history, tmp_path
):
    first = OrbitEphemerisMessage.from_orbits(
        heliocentric_history[:3], ORIGINATOR, creation_date="2026-10-06T00:00:00"
    ).to_kvn()
    second = OrbitEphemerisMessage.from_orbits(
        heliocentric_history[3:], ORIGINATOR
    ).to_kvn()
    second_segment = second[second.index("META_START") :]
    second_segment = second_segment.replace(
        "TIME_SYSTEM = TDB\n",
        "TIME_SYSTEM = TDB\nUSEABLE_START_TIME = 2028-03-01T00:00:00.000\n",
    ).replace(
        "STOP_TIME = 2028-04-01T00:00:00.000\n",
        "STOP_TIME = 2028-04-01T00:00:00.000\nUSER_KEY = user value\n",
        1,
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


def test_from_kvn_rejects_unmapped_labels(heliocentric_history, tmp_path):
    text = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR).to_kvn()
    path = tmp_path / "bad.oem"
    path.write_text(text.replace("REF_FRAME = ICRF", "REF_FRAME = TEME"))
    with pytest.raises(ValueError, match="Unsupported OEM REF_FRAME"):
        OrbitEphemerisMessage.from_kvn(path)
    path.write_text(text.replace("TIME_SYSTEM = TDB", "TIME_SYSTEM = GPS"))
    with pytest.raises(ValueError, match="Unsupported OEM TIME_SYSTEM"):
        OrbitEphemerisMessage.from_kvn(path)


def test_segment_metadata_mismatch_is_caught(heliocentric_history):
    message = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR)
    segment = message.segments[0]
    wrong_frame = OemSegment(
        metadata=OemSegmentMetadata(
            **{**segment.metadata.__dict__, "ref_frame": "ITRF-93"}
        ),
        states=segment.states,
    )
    with pytest.raises(ValueError, match="REF_FRAME"):
        OrbitEphemerisMessage(message.header, (wrong_frame,)).to_kvn()
    wrong_scale = OemSegment(
        metadata=OemSegmentMetadata(
            **{**segment.metadata.__dict__, "time_system": "UTC"}
        ),
        states=segment.states,
    )
    with pytest.raises(ValueError, match="TIME_SYSTEM"):
        OrbitEphemerisMessage(message.header, (wrong_scale,)).to_kvn()


def test_origin_must_be_single_valued_and_match_center_name(heliocentric_history):
    mixed = heliocentric_history.set_column(
        "coordinates.origin",
        Origin.from_kwargs(
            code=["SUN", "SUN", "SUN", "SOLAR_SYSTEM_BARYCENTER", "SUN", "SUN"]
        ),
    )
    with pytest.raises(ValueError, match="same origin"):
        OrbitEphemerisMessage.from_orbits(mixed, ORIGINATOR)

    message = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR)
    segment = message.segments[0]
    wrong_center = OemSegment(
        metadata=OemSegmentMetadata(
            **{**segment.metadata.__dict__, "center_name": "SOLAR SYSTEM BARYCENTER"}
        ),
        states=segment.states,
    )
    with pytest.raises(ValueError, match="CENTER_NAME"):
        OrbitEphemerisMessage(message.header, (wrong_center,)).to_kvn()


def test_read_local_covariance_block_is_written_back(heliocentric_history, tmp_path):
    message = OrbitEphemerisMessage.from_orbits(
        heliocentric_history, ORIGINATOR, creation_date="2026-10-06T00:00:00"
    )
    original = message.to_kvn(covariance_frame="TNW")
    path = tmp_path / "tnw.oem"
    path.write_text(original)
    loaded = OrbitEphemerisMessage.from_kvn(path)
    assert loaded.segments[0].states.coordinates.covariance.is_all_nan()
    assert loaded.segments[0].local_covariances.frame == "TNW"
    # Writing the same label writes the stored block back, byte for byte.
    assert loaded.to_kvn(covariance_frame="TNW") == original
    # Another local frame cannot be derived without a state covariance.
    with pytest.raises(ValueError, match="carry no covariance"):
        loaded.to_kvn(covariance_frame="VNC_ROTATING")


def test_legacy_reader_keeps_per_state_orbit_ids(heliocentric_history, tmp_path):
    path = OrbitEphemerisMessage.from_orbits(heliocentric_history, ORIGINATOR).write(
        tmp_path / "icrf_ids.oem"
    )
    orbits = orbit_from_oem(path)
    ids = orbits.orbit_id.to_pylist()
    assert len(set(ids)) == len(ids)
    assert ids[0] == "TEST OBJECT_seg_0_2027-03-01T00:00:00.000"


def test_rounding_cannot_collapse_two_epochs(heliocentric_history):
    time = heliocentric_history.coordinates.time
    days = time.days.to_numpy(zero_copy_only=False).copy()
    nanos = time.nanos.to_numpy(zero_copy_only=False).copy()
    days[1] = days[0]
    nanos[1] = nanos[0] + 400_000  # same millisecond as row 0
    close = heliocentric_history.set_column(
        "coordinates.time",
        Timestamp.from_kwargs(days=days, nanos=nanos, scale=time.scale),
    )
    message = OrbitEphemerisMessage.from_orbits(close, ORIGINATOR)
    with pytest.raises(ValueError, match="same millisecond"):
        message.to_kvn(allow_epoch_rounding=True)
