import json
import re

import numpy as np
import pyarrow as pa
import pytest

from adam_core import _rust_native

from ...coordinates import (
    CoordinateCovariances,
    LocalFrameCovariances,
    Origin,
    transform_coordinates,
)
from ...coordinates.units import convert_cartesian_covariance_au_to_km
from ...dynamics.propagation import propagate_2body
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from ..oem_io import orbit_from_oem, orbit_to_oem
from ..orbits import Orbits

ORIGINATOR = "TEST ORIGINATOR"
CREATION_DATE = "2026-10-06T00:00:00"
EPOCHS = [
    "2027-03-01T00:00:00",
    "2027-03-15T12:00:00",
    "2027-04-01T00:00:00",
    "2028-03-01T00:00:00",
    "2028-03-15T12:00:00",
    "2028-04-01T00:00:00",
]
ANNEX_B5 = "COMMENT COV_REF_FRAME VNC_ROTATING follows the SANA"


@pytest.fixture
def history() -> Orbits:
    seed = make_real_orbits(1)
    seed = seed.set_column(
        "coordinates", transform_coordinates(seed.coordinates, frame_out="equatorial")
    )
    seed = seed.set_column(
        "object_id", pa.array(["TEST OBJECT"], type=pa.large_string())
    )
    propagated = propagate_2body(seed, Timestamp.from_iso8601(EPOCHS, scale="tdb"))
    factors = np.random.default_rng(1).normal(size=(6, 6, 6)) * 1e-7
    factors[:, :, 3:] *= 1e-2
    return propagated.set_column(
        "coordinates.covariance",
        CoordinateCovariances.from_matrix(np.einsum("nij,nkj->nik", factors, factors)),
    )


def write(history, path, **options) -> str:
    orbit_to_oem(
        history,
        str(path),
        ORIGINATOR,
        version="3.0",
        creation_date=CREATION_DATE,
        **options,
    )
    return open(path).read()


def test_round_trip_version_3(history, tmp_path):
    text = write(history, tmp_path / "states.oem")
    lines = text.split("\n")
    assert lines[0] == "CCSDS_OEM_VERS = 3.0"
    for line in (
        f"CREATION_DATE = {CREATION_DATE}",
        "OBJECT_NAME = TEST OBJECT",
        "CENTER_NAME = SUN",
        "REF_FRAME = ICRF",
        "TIME_SYSTEM = TDB",
        "START_TIME = 2027-03-01T00:00:00.000",
        "STOP_TIME = 2028-04-01T00:00:00.000",
    ):
        assert line in lines
    # State covariance blocks, in REF_FRAME, carry no COV_REF_FRAME line.
    assert text.count("EPOCH = ") == 6 and "COV_REF_FRAME" not in text
    # 16 significant digits by default, the most CCSDS 502.0-B-3 7.5.7 allows.
    numbers = [
        token
        for line in lines
        if line and "=" not in line and not line.isupper()
        for token in (line.split()[1:] if line[:2] == "20" else line.split())
    ]
    assert len(numbers) == 6 * 6 + 6 * 21
    assert all(re.fullmatch(r"-?\d\.\d{15}e[+-]\d{2}", t) for t in numbers)

    loaded = orbit_from_oem(str(tmp_path / "states.oem"))
    assert loaded.coordinates.frame == "equatorial"
    assert loaded.coordinates.origin.code.to_pylist() == ["SUN"] * 6
    assert loaded.coordinates.time.equals(history.coordinates.time)
    np.testing.assert_allclose(
        loaded.coordinates.values, history.coordinates.values, rtol=1e-15
    )
    np.testing.assert_allclose(
        loaded.coordinates.covariance.to_matrix(),
        history.coordinates.covariance.to_matrix(),
        rtol=1e-15,
    )


def test_local_frame_covariance_blocks(history, tmp_path):
    # TNW is in the table 5-4 set, VNC_ROTATING only through annex B5.
    tnw = write(
        history, tmp_path / "tnw.oem", covariance_frame="tnw", table_frames_only=True
    )
    assert tnw.count("COV_REF_FRAME = TNW") == 6 and "SANA" not in tnw
    with pytest.raises(ValueError, match="outside the RSW, RTN, TNW set of table 5-4"):
        write(
            history,
            tmp_path / "x.oem",
            covariance_frame="VNC_ROTATING",
            table_frames_only=True,
        )
    path = str(tmp_path / "vnc.oem")
    text = write(history, path, covariance_frame="VNC_ROTATING")
    assert text.count("COV_REF_FRAME = VNC_ROTATING") == 6 and ANNEX_B5 in text

    # The blocks hold the separate product's matrices in km.
    records = json.loads(_rust_native.oem_parse_kvn(path))["segments"][0]["covariances"]
    written = np.array([record["matrix"] for record in records]).reshape(6, 6, 6)
    expected = convert_cartesian_covariance_au_to_km(
        LocalFrameCovariances.from_orbits(
            history, "VNC_ROTATING"
        ).covariance.to_matrix()
    )
    sigma = np.sqrt(np.diagonal(expected, axis1=1, axis2=2))
    assert np.all(
        np.abs(written - expected) <= 1e-12 * sigma[:, :, None] * sigma[:, None, :]
    )

    # The reader keeps the equatorial states and time scale and ignores the block.
    loaded = orbit_from_oem(path)
    assert loaded.coordinates.frame == "equatorial"
    assert loaded.coordinates.time.scale == "tdb"
    assert loaded.coordinates.time.equals(history.coordinates.time)
    assert loaded.coordinates.covariance.is_all_nan()

    states_only = write(
        history,
        tmp_path / "states.oem",
        covariance_frame="VNC_ROTATING",
        include_covariance=False,
    )
    assert "COVARIANCE_START" not in states_only


def test_input_rules(history, tmp_path):
    path = tmp_path / "x.oem"
    two_ids = pa.array(["A"] * 3 + ["B"] * 3, type=pa.large_string())
    with pytest.raises(AssertionError, match="Only one object_id"):
        write(history.set_column("object_id", two_ids), path)
    two_origins = Origin.from_kwargs(code=["SUN"] * 5 + ["SOLAR_SYSTEM_BARYCENTER"])
    with pytest.raises(ValueError, match="one object about one center per file"):
        write(history.set_column("coordinates.origin", two_origins), path)
    with pytest.raises(ValueError, match="2027-03-01T00:00:00.000 appears twice"):
        write(history.take(pa.array([0, 0, 1, 2, 3, 4])), path)
    nulls = history.set_column("coordinates.covariance", CoordinateCovariances.nulls(6))
    with pytest.raises(ValueError, match="The states carry no covariance."):
        write(nulls, path, covariance_frame="VNC_ROTATING")
    with pytest.raises(ValueError, match="between 1 and 16, got 17"):
        write(history, path, significant_digits=17)
    with pytest.raises(ValueError, match="'comments' must be a list of strings"):
        write(history, path, comments="abc")

    # Epochs off the millisecond grid are moved onto it with a warning.
    time = history.coordinates.time
    nanos = time.nanos.to_numpy(zero_copy_only=False).copy()
    nanos[1] += 123_456
    shifted = history.set_column(
        "coordinates.time",
        Timestamp.from_kwargs(days=time.days, nanos=nanos, scale="tdb"),
    )
    with pytest.warns(UserWarning, match="1 of 6 epochs rounded to the millisecond"):
        write(shifted, path)
    assert orbit_from_oem(str(path)).coordinates.time.equals(time)


def test_nine_by_nine_covariances_write_their_state_block(history, tmp_path):
    # Orbits with non-gravitational parameters carry 9x9 covariances.
    full = np.full((6, 9, 9), 1e-22)
    full[:, :6, :6] = history.coordinates.covariance.to_matrix()
    nongrav = history.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(full)
    )
    for options in ({}, {"covariance_frame": "VNC_ROTATING"}):
        text = write(nongrav, tmp_path / "nongrav.oem", **options)
        assert text.count("EPOCH = ") == 6
        assert text == write(history, tmp_path / "states.oem", **options)


def test_comments_follow_meta_start(history, tmp_path):
    lines = write(
        history,
        tmp_path / "x.oem",
        comments=["first", "second"],
        covariance_frame="VNC_ROTATING",
    ).split("\n")
    start = lines.index("META_START") + 1
    assert lines[start : start + 2] == ["COMMENT first", "COMMENT second"]
    assert lines[start + 2].startswith(ANNEX_B5)
    assert lines[start + 3] == "OBJECT_NAME = TEST OBJECT"


def test_version_3_differs_from_2_only_in_labels(history, tmp_path):
    legacy_path = str(tmp_path / "legacy.oem")
    orbit_to_oem(history, legacy_path, ORIGINATOR, creation_date=CREATION_DATE)
    legacy = open(legacy_path).read()
    expected = legacy.replace("CCSDS_OEM_VERS = 2.0", "CCSDS_OEM_VERS = 3.0").replace(
        "REF_FRAME = EME2000", "REF_FRAME = ICRF"
    )
    assert expected != legacy
    assert write(history, tmp_path / "new.oem", significant_digits=15) == expected
