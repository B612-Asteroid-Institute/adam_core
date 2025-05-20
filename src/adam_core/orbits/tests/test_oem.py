import os

import pyarrow as pa
import pytest
import quivr as qv
from adam_assist import ASSISTPropagator

from ...coordinates import CartesianCoordinates
from ...coordinates.origin import Origin
from ...time import Timestamp
from ..oem_io import orbit_from_oem, orbit_to_oem
from ..orbits import Orbits

astroforge_optical_path = (
    f"{os.path.dirname(__file__)}/testdata/AstroForgeSC_Optical.oem"
)


def test_orbit_from_oem():
    optical_orbits = orbit_from_oem(astroforge_optical_path)

    optical_orbits_df = optical_orbits.to_dataframe()

    # print(optical_orbits_df)

    orbit_ids = optical_orbits_df["orbit_id"]
    first_segment_ids = [orbit_id for orbit_id in orbit_ids if "seg_0" in orbit_id]
    second_segment_ids = [orbit_id for orbit_id in orbit_ids if "seg_1" in orbit_id]

    assert len(orbit_ids) == 15
    assert len(first_segment_ids) == 10
    assert len(second_segment_ids) == 5

    assert optical_orbits.coordinates.frame == "equatorial"
    assert optical_orbits.coordinates.origin.code[0].as_py() == "EARTH"


def test_orbit_to_oem(tmp_path):
    # generate an orbit

    mjds = [60000.0, 60001.0, 60002.0, 60003.0, 60004.0]
    times = Timestamp.from_mjd(mjds, scale="tdb")
    test_orbit = Orbits.from_kwargs(
        object_id=["test"],
        orbit_id=["test_orbit"],
        coordinates=CartesianCoordinates.from_kwargs(
            time=times[0],
            x=[1.0],
            y=[1.1],
            z=[1.2],
            vx=[1.0],
            vy=[0.1],
            vz=[0.01],
            frame="equatorial",
            origin=Origin.from_kwargs(code=["EARTH"]),
        ),
    )

    oem_path = f"{tmp_path}/test.oem"
    oem_data = orbit_to_oem(test_orbit, oem_path, times, ASSISTPropagator)

    # test load
    orbits_rt = orbit_from_oem(oem_path)

    orbits_rt_df = orbits_rt.to_dataframe()

    assert orbits_rt.coordinates.frame == "equatorial"
    assert orbits_rt.coordinates.origin.code[0].as_py() == "EARTH"
    assert len(orbits_rt_df) == 5
    assert orbits_rt.coordinates.time.scale == "tdb"
