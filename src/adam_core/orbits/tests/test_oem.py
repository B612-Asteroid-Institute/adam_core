import os

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import quivr as qv
import numpy as np
from adam_assist import ASSISTPropagator

from ...coordinates import CartesianCoordinates
from ...coordinates.origin import Origin
from ...coordinates.covariances import CoordinateCovariances
from ...time import Timestamp
from ..oem_io import orbit_from_oem, orbit_to_oem
from ..orbits import Orbits

astroforge_optical_path = (
    f"{os.path.dirname(__file__)}/testdata/AstroForgeSC_Optical.oem"
)

covariance_example_path = (
    f"{os.path.dirname(__file__)}/testdata/CovarianceExample.oem"
)


def test_orbit_from_oem():
    optical_orbits = orbit_from_oem(astroforge_optical_path)

    first_segment = optical_orbits.apply_mask(pc.match_substring(optical_orbits.orbit_id, "seg_0"))
    second_segment = optical_orbits.apply_mask(pc.match_substring(optical_orbits.orbit_id, "seg_1"))

    assert len(first_segment) == 10
    assert len(second_segment) == 5
    assert len(optical_orbits) == 15

    assert optical_orbits.coordinates.frame == "equatorial" # J2000
    assert optical_orbits.coordinates.origin.code[0].as_py() == "EARTH"


def test_orbit_from_oem_covariance():
    orbits = orbit_from_oem(covariance_example_path)

    assert len(orbits) == 4

    assert not orbits.coordinates.covariance[0].is_all_nan()
    assert orbits.coordinates.covariance[1:].is_all_nan()



def test_orbit_to_oem(tmp_path):
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
            covariance=CoordinateCovariances.from_sigmas(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])),
        ),
    )

    oem_path = f"{tmp_path}/test.oem"
    oem_data = orbit_to_oem(test_orbit, oem_path, times, ASSISTPropagator)

    # test load
    orbits_rt = orbit_from_oem(oem_path)

    orbits_rt_df = orbits_rt.to_dataframe()

    assert orbits_rt.coordinates.frame == "equatorial"
    assert orbits_rt.coordinates.origin.code[0].as_py() == "SUN"
    assert len(orbits_rt_df) == 5
    assert orbits_rt.coordinates.time.scale == "tdb"
    assert not np.any([cov.is_all_nan() for cov in orbits_rt.coordinates.covariance])