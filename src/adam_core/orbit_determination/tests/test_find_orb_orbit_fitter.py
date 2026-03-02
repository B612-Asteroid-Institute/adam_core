import pickle
import tempfile

import numpy as np
import pyarrow.compute as pc
import pytest
from adam_fo.build import main as build_fo
from adam_fo.config import check_build_exists

from ...coordinates import SphericalCoordinates
from ...coordinates.origin import Origin
from ...observers import Observers
from ...time import Timestamp
from ..evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry
from ..find_orb_orbit_fitter import FindOrbOrbitFitter


@pytest.fixture
def real_data():
    # Actual observations for "2009 JY22"
    obstimes = Timestamp.from_kwargs(
        days=[54952, 54952, 54952, 54952, 54977, 54977, 54977, 54977, 56209, 56209],
        nanos=[
            15930432000000,
            16879968000000,
            17813088000000,
            18760032000000,
            14122080000000,
            14906592000000,
            15680736000000,
            16459200000000,
            31688237000000,
            32917536000000,
        ],
        scale="utc",
    )
    obscodes = ["G96", "G96", "G96", "G96", "G96", "G96", "G96", "G96", "F51", "F51"]
    lon = [
        173.174080,
        173.173500,
        173.173170,
        173.172420,
        174.067330,
        174.068380,
        174.069290,
        174.070500,
        20.118004,
        20.114975,
    ]
    lat = [
        7.762110,
        7.760780,
        7.759580,
        7.758310,
        4.412810,
        4.411390,
        4.410170,
        4.408890,
        8.454381,
        8.453956,
    ]
    obsids = [f"KG0CNl00000055470100001d{i}" for i in range(10)]
    bands = ["V", "V", "V", "V", "V", "V", "V", "V", "w", "w"]
    mags = [21.1, 20.5, 21.2, 21.1, 21.9, 21.2, 21.8, 21.4, 22.0, 21.9]

    coords = SphericalCoordinates.from_kwargs(
        lon=lon,
        lat=lat,
        time=obstimes,
        origin=Origin.from_kwargs(code=["SUN"] * 10),
        frame="equatorial",
    )
    observers = Observers.from_codes(codes=obscodes, times=obstimes)

    photometry = OrbitDeterminationPhotometry.from_kwargs(
        mag=mags,
        band=bands,
    )

    observations = OrbitDeterminationObservations.from_kwargs(
        id=obsids,
        coordinates=coords,
        observers=observers,
        photometry=photometry,
    )
    return observations


def force_adam_fo_install():
    try:
        check_build_exists()
    except RuntimeError:
        build_fo()


def test_pickle():
    results_dir = "/some/path"
    fitter = FindOrbOrbitFitter(fo_result_dir=results_dir)
    assert len(fitter.obscodes) > 0, "Obscodes should be loaded by the constructor"
    saved = pickle.dumps(fitter)

    new_fitter = pickle.loads(saved)
    assert new_fitter.fo_result_dir == results_dir
    assert len(new_fitter.obscodes) == len(fitter.obscodes)


def test_success(real_data):
    force_adam_fo_install()
    observations = real_data
    out_dir = tempfile.TemporaryDirectory()
    fitter = FindOrbOrbitFitter(fo_result_dir=out_dir.name)
    object_id = "2009 JY22"
    fitted_orbit, fitted_members = fitter.initial_fit(object_id, observations)
    assert len(fitted_orbit) == 1
    assert fitted_orbit.object_id[0].as_py() == object_id
    assert len(fitted_members) == len(observations)
    outliers = fitted_members.outlier
    solution = fitted_members.solution
    assert pc.invert(outliers) == solution
    outlier_count = np.sum(outliers.to_pylist())
    # FO rejects the last two of the observations
    assert outlier_count > 0
    assert outlier_count < len(observations)


def test_not_enough_data(real_data):
    force_adam_fo_install()
    observations = real_data[:2]
    out_dir = tempfile.TemporaryDirectory()
    fitter = FindOrbOrbitFitter(fo_result_dir=out_dir.name)
    object_id = "2009 JY22"
    fitted_orbit, fitted_members = fitter.initial_fit(object_id, observations)
    assert len(fitted_orbit) == 0
    assert len(fitted_members) == 0
