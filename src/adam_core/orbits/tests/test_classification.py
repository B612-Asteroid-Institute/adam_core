from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from ...coordinates.cometary import CometaryCoordinates
from ...coordinates.keplerian import KeplerianCoordinates
from ...coordinates.origin import Origin
from ...time.time import Timestamp
from ..classification import calc_orbit_class
from ..orbits import Orbits

dynamical_classes_dir = files("adam_core.orbits.tests.testdata")


def sbdb_df_to_cometary(df: pd.DataFrame) -> CometaryCoordinates:
    return CometaryCoordinates.from_kwargs(
        q=df["q"],
        e=df["e"],
        i=df["i"],
        raan=df["om"],
        ap=df["w"],
        tp=df["tp"] - 2400000.5,
        time=Timestamp.from_mjd(df["epoch_mjd"].values, scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN" for i in range(len(df))]),
        frame="ecliptic",
    )


def sbdb_df_to_keplerian(df: pd.DataFrame) -> KeplerianCoordinates:
    return KeplerianCoordinates.from_kwargs(
        a=df["a"],
        e=df["e"],
        i=df["i"],
        raan=df["om"],
        ap=df["w"],
        M=df["ma"],
        time=Timestamp.from_mjd(df["epoch_mjd"].values, scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN" for i in range(len(df))]),
        frame="ecliptic",
    )


def sbdb_df_to_orbits(df: pd.DataFrame) -> Orbits:
    return Orbits.from_kwargs(
        object_id=df["full_name"],
        coordinates=sbdb_df_to_cometary(df).to_cartesian(),
    )


def ieo_sample():
    df = pd.read_csv(dynamical_classes_dir / "ieo.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("IEO", len(df))


def ate_sample():
    df = pd.read_csv(dynamical_classes_dir / "ate.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("ATE", len(df))


def apo_sample():
    df = pd.read_csv(dynamical_classes_dir / "apo.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("APO", len(df))


def amo_sample():
    df = pd.read_csv(dynamical_classes_dir / "amo.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("AMO", len(df))


def mca_sample():
    df = pd.read_csv(dynamical_classes_dir / "mca.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("MCA", len(df))


def imb_sample():
    df = pd.read_csv(dynamical_classes_dir / "imb.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("IMB", len(df))


def mba_sample():
    df = pd.read_csv(dynamical_classes_dir / "mba.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("MBA", len(df))


def omb_sample():
    df = pd.read_csv(dynamical_classes_dir / "omb.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("OMB", len(df))


def tjn_sample():
    df = pd.read_csv(dynamical_classes_dir / "tjn.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("TJN", len(df))


def cen_sample():
    df = pd.read_csv(dynamical_classes_dir / "cen.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("CEN", len(df))


def tno_sample():
    df = pd.read_csv(dynamical_classes_dir / "tno.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("TNO", len(df))


def hya_sample():
    df = pd.read_csv(dynamical_classes_dir / "hya.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("HYA", len(df))


def ast_sample():
    df = pd.read_csv(dynamical_classes_dir / "ast.csv")
    cometary = sbdb_df_to_cometary(df)
    keplerian = sbdb_df_to_keplerian(df)
    orbits = sbdb_df_to_orbits(df)
    return orbits, cometary, keplerian, np.repeat("AST", len(df))


SAMPLES = [
    ieo_sample(),
    ate_sample(),
    apo_sample(),
    amo_sample(),
    mca_sample(),
    imb_sample(),
    mba_sample(),
    omb_sample(),
    tjn_sample(),
    cen_sample(),
    tno_sample(),
    hya_sample(),
    ast_sample(),
]


@pytest.mark.parametrize(
    "sample",
    SAMPLES,
)
def test_calc_orbit_class_cometary(sample):
    _, cometary, _, expected_classes = sample
    classes = calc_orbit_class(cometary)
    np.testing.assert_array_equal(classes, expected_classes)


@pytest.mark.parametrize(
    "sample",
    SAMPLES,
)
def test_calc_orbit_class_keplerian(sample):
    _, _, keplerian, expected_classes = sample
    classes = calc_orbit_class(keplerian)
    np.testing.assert_array_equal(classes, expected_classes)


@pytest.mark.parametrize(
    "sample",
    SAMPLES,
)
def test_calc_orbit_class_orbits(sample):
    orbits, _, _, expected_classes = sample
    classes = orbits.dynamical_class()
    np.testing.assert_array_equal(classes, expected_classes)
