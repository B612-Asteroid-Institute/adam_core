from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from ...coordinates.cometary import CometaryCoordinates
from ...coordinates.origin import Origin
from ...time.time import Timestamp
from ..classification import calc_orbit_class

dynamical_classes_dir = files("adam_core.orbits.tests.testdata")


def sbdb_df_to_cometary(df):
    return CometaryCoordinates.from_kwargs(
        q=df["q"],
        e=df["e"],
        i=df["i"],
        raan=df["om"],
        ap=df["w"],
        tp=df["tp"],
        time=Timestamp.from_mjd(df["epoch_mjd"].values, scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN" for i in range(len(df))]),
        frame="ecliptic",
    )


def ate_sample():
    df = pd.read_csv(dynamical_classes_dir / "ate.csv")
    return sbdb_df_to_cometary(df), np.repeat("ATE", len(df))


def apo_sample():
    df = pd.read_csv(dynamical_classes_dir / "apo.csv")
    return sbdb_df_to_cometary(df), np.repeat("APO", len(df))


def amo_sample():
    df = pd.read_csv(dynamical_classes_dir / "amo.csv")
    return sbdb_df_to_cometary(df), np.repeat("AMO", len(df))


def mca_sample():
    df = pd.read_csv(dynamical_classes_dir / "mca.csv")
    return sbdb_df_to_cometary(df), np.repeat("MCA", len(df))


def imb_sample():
    df = pd.read_csv(dynamical_classes_dir / "imb.csv")
    return sbdb_df_to_cometary(df), np.repeat("IMB", len(df))


def mba_sample():
    df = pd.read_csv(dynamical_classes_dir / "mba.csv")
    return sbdb_df_to_cometary(df), np.repeat("MBA", len(df))


def omb_sample():
    df = pd.read_csv(dynamical_classes_dir / "omb.csv")
    return sbdb_df_to_cometary(df), np.repeat("OMB", len(df))


def tjn_sample():
    df = pd.read_csv(dynamical_classes_dir / "tjn.csv")
    return sbdb_df_to_cometary(df), np.repeat("TJN", len(df))


def cen_sample():
    df = pd.read_csv(dynamical_classes_dir / "cen.csv")
    return sbdb_df_to_cometary(df), np.repeat("CEN", len(df))


def tno_sample():
    df = pd.read_csv(dynamical_classes_dir / "tno.csv")
    return sbdb_df_to_cometary(df), np.repeat("TNO", len(df))


def hya_sample():
    df = pd.read_csv(dynamical_classes_dir / "hya.csv")
    return sbdb_df_to_cometary(df), np.repeat("HYA", len(df))


def ast_sample():
    df = pd.read_csv(dynamical_classes_dir / "ast.csv")
    return sbdb_df_to_cometary(df), np.repeat("AST", len(df))


@pytest.mark.parametrize(
    "sample",
    [
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
    ],
)
def test_calc_orbit_class(sample):
    orbits, expected_classes = sample
    classes = calc_orbit_class(orbits)
    np.testing.assert_array_equal(classes, expected_classes)
