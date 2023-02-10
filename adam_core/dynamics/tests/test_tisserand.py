import numpy as np
import numpy.testing as npt
import pytest

from ..tisserand import calc_tisserand_parameter

# --- Tests last updated: 2023-02-09

# From JPL's SBDB (2023-02-01) https://ssd.jpl.nasa.gov/sbdb.cgi
# a, e, i
DAMOCLES = {
    "aei": (11.87052898293103, 0.8659079806381217, 61.6840796497527),
    "T_jupiter": 1.155,
}
CERES = {
    "aei": (2.767181743149466, 0.07881745101960996, 10.58634326912728),
    "T_jupiter": 3.310,
}
# 3D/Biela
BIELA = {
    "aei": (3.53465808340135, 0.751299, 13.2164),
    "T_jupiter": 2.531,
}


def test_calc_tisserand_parameter_damocloids():
    Tp = calc_tisserand_parameter(*DAMOCLES["aei"], third_body="jupiter")

    npt.assert_allclose(np.round(Tp, 3), DAMOCLES["T_jupiter"])
    return


def test_calc_tisserand_parameter_asteroids():
    Tp = calc_tisserand_parameter(*CERES["aei"], third_body="jupiter")

    npt.assert_allclose(np.round(Tp, 3), CERES["T_jupiter"])
    return


def test_calc_tisserand_parameter_jupiter_family_comets():
    Tp = calc_tisserand_parameter(*BIELA["aei"], third_body="jupiter")

    npt.assert_allclose(np.round(Tp, 3), BIELA["T_jupiter"])
    return


def test_calc_tisserand_parameter_raise():
    # Not a valid planet name
    with pytest.raises(ValueError):
        calc_tisserand_parameter(*CERES["aei"], third_body="")
