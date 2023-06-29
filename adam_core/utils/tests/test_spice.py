import numpy as np
import spiceypy as sp
from astropy.time import Time
from naif_leapseconds import leapseconds

from ..spice import _jd_tdb_to_et


def test__jd_tdb_to_et():
    # Test that _jd_tdb_to_et returns the same values as SPICE's str2et
    sp.furnsh(leapseconds)

    times = Time(np.arange(40000, 70000, 5), scale="tdb", format="mjd")
    jd_tdb = times.tdb.jd

    et_actual = _jd_tdb_to_et(jd_tdb)
    et_expected = np.array([sp.str2et(f"JD {i:.16f} TDB") for i in jd_tdb])

    np.testing.assert_equal(et_actual, et_expected)
