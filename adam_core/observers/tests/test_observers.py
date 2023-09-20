import numpy as np
import quivr as qv
from astropy.time import Time

from ..observers import Observers


def test_observers_sort_by():
    # Test that Observers.sort_by works for both
    # ascending and descending order and different
    # order of columns.
    observers = qv.concatenate(
        [
            Observers.from_code(
                "X05", Time([59001, 59002, 59003], scale="tdb", format="mjd")
            ),
            Observers.from_code(
                "I41", Time([59003, 59004, 59005], scale="tdb", format="mjd")
            ),
        ]
    )
    observers_sorted = observers.sort_by(by=["time", "code"], ascending=True)
    np.testing.assert_equal(
        observers_sorted.code.to_numpy(zero_copy_only=False),
        np.array(["X05", "X05", "I41", "X05", "I41", "I41"]),
    )
    np.testing.assert_almost_equal(
        observers_sorted.coordinates.time.mjd(),
        np.array([59001, 59002, 59003, 59003, 59004, 59005]),
    )

    observers_sorted = observers.sort_by(by=["time", "code"], ascending=False)
    np.testing.assert_equal(
        observers_sorted.code.to_numpy(zero_copy_only=False),
        np.array(["I41", "I41", "X05", "I41", "X05", "X05"]),
    )
    np.testing.assert_almost_equal(
        observers_sorted.coordinates.time.mjd(),
        np.array([59005, 59004, 59003, 59003, 59002, 59001]),
    )

    observers_sorted = observers.sort_by(by=["code", "time"], ascending=True)
    np.testing.assert_equal(
        observers_sorted.code.to_numpy(zero_copy_only=False),
        np.array(["I41", "I41", "I41", "X05", "X05", "X05"]),
    )
    np.testing.assert_almost_equal(
        observers_sorted.coordinates.time.mjd(),
        np.array([59003, 59004, 59005, 59001, 59002, 59003]),
    )
