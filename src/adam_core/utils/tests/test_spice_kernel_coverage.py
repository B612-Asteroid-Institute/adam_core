"""Per-epoch kernel resolution across the Earth orientation kernels.

The three Earth orientation kernels adam-core furnishes have windows that
never merge into one. An arc reaching from the 1970s to the 2030s is covered
by their union and by no one file, which used to fail as a batch even though
every epoch in it resolved on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from ...observers import Observers
from ...time import Timestamp
from ..spice import _jd_tdb_to_et, setup_SPICE
from ..spice_backend import NotCovered, get_backend

# 1978 only ever lives in the historical kernel and 2030 only ever lives in the
# prediction kernel, whatever release of each is installed.
THREE_KERNEL_ARC = [
    "1978-12-23T00:00:00",
    "2020-06-01T00:00:00",
    "2030-01-01T00:00:00",
]
# Before 1962-01-20, where the historical kernel starts in every release.
UNCOVERED = "1950-03-04T05:06:07"


def test_observers_from_codes_spans_three_kernels():
    times = Timestamp.from_iso8601(THREE_KERNEL_ARC, scale="utc")
    values = Observers.from_codes(["T05"] * 3, times).coordinates.values
    assert values.shape == (3, 6)
    assert np.isfinite(values).all()

    # Splitting the batch across kernels must not change a single value.
    singles = np.vstack(
        [
            Observers.from_codes(
                ["T05"], Timestamp.from_iso8601([isot], scale="utc")
            ).coordinates.values
            for isot in THREE_KERNEL_ARC
        ]
    )
    np.testing.assert_array_equal(values, singles)


def test_an_epoch_no_kernel_covers_is_named_in_the_error():
    setup_SPICE()
    times = Timestamp.from_iso8601(
        [UNCOVERED, THREE_KERNEL_ARC[1]], scale="utc"
    ).rescale("tdb")
    ets = np.ascontiguousarray(
        _jd_tdb_to_et(times.jd().to_numpy(zero_copy_only=False)), dtype=np.float64
    )

    with pytest.raises(NotCovered) as excinfo:
        get_backend().pxform_batch("ITRF93", "ECLIPJ2000", ets)

    message = str(excinfo.value)
    # MJD 33344 is the uncovered 1950 epoch; 59001 is the covered 2020 one.
    assert "MJD 33344" in message, message
    assert "59001" not in message, message
