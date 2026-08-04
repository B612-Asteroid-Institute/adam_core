import pyarrow as pa
import pyarrow.compute as pc

from ...time import Timestamp
from ..utils import calculate_observing_night


def test_calculate_observing_night() -> None:
    # Rubin Observatory is UTC-7
    # Reasonable observating times would be +- 12 hours from local midnight
    # or 7:00 to 17:00 UTC

    # ZTF is UTC-8
    # Reasonable observating times would be +- 12 hours from local midnight
    # or 8:00 to 16:00 UTC

    # M22 is UTC+2
    # Reasonable observating times would be +- 12 hours from local midnight
    # or 22:00 to 10:00 UTC

    # 000 is UTC
    # Reasonable observating times would be +- 12 hours from local midnight
    # or 0:00 to 12:00 UTC

    codes = pa.array(
        [
            "X05",
            "X05",
            "X05",
            "X05",
            "X05",
            "I41",
            "I41",
            "I41",
            "I41",
            "I41",
            "M22",
            "M22",
            "M22",
            "M22",
            "M22",
            "000",
            "000",
            "000",
            "000",
            "000",
        ]
    )
    times_utc = Timestamp.from_mjd(
        [
            # Rubin Observatory
            59000 + 7 / 24 - 12 / 24,
            59000 + 7 / 24 - 6 / 24,
            59000 + 7 / 24,
            59000 + 7 / 24 + 6 / 24,
            59000 + 7 / 24 + 12 / 24,
            # ZTF
            59000 + 8 / 24 - 12 / 24,
            59000 + 8 / 24 - 4 / 24,
            59000 + 8 / 24,
            59000 + 8 / 24 + 4 / 24,
            59000 + 8 / 24 + 12 / 24,
            # M22
            59000 - 2 / 24 - 12 / 24,
            59000 - 2 / 24 - 6 / 24,
            59000 - 2 / 24,
            59000 - 2 / 24 + 6 / 24,
            59000 - 2 / 24 + 12 / 24,
            # 000
            59000 - 12 / 24,
            59000 - 6 / 24,
            59000,
            59000 + 6 / 24,
            59000 + 12 / 24,
        ],
        scale="utc",
    )

    observing_night = calculate_observing_night(codes, times_utc)
    desired = pa.array(
        [
            58999,
            58999,
            58999,
            58999,
            59000,
            58999,
            58999,
            58999,
            58999,
            59000,
            58999,
            58999,
            58999,
            58999,
            59000,
            58999,
            58999,
            58999,
            58999,
            59000,
        ]
    )
    assert pc.all(pc.equal(observing_night, desired)).as_py()
