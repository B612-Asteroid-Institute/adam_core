from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from astropy.time import Time

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


def test_calculate_observing_night_dst_parity_and_native_timing() -> None:
    from adam_core import _rust_native as _rn
    from adam_core.observers.observers import OBSERVATORY_PARALLAX_COEFFICIENTS

    values = [
        "2024-03-10T09:59:59.000",
        "2024-03-10T10:00:00.000",
        "2024-11-03T08:59:59.000",
        "2024-11-03T09:00:00.000",
        "2024-06-01T00:00:00.000",
    ]
    codes = pa.array(["I41", "I41", "I41", "I41", "000"])
    times = Timestamp.from_iso8601(values, scale="utc")

    have = calculate_observing_night(codes, times)

    want = np.empty(len(times), dtype=np.int64)
    timezone_names = np.empty(len(times), dtype=object)
    for code in pc.unique(codes):
        coefficients = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", code)
        timezone_name = str(coefficients.timezone()[0])
        timezone = ZoneInfo(timezone_name)
        mask = pc.equal(codes, code).to_numpy(zero_copy_only=False)
        timezone_names[mask] = timezone_name
        selected = times.apply_mask(pa.array(mask))
        legacy = Time(
            [
                value + value.astimezone(timezone).utcoffset() - timedelta(hours=12)
                for value in selected.to_astropy().datetime
            ],
            format="datetime",
            scale="utc",
        )
        want[mask] = legacy.mjd.astype(int)

    assert have.to_pylist() == want.tolist()

    samples = _rn.benchmark_observing_nights_numpy(
        times.days.to_numpy(zero_copy_only=False),
        times.nanos.to_numpy(zero_copy_only=False),
        timezone_names.tolist(),
        1,
        2,
        1,
    )
    assert len(samples) == 2
    assert all(sample[0] >= 0.0 for sample in samples)
