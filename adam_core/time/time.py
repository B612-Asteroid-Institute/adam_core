from __future__ import annotations

import astropy
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

SCALES = {
    "tai",
    "tt",
    "ut1",
    "utc",
    "tdb",
}


class Timestamp(qv.Table):
    # Scale, the rate at which time passes:
    scale = qv.StringAttribute(default="tai")

    # Days since MJD epoch (1858-11-17T00:00:00):
    days = qv.Int64Column()

    # Nanos since start of day:
    nanos = qv.Int64Column()

    def micros(self) -> pa.Int64Array:
        return pc.divide(self.nanos, 1_000)

    def millis(self) -> pa.Int64Array:
        return pc.divide(self.nanos, 1_000_000)

    def seconds(self) -> pa.Int64Array:
        return pc.divide(self.nanos, 1_000_000_000)

    def mjd(self) -> pa.lib.DoubleArray:
        return pc.add(self.days, self.fractional_days())

    def fractional_days(self) -> pa.lib.DoubleArray:
        return pc.divide(self.nanos, 86400 * 1e9)

    def rounded(self, precision: str = "ns") -> Timestamp:
        if precision == "ns":
            return self
        elif precision == "us":
            nanos = pc.multiply(self.micros(), 1_000)
        elif precision == "ms":
            nanos = pc.multiply(self.millis(), 1_000_000)
        elif precision == "s":
            nanos = pc.multiply(self.seconds(), 1_000_000_000)
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        return self.set_column("nanos", nanos)

    def equals(self, other: Timestamp, precision: str = "ns") -> pa.BooleanArray:
        if self.scale != other.scale:
            raise ValueError("Cannot compare timestamps with different scales")
        if len(other) == 1:
            return self.equals_scalar(other.days[0], other.nanos[0], precision)
        else:
            return self.equals_array(other, precision)

    def equals_scalar(
        self, days: int, nanos: int, precision: str = "ns"
    ) -> pa.BooleanArray:
        days_equal = pc.equal(self.days, days)
        if precision == "ns":
            times_equal = pc.equal(self.nanos, nanos)
        elif precision == "us":
            times_equal = pc.equal(self.micros(), nanos // 1_000)
        elif precision == "ms":
            times_equal = pc.equal(self.millis(), nanos // 1_000_000)
        elif precision == "s":
            times_equal = pc.equal(self.seconds(), nanos // 1_000_000_000)
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        return pc.and_(days_equal, times_equal)

    def equals_array(self, other: Timestamp, precision: str = "ns") -> pa.BooleanArray:
        days_equal = pc.equal(self.days, other.days)
        if precision == "ns":
            times_equal = pc.equal(self.nanos, other.nanos)
        elif precision == "us":
            times_equal = pc.equal(self.micros(), other.micros())
        elif precision == "ms":
            times_equal = pc.equal(self.millis(), other.millis())
        elif precision == "s":
            times_equal = pc.equal(self.seconds(), other.seconds())
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        return pc.and_(days_equal, times_equal)

    @classmethod
    def from_astropy(cls, astropy_time: astropy.time.Time) -> Timestamp:
        if astropy_time.scale not in SCALES:
            raise ValueError(f"Unsupported scale: {astropy_time.scale}")

        jd1, jd2 = astropy_time.jd1, astropy_time.jd2
        days, remainder = divmod(jd1 - 2400000.5, 1)
        remainder += jd2

        mask = remainder < 0
        np.add(remainder, 1, where=mask, out=remainder)
        np.subtract(days, 1, where=mask, out=days)

        mask = remainder >= 1
        np.subtract(remainder, 1, where=mask, out=remainder)
        np.add(days, 1, where=mask, out=days)

        nanos = np.round(remainder * 86400 * 1e9)
        return cls.from_kwargs(
            scale=astropy_time.scale,
            days=days.astype(np.int64),
            nanos=nanos,
        )

    def to_astropy(self) -> astropy.time.Time:
        fractional_days = self.fractional_days()
        return astropy.time.Time(
            val=self.days,
            val2=fractional_days,
            format="mjd",
            scale=self.scale,
        )
