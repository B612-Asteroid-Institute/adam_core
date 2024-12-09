from __future__ import annotations

import astropy.time
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

# The Modified Julian Date of the J2000 epoch in TDB scale:
_J2000_TDB_MJD = 51544.5


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

    def jd(self) -> pa.lib.DoubleArray:
        return pc.add(self.mjd(), 2400000.5)

    def et(self) -> pa.lib.DoubleArray:
        """
        Returns the times as ET seconds in a pyarrow array.
        """
        tdb = self.rescale("tdb")
        mjd = tdb.mjd()
        return pc.multiply(pc.subtract(mjd, _J2000_TDB_MJD), 86400)

    def to_numpy(self) -> np.ndarray:
        """
        Returns the times as TDB MJDs in a numpy array.
        """
        return self.rescale("tdb").mjd().to_numpy(False)

    @classmethod
    def from_iso8601(
        cls, iso: pa.lib.StringArray | list[str], scale="utc"
    ) -> Timestamp:
        """
        Create a Timestamp from ISO 8601 strings (for example, '2020-01-02T14:15:16').
        """
        return cls.from_astropy(astropy.time.Time(iso, format="isot", scale=scale))

    @classmethod
    def from_mjd(cls, mjd: pa.lib.DoubleArray, scale: str = "tai") -> Timestamp:
        days = pc.floor(mjd)
        fractional_days = pc.subtract(mjd, days)
        days = pc.cast(days, pa.int64())
        nanos = pc.cast(pc.round(pc.multiply(fractional_days, 86400 * 1e9)), pa.int64())
        return cls.from_kwargs(days=days, nanos=nanos, scale=scale)

    @classmethod
    def from_jd(cls, jd: pa.lib.DoubleArray, scale: str = "tai") -> Timestamp:
        return cls.from_mjd(pc.subtract(jd, 2400000.5), scale)

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
        delta_days, delta_nanos = self.difference_scalar(days, nanos)
        if precision == "ns":
            max_deviation = 0
        elif precision == "us":
            max_deviation = 999
        elif precision == "ms":
            max_deviation = 999_999
        elif precision == "s":
            max_deviation = 999_999_999
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        return _duration_arrays_within_tolerance(delta_days, delta_nanos, max_deviation)

    def equals_array(self, other: Timestamp, precision: str = "ns") -> pa.BooleanArray:
        """
        Compare two Timestamps, returning a BooleanArray indicating
        whether each element is equal.

        The Timestamps must have the same scale, and the same length.
        """
        if self.scale != other.scale:
            raise ValueError("Cannot compare timestamps with different scales")
        if len(self) != len(other):
            raise ValueError("Timestamps must have the same length")

        delta_days, delta_nanos = self.difference(other)
        if precision == "ns":
            max_deviation = 0
        elif precision == "us":
            max_deviation = 999
        elif precision == "ms":
            max_deviation = 999_999
        elif precision == "s":
            max_deviation = 999_999_999
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        return _duration_arrays_within_tolerance(delta_days, delta_nanos, max_deviation)

    def max(self) -> Timestamp:
        """
        Compute the maximum time.

        Returns
        -------
        max_time : `~adam_core.time.Timestamp`
            The maximum time. If there are multiple maximum times,
            one of them is returned.
        """
        # Compute the maximum day
        max_day = pc.max(self.days)
        days_mask = pc.equal(self.days, max_day)

        # Compute the maximum nanos for the maximum day
        max_nanos = pc.max(self.nanos.filter(days_mask))
        nanos_mask = pc.equal(self.nanos, max_nanos)

        return self.apply_mask(pc.and_(days_mask, nanos_mask))[0]

    def min(self) -> Timestamp:
        """
        Compute the minimum time.

        Returns
        -------
        min_time : `~adam_core.time.Timestamp`
            The minimum time. If there are multiple minimum times,
            one of them is returned.
        """
        # Compute the minimum day
        min_day = pc.min(self.days)
        days_mask = pc.equal(self.days, min_day)

        # Compute the minimum nanos for the minimum day
        min_nanos = pc.min(self.nanos.filter(days_mask))
        nanos_mask = pc.equal(self.nanos, min_nanos)

        return self.apply_mask(pc.and_(days_mask, nanos_mask))[0]

    @classmethod
    def from_astropy(cls, astropy_time: astropy.time.Time) -> Timestamp:
        """Convert an astropy time to a quivr timestamp.

        This is a lossy conversion, since astropy uses floating point
        to represent times, while quivr uses integers.

        The astropy time must use a scale supported by quivr. The
        supported scales are "tai", "tt", "ut1", "utc", and "tdb".

        """
        if astropy_time.scale not in SCALES:
            raise ValueError(f"Unsupported scale: {astropy_time.scale}")
        if astropy_time.isscalar:
            return cls._from_astropy_scalar(astropy_time)

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

    @classmethod
    def _from_astropy_scalar(cls, astropy_time: astropy.time.Time) -> Timestamp:
        """Astropy times can be scalar-valued, which requires
        separate handling because the numpy array functions won't
        work.
        """
        jd1, jd2 = astropy_time.jd1, astropy_time.jd2
        days, remainder = divmod(jd1 - 2400000.5, 1)
        remainder += jd2

        if remainder < 0:
            remainder += 1
            days -= 1

        if remainder >= 1:
            remainder -= 1
            days += 1

        nanos = round(remainder * 86400 * 1e9)
        return cls.from_kwargs(
            scale=astropy_time.scale,
            days=[int(days)],
            nanos=[nanos],
        )

    def to_astropy(self) -> astropy.time.Time:
        """
        Convert the timestamp to an astropy time.
        """
        fractional_days = self.fractional_days()
        return astropy.time.Time(
            val=self.days,
            val2=fractional_days,
            format="mjd",
            scale=self.scale,
        )

    def add_nanos(
        self, nanos: pa.lib.Int64Array | int, check_range: bool = True
    ) -> Timestamp:
        """
        Add nanoseconds to the timestamp. Negative nanoseconds are
        allowed.

        Parameters
        ----------
        nanos : The nanoseconds to add. Can be a scalar or an array of
            the same length as the timestamp. Must be in the range [-86400e9, 86400e9).
        check_range : If True, check that the nanoseconds are in the
            range [-86400e9, 86400e9). If False, the caller is
            responsible for ensuring that the nanoseconds are in the
            correct range.
        """
        if check_range:
            if isinstance(nanos, int):
                if not -86400e9 <= nanos < 86400e9:
                    raise ValueError("Nanoseconds out of range")
            else:
                if not pc.all(
                    pc.and_(pc.greater_equal(nanos, -86400e9), pc.less(nanos, 86400e9))
                ).as_py():
                    raise ValueError("Nanoseconds out of range")

        nanos = pc.add_checked(self.nanos, nanos)
        overflows = pc.greater_equal(nanos, 86400 * 1e9)
        underflows = pc.less(nanos, 0)

        mask = pa.StructArray.from_arrays(
            [overflows, underflows], names=["overflows", "underflows"]
        )
        nanos = pc.case_when(
            mask,
            pc.subtract(nanos, int(86400 * 1e9)),
            pc.add(nanos, int(86400 * 1e9)),
            nanos,
        )

        days = pc.case_when(
            mask,
            pc.add(self.days, 1),
            pc.subtract(self.days, 1),
            self.days,
        )
        v1 = self.set_column("days", days)
        v2 = v1.set_column("nanos", nanos)
        return v2

    def add_seconds(
        self, seconds: pa.lib.Int64Array | int | pa.DoubleArray | float
    ) -> Timestamp:
        """
        Add seconds to the timestamp. Negative seconds are supported.

        Parameters
        ----------
        seconds : The seconds to add. Can be a scalar or an array of
            the same length as the timestamp. Must be in the range [-86400, 86400).

        See Also
        --------
        add_nanos : Add nanoseconds to the timestamp. This method includes
            a 'check_range' parameter that allows the caller to disable range
            checking for performance reasons.
        """
        nanos = pc.cast(pc.round(pc.multiply(seconds, 1_000_000_000)), pa.int64())
        return self.add_nanos(nanos)

    def add_millis(self, millis: pa.lib.Int64Array | int) -> Timestamp:
        """
        Add milliseconds to the timestamp. Negative milliseconds are
        supported.

        Parameters
        ----------
        millis : The milliseconds to add. Can be a scalar or an array of
            the same length as the timestamp. Must be in the range [-86400e3, 86400e3).

        See Also
        --------
        add_nanos : Add nanoseconds to the timestamp. This method includes
            a 'check_range' parameter that allows the caller to disable range
            checking for performance reasons.
        """
        nanos = pc.cast(pc.round(pc.multiply(millis, 1_000_000)), pa.int64())
        return self.add_nanos(nanos)

    def add_micros(self, micros: pa.lib.Int64Array | int) -> Timestamp:
        """
        Add microseconds to the timestamp. Negative microseconds are
        supported.

        Parameters
        ----------
        micros : The microseconds to add. Can be a scalar or an array of
            the same length as the timestamp. Must be in the range [-86400e6, 86400e6).

        See Also
        --------
        add_nanos : Add nanoseconds to the timestamp. This method includes
            a 'check_range' parameter that allows the caller to disable range
            checking for performance reasons.
        """
        nanos = pc.cast(pc.round(pc.multiply(micros, 1_000)), pa.int64())
        return self.add_nanos(nanos)

    def add_days(self, days: pa.lib.Int64Array | int) -> Timestamp:
        """Add days to the timestamp.

        Parameters
        ----------
        days : The days to add. Can be a scalar or an array of the
            same length as the timestamp. Use negative values to
            subtract days.

        """
        return self.set_column("days", pc.add(self.days, days))

    def add_fractional_days(
        self, fractional_days: pa.lib.DoubleArray | float
    ) -> Timestamp:
        """
        Add fractional days to the timestamp.

        Parameters
        ----------
        fractional_days : The fractional days to add. Can be a scalar
            or an array of the same length as the timestamp. Use
            negative values to subtract fractional days.
        """
        day_part = pc.floor(fractional_days)
        nano_part = pc.subtract(fractional_days, day_part)

        days = pc.cast(day_part, pa.int64())
        nanos = pc.cast(
            pc.multiply(nano_part, 86400 * 1e9),
            options=pc.CastOptions(target_type=pa.int64(), allow_float_truncate=True),
        )
        return self.add_days(days).add_nanos(nanos)

    def difference_scalar(
        self, days: int, nanos: int
    ) -> tuple[pa.Int64Array, pa.Int64Array]:
        """
        Compute the difference between this timestamp and a scalar
        timestamp.

        The difference is computed as (self - scalar). The result is
        presented as a tuple of (days, nanos). The nanos value is
        always non-negative, in the range [0, 86400e9).

        Parameters
        ----------
        days : The days of the scalar timestamp.
        nanos : The nanoseconds of the scalar timestamp.

        Returns
        -------
        days : The difference in days. This value can be negative.
        nanos : The difference in nanoseconds. This value is always
            non-negative, in the range [0, 86400e9).

        Examples
        --------
        >>> from adam_core.time import Timestamp
        >>> ts = Timestamp.from_kwargs(days=[0, 1, 2], nanos=[200, 0, 100])
        >>> have_days, have_nanos = ts.difference_scalar(1, 100)
        >>> have_days.to_pylist()
        [-1, -1, 1]
        >>> have_nanos.to_pylist()
        [100, 86399999999900, 0]

        """
        days1 = pc.subtract(self.days, days)
        nanos1 = pc.subtract(self.nanos, nanos)
        overflows = pc.greater_equal(nanos1, 86400 * 1e9)
        underflows = pc.less(nanos1, 0)
        mask = pa.StructArray.from_arrays(
            [overflows, underflows], names=["overflows", "underflows"]
        )
        nanos2 = pc.case_when(
            mask,
            pc.subtract(nanos1, int(86400 * 1e9)),
            pc.add(nanos1, int(86400 * 1e9)),
            nanos1,
        )
        days2 = pc.case_when(
            mask,
            pc.add(days1, 1),
            pc.subtract(days1, 1),
            days1,
        )
        return days2, nanos2

    def difference(self, other: Timestamp) -> tuple[pa.Int64Array, pa.Int64Array]:
        """
        Compute the element-wise difference between this timestamp and another.
        """
        if self.scale != other.scale:
            raise ValueError(
                "Cannot compute difference between timestamps with different scales"
            )
        days1 = pc.subtract(self.days, other.days)
        nanos1 = pc.subtract(self.nanos, other.nanos)

        overflows = pc.greater_equal(nanos1, 86400 * 1e9)
        underflows = pc.less(nanos1, 0)
        mask = pa.StructArray.from_arrays(
            [overflows, underflows], names=["overflows", "underflows"]
        )
        nanos2 = pc.case_when(
            mask,
            pc.subtract(nanos1, int(86400 * 1e9)),
            pc.add(nanos1, int(86400 * 1e9)),
            nanos1,
        )
        days2 = pc.case_when(
            mask,
            pc.add(days1, 1),
            pc.subtract(days1, 1),
            days1,
        )
        return days2, nanos2

    def unique(self) -> Timestamp:
        """Return a new Timestamp table containing only the unique
        elements from self. Order is not necessarily preserved.

        """
        uniqued = self.table.group_by(["days", "nanos"]).aggregate([])
        uniqued = uniqued.replace_schema_metadata(self.table.schema.metadata)
        return Timestamp.from_pyarrow(uniqued)

    def rescale(self, new_scale: str) -> Timestamp:
        if self.scale == new_scale:
            return self
        elif new_scale == "tai":
            return Timestamp.from_astropy(self.to_astropy().tai)
        elif new_scale == "utc":
            return Timestamp.from_astropy(self.to_astropy().utc)
        elif new_scale == "tt":
            return Timestamp.from_astropy(self.to_astropy().tt)
        elif new_scale == "ut1":
            return Timestamp.from_astropy(self.to_astropy().ut1)
        elif new_scale == "tdb":
            return Timestamp.from_astropy(self.to_astropy().tdb)
        else:
            raise ValueError("Unknown scale: {}".format(new_scale))

    def link(
        self, other: Timestamp, precision: str = "ns"
    ) -> qv.MultiKeyLinkage[Timestamp, Timestamp]:
        """
        Link this Timestamp to another. The default precision is nanoseconds, but if
        other precisions are desired then both this class and the other Timestamp will
        be rounded to the desired precision.

        If the timescales are different, the other Timestamp will be rescaled to
        this Timestamp's timescale.

        Parameters
        ----------
        other : The Timestamp to link to.
        precision : The precision to use when linking. The default is 'ns'.

        Returns
        -------
        linkage : A MultiKeyLinkage object that can be used to join the two Timestamps.
        """
        if self.scale != other.scale:
            other = other.rescale(self.scale)

        rounded = self.rounded(precision)
        other_rounded = other.rounded(precision)

        left_keys = {"days": rounded.days, "nanos": rounded.nanos}
        right_keys = {"days": other_rounded.days, "nanos": other_rounded.nanos}
        return qv.MultiKeyLinkage(self, other, left_keys, right_keys)


def _duration_arrays_within_tolerance(
    delta_days: pa.Int64Array, delta_nanos: pa.Int64Array, max_nanos_deviation: int
) -> pa.BooleanArray:
    """Return a boolean array indicating whether the delta_days and delta_nanos
    arrays are within the specified tolerance.

    The max_nanos_deviation should be the maximum number of
    nanoseconds that the the two arrays can deviate to still be
    considered 'within tolerance'.
    """
    if max_nanos_deviation == 0:
        return pc.and_(pc.equal(delta_days, 0), pc.equal(delta_nanos, 0))

    cond1 = pc.and_(
        pc.equal(delta_days, 0), pc.less(pc.abs(delta_nanos), max_nanos_deviation)
    )
    cond2 = pc.and_(
        pc.equal(delta_days, -1),
        pc.greater_equal(pc.abs(delta_nanos), 86400 * 1e9 - max_nanos_deviation),
    )
    return pc.or_(cond1, cond2)
