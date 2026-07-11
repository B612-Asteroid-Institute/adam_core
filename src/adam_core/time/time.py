"""
Time utilities and a fast, integer-backed timestamp representation.

This module defines `Timestamp`, a quivr table storing times as integer MJD days plus
integer nanoseconds within the day, tagged with a time scale (`tai`, `tt`, `utc`, `tdb`, `ut1`).

`Timestamp.rescale()` implements fast conversions between supported scales. For UT1, we
delegate to astropy because UT1 requires IERS tables and interpolation.

### Empirical validation summary (2026-01)

We validated the impact of the `rescale()` implementation choice (default implementation vs
an astropy-backed baseline) on an Asteroid Institute “real world” pipeline:

- Rubin X05 MPC observation timestamps (UTC) + RA/Dec astrometry
- SBDB orbit queries (for known objects)
- Ephemeris generation with `adam_assist.ASSISTPropagator`
- Residuals computed via `adam_core.coordinates.residuals.Residuals` on equatorial spherical
  coordinates (RA/Dec as lon/lat)
- Two runs: default `Timestamp.rescale` vs a global override where `Timestamp.rescale`
  dispatches to `Timestamp.rescale_astropy` for all internal conversions

Results (largest run: 500 objects, max 100 observations/object; 43,919 ephemeris points):

- Predicted position delta (default vs astropy baseline): max ~0.001 mas.
- Residuals to Rubin astrometry (arcsec): p50 ~0.031, p90 ~0.088 for both methods.
- Residual delta (default - baseline): ~0 mas at p50/p90 (no measurable change).

We also confirmed these UTC observation times exhibit a default-vs-astropy UTC→TDB offset
in the expected tens-of-microseconds range (~14.6–21.8 µs for a sampled subset).
"""

from __future__ import annotations

import hashlib
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

# Time constants
_SECONDS_IN_DAY = 86_400
_NANOS_IN_DAY = 86_400_000_000_000  # int, exact
_NANOS_IN_DAY_F64 = float(_NANOS_IN_DAY)  # for pyarrow float division
_TAI_TT_NS_CORRECTION = 32_184_000_000  # TT = TAI + 32.184s

# Scales served by the Rust rescale backend (ut1 requires IERS tables and is
# delegated to astropy; other scales are unsupported, matching legacy).
_RUST_RESCALE_SCALES = {"tai", "tt", "utc", "tdb"}


def _int64_values(
    values: pa.lib.Int64Array | pa.ChunkedArray | np.ndarray | int, length: int
) -> np.ndarray:
    """Normalize a scalar or array delta argument to a contiguous int64 array."""
    if isinstance(values, pa.Scalar):
        values = values.as_py()
    if isinstance(values, (int, np.integer)):
        return np.full(length, int(values), dtype=np.int64)
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        values = values.to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(values, dtype=np.int64)


def _float64_values(
    values: pa.lib.DoubleArray | pa.ChunkedArray | np.ndarray | float, length: int
) -> np.ndarray:
    """Normalize a scalar or array argument to a contiguous float64 array."""
    if isinstance(values, pa.Scalar):
        values = values.as_py()
    if isinstance(values, (int, float, np.floating, np.integer)):
        return np.full(length, float(values), dtype=np.float64)
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        values = values.to_numpy(zero_copy_only=False)
    return np.ascontiguousarray(values, dtype=np.float64)


def _has_nulls(*values) -> bool:
    """True when any pyarrow argument carries nulls. The Rust fast paths
    operate on dense arrays; null-carrying inputs take the retained pyarrow
    expressions so legacy null propagation is preserved."""
    for value in values:
        if isinstance(value, (pa.Array, pa.ChunkedArray)) and value.null_count:
            return True
        if isinstance(value, pa.Scalar) and not value.is_valid:
            return True
    return False


class Timestamp(qv.Table):
    # Scale, the rate at which time passes:
    scale = qv.StringAttribute(default="tai")

    # Days since MJD epoch (1858-11-17T00:00:00):
    days = qv.Int64Column()

    # Nanos since start of day:
    nanos = qv.Int64Column()

    def micros(self) -> pa.Int64Array:
        return self._unit_floor(1_000)

    def millis(self) -> pa.Int64Array:
        return self._unit_floor(1_000_000)

    def seconds(self) -> pa.Int64Array:
        return self._unit_floor(1_000_000_000)

    def _unit_floor(self, divisor: int) -> pa.Int64Array:
        if _has_nulls(self.nanos):
            return pc.divide(self.nanos, divisor)
        from adam_core import _rust_native as _rn

        return pa.array(
            _rn.timestamp_unit_floor_numpy(
                self.nanos.to_numpy(zero_copy_only=False), divisor
            ),
            type=pa.int64(),
        )

    def key(self, *, scale: str | None = "tdb") -> np.ndarray:
        """
        Return an int64 key for each timestamp: (days * NANOS_IN_DAY + nanos).

        This is useful for fast grouping/uniquing and as a stable cache key when paired
        with a specific time scale.
        """
        if len(self) == 0:
            return np.empty(0, dtype=np.int64)

        from adam_core import _rust_native as _rn

        # One Rust crossing owns the optional rescale plus key assembly.
        return np.asarray(
            _rn.timestamp_key_numpy(
                self.days.to_numpy(zero_copy_only=False),
                self.nanos.to_numpy(zero_copy_only=False),
                self.scale,
                scale,
            ),
            dtype=np.int64,
        )

    def signature(self, *, scale: str | None = "tdb") -> tuple[int, int, int, int]:
        """
        Return a cheap signature for this Timestamp array.

        The signature is (n, first_key, last_key, sum_mod) where keys are produced by `key()`.
        """
        if len(self) == 0:
            return 0, 0, 0, 0

        from adam_core import _rust_native as _rn

        n, first, last, sum_mod = _rn.timestamp_signature_numpy(
            self.days.to_numpy(zero_copy_only=False),
            self.nanos.to_numpy(zero_copy_only=False),
            self.scale,
            scale,
        )
        return int(n), int(first), int(last), int(sum_mod)

    def cache_digest(self, *, scale: str | None = "tdb") -> int:
        """
        Return an order-sensitive 64-bit digest of timestamp keys.

        This is intended for cache keys where row alignment matters.
        """
        if len(self) == 0:
            return 0

        key = np.asarray(self.key(scale=scale), dtype="<i8")
        payload = np.ascontiguousarray(key).view(np.uint8)
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def mjd(self) -> pa.lib.DoubleArray:
        from adam_core import _rust_native as _rn

        if _has_nulls(self.days, self.nanos):
            return pc.add(self.days, self.fractional_days())
        return pa.array(
            _rn.timestamp_mjd(
                self.days.to_numpy(zero_copy_only=False),
                self.nanos.to_numpy(zero_copy_only=False),
            ),
            type=pa.float64(),
        )

    def jd(self) -> pa.lib.DoubleArray:
        if _has_nulls(self.days, self.nanos):
            return pc.add(self.mjd(), 2400000.5)
        from adam_core import _rust_native as _rn

        return pa.array(
            _rn.timestamp_jd_numpy(
                self.days.to_numpy(zero_copy_only=False),
                self.nanos.to_numpy(zero_copy_only=False),
            ),
            type=pa.float64(),
        )

    def et(self) -> pa.lib.DoubleArray:
        """
        Returns the times as ET seconds in a pyarrow array.
        """
        if self.scale in _RUST_RESCALE_SCALES and not _has_nulls(self.days, self.nanos):
            from adam_core import _rust_native as _rn

            # Fused rescale-to-TDB plus ET conversion in one crossing.
            return pa.array(
                _rn.timestamp_et_numpy(
                    self.days.to_numpy(zero_copy_only=False),
                    self.nanos.to_numpy(zero_copy_only=False),
                    self.scale,
                ),
                type=pa.float64(),
            )
        tdb = self.rescale("tdb")
        mjd = tdb.mjd()
        return pc.multiply(pc.subtract(mjd, _J2000_TDB_MJD), _SECONDS_IN_DAY)

    def to_numpy(self) -> np.ndarray:
        """
        Returns the times as TDB MJDs in a numpy array.
        """
        return self.rescale("tdb").mjd().to_numpy(False)

    def to_iso8601(self) -> pa.lib.StringArray:
        """
        Returns the times as ISO 8601 strings in a pyarrow array.
        """
        if len(self) == 0:
            return pa.array([], type=pa.string())
        return pa.array(self.to_astropy().isot, type=pa.string())

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
        from adam_core import _rust_native as _rn

        if isinstance(mjd, (pa.Array, pa.ChunkedArray)) and mjd.null_count:
            # Null-propagation compatibility path (legacy pyarrow kernels).
            days = pc.floor(mjd)
            fractional_days = pc.subtract(mjd, days)
            days = pc.cast(days, pa.int64())
            nanos = pc.cast(
                pc.round(pc.multiply(fractional_days, _NANOS_IN_DAY)), pa.int64()
            )
            return cls.from_kwargs(days=days, nanos=nanos, scale=scale)
        if isinstance(mjd, (pa.Array, pa.ChunkedArray)):
            values = mjd.to_numpy(zero_copy_only=False)
        else:
            values = mjd
        days, nanos = _rn.timestamp_from_mjd(
            np.ascontiguousarray(values, dtype=np.float64)
        )
        return cls.from_kwargs(days=days, nanos=nanos, scale=scale)

    @classmethod
    def from_jd(cls, jd: pa.lib.DoubleArray, scale: str = "tai") -> Timestamp:
        return cls.from_mjd(pc.subtract(jd, 2400000.5), scale)

    @classmethod
    def from_et(cls, et: pa.lib.DoubleArray, scale: str = "tdb") -> Timestamp:
        return cls.from_mjd(pc.divide(et, 86400), scale)

    def fractional_days(self) -> pa.lib.DoubleArray:
        # IMPORTANT: pyarrow integer / integer division truncates, so ensure float divisor.
        return pc.divide(self.nanos, _NANOS_IN_DAY_F64)

    def rounded(self, precision: str = "ns") -> Timestamp:
        if precision == "ns":
            return self
        elif precision == "us":
            divisor = 1_000
        elif precision == "ms":
            divisor = 1_000_000
        elif precision == "s":
            divisor = 1_000_000_000
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        if _has_nulls(self.nanos):
            nanos = pc.multiply(pc.divide(self.nanos, divisor), divisor)
        else:
            from adam_core import _rust_native as _rn

            nanos = pa.array(
                _rn.timestamp_rounded_nanos_numpy(
                    self.nanos.to_numpy(zero_copy_only=False), divisor
                ),
                type=pa.int64(),
            )
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
        max_deviation = _precision_max_deviation(precision)
        from adam_core import _rust_native as _rn

        return pa.array(
            _rn.timestamp_equals_numpy(
                self.days.to_numpy(zero_copy_only=False),
                self.nanos.to_numpy(zero_copy_only=False),
                _int64_values(days, 1),
                _int64_values(nanos, 1),
                max_deviation,
            ),
            type=pa.bool_(),
        )

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

        max_deviation = _precision_max_deviation(precision)
        from adam_core import _rust_native as _rn

        return pa.array(
            _rn.timestamp_equals_numpy(
                self.days.to_numpy(zero_copy_only=False),
                self.nanos.to_numpy(zero_copy_only=False),
                other.days.to_numpy(zero_copy_only=False),
                other.nanos.to_numpy(zero_copy_only=False),
                max_deviation,
            ),
            type=pa.bool_(),
        )

    def max(self) -> Timestamp:
        """
        Compute the maximum time.

        Returns
        -------
        max_time : `~adam_core.time.Timestamp`
            The maximum time. If there are multiple maximum times,
            one of them is returned.
        """
        from adam_core import _rust_native as _rn

        index = _rn.timestamp_extremum_index_numpy(
            self.days.to_numpy(zero_copy_only=False),
            self.nanos.to_numpy(zero_copy_only=False),
            True,
        )
        return self[index : index + 1]

    def min(self) -> Timestamp:
        """
        Compute the minimum time.

        Returns
        -------
        min_time : `~adam_core.time.Timestamp`
            The minimum time. If there are multiple minimum times,
            one of them is returned.
        """
        from adam_core import _rust_native as _rn

        index = _rn.timestamp_extremum_index_numpy(
            self.days.to_numpy(zero_copy_only=False),
            self.nanos.to_numpy(zero_copy_only=False),
            False,
        )
        return self[index : index + 1]

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

        # Floating point arithmetic can cause nanos to be 86400e9
        # instead of 0, so we need to handle this case
        nanos_full_day = nanos == 86400 * 1e9
        days = np.where(nanos_full_day, days + 1, days).astype(np.int64)
        nanos = np.where(nanos_full_day, 0, nanos)

        return cls.from_kwargs(
            scale=astropy_time.scale,
            days=days,
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
                if not -_NANOS_IN_DAY <= nanos < _NANOS_IN_DAY:
                    raise ValueError("Nanoseconds out of range")
            else:
                if not pc.all(
                    pc.and_(
                        pc.greater_equal(nanos, -_NANOS_IN_DAY),
                        pc.less(nanos, _NANOS_IN_DAY),
                    )
                ).as_py():
                    raise ValueError("Nanoseconds out of range")

        from adam_core import _rust_native as _rn

        days_out, nanos_out = _rn.timestamp_add_nanos(
            self.days.to_numpy(zero_copy_only=False),
            self.nanos.to_numpy(zero_copy_only=False),
            _int64_values(nanos, len(self)),
            False,  # range validated above with legacy pyarrow semantics
        )
        v1 = self.set_column("days", pa.array(days_out))
        v2 = v1.set_column("nanos", pa.array(nanos_out))
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
        from adam_core import _rust_native as _rn

        days_out, _nanos_out = _rn.timestamp_add_days(
            self.days.to_numpy(zero_copy_only=False),
            self.nanos.to_numpy(zero_copy_only=False),
            _int64_values(days, len(self)),
        )
        return self.set_column("days", pa.array(days_out))

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
        from adam_core import _rust_native as _rn

        days_out, nanos_out = _rn.timestamp_add_fractional_days(
            self.days.to_numpy(zero_copy_only=False),
            self.nanos.to_numpy(zero_copy_only=False),
            _float64_values(fractional_days, len(self)),
        )
        v1 = self.set_column("days", pa.array(days_out))
        v2 = v1.set_column("nanos", pa.array(nanos_out))
        return v2

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
        from adam_core import _rust_native as _rn

        days_out, nanos_out = _rn.timestamp_difference(
            self.days.to_numpy(zero_copy_only=False),
            self.nanos.to_numpy(zero_copy_only=False),
            _int64_values(days, len(self)),
            _int64_values(nanos, len(self)),
        )
        return pa.array(days_out), pa.array(nanos_out)

    def difference(self, other: Timestamp) -> tuple[pa.Int64Array, pa.Int64Array]:
        """
        Compute the element-wise difference between this timestamp and another.
        """
        if self.scale != other.scale:
            raise ValueError(
                "Cannot compute difference between timestamps with different scales"
            )
        from adam_core import _rust_native as _rn

        days_out, nanos_out = _rn.timestamp_difference(
            self.days.to_numpy(zero_copy_only=False),
            self.nanos.to_numpy(zero_copy_only=False),
            other.days.to_numpy(zero_copy_only=False),
            other.nanos.to_numpy(zero_copy_only=False),
        )
        return pa.array(days_out), pa.array(nanos_out)

    def unique(self) -> Timestamp:
        """Return a new Timestamp table containing only the unique
        elements from self. Order is not necessarily preserved.

        """
        from adam_core import _rust_native as _rn

        days, nanos = _rn.timestamp_unique_numpy(
            self.days.to_numpy(zero_copy_only=False),
            self.nanos.to_numpy(zero_copy_only=False),
        )
        return Timestamp.from_kwargs(days=days, nanos=nanos, scale=self.scale)

    def rescale_astropy(self, new_scale: str) -> Timestamp:

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

    def rescale(self, new_scale: str) -> Timestamp:
        """
        Convert this `Timestamp` to `new_scale`.

        Notes
        -----
        - For `ut1` conversions, we delegate to `rescale_astropy()` because UT1 requires IERS
          tables and interpolation that astropy manages.
        - For conversions involving TDB, this implementation uses a different (non-Earth-location-
          dependent) approximation than astropy/ERFA, and differences of 10-30 microseconds
          versus astropy can occur. Round-trip conversions within this implementation are designed
          to be stable (see tests).

        See also
        --------
        `src/adam_core/time/tests/test_time.py`: correctness + benchmark tests for rescale.
        """
        if self.scale == new_scale:
            return self

        # ut1 requires delta interpolated from tables published by the IERS.
        # ERFA functions expect the delta to be provided. Astropy does the
        # file downloading and interpolating, so stick with it here.
        if self.scale == "ut1" or new_scale == "ut1":
            return self.rescale_astropy(new_scale)

        # The tai/tt/utc/tdb conversions run in the Rust backend (bead
        # personal-cmy.25), which ports the same ERFA leap-second and
        # TT<->TDB correction policies; both sides are gated bit-exactly by
        # the frozen fixture in
        # migration/artifacts/time_scale_rescale_fixture_2026-05-15.json.
        if self.scale in _RUST_RESCALE_SCALES and new_scale in _RUST_RESCALE_SCALES:
            from adam_core import _rust_native as _rn

            days, nanos = _rn.timestamp_rescale(
                self.days.to_numpy(zero_copy_only=False),
                self.nanos.to_numpy(zero_copy_only=False),
                self.scale,
                new_scale,
            )
            return Timestamp.from_kwargs(days=days, nanos=nanos, scale=new_scale)

        raise ValueError(
            "Rescale from {} to {} is not supported".format(self.scale, new_scale)
        )

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


def _precision_max_deviation(precision: str) -> int:
    if precision == "ns":
        return 0
    if precision == "us":
        return 999
    if precision == "ms":
        return 999_999
    if precision == "s":
        return 999_999_999
    raise ValueError(f"Unsupported precision: {precision}")


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
