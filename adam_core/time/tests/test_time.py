import astropy.time
import numpy.testing as npt
import quivr as qv

from .. import time


class Wrapper(qv.Table):
    id = qv.StringColumn()
    times = time.Timestamp.as_column(nullable=True)


class TestTimeUnits:

    ts = time.Timestamp.from_kwargs(
        days=[0, 10000, 20000],
        nanos=[0, 1_000_000_000, 2_000_000_000],
    )

    empty = time.Timestamp.empty()

    with_nulls = qv.concatenate(
        [
            Wrapper.from_kwargs(times=ts, id=["a", "b", "c"]),
            Wrapper.from_kwargs(times=None, id=["d", "e", "f"]),
        ]
    ).times

    def test_micros(self):
        have = self.ts.micros()
        assert have.to_pylist() == [0, 1_000_000, 2_000_000]

        have = self.empty.micros()
        assert len(have) == 0

        have = self.with_nulls.micros()
        assert have.to_pylist() == [0, 1_000_000, 2_000_000, None, None, None]

    def test_millis(self):
        have = self.ts.millis()
        assert have.to_pylist() == [0, 1000, 2000]

        have = self.empty.millis()
        assert len(have) == 0

        have = self.with_nulls.millis()
        assert have.to_pylist() == [0, 1000, 2000, None, None, None]

    def test_seconds(self):
        have = self.ts.seconds()
        assert have.to_pylist() == [0, 1, 2]

        have = self.empty.seconds()
        assert len(have) == 0

        have = self.with_nulls.seconds()
        assert have.to_pylist() == [0, 1, 2, None, None, None]


class TestMJDConversions:
    ts = time.Timestamp.from_kwargs(
        days=[0, 10000, 20000],
        nanos=[0, 43200_000_000_000, 86400_000_000_000 - 1],
    )

    empty = time.Timestamp.empty()

    with_nulls = qv.concatenate(
        [
            Wrapper.from_kwargs(times=ts, id=["a", "b", "c"]),
            Wrapper.from_kwargs(times=None, id=["d", "e", "f"]),
        ]
    ).times

    def test_mjd(self):
        have = self.ts.mjd()
        assert have.to_pylist() == [0, 10000.5, 20000.999999999999]

    def test_empty(self):
        have = self.empty.mjd()
        assert len(have) == 0

    def test_with_nulls(self):
        have = self.with_nulls.mjd()
        assert have.to_pylist() == [0, 10000.5, 20000.999999999999, None, None, None]


class TestAstropyTime:
    ts = time.Timestamp.from_kwargs(
        days=[0, 50000, 60000],
        nanos=[0, 43200_000_000_000, 86400_000_000_000 - 1],
    )

    empty = time.Timestamp.empty()

    def test_to_astropy(self):
        have = self.ts.to_astropy()
        want = astropy.time.Time(
            [
                "1858-11-17T00:00:00.000000000",
                "1995-10-10T12:00:00.000000000",
                "2023-02-25T23:59:59.999999999",
            ],
            scale="tai",
        )

        npt.assert_allclose(have.mjd, want.mjd)

    def test_from_astropy(self):
        at = astropy.time.Time(
            [
                "1858-11-17T00:00:00.000000000",
                "1995-10-10T12:00:00.000000000",
                "2023-02-25T23:59:59.999999999",
            ],
            scale="tai",
        )
        have = time.Timestamp.from_astropy(at)
        assert have == self.ts

    def test_to_astropy_singleton(self):
        have = self.ts[0].to_astropy()
        want = astropy.time.Time("1858-11-17T00:00:00.000000000", scale="tai")
        assert have.mjd == want.mjd

    def test_from_astropy_scalar(self):
        at = astropy.time.Time("1858-11-17T00:00:00.000000000", scale="tai")
        have = time.Timestamp.from_astropy(at)
        assert have == self.ts[0]

    def test_empty(self):
        have = self.empty.to_astropy()
        assert len(have) == 0

    def test_roundtrips(self):
        zero_astropy_time = astropy.time.Time(0, val2=0, format="mjd")

        from_at = time.Timestamp.from_astropy(zero_astropy_time)
        assert from_at.days[0].as_py() == 0
        assert from_at.nanos[0].as_py() == 0

        roundtrip = from_at.to_astropy()
        assert zero_astropy_time == roundtrip
