import astropy.time
import astropy.units
import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import quivr as qv

from ..time import Timestamp


class Wrapper(qv.Table):
    id = qv.LargeStringColumn()
    times = Timestamp.as_column(nullable=True)


class TestTimeUnits:

    ts = Timestamp.from_kwargs(
        days=[0, 10000, 20000],
        nanos=[0, 1_000_000_000, 2_000_000_000],
    )

    empty = Timestamp.empty()

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
    ts = Timestamp.from_kwargs(
        days=[0, 10000, 20000],
        nanos=[0, 43200_000_000_000, 86400_000_000_000 - 1],
    )

    empty = Timestamp.empty()

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

    def test_from_list_ints(self):
        data = [0, 10000, 20000]
        have = Timestamp.from_mjd(data)
        assert have.days.to_pylist() == [0, 10000, 20000]
        assert have.nanos.to_pylist() == [0, 0, 0]

    def test_from_list_floats(self):
        data = [0.0, 10000.5, 20000.75]
        have = Timestamp.from_mjd(data)
        assert have.days.to_pylist() == [0, 10000, 20000]
        assert have.nanos.to_pylist() == [0, 43200_000_000_000, 64800_000_000_000]

    def test_from_numpy_ints(self):
        data = np.array([0, 10000, 20000], dtype=np.int64)
        have = Timestamp.from_mjd(data)
        assert have.days.to_pylist() == [0, 10000, 20000]
        assert have.nanos.to_pylist() == [0, 0, 0]

    def test_from_numpy_floats(self):
        data = np.array([0.0, 10000.5, 20000.75], dtype=np.float64)
        have = Timestamp.from_mjd(data)
        assert have.days.to_pylist() == [0, 10000, 20000]
        assert have.nanos.to_pylist() == [0, 43200_000_000_000, 64800_000_000_000]

    def test_from_pyarrow_ints(self):
        data = pa.array([0, 10000, 20000], type=pa.int64())
        have = Timestamp.from_mjd(data)
        assert have.days.to_pylist() == [0, 10000, 20000]
        assert have.nanos.to_pylist() == [0, 0, 0]

    def test_from_pyarrow_floats(self):
        data = pa.array([0.0, 10000.5, 20000.75], type=pa.float64())
        have = Timestamp.from_mjd(data)
        assert have.days.to_pylist() == [0, 10000, 20000]
        assert have.nanos.to_pylist() == [0, 43200_000_000_000, 64800_000_000_000]

    def test_from_pyarrow_table_float(self):
        """
        This test case exercises chunked arrays, which have slightly different behavior
        """
        data = pa.table({"mjd": [0.0, 10000.5, 20000.75]})
        have = Timestamp.from_mjd(data["mjd"])
        assert have.days.to_pylist() == [0, 10000, 20000]
        assert have.nanos.to_pylist() == [0, 43200_000_000_000, 64800_000_000_000]


class TestAstropyTime:
    ts = Timestamp.from_kwargs(
        days=[0, 50000, 60000],
        nanos=[0, 43200_000_000_000, 86400_000_000_000 - 1],
    )

    empty = Timestamp.empty()

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
        have = Timestamp.from_astropy(at)
        assert have == self.ts

    def test_to_astropy_singleton(self):
        have = self.ts[0].to_astropy()
        want = astropy.time.Time("1858-11-17T00:00:00.000000000", scale="tai")
        assert have.mjd == want.mjd

    def test_from_astropy_scalar(self):
        at = astropy.time.Time("1858-11-17T00:00:00.000000000", scale="tai")
        have = Timestamp.from_astropy(at)
        assert have == self.ts[0]

    def test_empty(self):
        have = self.empty.to_astropy()
        assert len(have) == 0

    def test_roundtrip_scalar_zero(self):
        zero_astropy_time = astropy.time.Time(0, val2=0, format="mjd")

        from_at = Timestamp.from_astropy(zero_astropy_time)
        assert from_at.days[0].as_py() == 0
        assert from_at.nanos[0].as_py() == 0

        roundtrip = from_at.to_astropy()
        assert zero_astropy_time == roundtrip

    def test_roundtrip_second_precision(self):
        t1 = astropy.time.Time("2020-01-01T00:00:00.000000000", scale="tai")
        t2 = t1 + 1 * astropy.units.second

        have1 = Timestamp.from_astropy(t1)
        have2 = have1.add_seconds(1)

        assert have2.to_astropy() == t2


class TestTimeMath:

    t1 = Timestamp.from_kwargs(
        days=[0, 50000, 60000],
        nanos=[0, 43200_000_000_000, 86400_000_000_000 - 1],
    )

    def test_add_nanos_out_of_range(self):
        MIN_VAL = -86400_000_000_000
        MAX_VAL = 86400_000_000_000 - 1
        # Scalars:
        with pytest.raises(ValueError):
            self.t1.add_nanos(MIN_VAL - 1)
        with pytest.raises(ValueError):
            self.t1.add_nanos(MAX_VAL + 1)

        # Arrays:
        with pytest.raises(ValueError):
            self.t1.add_nanos([MIN_VAL - 1, 0, 0])
        with pytest.raises(ValueError):
            self.t1.add_nanos([0, MAX_VAL + 1, 0])
        with pytest.raises(ValueError):
            self.t1.add_nanos([0, 0, MIN_VAL - 1])
        with pytest.raises(ValueError):
            self.t1.add_nanos([0, 0, MAX_VAL + 1])

    def test_add_seconds_out_of_range(self):
        MIN_VAL = -86400
        MAX_VAL = 86400 - 1
        # Scalars:
        with pytest.raises(ValueError):
            self.t1.add_seconds(MIN_VAL - 1)
        with pytest.raises(ValueError):
            self.t1.add_seconds(MAX_VAL + 1)

        # Arrays:
        with pytest.raises(ValueError):
            self.t1.add_seconds([MIN_VAL - 1, 0, 0])
        with pytest.raises(ValueError):
            self.t1.add_seconds([0, MAX_VAL + 1, 0])

    def test_add_millis_out_of_range(self):
        MIN_VAL = -86400_000
        MAX_VAL = 86400_000 - 1
        # Scalars:
        with pytest.raises(ValueError):
            self.t1.add_millis(MIN_VAL - 1)
        with pytest.raises(ValueError):
            self.t1.add_millis(MAX_VAL + 1)

        # Arrays:
        with pytest.raises(ValueError):
            self.t1.add_millis([MIN_VAL - 1, 0, 0])
        with pytest.raises(ValueError):
            self.t1.add_millis([0, MAX_VAL + 1, 0])

    def test_add_micros_out_of_range(self):
        MIN_VAL = -86400_000_000
        MAX_VAL = 86400_000_000 - 1
        # Scalars:
        with pytest.raises(ValueError):
            self.t1.add_micros(MIN_VAL - 1)
        with pytest.raises(ValueError):
            self.t1.add_micros(MAX_VAL + 1)

        # Arrays:
        with pytest.raises(ValueError):
            self.t1.add_micros([MIN_VAL - 1, 0, 0])
        with pytest.raises(ValueError):
            self.t1.add_micros([0, MAX_VAL + 1, 0])

    def test_add_nanos_scalar(self):
        have = self.t1.add_nanos(1)
        assert have.days.to_pylist() == [0, 50000, 60001]
        assert have.nanos.to_pylist() == [1, 43200_000_000_001, 0]

        have = self.t1.add_nanos(-1)
        assert have.days.to_pylist() == [-1, 50000, 60000]
        assert have.nanos.to_pylist() == [
            86400_000_000_000 - 1,
            43200_000_000_000 - 1,
            86400_000_000_000 - 2,
        ]

        have = self.t1.add_nanos(43200_000_000_000)
        assert have.days.to_pylist() == [0, 50001, 60001]
        assert have.nanos.to_pylist() == [43200_000_000_000, 0, 43200_000_000_000 - 1]

    def test_add_nanos_array(self):
        have = self.t1.add_nanos([1, 2, 3])
        assert have.days.to_pylist() == [0, 50000, 60001]
        assert have.nanos.to_pylist() == [1, 43200_000_000_002, 2]

        have = self.t1.add_nanos([-1, -2, -3])
        assert have.days.to_pylist() == [-1, 50000, 60000]
        assert have.nanos.to_pylist() == [
            86400_000_000_000 - 1,
            43200_000_000_000 - 2,
            86400_000_000_000 - 4,
        ]

    def test_add_seconds(self):
        have = self.t1.add_seconds(1)
        assert have.days.to_pylist() == [0, 50000, 60001]
        assert have.nanos.to_pylist() == [
            1_000_000_000,
            43200_000_000_000 + 1_000_000_000,
            1_000_000_000 - 1,
        ]

        have = self.t1.add_seconds(-1)
        assert have.days.to_pylist() == [-1, 50000, 60000]
        assert have.nanos.to_pylist() == [
            86400_000_000_000 - 1_000_000_000,
            43200_000_000_000 - 1_000_000_000,
            86400_000_000_000 - 1_000_000_000 - 1,
        ]

        have = self.t1.add_seconds([1, 2, 3])
        assert have.days.to_pylist() == [0, 50000, 60001]
        assert have.nanos.to_pylist() == [
            1_000_000_000,
            43200_000_000_000 + 2_000_000_000,
            3_000_000_000 - 1,
        ]

    def test_add_millis(self):
        have = self.t1.add_millis(1)
        assert have.days.to_pylist() == [0, 50000, 60001]
        assert have.nanos.to_pylist() == [
            1_000_000,
            43200_000_000_000 + 1_000_000,
            1_000_000 - 1,
        ]

        have = self.t1.add_millis(-1)
        assert have.days.to_pylist() == [-1, 50000, 60000]
        assert have.nanos.to_pylist() == [
            86400_000_000_000 - 1_000_000,
            43200_000_000_000 - 1_000_000,
            86400_000_000_000 - 1_000_000 - 1,
        ]

        have = self.t1.add_millis([1, 2, 3])
        assert have.days.to_pylist() == [0, 50000, 60001]
        assert have.nanos.to_pylist() == [
            1_000_000,
            43200_000_000_000 + 2_000_000,
            3_000_000 - 1,
        ]

    def test_add_micros(self):
        have = self.t1.add_micros(1)
        assert have.days.to_pylist() == [0, 50000, 60001]
        assert have.nanos.to_pylist() == [1_000, 43200_000_000_000 + 1_000, 1_000 - 1]

        have = self.t1.add_micros(-1)
        assert have.days.to_pylist() == [-1, 50000, 60000]
        assert have.nanos.to_pylist() == [
            86400_000_000_000 - 1_000,
            43200_000_000_000 - 1_000,
            86400_000_000_000 - 1_000 - 1,
        ]

        have = self.t1.add_micros([1, 2, 3])
        assert have.days.to_pylist() == [0, 50000, 60001]
        assert have.nanos.to_pylist() == [1_000, 43200_000_000_000 + 2_000, 3_000 - 1]

    def test_add_days(self):
        have = self.t1.add_days(1)
        assert have.days.to_pylist() == [1, 50001, 60001]
        assert have.nanos.to_pylist() == self.t1.nanos.to_pylist()

        have = self.t1.add_days(-1)
        assert have.days.to_pylist() == [-1, 49999, 59999]
        assert have.nanos.to_pylist() == self.t1.nanos.to_pylist()

        have = self.t1.add_days([1, 2, 3])
        assert have.days.to_pylist() == [1, 50002, 60003]
        assert have.nanos.to_pylist() == self.t1.nanos.to_pylist()

    def test_equals_array(self):
        t1 = self.t1
        t2 = self.t1

        assert pc.all(t1.equals_array(t2)).as_py()

        t3 = t2.add_nanos(1)
        assert not pc.all(t1.equals_array(t3)).as_py()
        assert pc.all(t1.equals_array(t3, precision="us")).as_py()

        t4 = Timestamp.from_kwargs(
            days=t1.days,
            nanos=t1.nanos,
            scale="utc",
        )
        with pytest.raises(ValueError):
            t1.equals_array(t4)

    def test_equals_scalar(self):
        t1 = Timestamp.from_kwargs(
            days=[50000, 60000, 70000],
            nanos=[0, 1, 2],
        )

        have = t1.equals_scalar(days=50000, nanos=0)
        assert have.to_pylist() == [True, False, False]

    def test_equals_scalar_precision(self):
        t1 = Timestamp.from_kwargs(
            days=[0, 0, 1, 1, 2, 2],
            nanos=[
                500,
                86400_000_000_000 - 500,
                500,
                86400_000_000_000 - 500,
                500,
                86400_000_000_000 - 500,
            ],
        )
        have = t1.equals_scalar(days=1, nanos=0, precision="us")
        assert have.to_pylist() == [False, True, True, False, False, False]

    def test_difference_scalar(self):
        # Compute difference from days=1, nanos=100
        cases = [
            {
                "in_days": 0,
                "in_nanos": 0,
                "out_days": -2,
                "out_nanos": 86400_000_000_000 - 100,
            },
            {
                "in_days": 0,
                "in_nanos": 50,
                "out_days": -2,
                "out_nanos": 86400_000_000_000 - 50,
            },
            {"in_days": 0, "in_nanos": 200, "out_days": -1, "out_nanos": 100},
            {
                "in_days": 1,
                "in_nanos": 50,
                "out_days": -1,
                "out_nanos": 86400_000_000_000 - 50,
            },
            {"in_days": 1, "in_nanos": 100, "out_days": 0, "out_nanos": 0},
            {"in_days": 1, "in_nanos": 200, "out_days": 0, "out_nanos": 100},
            {
                "in_days": 2,
                "in_nanos": 50,
                "out_days": 0,
                "out_nanos": 86400_000_000_000 - 50,
            },
            {"in_days": 2, "in_nanos": 100, "out_days": 1, "out_nanos": 0},
            {"in_days": 2, "in_nanos": 200, "out_days": 1, "out_nanos": 100},
        ]

        for i, c in enumerate(cases):
            t1 = Timestamp.from_kwargs(
                days=[c["in_days"]],
                nanos=[c["in_nanos"]],
            )
            have_days, have_nanos = t1.difference_scalar(days=1, nanos=100)
            assert have_days[0].as_py() == c["out_days"], f"case {i}"
            assert have_nanos[0].as_py() == c["out_nanos"], f"case {i}"

    def test_difference(self):
        t1 = Timestamp.from_kwargs(
            days=[50000, 60000, 70000],
            nanos=[0, 1, 2],
        )
        t2 = Timestamp.from_kwargs(
            days=[50000, 60000, 70000],
            nanos=[100, 200, 300],
        )
        have_days, have_nanos = t1.difference(t2)
        assert have_days.to_pylist() == [-1, -1, -1]
        assert have_nanos.to_pylist() == [
            86400_000_000_000 - 100,
            86400_000_000_000 - 199,
            86400_000_000_000 - 298,
        ]


def test_dataframe_roundtrip():
    t1 = Timestamp.from_kwargs(
        days=[50000, 60000, 70000],
        nanos=[0, 1, 2],
        scale="tdb",
    )

    df = t1.to_dataframe()

    t2 = Timestamp.from_dataframe(df)

    assert t1 == t2
    assert t1.scale == t2.scale


def test_dataframe_roundtrip_nested():
    t1 = Timestamp.from_kwargs(
        days=[50000, 60000, 70000],
        nanos=[0, 1, 2],
        scale="tdb",
    )
    w = Wrapper.from_kwargs(
        id=["a", "b", "c"],
        times=t1,
    )

    df = w.to_dataframe()

    w2 = Wrapper.from_flat_dataframe(df)

    assert w == w2
    assert w.times.scale == w2.times.scale


class TestUnique:
    def test_empty(self):
        t = Timestamp.empty(scale="tdb")
        have = t.unique()
        assert have.scale == "tdb"
        assert len(have) == 0

        t2 = Timestamp.empty(scale="utc")
        have = t2.unique()
        assert have.scale == "utc"
        assert len(have) == 0

    def test_one(self):
        t = Timestamp.from_kwargs(days=[1], nanos=[2], scale="tdb")
        have = t.unique()
        assert have.scale == "tdb"
        assert len(have) == 1
        assert have.days.to_pylist() == [1]
        assert have.nanos.to_pylist() == [2]

    def test_multple(self):
        t = Timestamp.from_kwargs(
            days=[1, 1, 2, 2, 1, 1], nanos=[2, 3, 4, 5, 2, 3], scale="tdb"
        )
        have = t.unique()
        assert have.scale == "tdb"
        assert len(have) == 4
        pairs = zip(have.days.to_pylist(), have.nanos.to_pylist())
        assert sorted(list(pairs)) == [(1, 2), (1, 3), (2, 4), (2, 5)]


@pytest.mark.parametrize("rescale", [True, False])
def test_Timestamp_link(rescale):
    # Test that the link method works as expected
    time = Timestamp.from_kwargs(
        days=[68000, 68000, 68010, 68010, 68020, 68020],
        nanos=[1, 2, 3, 3, 4, 4],
        scale="tdb",
    )

    other = Timestamp.from_kwargs(
        days=[68000, 68000, 68010, 68020], nanos=[1, 2, 3, 4], scale="tdb"
    )
    if rescale:
        other = other.rescale("utc")

    linkage = time.link(other)
    assert len(linkage.all_unique_values) == 4

    key = (68000, 1)
    left_table, right_table = linkage.select(key)
    assert len(left_table) == 1
    assert len(right_table) == 1
    assert left_table.days.to_pylist() == [68000]
    assert left_table.nanos.to_pylist() == [1]
    assert right_table.days.to_pylist() == [68000]
    assert right_table.nanos.to_pylist() == [1]

    key = (68000, 2)
    left_table, right_table = linkage.select(key)
    assert len(left_table) == 1
    assert len(right_table) == 1
    assert left_table.days.to_pylist() == [68000]
    assert left_table.nanos.to_pylist() == [2]
    assert right_table.days.to_pylist() == [68000]
    assert right_table.nanos.to_pylist() == [2]

    key = (68010, 3)
    left_table, right_table = linkage.select(key)
    assert len(left_table) == 2
    assert len(right_table) == 1
    assert left_table.days.to_pylist() == [68010, 68010]
    assert left_table.nanos.to_pylist() == [3, 3]
    assert right_table.days.to_pylist() == [68010]
    assert right_table.nanos.to_pylist() == [3]

    key = (68020, 4)
    left_table, right_table = linkage.select(key)
    assert len(left_table) == 2
    assert len(right_table) == 1
    assert left_table.days.to_pylist() == [68020, 68020]
    assert left_table.nanos.to_pylist() == [4, 4]
    assert right_table.days.to_pylist() == [68020]
    assert right_table.nanos.to_pylist() == [4]


def test_Timestamp_link_precision():
    # Test that the link method works as expected with user defined precisions
    time = Timestamp.from_kwargs(
        days=[1, 1, 1, 1, 1, 1, 1, 1],
        nanos=[9, 99, 999, 9_999, 99_999, 999_999, 9_999_999, 99_999_999],
        scale="tdb",
    )

    # 1 nano per nano :)
    time_ns = Timestamp.from_kwargs(
        days=[1, 1, 1, 1, 1, 1, 1, 1],
        nanos=[9, 99, 999, 9_999, 99_999, 999_999, 9_999_999, 99_999_999],
        scale="tdb",
    )
    linkage = time.link(time_ns, precision="ns")
    assert len(linkage.all_unique_values) == 8

    # 1_000 nanos per micro
    time_micros = Timestamp.from_kwargs(
        days=[1, 1, 1, 1, 1, 1, 1, 1],
        nanos=[0, 0, 0, 9_999, 99_999, 999_999, 9_999_999, 99_999_999],
        scale="tdb",
    )
    linkage = time.link(time_micros, precision="us")
    assert len(linkage.all_unique_values) == 6

    # 1_000_000 nanos per milli
    time_millis = Timestamp.from_kwargs(
        days=[1, 1, 1, 1, 1, 1, 1, 1],
        nanos=[0, 0, 0, 0, 0, 0, 9_999_999, 99_999_999],
        scale="tdb",
    )
    linkage = time.link(time_millis, precision="ms")
    assert len(linkage.all_unique_values) == 3

    # 1_000_000_000 nanos per second
    time_s = Timestamp.from_kwargs(
        days=[1, 1, 1, 1, 1, 1, 1, 1],
        nanos=[
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        scale="tdb",
    )
    linkage = time.link(time_s, precision="s")
    assert len(linkage.all_unique_values) == 1


def test_Timestamp_max():
    # Test that the max method correctly returns
    # the maximum time
    times = Timestamp.from_kwargs(
        days=[5, 1, 3, 2, 4, 1, 2, 3, 4, 5],
        nanos=[2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
    )
    times = qv.concatenate([times for _ in range(10)])

    max_time = times.max()
    assert len(max_time) == 1
    assert max_time.scale == times.scale
    assert max_time.days.to_pylist() == [5]
    assert max_time.nanos.to_pylist() == [2]


def test_Timestamp_min():
    # Test that the min method correctly returns
    # the minimum time
    times = Timestamp.from_kwargs(
        days=[5, 1, 3, 2, 4, 1, 2, 3, 4, 5],
        nanos=[2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
    )
    times = qv.concatenate([times for _ in range(10)])

    min_time = times.min()
    assert len(min_time) == 1
    assert min_time.scale == times.scale
    assert min_time.days.to_pylist() == [1]
    assert min_time.nanos.to_pylist() == [1]
