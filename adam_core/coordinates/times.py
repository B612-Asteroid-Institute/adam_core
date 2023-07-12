import pandas as pd
import pyarrow as pa
from astropy.time import Time
import erfa
from quivr import Float64Column, StringAttribute, Table


class Times(Table):

    # Stores the time as a pair of float64 values in the same style as erfa/astropy:
    # The first one is the day-part of a Julian date, and the second is
    # the fractional day-part.
    jd1 = Float64Column(nullable=False)
    jd2 = Float64Column(nullable=False)
    scale = StringAttribute()

    @classmethod
    def from_astropy(cls, time: Time):
        return cls.from_kwargs(jd1=time.jd1, jd2=time.jd2, scale=time.scale)

    def mjd(self) -> pa.lib.DoubleArray:
        """
        Convert to modified Julian date.
        """
        return pa.compute.add(pa.compute.add(self.jd1, -2400000.5), self.jd2)

    def set_scale(self, scale: str):
        """
        Set the time scale.
        """
        if scale == self.scale:
            return
        if scale not in ["utc", "tai", "tt", "tdb"]:
            raise ValueError(f"Invalid time scale: {scale}")
        if self.scale == "utc":
            if scale == "tai":
                self._utc_to_tai()
            elif scale == "tt":
                self._utc_to_tai()
                self._tai_to_tt()
            elif scale == "tdb":
                self._utc_to_tai()
                self._tai_to_tt()
                self._tt_to_tdb_approx()
        elif self.scale == "tai":
            if scale == "utc":
                self._tai_to_utc()
            elif scale == "tt":
                self._tai_to_tt()
            elif scale == "tdb":
                self._tai_to_tt()
                self._tt_to_tdb_approx()
        elif self.scale == "tt":
            if scale == "utc":
                self._tt_to_tai()
                self._tai_to_utc()
            elif scale == "tai":
                self._tt_to_tai()
            elif scale == "tdb":
                self._tt_to_tdb_approx()
        elif self.scale == "tdb":
            if scale == "utc":
                self._tdb_to_tt_approx()
                self._tt_to_tai()
                self._tai_to_utc()
            elif scale == "tai":
                self._tdb_to_tt_approx()
                self._tt_to_tai()
            elif scale == "tt":
                self._tdb_to_tt_approx()

    def _utc_to_ut1(self):
        """
        Convert self.scale from UTC to UT1.
        """
        new_jd1, new_jd2 = erfa.utcut1(self.jd1, self.jd2)
        self.jd1 = pa.array(new_jd1)
        self.jd2 = pa.array(new_jd2)
        self.scale = "ut1"

    def _utc_to_tai(self):
        """
        Convert self.scale from UTC to TAI.
        """
        new_jd1, new_jd2 = erfa.utctai(self.jd1, self.jd2)
        self.jd1 = pa.array(new_jd1)
        self.jd2 = pa.array(new_jd2)
        self.scale = "tai"

    def _tai_to_tt(self):
        """
        Convert self.scale from TAI to TT.
        """
        new_jd1, new_jd2 = erfa.taitt(self.jd1, self.jd2)
        self.jd1 = pa.array(new_jd1)
        self.jd2 = pa.array(new_jd2)
        self.scale = "tt"

    def _tt_to_tdb_approx(self):
        """
        Convert self.scale from TT to TDB using an approximate model, and assuming time from the geocenter.

        UT1 is approximated to be equal to UTC, which is wrong by up to a few hundred milliseconds.
        """
        # Need to get UT1 corresponding to the TT time
        tai1, tai2 = erfa.tttai(self.jd1, self.jd2)
        utc1, utc2 = erfa.taiutc(tai1, tai2)

        ut = (utc1 - 0.5 + utc2) % 1.0
        delta_sec = erfa.dtdb(self.jd1, self.jd2, ut, 0, 0, 0)
        self.jd2 = pc.compute.add(self.jd2, pa.array(delta_sec / 86400.0))
        self.scale = "tdb"

    def _tt_to_tai(self):
        """
        Convert self.scale from TT to TAI.
        """
        new_jd1, new_jd2 = erfa.tttai(self.jd1, self.jd2)
        self.jd1 = pa.array(new_jd1)
        self.jd2 = pa.array(new_jd2)
        self.scale = "tai"

    def _tai_to_utc(self):
        """
        Convert self.scale from TAI to UTC.
        """
        new_jd1, new_jd2 = erfa.taiutc(self.jd1, self.jd2)
        self.jd1 = pa.array(new_jd1)
        self.jd2 = pa.array(new_jd2)
        self.scale = "utc"

    def _tdb_to_tai_approx(self):
        # Need to get UT1 corresponding to the TT time
        tai1, tai2 = erfa.tttai(self.jd1, self.jd2)
        utc1, utc2 = erfa.taiutc(tai1, tai2)

        ut = (utc1 - 0.5 + utc2) % 1.0
        delta_sec = erfa.dtdb(self.jd1, self.jd2, ut, 0, 0, 0)
        self.jd2 = pc.compute.add(self.jd2, pa.array(-delta_sec / 86400.0))
        self.scale = "tai"

    def to_astropy(self, format: str = "jd") -> Time:
        t = Time(
            val=self.jd1.to_numpy(),
            val2=self.jd2.to_numpy(),
            format="jd",
            scale=self.scale,
        )
        if format == "jd":
            return t
        t.format = format
        return t

    def to_dataframe(self, flatten: bool = True) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame. Time scale is added as a suffix to the column names.

        Parameters
        ----------
        flatten : bool
            If True, flatten any tested tables.

        Returns
        -------
        df : `~pandas.DataFrame`
            A pandas DataFrame with two columns storing the times.
        """
        df = super().to_dataframe(flatten)
        df.rename(
            columns={col: f"{col}_{self.scale}" for col in df.columns}, inplace=True
        )
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Times":
        """
        Convert from a pandas DataFrame. Time scale is expected to be a suffix to the column names.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            A pandas DataFrame with two columns (*jd1_*, *jd2_*) storing the times.
        """
        # Column names may either start with times. or just jd1_ or jd2_
        df_filtered = df.filter(regex=".*jd[12]_.*", axis=1)

        # Extract time scale from column name
        scale = df_filtered.columns[0].split("_")[-1]

        # Rename columns to remove times. prefix
        for col in df_filtered.columns:
            if "time." in col:
                df_filtered.rename(columns={col: col.split("time.")[-1]}, inplace=True)

        df_renamed = df_filtered.rename(
            columns={col: col.split("_")[0] for col in df_filtered.columns}
        )
        return cls.from_kwargs(jd1=df_renamed.jd1, jd2=df_renamed.jd2, scale=scale)
