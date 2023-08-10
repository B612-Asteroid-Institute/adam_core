import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from astropy.time import Time
from typing_extensions import Self


class Times(qv.Table):

    # Stores the time as a pair of float64 values in the same style as erfa/astropy:
    # The first one is the day-part of a Julian date, and the second is
    # the fractional day-part.
    jd1 = qv.Float64Column()
    jd2 = qv.Float64Column()
    scale = qv.StringAttribute(default="utc")

    @classmethod
    def from_astropy(cls, time: Time):
        if time.isscalar == 1:
            jd1 = [time.jd1]
            jd2 = [time.jd2]
        else:
            jd1 = time.jd1
            jd2 = time.jd2
        return cls.from_kwargs(jd1=jd1, jd2=jd2, scale=time.scale)

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

    def to_scale(self, scale: str) -> Self:
        """
        Convert to a different time scale.

        Parameters
        ----------
        scale : str
            The time scale to convert to.

        Returns
        -------
        times : `~adam_core.coordinates.times.Times`
            The times in the new scale.
        """
        time = self.to_astropy()
        time._set_scale(scale)
        return self.from_astropy(time)

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

    def jd(self) -> pa.lib.DoubleArray:
        """
        Returns the times as a double-precision array of julian date values.

        Returns
        -------
        jd : `~pyarrow.DoubleArray`
            The times as a double-precision array of julian date values.
        """
        return pc.add(self.jd1, self.jd2)

    def mjd(self) -> pa.lib.DoubleArray:
        """
        Returns the times as a double-precision array of modified julian date values.

        Returns
        -------
        mjd : `~pyarrow.DoubleArray`
            The times as a double-precision array of modified julian date values.
        """
        return pc.add(pc.add(self.jd1, -2400000.5), self.jd2)
