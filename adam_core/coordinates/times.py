import numpy as np
from astropy.time import Time
from quivr import Float64Field, StringField, Table


class Times(Table):

    # Stores the time as a pair of float64 values in the same style as erfa/astropy:
    # The first one is the day-part of a Julian date, and the second is
    # the fractional day-part.
    jd1 = Float64Field(nullable=False)
    jd2 = Float64Field(nullable=False)
    scale = StringField(nullable=False)

    @classmethod
    def from_astropy(cls, time: Time):
        scale = np.array([time.scale for i in range(len(time))])
        return cls.from_kwargs(jd1=time.jd1, jd2=time.jd2, scale=scale)

    def to_astropy(self, format: str = "jd") -> Time:
        t = Time(
            val=self.jd1.to_numpy(),
            val2=self.jd2.to_numpy(),
            format="jd",
            scale=self.scale[0].as_py(),
        )
        if format == "jd":
            return t
        t.format = format
        return t
