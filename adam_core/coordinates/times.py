import numpy as np
from astropy.time import Time
from quivr import Float64Field, StringField, Table


class Times(Table):

    # TODO: @spenczar - We could have this class store times
    # as two integers... :)
    mjd = Float64Field(nullable=False)

    # Could replace this with an enum of some kind
    scale = StringField(nullable=False)

    @classmethod
    def from_astropy(cls, time: Time):
        scale = np.array([time.scale for i in range(len(time))])
        return cls.from_kwargs(mjd=time.mjd, scale=scale)

    def to_astropy(self) -> Time:
        return Time(self.mjd.to_numpy(), format="mjd", scale=str(self.scale[0]))
