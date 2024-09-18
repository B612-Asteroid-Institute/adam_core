import quivr as qv

from ..time import Timestamp


class Photometry(qv.Table):

    time = Timestamp.as_column()
    mag = qv.Float64Column(nullable=True)
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn(nullable=True)
