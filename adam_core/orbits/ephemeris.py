from quivr import Float64Column, StringAttribute, StringColumn, Table

from ..coordinates.origin import Origin
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.times import Times


class Ephemeris(Table):

    orbit_id = StringColumn(nullable=False)
    object_id = StringColumn(nullable=False)
    time = Times.as_column(nullable=False)
    rho = Float64Column(nullable=False)
    ra = Float64Column(nullable=False)
    dec = Float64Column(nullable=False)
    vrho = Float64Column(nullable=False)
    vra = Float64Column(nullable=False)
    vdec = Float64Column(nullable=False)
    origin = Origin.as_column(nullable=False)
    frame = StringAttribute()

    def as_spherical_coordinates(self):
        return SphericalCoordinates.from_kwargs(
            rho=self.rho,
            lon=self.ra,
            lat=self.dec,
            vrho=self.vrho,
            vlon=self.vra,
            vlat=self.vdec,
            time=self.time,
            origin=self.origin,
            frame=self.frame,
        )
