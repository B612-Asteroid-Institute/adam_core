from quivr import Float64Column, StringAttribute, StringColumn, Table

from ..coordinates.spherical import SphericalCoordinates
from ..observers import Observers


class Ephemeris(Table):

    orbit_id = StringColumn(nullable=False)
    object_id = StringColumn(nullable=False)
    observer = Observers.as_column(nullable=True)
    rho = Float64Column(nullable=True)
    ra = Float64Column(nullable=False)
    dec = Float64Column(nullable=False)
    vrho = Float64Column(nullable=True)
    vra = Float64Column(nullable=True)
    vdec = Float64Column(nullable=True)
    frame = StringAttribute()

    def as_spherical_coordinates(self):
        return SphericalCoordinates.from_kwargs(
            rho=self.rho,
            lon=self.ra,
            lat=self.dec,
            vrho=self.vrho,
            vlon=self.vra,
            vlat=self.vdec,
            time=self.observer.coordinates.time,
            origin=self.observer.code,
            frame=self.frame,
        )
