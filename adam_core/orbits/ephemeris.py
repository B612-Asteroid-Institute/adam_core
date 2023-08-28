from quivr import StringColumn, Table

from ..coordinates.spherical import SphericalCoordinates


class Ephemeris(Table):

    orbit_id = StringColumn(nullable=False)
    object_id = StringColumn()
    coordinates = SphericalCoordinates.as_column()
