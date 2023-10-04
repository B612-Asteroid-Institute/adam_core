import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.spherical import SphericalCoordinates


class Ephemeris(qv.Table):

    orbit_id = qv.StringColumn()
    object_id = qv.StringColumn(nullable=True)
    coordinates = SphericalCoordinates.as_column()

    # The coordinates as observed by the observer will be the result of
    # light emitted or reflected from the object at the time of the observation.
    # Light, however, has a finite speed and so the object's observed cooordinates
    # will be different from its actual geometric coordinates at the time of observation.
    # Aberrated coordinates are coordinates that account for the light travel time
    # from the time of emission/reflection to the time of observation
    light_time = qv.Float64Column(nullable=True)
    aberrated_coordinates = CartesianCoordinates.as_column(nullable=True)
