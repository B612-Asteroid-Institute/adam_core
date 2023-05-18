import logging

from quivr import StringField, Table

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.cometary import CometaryCoordinates
from ..coordinates.keplerian import KeplerianCoordinates
from ..coordinates.spherical import SphericalCoordinates

logger = logging.getLogger(__name__)


class Orbits(Table):

    orbit_ids = StringField(nullable=True)
    object_ids = StringField(nullable=True)
    cartesian = CartesianCoordinates.as_field(nullable=True)
    keplerian = KeplerianCoordinates.as_field(nullable=True)
    cometary = CometaryCoordinates.as_field(nullable=True)
    spherical = SphericalCoordinates.as_field(nullable=True)
