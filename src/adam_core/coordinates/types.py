from __future__ import annotations

import typing

from .cartesian import CartesianCoordinates
from .cometary import CometaryCoordinates
from .geodetics import GeodeticCoordinates
from .keplerian import KeplerianCoordinates
from .spherical import SphericalCoordinates

#: Type alias that represents a generic coordinate system.
Coordinates = typing.Union[
    CartesianCoordinates,
    KeplerianCoordinates,
    SphericalCoordinates,
    CometaryCoordinates,
    GeodeticCoordinates,
]

#: Type variable which represents any of the coordinate classes.
CoordinateType = typing.TypeVar(
    "CoordinateType",
    bound=typing.Union[
        CartesianCoordinates,
        KeplerianCoordinates,
        CometaryCoordinates,
        SphericalCoordinates,
        GeodeticCoordinates,
    ],
)
