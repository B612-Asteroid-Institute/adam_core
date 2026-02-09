from typing import Union

from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..orbits.variants import VariantEphemeris, VariantOrbits
from ..time import Timestamp
from ray import ObjectRef

TimestampType = Union[Timestamp, ObjectRef]
OrbitType = Union[Orbits, VariantOrbits, ObjectRef]
EphemerisType = Union[Ephemeris, VariantEphemeris, ObjectRef]
ObserverType = Union[Observers, ObjectRef]
