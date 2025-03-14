from typing import Union

from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..orbits.variants import VariantEphemeris, VariantOrbits
from ..time import Timestamp

RAY_INSTALLED = False
try:
    from ray import ObjectRef

    RAY_INSTALLED = True
except ImportError:
    pass

TimestampType = Union[Timestamp, ObjectRef]
OrbitType = Union[Orbits, VariantOrbits, ObjectRef]
EphemerisType = Union[Ephemeris, VariantEphemeris, ObjectRef]
ObserverType = Union[Observers, ObjectRef]
