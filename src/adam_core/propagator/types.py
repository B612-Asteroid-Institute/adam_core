from typing import Union

from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..orbits.variants import VariantEphemeris, VariantOrbits
from ..time import Timestamp

# Only Rust backends are supported; the public propagator surfaces take/return
# concrete quivr tables in a single Python->Rust crossing (no Ray ObjectRef
# threading through the composition, which has been deleted).
TimestampType = Timestamp
OrbitType = Union[Orbits, VariantOrbits]
EphemerisType = Union[Ephemeris, VariantEphemeris]
ObserverType = Observers
