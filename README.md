# adam_core: ADAM Core Utilities
#### A Python package by the Asteroid Institute, a program of the B612 Foundation
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://img.shields.io/badge/Python-3.9%2B-blue)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![pip - Build, Lint, Test, and Coverage](https://github.com/B612-Asteroid-Institute/adam_core/actions/workflows/pip-build-lint-test-coverage.yml/badge.svg)](https://github.com/B612-Asteroid-Institute/adam_core/actions/workflows/pip-build-lint-test-coverage.yml)

`adam_core` is used by a variety of library and services at the Asteroid Institute. Sharing these common classes, types, and conversions amongst our tools ensures consistency and accuracy.

## Installation

ADAM Core is available on PyPI

```bash
pip install adam_core
```

## Usage

### Orbits

To define an orbit:
```python
from adam_core.coordinates import KeplerianCoordinates
from adam_core.coordinates import Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

keplerian_elements = KeplerianCoordinates.from_kwargs(
    time=Timestamp.from_mjd([59000.0], scale="tdb"),
    a=[1.0],
    e=[0.002],
    i=[10.],
    raan=[50.0],
    ap=[20.0],
    M=[30.0],
    origin=Origin.from_kwargs(code=["SUN"]),
    frame="ecliptic"
)
orbits = Orbits.from_kwargs(
    orbit_id=["1"],
    object_id=["Test Orbit"],
    coordinates=keplerian_elements.to_cartesian(),
)
```
Note that internally, all orbits are stored in Cartesian coordinates. Cartesian coordinates do not have any
singularities and are thus more robust for numerical integration. Any orbital element conversions to Cartesian
can be done on demand by calling `to_cartesian()` on the coordinates object.

The underlying orbits class is 2 dimensional and can store elements and covariances for multiple orbits.

```python
from adam_core.coordinates import KeplerianCoordinates
from adam_core.coordinates import Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

keplerian_elements = KeplerianCoordinates.from_kwargs(
    time=Timestamp.from_mjd([59000.0, 60000.0], scale="tdb"),
    a=[1.0, 3.0],
    e=[0.002, 0.0],
    i=[10., 30.],
    raan=[50.0, 32.0],
    ap=[20.0, 94.0],
    M=[30.0, 159.0],
    origin=Origin.from_kwargs(code=["SUN", "SUN"]),
    frame="ecliptic"
)
orbits = Orbits.from_kwargs(
    orbit_id=["1", "2"],
    object_id=["Test Orbit 1", "Test Orbit 2"],
    coordinates=keplerian_elements.to_cartesian(),
)
```

Orbits can be easily converted to a pandas DataFrame:
```python
orbits.to_dataframe()  
  orbit_id     object_id  coordinates.x  coordinates.y  coordinates.z  coordinates.vx  coordinates.vy  coordinates.vz  coordinates.time.days  coordinates.time.nanos                      coordinates.covariance.values coordinates.origin.code  
0        1  Test Orbit 1      -0.166403       0.975273       0.133015       -0.016838       -0.003117        0.001921                  59000                       0  [nan, nan, nan, nan, nan, nan, nan, nan, nan, ...                     SUN  
1        2  Test Orbit 2       0.572777      -2.571820      -1.434457        0.009387        0.002900       -0.001452                  60000                       0  [nan, nan, nan, nan, nan, nan, nan, nan, nan, ...                     SUN
```

Orbits can also be defined with uncertainties.
```python
import numpy as np
from adam_core.coordinates import KeplerianCoordinates
from adam_core.coordinates import Origin
from adam_core.coordinates import CoordinateCovariances
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

keplerian_elements = KeplerianCoordinates.from_kwargs(
    time=Timestamp.from_mjd([59000.0], scale="tdb"),
    a=[1.0],
    e=[0.002],
    i=[10.],
    raan=[50.0],
    ap=[20.0],
    M=[30.0],
    covariance=CoordinateCovariances.from_sigmas(
        np.array([[0.002, 0.001, 0.01, 0.01, 0.1, 0.1]])
    ),
    origin=Origin.from_kwargs(code=["SUN"]),
    frame="ecliptic"
)

orbits = Orbits.from_kwargs(
    orbit_id=["1"],
    object_id=["Test Orbit with Uncertainties"],
    coordinates=keplerian_elements.to_cartesian(),
)
orbits.to_dataframe()  
  orbit_id                      object_id  coordinates.x  coordinates.y  coordinates.z  coordinates.vx  coordinates.vy  coordinates.vz  coordinates.time.days  coordinates.time.nanos                      coordinates.covariance.values coordinates.origin.code  
0        1  Test Orbit with Uncertainties      -0.166403       0.975273       0.133015       -0.016838       -0.003117        0.001921                  59000                       0  [6.654136535278775e-06, 1.2935845684776213e-06...                     SUN
```

The covariance matrices can be extracted in matrix form by using the `.to_matrix()` method:
```python
orbits.coordinates.covariance.to_matrix()
```

Similarly, if you just want to access the orbital elements you can do the following:
```python
orbits.coordinates.values
```

To query orbits from JPL Horizons:
```python
from adam_core.orbits.query import query_horizons
from adam_core.time import Timestamp

times = Timestamp.from_mjd([60000.0], scale="tdb")
object_ids = ["Duende", "Eros", "Ceres"]
orbits = query_horizons(object_ids, times)
```

To query orbits from JPL SBDB:
```python
from adam_core.orbits.query import query_sbdb

object_ids = ["Duende", "Eros", "Ceres"]
orbits = query_sbdb(object_ids)
```

#### Orbital Element Conversions
Orbital elements can be accessed via the corresponding attribute. All conversions, including covariances, are done on demand and stored.

```python
# Cartesian Elements
orbits.coordinates

# To convert to other representations
cometary_elements = orbits.coordinates.to_cometary()
keplerian_elements = orbits.coordinates.to_keplerian()
spherical_elements = orbits.coordinates.to_spherical()
```

### Propagator
The propagator class in `adam_core` provides a generalized interface to the supported orbit integrators and ephemeris generators. The propagator class is designed to be used with the `Orbits` class and can handle multiple orbits and times. 

You will need to install either adam_core[assist], or another compatible propagator in order to use propagation, ephemeris generation, or impact analysis.

#### Propagation
To propagate orbits with ASSIST (here we grab some orbits from Horizons first):

```python
import numpy as np
from astropy import units as u

from adam_core.orbits.query import query_horizons
from adam_assist import ASSISTPropagator
from adam_core.time import Timestamp

# Get orbits to propagate
initial_time = Timestamp.from_mjd([60000.0], scale="tdb")
object_ids = ["Duende", "Eros", "Ceres"]
orbits = query_horizons(object_ids, initial_time)

# initialize the propagator
propagator = ASSISTPropagator()

# Define propagation times
times = initial_time.from_mjd(initial_time.mjd() + np.arange(0, 100))

# Propagate orbits! This function supports multiprocessing for large
# propagation jobs.
propagated_orbits = propagator.propagate_orbits(
    orbits,
    times,
    chunk_size=100,
    max_processes=1,
)
```

#### Ephemeris Generation
Ephemeris generation requires a propagator that implements the EphemerisMixin interface. This is currently only implemented by the PYOORB propagator. The ephemeris generator will automatically map the propagated covariance matrices to the sky-plane.

You will need to install adam-pyoorb in order to use the ephemeris generator, which is currently only available on GitHub.

```sh
pip install git+https://github.com/B612-Asteroid-Institute/adam-pyoorb.git
```


```python
import numpy as np
from astropy import units as u

from adam_core.orbits.query import query_horizons
from adam_core.propagator.adam_pyoorb import PYOORBPropagator
from adam_core.observers import Observers
from adam_core.time import Timestamp

# Get orbits to propagate
initial_time = Timestamp.from_mjd([60000.0], scale="tdb")
object_ids = ["Duende", "Eros", "Ceres"]
orbits = query_horizons(object_ids, initial_time)

# Make sure PYOORB is ready
propagator = PYOORBPropagator()

# Define a set of observers and observation times
times = Timestamp.from_mjd(initial_time.mjd() + np.arange(0, 100))
observers = Observers.from_code("I11", times)

# Generate ephemerides! This function supports multiprocessing for large
# propagation jobs.
ephemeris = propagator.generate_ephemeris(
    orbits,
    observers,
    chunk_size=100,
    max_processes=1
)
```


### Low-level APIs

#### State Vectors from Development Ephemeris files
Getting the heliocentric ecliptic state vector of a DE440 body at a given set of times (in this case the barycenter of the Jovian system):
```python
import numpy as np

from adam_core.coordinates import OriginCodes
from adam_core.utils import get_perturber_state
from adam_core.time import Timestamp

states = get_perturber_state(
    OriginCodes.JUPITER_BARYCENTER,
    Timetamp.from_mjd(np.arange(59000, 60000), scale="tdb"),
    frame="ecliptic",
    origin=OriginCodes.SUN,
)
```

#### 2-body Propagation
`adam_core` also has 2-body propagation functionality. To propagate any orbit with 2-body dynamics:
```python
import numpy as np
from astropy import units as u

from adam_core.orbits.query import query_sbdb
from adam_core.dynamics import propagate_2body
from adam_core.time import Timestamp

# Get orbit to propagate
object_ids = ["Duende", "Eros", "Ceres"]
orbits = query_sbdb(object_ids)

# Define propagation times
times = Timestamp.from_mjd(np.arange(59000, 60000), scale="tdb")

# Propagate orbits with 2-body dynamics
propagated_orbits = propagate_2body(
    orbits,
    times
)
```

#### 2-body Ephemeris Generation
This package also has functionality to generate ephemerides for a set of orbits. We do not recommend you use this with
2-body propagated orbits as it will not be accurate for more than a few days. However, if you used a N-body propagator
such as PYOORB, you can feed in the propagated orbits to this function to generate ephemerides. We call the ephemeris generator
2-body because the light-time correction is applied using a 2-body propagator.

Because the ephemeris generator was written in Jax, we can also map covariances directly to the sky-plane. To do this, we propagate
the covariance matrices with the orbits. This is done by passing `covariance=True` to the propagator. The ephemeris generator will
then automatically map the propagated covariance matrices to the sky-plane.

```python
import numpy as np
from astropy import units as u

from adam_core.orbits.query import query_sbdb
from adam_core.propagator.adam_pyoorb import PYOORBPropagator
from adam_core.observers import Observers
from adam_core.dynamics import generate_ephemeris_2body
from adam_core.time import Timestamp

# Get orbits to propagate
object_ids = ["Duende", "Eros", "Ceres"]
orbits = query_sbdb(object_ids)

# Make sure PYOORB is ready
propagator = PYOORBPropagator()

# Define a set of observers and observation times
times = Timestamp.from_mjd(np.arange(59000, 60000), scale="tdb")
observers = Observers.from_code("I11", times)

# Propagate orbits with PYOORB (note that we are propagating with covariances)
propagated_orbits = propagator.propagate_orbits(
    orbits,
    times,
    chunk_size=100,
    max_processes=1,
    covariance=True,
)

# Now generate ephemerides with the 2-body ephemeris generator
ephemeris = generate_ephemeris_2body(
    propagated_orbits,
    observers,
)
```

#### Gravitational parameter
Both the 2-body propagation and 2-body ephemeris generation code will determine the correct graviational parameter
to use from each orbit's origin.

To see the gravitational parameter used for each orbit:
```python
from adam_core.orbits.query import query_sbdb

# Get orbit to propagate
object_ids = ["Duende", "Eros", "Ceres"]
orbits = query_sbdb(object_ids)

# Get the gravitational parameter (these will all be the same -- heliocentric)
mu = orbits.coordinates.origin.mu()
```

## Package Structure

```bash
adam_core
├── constants.py  # Shared constants
├── coordinates   # Coordinate classes and transformations
├── dynamics      # Numerical solutions
├── orbits        # Orbits class and query utilities
└── utils         # Utility classes like Indexable or conversions like times_from_df
```


