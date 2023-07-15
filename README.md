# Asteroid Institute ADAM Core

A set of shared astrodynamics libraries and utilities.

`adam_core` is used by a variety of library and services at Asteroid Institute. Sharing these common classes, types, and conversions amongst our tools ensures consistency and accuracy.

## Usage

### Orbits

To define an orbit:
```python
from astropy.time import Time
from adam_core.coordinates import KeplerianCoordinates
from adam_core.coordinates import Times
from adam_core.coordinates import Origin
from adam_core.orbits import Orbits

keplerian_elements = KeplerianCoordinates.from_kwargs(
    time=Times.from_astropy(
        Time([59000.0], scale="tdb", format="mjd")
    ),
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
from astropy.time import Time
from adam_core.coordinates import KeplerianCoordinates
from adam_core.coordinates import Times
from adam_core.coordinates import Origin
from adam_core.orbits import Orbits

keplerian_elements = KeplerianCoordinates.from_kwargs(
    time=Times.from_astropy(
        Time([59000.0, 60000.0], scale="tdb", format="mjd")
    ),
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
 orbit_ids	object_ids	times.jd1	times.jd2	x	y	z	vx	vy	vz	...	cov_vy_y	cov_vy_z	cov_vy_vx	cov_vy_vy	cov_vz_x	cov_vz_y	cov_vz_z	cov_vz_vx	cov_vz_vy	cov_vz_vz
0	1	Test Orbit 1	2459000.0	0.5	-0.166403	0.975273	0.133015	-0.016838	-0.003117	0.001921	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	2	Test Orbit 2	2460000.0	0.5	0.572777	-2.571820	-1.434457	0.009387	0.002900	-0.001452	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
```

Orbits can also be defined with uncertainties.
```python
import numpy as np
from astropy.time import Time
from adam_core.coordinates import KeplerianCoordinates
from adam_core.coordinates import Times
from adam_core.coordinates import Origin
from adam_core.coordinates import CoordinateCovariances
from adam_core.orbits import Orbits

keplerian_elements = KeplerianCoordinates.from_kwargs(
    time=Times.from_astropy(
        Time([59000.0], scale="tdb", format="mjd")
    ),
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
orbits.to_dataframe(sigmas=True)
 orbit_ids	object_ids	times.jd1	times.jd2	x	y	z	vx	vy	vz	...	cov_vy_y	cov_vy_z	cov_vy_vx	cov_vy_vy	cov_vz_x	cov_vz_y	cov_vz_z	cov_vz_vx	cov_vz_vy	cov_vz_vz
0	1	Test Orbit with Uncertainties	2459000.0	0.5	-0.166403	0.975273	0.133015	-0.016838	-0.003117	0.001921	...	3.625729e-08	-1.059731e-08	-9.691716e-11	1.872922e-09	1.392222e-08	-1.759744e-09	-1.821839e-09	-7.865582e-11	2.237521e-10	3.971297e-11
```

To query orbits from JPL Horizons:
```python
from astropy.time import Time
from adam_core.orbits.query import query_horizons

times = Time([60000.0], scale="tdb", format="mjd")
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
The propagator class in `adam_core` provides a generalized interface to the supported orbit integrators and ephemeris generators. By default,
`adam_core` ships with PYOORB.

To propagate orbits with PYOORB (here we grab some orbits from Horizons first):
```python
import numpy as np
from astropy.time import Time
from astropy import units as u
from adam_core.orbits.query import query_horizons
from adam_core.propagator import PYOORB

# Get orbit to propagate
initial_time = Time([60000.0], scale="tdb", format="mjd")
object_ids = ["Duende", "Eros", "Ceres"]
orbits = query_horizons(object_ids, initial_time)

# Make sure PYOORB is ready
propagator = PYOORB()

# Define propagation times
times = initial_time + np.arange(0, 100) * u.d

# Propagate orbits! This function supports multiprocessing for large
# propagation jobs.
propagated_orbits = propagator.propagate_orbits(
    orbits,
    times,
    chunk_size=100,
    max_processes=1,
)
```

### Low-level APIs

#### State Vectors from Development Ephemeris files
Getting the heliocentric ecliptic state vector of a DE440 body at a given set of times (in this case the barycenter of the Jovian system):
```python
import numpy as np
from astropy.time import Time

from adam_core.coordinates.origin import OriginCodes
from adam_core.utils.spice import get_perturber_state

states = get_perturber_state(
    OriginCodes.JUPITER_BARYCENTER,
    Time(np.arange(59000, 60000), format="mjd", scale="tdb"),
    frame="ecliptic",
    origin=OriginCodes.SUN,
)
```

#### 2-body Propagation
`adam_core` also has 2-body propagation functionality. To propagate any orbit with 2-body dynamics:
```python
import numpy as np
from astropy.time import Time
from astropy import units as u
from adam_core.orbits.query import query_sbdb
from adam_core.dynamics.propagation import propagate_2body

# Get orbit to propagate
object_ids = ["Duende", "Eros", "Ceres"]
orbits = query_sbdb(object_ids)

# Define propagation times
times = Time(np.arange(59000, 60000), scale="tdb", format="mjd")

# Propagate orbits with 2-body dynamics
propagated_orbits = propagate_2body(
    orbits,
    times
)
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

## Installation

ADAM Core is available on PyPI

```bash
pip install adam_core
```

## Development

Development is made easy with our Docker container environment.

```bash
# Build the container
docker compose build

# Run tests in the container
docker compose run adam_core pytest .

# Run a shell in the container
docker compose run adam_core bash
```
