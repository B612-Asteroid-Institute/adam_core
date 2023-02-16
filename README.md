# Asteroid Institute ADAM Core

A set of shared astrodynamics libraries and utilities. 

`adam_core` is used by a variety of library and services at Asteroid Institute. Sharing these common classes, types, and conversions amongst our tools ensures consistency and accuracy.

## Usage

### Orbits

To define an orbit: 
```python
from astropy.time import Time
from adam_core.coordinates import KeplerianCoordinates
from adam_core.orbits import Orbits

keplerian_elements = KeplerianCoordinates(
    times=Time(59000.0, scale="tdb", format="mjd"),
    a=1.0, 
    e=0.002,
    i=10.,
    raan=50.0,
    ap=20.0,
    M=30.0, 
)
orbits = Orbits(
    keplerian_elements,
    orbit_ids="1",
    object_ids="Test Orbit"
)
```
The underlying orbits class is 2 dimensional and can store elements and covariances for multiple orbits.

```python
from astropy.time import Time
from adam_core.coordinates import KeplerianCoordinates
from adam_core.orbits import Orbits

keplerian_elements = KeplerianCoordinates(
    times=Time([59000.0, 60000.0], scale="tdb", format="mjd"),
    a=[1.0, 3.0], 
    e=[0.002, 0.0],
    i=[10., 30.],
    raan=[50.0, 32.0],
    ap=[20.0, 94.0],
    M=[30.0, 159.0],
)
orbits = Orbits(
    keplerian_elements,
    orbit_ids=["1", "2"],
    object_ids=["Test Orbit 1", "Test Orbit 2"]
)
```

Orbits can be easily converted to a pandas DataFrame:
```python
orbits.to_df()
  orbit_id     object_id  mjd_tdb    a      e     i  raan    ap      M       origin     frame
0        1  Test Orbit 1  59000.0  1.0  0.002  10.0  50.0  20.0   30.0  heliocenter  ecliptic
1        2  Test Orbit 2  60000.0  3.0  0.000  30.0  32.0  94.0  159.0  heliocenter  ecliptic
```

Orbits can also be defined with uncertainties. 
```python
from astropy.time import Time
from adam_core.coordinates import KeplerianCoordinates
from adam_core.orbits import Orbits

keplerian_elements = KeplerianCoordinates(
    times=Time(59000.0, scale="tdb", format="mjd"),
    a=1.0, 
    e=0.002,
    i=10.,
    raan=50.0,
    ap=20.0,
    M=30.0, 
    sigma_a=0.002,
    sigma_e=0.001,
    sigma_i=0.01,
    sigma_raan=0.01,
    sigma_ap=0.1,
    sigma_M=0.1,
)

orbits = Orbits(
    keplerian_elements,
    orbit_ids=["1"],
    object_ids=["Test Orbit with Uncertainties"]
)
orbits.to_df(sigmas=True)
  orbit_id                      object_id  mjd_tdb    a  ...  sigma_ap  sigma_M       origin     frame
0        1  Test Orbit with Uncertainties  59000.0  1.0  ...       0.1      0.1  heliocenter  ecliptic
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
# Keplerian Elements
orbits.keplerian

# Cometary Elements
orbits.cometary

# Cartesian Elements
orbits.cartesian

# Spherical Elements
orbits.spherical
```

### Lower Level Classes
These classes are designed more for developers interested in building codes derived from utilities 
in this package. 

#### Coordinates

```python
from astropy.time import Time
from adam_core.coordinates import CartesianCoordinates, transform_coordinates

# Instantiate a Cartesian coordinate
time = Time(
    [cartesian.mjd_tdb],
    scale="tdb",
    format="mjd",
)
cartesian_coordinates = CartesianCoordinates(
    x=np.array([cartesian.x]),
    y=np.array([cartesian.y]),
    z=np.array([cartesian.z]),
    vx=np.array([cartesian.vx]),
    vy=np.array([cartesian.vy]),
    vz=np.array([cartesian.vz]),
    times=time,
    origin="heliocenter",
    frame="ecliptic",
)

keplerian_coordinates = transform_coordinates(
    cartesian_coordinates,
    representation_out="keplerian",
    frame_out="ecliptic"
)

print(keplerian_coordinates)
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

