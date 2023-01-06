# Asteroid Institute ADAM Core

A set of shared astrodynamics libraries and utilities. 

`adam_core` is used by a variety of library and services at Asteroid Institute. Sharing these common classes, types, and conversions amongst our tools ensures consistency and accuracy.

## Usage

### Coordinates

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

