# OD Integration Test Fixtures

This directory holds small observation sets used by integration tests for
the orbit-determination backends.

## Adding fixtures

Each fixture should be a Parquet file serialisable by a
`OrbitDeterminationObservations` quivr table, e.g.

```python
from adam_core.orbit_determination import OrbitDeterminationObservations

obs = OrbitDeterminationObservations.from_parquet("ceres_10obs.parquet")
```

### Naming convention

`<object_name>_<n>obs.parquet` — e.g. `ceres_10obs.parquet`.

### Recommended content

* 5–20 observations for a well-characterised MPC object (e.g. 1 Ceres,
  2 Pallas, or any numbered asteroid with a stable solution).
* Observations must have:
  * Non-NaN RA/Dec covariance sigmas.
  * Observer states populated (`Observers.from_codes`).
  * Time scale: UTC (observation times) with TDB observer states.

### How to generate

```python
import numpy as np
import pyarrow as pa
from adam_core.coordinates import CoordinateCovariances, Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbit_determination import OrbitDeterminationObservations
from adam_core.time import Timestamp

# Example: pull 10 MPC observations for 1 Ceres from astroquery
# (adapt as needed)
times = Timestamp.from_mjd([...], scale="utc")
sigmas = np.full((len(times), 6), np.nan)
sigmas[:, 1] = ...  # RA sigma in degrees
sigmas[:, 2] = ...  # Dec sigma in degrees
coords = SphericalCoordinates.from_kwargs(
    lon=[...], lat=[...], time=times,
    origin=Origin.from_kwargs(code=["XXX"] * len(times)),
    frame="equatorial",
    covariance=CoordinateCovariances.from_sigmas(sigmas),
)
observers = Observers.from_codes(codes=["XXX"] * len(times), times=times)
obs = OrbitDeterminationObservations.from_kwargs(
    id=[...], coordinates=coords, observers=observers
)
obs.to_parquet("ceres_10obs.parquet")
```

## TODO(od-module)

Add at least one fixture for integration regression tests:
- `ceres_10obs.parquet` — 10 MPC observations for 1 Ceres
  Expected reduced chi2 (find_orb ADAM mode): < 5.0
