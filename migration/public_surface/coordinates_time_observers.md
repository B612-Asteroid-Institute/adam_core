# Public-surface audit: coordinates, time, origins, covariance, and observers

Audit date: 2026-07-10  
Owner bead: `personal-cmy.37.2`  
Compared trees: migration branch `f27370b2` and baseline `/Users/aleck/Code/adam-core`  

## Scope and classification

This is a source audit of every adam-core-owned public class, constructor,
column/property, method, function, and constant in:

- `adam_core.coordinates` (including `covariances`, `origin`, `residuals`,
  `transform`, `units`, and `variants`),
- `adam_core.time`, and
- `adam_core.observers`.

Names beginning with `_` are excluded except when needed to explain a public
fallback. A table's declared columns and attributes are included because they
are public descriptors. The common quivr surface is inventoried separately
rather than silently credited to adam-core.

Classifications used below:

- **Rust-only**: callable directly as a Rust API without Python orchestration.
- **one-crossing**: a compatible Python veneer makes one Rust call; Python may
  validate or wrap the result.
- **mixed**: meaningful work, branching, or multiple backend calls occur on
  both sides of the boundary.
- **Python-only**: adam-core behavior is implemented in Python/NumPy/PyArrow.
- **plotting**: display/URL/UI behavior may intentionally stay Python.
- **external inheritance**: behavior comes unchanged from quivr, `Enum`,
  `dataclass`, or typing infrastructure and is not an adam-core Rust gap.

**Important result:** there are no Rust-only public APIs in this Python package
slice. The best migrated surfaces are one-crossing veneers over Rust APIs.
Only three audited parity rows currently have qualifying Rust-owned
`std::time::Instant` measurements: `transform_coordinates`,
`Residuals.calculate`, and `Observers.from_codes`.

`Parity` below means a dedicated baseline-main constitutional parity registry
row, not ordinary unit tests or indirect exercise by another row. `Native`
means a Rust-internal `Instant` adapter, not Python `perf_counter` around PyO3.

## Shared constructors, descriptors, and inherited table API

All quivr tables in this audit expose a generated constructor and the following
unchanged quivr operations. They are classified **external inheritance** and
are neither parity nor native-timing gaps:

`__init__`, `from_kwargs`, `from_pyarrow`, `from_csv`, `from_dataframe`,
`from_feather`, `from_flat_dataframe`, `from_parquet`, `empty`, `nulls`,
`as_column`, `with_table`, `column`, `set_column`, `apply_mask`, `take`,
`select`, `where`, `sort_by`, `drop_duplicates`, `unique_indices`, `validate`,
`is_valid`, `invalid_mask`, `null_mask`, `separate_invalid`, `attributes`,
`chunk_counts`, `fragmented`, `flattened_table`, `to_structarray`,
`to_dataframe`, `to_csv`, `to_feather`, and `to_parquet`.

This applies to `CartesianCoordinates`, `CometaryCoordinates`,
`CoordinateCovariances`, `GeodeticCoordinates`, `KeplerianCoordinates`,
`Origin`, `Residuals`, `SphericalCoordinates`, `Timestamp`,
`ObservatoryParallaxCoefficients`, and `Observers`. Their declared columns are
external quivr descriptors, while adam-core-derived properties are audited
below. None of these generated constructors has a dedicated parity row.

## Coordinate representation tables

| Public API(s) | Classification | Parity | Native | Finding / gap |
|---|---|---:|---:|---|
| `CartesianCoordinates` columns `x`, `y`, `z`, `vx`, `vy`, `vz`, `time`, `covariance`, `origin`; attribute `frame` | external inheritance | no | n/a | Schema descriptors; constructor/table mechanics are quivr-owned. |
| `CartesianCoordinates.values`, `r`, `v` | Python-only | no | no | NumPy/PyArrow extraction. |
| `CartesianCoordinates.r_mag`, `r_hat`, `v_mag`, `v_hat`, `h`, `h_mag` | Python-only | no | no | Adam-core vector algebra omitted from migration governance. |
| `CartesianCoordinates.sigma_x`, `sigma_y`, `sigma_z`, `sigma_vx`, `sigma_vy`, `sigma_vz`, `sigma_r`, `sigma_r_mag`, `sigma_v`, `sigma_v_mag` | Python-only | no | no | Derived uncertainty accessors. |
| `CartesianCoordinates.values_km`, `r_km`, `v_km_s`, `covariance_km` | Python-only | no | no | Python unit conversion. |
| `CartesianCoordinates.ric3_matrix`, `ric6_matrix` | Python-only | no | no | NumPy RIC construction; no Rust API/parity/timing. |
| `CartesianCoordinates.rotate` | one-crossing | raw kernel only (`coordinates.rotate_cartesian_time_varying`) | no (`personal-98v.1`) | Public method wraps one Rust state/covariance rotation, but public method construction/near-zero semantics have no dedicated row. |
| `CartesianCoordinates.translate` | Python-only | indirect only | no | Masked-array addition and table reconstruction remain Python. |
| `CartesianCoordinates.to_cometary`, `from_cometary`, `to_keplerian`, `from_keplerian`, `to_spherical`, `from_spherical` | one-crossing | only raw representative directions | no | Thin delegation reaches one Rust representation kernel; public aliases and covariance-bearing variants are not independently registered. |
| `CometaryCoordinates` columns `q`, `e`, `i`, `raan`, `ap`, `tp`, `time`, `covariance`, `origin`; attribute `frame` | external inheritance | no | n/a | Quivr schema/constructor. |
| `CometaryCoordinates.values`, `sigma_q`, `sigma_e`, `sigma_i`, `sigma_raan`, `sigma_ap`, `sigma_tp` | Python-only | no | no | Arrow-to-NumPy/uncertainty accessors. |
| `CometaryCoordinates.a`, `Q`, `p`, `P` (including rejecting setters/deleters) | Python-only | no | no | Derived orbital quantities and mutation contracts remain NumPy/Python. |
| `CometaryCoordinates.n` (including rejecting setter/deleter) | one-crossing | indirect via `dynamics.calc_mean_motion` | no | Rust numeric kernel plus Python degree conversion; public property contract omitted. |
| `CometaryCoordinates.to_cartesian`, `from_cartesian` | one-crossing | yes | no (`personal-98v.1`) | Rust state/covariance transform plus Python table veneer. |
| `CometaryCoordinates.to_keplerian`, `from_keplerian`, `to_spherical`, `from_spherical` | mixed | no | no | Two representation crossings through Cartesian plus Python reconstruction. |
| `KeplerianCoordinates` columns `a`, `e`, `i`, `raan`, `ap`, `M`, `time`, `covariance`, `origin`; attribute `frame` | external inheritance | no | n/a | Quivr schema/constructor. |
| `KeplerianCoordinates.values`, `sigma_a`, `sigma_e`, `sigma_i`, `sigma_raan`, `sigma_ap`, `sigma_M` | Python-only | no | no | Arrow-to-NumPy/uncertainty accessors. |
| `KeplerianCoordinates.q`, `Q`, `p`, `P` (including rejecting setters/deleters) | Python-only | no | no | Derived orbital quantities and mutation contracts remain NumPy/Python. |
| `KeplerianCoordinates.n` (including rejecting setter/deleter) | one-crossing | indirect via `dynamics.calc_mean_motion` | no | Rust numeric kernel plus Python degree conversion; property omitted. |
| `KeplerianCoordinates.to_cartesian`, `from_cartesian` | one-crossing | `to_cartesian` and raw cart→kep only | no (`personal-98v.1`) | Rust state/covariance transform plus Python table veneer. |
| `KeplerianCoordinates.to_cometary`, `from_cometary`, `to_spherical`, `from_spherical` | mixed | no | no | Two representation crossings through Cartesian. |
| `SphericalCoordinates` columns `rho`, `lon`, `lat`, `vrho`, `vlon`, `vlat`, `time`, `covariance`, `origin`; attribute `frame` | external inheritance | no | n/a | Quivr schema/constructor. |
| `SphericalCoordinates.values`, `sigma_rho`, `sigma_lon`, `sigma_lat`, `sigma_vrho`, `sigma_vlon`, `sigma_vlat` | Python-only | no | no | Arrow-to-NumPy/uncertainty accessors. |
| `SphericalCoordinates.to_unit_sphere` | Python-only | no | no | Mutates a NumPy copy and rebuilds a table. |
| `SphericalCoordinates.to_cartesian`, `from_cartesian` | one-crossing | `to_cartesian` and raw cart→sph only | no (`personal-98v.1`) | Rust state/covariance transform plus Python table veneer. |
| `SphericalCoordinates.to_cometary`, `from_cometary`, `to_keplerian`, `from_keplerian` | mixed | no | no | Two representation crossings through Cartesian. |
| `SphericalCoordinates.from_spherical` | Python-only | no | no | Identity compatibility constructor. |
| `GeodeticCoordinates` columns `alt`, `lon`, `lat`, `vup`, `veast`, `vnorth`, `time`, `covariance`, `origin`; attribute `frame` | external inheritance | no | n/a | Quivr schema/constructor. |
| `GeodeticCoordinates.values`, `sigma_alt`, `sigma_lon`, `sigma_lat`, `sigma_vup`, `sigma_veast`, `sigma_vnorth` | Python-only | no | no | Arrow-to-NumPy/uncertainty accessors. |
| `GeodeticCoordinates.google_maps_url` | plotting | no | n/a | Intentional display/external-URL helper; exempt from Rust migration. |
| `GeodeticCoordinates.from_cartesian` | one-crossing | raw `cartesian_to_geodetic` only | no (`personal-98v.1`) | Rust state/covariance transform; assertions and table assembly remain veneer. |
| `GeodeticConstants(a,b,f)` and fields; `WGS84` | external inheritance | no | n/a | Dataclass-generated constructor/fields and a static data constant. Preserve in compatibility tests, not Rust migration governance. |

## Covariance, residual, units, and variants

| Public API(s) | Classification | Parity | Native | Finding / gap |
|---|---|---:|---:|---|
| `CoordinateCovariances.values` | external inheritance | no | n/a | Quivr column descriptor. |
| `CoordinateCovariances.sigmas`, `to_matrix`, `from_matrix`, `from_sigmas`, `nulls`, `is_all_nan` | Python-only | no | no | Core covariance construction/access remains Python/PyArrow/NumPy. |
| `sigmas_to_covariances` | Python-only | no | no | NumPy diagonal expansion. |
| `make_positive_semidefinite` | Python-only | no | no | NumPy eigendecomposition. |
| `sample_covariance_random` | Python-only | no | no | SciPy sampling. |
| `sample_covariance_sigma_points` | Python-only | no | no | Python/NumPy/SciPy square-root sampling. |
| `weighted_mean`, `weighted_covariance` | Python-only (BLAS policy) | yes under `statistics.*` aliases | no (`personal-98v.1`) | Deliberately retained NumPy/BLAS, but the ownership exception must be explicit; not Rust/one-crossing. |
| `transform_covariances_sampling` | Python-only | no | no | Per-row Python sampling loop. |
| `rust_covariance_transform` | one-crossing | yes as `coordinates.transform_coordinates_with_covariance` | no (`personal-98v.1`) | Direct Rust forward-AD veneer. |
| `Residuals` columns `values`, `chi2`, `dof`, `probability` | external inheritance | no | n/a | Quivr schema/constructor. |
| `compute_residuals_ndarray` | mixed | no | no | Python subtraction/orchestration; spherical longitude helper crosses Rust separately. |
| `apply_cosine_latitude_correction`, `bound_longitude_residuals`, `calculate_chi2` | one-crossing | yes | no (`personal-98v.1`) | Direct Rust numeric veneers. |
| `Residuals.calculate` | one-crossing for built-in coordinate classes; mixed fallback for `custom_coordinates=True` | yes | **yes** | Normal path is Arrow-native; custom path retains NumPy/SciPy assembly. |
| `Residuals.to_array`, `calculate_reduced_chi2` | Python-only | no | no | NumPy/PyArrow reductions omitted from parity. |
| `au_to_km`, `km_to_au`, `au_per_day_to_km_per_s`, `km_per_s_to_au_per_day` | Python-only | no | no | Scalar/vector arithmetic. |
| `convert_cartesian_values_au_to_km`, `convert_cartesian_values_km_to_au`, `convert_cartesian_covariance_au_to_km`, `convert_cartesian_covariance_km_to_au` | Python-only | no | no | NumPy copies/scaling. |
| `VariantCoordinatesTable.index`, `sample`, `weight`, `weight_cov` | external inheritance | no | n/a | Typing-only `Protocol`; no runtime implementation. |
| `create_coordinate_variants` | Python-only | no | no | Row loop, sampling policy, and dynamic quivr class/table assembly remain Python. |

## Transform functions

| Public API(s) | Classification | Parity | Native | Finding / gap |
|---|---|---:|---:|---|
| `clear_translation_cache` | Python-only | no | no | Cache governance helper; not numeric but adam-core-owned. |
| `cartesian_to_geodetic`, `cartesian_to_spherical`, `spherical_to_cartesian`, `cartesian_to_keplerian`, `keplerian_to_cartesian`, `cartesian_to_cometary`, `cometary_to_cartesian` | one-crossing | yes (registered under the representative IDs listed in the parity registry) | no (`personal-98v.1`) | Direct NumPy-compatible Rust veneers; all lack qualifying native timing. |
| `cartesian_to_origin` | mixed | indirect in `transform_coordinates` only | no | Python origin grouping/cache/state resolution followed by Python `translate`. |
| `apply_time_varying_rotation` | mixed | raw rotation and dispatcher subcases only | no | SPICE setup/dedup/index/unit conversion/table assembly surround a Rust apply kernel. |
| `cartesian_to_frame` | mixed | indirect in dispatcher only | no | Python branch router over constant/time-varying rotations. |
| `transform_coordinates` | one-crossing for the supported Arrow matrix; mixed for deliberate legacy fallthrough | yes, broad branch fixture | **yes** | Best-covered facade, but its fallback prevents an unconditional one-crossing classification. |

## Origin types and constants

| Public API(s) | Classification | Parity | Native | Finding / gap |
|---|---|---:|---:|---|
| `OriginCodes` constructor, iteration, comparisons, and members `SOLAR_SYSTEM_BARYCENTER`, `MERCURY_BARYCENTER`, `VENUS_BARYCENTER`, `EARTH_MOON_BARYCENTER`, `MARS_BARYCENTER`, `JUPITER_BARYCENTER`, `SATURN_BARYCENTER`, `URANUS_BARYCENTER`, `NEPTUNE_BARYCENTER`, `SUN`, `MERCURY`, `VENUS`, `EARTH`, `MOON`, `MARS`, `JUPITER`, `SATURN`, `URANUS`, `NEPTUNE` | external inheritance | no | n/a | Standard-library `Enum`; static compatibility data. |
| `OriginGravitationalParameters` constructor/float behavior and members `MERCURY_BARYCENTER`, `VENUS_BARYCENTER`, `MARS_BARYCENTER`, `JUPITER_BARYCENTER`, `SATURN_BARYCENTER`, `URANUS_BARYCENTER`, `NEPTUNE_BARYCENTER`, `PLUTO_BARYCENTER`, `SUN`, `MERCURY`, `VENUS`, `EARTH`, `MOON` | external inheritance | no | n/a | `float, Enum` behavior plus static data. |
| `OriginGravitationalParameters.SOLAR_SYSTEM_BARYCENTER` | Python-only | no | no | Python sum of enum constants. |
| `Origin.code` | external inheritance | no | n/a | Quivr descriptor. |
| `Origin.from_OriginCodes`, `as_OriginCodes`, `mu` | Python-only | no | no | PyArrow/Python enum conversion and per-code loop. |

## Timestamp

| Public API(s) | Classification | Parity | Native | Finding / gap |
|---|---|---:|---:|---|
| `Timestamp` columns `days`, `nanos`; attribute `scale` | external inheritance | no | n/a | Quivr schema/constructor. |
| `Timestamp.micros`, `millis`, `seconds`, `fractional_days`, `jd` | Python-only | no | no | PyArrow arithmetic. |
| `Timestamp.key`, `signature`, `cache_digest` | Python-only | no | no | NumPy/hash/cache helpers. |
| `Timestamp.mjd`, `from_mjd` | mixed | fixture/unit tests only | no | Dense input takes one Rust call; null-bearing input retains PyArrow fallback. |
| `Timestamp.from_jd`, `from_et`, `et`, `to_numpy` | mixed | no | no | Python/PyArrow normalization around Rust-backed MJD/rescale operations. |
| `Timestamp.to_iso8601`, `from_iso8601`, `from_astropy`, `to_astropy` | Python-only / external Astropy | no | no | Explicit external interoperability boundary; semantic parity still needs compatibility coverage, but direct Rust timing is not required. |
| `Timestamp.rounded`, `equals`, `equals_scalar`, `equals_array` | mixed | no | no | PyArrow policy composed with Rust-backed differences. |
| `Timestamp.max`, `min`, `unique` | Python-only | no | no | PyArrow reduction/grouping. |
| `Timestamp.add_nanos`, `add_days`, `add_fractional_days`, `difference_scalar`, `difference` | one-crossing | fixture/unit tests only | no | Direct Rust integer-time kernels; completely absent from constitutional parity/native timing. |
| `Timestamp.add_seconds`, `add_millis`, `add_micros` | one-crossing | no | no | PyArrow unit conversion followed by one Rust `add_nanos` call. |
| `Timestamp.rescale` | mixed | frozen fixture only | no | Rust handles TAI/TT/UTC/TDB; UT1 delegates to Astropy/IERS. |
| `Timestamp.rescale_astropy` | Python-only / external Astropy | no | no | Explicit oracle/fallback. |
| `Timestamp.link` | Python-only / external quivr | no | no | Python rescale/round and `MultiKeyLinkage` construction. |
| constants `SCALES` | Python-only data | no | n/a | Public module constant; compatibility data, not a hot numeric API. |

## Observers and observatory state

| Public API(s) | Classification | Parity | Native | Finding / gap |
|---|---|---:|---:|---|
| `ObservatoryParallaxCoefficients` columns `code`, `longitude`, `cos_phi`, `sin_phi`, `name` | external inheritance | no | n/a | Quivr schema/constructor. |
| `ObservatoryParallaxCoefficients.lon_lat` | Python-only | no | no | NumPy geodetic conversion. |
| `ObservatoryParallaxCoefficients.timezone` | Python-only / external data | no | no | Per-row `timezonefinder` lookup. |
| constants `R_EARTH_EQUATORIAL`, `R_EARTH_POLAR`, `E_EARTH`, `OBSCODES`, `OBSERVATORY_PARALLAX_COEFFICIENTS`, `OBSERVATORY_CODES` | Python-only data | no | n/a | Import-time pandas/data-package construction; data/discovery concern, not numeric Rust credit. |
| `Observers` columns `code`, `coordinates` | external inheritance | no | n/a | Quivr schema/constructor. |
| `Observers.from_codes` | one-crossing for supported ground MPC codes; mixed fallback otherwise | yes | **yes** | Rust Arrow fast path, Python per-code fallback for unsupported/space/custom cases. |
| `Observers.from_code` | mixed | indirect via `from_codes` only | no | Python type dispatch; OriginCodes path uses observer-state orchestration. |
| `Observers.iterate_codes` | Python-only | no | no | Python generator over quivr selections. |
| `clear_observer_state_cache` | Python-only | no | no | Cache governance helper. |
| `get_mpc_observer_state` | mixed | indirect via `Observers.from_codes` ground cases | no | Python geodetic/cache/frame orchestration around several Rust SPICE calls; not one crossing. |
| `get_observer_state` | mixed | no dedicated row | no | Python dispatcher among perturber, MPC, and custom SPICE body paths. |
| `calculate_observing_night` | Python-only / external Astropy/zoneinfo | no | no | Per-code Python loop, timezone lookup, datetime/Astropy conversion. Note: importable from `adam_core.observers`, but omitted from that package's `__all__`. |

## Implementation beads created from this audit

Each non-plotting Python/mixed implementation or governance gap above is owned
by a concrete child of `personal-cmy.37.2`:

| Child | Concrete closure scope |
|---|---|
| `personal-cmy.37.2.1` | Cartesian vectors, uncertainties, RIC, unit conversion, and translation |
| `personal-cmy.37.2.2` | Keplerian/Cometary derived orbital properties and mutation contracts |
| `personal-cmy.37.2.3` | Coordinate value/sigma accessors and spherical unit-sphere operation |
| `personal-cmy.37.2.4` | Multi-crossing chained representation conversions |
| `personal-cmy.37.2.5` | Public parity/native timing for direct conversion veneers and aliases |
| `personal-cmy.37.2.6` | `CoordinateCovariances` construction and access |
| `personal-cmy.37.2.7` | Covariance PSD, sampling, weighting, and sampling-transform algorithms/explicit exceptions |
| `personal-cmy.37.2.8` | `create_coordinate_variants` one-crossing workflow |
| `personal-cmy.37.2.9` | Origin/frame translation and transform fallthrough |
| `personal-cmy.37.2.10` | Residual convenience APIs, custom fallback, parity, and native timing |
| `personal-cmy.37.2.11` | Origin conversion and gravitational-parameter methods |
| `personal-cmy.37.2.12` | Timestamp conversions, keys, comparisons, and reductions |
| `personal-cmy.37.2.13` | Timestamp arithmetic parity and Rust-Instant timing |
| `personal-cmy.37.2.14` | Timestamp rescale plus explicit Astropy/quivr interoperability governance |
| `personal-cmy.37.2.15` | Observatory metadata, timezone, and observing-night calculation |
| `personal-cmy.37.2.16` | Observer-state dispatch and `Observers.from_codes` fallbacks |
| `personal-cmy.37.2.17` | `Observers.from_code` and `iterate_codes` |

No child was created for `GeodeticCoordinates.google_maps_url` (plotting/display),
unchanged quivr operations, standard Enum/dataclass/Protocol mechanics, or static
constant data. Those exclusions are explicit classifications, not migration
credit.

## Closure summary

### Already acceptable runtime shapes

- Direct transform numeric functions, Rust covariance transform, residual numeric
  helpers, and dense integer timestamp arithmetic are one-crossing veneers.
- `Residuals.calculate`, `transform_coordinates`, and `Observers.from_codes`
  have Arrow-native normal paths and qualifying native timing.
- Plot/display (`google_maps_url`), Astropy interoperability, unchanged quivr
  table operations, enum mechanics, protocol declarations, and static constants
  are explicitly classified rather than counted as Rust migration wins.

### Non-plotting gaps

Every remaining gap is represented by a child bead of `personal-cmy.37.2`.
Those beads require direct Rust or a one-crossing veneer, dedicated parity for
public semantics, and Rust-owned `Instant` timing when the API is a qualifying
Rust numeric/table operation. Deliberate BLAS and external-library exceptions
must remain explicit and tested rather than being reported as Rust-backed.
