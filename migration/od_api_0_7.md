# Orbit-determination API migration for Core 0.7

Core 0.7 replaces the executable Python observation-model surface with one
complete, explicit Rust-owned configuration.

## Required changes

- Pass an `OdConfig` to each high-level `run_iod`, `run_od`, `refine`, or
  `evaluate_orbits` call. No scientific configuration is selected when the
  argument is omitted.
- Bind a named profile explicitly:

  ```python
  from adam_core.orbit_determination import ASTEROID_INSTITUTE_OD_V1, ResourceBindings

  config = ASTEROID_INSTITUTE_OD_V1.bind(ResourceBindings({
      "station_catalog_uncertainty_table": sigma_path,
      "efcc18_catalog_bias_table": efcc18_path,
      "station_band_residual_covariance_table": residual_path,
  }))
  ```

- For astrometry that is already prepared, construct a complete manual config
  with `ObservationPipeline.as_supplied()`. Its provenance records
  `transforms=[]`.
- For a manual transformed pipeline, use `ObservationPipeline.ordered(...)` and
  bind every referenced semantic resource through `ResourceBindings`.

## Removed without aliases

`models=`, `models="default"`, `observatory_bias_model=`,
`ObservationUncertaintyModel`, `CompositeModel`, `apply_observation_models`,
`initial_orbit_determination`, `iod`, `differential_correction`,
`NativeOrbitFitter`, `OrbitFitter`, and the public linkage APIs are not part of
Core 0.7. Python transform objects are immutable declarative values; transform
validation, resource parsing, execution, and provenance are native Rust.
