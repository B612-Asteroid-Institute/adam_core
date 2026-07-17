# Complete registered-parity performance table

> **Scope warning:** this is complete only for the 44 APIs explicitly registered in the parity harness plus 26 adam-assist benchmark lanes. It is **not** an inventory of the complete adam-core public Python surface; class methods such as `Orbits.group_by_orbit_id`, `VariantOrbits.collapse`, and `Ephemeris.link_to_observers` were not registered and therefore do not appear below. That broader audit is tracked by `personal-cmy.37`.

Generated from the canonical `rm-p1-023-canonical-variant-create-v1` artifacts. Ratios are **legacy / implementation**, so larger is faster. p50/p95 are shown together. “—” means no qualifying Rust-owned `Instant` adapter exists; a PyO3 call is not mislabeled as native Rust.

## adam-core

| Surface | Lane / workload | Python veneer vs legacy (p50 / p95) | Native Rust vs legacy (p50 / p95) | Native evidence |
|---|---|---:|---:|---|
| `coordinates.cartesian_to_spherical` | tiny-n: rows=10 | 9.69× / 10.74× | — / — | unavailable (personal-98v.1) |
| `coordinates.transform_coordinates` | tiny-n: rows=12 | 5.52× / 5.12× | 351.24× / 393.46× | measured |
| `coordinates.transform_coordinates_with_covariance` | tiny-n: rows=4 | 392.22× / 416.64× | — / — | unavailable (personal-98v.1) |
| `coordinates.cartesian_to_geodetic` | tiny-n: rows=10 | 16.43× / 16.50× | — / — | unavailable (personal-98v.1) |
| `coordinates.cartesian_to_keplerian` | tiny-n: rows=10 | 20.86× / 39.89× | — / — | unavailable (personal-98v.1) |
| `coordinates.keplerian.to_cartesian` | tiny-n: rows=10 | 3.29× / 2.70× | — / — | unavailable (personal-98v.1) |
| `coordinates.cartesian_to_cometary` | tiny-n: rows=10 | 2.64× / 2.50× | — / — | unavailable (personal-98v.1) |
| `coordinates.cometary.to_cartesian` | tiny-n: rows=10 | 4.80× / 3.42× | — / — | unavailable (personal-98v.1) |
| `coordinates.spherical.to_cartesian` | tiny-n: rows=10 | 9.66× / 10.78× | — / — | unavailable (personal-98v.1) |
| `coordinates.rotate_cartesian_time_varying` | tiny-n: rows=10 | 0.72× / 0.54× | — / — | unavailable (personal-98v.1) |
| `coordinates.residuals.Residuals.calculate` | tiny-n: rows=10 | 1.49× / 1.23× | 77.99× / 92.49× | measured |
| `coordinates.residuals.calculate_chi2` | tiny-n: rows=10 | 9.09× / 9.56× | — / — | unavailable (personal-98v.1) |
| `coordinates.residuals.bound_longitude_residuals` | tiny-n: rows=10 | 7.21× / 7.57× | — / — | unavailable (personal-98v.1) |
| `coordinates.residuals.apply_cosine_latitude_correction` | tiny-n: rows=10 | 6.11× / 6.11× | — / — | unavailable (personal-98v.1) |
| `statistics.weighted_mean` | tiny-n: rows=10 | 0.60× / 0.61× | — / — | unavailable (personal-98v.1) |
| `statistics.weighted_covariance` | tiny-n: rows=10 | 1.58× / 1.63× | — / — | unavailable (personal-98v.1) |
| `dynamics.calc_mean_motion` | tiny-n: rows=10 | 1.65× / 1.70× | — / — | unavailable (personal-98v.1) |
| `dynamics.tisserand_parameter` | tiny-n: rows=10 | 1.55× / 1.59× | — / — | unavailable (personal-98v.1) |
| `orbits.classify_orbits` | tiny-n: rows=10 | 10.23× / 10.22× | — / — | unavailable (personal-98v.1) |
| `dynamics.calculate_moid` | tiny-n: rows=1 | 93.81× / 101.52× | — / — | unavailable (personal-98v.1) |
| `dynamics.calculate_moid_batch` | tiny-n: rows=1 | 100.45× / 109.18× | — / — | unavailable (personal-98v.1) |
| `dynamics.calculate_perturber_moids` | tiny-n: rows=1 | 171.56× / 126.76× | 1332.36× / 1423.97× | measured |
| `missions.porkchop_grid` | tiny-n: departures=2 × arrivals=2 (4 rows) | 21.89× / 22.74× | — / — | unavailable (personal-98v.1) |
| `dynamics.generate_porkchop_data` | tiny-n: departures=2 × arrivals=2 (4 rows) | 2.30× / 1.92× | 243.83× / 271.50× | measured |
| `dynamics.propagate_2body` | tiny-n: rows=10 | 2.31× / 2.11× | 97.28× / 112.94× | measured |
| `dynamics.propagate_2body_along_arc` | tiny-n: rows=10 | 12.76× / 17.58× | — / — | unavailable (personal-98v.1) |
| `dynamics.propagate_2body_arc_batch` | tiny-n: orbits=2 × epochs=5 (10 rows) | 1.77× / 1.61× | — / — | unavailable (personal-98v.1) |
| `dynamics.propagate_2body_with_covariance` | tiny-n: rows=10 | 495.68× / 257.61× | — / — | unavailable (personal-98v.1) |
| `dynamics.generate_ephemeris_2body` | tiny-n: rows=10 | 5.17× / 4.83× | 57.89× / 41.95× | measured |
| `dynamics.generate_ephemeris_2body_with_covariance` | tiny-n: rows=10 | 8.37× / 7.66× | 85.34× / 63.90× | measured |
| `dynamics.solve_lambert` | tiny-n: rows=10 | 2.71× / 2.45× | — / — | unavailable (personal-98v.1) |
| `dynamics.add_light_time` | tiny-n: rows=10 | 1.70× / 1.25× | — / — | unavailable (personal-cmy.36.5) |
| `photometry.calculate_phase_angle` | tiny-n: rows=10 | 6.38× / 6.82× | — / — | unavailable (personal-98v.1) |
| `photometry.calculate_apparent_magnitude_v` | tiny-n: rows=10 | 5.11× / 5.68× | — / — | unavailable (personal-98v.1) |
| `photometry.calculate_apparent_magnitude_v_and_phase_angle` | tiny-n: rows=10 | 5.19× / 5.17× | — / — | unavailable (personal-98v.1) |
| `photometry.predict_magnitudes` | tiny-n: rows=10 | 5.15× / 5.93× | — / — | unavailable (personal-98v.1) |
| `photometry.fit_absolute_magnitude_rows` | tiny-n: rows=10 | 11.12× / 11.97× | — / — | unavailable (personal-98v.1) |
| `photometry.fit_absolute_magnitude_grouped` | tiny-n: rows=10 | 3.14× / 1.83× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.calcGibbs` | tiny-n: rows=10 | 41.84× / 41.47× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.calcHerrickGibbs` | tiny-n: rows=10 | 3.87× / 3.90× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.calcGauss` | tiny-n: rows=10 | 2.30× / 2.36× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.gaussIOD` | tiny-n: triplets=1 (1 rows) | 7.55× / 6.87× | 155.51× / 179.26× | measured |
| `orbits.VariantOrbits.create` | tiny-n: rows=10 | 14.65× / 10.76× | 315.80× / 328.04× | measured |
| `observers.Observers.from_codes` | tiny-n: rows=10 | 54.05× / 43.52× | 848.01× / 906.13× | measured |
| `coordinates.cartesian_to_spherical` | small-n: rows=2000 | 1.70× / 2.18× | — / — | unavailable (personal-98v.1) |
| `coordinates.transform_coordinates` | small-n: rows=2000 | 5.65× / 5.79× | 29.85× / 28.44× | measured |
| `coordinates.transform_coordinates_with_covariance` | small-n: rows=2000 | 157.41× / 124.39× | — / — | unavailable (personal-98v.1) |
| `coordinates.cartesian_to_geodetic` | small-n: rows=2000 | 4.35× / 2.60× | — / — | unavailable (personal-98v.1) |
| `coordinates.cartesian_to_keplerian` | small-n: rows=2000 | 3.41× / 3.11× | — / — | unavailable (personal-98v.1) |
| `coordinates.keplerian.to_cartesian` | small-n: rows=2000 | 31.88× / 21.03× | — / — | unavailable (personal-98v.1) |
| `coordinates.cartesian_to_cometary` | small-n: rows=2000 | 3.34× / 3.71× | — / — | unavailable (personal-98v.1) |
| `coordinates.cometary.to_cartesian` | small-n: rows=2000 | 30.16× / 16.91× | — / — | unavailable (personal-98v.1) |
| `coordinates.spherical.to_cartesian` | small-n: rows=2000 | 2.14× / 1.91× | — / — | unavailable (personal-98v.1) |
| `coordinates.rotate_cartesian_time_varying` | small-n: rows=2000 | 14.53× / 9.23× | — / — | unavailable (personal-98v.1) |
| `coordinates.residuals.Residuals.calculate` | small-n: rows=2000 | 2.34× / 2.42× | 5.84× / 6.10× | measured |
| `coordinates.residuals.calculate_chi2` | small-n: rows=2000 | 3.02× / 2.50× | — / — | unavailable (personal-98v.1) |
| `coordinates.residuals.bound_longitude_residuals` | small-n: rows=2000 | 4.15× / 4.26× | — / — | unavailable (personal-98v.1) |
| `coordinates.residuals.apply_cosine_latitude_correction` | small-n: rows=2000 | 5.27× / 5.86× | — / — | unavailable (personal-98v.1) |
| `statistics.weighted_mean` | small-n: rows=2000 | 0.15× / 0.13× | — / — | unavailable (personal-98v.1) |
| `statistics.weighted_covariance` | small-n: rows=2000 | 0.53× / 0.54× | — / — | unavailable (personal-98v.1) |
| `dynamics.calc_mean_motion` | small-n: rows=2000 | 7.59× / 7.64× | — / — | unavailable (personal-98v.1) |
| `dynamics.tisserand_parameter` | small-n: rows=2000 | 2.24× / 2.41× | — / — | unavailable (personal-98v.1) |
| `orbits.classify_orbits` | small-n: rows=2000 | 9.56× / 9.62× | — / — | unavailable (personal-98v.1) |
| `dynamics.calculate_moid` | small-n: rows=8 | 94.37× / 97.00× | — / — | unavailable (personal-98v.1) |
| `dynamics.calculate_moid_batch` | small-n: rows=8 | 204.00× / 161.50× | — / — | unavailable (personal-98v.1) |
| `dynamics.calculate_perturber_moids` | small-n: rows=8 | 332.07× / 269.00× | 2613.12× / 1906.50× | measured |
| `missions.porkchop_grid` | small-n: departures=44 × arrivals=44 (1936 rows) | 6.63× / 3.99× | — / — | unavailable (personal-98v.1) |
| `dynamics.generate_porkchop_data` | small-n: departures=44 × arrivals=44 (1936 rows) | 2.20× / 2.22× | 18.24× / 16.34× | measured |
| `dynamics.propagate_2body` | small-n: rows=2000 | 1.38× / 1.36× | 3.24× / 3.23× | measured |
| `dynamics.propagate_2body_along_arc` | small-n: rows=100 | 2.98× / 4.29× | — / — | unavailable (personal-98v.1) |
| `dynamics.propagate_2body_arc_batch` | small-n: orbits=40 × epochs=50 (2000 rows) | 17.53× / 15.17× | — / — | unavailable (personal-98v.1) |
| `dynamics.propagate_2body_with_covariance` | small-n: rows=2000 | 17983.12× / 12019.77× | — / — | unavailable (personal-98v.1) |
| `dynamics.generate_ephemeris_2body` | small-n: rows=2000 | 8.66× / 8.30× | 11.99× / 11.47× | measured |
| `dynamics.generate_ephemeris_2body_with_covariance` | small-n: rows=2000 | 8.31× / 7.80× | 10.48× / 9.74× | measured |
| `dynamics.solve_lambert` | small-n: rows=2000 | 30.25× / 19.55× | — / — | unavailable (personal-98v.1) |
| `dynamics.add_light_time` | small-n: rows=2000 | 3.22× / 3.08× | — / — | unavailable (personal-cmy.36.5) |
| `photometry.calculate_phase_angle` | small-n: rows=2000 | 3.74× / 5.36× | — / — | unavailable (personal-98v.1) |
| `photometry.calculate_apparent_magnitude_v` | small-n: rows=2000 | 4.08× / 3.47× | — / — | unavailable (personal-98v.1) |
| `photometry.calculate_apparent_magnitude_v_and_phase_angle` | small-n: rows=2000 | 4.00× / 4.89× | — / — | unavailable (personal-98v.1) |
| `photometry.predict_magnitudes` | small-n: rows=2000 | 3.52× / 4.04× | — / — | unavailable (personal-98v.1) |
| `photometry.fit_absolute_magnitude_rows` | small-n: rows=2000 | 3.37× / 3.60× | — / — | unavailable (personal-98v.1) |
| `photometry.fit_absolute_magnitude_grouped` | small-n: rows=2000 | 112.64× / 101.60× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.calcGibbs` | small-n: rows=2000 | 43.87× / 42.68× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.calcHerrickGibbs` | small-n: rows=2000 | 3.96× / 3.78× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.calcGauss` | small-n: rows=2000 | 2.39× / 2.31× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.gaussIOD` | small-n: triplets=16 (16 rows) | 8.65× / 7.66× | 169.60× / 177.05× | measured |
| `orbits.VariantOrbits.create` | small-n: rows=2000 | 248.48× / 217.97× | 343.85× / 318.55× | measured |
| `observers.Observers.from_codes` | small-n: rows=2000 | 44.74× / 34.80× | 55.06× / 40.72× | measured |
| `coordinates.cartesian_to_spherical` | large-n: rows=20000 | 3.44× / 2.93× | — / — | unavailable (personal-98v.1) |
| `coordinates.transform_coordinates` | large-n: rows=12000 | 8.40× / 8.22× | 18.70× / 16.93× | measured |
| `coordinates.transform_coordinates_with_covariance` | large-n: rows=4000 | 156.54× / 108.00× | — / — | unavailable (personal-98v.1) |
| `coordinates.cartesian_to_geodetic` | large-n: rows=20000 | 11.68× / 7.37× | — / — | unavailable (personal-98v.1) |
| `coordinates.cartesian_to_keplerian` | large-n: rows=20000 | 2.75× / 2.78× | — / — | unavailable (personal-98v.1) |
| `coordinates.keplerian.to_cartesian` | large-n: rows=20000 | 33.40× / 27.49× | — / — | unavailable (personal-98v.1) |
| `coordinates.cartesian_to_cometary` | large-n: rows=20000 | 3.93× / 4.07× | — / — | unavailable (personal-98v.1) |
| `coordinates.cometary.to_cartesian` | large-n: rows=20000 | 32.08× / 28.86× | — / — | unavailable (personal-98v.1) |
| `coordinates.spherical.to_cartesian` | large-n: rows=20000 | 5.77× / 4.86× | — / — | unavailable (personal-98v.1) |
| `coordinates.rotate_cartesian_time_varying` | large-n: rows=50000 | 36.35× / 32.61× | — / — | unavailable (personal-98v.1) |
| `coordinates.residuals.Residuals.calculate` | large-n: rows=20000 | 2.95× / 2.88× | 6.08× / 5.72× | measured |
| `coordinates.residuals.calculate_chi2` | large-n: rows=50000 | 11.48× / 8.17× | — / — | unavailable (personal-98v.1) |
| `coordinates.residuals.bound_longitude_residuals` | large-n: rows=100000 | 1.58× / 1.51× | — / — | unavailable (personal-98v.1) |
| `coordinates.residuals.apply_cosine_latitude_correction` | large-n: rows=50000 | 12.06× / 11.71× | — / — | unavailable (personal-98v.1) |
| `statistics.weighted_mean` | large-n: rows=50000 | 0.12× / 0.12× | — / — | unavailable (personal-98v.1) |
| `statistics.weighted_covariance` | large-n: rows=50000 | 0.73× / 0.97× | — / — | unavailable (personal-98v.1) |
| `dynamics.calc_mean_motion` | large-n: rows=50000 | 13.15× / 13.39× | — / — | unavailable (personal-98v.1) |
| `dynamics.tisserand_parameter` | large-n: rows=100000 | 3.32× / 2.30× | — / — | unavailable (personal-98v.1) |
| `orbits.classify_orbits` | large-n: rows=50000 | 3.75× / 2.93× | — / — | unavailable (personal-98v.1) |
| `dynamics.calculate_moid` | large-n: rows=64 | 93.65× / 89.84× | — / — | unavailable (personal-98v.1) |
| `dynamics.calculate_moid_batch` | large-n: rows=64 | 397.63× / 347.38× | — / — | unavailable (personal-98v.1) |
| `dynamics.calculate_perturber_moids` | large-n: rows=64 | 1602.69× / 1456.03× | 4386.02× / 3862.77× | measured |
| `missions.porkchop_grid` | large-n: departures=64 × arrivals=64 (4096 rows) | 8.58× / 7.20× | — / — | unavailable (personal-98v.1) |
| `dynamics.generate_porkchop_data` | large-n: departures=64 × arrivals=64 (4096 rows) | 2.49× / 2.19× | 16.91× / 13.33× | measured |
| `dynamics.propagate_2body` | large-n: orbits=1000 × epochs=20 (20000 rows) | 2.85× / 2.73× | 3.82× / 3.72× | measured |
| `dynamics.propagate_2body_along_arc` | large-n: rows=400 | 1.34× / 2.04× | — / — | unavailable (personal-98v.1) |
| `dynamics.propagate_2body_arc_batch` | large-n: orbits=400 × epochs=50 (20000 rows) | 22.42× / 20.60× | — / — | unavailable (personal-98v.1) |
| `dynamics.propagate_2body_with_covariance` | large-n: orbits=200 × epochs=20 (4000 rows) | 17960.46× / 13735.45× | — / — | unavailable (personal-98v.1) |
| `dynamics.generate_ephemeris_2body` | large-n: orbits=400 × epochs=50 (20000 rows) | 1.50× / 1.49× | 1.63× / 1.67× | measured |
| `dynamics.generate_ephemeris_2body_with_covariance` | large-n: orbits=200 × epochs=20 (4000 rows) | 3.18× / 3.34× | 3.98× / 4.20× | measured |
| `dynamics.solve_lambert` | large-n: rows=12000 | 34.34× / 23.81× | — / — | unavailable (personal-98v.1) |
| `dynamics.add_light_time` | large-n: orbits=400 × observers=50 (20000 rows) | 3.85× / 2.96× | — / — | unavailable (personal-cmy.36.5) |
| `photometry.calculate_phase_angle` | large-n: orbits=1000 × observers=50 (50000 rows) | 3.10× / 2.35× | — / — | unavailable (personal-98v.1) |
| `photometry.calculate_apparent_magnitude_v` | large-n: orbits=1000 × observers=50 (50000 rows) | 1.93× / 1.53× | — / — | unavailable (personal-98v.1) |
| `photometry.calculate_apparent_magnitude_v_and_phase_angle` | large-n: orbits=1000 × observers=50 (50000 rows) | 2.33× / 1.59× | — / — | unavailable (personal-98v.1) |
| `photometry.predict_magnitudes` | large-n: orbits=1000 × observers=50 (50000 rows) | 1.37× / 1.45× | — / — | unavailable (personal-98v.1) |
| `photometry.fit_absolute_magnitude_rows` | large-n: rows=50000 | 2.65× / 2.15× | — / — | unavailable (personal-98v.1) |
| `photometry.fit_absolute_magnitude_grouped` | large-n: rows=50000 | 251.97× / 225.17× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.calcGibbs` | large-n: triplets=5000 (5000 rows) | 43.37× / 42.60× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.calcHerrickGibbs` | large-n: triplets=5000 (5000 rows) | 3.91× / 3.81× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.calcGauss` | large-n: triplets=5000 (5000 rows) | 2.33× / 2.40× | — / — | unavailable (personal-98v.1) |
| `orbit_determination.gaussIOD` | large-n: triplets=128 (128 rows) | 7.75× / 7.99× | 164.60× / 164.27× | measured |
| `orbits.VariantOrbits.create` | large-n: rows=5000 | 230.72× / 219.22× | 320.33× / 284.79× | measured |
| `observers.Observers.from_codes` | large-n: rows=50000 | 22.48× / 19.64× | 25.34× / 25.89× | measured |

## adam-assist

These are the latest committed two-runtime artifacts for the same Rust implementation now owned by downstream `adam-assist` (`rust-migration-assist`, commit `1ffda5a`). The compatible public method was timed against pinned legacy adam-assist; Rust-internal adapters have not yet been implemented, so native values are honestly blank.

| Surface | Lane / workload | Python veneer vs legacy (p50 / p95) | Native Rust vs legacy (p50 / p95) | Native evidence |
|---|---|---:|---:|---|
| `ASSISTPropagator.propagate_orbits` | tiny: `tiny_sun_ecliptic_tdb_2x2_fixture_shape` (n_orbits=2, n_target_times=2, output_rows=4) | 5.75× / 6.07× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | tiny: `tiny_sun_ecliptic_tdb_8x8_same_epoch` (n_orbits=8, n_target_times=8, output_rows=64) | 5.91× / 6.15× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | tiny: `tiny_ssb_equatorial_utc_8x8_same_epoch` (n_orbits=8, n_target_times=8, output_rows=64) | 6.06× / 6.48× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | tiny: `tiny_variant_sun_ecliptic_tdb_8x4` (n_orbits=8, n_target_times=4, output_rows=32) | 7.07× / 7.07× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | small: `small_sun_ecliptic_tdb_40x50` (n_orbits=40, n_target_times=50, output_rows=2000) | 2.58× / 2.68× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | small: `small_ssb_equatorial_utc_40x50` (n_orbits=40, n_target_times=50, output_rows=2000) | 2.07× / 2.54× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | small: `small_variant_sun_ecliptic_tdb_40x50` (n_orbits=40, n_target_times=50, output_rows=2000) | 2.48× / 2.33× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_sun_ecliptic_tdb_1000x20` (n_orbits=1000, n_target_times=20, output_rows=20000) | 1.93× / 1.99× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_sun_ecliptic_tdb_400x50_arc_shape` (n_orbits=400, n_target_times=50, output_rows=20000) | 1.68× / 1.50× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_ssb_equatorial_utc_400x50_arc_shape` (n_orbits=400, n_target_times=50, output_rows=20000) | 1.46× / 1.33× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_variant_sun_ecliptic_tdb_400x50` (n_orbits=400, n_target_times=50, output_rows=20000) | 1.52× / 1.66× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_sun_ecliptic_tdb_200x100_1yr` (n_orbits=200, n_target_times=100, output_rows=20000) | 1.46× / 1.45× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_sun_ecliptic_tdb_400x50_unique_input_epochs` (n_orbits=400, n_target_times=50, output_rows=20000) | 2.26× / 2.26× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_ssb_equatorial_utc_400x50_unique_input_epochs` (n_orbits=400, n_target_times=50, output_rows=20000) | 2.59× / 2.40× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_ssb_equatorial_utc_200x100_1yr` (n_orbits=200, n_target_times=100, output_rows=20000) | 1.26× / 1.30× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_sun_ecliptic_tdb_200x100_1yr_unique_input_epochs` (n_orbits=200, n_target_times=100, output_rows=20000) | 1.57× / 1.71× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits` | large: `large_variant_sun_ecliptic_tdb_200x100_1yr` (n_orbits=200, n_target_times=100, output_rows=20000) | 1.41× / 1.42× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits(covariance=True)` | tiny: `tiny_cov_sigma_point_sun_ecliptic_tdb_4x3` (n_orbits=4, n_target_times=3, output_rows=12) | 12.78× / 11.36× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits(covariance=True)` | small: `small_cov_sigma_point_sun_ecliptic_tdb_25x20` (n_orbits=25, n_target_times=20, output_rows=500) | 12.11× / 11.62× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits(covariance=True)` | small: `small_cov_auto_sun_ecliptic_tdb_25x20` (n_orbits=25, n_target_times=20, output_rows=500) | 12.44× / 12.52× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits(covariance=True)` | small: `small_cov_monte_carlo_sun_ecliptic_tdb_10x10` (n_orbits=10, n_target_times=10, output_rows=100) | 9.96× / 11.03× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits(covariance=True)` | large: `large_cov_sigma_point_sun_ecliptic_tdb_100x50_1yr` (n_orbits=100, n_target_times=50, output_rows=5000) | 11.04× / 10.86× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.propagate_orbits(covariance=True)` | large: `large_cov_sigma_point_unique_input_epochs_50x25_1yr` (n_orbits=50, n_target_times=25, output_rows=1250) | 8.95× / 8.74× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.detect_collisions` | orbits=10, days=30, impacts=2 | 6.71× / 6.50× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.detect_collisions` | orbits=50, days=30, impacts=10 | 3.95× / 3.92× | — / — | unavailable (personal-98v.1) |
| `ASSISTPropagator.detect_collisions` | orbits=200, days=30, impacts=40 | 2.17× / 2.16× | — / — | unavailable (personal-98v.1) |
