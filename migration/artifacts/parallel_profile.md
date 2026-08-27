# Parallel Backend Profile (RM-WD3-001 step 2 — dynamics-only)

- started_at: 2026-05-07T21:01:25Z
- cpu_count: 10

Surfaces that touch the n-body line (ASSIST-backed Propagator, impacts, OD, IOD) are deliberately out of scope here. Their Ray defaults are deferred until the n-body line itself is migrated to Rust (RM-FUTURE-002).

## propagate_2body
- workload: n_orbits=999, n_times=100

| max_processes | cold | warm_median | warm_min | warm_max | n_warm |
|---|---|---|---|---|---|
| 1 | 59.4ms | 36.3ms  (seq/this 1.00x) | 34.8ms | 37.8ms | 2 |
| 4 | 1.976s | 103.7ms  (seq/this 0.35x) | 103.6ms | 103.7ms | 2 |
| 8 | 2.293s | 106.8ms  (seq/this 0.34x) | 101.6ms | 112.0ms | 2 |

## generate_ephemeris_2body
- workload: n_paired_rows=12960

| max_processes | cold | warm_median | warm_min | warm_max | n_warm |
|---|---|---|---|---|---|
| 1 | 19.1ms | 14.0ms  (seq/this 1.00x) | 13.9ms | 14.2ms | 2 |
| 4 | 2.214s | 571.5ms  (seq/this 0.02x) | 562.2ms | 580.8ms | 2 |
| 8 | 2.663s | 530.0ms  (seq/this 0.03x) | 506.9ms | 553.1ms | 2 |

## propagator.propagate_orbits
- workload: n_orbits=999, n_times=100, propagator=_HarnessPropagator(2body)

| max_processes | cold | warm_median | warm_min | warm_max | n_warm |
|---|---|---|---|---|---|
| 1 | 96.6ms | 80.1ms  (seq/this 1.00x) | 78.9ms | 81.3ms | 2 |
| 4 | 1.820s | 151.6ms  (seq/this 0.53x) | 138.7ms | 164.5ms | 2 |
| 8 | 2.377s | 155.8ms  (seq/this 0.51x) | 151.8ms | 159.7ms | 2 |

## propagator.generate_ephemeris
- workload: n_orbits=216, n_observers=60, propagator=_HarnessPropagator(2body)

| max_processes | cold | warm_median | warm_min | warm_max | n_warm |
|---|---|---|---|---|---|
| 1 | 66.5ms | 57.8ms  (seq/this 1.00x) | 56.4ms | 59.2ms | 2 |
| 4 | 1.695s | 54.4ms  (seq/this 1.06x) | 53.9ms | 54.8ms | 2 |
| 8 | 2.182s | 83.6ms  (seq/this 0.69x) | 81.5ms | 85.6ms | 2 |

