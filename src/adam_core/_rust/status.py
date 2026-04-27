"""Single source of truth for per-API Rust-migration state.

This module is imported at runtime to drive backend dispatch and by the
migration governance scripts (`migration/scripts/*`) to produce benchmark
gates and state reports. There is no separate YAML registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class PerfGate:
    min_speedup_p50: float = 1.2
    min_speedup_p95: float = 1.2


@dataclass(frozen=True)
class ApiMigration:
    api_id: str
    status: str  # "legacy", "dual", "rust-default"
    boundary: str  # "numpy", "arrow", "python+numpy"
    default: str  # "legacy" or "rust"
    rust_module: str
    perf_gate: PerfGate = PerfGate()
    waiver: str = ""


API_MIGRATIONS: Final[tuple[ApiMigration, ...]] = (
    ApiMigration(
        api_id="coordinates.cartesian_to_spherical",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cartesian_to_spherical_numpy",
    ),
    ApiMigration(
        api_id="coordinates.cartesian_to_geodetic",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cartesian_to_geodetic_numpy",
    ),
    ApiMigration(
        api_id="coordinates.cartesian_to_keplerian",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cartesian_to_keplerian_numpy",
    ),
    ApiMigration(
        api_id="coordinates.keplerian.to_cartesian",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.keplerian_to_cartesian_numpy",
    ),
    ApiMigration(
        api_id="coordinates.cartesian_to_cometary",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cartesian_to_cometary_numpy",
    ),
    ApiMigration(
        api_id="coordinates.cometary.to_cartesian",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cometary_to_cartesian_numpy",
    ),
    ApiMigration(
        api_id="coordinates.transform_coordinates",
        status="dual",
        boundary="python+numpy",
        default="rust",
        rust_module="adam_core._rust_native.transform_coordinates_numpy",
    ),
    ApiMigration(
        api_id="coordinates.spherical.to_cartesian",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.spherical_to_cartesian_numpy",
    ),
    ApiMigration(
        api_id="dynamics.calc_mean_motion",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_mean_motion_numpy",
    ),
    ApiMigration(
        api_id="dynamics.propagate_2body",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.propagate_2body_numpy",
    ),
    ApiMigration(
        api_id="dynamics.propagate_2body_with_covariance",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.propagate_2body_with_covariance_numpy",
    ),
    ApiMigration(
        api_id="dynamics.generate_ephemeris_2body",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.generate_ephemeris_2body_numpy",
    ),
    ApiMigration(
        api_id="dynamics.generate_ephemeris_2body_with_covariance",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.generate_ephemeris_2body_with_covariance_numpy",
    ),
    ApiMigration(
        api_id="photometry.calculate_phase_angle",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_phase_angle_numpy",
    ),
    ApiMigration(
        api_id="photometry.calculate_apparent_magnitude_v",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_apparent_magnitude_v_numpy",
    ),
    ApiMigration(
        api_id="photometry.calculate_apparent_magnitude_v_and_phase_angle",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_apparent_magnitude_v_and_phase_angle_numpy",
    ),
    ApiMigration(
        api_id="photometry.predict_magnitudes",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.predict_magnitudes_bandpass_numpy",
    ),
    ApiMigration(
        api_id="orbit_determination.calcGibbs",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_gibbs_numpy",
    ),
    ApiMigration(
        api_id="orbit_determination.calcHerrickGibbs",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_herrick_gibbs_numpy",
    ),
    ApiMigration(
        api_id="orbit_determination.calcGauss",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_gauss_numpy",
    ),
    ApiMigration(
        api_id="orbit_determination.gaussIOD",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.gauss_iod_fused_numpy",
    ),
    ApiMigration(
        api_id="dynamics.solve_lambert",
        status="dual",
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.izzo_lambert_numpy",
    ),
)


API_MIGRATION_STATUS: Final[dict[str, dict[str, str]]] = {
    m.api_id: {
        "status": m.status,
        "boundary": m.boundary,
        "default": m.default,
        "waiver": m.waiver,
    }
    for m in API_MIGRATIONS
}
