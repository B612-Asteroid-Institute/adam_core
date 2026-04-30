"""Single source of truth for per-API Rust-migration state.

This module is imported at runtime to expose the migration registry and by the
governance scripts under ``migration/scripts`` to produce benchmark, parity,
and status reports. There is no separate YAML registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, get_args

MigrationStatus = Literal[
    "legacy",
    "dual",
    "public-rust-default",
    "rust-only",
    "raw-kernel-only",
    "orchestration-rust-default",
]
Boundary = Literal["numpy", "arrow", "python+numpy", "python+quivr", "rust"]
DefaultBackend = Literal["legacy", "rust"]
ParityCoverage = Literal[
    "random-fuzz",
    "random-fuzz-excluded",
    "orchestration-implied",
    "targeted-tests",
    "manual-only",
    "not-covered",
]

PUBLIC_RUST_DEFAULT: Final[MigrationStatus] = "public-rust-default"
RAW_KERNEL_ONLY: Final[MigrationStatus] = "raw-kernel-only"
ORCHESTRATION_RUST_DEFAULT: Final[MigrationStatus] = "orchestration-rust-default"


@dataclass(frozen=True)
class PerfGate:
    min_speedup_p50: float = 1.2
    min_speedup_p95: float = 1.2


@dataclass(frozen=True)
class ApiMigration:
    api_id: str
    status: MigrationStatus
    boundary: Boundary
    default: DefaultBackend
    rust_module: str = ""
    perf_gate: PerfGate = PerfGate()
    waiver: str = ""
    parity_coverage: ParityCoverage = "not-covered"
    coverage_note: str = ""
    covered_subcases: tuple[str, ...] = ()
    excluded_subcases: tuple[str, ...] = ()
    current_legacy_impl: bool = False
    latency_gate: bool = False

    @property
    def is_rust_default(self) -> bool:
        return self.default == "rust"


API_MIGRATIONS: Final[tuple[ApiMigration, ...]] = (
    ApiMigration(
        api_id="coordinates.cartesian_to_spherical",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cartesian_to_spherical_numpy",
        waiver="waiver-20260428-cartesian-to-spherical-warm-performance-temporary",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="coordinates.cartesian_to_geodetic",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cartesian_to_geodetic_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="coordinates.cartesian_to_keplerian",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cartesian_to_keplerian_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="coordinates.keplerian.to_cartesian",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.keplerian_to_cartesian_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="coordinates.cartesian_to_cometary",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cartesian_to_cometary_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="coordinates.cometary.to_cartesian",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.cometary_to_cartesian_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="coordinates.transform_coordinates",
        status=PUBLIC_RUST_DEFAULT,
        boundary="python+numpy",
        default="rust",
        rust_module="adam_core._rust_native.transform_coordinates_numpy",
        parity_coverage="random-fuzz",
        covered_subcases=(
            "Public dispatcher CartesianCoordinates ecliptic->equatorial "
            "to SphericalCoordinates",
        ),
        excluded_subcases=(
            "Cartesian->Cartesian frame-only public dispatcher fallthrough "
            "(intentional: cached cartesian_to_frame path is faster)",
            "time-varying ITRF93 rotations",
            "origin translation and user-kernel SPICE body coverage",
            "remaining non-Cartesian representation conversions",
        ),
        coverage_note=(
            "Random fuzz now exercises the public quivr-object dispatcher for "
            "a supported Cartesian->Spherical frame-change workload. The "
            "remaining excluded subcases are explicit and should not be "
            "inferred as fuzz-covered."
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="coordinates.spherical.to_cartesian",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.spherical_to_cartesian_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="coordinates.transform_coordinates_with_covariance",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module=(
            "adam_core._rust_native.transform_coordinates_with_covariance_numpy"
        ),
        parity_coverage="targeted-tests",
        coverage_note=(
            "Raw covariance-transform kernel used by coordinate-class "
            "dispatch; not part of baseline-main random-fuzz manifest."
        ),
    ),
    ApiMigration(
        api_id="coordinates.rotate_cartesian_time_varying",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.rotate_cartesian_time_varying_numpy",
        parity_coverage="targeted-tests",
        coverage_note=(
            "Raw rotation helper covered by SPICE/backend targeted tests; "
            "not part of the baseline-main random-fuzz manifest."
        ),
    ),
    ApiMigration(
        api_id="coordinates.residuals.apply_cosine_latitude_correction",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module=("adam_core._rust_native.apply_cosine_latitude_correction_numpy"),
        parity_coverage="targeted-tests",
        coverage_note="Wave E2 residual helper; covered by residuals tests.",
    ),
    ApiMigration(
        api_id="coordinates.residuals.bound_longitude_residuals",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.bound_longitude_residuals_numpy",
        parity_coverage="targeted-tests",
        coverage_note="Wave E2 residual helper; covered by residuals tests.",
    ),
    ApiMigration(
        api_id="coordinates.residuals.calculate_chi2",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_chi2_numpy",
        parity_coverage="targeted-tests",
        coverage_note=(
            "Wave E2 OD-inner-loop kernel; RM-P1-013 owns the SPD covariance "
            "contract and any expanded baseline-main governance."
        ),
    ),
    ApiMigration(
        api_id="dynamics.calc_mean_motion",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_mean_motion_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="dynamics.propagate_2body",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.propagate_2body_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="dynamics.propagate_2body_along_arc",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.propagate_2body_along_arc_numpy",
        parity_coverage="targeted-tests",
        coverage_note=(
            "Warm-started arc helper behind propagation dispatch; not a "
            "separate baseline-main random-fuzz API."
        ),
    ),
    ApiMigration(
        api_id="dynamics.propagate_2body_arc_batch",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.propagate_2body_arc_batch_numpy",
        parity_coverage="targeted-tests",
        coverage_note=(
            "Batched warm-start propagation helper; not a separate "
            "baseline-main random-fuzz API."
        ),
    ),
    ApiMigration(
        api_id="dynamics.propagate_2body_with_covariance",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.propagate_2body_with_covariance_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="dynamics.generate_ephemeris_2body",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.generate_ephemeris_2body_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="dynamics.generate_ephemeris_2body_with_covariance",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module=(
            "adam_core._rust_native.generate_ephemeris_2body_with_covariance_numpy"
        ),
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="dynamics.add_light_time",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.add_light_time_numpy",
        parity_coverage="random-fuzz",
        latency_gate=False,
    ),
    ApiMigration(
        api_id="dynamics.calculate_moid",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_moid_numpy",
        parity_coverage="targeted-tests",
        coverage_note=(
            "Single-pair MOID kernel covered by dynamics MOID tests; "
            "calculate_perturber_moids governs the public orchestration."
        ),
    ),
    ApiMigration(
        api_id="dynamics.calculate_moid_batch",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_moid_batch_numpy",
        parity_coverage="targeted-tests",
        coverage_note=(
            "Rayon batch kernel used by calculate_perturber_moids; direct "
            "random-fuzz coverage is represented by the orchestration row."
        ),
    ),
    ApiMigration(
        api_id="dynamics.calculate_perturber_moids",
        status=ORCHESTRATION_RUST_DEFAULT,
        boundary="python+quivr",
        default="rust",
        rust_module="adam_core._rust_native.calculate_moid_batch_numpy",
        parity_coverage="orchestration-implied",
        coverage_note=(
            "Public quivr orchestration over the Rust MOID batch kernel. "
            "Baseline-main random fuzz is deferred because it needs a "
            "heavier Orbits/SPK subprocess adapter."
        ),
    ),
    ApiMigration(
        api_id="dynamics.solve_lambert",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.izzo_lambert_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="dynamics.tisserand_parameter",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.tisserand_parameter_numpy",
        parity_coverage="targeted-tests",
        coverage_note="Small public dynamics helper covered by unit tests.",
    ),
    ApiMigration(
        api_id="missions.porkchop_grid",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.porkchop_grid_numpy",
        parity_coverage="targeted-tests",
        coverage_note=(
            "Raw fused grid kernel behind generate_porkchop_data; public "
            "orchestration coverage is tracked separately."
        ),
    ),
    ApiMigration(
        api_id="dynamics.generate_porkchop_data",
        status=ORCHESTRATION_RUST_DEFAULT,
        boundary="python+quivr",
        default="rust",
        rust_module="adam_core._rust_native.porkchop_grid_numpy",
        parity_coverage="orchestration-implied",
        coverage_note=(
            "Public quivr orchestration over Rust Lambert/grid kernels. "
            "Baseline-main random fuzz is deferred because it needs an "
            "Orbits-quivr subprocess adapter."
        ),
    ),
    ApiMigration(
        api_id="photometry.calculate_phase_angle",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_phase_angle_numpy",
        waiver="waiver-20260428-photometry-warm-performance-temporary",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="photometry.calculate_apparent_magnitude_v",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_apparent_magnitude_v_numpy",
        waiver="waiver-20260428-photometry-warm-performance-temporary",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="photometry.calculate_apparent_magnitude_v_and_phase_angle",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module=(
            "adam_core._rust_native.calculate_apparent_magnitude_v_and_phase_angle_numpy"
        ),
        waiver="waiver-20260428-photometry-warm-performance-temporary",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="photometry.predict_magnitudes",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.predict_magnitudes_bandpass_numpy",
        waiver="waiver-20260428-photometry-warm-performance-temporary",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="photometry.fit_absolute_magnitude_rows",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.fit_absolute_magnitude_rows_numpy",
        parity_coverage="targeted-tests",
        coverage_note="Grouped H-fit production path covered by photometry tests.",
    ),
    ApiMigration(
        api_id="photometry.fit_absolute_magnitude_grouped",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.fit_absolute_magnitude_grouped_numpy",
        parity_coverage="targeted-tests",
        coverage_note="Grouped H-fit production path covered by photometry tests.",
    ),
    ApiMigration(
        api_id="orbit_determination.calcGibbs",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_gibbs_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="orbit_determination.calcHerrickGibbs",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_herrick_gibbs_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="orbit_determination.calcGauss",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_gauss_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="orbit_determination.gaussIOD",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.gauss_iod_fused_numpy",
        parity_coverage="random-fuzz-excluded",
        coverage_note=(
            "Runner adapters are retained for fixed-fixture/manual parity, "
            "but randomized fuzz is intentionally disabled because Rust "
            "Laguerre+deflation and legacy np.roots/LAPACK accept different "
            "physical root subsets on some random triplets."
        ),
        covered_subcases=("well-conditioned fixed fixtures/manual triplets",),
        excluded_subcases=(
            "random triplets with root-subset disagreement between "
            "Laguerre+deflation and np.roots/LAPACK",
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="orbits.classify_orbits",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.classify_orbits_numpy",
        parity_coverage="targeted-tests",
        coverage_note="Wave E1 helper covered by classification tests.",
    ),
    ApiMigration(
        api_id="statistics.weighted_mean",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.weighted_mean_numpy",
        parity_coverage="targeted-tests",
        coverage_note="Raw statistics helper used by migration/Wave E tests.",
    ),
    ApiMigration(
        api_id="statistics.weighted_covariance",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.weighted_covariance_numpy",
        parity_coverage="targeted-tests",
        coverage_note="Raw statistics helper used by migration/Wave E tests.",
    ),
)


API_MIGRATIONS_BY_ID: Final[dict[str, ApiMigration]] = {
    m.api_id: m for m in API_MIGRATIONS
}


def validate_api_migrations(
    migrations: tuple[ApiMigration, ...] = API_MIGRATIONS,
) -> None:
    """Validate registry invariants used by governance scripts."""

    seen: set[str] = set()
    errors: list[str] = []
    allowed_statuses = set(get_args(MigrationStatus))
    allowed_boundaries = set(get_args(Boundary))
    allowed_defaults = set(get_args(DefaultBackend))
    allowed_coverage = set(get_args(ParityCoverage))

    for migration in migrations:
        if migration.api_id in seen:
            errors.append(f"{migration.api_id}: duplicate registry row")
        seen.add(migration.api_id)

        if migration.status not in allowed_statuses:
            errors.append(f"{migration.api_id}: invalid status {migration.status!r}")
        if migration.boundary not in allowed_boundaries:
            errors.append(
                f"{migration.api_id}: invalid boundary {migration.boundary!r}"
            )
        if migration.default not in allowed_defaults:
            errors.append(f"{migration.api_id}: invalid default {migration.default!r}")
        if migration.parity_coverage not in allowed_coverage:
            errors.append(
                f"{migration.api_id}: invalid parity_coverage "
                f"{migration.parity_coverage!r}"
            )

        if migration.status == "dual" and not migration.current_legacy_impl:
            errors.append(
                f"{migration.api_id}: status='dual' requires current_legacy_impl=True"
            )
        if migration.status == "legacy" and migration.default == "rust":
            errors.append(f"{migration.api_id}: legacy status cannot default to rust")
        if migration.latency_gate and not migration.rust_module:
            errors.append(f"{migration.api_id}: latency_gate requires rust_module")
        if migration.default == "rust" and not migration.rust_module:
            errors.append(f"{migration.api_id}: rust default requires rust_module")
        if (
            migration.parity_coverage == "random-fuzz-excluded"
            and not migration.excluded_subcases
        ):
            errors.append(
                f"{migration.api_id}: random-fuzz-excluded requires excluded_subcases"
            )
        if (
            migration.parity_coverage
            in {"random-fuzz-excluded", "orchestration-implied", "not-covered"}
            and not migration.coverage_note
        ):
            errors.append(
                f"{migration.api_id}: {migration.parity_coverage} requires "
                "coverage_note"
            )

    if errors:
        raise RuntimeError("Invalid Rust migration registry:\n- " + "\n- ".join(errors))


validate_api_migrations()


API_MIGRATION_STATUS: Final[dict[str, dict[str, Any]]] = {
    m.api_id: {
        "status": m.status,
        "boundary": m.boundary,
        "default": m.default,
        "rust_module": m.rust_module,
        "waiver": m.waiver,
        "parity_coverage": m.parity_coverage,
        "coverage_note": m.coverage_note,
        "covered_subcases": m.covered_subcases,
        "excluded_subcases": m.excluded_subcases,
        "current_legacy_impl": m.current_legacy_impl,
        "latency_gate": m.latency_gate,
    }
    for m in API_MIGRATIONS
}
