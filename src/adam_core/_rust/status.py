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
    "fixed-fixture",
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
            "and equatorial->ecliptic to SphericalCoordinates",
            "Public dispatcher SphericalCoordinates ecliptic->equatorial "
            "to CartesianCoordinates",
            "Public dispatcher KeplerianCoordinates ecliptic->equatorial "
            "to SphericalCoordinates",
            "Public dispatcher CometaryCoordinates equatorial->ecliptic "
            "to KeplerianCoordinates",
            "Public dispatcher SUN->EARTH and EARTH->SUN origin translations "
            "to SphericalCoordinates",
            "Public dispatcher Earth-centered ecliptic->ITRF93 and "
            "ITRF93->equatorial time-varying rotations to SphericalCoordinates",
            "Public dispatcher covariance-bearing Cartesian/Keplerian inputs "
            "through constant-frame and SUN->EARTH origin-translation outputs",
        ),
        excluded_subcases=(
            "Cartesian->Cartesian frame-only public dispatcher fallthrough "
            "(intentional: cached cartesian_to_frame path is faster)",
            "covariance-bearing ITRF93 public dispatcher subcases",
            "mixed-origin arrays, observatory origins, and user-furnished "
            "SPICE body coverage beyond the SUN/EARTH origin-translation matrix",
        ),
        coverage_note=(
            "Random fuzz exercises a public quivr-object dispatcher subcase "
            "matrix spanning constant-frame inverse directions, non-Cartesian "
            "inputs, representative covariance-bearing paths, SUN/EARTH origin "
            "translations, and ITRF93 time-varying rotations. Remaining exclusions "
            "are explicit and should not be inferred as fuzz-covered."
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
        api_id="coordinates.residuals.Residuals.calculate",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.compute_residuals_chi2_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "End-to-end Residuals.calculate fused path (RM-WE2-002). Inputs are "
            "the OD-inner-loop shape: spherical 6-D coordinates with only "
            "lon/lat observed (rho/vrho/vlon/vlat NaN), SPD 2x2 astrometric "
            "covariance lifted into a 6x6 with NaN-padded inactive dims. Outputs "
            "are the four quivr columns (values, chi2, dof, probability) compared "
            "as ndarrays. Underlying chi2 numpy kernel parity remains gated by "
            "coordinates.residuals.calculate_chi2."
        ),
    ),
    ApiMigration(
        api_id="coordinates.residuals.calculate_chi2",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_chi2_numpy",
        parity_coverage="random-fuzz",
        covered_subcases=(
            "2-D symmetric-positive-definite astrometric covariance rows",
        ),
        coverage_note=(
            "Wave E2 OD-inner-loop kernel; RM-P1-013 documents the SPD "
            "covariance contract and targeted tests cover non-SPD diagnostics."
        ),
        latency_gate=True,
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
        parity_coverage="random-fuzz",
        coverage_note=(
            "Direct single-pair NumPy MOID boundary is covered by randomized "
            "baseline-main parity and shaped speed lanes; "
            "calculate_perturber_moids orchestration is tracked separately."
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
        parity_coverage="random-fuzz",
        coverage_note=(
            "Public quivr orchestration over the Rust MOID batch kernel is "
            "fuzzed end-to-end against the baseline-main oracle, including "
            "Orbits construction, SPICE perturber-state lookup, batched MOID "
            "dispatch, and PerturberMOIDs table assembly."
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
        parity_coverage="random-fuzz",
        coverage_note=(
            "Public quivr orchestration over Rust Lambert/grid kernels is "
            "fuzzed end-to-end against the baseline-main oracle, including "
            "Orbits construction, time-order filtering, Rust porkchop grid "
            "dispatch, and LambertSolutions table assembly."
        ),
    ),
    ApiMigration(
        api_id="photometry.calculate_phase_angle",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_phase_angle_numpy",
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="photometry.calculate_apparent_magnitude_v",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_apparent_magnitude_v_numpy",
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
        parity_coverage="random-fuzz",
        latency_gate=True,
    ),
    ApiMigration(
        api_id="photometry.predict_magnitudes",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.predict_magnitudes_bandpass_numpy",
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
        parity_coverage="random-fuzz",
        coverage_note=(
            "Random fuzz is constrained to well-conditioned low-e, "
            "main-belt-like, multi-day triplets where Rust Laguerre+deflation "
            "and legacy np.roots/LAPACK share a physical best root. A "
            "deterministic fixed fixture is retained as supplemental governance."
        ),
        covered_subcases=(
            "randomized low-e main-belt-like triplets with shared |r2|>=1.5 AU "
            "best roots",
            "eight deterministic well-conditioned fixed-fixture triplets",
        ),
        excluded_subcases=(
            "ill-conditioned or multi-root random triplets with root-subset "
            "disagreement between Laguerre+deflation and np.roots/LAPACK",
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="orbits.classify_orbits",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.classify_orbits_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Randomized parity and shaped speed lanes cover the NumPy rule "
            "core over (a, e, q, Q); public coordinate-table extraction remains "
            "covered by classification tests."
        ),
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
        if migration.parity_coverage == "fixed-fixture" and not (
            migration.covered_subcases and migration.excluded_subcases
        ):
            errors.append(
                f"{migration.api_id}: fixed-fixture requires covered_subcases "
                "and excluded_subcases"
            )
        if (
            migration.parity_coverage
            in {
                "fixed-fixture",
                "random-fuzz-excluded",
                "orchestration-implied",
                "not-covered",
            }
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
