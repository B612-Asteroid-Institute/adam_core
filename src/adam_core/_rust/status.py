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
        boundary="arrow",
        default="rust",
        rust_module="adam_core._rust_native.transform_coordinates_arrow",
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
            "ITRF93->equatorial time-varying rotations to SphericalCoordinates "
            "at vetted PCK epochs",
            "Public dispatcher covariance-bearing Cartesian/Keplerian inputs "
            "through constant/time-varying frames and origin translations",
            "Public dispatcher mixed-origin preservation/translation and "
            "observatory-origin translation through the shared SPICE backend",
        ),
        excluded_subcases=(
            "non-Cartesian input into a time-varying ITRF93 frame change",
            "geodetic input representation transforms",
        ),
        coverage_note=(
            "Random fuzz exercises a public quivr-object dispatcher subcase "
            "matrix spanning constant-frame inverse directions, non-Cartesian "
            "inputs, representative covariance-bearing paths, SUN/EARTH origin "
            "translations, mixed/observatory origins, covariance AD, and ITRF93 "
            "time-varying rotations at vetted PCK epochs. The canonical surface "
            "crosses as one pyarrow RecordBatch and Rust assembles the returned "
            "RecordBatch. Remaining exclusions are explicit and should not be "
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
        parity_coverage="random-fuzz",
        coverage_note=(
            "Raw forward-mode AD covariance-transform kernel is fuzzed directly "
            "against baseline-main public covariance transforms with diagnostic "
            "raw-kernel comparisons. Any NaN in an input covariance row short-"
            "circuits that entire output covariance row to NaN while state values "
            "still transform normally, matching legacy JAX/NumPy covariance "
            "propagation and differing intentionally from the rotation kernel's "
            "legacy mask policy. Public coordinate-dispatch coverage remains "
            "tracked separately."
        ),
        covered_subcases=(
            "Cartesian covariance transform to spherical with ecliptic→equatorial frame rotation",
            "Cartesian covariance transform to Keplerian with equatorial→ecliptic frame rotation",
            "Keplerian covariance transform to Cartesian with ecliptic→equatorial frame rotation",
            "Keplerian covariance transform to spherical with equatorial→ecliptic frame rotation",
            "all-NaN covariance row exercising any-NaN row short-circuit policy",
        ),
    ),
    ApiMigration(
        api_id="coordinates.rotate_cartesian_time_varying",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.rotate_cartesian_time_varying_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Raw time-varying rotation kernel is fuzzed directly against a "
            "NumPy oracle with diagnostic raw-kernel comparisons. It preserves "
            "CartesianCoordinates.rotate's legacy zero-fill-then-restore NaN "
            "covariance mask policy; partial-NaN rows are compatibility-covered "
            "but are non-physical except as whole inactive-dimension masks. "
            "Public coordinate-dispatch coverage remains tracked separately."
        ),
        covered_subcases=(
            "sxform-like 6x6 matrix tables with per-row matrix indices",
            "Cartesian state rotation",
            "covariance rotation with all-NaN and partial-NaN masks",
        ),
    ),
    ApiMigration(
        api_id="coordinates.residuals.apply_cosine_latitude_correction",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module=("adam_core._rust_native.apply_cosine_latitude_correction_numpy"),
        parity_coverage="random-fuzz",
        coverage_note=(
            "Classified numpy-flat (bead personal-cmy.36.9): pure residual-"
            "array helper with no table identity. "
            "Randomized parity covers cos(latitude) scaling for six-column "
            "spherical residual rows and covariance matrices, including NaN "
            "covariance-cell preservation. End-to-end Residuals.calculate remains "
            "separately governed."
        ),
        covered_subcases=(
            "longitude and longitudinal-velocity residual scaling",
            "D·Σ·Dᵀ covariance scaling with NaN preservation",
        ),
    ),
    ApiMigration(
        api_id="coordinates.residuals.bound_longitude_residuals",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.bound_longitude_residuals_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Randomized parity covers longitude residual no-wrap rows plus both "
            ">180° and <-180° wrap branches on each side of the 0°/360° sign "
            "convention. End-to-end Residuals.calculate remains separately governed. "
            "Classified numpy-flat (bead personal-cmy.36.9): pure residual-array "
            "helper with no table identity."
        ),
        covered_subcases=(
            "no-wrap longitude residual rows",
            ">180° wrap rows with and without 0°/360° sign flip",
            "<-180° wrap rows with and without 0°/360° sign flip",
        ),
    ),
    ApiMigration(
        api_id="coordinates.residuals.Residuals.calculate",
        status=PUBLIC_RUST_DEFAULT,
        boundary="arrow",
        default="rust",
        rust_module="adam_core._rust_native.residuals_calculate_arrow",
        parity_coverage="random-fuzz",
        coverage_note=(
            "End-to-end Residuals.calculate crosses once as observed/predicted "
            "coordinate RecordBatches; Rust owns decode, broadcasting, "
            "longitude wrap, cos(lat) correction, covariance sum, Cholesky "
            "chi2, chi-squared survival probability, and finished Residuals "
            "RecordBatch assembly. Inputs are the OD-inner-loop shape: "
            "spherical 6-D coordinates with only lon/lat observed and SPD 2x2 "
            "astrometric covariance lifted into a NaN-padded 6x6. Custom "
            "coordinate classes keep the numpy fallback. Underlying chi2 "
            "numpy kernel parity remains gated by "
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
            "covariance contract and targeted tests cover non-SPD diagnostics. "
            "Classified numpy-flat (bead personal-cmy.36.9): pure residual/"
            "covariance array kernel with no table identity."
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
        boundary="arrow",
        default="rust",
        rust_module="adam_core._rust_native.propagate_orbits_arrow",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Public Orbits facade: one Arrow crossing owns the orbit×epoch "
            "cross product, time rescaling, origin-mu lookup, covariance "
            "transport, physical-parameter repetition, and output assembly."
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="dynamics.propagate_2body_along_arc",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.propagate_2body_along_arc_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Warm-started arc helper behind propagation dispatch; randomized "
            "diagnostic raw-kernel comparisons check unsorted dt arrays against "
            "baseline-main vector propagation."
        ),
        covered_subcases=(
            "Single orbit propagated over many unsorted dt values, including dt=0",
            "Row-order restoration after internal dt sorting and warm-starting",
        ),
    ),
    ApiMigration(
        api_id="dynamics.propagate_2body_arc_batch",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.propagate_2body_arc_batch_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Batched warm-start propagation helper; randomized diagnostic "
            "raw-kernel comparisons flatten orbits × dt grids against "
            "baseline-main vector propagation."
        ),
        covered_subcases=(
            "Multiple orbit blocks with per-orbit unsorted dt rows",
            "Orbit-major flattened output ordering for the batched arc helper",
        ),
    ),
    ApiMigration(
        api_id="dynamics.propagate_2body_with_covariance",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.propagate_2body_with_covariance_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Baseline-main random fuzz is supplemented by a high-a, 2516-day "
            "finite-difference covariance fixture that checks the Rust "
            "Dual-number tangent path against the scalar state map under a "
            "tightened rtol=1e-8 witness."
        ),
        covered_subcases=(
            "Random covariance rows against baseline-main Jacobian covariance propagation",
            "High-a slow-moving 2516-day finite-difference covariance witness",
        ),
    ),
    ApiMigration(
        api_id="dynamics.generate_ephemeris_2body",
        status=PUBLIC_RUST_DEFAULT,
        boundary="arrow",
        default="rust",
        rust_module="adam_core._rust_native.generate_ephemeris_arrow",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Public Orbits+Observers to Ephemeris facade crosses once through "
            "nested Arrow RecordBatches; Rust owns normalization, light-time, "
            "photometry, diagnostics, and finished table assembly."
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="dynamics.generate_ephemeris_2body_with_covariance",
        status=PUBLIC_RUST_DEFAULT,
        boundary="arrow",
        default="rust",
        rust_module="adam_core._rust_native.generate_ephemeris_arrow",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Baseline-main random fuzz is supplemented by a distant-object "
            "finite-difference covariance fixture with stellar aberration enabled "
            "under a tightened rtol=1e-6 witness."
        ),
        covered_subcases=(
            "Random ephemeris covariance rows against baseline-main covariance propagation",
            "Distant slow-moving stellar-aberration finite-difference covariance witness",
        ),
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
            "baseline-main parity and shaped speed lanes; supplemental fixed "
            "fixtures cover the identical-circular flat-minimum case where the "
            "distance is unique but the argmin time is only an optimizer witness, "
            "plus a non-degenerate unique-minimum pair that pins dt_at_min to "
            "1e-6 day. calculate_perturber_moids orchestration is tracked separately."
        ),
    ),
    ApiMigration(
        api_id="dynamics.calculate_moid_batch",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_moid_batch_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Rayon batch kernel used by calculate_perturber_moids; randomized "
            "diagnostic raw-kernel comparisons check direct batched pairs "
            "against the baseline-main scalar MOID oracle."
        ),
        covered_subcases=(
            "Batched primary/secondary Cartesian orbit pairs",
            "Per-row MOID distance and optimizer dt_at_min outputs",
        ),
    ),
    ApiMigration(
        api_id="observers.Observers.from_codes",
        status=ORCHESTRATION_RUST_DEFAULT,
        boundary="python+quivr",
        default="rust",
        rust_module="adam_core._rust_native.AdamCoreSpiceBackend",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Public MPC observatory-code observer-state orchestration: the "
            "DE440 SPK lookups and the ITRF93->J2000 ground-offset rotation "
            "run in the Rust spicekit backend (pxform_batch/spkez), with "
            "Python owning obscodes lookup and table assembly. Randomized "
            "baseline-main parity fuzzes per-row codes (geocenter + ground "
            "stations, both hemispheres) across UTC epochs; measured "
            "agreement is f64 rounding-level (<=1.2e-15 AU)."
        ),
    ),
    ApiMigration(
        api_id="dynamics.calculate_perturber_moids",
        status=PUBLIC_RUST_DEFAULT,
        boundary="arrow",
        default="rust",
        rust_module="adam_core._rust_native.calculate_perturber_moids_arrow",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Public facade crosses once as an Orbits RecordBatch; Rust owns "
            "the perturber loop, SPICE perturber-state lookup, batched MOID "
            "dispatch, and finished PerturberMOIDs RecordBatch assembly. "
            "Randomized parity fuzzes the facade end-to-end against the "
            "baseline-main oracle."
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
        parity_coverage="random-fuzz",
        coverage_note=(
            "Randomized parity covers the public helper over asteroid/comet "
            "semi-major axes, eccentricity/inclination edge cases, and valid "
            "perturbing-body lookup names."
        ),
        covered_subcases=(
            "public calc_tisserand_parameter array API",
            "valid third_body lookup path across seeded fuzz runs",
            "pinned near-circular, high-eccentricity, prograde, polar, and retrograde rows",
        ),
    ),
    ApiMigration(
        api_id="missions.porkchop_grid",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.porkchop_grid_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Raw fused grid kernel behind generate_porkchop_data; randomized "
            "diagnostic raw-kernel comparisons check direct grid filtering and "
            "Lambert velocity outputs against baseline-main Lambert solves."
        ),
        covered_subcases=(
            "Departure/arrival grid filtering where arrival MJD exceeds departure MJD",
            "Raw Lambert departure/arrival velocity outputs for valid grid pairs",
        ),
    ),
    ApiMigration(
        api_id="dynamics.generate_porkchop_data",
        status=PUBLIC_RUST_DEFAULT,
        boundary="arrow",
        default="rust",
        rust_module="adam_core._rust_native.generate_porkchop_data_arrow",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Public facade crosses once as departure/arrival Orbits "
            "RecordBatches; Rust owns chronological sorting, frame/origin "
            "contract validation, meshgrid time filtering, rayon-batched "
            "Lambert, and finished LambertSolutions RecordBatch assembly. "
            "Randomized parity fuzzes the facade end-to-end against the "
            "baseline-main oracle."
        ),
    ),
    ApiMigration(
        api_id="photometry.calculate_phase_angle",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_phase_angle_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Classified numpy-flat (bead personal-cmy.36.8): coordinate tables "
            "act only as position carriers and the output is a plain float "
            "array with no table assembly; the Arrow-native single-crossing "
            "bar does not apply."
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="photometry.calculate_apparent_magnitude_v",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calculate_apparent_magnitude_v_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Classified numpy-flat (bead personal-cmy.36.8): coordinate tables "
            "act only as position carriers and the output is a plain float "
            "array with no table assembly; the Arrow-native single-crossing "
            "bar does not apply."
        ),
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
        coverage_note=(
            "Classified numpy-flat (bead personal-cmy.36.8): coordinate tables "
            "act only as position carriers and the output is a plain float "
            "array with no table assembly; the Arrow-native single-crossing "
            "bar does not apply."
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="photometry.predict_magnitudes",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.predict_magnitudes_bandpass_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Classified numpy-flat (bead personal-cmy.36.8): coordinate and "
            "exposure tables act only as position/filter carriers and the "
            "output is a plain float array with no table assembly; the "
            "Arrow-native single-crossing bar does not apply."
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="photometry.fit_absolute_magnitude_rows",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.fit_absolute_magnitude_rows_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Randomized parity covers single-group H-fit rows with finite-sigma "
            "and missing-sigma branches; grouped production path is tracked "
            "separately. Classified numpy-flat (bead personal-cmy.36.8): pure "
            "numeric fitting rows with no table identity."
        ),
    ),
    ApiMigration(
        api_id="photometry.fit_absolute_magnitude_grouped",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.fit_absolute_magnitude_grouped_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Randomized parity covers grouped H-fit offsets with mixed group sizes "
            "and finite-/missing-sigma row groups. Classified numpy-flat (bead "
            "personal-cmy.36.8): pure numeric fitting rows with no table identity."
        ),
    ),
    ApiMigration(
        api_id="orbit_determination.calcGibbs",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_gibbs_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Classified numpy-flat (bead personal-cmy.36.7): scalar (3,)-vector "
            "velocity helper with no table identity; the Arrow-native "
            "single-crossing bar does not apply."
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="orbit_determination.calcHerrickGibbs",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_herrick_gibbs_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Classified numpy-flat (bead personal-cmy.36.7): scalar (3,)-vector "
            "velocity helper with no table identity; the Arrow-native "
            "single-crossing bar does not apply."
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="orbit_determination.calcGauss",
        status=PUBLIC_RUST_DEFAULT,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.calc_gauss_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Classified numpy-flat (bead personal-cmy.36.7): scalar (3,)-vector "
            "velocity helper with no table identity; the Arrow-native "
            "single-crossing bar does not apply."
        ),
        latency_gate=True,
    ),
    ApiMigration(
        api_id="orbit_determination.gaussIOD",
        status=PUBLIC_RUST_DEFAULT,
        boundary="arrow",
        default="rust",
        rust_module="adam_core._rust_native.gauss_iod_orbits_arrow",
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
            "covered by classification tests. Classified numpy-flat (bead "
            "personal-cmy.36.9): element arrays in, label array out, no table "
            "assembly."
        ),
    ),
    ApiMigration(
        api_id="orbits.VariantOrbits.create",
        status=PUBLIC_RUST_DEFAULT,
        boundary="arrow",
        default="rust",
        rust_module="adam_core._rust_native.sample_orbit_variants_arrow",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Public covariance-variant sampler crosses once as an Orbits "
            "RecordBatch; Rust owns sigma-point/Monte-Carlo/auto sampling and "
            "finished VariantOrbits RecordBatch assembly. Sigma-point output "
            "is deterministic and gated elementwise against pinned legacy; "
            "Monte Carlo and the auto fallback use the Rust-native RNG "
            "(statistically equivalent, not bit-identical; decision "
            "2026-07-03) and are covered by statistical unit tests. Replaces "
            "the retired temporary variant-sampler candidate lane (bead "
            "personal-cmy.36.10); historical IDs remain documented only in "
            "migration/parity/README.md."
        ),
    ),
    ApiMigration(
        api_id="statistics.weighted_mean",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.weighted_mean_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Raw statistics helper: randomized fuzz compares the Rust kernel "
            "directly with the baseline-main NumPy/BLAS formula. Public "
            "coordinate covariance wrappers remain BLAS-backed for performance, "
            "so speed rows are diagnostic raw-kernel comparisons rather than "
            "public-dispatch promotion gates. Classified numpy-flat (bead "
            "personal-cmy.36.9): pure array statistics with no table identity."
        ),
        covered_subcases=("normalized finite weights over 6-D sample matrices",),
    ),
    ApiMigration(
        api_id="statistics.weighted_covariance",
        status=RAW_KERNEL_ONLY,
        boundary="numpy",
        default="rust",
        rust_module="adam_core._rust_native.weighted_covariance_numpy",
        parity_coverage="random-fuzz",
        coverage_note=(
            "Raw statistics helper: randomized fuzz compares the Rust kernel "
            "directly with the baseline-main NumPy/BLAS formula. Public "
            "coordinate covariance wrappers remain BLAS-backed for performance, "
            "so speed rows are diagnostic raw-kernel comparisons rather than "
            "public-dispatch promotion gates."
        ),
        covered_subcases=(
            "normalized finite weights over 6-D sample matrices and 6x6 "
            "covariance output",
        ),
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
