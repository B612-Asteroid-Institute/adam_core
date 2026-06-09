"""Per-API baseline-main randomized parity tolerance table.

Every registry row in ``src/adam_core/_rust/status.py`` with
``parity_coverage`` set to ``"random-fuzz"``, ``"fixed-fixture"``,
``"random-fuzz-excluded"``, or ``"orchestration-implied"`` must have an entry
here. Targeted-test-only rows are intentionally tracked in the registry instead
of this baseline-main random-fuzz manifest. Raw-kernel-only rows may appear here
when they have explicit fuzz/performance governance; their speed rows remain
diagnostic rather than public-promotion gates.

`atol`/`rtol` apply via ``np.testing.assert_allclose(rust, legacy, atol=atol, rtol=rtol)``.
For tuple outputs (e.g. ephemeris emits ``(spherical, lt, cart)``) we
key by output role: ``ToleranceSpec.outputs`` maps the output name to
its (atol, rtol) pair.

Any tolerance flagged ``investigate=True`` is treated by the gate as a
known-loose entry pending root-cause review (Task #137 et al). The gate
still enforces the spec but emits a banner so we don't lose track of
loose tolerances over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class OutputTol:
    """Tolerance pair for a single output array."""

    atol: float
    rtol: float = 0.0


@dataclass(frozen=True)
class ToleranceSpec:
    """Parity tolerance specification for one rust-default API.

    For single-output kernels, ``outputs`` has one entry under the key
    ``"out"``. For multi-output kernels (e.g. ephemeris returns three
    arrays), each output keys its own tolerance pair.

    ``rationale`` documents WHY the tolerance is set where it is — pulled
    from the journal entry for the corresponding port.

    ``investigate`` flags entries that are known-loose pending RCA. The
    gate runs them at this tolerance but reports them on every run.

    Structured RCA fields (optional, populated for loose-margin entries):
      - ``dominant_column``: which column / output dimension the worst
        rust-vs-legacy diff lives in (e.g. ``"tp (mjd)"``).
      - ``physical_magnitude``: the worst diff translated to physically
        meaningful units (e.g. ``"24 microseconds"``).
      - ``root_cause``: which composed transcendental / fp operation in
        the rust kernel introduces the divergence.
      - ``verdict``: rust's accuracy vs legacy on the standard candle
        (JPL Horizons fixture in `utils/helpers/data/`):
          * ``"bit-parity"``      — rust-vs-legacy ≤ 1-2 ulps; both
                                    pass the same external truth check.
          * ``"equally accurate"`` — rust and legacy each pass the
                                    Horizons candle test at the same
                                    tolerance; their last-ulp choices
                                    differ but neither is provably
                                    closer to the candle.
          * ``"more accurate"``    — rust passes the candle at tighter
                                    tolerance than legacy did.
          * ``"less accurate"``    — rust drifts further from candle
                                    than legacy. Action required.
          * ``""`` (empty)         — bit-parity case; verdict not
                                    applicable.
    """

    outputs: Mapping[str, OutputTol]
    rationale: str
    investigate: bool = False
    investigate_task: str = ""
    dominant_column: str = ""
    physical_magnitude: str = ""
    root_cause: str = ""
    verdict: str = ""

    def primary(self) -> OutputTol:
        if "out" in self.outputs:
            return self.outputs["out"]
        # First entry is the primary output for multi-output APIs
        return next(iter(self.outputs.values()))


# ---------------------------------------------------------------------------
# Per-API tolerance table.
#
# Coverage: randomized baseline-main parity entries are declared by
# `src/adam_core/_rust/status.py` via `parity_coverage`. This table stores the
# tolerance/RCA data for those entries.
# ---------------------------------------------------------------------------

TOLERANCES: dict[str, ToleranceSpec] = {
    # ---- coordinates representation transforms (numpy boundary) ----
    "coordinates.cartesian_to_spherical": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-11)},
        rationale=("Spherical output of cart→sph composed of sqrt + atan2 + asin."),
        dominant_column="lon (deg) at 0°/360° wraparound",
        physical_magnitude="1.4e-14 deg ≈ 50 femtoarcsec.",
        root_cause="atan2 + asin compose ~1-2 ulps near wraparound.",
        verdict="bit-parity.",
    ),
    "coordinates.cartesian_to_geodetic": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12)},
        rationale="Iterative geodetic latitude solve (Bowring).",
        dominant_column="latitude (deg)",
        physical_magnitude="1.4e-14 = 1 ulp at degree scale.",
        root_cause="Bowring iteration converges to same root within 1 ulp.",
        verdict="bit-parity.",
    ),
    "coordinates.cartesian_to_keplerian": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-9, rtol=1e-12)},
        rationale=(
            "13-column output (a, p, q, Q, e, i°, raan°, omega°, M°, "
            "nu°, n°/d, P d, tp mjd). Distance and e columns at "
            "≤1-3 ulps absolute; angular elements (i, raan, ap, M, nu) "
            "compose acos + atan2 with degree-scale outputs that "
            "accumulate ~1-100 ulps of degree per ulp of input."
        ),
        dominant_column="tp (mjd)",
        physical_magnitude=(
            "2.76e-10 days = 24 microseconds, on tp ~ MJD 60000-62000. "
            "All other columns ≤1-100 ulps absolute; all angular "
            "columns ≤ 0.05 mas (microarcsec)."
        ),
        root_cause=(
            "tp = t0 + dtp where dtp = -M_anom/n_mean. M_anom is "
            "computed via calc_mean_anomaly(nu, e), which composes "
            "atan2 + ln (for hyperbolic) / arcsin (for elliptic). "
            "Each transcendental contributes ~1 ulp at degree scale; "
            "division by small n_mean amplifies the ulp drift. "
            "Rust's f64 stdlib implementations of atan2/ln differ from "
            "JAX's XLA-compiled versions at the last 1-2 ulps."
        ),
        verdict=(
            "equally accurate. The Horizons standard candle (test_keplerian.py) "
            "holds rust to atol=1e-12 on q, Q, p, n, atol=1e-10 on P "
            "(period). Rust passes those with margin; the 24μs "
            "rust-vs-legacy tp drift is well below tp's intrinsic "
            "precision (n-body perturbations move tp by seconds/year). "
            "Neither rust nor legacy is provably closer to truth on tp — "
            "JPL Horizons publishes tp with ~1-second precision."
        ),
    ),
    "coordinates.keplerian.to_cartesian": ToleranceSpec(
        outputs={"out": OutputTol(atol=3e-12)},
        rationale=(
            "Closed-form keplerian→cartesian via Kepler equation Newton "
            "solve. 6-column output (x, y, z, vx, vy, vz)."
        ),
        dominant_column="x (AU)",
        physical_magnitude=(
            "Position columns ≤ 6.9e-14 AU ≈ 10 picometers. Velocity "
            "columns ≤ 1.2e-17 AU/d ≈ 0.2 mm/year — true bit-parity."
        ),
        root_cause=(
            "Newton iteration on Kepler's equation E - e·sin(E) = M. "
            "Rust's stdlib sin/cos differ from XLA at last ulp; "
            "iteration converges to same root within ~2 ulps."
        ),
        verdict="bit-parity.",
    ),
    "coordinates.cartesian_to_cometary": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-9, rtol=1e-12)},
        rationale=(
            "Same composition pattern as cart→keplerian — cometary "
            "(q, e, i°, raan°, omega°, Tp) differs only in periapsis "
            "vs anomaly representation. Same ulp accumulation."
        ),
        dominant_column="Tp (mjd)",
        physical_magnitude=(
            "1.89e-10 days ≈ 16 microseconds on Tp ≈ MJD 60000. "
            "omega column: 1.51e-11 deg ≈ 50 microarcsec."
        ),
        root_cause=(
            "Same as cart→keplerian: M_anom → Tp conversion accumulates "
            "atan2 + division by small n_mean. The omega column hits "
            "100 ulps of degree because raan+omega is a composed angle."
        ),
        verdict=(
            "equally accurate. JPL Horizons publishes Tp at ~1-second "
            "precision; rust-vs-legacy 16μs drift is below the candle's "
            "publishable resolution."
        ),
    ),
    "coordinates.cometary.to_cartesian": ToleranceSpec(
        outputs={"out": OutputTol(atol=3e-12)},
        rationale="Mirror of keplerian→cartesian — same kernel, same precision.",
        dominant_column="x (AU)",
        physical_magnitude=(
            "Position ≤ 4.3e-14 AU ≈ 6 picometers; velocity ≤ 3.2e-17 AU/d."
        ),
        root_cause=(
            "Same Newton iteration as keplerian→cartesian, after q→a "
            "conversion. Same ~2 ulp last-bit drift."
        ),
        verdict="bit-parity.",
    ),
    "coordinates.spherical.to_cartesian": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-11)},
        rationale="Inverse of cart→sph. 1-2 ulp angular composition ceiling.",
        dominant_column="x (AU)",
        physical_magnitude="3.5e-14 AU = 5 picometers.",
        root_cause="sin/cos chain composes ~2 ulps at AU scale.",
        verdict="bit-parity.",
    ),
    "coordinates.transform_coordinates": ToleranceSpec(
        outputs={
            "cart_ec_to_sph_eq": OutputTol(atol=1e-10, rtol=1e-12),
            "cart_eq_to_sph_ec": OutputTol(atol=1e-10, rtol=1e-12),
            "sph_ec_to_cart_eq": OutputTol(atol=1e-10, rtol=1e-12),
            "kep_ec_to_sph_eq": OutputTol(atol=1e-10, rtol=1e-12),
            "com_eq_to_kep_ec": OutputTol(atol=1e-9, rtol=1e-12),
            "cart_ec_sun_to_sph_ec_earth": OutputTol(atol=1e-11, rtol=1e-12),
            "cart_ec_earth_to_sph_ec_sun": OutputTol(atol=1e-11, rtol=1e-12),
            "cart_ec_earth_to_sph_itrf93": OutputTol(atol=3e-8, rtol=1e-12),
            "cart_itrf93_earth_to_sph_eq": OutputTol(atol=3e-8, rtol=1e-12),
            "cart_cov_ec_to_sph_eq": OutputTol(atol=1e-10, rtol=1e-12),
            "cart_cov_ec_to_sph_eq_covariance": OutputTol(atol=1e-21, rtol=1e-10),
            "cart_cov_ec_sun_to_sph_ec_earth": OutputTol(atol=1e-11, rtol=1e-12),
            "cart_cov_ec_sun_to_sph_ec_earth_covariance": OutputTol(
                atol=1e-21, rtol=1e-10
            ),
            "kep_cov_ec_to_sph_eq": OutputTol(atol=1e-10, rtol=1e-12),
            "kep_cov_ec_to_sph_eq_covariance": OutputTol(atol=1e-21, rtol=1e-10),
        },
        rationale=(
            "Public dispatcher matrix covering Cartesian constant-frame "
            "directions, Spherical/Keplerian/Cometary non-Cartesian inputs, "
            "representative covariance-bearing Cartesian/Keplerian dispatcher "
            "paths, SUN↔EARTH origin translations, and Earth-centered ITRF93 "
            "time-varying rotations. Exercises quivr object construction, "
            "public transform_coordinates dispatch, Rust fused dispatch where "
            "supported, and baseline-main public dispatch."
        ),
        dominant_column="spherical angle / Keplerian epoch-like output",
        physical_magnitude=(
            "1e-10 deg is 0.36 microarcsec for constant-frame angular columns; "
            "1e-9 days is 86 microseconds for the Cometary→Keplerian epoch-like "
            "column. Covariance rows use 1e-21 absolute for near-zero covariance "
            "cells plus 1e-10 relative for finite propagated covariance terms. "
            "SUN↔EARTH origin translations are held to 1e-11 in mixed "
            "spherical units, i.e. 1.5 m in range or 0.036 microarcsec in angular "
            "columns. The ITRF93 rows keep 3e-8 deg/day only for spherical "
            "velocity-angle columns: 0.108 mas/day, with deterministic PCK "
            "epochs chosen to keep accepted CSPICE-vs-spicekit Earth-rotation "
            "drift below the gate with >3x headroom."
        ),
        root_cause=(
            "Constant 6×6 frame rotations and representation conversions compose "
            "sqrt/atan2/asin/sin/cos last-ulp differences between Rust stdlib "
            "and baseline JAX/XLA; covariance subcases additionally compare "
            "Rust forward-mode AD Jacobian propagation against baseline-main's "
            "public covariance transform path. SUN↔EARTH origin translations use the same "
            "DE440 state semantics and show only final representation-conversion "
            "plus last-ulp SPK evaluation drift, not a broad SPICE slack. ITRF93 "
            "subcases additionally compare baseline CSPICE/spiceypy PCK evaluation "
            "against spicekit's pure-Rust PCK path. The underlying accepted "
            "rotation-matrix difference is at the last-ulp level, but applying "
            "the time-varying rotation differentiates through r×ω and the "
            "spherical velocity-angle Jacobian, amplifying that backend "
            "difference to ~1e-8 deg/day in vlon/vlat."
        ),
        verdict=(
            "public-dispatch parity across the covered subcase matrix; remaining "
            "exclusions are explicit in the migration registry."
        ),
    ),
    "coordinates.transform_coordinates_with_covariance": ToleranceSpec(
        outputs={
            "raw_cart_cov_ec_to_sph_eq": OutputTol(atol=1e-10, rtol=1e-12),
            "raw_cart_cov_ec_to_sph_eq_covariance": OutputTol(atol=1e-21, rtol=1e-10),
            "raw_cart_cov_eq_to_kep_ec": OutputTol(atol=1e-9, rtol=1e-12),
            "raw_cart_cov_eq_to_kep_ec_covariance": OutputTol(atol=1e-21, rtol=1e-10),
            "raw_kep_cov_ec_to_cart_eq": OutputTol(atol=3e-12, rtol=1e-12),
            "raw_kep_cov_ec_to_cart_eq_covariance": OutputTol(atol=1e-21, rtol=1e-10),
            "raw_kep_cov_eq_to_sph_ec": OutputTol(atol=1e-10, rtol=1e-12),
            "raw_kep_cov_eq_to_sph_ec_covariance": OutputTol(atol=1e-21, rtol=1e-10),
        },
        rationale=(
            "Raw forward-mode AD covariance transform kernel over representative "
            "constant-frame Cartesian and Keplerian representation chains, with "
            "one all-NaN covariance row per subcase to pin the all-or-nothing "
            "row-level NaN policy."
        ),
        dominant_column="Keplerian angular outputs and propagated 6×6 covariance cells",
        physical_magnitude=(
            "State outputs inherit the direct transform-coordinate scale: "
            "3e-12 AU for Keplerian→Cartesian, 1e-10 deg for spherical angles, "
            "and 1e-9 on Cartesian→Keplerian epoch/angle-like fields. "
            "Covariance rows use 1e-21 absolute for near-zero cells plus 1e-10 "
            "relative for finite Jacobian-propagated covariance terms."
        ),
        root_cause=(
            "Rust evaluates a Dual<6> transform chain and applies J·Σ·Jᵀ in "
            "row-major loops while the baseline-main oracle reaches the same "
            "public covariance transform through JAX jacfwd and NumPy matrix "
            "multiplication. Finite drift is last-ulp representation conversion "
            "and matrix-product ordering noise. Any NaN in an input covariance "
            "row intentionally produces an all-NaN output covariance row because "
            "that is the legacy JAX/NumPy matrix-product behavior."
        ),
        verdict=(
            "raw-kernel parity; performance rows are diagnostic, not public "
            "dispatcher promotion gates. The tightest finite-covariance budget "
            "observed in the 2026-05-14 targeted fuzz was "
            "raw_cart_cov_eq_to_kep_ec_covariance at ~10x headroom "
            "(max tolerance ratio ≈0.095); treat that row as a canary before "
            "any future covariance-rtol tightening rather than loosening "
            "preemptively."
        ),
    ),
    "coordinates.rotate_cartesian_time_varying": ToleranceSpec(
        outputs={
            "coords": OutputTol(atol=1e-12, rtol=1e-12),
            "covariances": OutputTol(atol=1e-12, rtol=1e-12),
        },
        rationale=(
            "Raw time-varying Cartesian rotation kernel over sxform-like 6×6 "
            "matrix tables, per-row matrix indices, Cartesian states, and "
            "covariance rows with all-NaN and legacy partial-NaN masks."
        ),
        dominant_column="rotated Cartesian state and 6×6 covariance cells",
        physical_magnitude=(
            "≤1e-12 absolute on AU/AU-day states and covariance cells; NaN "
            "covariance locations must match exactly. Partial-NaN rows preserve "
            "legacy compatibility, not a physically meaningful covariance model."
        ),
        root_cause=(
            "Rust evaluates explicit M·x and M·Σ·Mᵀ loops while the baseline "
            "oracle uses vectorized NumPy contractions; finite drift is last-ulp "
            "matrix multiplication order noise. The NaN mask behavior deliberately "
            "matches CartesianCoordinates.rotate's zero-fill-then-restore legacy "
            "policy."
        ),
        verdict=(
            "raw-kernel parity for the legacy rotation semantics; partial-NaN "
            "covariance rows are quarantined as compatibility behavior. "
            "Performance rows are diagnostic, not promotion gates."
        ),
    ),
    "coordinates.residuals.Residuals.calculate": ToleranceSpec(
        outputs={
            "values": OutputTol(atol=1e-12, rtol=1e-12),
            "chi2": OutputTol(atol=1e-12, rtol=1e-12),
            "dof": OutputTol(atol=0.0, rtol=0.0),
            "probability": OutputTol(atol=1e-12, rtol=1e-12),
        },
        rationale=(
            "End-to-end Residuals.calculate over OD-inner-loop spherical inputs "
            "(only lon/lat observed; SPD 2x2 astrometric covariance on the "
            "(lon, lat) sub-block; predicted is the full 6-D propagator output). "
            "`values` matches bit-for-bit modulo the same Cholesky/explicit-inverse "
            "drift as `coordinates.residuals.calculate_chi2`. `dof` is an integer "
            "count and must match exactly. `probability = 1 - scipy.stats.chi2.cdf` "
            "is identical between paths because the chi2 CDF call stays in Python."
        ),
        dominant_column="chi2 scalar",
        physical_magnitude="≤1e-13 absolute on unit-scale chi² values; dof is exact.",
        root_cause=(
            "Same as the underlying chi2 kernel: Cholesky triangular solve and "
            "explicit matrix inverse accumulate floating-point products in "
            "different orders."
        ),
        verdict="end-to-end parity for valid SPD covariance matrices.",
    ),
    "coordinates.residuals.calculate_chi2": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12, rtol=1e-12)},
        rationale=(
            "Mahalanobis χ² over representative 2-D astrometric residual rows. "
            "Rust uses per-row Cholesky solve while baseline-main forms an "
            "explicit inverse; both evaluate r·Σ⁻¹·rᵀ for SPD covariance input."
        ),
        dominant_column="chi2 scalar",
        physical_magnitude="≤1e-13 absolute on unit-scale χ² values.",
        root_cause=(
            "Cholesky triangular solve and explicit matrix inverse accumulate "
            "floating-point products in different orders; observed drift is at "
            "the last few ulps for 2×2 SPD matrices."
        ),
        verdict="equally accurate for valid SPD covariance matrices.",
    ),
    "coordinates.residuals.bound_longitude_residuals": ToleranceSpec(
        outputs={"out": OutputTol(atol=0.0, rtol=0.0)},
        rationale=(
            "Longitude residual wrapping over six-column spherical residual rows. "
            "Random fuzz exercises no-wrap rows plus both >180° and <-180° wrap "
            "branches with each side of the 0°/360° sign convention."
        ),
        dominant_column="longitude residual column",
        physical_magnitude="exact categorical branch parity in degrees.",
        root_cause=(
            "Pure comparisons, ±360° shifts, and sign flips. No transcendental "
            "or reduction-order drift is expected."
        ),
        verdict="bit-parity for every longitude-wrap branch.",
    ),
    "coordinates.residuals.apply_cosine_latitude_correction": ToleranceSpec(
        outputs={
            "residuals": OutputTol(atol=1e-12, rtol=1e-12),
            "covariances": OutputTol(atol=1e-12, rtol=1e-12),
        },
        rationale=(
            "Cos(latitude) correction for spherical residuals and covariance rows. "
            "Random fuzz scales residual columns 1 and 4 plus covariance rows/cols "
            "1 and 4 over representative latitudes, preserving NaN covariance cells."
        ),
        dominant_column="longitude/longitudinal-velocity residual and covariance cells",
        physical_magnitude=(
            "≤1e-12 absolute on degree-scale residuals and covariance cells; NaN "
            "covariance locations must match exactly."
        ),
        root_cause=(
            "Rust evaluates the reduced diagonal form D·Σ·Dᵀ directly while "
            "baseline-main constructs D matrices and performs batched matmul; "
            "any finite drift is last-ulp multiplication-order noise."
        ),
        verdict="algorithm-equivalent cos-lat residual/covariance parity.",
    ),
    # ---- raw statistics kernels ----
    "statistics.weighted_mean": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12, rtol=1e-12)},
        rationale=(
            "Raw Rust weighted-mean kernel compared directly against the "
            "baseline-main NumPy/BLAS formula np.dot(weights, samples). Public "
            "coordinate covariance wrappers intentionally stay BLAS-backed for "
            "performance; this entry governs raw-kernel correctness."
        ),
        dominant_column="weighted sample component",
        physical_magnitude="≤1e-13 absolute on O(1-10) sample means.",
        root_cause=(
            "Rust's serial accumulation and BLAS GEMV reduce terms in different "
            "orders, yielding only last-ulp reduction noise for normalized weights."
        ),
        verdict="raw-kernel parity; performance rows are diagnostic, not promotion gates.",
    ),
    "statistics.weighted_covariance": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12, rtol=1e-12)},
        rationale=(
            "Raw Rust weighted-covariance kernel compared directly against the "
            "baseline-main NumPy/BLAS formula (weights * residual.T) @ residual. "
            "Public coordinate covariance wrappers intentionally stay BLAS-backed "
            "for performance; this entry governs raw-kernel correctness."
        ),
        dominant_column="6x6 covariance cell",
        physical_magnitude="≤1e-12 absolute on O(100) covariance cells.",
        root_cause=(
            "Rust loops and BLAS GEMM use different accumulation trees; drift is "
            "bounded by last-ulp reduction-order noise for finite normalized weights."
        ),
        verdict="raw-kernel parity; performance rows are diagnostic, not promotion gates.",
    ),
    # ---- dynamics ----
    "dynamics.calc_mean_motion": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-13)},
        rationale="Scalar n = sqrt(mu / a^3).",
        dominant_column="n",
        physical_magnitude="3.5e-18 = exactly 1 ulp at AU³/d² scale.",
        root_cause="Single sqrt — atomic op.",
        verdict="bit-parity.",
    ),
    "dynamics.tisserand_parameter": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12, rtol=1e-13)},
        rationale=(
            "Public Tisserand helper over asteroid/comet orbital ranges and all "
            "supported perturbing-body semi-major axes."
        ),
        dominant_column="dimensionless Tisserand parameter",
        physical_magnitude=(
            "1e-12 is twelve orders below the unit-scale classifier thresholds "
            "near Tp≈2–3."
        ),
        root_cause="Equivalent cos/sqrt expression with vectorized transcendental libraries.",
        verdict="science-grade parity for a dimensionless classification helper.",
    ),
    "dynamics.calculate_moid": ToleranceSpec(
        outputs={
            "moid": OutputTol(atol=1e-9, rtol=1e-8),
            "dt_at_min": OutputTol(atol=1e-3, rtol=1e-8),
        },
        rationale=(
            "Single-pair MOID over two Cartesian orbit states. Both sides use "
            "the same nested bounded-minimization formulation: outer Brent "
            "search over primary-orbit dt and inner bounded search over the "
            "secondary ellipse anomaly. A supplemental fixed fixture covers "
            "the identical-circular flat-minimum case and compares only the "
            "unique distance; a second non-degenerate fixture pins dt_at_min at "
            "1e-6 day so optimizer time regressions cannot hide behind the "
            "randomized flat-minimum slack."
        ),
        dominant_column="dt_at_min from the outer bounded minimizer",
        physical_magnitude=(
            "MOID atol 1e-9 AU ≈ 150 m in random fuzz; the flat-minimum "
            "fixed fixtures tighten the distance to 1e-12 AU and pin a "
            "well-conditioned unique argmin to 1e-6 day ≈ 86 ms. dt_at_min "
            "atol 1e-3 day ≈ 86 s remains only as the randomized optimizer "
            "envelope; degenerate flat minima treat the returned time as a "
            "finite witness, not a unique science output."
        ),
        root_cause=(
            "Rust carries a scipy-style bounded minimizer but not scipy's exact "
            "floating-point branch history. The baseline oracle keeps upstream "
            "tolerances (inner xatol via tol=1e-12, outer tol=1e-14, "
            "propagation max_iter=1000/tol=1e-14). Very flat minima can shift "
            "the reported argmin substantially while preserving the distance; "
            "identical circular orbits make this non-uniqueness explicit."
        ),
        verdict="algorithm-equivalent optimizer parity for the direct NumPy boundary.",
    ),
    "dynamics.calculate_moid_batch": ToleranceSpec(
        outputs={
            "moid": OutputTol(atol=1e-9, rtol=1e-8),
            "dt_at_min": OutputTol(atol=1e-3, rtol=1e-8),
        },
        rationale=(
            "Raw Rayon batch MOID kernel over Cartesian primary/secondary orbit "
            "pairs. The legacy oracle loops over the same baseline-main scalar "
            "MOID formulation used by dynamics.calculate_moid."
        ),
        dominant_column="dt_at_min from the outer bounded minimizer",
        physical_magnitude=(
            "Same randomized optimizer envelope as dynamics.calculate_moid: "
            "MOID atol 1e-9 AU ≈ 150 m and dt_at_min atol 1e-3 day ≈ 86 s."
        ),
        root_cause=(
            "The Rust batch path only changes dispatch and Rayon scheduling over "
            "independent scalar MOID solves; per-row optimizer branch-history "
            "differences are the same as dynamics.calculate_moid."
        ),
        verdict="raw-kernel parity; performance rows are diagnostic, not promotion gates.",
    ),
    "dynamics.calculate_perturber_moids": ToleranceSpec(
        outputs={
            "orbit_index": OutputTol(atol=0.0, rtol=0.0),
            "perturber_code": OutputTol(atol=0.0, rtol=0.0),
            "moid": OutputTol(atol=1e-9, rtol=1e-8),
            "time_mjd": OutputTol(atol=1e-3, rtol=1e-8),
        },
        rationale=(
            "Public PerturberMOIDs orchestration over Orbits inputs. The gate "
            "compares stable numeric encodings of orbit_id/perturber ordering "
            "plus the science outputs from the same nested MOID optimizer used "
            "by dynamics.calculate_moid."
        ),
        dominant_column="time_mjd / dt_at_min from the outer bounded minimizer",
        physical_magnitude=(
            "MOID atol 1e-9 AU ≈ 150 m. time_mjd atol 1e-3 day ≈ 86 s; "
            "row identity columns are exact."
        ),
        root_cause=(
            "Same optimizer branch-history drift as the direct MOID boundary, "
            "plus last-ulp SPICE state differences between baseline-main and "
            "migration SPICE backends."
        ),
        verdict="public quivr orchestration parity for DE440 perturber rows.",
    ),
    "dynamics.propagate_2body": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-11)},
        rationale=(
            "Universal-Kepler 2-body, Newton chi iteration (reverted "
            "from Laguerre on 2026-04-25 — see journal #138)."
        ),
        dominant_column="position (x, y, z)",
        physical_magnitude="≤8.9e-14 AU = ~13 picometers.",
        root_cause=(
            "Newton iteration on universal-Kepler equation. Chi solver "
            "converges to same root within ~2 ulps of f64."
        ),
        verdict=(
            "bit-parity. The hyperbolic Oumuamua orbit roundtrip test "
            "(test_propagation.py) holds rust to atol=100m position / "
            "1mm/s velocity over 10000-day span — far tighter than "
            "the 13-pm rust-vs-legacy drift."
        ),
    ),
    "dynamics.propagate_2body_along_arc": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12)},
        rationale=(
            "Raw warm-started single-orbit arc helper compared against "
            "baseline-main _propagate_2body_vmap on the same orbit and "
            "random unsorted dt arrays."
        ),
        dominant_column="position (x, y, z)",
        physical_magnitude=(
            "Same units as propagate_2body; 1e-12 AU is 0.15 m and observed "
            "warm-start drift is expected at picometer/ulp scale."
        ),
        root_cause=(
            "Rust reuses orbit constants and warm-starts chi across sorted dt "
            "values, then restores input order. Baseline-main cold-starts each "
            "row through the vectorized propagation path; any drift should be "
            "limited to the same Newton/universal-Kepler last-bit choices as "
            "dynamics.propagate_2body."
        ),
        verdict="raw-kernel parity; performance rows are diagnostic, not promotion gates.",
    ),
    "dynamics.propagate_2body_arc_batch": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12)},
        rationale=(
            "Raw batched warm-started arc helper compared against baseline-main "
            "_propagate_2body_vmap after flattening an orbits × dt grid."
        ),
        dominant_column="position (x, y, z)",
        physical_magnitude=(
            "Same units as propagate_2body; 1e-12 AU is 0.15 m and covers only "
            "last-bit Newton/universal-Kepler drift."
        ),
        root_cause=(
            "Each orbit block warm-starts chi serially within its dt row while "
            "Rayon distributes blocks across orbits. Baseline-main cold-starts "
            "the flattened rows; differences should mirror the single-row "
            "propagation kernel and row-order restoration."
        ),
        verdict="raw-kernel parity; performance rows are diagnostic, not promotion gates.",
    ),
    "dynamics.propagate_2body_with_covariance": ToleranceSpec(
        outputs={
            "state": OutputTol(atol=1e-11),
            "covariance": OutputTol(atol=1e-14, rtol=1e-6),
        },
        rationale=(
            "State mirrors propagate_2body. Covariance path uses Rust "
            "Dual<6> autodiff to compute Σ_out = J Σ_in J^T per row."
        ),
        dominant_column="covariance off-diagonal",
        physical_magnitude=(
            "State worst 8.9e-14 AU; covariance worst 1.7e-14 (AU,AU/d)² "
            "with worst-rel 5.2e-9 in random fuzz. The supplemental finite-"
            "difference fixture is held to rtol=1e-8."
        ),
        root_cause=(
            "Σ_out = J Σ_in J^T = 6×6×6 matmul. Each multiplication "
            "accumulates ~6 sig figs of fp drift relative to JAX's "
            "jacfwd path; rtol=1e-6 captures this without rejecting "
            "bit-clean cells."
        ),
        verdict=(
            "more accurate than legacy on stiff inputs. Per journal "
            "2026-04-22: rust Dual<6> AD produces non-NaN tangents on "
            "2/2000 chaotic inputs where JAX's jacfwd overflows to "
            "~1e47 magnitude. Supplemental fixed-fixture governance now "
            "codifies a high-a finite-difference covariance witness."
        ),
    ),
    "dynamics.generate_ephemeris_2body": ToleranceSpec(
        outputs={
            "spherical": OutputTol(atol=1e-10, rtol=1e-12),
            "light_time": OutputTol(atol=1e-14),
            "aberrated_state": OutputTol(atol=1e-11),
        },
        rationale=(
            "Composed: LT Newton + universal-Kepler back-prop + "
            "stellar aberration + ec→eq rotation + cart→sph."
        ),
        dominant_column="spherical lon/lat near 0°/360° wraparound",
        physical_magnitude=(
            "spherical worst 8.5e-14 deg = 0.3 picoarcsec; "
            "light_time 1.1e-16 day = 9.6 fs; "
            "aberrated_state 1.4e-14 AU = 2 pm."
        ),
        root_cause=(
            "Composed ~5 transcendentals (sqrt, atan2, sin, cos, "
            "asin). 1-2 ulp per op accumulates to ~1e-13 abs."
        ),
        verdict=(
            "bit-parity. Horizons-fixture ephemeris test "
            "(test_ephemeris.py) holds rust to atol=1e-10 deg "
            "(0.36 mas) — well below DE440's 10 mas noise."
        ),
    ),
    "dynamics.generate_ephemeris_2body_with_covariance": ToleranceSpec(
        outputs={
            "spherical": OutputTol(atol=1e-10, rtol=1e-12),
            "light_time": OutputTol(atol=1e-14),
            "aberrated_state": OutputTol(atol=1e-11),
            "covariance": OutputTol(atol=1e-14, rtol=1e-6),
        },
        rationale="Mirrors generate_ephemeris_2body + covariance.",
        dominant_column="covariance",
        physical_magnitude=(
            "Same as generate_ephemeris_2body for state outputs; "
            "covariance worst-rel 6.7e-5 in random fuzz, driven by near-zero "
            "covariance cells in the atan2 Jacobian. The supplemental finite-"
            "difference fixture is held to rtol=1e-6."
        ),
        root_cause=(
            "Same Dual<6> AD pass as propagate_2body_with_covariance "
            "but composed through more transcendentals."
        ),
        verdict=(
            "more accurate on stiff inputs (same as propagate "
            "covariance) — rust Dual AD doesn't blow up where JAX "
            "jacfwd does. Supplemental fixed-fixture governance now "
            "codifies a distant-object finite-difference covariance witness "
            "with stellar aberration enabled."
        ),
    ),
    "dynamics.solve_lambert": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-13)},
        rationale="Izzo Lambert solver, Householder root-finding.",
        dominant_column="velocity vector (v1, v2)",
        physical_magnitude="3.8e-15 AU/d ≈ 0.0001 mm/s.",
        root_cause=(
            "Householder iteration converges to atol/rtol=1e-10 "
            "internally; output is at fp precision."
        ),
        verdict="bit-parity.",
    ),
    "dynamics.add_light_time": ToleranceSpec(
        outputs={
            "aberrated_orbit": OutputTol(atol=1e-13),
            "light_time": OutputTol(atol=1e-15),
        },
        rationale=(
            "Newton fixed-point LT iteration + universal-Kepler "
            "back-prop. Convergence threshold lt_tol=1e-10 day; the "
            "actual residual is at fp precision."
        ),
        dominant_column="aberrated_orbit position (x, y, z)",
        physical_magnitude=(
            "7.1e-15 AU = 1 picometer. light_time worst 1.1e-16 day = "
            "9.6 femtoseconds."
        ),
        root_cause=(
            "Each Newton LT iteration solves chi via universal-Kepler "
            "(per row), then back-propagates. ~3-4 iterations × 1 ulp "
            "drift = ~4 ulps total. Rust f64 sqrt/log differs from "
            "XLA's at last ulp."
        ),
        verdict=(
            "bit-parity. Both rust and legacy converge to the same "
            "physical root within ~1 ulp; differ only in last-ulp "
            "rounding. Picometer-scale aberration drift is 14+ "
            "orders below astrometric noise."
        ),
    ),
    # ---- photometry ----
    "photometry.calculate_phase_angle": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-10)},
        rationale=("Stable 2·atan2(sqrt(1-cos), sqrt(1+cos)) form for cos→α."),
        dominant_column="α (deg)",
        physical_magnitude="7.87e-12 deg ≈ 28 picoarcsec.",
        root_cause=(
            "Composed sqrt(1−cos) and sqrt(1+cos) lose precision at "
            "small phase angles (near 0°), where 1−cos≈α²/2 has "
            "relative error 2 ulp. The half-angle atan2 then doubles "
            "this. JAX uses XLA's vectorized `atan2`, rust uses f64 "
            "stdlib — they agree to ~1-2 ulps on the small-angle limb."
        ),
        verdict=(
            "equally accurate. LSST single-observation astrometric "
            "uncertainty is ~1 mas = 10⁹ picoarcsec; DE440 ephemeris "
            "noise is ~10 mas. 28 picoarcsec rust-vs-legacy drift is "
            "11 orders of magnitude below either."
        ),
    ),
    "photometry.calculate_apparent_magnitude_v": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12, rtol=1e-12)},
        rationale="H-G phase function in V band. Pure element-wise.",
        dominant_column="magnitude",
        physical_magnitude=(
            "3.6e-12 mag ≈ ~500 ulps at magnitude ~30, but >1e8× below "
            "0.01 mag survey photometric noise."
        ),
        root_cause="log10 + powf chain composes hundreds of output ulps at mag scale.",
        verdict=(
            "science-grade parity: log10/powf composition differs by ~500 ulps "
            "but remains many orders below photometric noise."
        ),
    ),
    "photometry.calculate_apparent_magnitude_v_and_phase_angle": ToleranceSpec(
        outputs={
            "magnitude": OutputTol(atol=1e-12, rtol=1e-12),
            "phase_angle": OutputTol(atol=1e-10),
        },
        rationale=(
            "Fused mag+alpha — same kernels as standalone; same physical "
            "tolerance envelope."
        ),
        dominant_column="phase_angle (deg)",
        physical_magnitude=(
            "magnitude drift is a few 1e-12 mag (~hundreds of ulps at mag scale); "
            "phase_angle 7.9e-12 deg = 28 picoarcsec (same as standalone "
            "calculate_phase_angle)."
        ),
        root_cause=(
            "Same composed atan2 + small-angle ceiling as standalone "
            "phase_angle. 1 picoarcsec scale is far below any noise."
        ),
        verdict=(
            "science-grade parity for magnitude plus equally accurate phase-angle "
            "parity (see calculate_phase_angle entry)."
        ),
    ),
    "photometry.predict_magnitudes": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12, rtol=1e-12)},
        rationale=(
            "H-G V-band magnitude + per-target-filter delta lookup — adds "
            "at most 1 fma over standalone V-band kernel."
        ),
        dominant_column="magnitude",
        physical_magnitude=(
            "3.6e-12 mag — same order as standalone V-band and >1e8× below "
            "0.01 mag survey photometric noise."
        ),
        root_cause=(
            "Same log10/powf composition as calculate_apparent_magnitude_v + "
            "1 fma for delta."
        ),
        verdict=(
            "science-grade parity: magnitude drift is hundreds of ulps at mag "
            "scale but many orders below photometric noise."
        ),
    ),
    "photometry.fit_absolute_magnitude_rows": ToleranceSpec(
        outputs={
            "h_hat": OutputTol(atol=1e-12, rtol=1e-12),
            "h_sigma": OutputTol(atol=1e-12, rtol=1e-12),
            "sigma_eff": OutputTol(atol=1e-12, rtol=1e-12),
            "chi2_red": OutputTol(atol=1e-12, rtol=1e-12),
            "n_used": OutputTol(atol=0.0),
        },
        rationale=(
            "Single-group H-fit statistics over finite H rows with either "
            "all finite magnitude sigmas (inverse-variance weighted mean, "
            "H_sigma, chi2_red) or missing sigmas (arithmetic mean plus MAD "
            "scatter estimate)."
        ),
        dominant_column="H_hat / H_sigma statistics",
        physical_magnitude="1e-12 mag-level agreement on H statistics; n_used exact.",
        root_cause=(
            "Rust and legacy evaluate the same weighted sums and MAD median. "
            "Remaining differences are last-ulp reduction-order noise."
        ),
        verdict="bit-parity for finite, non-degenerate H-fit rows.",
    ),
    "photometry.fit_absolute_magnitude_grouped": ToleranceSpec(
        outputs={
            "h_hat": OutputTol(atol=1e-12, rtol=1e-12),
            "h_sigma": OutputTol(atol=1e-12, rtol=1e-12),
            "sigma_eff": OutputTol(atol=1e-12, rtol=1e-12),
            "chi2_red": OutputTol(atol=1e-12, rtol=1e-12),
            "n_used": OutputTol(atol=0.0),
        },
        rationale=(
            "Grouped H-fit wrapper over contiguous per-object row groups. "
            "Random fuzz covers mixed group sizes plus finite-sigma and "
            "missing-sigma groups."
        ),
        dominant_column="per-group H statistics",
        physical_magnitude="1e-12 mag-level agreement per group; n_used exact.",
        root_cause=(
            "Same single-group formula as fit_absolute_magnitude_rows; grouped "
            "Rust only changes orchestration/parallelization over offsets."
        ),
        verdict="bit-parity for finite, non-degenerate grouped H-fit rows.",
    ),
    # ---- orbits ----
    "orbits.classify_orbits": ToleranceSpec(
        outputs={"out": OutputTol(atol=0.0)},
        rationale=(
            "PDS Small Bodies Node rule table over (a, e, q, Q). The Rust "
            "kernel returns integer class codes and the Python wrapper maps "
            "them to class strings. Random fuzz covers the registered NumPy "
            "rule core; public coordinate-table extraction stays covered by "
            "classification tests."
        ),
        dominant_column="integer class code",
        physical_magnitude="exact categorical equality required.",
        root_cause="Pure ordered boolean rules; no floating-point tolerance is needed.",
        verdict="exact rule parity.",
    ),
    # ---- orbit determination ----
    "orbit_determination.calcGibbs": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12)},
        rationale=(
            "Gibbs three-vector velocity reconstruction. Note: Gibbs "
            "amplifies input ULPs by ~1e8× on close-spaced triplets, "
            "so the fuzzer enforces ≥1-day spacing between observations."
        ),
        dominant_column="velocity (vx, vy, vz)",
        physical_magnitude="0.0e+00 — bit-identical (FP-deterministic chain).",
        root_cause="Pure cross-products + scalar ops, no transcendentals.",
        verdict="bit-parity (exact: 0e+00).",
    ),
    "orbit_determination.calcHerrickGibbs": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12)},
        rationale="Herrick-Gibbs short-arc velocity.",
        dominant_column="velocity (vx, vy, vz)",
        physical_magnitude="0.0e+00 — bit-identical.",
        root_cause="Pure scalar arithmetic, no transcendentals.",
        verdict="bit-parity (exact: 0e+00).",
    ),
    "orbit_determination.calcGauss": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12)},
        rationale="Gauss two-vector velocity.",
        dominant_column="velocity (vx, vy, vz)",
        physical_magnitude="0.0e+00 — bit-identical.",
        root_cause="Closed-form Lagrange coefficients only.",
        verdict="bit-parity (exact: 0e+00).",
    ),
    "orbit_determination.gaussIOD": ToleranceSpec(
        outputs={
            "epoch": OutputTol(atol=1e-10),
            "orbit": OutputTol(atol=1e-11, rtol=1e-9),
        },
        rationale=(
            "Full Gauss IOD: equatorial→ecliptic rotation, 8th-order polynomial "
            "root finding, and per-root orbit construction. Random fuzz is "
            "constrained to well-conditioned low-e, main-belt-like, multi-day "
            "triplets where Rust Laguerre+deflation and legacy np.roots/LAPACK "
            "share a physical best root; deterministic fixed fixtures retain "
            "coverage for the same shared-root regime. Unconstrained random "
            "triplets remain excluded because the two root solvers can accept "
            "different physical-root subsets."
        ),
        dominant_column="epoch and 6-D Cartesian orbit",
        physical_magnitude=(
            "Random shared-root fuzz holds epoch to 1e-10 day = 8.6 microseconds "
            "and orbit to 1e-11 AU ≈ 1.5 m with 1 ppb relative tolerance. The "
            "supplemental fixed fixture keeps its own looser eight-triplet branch-"
            "history tolerance while remaining below kilometre-scale orbit drift."
        ),
        root_cause=(
            "Both sides solve the same 8th-order Gauss-IOD polynomial, but "
            "Laguerre+deflation and LAPACK root finding have different last-bit "
            "branch histories and can disagree on marginal root acceptance. The "
            "randomized gate and fixed fixture therefore cover the shared best-root "
            "regime used downstream after near-observer roots are filtered."
        ),
        verdict=(
            "randomized shared-root parity plus supplemental fixed-fixture parity; "
            "unconstrained multi-root subset parity remains excluded by design."
        ),
    ),
    # ---- orchestration (compose multiple rust-default APIs) ----
    #
    # `gaussIOD` has a dedicated entry above documenting its intrinsic
    # Laguerre-vs-LAPACK randomized-root divergence.
    "missions.porkchop_grid": ToleranceSpec(
        outputs={
            "departure_index": OutputTol(atol=0.0, rtol=0.0),
            "arrival_index": OutputTol(atol=0.0, rtol=0.0),
            "solution_departure_velocity": OutputTol(atol=1e-13, rtol=1e-12),
            "solution_arrival_velocity": OutputTol(atol=1e-13, rtol=1e-12),
        },
        rationale=(
            "Raw fused porkchop grid kernel: enumerate arrival-after-departure "
            "grid pairs and solve Lambert for each valid pair. The legacy oracle "
            "uses baseline-main _izzo_lambert_vmap over the same filtered grid."
        ),
        dominant_column="solution_departure_velocity / solution_arrival_velocity",
        physical_magnitude="1e-13 AU/day ≈ 0.00017 mm/s; grid indices exact.",
        root_cause=(
            "Same Householder iteration floating-point branch history as "
            "dynamics.solve_lambert; the raw kernel only fuses time-order "
            "filtering and batched Lambert dispatch."
        ),
        verdict="raw-kernel parity; performance rows are diagnostic, not promotion gates.",
    ),
    "dynamics.generate_porkchop_data": ToleranceSpec(
        outputs={
            "departure_index": OutputTol(atol=0.0, rtol=0.0),
            "arrival_index": OutputTol(atol=0.0, rtol=0.0),
            "departure_time_mjd": OutputTol(atol=0.0, rtol=0.0),
            "arrival_time_mjd": OutputTol(atol=0.0, rtol=0.0),
            "solution_departure_velocity": OutputTol(atol=1e-13, rtol=1e-12),
            "solution_arrival_velocity": OutputTol(atol=1e-13, rtol=1e-12),
            "c3_departure": OutputTol(atol=1e-12, rtol=1e-10),
            "vinf_arrival": OutputTol(atol=1e-12, rtol=1e-10),
            "time_of_flight": OutputTol(atol=0.0, rtol=0.0),
        },
        rationale=(
            "Public LambertSolutions orchestration over Orbits inputs. The gate "
            "compares stable numeric encodings of departure/arrival row identity, "
            "time-order filtering, solution velocities, and derived C3/v∞ values "
            "from the same Izzo Lambert solver covered by dynamics.solve_lambert."
        ),
        dominant_column="solution_departure_velocity / solution_arrival_velocity",
        physical_magnitude="1e-13 AU/day ≈ 0.00017 mm/s; identity/time columns exact.",
        root_cause=(
            "Same Householder iteration floating-point branch history as the "
            "direct solve_lambert boundary plus quivr table round-trip."
        ),
        verdict="public quivr orchestration parity for Lambert porkchop grids.",
    ),
}


def all_api_ids() -> tuple[str, ...]:
    return tuple(TOLERANCES.keys())


def get(api_id: str) -> ToleranceSpec:
    if api_id not in TOLERANCES:
        raise KeyError(
            f"No tolerance spec for {api_id!r}. Add an entry to "
            f"migration/parity/tolerances.py before using this API in the gate."
        )
    return TOLERANCES[api_id]
