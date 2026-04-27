"""Per-API parity tolerance table — single source of truth.

Every rust-default API in ``src/adam_core/_rust/status.py`` MUST have an
entry here. The values mirror the per-API tolerances established when
each port was shipped (see journal.md entries by date).

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

from dataclasses import dataclass, field
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
# Coverage: every rust-default API in src/adam_core/_rust/status.py PLUS
# the orchestration functions (calculate_perturber_moids, generate_porkchop_data,
# add_light_time) that compose them.
# ---------------------------------------------------------------------------

TOLERANCES: dict[str, ToleranceSpec] = {
    # ---- coordinates representation transforms (numpy boundary) ----
    "coordinates.cartesian_to_spherical": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-11)},
        rationale=(
            "Spherical output of cart→sph composed of sqrt + atan2 + asin."
        ),
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
        outputs={"out": OutputTol(atol=1e-10, rtol=1e-12)},
        rationale="Cart→cart frame rotation (ec↔eq).",
        dominant_column="x (AU)",
        physical_magnitude="1.7e-18 AU = sub-femtometer (essentially exact).",
        root_cause="6×6 constant rotation matrix multiply only — no transcendentals.",
        verdict="bit-parity.",
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
            "with worst-rel 5.2e-9 — ~6 sig figs accurate."
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
            "~1e47 magnitude. Verified rust matches finite-difference "
            "Jacobian where JAX diverges."
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
            "covariance worst-rel 6.7e-5 (driven by atan2 jacobian)."
        ),
        root_cause=(
            "Same Dual<6> AD pass as propagate_2body_with_covariance "
            "but composed through more transcendentals."
        ),
        verdict=(
            "more accurate on stiff inputs (same as propagate "
            "covariance) — rust Dual AD doesn't blow up where JAX "
            "jacfwd does."
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
        rationale=(
            "Stable 2·atan2(sqrt(1-cos), sqrt(1+cos)) form for cos→α."
        ),
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
        physical_magnitude="2.8e-12 mag ≈ 1 ulp at H-G mag scale.",
        root_cause="log10 + powf chain composes ~1-2 ulps.",
        verdict="bit-parity.",
    ),
    "photometry.calculate_apparent_magnitude_v_and_phase_angle": ToleranceSpec(
        outputs={
            "magnitude": OutputTol(atol=1e-12, rtol=1e-12),
            "phase_angle": OutputTol(atol=1e-10),
        },
        rationale=(
            "Fused mag+alpha — same kernels as standalone; same ulp ceiling."
        ),
        dominant_column="phase_angle (deg)",
        physical_magnitude=(
            "magnitude 2.1e-12 mag (bit-parity); phase_angle 7.9e-12 deg "
            "= 28 picoarcsec (same as standalone calculate_phase_angle)."
        ),
        root_cause=(
            "Same composed atan2 + small-angle ceiling as standalone "
            "phase_angle. 1 picoarcsec scale is far below any noise."
        ),
        verdict="equally accurate (see calculate_phase_angle entry).",
    ),
    "photometry.predict_magnitudes": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-12, rtol=1e-12)},
        rationale=(
            "H-G V-band magnitude + per-target-filter delta lookup — adds "
            "at most 1 fma over standalone V-band kernel."
        ),
        dominant_column="magnitude",
        physical_magnitude="2.8e-12 mag — same as standalone V-band.",
        root_cause="Same as calculate_apparent_magnitude_v + 1 fma for delta.",
        verdict="bit-parity.",
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
            "epoch": OutputTol(atol=1e-6),
            "orbit": OutputTol(atol=1e-5, rtol=1e-6),
        },
        rationale=(
            "Full Gauss IOD: equatorial→ecliptic rotation, 8th-order poly "
            "root finding (Laguerre+deflation), per-root orbit "
            "construction. INTENTIONALLY UNWIRED FROM RANDOMIZED FUZZ "
            "(see comment in _inputs.GENERATORS): rust Laguerre+deflation "
            "and legacy np.roots/LAPACK find different SUBSETS of the "
            "8th-order polynomial's roots on a non-trivial fraction of "
            "random triplets — intrinsic to the algorithms, not a kernel "
            "bug. Best-root parity holds at ~1e-10 AU when both sides "
            "find any |r2|≥1.5 AU root, but ~15% of random triplets show "
            "NaN-mismatch (one side rejects the root the other accepts). "
            "Tolerances retained for fixed-fixture / manual parity. "
            "Epoch atol=1e-6 day = 0.1 s; orbit atol=1e-5 AU = 1500 m."
        ),
        investigate=True,
        investigate_task="randomized-parity unwired by design",
    ),
    # ---- orchestration (compose multiple rust-default APIs) ----
    #
    # NOTE on coverage: these three orchestration APIs are NOT directly
    # wired in random-fuzz GENERATORS because their parity is structurally
    # implied by the underlying kernel parity entries. They consist of
    # meshgrid + filter + per-pair calls to kernels that ARE wired:
    #
    #   `calculate_perturber_moids` → loops `calculate_moid_batch` (in
    #     turn `calculate_moid` per pair, which iterates 2-body
    #     propagate + Brent root-finding — bounded by `propagate_2body`
    #     parity at atol=1e-11).
    #   `generate_porkchop_data` → meshgrid over dep×arr times + filter
    #     + per-pair `solve_lambert` — bounded by `solve_lambert` parity
    #     at atol=1e-13.
    #   `gaussIOD` → 8th-order poly root-find + per-root construction
    #     via `calcGibbs`/`calcGauss` — see its entry below for the
    #     intrinsic Laguerre-vs-LAPACK divergence.
    #
    # Independent random-fuzz parity of the orchestration would require
    # quivr-Orbits round-trip in the legacy subprocess (heavy adapter
    # work) and would only re-test the underlying kernels we already
    # gate. The tolerance specs are retained as a coverage manifest;
    # `state="orchestration-implied"` is reported in the parity table.
    "dynamics.calculate_perturber_moids": ToleranceSpec(
        outputs={
            "moid": OutputTol(atol=1e-10),
            "dt_at_min": OutputTol(atol=1e-6),
        },
        rationale=(
            "ORCHESTRATION (covered indirectly). MOID = min |r1(t) - "
            "r2(t+dt)| over analytical 2-body grid + Brent refinement. "
            "Bounded by `propagate_2body` kernel parity (atol=1e-11) "
            "plus optimizer xtol. Direct random-fuzz wiring would require "
            "SPK kernels in subprocess; deferred."
        ),
    ),
    "dynamics.generate_porkchop_data": ToleranceSpec(
        outputs={"out": OutputTol(atol=1e-13)},
        rationale=(
            "ORCHESTRATION (covered indirectly). Meshgrid + time-order "
            "filter + batched Lambert per (dep, arr) pair. Bounded by "
            "`solve_lambert` kernel parity (atol=1e-13). Direct wiring "
            "would require Orbits-quivr round-trip; deferred."
        ),
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
