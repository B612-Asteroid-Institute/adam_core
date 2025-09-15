"""
SBDB-based end-to-end recovery test for anomaly-gated filtering pipeline.

This test verifies that the complete pipeline (BVH indexing, ray construction,
geometric overlap) can successfully recover synthetic observations generated
from real orbital data with covariance sampling.
"""

import numpy as np
import pytest
from adam_assist import ASSISTPropagator

from adam_core.geometry import (
    build_bvh_index_from_segments,
    ephemeris_to_rays,
    query_bvh,
)
from adam_core.observers.observers import Observers
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.orbits.query.sbdb import query_sbdb
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp


class TestSBDBEndToEnd:
    """SBDB-based end-to-end recovery tests."""

    def test_sbdb_e2e_recovery_ci(self):
        """
        CI-sized SBDB end-to-end recovery test.

        Uses 10 variants × 21 epochs × 3 stations = 630 observations.
        Verifies 100% recovery with 1 arcmin guard radius.
        """
        self._run_sbdb_e2e_test(
            sbdb_id="1998 SG172",
            n_variants=10,
            n_epochs=21,
            guard_arcmin=1.0,
            max_chord_arcmin=2.0,  # Coarser to avoid segment limit
            expected_recovery=1.0,
        )

    @pytest.mark.sbdb_full
    def test_sbdb_e2e_recovery_full(self):
        """
        Full-scale SBDB end-to-end recovery test.

        Uses 100 variants × 21 epochs × 3 stations = 6300 observations.
        Verifies 100% recovery with 1 arcmin guard radius.
        """
        self._run_sbdb_e2e_test(
            sbdb_id="1998 SG172",
            n_variants=100,
            n_epochs=21,
            guard_arcmin=1.0,
            max_chord_arcmin=2.0,  # Coarser to avoid segment limit
            expected_recovery=1.0,
        )

    @pytest.mark.parametrize("guard_arcmin", [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01])
    def test_sbdb_guard_sweep_ci(self, guard_arcmin):
        """
        CI-sized guard tolerance sweep for 1998 SG172.

        Tests recovery at different guard radii to find failure threshold.
        Uses dynamic polyline sampling that gets tighter for smaller guards.
        """
        # Dynamic polyline sampling: much tighter for very small guards
        if guard_arcmin >= 0.5:
            max_chord_arcmin = 2.0  # Coarse for large guards
        elif guard_arcmin >= 0.1:
            max_chord_arcmin = guard_arcmin * 2  # 2x guard size
        elif guard_arcmin >= 0.02:
            max_chord_arcmin = guard_arcmin * 1.5  # 1.5x guard size for tiny guards
        else:
            max_chord_arcmin = guard_arcmin * 1.2  # Very tight for sub-arcsec guards

        # Expect 100% recovery at all guard sizes; fail fast if not met
        expected_recovery = 1.0

        self._run_sbdb_e2e_test(
            sbdb_id="1998 SG172",
            n_variants=10,
            n_epochs=21,
            guard_arcmin=guard_arcmin,
            max_chord_arcmin=max_chord_arcmin,
            expected_recovery=expected_recovery,
        )

    @pytest.mark.sbdb_full
    @pytest.mark.parametrize(
        "sbdb_id,guard_arcmin",
        [
            # Inner main-belt, low-e, low-i (~2.2 AU)
            ("8 Flora", 1.0),
            ("8 Flora", 0.5),
            ("8 Flora", 0.25),
            ("8 Flora", 0.1),
            # High-e NEO (perihelion ~0.6 AU, aphelion ~1.5 AU)
            ("1862 Apollo", 1.0),
            ("1862 Apollo", 0.5),
            ("1862 Apollo", 0.25),
            ("1862 Apollo", 0.1),
            # High-i main-belt (~2.8 AU, i~34°)
            ("20 Massalia", 1.0),
            ("20 Massalia", 0.5),
            ("20 Massalia", 0.25),
            ("20 Massalia", 0.1),
            # High-e Amor NEO (perihelion ~1.1 AU, aphelion ~2.8 AU)
            ("1221 Amor", 1.0),
            ("1221 Amor", 0.5),
            ("1221 Amor", 0.25),
            ("1221 Amor", 0.1),
            # Outer main-belt, moderate-e (~3.1 AU)
            ("324 Bamberga", 1.0),
            ("324 Bamberga", 0.5),
            ("324 Bamberga", 0.25),
            ("324 Bamberga", 0.1),
            # Very high-i (~63°), moderate distance (~2.4 AU)
            ("944 Hidalgo", 1.0),
            ("944 Hidalgo", 0.5),
            ("944 Hidalgo", 0.25),
            ("944 Hidalgo", 0.1),
            # Jupiter Trojan (~5.2 AU, L4 point)
            ("588 Achilles", 1.0),
            ("588 Achilles", 0.5),
            ("588 Achilles", 0.25),
            ("588 Achilles", 0.1),
            # Baseline (Izevic) - moderate distance, moderate-e
            ("1998 SG172", 1.0),
            ("1998 SG172", 0.5),
            ("1998 SG172", 0.25),
            ("1998 SG172", 0.1),
        ],
    )
    def test_sbdb_guard_sweep_full(self, sbdb_id, guard_arcmin):
        """
        Full-scale guard tolerance sweep across diverse orbit types.

        Tests recovery across different distances, eccentricities, and inclinations
        to find failure thresholds and understand geometric sensitivity.
        """
        # Dynamic polyline sampling: much tighter for very small guards
        if guard_arcmin >= 0.5:
            max_chord_arcmin = 2.0  # Coarse for large guards
        elif guard_arcmin >= 0.1:
            max_chord_arcmin = guard_arcmin * 2  # 2x guard size
        else:
            max_chord_arcmin = guard_arcmin * 1.5  # 1.5x guard size for tiny guards

        # Expect 100% recovery at all guard sizes; fail fast if not met
        expected_recovery = 1.0

        self._run_sbdb_e2e_test(
            sbdb_id=sbdb_id,
            n_variants=100,
            n_epochs=21,
            guard_arcmin=guard_arcmin,
            max_chord_arcmin=max_chord_arcmin,
            expected_recovery=expected_recovery,
        )

    def _run_sbdb_e2e_test(
        self,
        sbdb_id: str,
        n_variants: int,
        n_epochs: int,
        guard_arcmin: float,
        max_chord_arcmin: float,
        expected_recovery: float,
    ):
        """
        Run the SBDB end-to-end recovery test with specified parameters.

        Parameters
        ----------
        sbdb_id : str
            SBDB object identifier (e.g., "1998 SG172", "4 Vesta")
        n_variants : int
            Number of orbit variants to generate from covariance
        n_epochs : int
            Number of observation epochs (must be multiple of 3 for 3 stations)
        guard_arcmin : float
            Guard radius in arcminutes for geometric overlap
        max_chord_arcmin : float
            Maximum chord length in arcminutes for polyline sampling
        expected_recovery : float
            Expected recovery fraction (0.0 = log only, >0.0 = assert threshold)
        """
        # Step 1: Fetch nominal orbit from SBDB
        print(f"Fetching orbit for {sbdb_id} from SBDB...")
        nominal_orbits = query_sbdb([sbdb_id])
        assert len(nominal_orbits) == 1
        nominal_orbit = nominal_orbits[0:1]  # Keep as table with 1 row

        print(
            f"Nominal orbit epoch: {nominal_orbit.coordinates.time.mjd().to_numpy()[0]:.1f} MJD"
        )
        print(
            f"Nominal orbit has covariance: {nominal_orbit.coordinates.covariance is not None}"
        )

        # Step 2: Generate orbit variants from covariance
        print(f"Generating {n_variants} orbit variants from covariance...")
        variant_orbits = VariantOrbits.create(
            nominal_orbit,
            method="monte-carlo",
            num_samples=n_variants,
            seed=42,
        )

        print(f"Generated {len(variant_orbits)} variants")

        # Step 3: Create observation epochs (spanning ~3 months)
        epoch_start_mjd = nominal_orbit.coordinates.time.mjd().to_numpy()[0]
        epoch_span_days = 90  # 3 months

        if n_epochs % 3 != 0:
            raise ValueError(
                f"n_epochs ({n_epochs}) must be multiple of 3 for 3 stations"
            )

        epochs_per_station = n_epochs // 3
        epoch_mjds = np.linspace(
            epoch_start_mjd, epoch_start_mjd + epoch_span_days, n_epochs
        )

        # Assign stations: first 1/3 to X05, middle 1/3 to T08, last 1/3 to I41
        station_codes = (
            ["X05"] * epochs_per_station
            + ["T08"] * epochs_per_station
            + ["I41"] * epochs_per_station
        )

        observation_times = Timestamp.from_mjd(epoch_mjds, scale="tdb")

        print(
            f"Created {n_epochs} epochs from {epoch_start_mjd:.1f} to {epoch_start_mjd + epoch_span_days:.1f} MJD"
        )
        print(f"Station assignment: {epochs_per_station} epochs each for X05, T08, I41")

        # Step 4: Generate ephemerides for all variants
        print(f"Generating ephemerides for {len(variant_orbits)} variants...")

        # Initialize assist propagator (following existing test patterns)
        propagator = ASSISTPropagator()

        # Create observers for each epoch/station combination
        observers = Observers.from_codes(
            times=observation_times,
            codes=station_codes,
        )

        # Generate ephemerides for all variants (with error handling for invalid variants)
        try:
            ephemerides = propagator.generate_ephemeris(
                variant_orbits,
                observers,
                max_processes=1,  # Keep CI-friendly
            )
        except ValueError as e:
            if "Distance from observer is NaN" in str(e):
                # Some covariance samples may be invalid; reduce sample size and retry
                print(
                    f"Warning: Invalid variants detected, reducing sample size from {n_variants} to {n_variants//2}"
                )
                variant_orbits = VariantOrbits.create(
                    nominal_orbit,
                    method="monte-carlo",
                    num_samples=n_variants // 2,
                    seed=42,
                )
                ephemerides = propagator.generate_ephemeris(
                    variant_orbits,
                    observers,
                    max_processes=1,
                )
                print(
                    f"Successfully generated ephemerides with {len(variant_orbits)} variants"
                )
            else:
                raise

        print(f"Generated {len(ephemerides)} ephemeris points")

        # Step 5: Convert ephemerides to observation rays
        print("Converting ephemerides to observation rays...")

        # Create detection IDs
        det_ids = [
            f"{ephemerides.orbit_id[i].as_py()}:{ephemerides.coordinates.origin.code[i].as_py()}:{i}"
            for i in range(len(ephemerides))
        ]

        # Use convenience function and reuse already-created observers
        rays = ephemeris_to_rays(ephemerides, observers=observers, det_id=det_ids)

        print(f"Created {len(rays)} observation rays")

        # Step 6: Build BVH from nominal orbit polyline
        print("Building BVH from nominal orbit polyline...")

        # Sample nominal orbit polyline
        plane_params, segments = sample_ellipse_adaptive(
            nominal_orbit,
            max_chord_arcmin=max_chord_arcmin,
        )

        print(
            f"Sampled {len(segments)} polyline segments with max chord {max_chord_arcmin} arcmin"
        )

        # Compute segment AABBs and build BVH index
        segments_with_aabbs = compute_segment_aabbs(segments)
        bvh = build_bvh_index_from_segments(segments_with_aabbs)
        print(f"Built BVH with {len(bvh.nodes)} nodes")

        # Step 7: Run geometric overlap query
        print(f"Running geometric overlap with {guard_arcmin} arcmin guard...")

        # Query BVH for overlaps using high-level API
        overlap_hits = query_bvh(
            bvh,
            rays,
            guard_arcmin=guard_arcmin,
        )

        print(f"Found {len(overlap_hits)} total hits")

        # Step 8: Verify recovery
        print("Verifying recovery...")

        # Extract hit detection IDs
        if len(overlap_hits) > 0:
            hit_det_ids = set(overlap_hits.det_id.to_pylist())
        else:
            hit_det_ids = set()

        # All rays should have hits (100% recovery)
        all_det_ids = set(det_ids)
        missed_det_ids = all_det_ids - hit_det_ids

        recovery_fraction = len(hit_det_ids) / len(all_det_ids)

        print(
            f"Recovery: {len(hit_det_ids)}/{len(all_det_ids)} = {recovery_fraction:.3f}"
        )
        print(
            f"Guard: {guard_arcmin} arcmin, Polyline: {max_chord_arcmin} arcmin, Segments: {len(segments)}"
        )

        if missed_det_ids:
            print(
                f"Missed detections: {sorted(list(missed_det_ids))[:10]}..."
            )  # Show first 10

        # Assert recovery based on expected threshold
        if expected_recovery > 0.0:
            assert recovery_fraction >= expected_recovery, (
                f"Expected ≥{expected_recovery:.1%} recovery, got {recovery_fraction:.3f}. "
                f"Missed {len(missed_det_ids)} detections. "
                f"Object: {sbdb_id}, Guard: {guard_arcmin} arcmin. "
                f"Consider increasing guard radius or tightening polyline max chord."
            )
            print(
                f"✓ SBDB E2E recovery test passed! ({recovery_fraction:.1%} ≥ {expected_recovery:.1%})"
            )
        else:
            print(f"ℹ SBDB E2E recovery logged: {recovery_fraction:.1%} (no assertion)")

    def _measure_recovery_for(
        self,
        sbdb_id: str,
        n_variants: int,
        n_epochs: int,
        guard_arcmin: float,
        max_chord_arcmin: float,
    ) -> dict:
        """
        Measure recovery fraction and index/query metrics for a given guard/chord.

        Returns a metrics dictionary without assertions.
        """
        # Fetch nominal orbit
        nominal_orbits = query_sbdb([sbdb_id])
        nominal_orbit = nominal_orbits[0:1]

        # Generate variants
        variant_orbits = VariantOrbits.create(
            nominal_orbit,
            method="monte-carlo",
            num_samples=n_variants,
            seed=42,
        )

        # Epochs/stations
        epoch_start_mjd = nominal_orbit.coordinates.time.mjd().to_numpy()[0]
        epoch_span_days = 90
        if n_epochs % 3 != 0:
            raise ValueError("n_epochs must be multiple of 3")
        epochs_per_station = n_epochs // 3
        epoch_mjds = np.linspace(
            epoch_start_mjd, epoch_start_mjd + epoch_span_days, n_epochs
        )
        station_codes = (
            ["X05"] * epochs_per_station
            + ["T08"] * epochs_per_station
            + ["I41"] * epochs_per_station
        )
        observation_times = Timestamp.from_mjd(epoch_mjds, scale="tdb")

        # Ephemerides
        propagator = ASSISTPropagator()
        observers = Observers.from_codes(times=observation_times, codes=station_codes)
        try:
            ephemerides = propagator.generate_ephemeris(
                variant_orbits,
                observers,
                max_processes=1,
            )
        except ValueError as e:
            if "Distance from observer is NaN" in str(e):
                variant_orbits = VariantOrbits.create(
                    nominal_orbit,
                    method="monte-carlo",
                    num_samples=max(1, n_variants // 2),
                    seed=42,
                )
                ephemerides = propagator.generate_ephemeris(
                    variant_orbits,
                    observers,
                    max_processes=1,
                )
            else:
                raise

        # Rays
        det_ids = [
            f"{ephemerides.orbit_id[i].as_py()}:{ephemerides.coordinates.origin.code[i].as_py()}:{i}"
            for i in range(len(ephemerides))
        ]
        rays = ephemeris_to_rays(ephemerides, observers=observers, det_id=det_ids)

        # Build BVH
        _, segments = sample_ellipse_adaptive(
            nominal_orbit,
            max_chord_arcmin=max_chord_arcmin,
        )
        segments_with_aabbs = compute_segment_aabbs(segments)
        bvh = build_bvh_index_from_segments(segments_with_aabbs)

        # Query
        overlap_hits = query_bvh(
            bvh,
            rays,
            guard_arcmin=guard_arcmin,
        )

        hit_det_ids = (
            set(overlap_hits.det_id.to_pylist()) if len(overlap_hits) > 0 else set()
        )
        all_det_ids = set(det_ids)
        recovery_fraction = len(hit_det_ids) / len(all_det_ids)

        metrics = {
            "sbdb_id": sbdb_id,
            "guard_arcmin": guard_arcmin,
            "max_chord_arcmin": max_chord_arcmin,
            "num_segments": len(segments),
            "num_bvh_nodes": len(bvh.nodes),
            "num_hits": len(overlap_hits),
            "num_rays": len(rays),
            "recovery_fraction": recovery_fraction,
        }
        print(
            f"MEASURE: {sbdb_id} guard={guard_arcmin}′ chord={max_chord_arcmin}′ "
            f"segs={metrics['num_segments']} nodes={metrics['num_bvh_nodes']} "
            f"hits={metrics['num_hits']} rays={metrics['num_rays']} rec={recovery_fraction:.3f}"
        )
        return metrics

    @pytest.mark.parametrize("guard_arcmin", [0.5, 0.25, 0.1])
    def test_min_segments_for_100pct_ci(self, guard_arcmin):
        """
        For each guard, sweep chord sizes and find the coarsest chord that yields 100% recovery.
        """
        sbdb_id = "1998 SG172"
        n_variants = 10
        n_epochs = 21
        # Coarse → fine
        chord_sweep = [60.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1]

        found_chord = None
        found_metrics = None
        for chord in chord_sweep:
            m = self._measure_recovery_for(
                sbdb_id=sbdb_id,
                n_variants=n_variants,
                n_epochs=n_epochs,
                guard_arcmin=guard_arcmin,
                max_chord_arcmin=chord,
            )
            if m["recovery_fraction"] >= 1.0:
                found_chord = chord
                found_metrics = m
                break

        assert (
            found_chord is not None
        ), f"No chord in sweep achieved 100% recovery for guard={guard_arcmin} arcmin"
        print(
            f"RESULT: guard={guard_arcmin}′ minimal chord={found_chord}′ "
            f"segs/orbit={found_metrics['num_segments']} nodes={found_metrics['num_bvh_nodes']}"
        )

    @pytest.mark.parametrize(
        "sbdb_id",
        [
            "1862 Apollo",  # high-e NEO
            "944 Hidalgo",  # very high-i
            "1221 Amor",  # high-e Amor NEO
        ],
    )
    def test_min_segments_for_100pct_ci_extremes(self, sbdb_id):
        """
        For high-e and high-i objects, sweep chord sizes at a tight guard (0.1′)
        and find the coarsest chord that yields 100% recovery.
        """
        guard_arcmin = 0.1
        n_variants = 10
        n_epochs = 21
        # Coarse → fine
        chord_sweep = [60.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25]

        found_chord = None
        found_metrics = None
        for chord in chord_sweep:
            m = self._measure_recovery_for(
                sbdb_id=sbdb_id,
                n_variants=n_variants,
                n_epochs=n_epochs,
                guard_arcmin=guard_arcmin,
                max_chord_arcmin=chord,
            )
            if m["recovery_fraction"] >= 1.0:
                found_chord = chord
                found_metrics = m
                break

        assert (
            found_chord is not None
        ), f"No chord in sweep achieved 100% recovery for {sbdb_id} at guard={guard_arcmin} arcmin"
        print(
            f"RESULT: {sbdb_id} guard={guard_arcmin}′ minimal chord={found_chord}′ "
            f"segs/orbit={found_metrics['num_segments']} nodes={found_metrics['num_bvh_nodes']}"
        )
