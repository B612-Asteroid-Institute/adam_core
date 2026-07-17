import uuid
from typing import Literal, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from .._rust.api import add_light_time_numpy
from ..constants import Constants as c
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import transform_coordinates
from ..observers.observers import Observers
from ..time import Timestamp
from .ephemeris import Ephemeris
from .orbits import Orbits
from .physical_parameters import PhysicalParameters


class VariantOrbits(qv.Table):
    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    variant_id = qv.LargeStringColumn(nullable=True)
    weights = qv.Float64Column(nullable=True)
    weights_cov = qv.Float64Column(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    physical_parameters = PhysicalParameters.as_column(nullable=True)

    @classmethod
    def create(
        cls,
        orbits: Orbits,
        method: Literal["auto", "sigma-point", "monte-carlo"] = "auto",
        num_samples: int = 10000,
        alpha: float = 1,
        beta: float = 0,
        kappa: float = 0,
        seed: Optional[int] = None,
    ) -> "VariantOrbits":
        """
        Sample and create variants for the given orbits by sampling the covariance matrices.
        There are three supported methods:
        - sigma-point: Sample the covariance matrix using sigma points. This is the fastest method,
        but can be inaccurate if the covariance matrix is not well behaved.
        - monte-carlo: Sample the covariance matrix using a monte carlo method.
        This is the slowest method, but is the most accurate. 10k samples are drawn.
        - auto: Automatically select the best method based on the covariance matrix.
        If the covariance matrix is well behaved then sigma-point sampling will be used.
        If the covariance matrix is not well behaved then monte-carlo sampling will be used.

        When sampling with monte-carlo, 10k samples are drawn. Sigma-point sampling draws 13 samples
        for 6-dimensional coordinates.

        Parameters
        ----------
        orbits : '~adam_core.orbits.orbits.Orbits'
            The orbits for which to create variant orbits.
        method : {'sigma-point', 'monte-carlo', 'auto'}, optional
            The method to use for sampling the covariance matrix. If 'auto' is selected then the method
            will be automatically selected based on the covariance matrix. The default is 'auto'.
        num_samples : int, optional
            The number of samples to draw when sampling with monte-carlo.
        alpha : float, optional
            Spread of the sigma points between 1e^-2 and 1.
        beta : float, optional
            Prior knowledge of the distribution when generating sigma points usually set to 2 for a Gaussian.
        kappa : float, optional
            Secondary scaling parameter when generating sigma points usually set to 0.
        seed : int, optional
            Seed for monte-carlo sampling (including the auto-mode fallback).
            Given a seed the samples are reproducible; without one a fresh
            random seed is drawn per call.

        Returns
        -------
        variants_orbits : '~adam_core.orbits.variants.VariantOrbits'
            The variant orbits.

        Notes
        -----
        Sampling runs in the Rust backend. Sigma-point output is deterministic
        and matches the legacy Python implementation. Monte Carlo draws use a
        Rust-native RNG that is statistically equivalent to, but not
        bit-identical with, the legacy scipy sampler (decision 2026-07-03).
        Unlike legacy, ``seed`` also applies to the auto-mode Monte Carlo
        fallback so auto-mode is reproducible given a seed.
        """
        if method not in ("auto", "sigma-point", "monte-carlo"):
            raise ValueError(f"Unknown coordinate covariance sampling method: {method}")

        # All three methods run through the Rust sampler (decision 2026-07-03:
        # exact scipy RNG parity is not required). Monte Carlo draws are
        # statistically equivalent to, but not bit-identical with, the legacy
        # scipy path; sigma-point output is deterministic and matches legacy.
        # One intentional improvement over legacy: ``seed`` now also applies
        # to the auto-mode Monte Carlo fallback (legacy ignored it there).
        from .arrow_bridge import _sample_orbit_variants_arrow

        if seed is None and method in ("auto", "monte-carlo"):
            # The Rust sampler is deterministic given a seed; draw a fresh one
            # so unseeded Monte Carlo sampling stays nondeterministic, matching
            # the legacy contract.
            seed = int(np.random.default_rng().integers(0, 2**63, dtype=np.int64))

        return _sample_orbit_variants_arrow(
            orbits,
            method=method,
            num_samples=num_samples,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            seed=seed,
        )

    def link_to_orbits(
        self, orbits: Orbits
    ) -> qv.MultiKeyLinkage[Orbits, "VariantOrbits"]:
        """
        Link variants to the orbits from which they were generated.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits`
            Orbits from which the variants were generated.

        Returns
        -------
        linkage : `~quivr.MultiKeyLinkage[Orbits, VariantOrbits]`
            Linkage between variants and orbits.
        """
        assert orbits.coordinates.time.scale == self.coordinates.time.scale

        # We might want to replace linking on jd1 and jd2 with just linking on mjd
        # once the changes have been merged
        return qv.MultiKeyLinkage(
            orbits,
            self,
            left_keys={
                "orbit_id": orbits.orbit_id,
                "day": orbits.coordinates.time.days,
                "millis": orbits.coordinates.time.millis(),
            },
            right_keys={
                "orbit_id": self.orbit_id,
                "day": self.coordinates.time.days,
                "millis": self.coordinates.time.millis(),
            },
        )

    def collapse(self, orbits: Orbits) -> Orbits:
        """
        Collapse the variants and recalculate the covariance matrix for each
        each orbit at each epoch. The mean state is taken from the orbits class and
        is not calculated from the variants.

        Parameters
        ----------
        orbits : `~adam_core.orbits.orbits.Orbits`
            Orbits from which the variants were generated.

        Returns
        -------
        collapsed_orbits : `~adam_core.orbits.orbits.Orbits`
            The collapsed orbits.
        """
        from adam_core import _rust_native

        from .arrow_bridge import (
            orbits_from_record_batch,
            orbits_to_record_batch,
            variants_to_record_batch,
        )

        # Legacy linkage asserted matching time scales before collapsing.
        assert orbits.coordinates.time.scale == self.coordinates.time.scale

        # One Rust crossing owns linkage grouping, weighted covariance, and
        # finished-table assembly; the mean state stays the orbit row state.
        collapsed = _rust_native.collapse_variant_orbits_arrow(
            orbits_to_record_batch(orbits),
            variants_to_record_batch(self),
        )
        return orbits_from_record_batch(collapsed)

    def collapse_by_object_id(self) -> Orbits:
        """
        Collapse the variant orbits into a mean and covariance matrix.

        Returns
        -------
        collapsed_orbits : `~adam_core.orbits.orbits.Orbits`
            The collapsed orbits.
        """
        from adam_core import _rust_native

        from .arrow_bridge import (
            orbits_from_record_batch,
            variants_to_record_batch,
        )

        n_objects = len(self.object_id.unique())
        if n_objects == 0:
            return Orbits.empty()

        # Fresh random output identities are input preparation; grouping,
        # single-epoch/origin validation, means, covariances, and finished
        # table assembly are one Rust crossing.
        orbit_ids = [uuid.uuid4().hex for _ in range(n_objects)]
        try:
            collapsed = _rust_native.collapse_variant_orbits_by_object_id_arrow(
                variants_to_record_batch(self), orbit_ids
            )
        except ValueError as exc:
            if "assertion failed" in str(exc):
                raise AssertionError(str(exc)) from None
            raise
        return orbits_from_record_batch(collapsed)


class VariantEphemeris(qv.Table):

    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    variant_id = qv.LargeStringColumn(nullable=True)
    weights = qv.Float64Column(nullable=True)
    weights_cov = qv.Float64Column(nullable=True)
    coordinates = SphericalCoordinates.as_column()
    aberrated_coordinates = CartesianCoordinates.as_column(nullable=True)
    predicted_magnitude_v = qv.Float64Column(nullable=True)
    alpha = qv.Float64Column(nullable=True)
    light_time = qv.Float64Column(nullable=True)

    def link_to_ephemeris(
        self, ephemeris: Ephemeris
    ) -> qv.MultiKeyLinkage[Ephemeris, "VariantEphemeris"]:
        """
        Link variants to the ephemeris for which they were generated.

        Parameters
        ----------
        ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
            Ephemeris for which the variants were generated.

        Returns
        -------
        linkage : `~quivr.MultiKeyLinkage[Ephemeris, EphemerisVariants]`
            Linkage between variants and ephemeris.
        """
        assert ephemeris.coordinates.time.scale == self.coordinates.time.scale

        # We might want to replace linking on jd1 and jd2 with just linking on mjd
        # once the changes have been merged
        return qv.MultiKeyLinkage(
            ephemeris,
            self,
            left_keys={
                "orbit_id": ephemeris.orbit_id,
                "day": ephemeris.coordinates.time.days,
                "millis": ephemeris.coordinates.time.millis(),
            },
            right_keys={
                "orbit_id": self.orbit_id,
                "day": self.coordinates.time.days,
                "millis": self.coordinates.time.millis(),
            },
        )

    def collapse(self, ephemeris: Ephemeris) -> Ephemeris:
        """
        Collapse the variants and recalculate the covariance matrix for each
        each ephemeris at each epoch. The mean state is taken from the ephemeris class and
        is not calculate from the variants.

        Parameters
        ----------
        ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
            Ephemeris for which the variants were generated.

        Returns
        -------
        collapsed_ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
            The collapsed ephemeris (with covariance matrices calculated based
            on the samples).
        """
        from adam_core import _rust_native

        # Legacy linkage asserted matching time scales before collapsing.
        assert ephemeris.coordinates.time.scale == self.coordinates.time.scale

        # Legacy collapsed aberrated covariances whenever any variant carried
        # aberrated coordinates; preserve that global switch.
        has_aberrated = not pc.all(pc.is_null(self.aberrated_coordinates.x)).as_py()
        aberrated_kwargs = {}
        if has_aberrated:
            aberrated_kwargs = {
                "orbit_aberrated_means": np.ascontiguousarray(
                    ephemeris.aberrated_coordinates.values, dtype=np.float64
                ),
                "variant_aberrated": np.ascontiguousarray(
                    self.aberrated_coordinates.values, dtype=np.float64
                ),
            }

        # One Rust crossing owns the linkage grouping and every weighted
        # covariance; the mean states stay the ephemeris row states.
        covariance_spherical, covariance_aberrated = (
            _rust_native.collapse_variant_ephemeris_covariances_numpy(
                ephemeris.orbit_id.to_pylist(),
                ephemeris.coordinates.time.days.to_numpy(zero_copy_only=False),
                ephemeris.coordinates.time.millis().to_numpy(zero_copy_only=False),
                np.ascontiguousarray(ephemeris.coordinates.values, dtype=np.float64),
                self.orbit_id.to_pylist(),
                self.coordinates.time.days.to_numpy(zero_copy_only=False),
                self.coordinates.time.millis().to_numpy(zero_copy_only=False),
                np.ascontiguousarray(self.coordinates.values, dtype=np.float64),
                self.weights_cov.to_numpy(zero_copy_only=False),
                **aberrated_kwargs,
            )
        )

        collapsed = ephemeris.set_column(
            "coordinates.covariance",
            CoordinateCovariances.from_matrix(covariance_spherical.reshape(-1, 6, 6)),
        )
        if covariance_aberrated is not None:
            collapsed = collapsed.set_column(
                "aberrated_coordinates.covariance",
                CoordinateCovariances.from_matrix(
                    covariance_aberrated.reshape(-1, 6, 6)
                ),
            )
        return collapsed

    def collapse_by_object_id(
        self,
        *,
        aberration_mode: Literal["recompute", "collapse", "none"] = "recompute",
        group_chunk_size: int = 200_000,
    ) -> Ephemeris:
        """
        Collapse the variant ephemerides into mean ephemerides and covariance matrices
        grouped by object_id, time, and observatory (origin code).

        Returns
        -------
        collapsed_ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
            The collapsed ephemeris.
        """
        if len(self) == 0:
            return Ephemeris.empty()

        aberration_mode = str(aberration_mode).strip().lower()
        if aberration_mode not in {"recompute", "collapse", "none"}:
            raise ValueError(
                "aberration_mode must be one of {'recompute','collapse','none'}; "
                f"got {aberration_mode!r}"
            )
        group_chunk_size = int(group_chunk_size)
        if group_chunk_size <= 0:
            raise ValueError("group_chunk_size must be > 0")

        from adam_core import _rust_native

        # Legacy behavior: aberrated collapse required constant group sizes.
        # Detect that before the crossing so mode='collapse' can degrade the
        # same way, and pass aberrated inputs only when they can be used.
        try:
            has_ab = not pc.all(pc.is_null(self.aberrated_coordinates.x)).as_py()
            has_lt = not pc.all(pc.is_null(self.light_time)).as_py()
        except Exception:
            has_ab = False
            has_lt = False
        collapse_aberrated = aberration_mode == "collapse" and has_ab and has_lt
        aberrated_kwargs = {}
        if collapse_aberrated:
            aberrated_kwargs = {
                "aberrated_values": np.ascontiguousarray(
                    self.aberrated_coordinates.values, dtype=np.float64
                ),
                "light_times": pc.fill_null(self.light_time, np.nan).to_numpy(
                    zero_copy_only=False
                ),
            }

        # One Rust crossing owns the stable sort, grouping, circular longitude
        # statistics, weighted means/covariances, and magnitude reductions.
        (
            sorted_indices,
            group_starts,
            means_sph,
            cov_flat,
            mags_out,
            aberrated_out,
        ) = _rust_native.collapse_variant_ephemeris_by_object_numpy(
            self.object_id.to_pylist(),
            self.coordinates.time.days.to_numpy(zero_copy_only=False),
            self.coordinates.time.nanos.to_numpy(zero_copy_only=False),
            self.coordinates.origin.code.to_pylist(),
            np.ascontiguousarray(self.coordinates.values, dtype=np.float64),
            pc.fill_null(self.weights, np.nan).to_numpy(zero_copy_only=False),
            pc.fill_null(self.weights_cov, np.nan).to_numpy(zero_copy_only=False),
            pc.fill_null(self.predicted_magnitude_v, np.nan).to_numpy(
                zero_copy_only=False
            ),
            **aberrated_kwargs,
        )

        n_groups = int(len(group_starts))
        start_rows = sorted_indices[group_starts].tolist()
        counts = np.diff(np.concatenate([group_starts, [len(self)]])).astype(
            np.int64, copy=False
        )
        uniform_group_size = bool(counts.size > 0 and np.all(counts == counts[0]))
        K = int(counts[0]) if uniform_group_size else -1

        # Preserve orbit_id (critical for downstream pipelines that key on orbit_id).
        orbit_id = pc.take(self.orbit_id, start_rows)
        object_id = pc.take(self.object_id, start_rows)
        out_time = Timestamp.from_kwargs(
            days=pc.take(self.coordinates.time.days, start_rows),
            nanos=pc.take(self.coordinates.time.nanos, start_rows),
            scale=self.coordinates.time.scale,
        )
        out_origin = Origin.from_kwargs(
            code=pc.take(self.coordinates.origin.code, start_rows)
        )
        predicted_magnitude_v: list[float | None] = [
            float(value) if np.isfinite(value) else None for value in mags_out.tolist()
        ]

        collapsed_coordinates = SphericalCoordinates.from_kwargs(
            rho=means_sph[:, 0],
            lon=means_sph[:, 1],
            lat=means_sph[:, 2],
            vrho=means_sph[:, 3],
            vlon=means_sph[:, 4],
            vlat=means_sph[:, 5],
            covariance=CoordinateCovariances.from_matrix(
                cov_flat.reshape(n_groups, 6, 6)
            ),
            time=out_time,
            origin=out_origin,
            frame=self.coordinates.frame,
        )

        # The collapsed mean ephemeris (UT mean + covariance).
        ephemeris = Ephemeris.from_kwargs(
            orbit_id=orbit_id,
            object_id=object_id,
            coordinates=collapsed_coordinates,
            predicted_magnitude_v=predicted_magnitude_v,
        )

        if aberration_mode == "none":
            return ephemeris

        if aberration_mode == "collapse":
            # If the variants already contain aberrated coordinates + light_time, avoid any
            # regeneration and just collapse those too (weighted, in the same crossing).
            if aberrated_out is not None and uniform_group_size and K > 0:
                out_ab, out_lt = aberrated_out
                if np.all(np.isfinite(out_ab)) and np.any(np.isfinite(out_lt)):
                    light_time = np.where(
                        np.isfinite(out_lt),
                        out_lt,
                        np.linalg.norm(out_ab[:, :3], axis=1) / c.C,
                    )
                    emission_times = ephemeris.coordinates.time.rescale(
                        "tdb"
                    ).add_fractional_days(pa.array(-light_time, type=pa.float64()))
                    aberrated_coordinates = CartesianCoordinates.from_kwargs(
                        x=out_ab[:, 0],
                        y=out_ab[:, 1],
                        z=out_ab[:, 2],
                        vx=out_ab[:, 3],
                        vy=out_ab[:, 4],
                        vz=out_ab[:, 5],
                        time=emission_times,
                        origin=Origin.from_kwargs(
                            code=np.full(
                                len(ephemeris), OriginCodes.SOLAR_SYSTEM_BARYCENTER.name
                            )
                        ),
                        frame="ecliptic",
                    )
                    return ephemeris.set_column(
                        "light_time", pa.array(light_time, type=pa.float64())
                    ).set_column("aberrated_coordinates", aberrated_coordinates)
            # If collapse isn't possible, fall back to returning no aberrations (do not recompute).
            return ephemeris

        # aberration_mode == "recompute": regenerate aberrated coordinates + light_time
        # for the collapsed mean states.
        observers = Observers.from_codes(
            ephemeris.coordinates.origin.code, ephemeris.coordinates.time
        )
        observers_barycentric = observers.set_column(
            "coordinates",
            transform_coordinates(
                observers.coordinates,
                CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
            ),
        )

        topocentric = transform_coordinates(
            ephemeris.coordinates, CartesianCoordinates, frame_out="ecliptic"
        )
        topo_vals = topocentric.values
        obs_vals = observers_barycentric.coordinates.values
        barycentric_vals = topo_vals + obs_vals

        times_tdb_mjd = (
            ephemeris.coordinates.time.rescale("tdb")
            .mjd()
            .to_numpy(zero_copy_only=False)
        )
        observer_positions = observers_barycentric.coordinates.r

        # Use the same iterative light-time correction as ephemeris generation.
        n = barycentric_vals.shape[0]
        rust_out = add_light_time_numpy(
            np.ascontiguousarray(barycentric_vals, dtype=np.float64),
            np.ascontiguousarray(observer_positions, dtype=np.float64),
            np.full(n, c.MU, dtype=np.float64),
            lt_tol=1e-12,
            max_iter=100,
            tol=1e-15,
            max_lt_iter=10,
        )
        aberrated_vals, light_time = rust_out
        del times_tdb_mjd  # not needed; LT depends only on relative position

        # add_light_time assumes a physically valid inertial state; if the collapsed
        # topocentric state is not dynamically consistent, fall back to geometric
        # light-time using topocentric range and keep the non-propagated barycentric state.
        if not np.all(np.isfinite(light_time)):
            light_time = np.linalg.norm(topo_vals[:, :3], axis=1) / c.C
            aberrated_vals = barycentric_vals

        emission_times = ephemeris.coordinates.time.rescale("tdb").add_fractional_days(
            pa.array(-light_time, type=pa.float64())
        )

        aberrated_coordinates = CartesianCoordinates.from_kwargs(
            x=aberrated_vals[:, 0],
            y=aberrated_vals[:, 1],
            z=aberrated_vals[:, 2],
            vx=aberrated_vals[:, 3],
            vy=aberrated_vals[:, 4],
            vz=aberrated_vals[:, 5],
            time=emission_times,
            origin=Origin.from_kwargs(
                code=np.full(len(ephemeris), OriginCodes.SOLAR_SYSTEM_BARYCENTER.name)
            ),
            frame="ecliptic",
        )

        return ephemeris.set_column(
            "light_time", pa.array(light_time, type=pa.float64())
        ).set_column("aberrated_coordinates", aberrated_coordinates)

    def collapse_sigma_points_orbit_major(
        self,
        *,
        n_times: int,
        n_variants: int = 13,
    ) -> Ephemeris:
        """
        Fast sigma-point collapse for the common layout produced by:
        `VariantOrbits.create(method="sigma-point")` + `propagate_2body(variants, times)`.

        Assumptions
        -----------
        - The variant ephemeris rows are ordered in *base-variant-major* blocks, where each
          base variant has a contiguous block of `n_times` rows in time order.
        - Base variants are ordered in *orbit-major* order, with `variant_id` cycling
          as "0".."n_variants-1" for each orbit.

        This method avoids the expensive `sort_by` in `collapse_by_object_id()` and computes
        UT mean + covariance directly via reshaping and vectorized numpy operations.
        """
        if len(self) == 0:
            return Ephemeris.empty()

        n_times = int(n_times)
        n_variants = int(n_variants)
        if n_times <= 0:
            raise ValueError("n_times must be > 0")
        if n_variants <= 0:
            raise ValueError("n_variants must be > 0")

        total = int(len(self))
        denom = int(n_times * n_variants)
        if denom <= 0 or (total % denom) != 0:
            raise ValueError(
                "Variant ephemeris length must be a multiple of n_times*n_variants. "
                f"len={total} n_times={n_times} n_variants={n_variants}"
            )
        n_orbits = int(total // denom)

        # Validate that base variants are orbit-major (n_variants consecutive base variants per orbit).
        base0 = np.arange(0, total, n_times, dtype=np.int64)  # (n_base,)
        orbit_id_base = (
            pc.take(pc.cast(self.orbit_id, pa.large_string()), base0.tolist())
            .to_numpy(zero_copy_only=False)
            .astype(object)
        )
        object_id_base = (
            pc.take(pc.cast(self.object_id, pa.large_string()), base0.tolist())
            .to_numpy(zero_copy_only=False)
            .astype(object)
        )
        orbit_id_base2 = orbit_id_base.reshape(n_orbits, n_variants)
        object_id_base2 = object_id_base.reshape(n_orbits, n_variants)
        if not (
            np.all(orbit_id_base2 == orbit_id_base2[:, [0]])
            and np.all(object_id_base2 == object_id_base2[:, [0]])
        ):
            # Unknown layout; fall back to generic (sorted) implementation.
            return self.collapse_by_object_id(aberration_mode="none")

        # One orbit_id/object_id per orbit (use the first base variant for that orbit).
        orbit0 = np.arange(0, total, denom, dtype=np.int64)  # (n_orbits,)
        orbit_id_orbit = (
            pc.take(pc.cast(self.orbit_id, pa.large_string()), orbit0.tolist())
            .to_numpy(zero_copy_only=False)
            .astype(object)
        )
        object_id_orbit = (
            pc.take(pc.cast(self.object_id, pa.large_string()), orbit0.tolist())
            .to_numpy(zero_copy_only=False)
            .astype(object)
        )

        # Extract shared time/origin grid from first base variant (first n_times rows).
        days0 = (
            pc.take(self.coordinates.time.days, list(range(n_times)))
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        nanos0 = (
            pc.take(self.coordinates.time.nanos, list(range(n_times)))
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        origin0 = (
            pc.take(
                pc.cast(self.coordinates.origin.code, pa.large_string()),
                list(range(n_times)),
            )
            .to_numpy(zero_copy_only=False)
            .astype(object)
        )

        out_days = np.tile(days0, n_orbits)
        out_nanos = np.tile(nanos0, n_orbits)
        out_origin = np.tile(origin0, n_orbits)

        out_time = Timestamp.from_kwargs(
            days=pa.array(out_days, type=pa.int64()),
            nanos=pa.array(out_nanos, type=pa.int64()),
            scale=self.coordinates.time.scale,
        )
        out_origin_tbl = Origin.from_kwargs(
            code=pa.array(out_origin, type=pa.large_string())
        )

        from adam_core import _rust_native

        # UT weights per (orbit, variant) (constant across times); legacy
        # applies the base-variant weight to every epoch of that variant.
        w_mean_rows = np.repeat(
            pc.take(self.weights, base0.tolist())
            .cast(pa.float64())
            .fill_null(pa.scalar(np.nan, type=pa.float64()))
            .to_numpy(zero_copy_only=False)
            .astype(np.float64, copy=False),
            n_times,
        )
        w_cov_rows = np.repeat(
            pc.take(self.weights_cov, base0.tolist())
            .cast(pa.float64())
            .fill_null(pa.scalar(np.nan, type=pa.float64()))
            .to_numpy(zero_copy_only=False)
            .astype(np.float64, copy=False),
            n_times,
        )

        # Synthesized orbit-major keys: sorting by (orbit, time) recovers the
        # output layout while the Rust core owns circular statistics, weight
        # normalization, and weighted covariance per (orbit, time) group.
        row_orbit = np.arange(total, dtype=np.int64) // denom
        row_time = np.arange(total, dtype=np.int64) % n_times
        synthetic_ids = [f"{orbit:012d}" for orbit in row_orbit.tolist()]
        (
            _sorted_indices,
            group_starts,
            mean_flat,
            cov_groups,
            _mags_out,
            _aberrated_out,
        ) = _rust_native.collapse_variant_ephemeris_by_object_numpy(
            synthetic_ids,
            row_time,
            np.zeros(total, dtype=np.int64),
            [""] * total,
            np.ascontiguousarray(self.coordinates.values, dtype=np.float64),
            w_mean_rows,
            w_cov_rows,
            np.full(total, np.nan, dtype=np.float64),
        )
        if len(group_starts) != n_orbits * n_times:
            raise ValueError(
                "sigma-point collapse produced an unexpected group count: "
                f"{len(group_starts)} != {n_orbits * n_times}"
            )
        cov_flat = cov_groups.reshape(n_orbits * n_times, 6, 6)

        collapsed_coordinates = SphericalCoordinates.from_kwargs(
            rho=mean_flat[:, 0],
            lon=mean_flat[:, 1],
            lat=mean_flat[:, 2],
            vrho=mean_flat[:, 3],
            vlon=mean_flat[:, 4],
            vlat=mean_flat[:, 5],
            covariance=CoordinateCovariances.from_matrix(cov_flat),
            time=out_time,
            origin=out_origin_tbl,
            frame=self.coordinates.frame,
        )

        out_orbit_id = pa.array(
            np.repeat(orbit_id_orbit, n_times), type=pa.large_string()
        )
        out_object_id = pa.array(
            np.repeat(object_id_orbit, n_times), type=pa.large_string()
        )

        return Ephemeris.from_kwargs(
            orbit_id=out_orbit_id,
            object_id=out_object_id,
            coordinates=collapsed_coordinates,
            predicted_magnitude_v=pa.nulls(n_orbits * n_times, type=pa.float64()),
        )
