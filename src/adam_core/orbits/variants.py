import uuid
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances, weighted_covariance
from ..coordinates.origin import Origin
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.variants import VariantCoordinatesTable, create_coordinate_variants
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

        Returns
        -------
        variants_orbits : '~adam_core.orbits.variants.VariantOrbits'
            The variant orbits.
        """
        variant_coordinates: VariantCoordinatesTable = create_coordinate_variants(
            orbits.coordinates,
            method=method,
            num_samples=num_samples,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            seed=seed,
        )

        return cls.from_kwargs(
            orbit_id=pc.take(orbits.orbit_id, variant_coordinates.index),
            object_id=pc.take(orbits.object_id, variant_coordinates.index),
            variant_id=np.array(
                np.arange(len(variant_coordinates)).astype(str), dtype="object"
            ),
            weights=variant_coordinates.weight,
            weights_cov=variant_coordinates.weight_cov,
            coordinates=variant_coordinates.sample,
            physical_parameters=orbits.physical_parameters.take(
                variant_coordinates.index
            ),
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
        link = self.link_to_orbits(orbits)

        # Iterate over the variants and calculate the mean state and covariance matrix
        # for each orbit at each epoch then create a new orbit with the calculated covariance matrix
        orbits_list = []
        for orbit in orbits:
            assert len(orbit) == 1

            key = link.key(
                orbit_id=orbit.orbit_id[0].as_py(),
                day=orbit.coordinates.time.days[0].as_py(),
                millis=orbit.coordinates.time.millis()[0].as_py(),
            )
            variants = link.select_right(key)

            samples = variants.coordinates.values
            mean = orbit.coordinates.values[0]
            covariance = weighted_covariance(
                mean, samples, variants.weights_cov.to_numpy(zero_copy_only=False)
            ).reshape(1, 6, 6)

            orbit_collapsed = orbit.set_column(
                "coordinates.covariance", CoordinateCovariances.from_matrix(covariance)
            )

            orbits_list.append(orbit_collapsed)

        return qv.concatenate(orbits_list)

    def collapse_by_object_id(self) -> Orbits:
        """
        Collapse the variant orbits into a mean and covariance matrix.

        Returns
        -------
        collapsed_orbits : `~adam_core.orbits.orbits.Orbits`
            The collapsed orbits.
        """
        # Group the variants by object_id
        unique_object_ids = self.object_id.unique()

        orbits = Orbits.empty()
        for object_id in unique_object_ids:
            object_variants = self.select("object_id", object_id)

            # All the variants must have the same epoch
            assert len(object_variants.coordinates.time.unique()) == 1
            assert len(pc.unique(object_variants.coordinates.origin.code)) == 1

            # Calculate the mean
            mean = np.average(
                object_variants.coordinates.values,
                axis=0,
            )

            # Calculate the covariance matrix
            covariance = weighted_covariance(
                mean,
                object_variants.coordinates.values,
                np.ones(len(object_variants), dtype=np.float64) / len(object_variants),
            ).reshape(1, 6, 6)

            # Create the collapsed orbit
            orbit = Orbits.from_kwargs(
                orbit_id=[uuid.uuid4().hex],
                object_id=[object_id],
                physical_parameters=object_variants.physical_parameters.take([0]),
                coordinates=CartesianCoordinates.from_kwargs(
                    x=[mean[0]],
                    y=[mean[1]],
                    z=[mean[2]],
                    vx=[mean[3]],
                    vy=[mean[4]],
                    vz=[mean[5]],
                    covariance=CoordinateCovariances.from_matrix(covariance),
                    time=object_variants.coordinates.time[0],
                    origin=object_variants.coordinates.origin[0],
                    frame=object_variants.coordinates.frame,
                ),
            )
            orbits = qv.concatenate([orbits, orbit])

        return orbits


class VariantEphemeris(qv.Table):

    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    variant_id = qv.LargeStringColumn(nullable=True)
    weights = qv.Float64Column(nullable=True)
    weights_cov = qv.Float64Column(nullable=True)
    coordinates = SphericalCoordinates.as_column()
    aberrated_coordinates = CartesianCoordinates.as_column(nullable=True)
    predicted_magnitude_v = qv.Float64Column(nullable=True)

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
        link = self.link_to_ephemeris(ephemeris)

        # Iterate over the variants and calculate the mean state and covariance matrix
        # for each orbit at each epoch then create a new orbit with the calculated covariance matrix
        ephemeris_list = []
        for ephemeris_i in ephemeris:
            assert len(ephemeris_i) == 1

            key = link.key(
                orbit_id=ephemeris_i.orbit_id[0].as_py(),
                day=ephemeris_i.coordinates.time.days[0].as_py(),
                millis=ephemeris_i.coordinates.time.millis()[0].as_py(),
            )
            variants = link.select_right(key)

            # Collapse topocentric spherical coordinate covariances
            samples_spherical = variants.coordinates.values
            mean_spherical = ephemeris_i.coordinates.values[0]
            covariance_spherical = weighted_covariance(
                mean_spherical,
                samples_spherical,
                variants.weights_cov.to_numpy(zero_copy_only=False),
            ).reshape(1, 6, 6)

            ephemeris_collapsed = ephemeris_i.set_column(
                "coordinates.covariance",
                CoordinateCovariances.from_matrix(covariance_spherical),
            )

            # If aberrated coordinates were provided, also collapse their covariances
            if not pc.all(pc.is_null(variants.aberrated_coordinates.x)).as_py():
                samples_aberrated = variants.aberrated_coordinates.values
                mean_aberrated = ephemeris_i.aberrated_coordinates.values[0]
                covariance_aberrated = weighted_covariance(
                    mean_aberrated,
                    samples_aberrated,
                    variants.weights_cov.to_numpy(zero_copy_only=False),
                ).reshape(1, 6, 6)
                ephemeris_collapsed = ephemeris_collapsed.set_column(
                    "aberrated_coordinates.covariance",
                    CoordinateCovariances.from_matrix(covariance_aberrated),
                )

            ephemeris_list.append(ephemeris_collapsed)

        return qv.concatenate(ephemeris_list)

    def collapse_by_object_id(self) -> Ephemeris:
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

        variants = self.sort_by(
            [
                "object_id",
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        object_ids = variants.object_id.to_numpy(zero_copy_only=False)
        days = variants.coordinates.time.days.to_numpy(zero_copy_only=False)
        nanos = variants.coordinates.time.nanos.to_numpy(zero_copy_only=False)
        origin_codes = variants.coordinates.origin.code.to_numpy(zero_copy_only=False)

        key_change = (
            (object_ids[1:] != object_ids[:-1])
            | (days[1:] != days[:-1])
            | (nanos[1:] != nanos[:-1])
            | (origin_codes[1:] != origin_codes[:-1])
        )
        bounds = np.concatenate(([0], np.nonzero(key_change)[0] + 1, [len(variants)]))
        if len(bounds) <= 2:
            starts = np.array([0], dtype=np.int64)
        else:
            starts = bounds[:-1]

        n_groups = len(bounds) - 1
        orbit_id = [uuid.uuid4().hex for _ in range(n_groups)]

        object_id = pc.take(variants.object_id, starts.tolist())
        out_time = Timestamp.from_kwargs(
            days=pc.take(variants.coordinates.time.days, starts.tolist()),
            nanos=pc.take(variants.coordinates.time.nanos, starts.tolist()),
            scale=variants.coordinates.time.scale,
        )
        out_origin = Origin.from_kwargs(
            code=pc.take(variants.coordinates.origin.code, starts.tolist())
        )

        mags = pc.fill_null(variants.predicted_magnitude_v, np.nan).to_numpy(
            zero_copy_only=False
        )

        spherical_values = variants.coordinates.values
        means_sph = np.empty((n_groups, 6), dtype=np.float64)
        cov_sph = np.empty((n_groups, 6, 6), dtype=np.float64)
        predicted_magnitude_v: list[float | None] = [None] * n_groups

        aberrated_x = variants.aberrated_coordinates.x
        has_aberrated = not pc.all(pc.is_null(aberrated_x)).as_py()
        if has_aberrated:
            assert not pc.any(
                pc.is_null(aberrated_x)
            ).as_py(), "aberrated_coordinates must be provided for all rows or none"
            ab_values = variants.aberrated_coordinates.values
            ab_days = variants.aberrated_coordinates.time.days.to_numpy(
                zero_copy_only=False
            )
            ab_nanos = variants.aberrated_coordinates.time.nanos.to_numpy(
                zero_copy_only=False
            )
            ab_origin_codes = variants.aberrated_coordinates.origin.code.to_numpy(
                zero_copy_only=False
            )
            means_ab = np.empty((n_groups, 6), dtype=np.float64)
            cov_ab = np.empty((n_groups, 6, 6), dtype=np.float64)

        for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            n = end - start
            if n == 0:
                continue

            samples = spherical_values[start:end].copy()

            # Circular mean in degrees for longitude, with wrap-aware covariance.
            lon = samples[:, 1]
            lon_mean = float(
                (
                    np.degrees(
                        np.arctan2(
                            np.mean(np.sin(np.deg2rad(lon))),
                            np.mean(np.cos(np.deg2rad(lon))),
                        )
                    )
                    + 360.0
                )
                % 360.0
            )
            samples[:, 1] = lon_mean + (((lon - lon_mean + 180.0) % 360.0) - 180.0)

            mean = np.average(samples, axis=0)
            mean[1] = lon_mean
            means_sph[i] = mean
            cov_sph[i] = weighted_covariance(
                mean, samples, np.ones(n, dtype=np.float64) / n
            ).reshape(6, 6)

            mags_i = mags[start:end]
            if not np.all(np.isnan(mags_i)):
                predicted_magnitude_v[i] = float(np.nanmean(mags_i))

            if has_aberrated:
                assert np.all(ab_days[start:end] == ab_days[start])
                assert np.all(ab_nanos[start:end] == ab_nanos[start])
                assert np.all(ab_origin_codes[start:end] == ab_origin_codes[start])
                ab_samples = ab_values[start:end]
                ab_mean = np.average(ab_samples, axis=0)
                means_ab[i] = ab_mean
                cov_ab[i] = weighted_covariance(
                    ab_mean, ab_samples, np.ones(n, dtype=np.float64) / n
                ).reshape(6, 6)

        collapsed_coordinates = SphericalCoordinates.from_kwargs(
            rho=means_sph[:, 0],
            lon=means_sph[:, 1],
            lat=means_sph[:, 2],
            vrho=means_sph[:, 3],
            vlon=means_sph[:, 4],
            vlat=means_sph[:, 5],
            covariance=CoordinateCovariances.from_matrix(cov_sph),
            time=out_time,
            origin=out_origin,
            frame=variants.coordinates.frame,
        )

        if not has_aberrated:
            return Ephemeris.from_kwargs(
                orbit_id=orbit_id,
                object_id=object_id,
                coordinates=collapsed_coordinates,
                predicted_magnitude_v=predicted_magnitude_v,
            )

        ab_time = Timestamp.from_kwargs(
            days=pc.take(variants.aberrated_coordinates.time.days, starts.tolist()),
            nanos=pc.take(variants.aberrated_coordinates.time.nanos, starts.tolist()),
            scale=variants.aberrated_coordinates.time.scale,
        )
        ab_origin = Origin.from_kwargs(
            code=pc.take(variants.aberrated_coordinates.origin.code, starts.tolist())
        )
        collapsed_aberrated = CartesianCoordinates.from_kwargs(
            x=means_ab[:, 0],
            y=means_ab[:, 1],
            z=means_ab[:, 2],
            vx=means_ab[:, 3],
            vy=means_ab[:, 4],
            vz=means_ab[:, 5],
            covariance=CoordinateCovariances.from_matrix(cov_ab),
            time=ab_time,
            origin=ab_origin,
            frame=variants.aberrated_coordinates.frame,
        )
        return Ephemeris.from_kwargs(
            orbit_id=orbit_id,
            object_id=object_id,
            coordinates=collapsed_coordinates,
            predicted_magnitude_v=predicted_magnitude_v,
            aberrated_coordinates=collapsed_aberrated,
        )
