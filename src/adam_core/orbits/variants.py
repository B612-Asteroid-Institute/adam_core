import uuid
from typing import Literal, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..constants import Constants as c
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import (
    CoordinateCovariances,
    sample_covariance_random,
    sample_covariance_sigma_points,
    weighted_covariance,
    weighted_mean,
)
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import transform_coordinates
from ..coordinates.variants import VariantCoordinatesTable, create_coordinate_variants
from ..dynamics.aberrations import add_light_time
from ..observers.observers import Observers
from ..time import Timestamp
from ..utils.chunking import process_in_chunks
from .ephemeris import Ephemeris
from .non_gravitational_parameters import NonGravitationalParameters
from .orbits import Orbits
from .physical_parameters import PhysicalParameters
from .solved_state_covariances import (
    NON_GRAVITATIONAL_PARAMETER_NAMES,
    ORBITAL_PARAMETER_NAMES,
    SolvedStateCovariances,
    build_solved_state,
    parse_parameter_names,
)


def _sample_covariance(
    mean: np.ndarray,
    cov: np.ndarray,
    method: Literal["auto", "sigma-point", "monte-carlo"],
    num_samples: int,
    alpha: float,
    beta: float,
    kappa: float,
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "sigma-point":
        return sample_covariance_sigma_points(
            mean, cov, alpha=alpha, beta=beta, kappa=kappa
        )
    if method == "monte-carlo":
        return sample_covariance_random(mean, cov, num_samples=num_samples, seed=seed)
    if method == "auto":
        samples, weights, weights_cov = sample_covariance_sigma_points(
            mean, cov, alpha=alpha, beta=beta, kappa=kappa
        )
        mean_sg = weighted_mean(samples, weights)
        cov_sg = weighted_covariance(mean_sg, samples, weights_cov)
        if np.any(np.abs(mean_sg - mean) >= 1e-12) or np.any(
            np.abs(cov_sg - cov) >= 1e-12
        ):
            return sample_covariance_random(mean, cov, num_samples=num_samples, seed=seed)
        return samples, weights, weights_cov
    raise ValueError(f"Unknown covariance sampling method: {method}")


def _nongrav_columns_dict() -> dict[str, list[object]]:
    columns = {}
    for name in NonGravitationalParameters.schema.names:
        columns[name] = []
    return columns


def _append_nongrav_variant_rows(
    columns: dict[str, list[object]],
    base: NonGravitationalParameters,
    row_index: int,
    parameter_names: list[str],
    samples: np.ndarray,
) -> None:
    base_row = {
        name: getattr(base, name)[row_index].as_py()
        for name in NonGravitationalParameters.schema.names
    }
    nongrav_names = [name for name in parameter_names if name in NON_GRAVITATIONAL_PARAMETER_NAMES]
    for sample in samples:
        row = dict(base_row)
        for offset, name in enumerate(parameter_names):
            if name in nongrav_names:
                row[name] = float(sample[offset])
        for name in columns:
            columns[name].append(row.get(name))


def _apply_solved_state_mean_to_nongrav_row(
    base: NonGravitationalParameters,
    row_index: int,
    parameter_names: list[str],
    mean_state: np.ndarray,
) -> NonGravitationalParameters:
    row = {
        name: getattr(base, name)[row_index].as_py()
        for name in NonGravitationalParameters.schema.names
    }
    for name, value in zip(parameter_names[6:], mean_state[6:]):
        if name in NON_GRAVITATIONAL_PARAMETER_NAMES:
            row[name] = float(value)
    return NonGravitationalParameters.from_kwargs(
        **{name: [row.get(name)] for name in NonGravitationalParameters.schema.names}
    )


def _joint_sample_variants(
    orbits: Orbits,
    method: Literal["auto", "sigma-point", "monte-carlo"],
    num_samples: int,
    alpha: float,
    beta: float,
    kappa: float,
    seed: Optional[int],
) -> "VariantOrbits":
    solved_covariances = orbits.solved_state_covariance.to_matrix()
    solved_names = orbits.solved_state_covariance.parameter_names_list()

    orbit_id_rows: list[str] = []
    object_id_rows: list[object] = []
    variant_id_rows: list[str] = []
    weight_rows: list[float] = []
    weight_cov_rows: list[float] = []
    coordinate_rows: list[np.ndarray] = []
    solved_covariance_rows: list[np.ndarray | None] = []
    solved_parameter_name_rows: list[list[str] | None] = []
    physical_index: list[int] = []
    nongrav_columns = _nongrav_columns_dict()

    for orbit_index, orbit in enumerate(orbits):
        solved_covariance = solved_covariances[orbit_index]
        parameter_names = solved_names[orbit_index]

        if solved_covariance is None or not parameter_names:
            raise ValueError(
                "Solved-state covariance sampling requires a covariance matrix and parameter names for every orbit."
            )
        if tuple(parameter_names[:6]) != ORBITAL_PARAMETER_NAMES:
            raise ValueError(
                "Solved-state covariance parameter ordering must begin with x,y,z,vx,vy,vz."
            )

        mean = build_solved_state(
            orbit.coordinates.values[0],
            orbit.non_gravitational_parameters,
            0,
            parameter_names,
        )
        samples, weights, weights_cov = _sample_covariance(
            mean,
            solved_covariance,
            method=method,
            num_samples=num_samples,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            seed=seed,
        )
        orbit_id = orbit.orbit_id[0].as_py()
        object_id = orbit.object_id[0].as_py()
        for sample_index in range(len(samples)):
            orbit_id_rows.append(orbit_id)
            object_id_rows.append(object_id)
            variant_id_rows.append(str(len(variant_id_rows)))
            weight_rows.append(float(weights[sample_index]))
            weight_cov_rows.append(float(weights_cov[sample_index]))
            coordinate_rows.append(samples[sample_index, :6])
            solved_covariance_rows.append(solved_covariance)
            solved_parameter_name_rows.append(parameter_names)
            physical_index.append(orbit_index)

        _append_nongrav_variant_rows(
            nongrav_columns,
            orbits.non_gravitational_parameters,
            orbit_index,
            parameter_names,
            samples,
        )

    coordinate_array = np.asarray(coordinate_rows, dtype=np.float64)
    repeated_indices = np.asarray(physical_index, dtype=np.int64)
    repeated_time = orbits.coordinates.time.take(repeated_indices)
    repeated_origin = orbits.coordinates.origin.take(repeated_indices)

    return VariantOrbits.from_kwargs(
        orbit_id=np.array(orbit_id_rows, dtype="object"),
        object_id=np.array(object_id_rows, dtype="object"),
        variant_id=np.array(variant_id_rows, dtype="object"),
        weights=np.asarray(weight_rows, dtype=np.float64),
        weights_cov=np.asarray(weight_cov_rows, dtype=np.float64),
        coordinates=CartesianCoordinates.from_kwargs(
            x=coordinate_array[:, 0],
            y=coordinate_array[:, 1],
            z=coordinate_array[:, 2],
            vx=coordinate_array[:, 3],
            vy=coordinate_array[:, 4],
            vz=coordinate_array[:, 5],
            time=repeated_time,
            origin=repeated_origin,
            frame=orbits.coordinates.frame,
        ),
        physical_parameters=orbits.physical_parameters.take(repeated_indices),
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(
            **nongrav_columns
        ),
        solved_state_covariance=SolvedStateCovariances.from_matrix(
            solved_covariance_rows, solved_parameter_name_rows
        ),
    )


class VariantOrbits(qv.Table):
    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    variant_id = qv.LargeStringColumn(nullable=True)
    weights = qv.Float64Column(nullable=True)
    weights_cov = qv.Float64Column(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    physical_parameters = PhysicalParameters.as_column(nullable=True)
    non_gravitational_parameters = NonGravitationalParameters.as_column(nullable=True)
    solved_state_covariance = SolvedStateCovariances.as_column(nullable=True)

    def without_non_gravitational_parameters(self) -> "VariantOrbits":
        return self.set_column(
            "non_gravitational_parameters",
            NonGravitationalParameters.nulls(len(self)),
        ).set_column(
            "solved_state_covariance",
            SolvedStateCovariances.nulls(len(self)),
        )

    def with_non_gravitational_parameters(
        self, enabled: bool = True
    ) -> "VariantOrbits":
        if enabled:
            return self
        return self.without_non_gravitational_parameters()

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
        include_nongrav: bool = True,
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
        if not include_nongrav:
            orbits = orbits.without_non_gravitational_parameters()

        if not orbits.solved_state_covariance.is_all_null():
            return _joint_sample_variants(
                orbits,
                method=method,
                num_samples=num_samples,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                seed=seed,
            )

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
            non_gravitational_parameters=orbits.non_gravitational_parameters.take(
                variant_coordinates.index
            ),
            solved_state_covariance=orbits.solved_state_covariance.take(
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
            solved_covariances = orbit.solved_state_covariance.to_matrix()
            solved_parameter_names = orbit.solved_state_covariance.parameter_names_list()
            if solved_covariances[0] is not None and solved_parameter_names[0]:
                parameter_names = solved_parameter_names[0]
                samples_full = []
                for i in range(len(variants)):
                    samples_full.append(
                        build_solved_state(
                            variants.coordinates.values[i],
                            variants.non_gravitational_parameters,
                            i,
                            parameter_names,
                        )
                    )
                covariance_full = weighted_covariance(
                    build_solved_state(
                        mean,
                        orbit.non_gravitational_parameters,
                        0,
                        parameter_names,
                    ),
                    np.asarray(samples_full, dtype=np.float64),
                    variants.weights_cov.to_numpy(zero_copy_only=False),
                )
                orbit_collapsed = orbit_collapsed.set_column(
                    "solved_state_covariance",
                    SolvedStateCovariances.from_matrix(
                        [covariance_full], [parameter_names]
                    ),
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
            if not object_variants.solved_state_covariance.is_all_null():
                parameter_names = object_variants.solved_state_covariance.parameter_names_list()[0]
                full_samples = np.asarray(
                    [
                        build_solved_state(
                            object_variants.coordinates.values[i],
                            object_variants.non_gravitational_parameters,
                            i,
                            parameter_names,
                        )
                        for i in range(len(object_variants))
                    ],
                    dtype=np.float64,
                )
                mean_full = np.average(full_samples, axis=0)
                covariance_full = weighted_covariance(
                    mean_full,
                    full_samples,
                    np.ones(len(object_variants), dtype=np.float64) / len(object_variants),
                )
                orbit = orbit.set_column(
                    "non_gravitational_parameters",
                    _apply_solved_state_mean_to_nongrav_row(
                        object_variants.non_gravitational_parameters,
                        0,
                        parameter_names,
                        mean_full,
                    ),
                ).set_column(
                    "solved_state_covariance",
                    SolvedStateCovariances.from_matrix(
                        [covariance_full], [parameter_names]
                    ),
                )
            else:
                orbit = orbit.set_column(
                    "non_gravitational_parameters",
                    object_variants.non_gravitational_parameters.take([0]),
                ).set_column(
                    "solved_state_covariance",
                    object_variants.solved_state_covariance.take([0]),
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

        # Drop any existing aberrated coordinates and regenerate from the collapsed topocentric state.
        ephemeris = Ephemeris.from_kwargs(
            orbit_id=orbit_id,
            object_id=object_id,
            coordinates=collapsed_coordinates,
            predicted_magnitude_v=predicted_magnitude_v,
        )

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
        chunk_size = 200
        aberrated_vals: np.ndarray = np.empty((0, 6), dtype=np.float64)
        light_time: np.ndarray = np.empty((0,), dtype=np.float64)
        for barycentric_chunk, times_chunk, observers_chunk in zip(
            process_in_chunks(barycentric_vals, chunk_size),
            process_in_chunks(times_tdb_mjd, chunk_size),
            process_in_chunks(observer_positions, chunk_size),
        ):
            aberrated_chunk, light_time_chunk = add_light_time(
                barycentric_chunk,
                times_chunk,
                observers_chunk,
                lt_tol=1e-12,
                mu=c.MU,
                max_iter=100,
                tol=1e-15,
            )
            aberrated_vals = np.concatenate(
                (aberrated_vals, np.array(aberrated_chunk)), axis=0
            )
            light_time = np.concatenate((light_time, np.array(light_time_chunk)))

        aberrated_vals = aberrated_vals[: len(ephemeris)]
        light_time = light_time[: len(ephemeris)]

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
