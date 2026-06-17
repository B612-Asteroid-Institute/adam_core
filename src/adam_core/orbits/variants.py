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
)


def _sample_covariance(
    mean: np.ndarray,
    cov: np.ndarray,
    method: Literal["auto", "sigma-point", "monte-carlo"],
    num_samples: int,
    alpha: float,
    beta: float,
    kappa: float,
    seed,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sample in a whitened basis: solved-state covariances mix orbital
    # variances (~1e-8) with non-grav variances (~1e-26). At that conditioning
    # both scipy's PSD factorization and sqrtm truncate the small eigenvalues,
    # silently producing zero spread in the non-grav dimensions.
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    scales = np.sqrt(np.diag(cov))
    scales[scales == 0.0] = 1.0
    whitened_cov = cov / np.outer(scales, scales)
    whitened_mean = np.zeros_like(mean)

    if method == "sigma-point":
        samples, weights, weights_cov = sample_covariance_sigma_points(
            whitened_mean, whitened_cov, alpha=alpha, beta=beta, kappa=kappa
        )
    elif method == "monte-carlo":
        samples, weights, weights_cov = sample_covariance_random(
            whitened_mean, whitened_cov, num_samples=num_samples, seed=seed
        )
    elif method == "auto":
        samples, weights, weights_cov = sample_covariance_sigma_points(
            whitened_mean, whitened_cov, alpha=alpha, beta=beta, kappa=kappa
        )
        mean_sg = weighted_mean(samples, weights)
        cov_sg = weighted_covariance(mean_sg, samples, weights_cov)
        # In the whitened basis every variance is ~1, so an absolute tolerance
        # is meaningful for all dimensions.
        if np.any(np.abs(mean_sg - whitened_mean) >= 1e-12) or np.any(
            np.abs(cov_sg - whitened_cov) >= 1e-12
        ):
            samples, weights, weights_cov = sample_covariance_random(
                whitened_mean, whitened_cov, num_samples=num_samples, seed=seed
            )
    else:
        raise ValueError(f"Unknown covariance sampling method: {method}")

    return mean + samples * scales, weights, weights_cov


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
    for sample in samples:
        row = dict(base_row)
        for offset, name in enumerate(parameter_names):
            if name in NON_GRAVITATIONAL_PARAMETER_NAMES:
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

    # Spawn an independent child seed per orbit: reusing the same seed for
    # every orbit would draw identical underlying normals, perfectly
    # correlating the variants across orbits.
    if seed is not None:
        child_seeds = np.random.SeedSequence(seed).spawn(len(orbits))
        orbit_seeds = [np.random.default_rng(child) for child in child_seeds]
    else:
        orbit_seeds = [None] * len(orbits)

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

        orbit_id = orbit.orbit_id[0].as_py()
        object_id = orbit.object_id[0].as_py()

        mean = build_solved_state(
            orbit.coordinates.values[0],
            orbit.non_gravitational_parameters,
            0,
            parameter_names,
        )
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(solved_covariance))):
            raise ValueError(
                f"Solved-state covariance sampling requires finite mean and "
                f"covariance values for orbit {orbit_id}."
            )
        samples, weights, weights_cov = _sample_covariance(
            mean,
            solved_covariance,
            method=method,
            num_samples=num_samples,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            seed=orbit_seeds[orbit_index],
        )
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

    def has_non_gravitational_parameters(self) -> bool:
        """
        Return True if any variant carries a non-zero non-gravitational
        parameter value.
        """
        return self.non_gravitational_parameters.has_values()

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
        for 6-dimensional coordinates. Orbits that carry a solved-state covariance are jointly
        sampled in the full 6+k dimensional solved state (orbital state plus non-gravitational
        parameters), in which case sigma-point sampling draws 2*(6+k)+1 samples per orbit.

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
            Seed for reproducible monte-carlo sampling. Each orbit receives an
            independent child seed spawned from this value.
        include_nongrav : bool, optional
            If True (default), orbits with a solved-state covariance are jointly sampled
            in the full 6+k dimensional state. If False, the non-gravitational parameter
            and solved-state covariance columns are stripped and only the 6x6 coordinate
            covariance is sampled.

        Returns
        -------
        variants_orbits : '~adam_core.orbits.variants.VariantOrbits'
            The variant orbits. When the input mixes orbits with and without solved-state
            covariances, variants for the solved-state orbits are returned first, so row
            order does not necessarily follow the input orbit order.
        """
        if not include_nongrav:
            orbits = orbits.without_non_gravitational_parameters()

        if not orbits.solved_state_covariance.is_all_null():
            solved_mask = pc.is_valid(orbits.solved_state_covariance.dimension)
            if pc.all(solved_mask).as_py():
                return _joint_sample_variants(
                    orbits,
                    method=method,
                    num_samples=num_samples,
                    alpha=alpha,
                    beta=beta,
                    kappa=kappa,
                    seed=seed,
                )
            # Mixed coverage: jointly sample the orbits that have a solved-state
            # covariance and route the rest through the 6D coordinate path.
            joint_variants = _joint_sample_variants(
                orbits.apply_mask(solved_mask),
                method=method,
                num_samples=num_samples,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                seed=seed,
            )
            coordinate_only_variants = cls.create(
                orbits.apply_mask(pc.invert(solved_mask)),
                method=method,
                num_samples=num_samples,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                seed=seed,
                include_nongrav=include_nongrav,
            )
            combined = qv.concatenate([joint_variants, coordinate_only_variants])
            return combined.set_column(
                "variant_id",
                pa.array(np.arange(len(combined)).astype(str), type=pa.large_string()),
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
            solved_parameter_names = (
                orbit.solved_state_covariance.parameter_names_list()
            )
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
                # All variants of an object must share the same solved-state
                # parameter set; take the names from the first non-null row.
                names_list = (
                    object_variants.solved_state_covariance.parameter_names_list()
                )
                non_null_names = [names for names in names_list if names]
                parameter_names = non_null_names[0]
                assert all(names == parameter_names for names in non_null_names)
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
                    np.ones(len(object_variants), dtype=np.float64)
                    / len(object_variants),
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
        # Preserve orbit_id (critical for downstream pipelines that key on orbit_id).
        orbit_id = pc.take(variants.orbit_id, starts.tolist())
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

        # Weights (nullable). If present, use UT weights for mean and weights_cov for covariance.
        w_mean_all = pc.fill_null(variants.weights, np.nan).to_numpy(
            zero_copy_only=False
        )
        w_cov_all = pc.fill_null(variants.weights_cov, np.nan).to_numpy(
            zero_copy_only=False
        )

        counts = np.diff(bounds).astype(np.int64, copy=False)
        uniform_group_size = bool(counts.size > 0 and np.all(counts == counts[0]))
        K = int(counts[0]) if uniform_group_size else -1

        def _normalize_rows(w: np.ndarray) -> np.ndarray:
            s = np.sum(w, axis=1, keepdims=True)
            ok = np.isfinite(s) & (s != 0.0)
            return np.where(ok, w / s, 1.0 / float(w.shape[1]))

        if uniform_group_size and K > 0:
            # Fast path: constant group size (common for sigma points and MC sampling).
            for g0 in range(0, int(n_groups), int(group_chunk_size)):
                g1 = int(min(int(n_groups), int(g0) + int(group_chunk_size)))
                r0 = int(bounds[g0])
                r1 = int(bounds[g1])
                n_g = int(g1 - g0)
                if n_g <= 0:
                    continue

                samp = np.asarray(spherical_values[r0:r1], dtype=np.float64).reshape(
                    n_g, K, 6
                )

                w_mean = np.asarray(w_mean_all[r0:r1], dtype=np.float64).reshape(n_g, K)
                w_cov = np.asarray(w_cov_all[r0:r1], dtype=np.float64).reshape(n_g, K)

                valid_mean = np.all(np.isfinite(w_mean), axis=1)
                valid_cov = np.all(np.isfinite(w_cov), axis=1)
                # Default to uniform weights when missing.
                w_mean = np.where(valid_mean[:, None], w_mean, 1.0 / float(K))
                # If covariance weights missing, fall back to mean weights.
                w_cov = np.where(valid_cov[:, None], w_cov, w_mean)

                w_mean = _normalize_rows(w_mean)
                w_cov = _normalize_rows(w_cov)

                # Circular weighted mean for longitude (degrees).
                lon = samp[:, :, 1]
                lon_rad = np.deg2rad(lon)
                s_sin = np.sum(w_mean * np.sin(lon_rad), axis=1)
                s_cos = np.sum(w_mean * np.cos(lon_rad), axis=1)
                lon_mean = (np.degrees(np.arctan2(s_sin, s_cos)) + 360.0) % 360.0

                # Wrap longitude samples around the circular mean for covariance.
                lon_wrapped = lon_mean[:, None] + (
                    ((lon - lon_mean[:, None] + 180.0) % 360.0) - 180.0
                )
                samp2 = samp.copy()
                samp2[:, :, 1] = lon_wrapped

                mean = np.sum(w_mean[:, :, None] * samp2, axis=1)
                mean[:, 1] = lon_mean
                means_sph[g0:g1] = mean

                resid = samp2 - mean[:, None, :]
                cov = np.einsum("gk,gki,gkj->gij", w_cov, resid, resid, optimize=True)
                cov_sph[g0:g1] = cov

                # Weighted mean of predicted_magnitude_v (ignore NaNs).
                mags_chunk = np.asarray(mags[r0:r1], dtype=np.float64).reshape(n_g, K)
                is_ok = np.isfinite(mags_chunk)
                w_m = np.where(is_ok, w_mean, 0.0)
                denom = np.sum(w_m, axis=1)
                num = np.sum(w_m * np.where(is_ok, mags_chunk, 0.0), axis=1)
                out = np.full_like(num, np.nan, dtype=np.float64)
                np.divide(num, denom, out=out, where=(denom > 0))
                for i, v in enumerate(out.tolist()):
                    if np.isfinite(v):
                        predicted_magnitude_v[g0 + i] = float(v)
        else:
            # Fallback: variable group sizes (rare). Still uses weights when present.
            for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
                n = int(end - start)
                if n <= 0:
                    continue
                samples = np.asarray(
                    spherical_values[start:end], dtype=np.float64
                ).copy()

                w_mean = np.asarray(w_mean_all[start:end], dtype=np.float64)
                w_cov = np.asarray(w_cov_all[start:end], dtype=np.float64)
                if not np.all(np.isfinite(w_mean)):
                    w_mean = np.full(n, 1.0 / float(n), dtype=np.float64)
                else:
                    s = float(np.sum(w_mean))
                    w_mean = (
                        w_mean / s
                        if s != 0.0
                        else np.full(n, 1.0 / float(n), dtype=np.float64)
                    )
                if not np.all(np.isfinite(w_cov)):
                    w_cov = w_mean
                else:
                    s = float(np.sum(w_cov))
                    w_cov = w_cov / s if s != 0.0 else w_mean

                lon = samples[:, 1]
                lon_rad = np.deg2rad(lon)
                lon_mean = float(
                    (
                        np.degrees(
                            np.arctan2(
                                np.sum(w_mean * np.sin(lon_rad)),
                                np.sum(w_mean * np.cos(lon_rad)),
                            )
                        )
                        + 360.0
                    )
                    % 360.0
                )
                samples[:, 1] = lon_mean + (((lon - lon_mean + 180.0) % 360.0) - 180.0)

                mean = np.dot(w_mean, samples)
                mean[1] = lon_mean
                means_sph[i] = mean
                cov_sph[i] = weighted_covariance(mean, samples, w_cov).reshape(6, 6)

                mags_i = mags[start:end]
                if not np.all(np.isnan(mags_i)):
                    ok = np.isfinite(mags_i)
                    denom = float(np.sum(w_mean[ok]))
                    predicted_magnitude_v[i] = (
                        None
                        if denom <= 0
                        else float(np.sum(w_mean[ok] * mags_i[ok]) / denom)
                    )

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
            # regeneration and just collapse those too (weighted).
            try:
                has_ab = not pc.all(
                    pc.is_null(variants.aberrated_coordinates.x)
                ).as_py()
                has_lt = not pc.all(pc.is_null(variants.light_time)).as_py()
            except Exception:
                has_ab = False
                has_lt = False
            if has_ab and has_lt and uniform_group_size and K > 0:
                ab_vals = np.asarray(
                    variants.aberrated_coordinates.values, dtype=np.float64
                )
                lt_vals = pc.fill_null(variants.light_time, np.nan).to_numpy(
                    zero_copy_only=False
                )
                out_ab = np.empty((n_groups, 6), dtype=np.float64)
                out_lt = np.empty((n_groups,), dtype=np.float64)
                for g0 in range(0, int(n_groups), int(group_chunk_size)):
                    g1 = int(min(int(n_groups), int(g0) + int(group_chunk_size)))
                    r0 = int(bounds[g0])
                    r1 = int(bounds[g1])
                    n_g = int(g1 - g0)
                    if n_g <= 0:
                        continue
                    w_mean = np.asarray(w_mean_all[r0:r1], dtype=np.float64).reshape(
                        n_g, K
                    )
                    valid_mean = np.all(np.isfinite(w_mean), axis=1)
                    w_mean = np.where(valid_mean[:, None], w_mean, 1.0 / float(K))
                    w_mean = _normalize_rows(w_mean)
                    ab = np.asarray(ab_vals[r0:r1], dtype=np.float64).reshape(n_g, K, 6)
                    lt = np.asarray(lt_vals[r0:r1], dtype=np.float64).reshape(n_g, K)
                    # Weighted mean; ignore non-finite lt.
                    out_ab[g0:g1] = np.sum(w_mean[:, :, None] * ab, axis=1)
                    ok = np.isfinite(lt)
                    w2 = np.where(ok, w_mean, 0.0)
                    denom = np.sum(w2, axis=1)
                    num = np.sum(w2 * np.where(ok, lt, 0.0), axis=1)
                    out_lt[g0:g1] = np.where(denom > 0, num / denom, np.nan)

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
        n_base = int(n_orbits * n_variants)  # base variants (orbit, variant_id)

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

        # UT weights per (orbit, variant) (constant across times).
        w_mean_base = (
            pc.take(self.weights, base0.tolist())
            .cast(pa.float64())
            .fill_null(pa.scalar(np.nan, type=pa.float64()))
            .to_numpy(zero_copy_only=False)
            .astype(np.float64, copy=False)
            .reshape(n_orbits, n_variants)
        )
        w_cov_base = (
            pc.take(self.weights_cov, base0.tolist())
            .cast(pa.float64())
            .fill_null(pa.scalar(np.nan, type=pa.float64()))
            .to_numpy(zero_copy_only=False)
            .astype(np.float64, copy=False)
            .reshape(n_orbits, n_variants)
        )

        def _normalize_rows(w: np.ndarray) -> np.ndarray:
            s = np.sum(w, axis=1, keepdims=True)
            ok = np.isfinite(s) & (s != 0.0)
            return np.where(ok, w / s, 1.0 / float(w.shape[1]))

        valid_mean = np.all(np.isfinite(w_mean_base), axis=1)
        w_mean = np.where(valid_mean[:, None], w_mean_base, 1.0 / float(n_variants))
        valid_cov = np.all(np.isfinite(w_cov_base), axis=1)
        w_cov = np.where(valid_cov[:, None], w_cov_base, w_mean)
        w_mean = _normalize_rows(w_mean)
        w_cov = _normalize_rows(w_cov)

        # Values: reshape to (orbit, variant, time, dim).
        vals = np.asarray(self.coordinates.values, dtype=np.float64).reshape(
            n_base, n_times, 6
        )
        vals = vals.reshape(n_orbits, n_variants, n_times, 6)

        # Circular weighted mean for longitude (deg).
        lon = vals[:, :, :, 1]  # (O,K,N)
        lon_rad = np.deg2rad(lon)
        w_mean3 = w_mean[:, :, None]  # (O,K,1)
        s_sin = np.sum(w_mean3 * np.sin(lon_rad), axis=1)  # (O,N)
        s_cos = np.sum(w_mean3 * np.cos(lon_rad), axis=1)
        lon_mean = (np.degrees(np.arctan2(s_sin, s_cos)) + 360.0) % 360.0  # (O,N)

        lon_wrapped = lon_mean[:, None, :] + (
            ((lon - lon_mean[:, None, :] + 180.0) % 360.0) - 180.0
        )
        vals2 = vals.copy()
        vals2[:, :, :, 1] = lon_wrapped

        mean = np.sum(w_mean[:, :, None, None] * vals2, axis=1)  # (O,N,6)
        mean[:, :, 1] = lon_mean

        resid = vals2 - mean[:, None, :, :]  # (O,K,N,6)
        cov = np.einsum(
            "ok,okni,oknj->onij", w_cov, resid, resid, optimize=True
        )  # (O,N,6,6)

        mean_flat = mean.reshape(n_orbits * n_times, 6)
        cov_flat = cov.reshape(n_orbits * n_times, 6, 6)

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
