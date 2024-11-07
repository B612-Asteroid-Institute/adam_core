import uuid
from typing import Literal, Optional

import numpy as np
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances, weighted_covariance
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.variants import VariantCoordinatesTable, create_coordinate_variants
from .ephemeris import Ephemeris
from .orbits import Orbits


class VariantOrbits(qv.Table):
    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    variant_id = qv.LargeStringColumn(nullable=True)
    weights = qv.Float64Column(nullable=True)
    weights_cov = qv.Float64Column(nullable=True)
    coordinates = CartesianCoordinates.as_column()

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
                coordinates=CartesianCoordinates.from_kwargs(
                    x=[mean[0]],
                    y=[mean[1]],
                    z=[mean[2]],
                    vx=[mean[3]],
                    vy=[mean[4]],
                    vz=[mean[5]],
                    covariance=CoordinateCovariances.from_matrix(covariance),
                    time=object_variants.coordinates.time[0],
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

            samples = variants.coordinates.values
            mean = ephemeris_i.coordinates.values[0]
            covariance = weighted_covariance(
                mean, samples, variants.weights_cov.to_numpy(zero_copy_only=False)
            ).reshape(1, 6, 6)

            ephemeris_collapsed = ephemeris_i.set_column(
                "coordinates.covariance", CoordinateCovariances.from_matrix(covariance)
            )

            ephemeris_list.append(ephemeris_collapsed)

        return qv.concatenate(ephemeris_list)
