import uuid
from typing import Literal

import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances, weighted_covariance
from ..coordinates.variants import VariantCoordinatesTable, create_coordinate_variants
from .orbits import Orbits


class VariantOrbits(qv.Table):

    orbit_id = qv.StringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.StringColumn(nullable=True)
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
        )
        return cls.from_kwargs(
            orbit_id=pc.take(orbits.orbit_id, variant_coordinates.index),
            object_id=pc.take(orbits.object_id, variant_coordinates.index),
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
                "jd1": orbits.coordinates.time.jd1,
                "jd2": orbits.coordinates.time.jd2,
            },
            right_keys={
                "orbit_id": self.orbit_id,
                "jd1": self.coordinates.time.jd1,
                "jd2": self.coordinates.time.jd2,
            },
        )

    def collapse(self, orbits: Orbits) -> Orbits:
        """
        Collapse the variants and recalculate the covariance matrix for each
        each orbit at each epoch. The mean state is taken from the orbits class and
        is not calculate from the variants.

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
                jd1=orbit.coordinates.time.jd1[0].as_py(),
                jd2=orbit.coordinates.time.jd2[0].as_py(),
            )
            variants = link.select_right(key)

            samples = variants.coordinates.values
            mean = orbit.coordinates.values[0]
            covariance = weighted_covariance(
                mean, samples, variants.weights_cov.to_numpy()
            )

            orbit_collapsed = orbit.set_column(
                "coordinates.covariance", CoordinateCovariances.from_matrix(covariance)
            )

            orbits_list.append(orbit_collapsed)

        return qv.concatenate(orbits_list)
