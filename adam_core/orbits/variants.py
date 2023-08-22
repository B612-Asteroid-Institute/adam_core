import uuid
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances, weighted_covariance
from ..coordinates.variants import create_coordinate_variants
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

        Returns
        -------
        variants_orbits : '~adam_core.orbits.variants.VariantOrbits'
            The variant orbits.
        """
        idx, W, W_cov, variant_coordinates = create_coordinate_variants(
            orbits.coordinates, method=method
        )
        return cls.from_kwargs(
            orbit_id=pc.take(orbits.orbit_id, idx),
            object_id=pc.take(orbits.object_id, idx),
            weights=W,
            weights_cov=W_cov,
            coordinates=variant_coordinates,
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
        for key, orbit, variants in link.iterate():
            key = key.as_py()

            assert len(orbit) == 1

            samples = variants.coordinates.values
            mean = orbit.coordinates.values[0]
            covariance = weighted_covariance(
                mean, samples, variants.weights_cov.to_numpy()
            )

            orbit_collapsed = orbit.set_column(
                "coordinates.covariance", CoordinateCovariances.from_matrix(covariance)
            )

            orbits_list.append(orbit_collapsed)

        orbits_collapsed = qv.concatenate(orbits_list)

        # Array of indices into the collapsed orbits
        orbits_idx = pa.array(np.arange(0, len(orbits_collapsed)))

        # Make a list of arrays that will be used to sort the orbits
        orbits_idx_sorted_list = []

        # Loop over input orbits and figure out where in the collapsed orbits they occur
        # There has to be an easier or better way to do this?
        for orbit in orbits:
            mask_orbit_id = pc.equal(orbits_collapsed.orbit_id, orbit.orbit_id[0])
            mask_jd1 = pc.equal(
                orbits_collapsed.coordinates.time.jd1, orbit.coordinates.time.jd1[0]
            )
            mask_jd2 = pc.equal(
                orbits_collapsed.coordinates.time.jd2, orbit.coordinates.time.jd2[0]
            )
            mask = pc.and_(mask_orbit_id, pc.and_(mask_jd1, mask_jd2))
            orbits_idx_sorted_list.append(orbits_idx.filter(mask))

        orbits_idx_sorted = pa.concat_arrays(orbits_idx_sorted_list)
        return orbits_collapsed.take(orbits_idx_sorted)
