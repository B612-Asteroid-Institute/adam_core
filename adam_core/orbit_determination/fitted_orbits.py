import uuid

import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.residuals import Residuals
from ..orbits.orbits import Orbits


class FittedOrbits(qv.Table):

    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()
    chi2 = qv.Float64Column()
    reduced_chi2 = qv.Float64Column()
    iterations = qv.Int64Column(nullable=True)
    success = qv.BooleanColumn(nullable=True)
    status_code = qv.Int64Column(nullable=True)

    def to_orbits(self) -> Orbits:
        """
        Convert fitted orbits to orbits that can be used by
        a Propagator.

        Returns
        -------
        orbits : `~adam_core.orbits.Orbits`
            Orbits.
        """
        return Orbits.from_kwargs(
            orbit_id=self.orbit_id,
            object_id=self.object_id,
            coordinates=self.coordinates,
        )


class FittedOrbitMembers(qv.Table):

    orbit_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()
    residuals = Residuals.as_column(nullable=True)
    solution = qv.BooleanColumn(nullable=True)
    outlier = qv.BooleanColumn(nullable=True)
