import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

# Non-gravitational parameter values supported for ingestion and storage,
# in the order of the trailing dimensions of an extended (9x9) coordinate
# covariance matrix. See `~adam_core.coordinates.covariances.CoordinateCovariances`.
NON_GRAVITATIONAL_VALUE_FIELDS = (
    "A1",
    "A2",
    "A3",
)

# Marsden g(r) model constants. These are fixed per solution -- sources report
# them without uncertainties and they never appear in solution covariances --
# so they are stored as plain values and are NOT dimensions of the extended
# coordinate covariance. They select the force law under which the A1/A2/A3
# accelerations were fit:
#
#     g(r) = ALN * (r/R0)^-NM * (1 + (r/R0)^NN)^-NK
#
# Null constants mean the standard asteroid convention g(r) = (1 au / r)^2
# (ALN=1, NK=0, NM=2, NN=5.093, R0=1 -- the defaults ASSIST applies).
MARSDEN_CONSTANT_FIELDS = (
    "ALN",
    "NK",
    "NM",
    "NN",
    "R0",
)

# The classic Marsden, Sekanina & Yeomans (1973) water-ice sublimation
# constants -- the convention comet solutions are fit under. JPL/SBDB lists
# g(r) constants on a solution only when they differ from this standard set,
# so absent constants on a solution with fitted A parameters mean these values.
MARSDEN_STANDARD_CONSTANTS = {
    "ALN": 0.1112620426,
    "NK": 4.6142,
    "NM": 2.15,
    "NN": 5.093,
    "R0": 2.808,
}

# The asteroid (Yarkovsky-style) convention: g(r) degrades to (1 au / r)^2.
ASTEROID_MARSDEN_CONSTANTS = {
    "ALN": 1.0,
    "NK": 0.0,
    "NM": 2.0,
    "NN": 5.093,
    "R0": 1.0,
}


class NonGravitationalParameters(qv.Table):
    """
    Non-gravitational parameters attached to an orbit solution.

    Only the Marsden-style radial/transverse/normal accelerations (A1, A2, A3)
    are supported as fitted parameters, stored in au / d^2. Their
    uncertainties and cross covariances with the orbital state live in the
    orbit's coordinate covariance, which is extended to 9x9 for orbits with a
    non-gravitational solution (see
    `~adam_core.coordinates.covariances.CoordinateCovariances`).

    The Marsden g(r) constants (ALN, NK, NM, NN, R0; R0 in au, the rest
    dimensionless) are fixed model constants, not fitted parameters: they
    carry no uncertainty and are not covariance dimensions, but they select
    the force law under which A1/A2/A3 were fit and must be fed to the
    propagator. Null constants mean the standard asteroid convention
    g(r) = (1 au / r)^2.
    """

    source = qv.LargeStringColumn(nullable=True)

    A1 = qv.Float64Column(nullable=True)
    A2 = qv.Float64Column(nullable=True)
    A3 = qv.Float64Column(nullable=True)

    ALN = qv.Float64Column(nullable=True)
    NK = qv.Float64Column(nullable=True)
    NM = qv.Float64Column(nullable=True)
    NN = qv.Float64Column(nullable=True)
    R0 = qv.Float64Column(nullable=True)

    def has_values(self) -> bool:
        """
        Return True if any row has a non-zero non-gravitational acceleration
        value (A1, A2, A3).

        Parameters that are explicitly solved to zero are treated as absent:
        they exert no force, so a gravity-only propagation of such an orbit is
        still exact. The g(r) constants are not considered: without a non-zero
        acceleration they select a force law that is never applied.
        """
        if len(self) == 0:
            return False
        for field in NON_GRAVITATIONAL_VALUE_FIELDS:
            values = pc.drop_null(getattr(self, field))
            if len(values) > 0 and pc.any(pc.not_equal(values, 0.0)).as_py():
                return True
        return False

    def to_array(self) -> npt.NDArray[np.float64]:
        """
        Return the acceleration values as an (N, 3) array in (A1, A2, A3)
        order, with nulls replaced by 0.0 (no force).
        """
        columns = [
            pc.fill_null(getattr(self, field), 0.0).to_numpy(zero_copy_only=False)
            for field in NON_GRAVITATIONAL_VALUE_FIELDS
        ]
        return np.stack(columns, axis=1)

    def marsden_constants_array(self) -> npt.NDArray[np.float64]:
        """
        Return the g(r) constants as an (N, 5) array in (ALN, NK, NM, NN, R0)
        order, with nulls replaced by the standard asteroid convention
        (g(r) = (1 au / r)^2).
        """
        columns = [
            pc.fill_null(
                getattr(self, field), ASTEROID_MARSDEN_CONSTANTS[field]
            ).to_numpy(zero_copy_only=False)
            for field in MARSDEN_CONSTANT_FIELDS
        ]
        return np.stack(columns, axis=1)

    @classmethod
    def nulls(
        cls, length: int, **kwargs: int | float | str
    ) -> "NonGravitationalParameters":
        null_float = [None] * length
        null_str = [None] * length
        return cls.from_kwargs(
            source=null_str,
            A1=null_float,
            A2=null_float,
            A3=null_float,
            ALN=null_float,
            NK=null_float,
            NM=null_float,
            NN=null_float,
            R0=null_float,
            **kwargs,
        )
