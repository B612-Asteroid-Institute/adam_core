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


class NonGravitationalParameters(qv.Table):
    """
    Non-gravitational parameters attached to an orbit solution.

    Only the Marsden-style radial/transverse/normal accelerations (A1, A2, A3)
    are supported, stored in au / d^2. Their uncertainties and cross
    covariances with the orbital state live in the orbit's coordinate
    covariance, which is extended to 9x9 for orbits with a non-gravitational
    solution (see `~adam_core.coordinates.covariances.CoordinateCovariances`).
    """

    source = qv.LargeStringColumn(nullable=True)

    A1 = qv.Float64Column(nullable=True)
    A2 = qv.Float64Column(nullable=True)
    A3 = qv.Float64Column(nullable=True)

    def has_values(self) -> bool:
        """
        Return True if any row has a non-zero non-gravitational parameter value.

        Parameters that are explicitly solved to zero are treated as absent:
        they exert no force, so a gravity-only propagation of such an orbit is
        still exact.
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
        Return the parameter values as an (N, 3) array in (A1, A2, A3) order,
        with nulls replaced by 0.0 (no force).
        """
        columns = [
            pc.fill_null(getattr(self, field), 0.0).to_numpy(zero_copy_only=False)
            for field in NON_GRAVITATIONAL_VALUE_FIELDS
        ]
        return np.stack(columns, axis=1)

    @classmethod
    def nulls(cls, length: int) -> "NonGravitationalParameters":
        null_float = [None] * length
        null_str = [None] * length
        return cls.from_kwargs(
            source=null_str,
            A1=null_float,
            A2=null_float,
            A3=null_float,
        )
