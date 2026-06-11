import pyarrow.compute as pc
import quivr as qv

# Fields that hold non-gravitational parameter values (as opposed to
# sigmas or solution metadata).
NON_GRAVITATIONAL_VALUE_FIELDS = (
    "A1",
    "A2",
    "A3",
    "DT",
    "R0",
    "ALN",
    "NK",
    "NM",
    "NN",
    "AMRAT",
    "RHO",
)


class NonGravitationalParameters(qv.Table):
    """
    Canonical non-gravitational parameters attached to an orbit solution.

    Notes
    -----
    Values are stored in canonical units where known:
    - `A1`, `A2`, `A3`: au / d^2
    - `DT`: d
    - `R0`: au
    - `AMRAT`: m^2 / kg
    - `RHO`: kg / m^3

    Larger solved states and cross-covariances with these parameters are tracked via
    metadata fields, but coordinate covariance propagation remains 6D for now.
    """

    source = qv.LargeStringColumn(nullable=True)
    model = qv.LargeStringColumn(nullable=True)
    solution_dimension = qv.Int64Column(nullable=True)
    parameter_count = qv.Int64Column(nullable=True)
    estimated_parameter_names = qv.LargeStringColumn(nullable=True)

    A1 = qv.Float64Column(nullable=True)
    A1_sigma = qv.Float64Column(nullable=True)
    A2 = qv.Float64Column(nullable=True)
    A2_sigma = qv.Float64Column(nullable=True)
    A3 = qv.Float64Column(nullable=True)
    A3_sigma = qv.Float64Column(nullable=True)

    DT = qv.Float64Column(nullable=True)
    DT_sigma = qv.Float64Column(nullable=True)
    R0 = qv.Float64Column(nullable=True)
    R0_sigma = qv.Float64Column(nullable=True)
    ALN = qv.Float64Column(nullable=True)
    ALN_sigma = qv.Float64Column(nullable=True)
    NK = qv.Float64Column(nullable=True)
    NK_sigma = qv.Float64Column(nullable=True)
    NM = qv.Float64Column(nullable=True)
    NM_sigma = qv.Float64Column(nullable=True)
    NN = qv.Float64Column(nullable=True)
    NN_sigma = qv.Float64Column(nullable=True)

    AMRAT = qv.Float64Column(nullable=True)
    AMRAT_sigma = qv.Float64Column(nullable=True)
    RHO = qv.Float64Column(nullable=True)
    RHO_sigma = qv.Float64Column(nullable=True)

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

    @classmethod
    def nulls(cls, length: int) -> "NonGravitationalParameters":
        null_float = [None] * length
        null_int = [None] * length
        null_str = [None] * length
        return cls.from_kwargs(
            source=null_str,
            model=null_str,
            solution_dimension=null_int,
            parameter_count=null_int,
            estimated_parameter_names=null_str,
            A1=null_float,
            A1_sigma=null_float,
            A2=null_float,
            A2_sigma=null_float,
            A3=null_float,
            A3_sigma=null_float,
            DT=null_float,
            DT_sigma=null_float,
            R0=null_float,
            R0_sigma=null_float,
            ALN=null_float,
            ALN_sigma=null_float,
            NK=null_float,
            NK_sigma=null_float,
            NM=null_float,
            NM_sigma=null_float,
            NN=null_float,
            NN_sigma=null_float,
            AMRAT=null_float,
            AMRAT_sigma=null_float,
            RHO=null_float,
            RHO_sigma=null_float,
        )
