import json
import logging
from typing import Any, Dict, List, Literal, Union

import numpy as np
import numpy.typing as npt

from ...coordinates.covariances import COORD_DIM, FULL_DIM
from ...orbits import Orbits
from ...utils.http import _raise_compatible_http_error
from ..non_gravitational_parameters import (
    NON_GRAVITATIONAL_VALUE_FIELDS,
    NonGravitationalParameters,
)
from ..physical_parameters import PhysicalParameters

logger = logging.getLogger(__name__)

_NEOCC_UNIT_FACTORS = {
    "AMRAT": 1e-3,
    "A2": 1e-10,
}
_NEOCC_SUPPORTED_SOLVE_FOR = frozenset(_NEOCC_UNIT_FACTORS)


def _parse_oef(data: str) -> Dict[str, Any]:
    """
    Parse a OEF file and return the stored orbital elements.

    Parameters
    ----------
    data: str
        The content of the OEF file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the parsed orbital elements and metadata.


    Examples
    --------
    format  = 'OEF2.0'       ! file format
    rectype = 'ML'           ! record type (1L/ML)
    refsys  = ECLM J2000     ! default reference system
    END_OF_HEADER
    2024YR4
    ! Keplerian elements: a, e, i, long. node, arg. peric., mean anomaly
     KEP   2.5158127507489616E+00  6.6154036821914619E-01   3.4081393687180 271.3655954496424 134.3614240204325  4.0403920526717883E+01
     MJD     60800.000000000 TDT
     MAG  23.876  0.150
    ! Non-gravitational parameters: model used, number of model parameters, dimension
     LSP   0  0    6
    ! PERIHELION   8.5150105724807035E-01
    ! APHELION   4.1801244442498522E+00
    ! ANODE   1.6132920678553648E+00
    ! DNODE  -1.6139338737144644E-02
    ! MOID   2.8281976977061222E-03
    ! PERIOD   1.4575246142278593E+03
    ! PHA F
    ! VINFTY    14.2161102117779
    ! U_PAR 5.5
    ! ORB_TYPE Apollo
    ! RMS    1.45945E-04   2.08511E-05   7.80533E-05   1.08159E-05   9.07220E-05   3.57225E-03
     COV   2.129990103626278E-08  3.043103695236090E-09  1.138994073085263E-08
     COV  -1.297300567885438E-09 -1.321094812357632E-08 -5.213516734886736E-07
     COV   4.347664336862408E-10  1.627274457493249E-09 -1.853526029216412E-10
     COV  -1.887473042074563E-09 -7.448518590163292E-08  6.092321667952164E-09
     COV  -6.879908552449990E-10 -7.069797588501348E-09 -2.787885242390572E-07
     COV   1.169847013231039E-10  7.781995171805923E-10  3.175091792990313E-08
     COV   8.230474520730192E-09  3.233601008844550E-07  1.276097850815426E-05
     COR   1.000000000000000E+00  9.999998967955239E-01  9.998647520507938E-01
     COR  -8.218400094356861E-01 -9.977753189147101E-01 -9.999999781092901E-01
     COR   9.999999999999999E-01  9.998650578225570E-01 -8.218757198089662E-01
     COR  -9.977926872410703E-01 -9.999998019539900E-01  9.999999999999998E-01
     COR  -8.149420314930980E-01 -9.983966836512449E-01 -9.998652884032685E-01
     COR   1.000000000000000E+00  7.930744967922404E-01  8.217690139449345E-01
     COR   1.000000000000000E+00  9.977735927640640E-01  1.000000000000000E+00

    """
    from adam_core import _rust_native as _rn

    result = json.loads(_rn.query_neocc_parse_oef(data))
    solved_dimension = int((result.get("nongrav") or {}).get("dimension", COORD_DIM))
    for matrix_name in ("covariance", "correlation"):
        if matrix_name in result:
            result[matrix_name] = np.asarray(
                result[matrix_name], dtype=np.float64
            ).reshape(COORD_DIM, COORD_DIM)
        full_name = f"{matrix_name}_full"
        if full_name in result:
            result[full_name] = np.asarray(result[full_name], dtype=np.float64).reshape(
                solved_dimension, solved_dimension
            )
    return result


def _physical_parameters_from_neocc(data: Dict[str, Any]) -> PhysicalParameters:
    """
    Build one-row PhysicalParameters from parsed NEOCC OEF data.

    OEF MAG line is (H, G); no uncertainties in NEOCC OEF. V-band per ESA doc.
    Ref: https://neo.ssa.esa.int/computer-access
    """
    mag = data.get("magnitude")
    if mag is not None and "H" in mag:
        h = float(mag["H"])
        g = float(mag["G"]) if mag.get("G") is not None else np.nan
        return PhysicalParameters.from_kwargs(
            H_v=[h],
            H_v_sigma=[np.nan],
            G=[g],
            G_sigma=[np.nan],
        )
    return PhysicalParameters.from_kwargs(
        H_v=[np.nan],
        H_v_sigma=[np.nan],
        G=[np.nan],
        G_sigma=[np.nan],
    )


def _upper_triangular_to_full_dimension(
    upper_triangular: npt.NDArray[np.float64], dimension: int
) -> npt.NDArray[np.float64]:
    expected = dimension * (dimension + 1) // 2
    if len(upper_triangular) != expected:
        raise ValueError(
            f"Upper triangular matrix for dimension {dimension} should have "
            f"{expected} elements, got {len(upper_triangular)}"
        )
    full = np.zeros((dimension, dimension), dtype=np.float64)
    full[np.triu_indices(dimension)] = upper_triangular
    full[np.tril_indices(dimension, -1)] = full.T[np.tril_indices(dimension, -1)]
    return full


def _full_covariance_dimension(n_elements: int) -> int:
    dimension = int((np.sqrt(8 * n_elements + 1) - 1) // 2)
    if dimension * (dimension + 1) // 2 != n_elements:
        raise ValueError(
            f"Covariance upper-triangular length {n_elements} is not a valid "
            "triangular number."
        )
    return dimension


def _full_covariance_from_upper_triangular(
    upper_triangular: List[float], solved_dimension: int
) -> npt.NDArray[np.float64]:
    full = _upper_triangular_to_full_dimension(
        np.asarray(upper_triangular, dtype=np.float64),
        _full_covariance_dimension(len(upper_triangular)),
    )
    return full[:solved_dimension, :solved_dimension]


def _solve_for_codes_to_names(codes: list[int]) -> list[str]:
    mapping = {1: "AMRAT", 2: "A2", 3: "A1", 4: "A3", 5: "DT"}
    return [mapping[code] for code in codes if code in mapping]


def _neocc_nongrav_solution_is_decodable(data: Dict[str, Any]) -> bool:
    info = data.get("nongrav") or {}
    codes = info.get("solve_for_parameter_codes") or []
    solve_for = _solve_for_codes_to_names(codes)
    if len(solve_for) != len(codes):
        return False
    dimension = info.get("dimension") or COORD_DIM
    if dimension - COORD_DIM != len(solve_for):
        return False
    if info.get("model_used") not in (None, 0, 1):
        return False
    vector = info.get("vector") or []
    if vector and len(vector) != 2:
        return False
    return all(name in _NEOCC_SUPPORTED_SOLVE_FOR for name in solve_for)


def _non_gravitational_parameters_from_neocc(
    data: Dict[str, Any],
) -> NonGravitationalParameters:
    info = data.get("nongrav") or {}
    if not info:
        return NonGravitationalParameters.nulls(1)
    solve_for = _solve_for_codes_to_names(info.get("solve_for_parameter_codes") or [])
    vector = info.get("vector") or []
    a2 = None
    if _neocc_nongrav_solution_is_decodable(data):
        if len(vector) > 1:
            a2 = float(vector[1]) * _NEOCC_UNIT_FACTORS["A2"]
        if "AMRAT" in solve_for or (vector and float(vector[0]) != 0.0):
            logger.warning(
                "NEOCC solution for object %s includes an area-to-mass ratio "
                "(AMRAT), which is not supported for storage (only A1, A2, "
                "A3); dropping its value and marginalizing it out of the "
                "covariance.",
                data.get("object_id"),
            )
    elif vector:
        logger.warning(
            "NEOCC non-grav solution for object %s uses an unsupported model or "
            "solve-for parameters (%s); the nominal values and the "
            "non-gravitational covariance block are left null.",
            data.get("object_id"),
            ",".join(solve_for) if solve_for else "unknown",
        )
    return NonGravitationalParameters.from_kwargs(
        source=["NEOCC"], A1=[None], A2=[a2], A3=[None]
    )


def _neocc_extended_covariance(
    data: Dict[str, Any],
) -> npt.NDArray[np.float64] | None:
    covariance_native = data.get("covariance_full")
    if covariance_native is None or covariance_native.shape[0] <= COORD_DIM:
        return None
    if not _neocc_nongrav_solution_is_decodable(data):
        return None
    info = data.get("nongrav") or {}
    solve_for = _solve_for_codes_to_names(info.get("solve_for_parameter_codes") or [])
    source_indices = list(range(COORD_DIM))
    target_indices = list(range(COORD_DIM))
    factors = np.ones(FULL_DIM, dtype=np.float64)
    for offset, name in enumerate(NON_GRAVITATIONAL_VALUE_FIELDS):
        if name in solve_for:
            source_indices.append(COORD_DIM + solve_for.index(name))
            target_indices.append(COORD_DIM + offset)
            factors[COORD_DIM + offset] = _NEOCC_UNIT_FACTORS[name]
    if len(target_indices) == COORD_DIM:
        return None
    full = np.zeros((FULL_DIM, FULL_DIM), dtype=np.float64)
    full[np.ix_(target_indices, target_indices)] = covariance_native[
        np.ix_(source_indices, source_indices)
    ]
    full *= np.outer(factors, factors)
    return full


def query_neocc(
    object_ids: Union[List, npt.ArrayLike],
    orbit_type: Literal["ke", "eq"] = "ke",
    orbit_epoch: Literal["middle", "present-day"] = "present-day",
    *,
    include_nongrav: bool = True,
) -> Orbits:
    """
    Query ESA's Near-Earth Object Coordination Centre (NEOCC) database for orbital elements of the specified NEOs.

    Parameters
    ----------
    object_ids : Union[List, npt.ArrayLike]
        Object IDs / designations recognizable by NEOCC.
    orbit_type : ["ke", "eq"]
        Type of orbital elements to query.
    orbit_epoch : ["middle", "present-day"]
        Epoch of the orbital elements to query.

    Returns
    -------
    orbits : `~adam_core.orbits.Orbits`
        Orbits object containing the orbital elements of the specified NEOs.
    """
    from adam_core import _rust_native

    from ..._rust.arrow import table_from_record_batch

    try:
        batch, warnings = _rust_native.query_neocc_arrow(
            [str(value) for value in object_ids],
            orbit_type,
            orbit_epoch,
            None,
            include_nongrav,
        )
    except RuntimeError as error:
        _raise_compatible_http_error(error)
    for warning in warnings:
        logger.warning("%s", warning)
    return table_from_record_batch(Orbits, batch)
