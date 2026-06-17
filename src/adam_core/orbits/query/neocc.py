import logging
from typing import Any, Dict, List, Literal, Union

import numpy as np
import numpy.typing as npt
import quivr as qv
import requests

from adam_core.coordinates.covariances import (
    transform_solved_state_covariances_jacobian,
)

from ...coordinates import CoordinateCovariances, KeplerianCoordinates, Origin
from ...coordinates.transform import _keplerian_to_cartesian_a
from ...orbits import Orbits
from ...time import Timestamp
from ..non_gravitational_parameters import NonGravitationalParameters
from ..physical_parameters import PhysicalParameters
from ..solved_state_covariances import ORBITAL_PARAMETER_NAMES, SolvedStateCovariances

logger = logging.getLogger(__name__)

# Unit conversions from NEOCC OEF native units to the canonical units declared
# by NonGravitationalParameters. The OEF header documents them as
# "Area-to-mass ratio [m^2/t], Yarkovsky parameter [1E-10 au/day^2]".
_NEOCC_UNIT_FACTORS = {
    "AMRAT": 1e-3,  # m^2/t -> m^2/kg
    "A2": 1e-10,  # 1e-10 au/d^2 -> au/d^2
}

# The NGR vector decoding below assumes the Yarkovsky model layout
# (AMRAT, A2). Solutions fit with other non-grav models (A1/A3/DT) carry a
# different vector layout and are degraded to metadata-only rows.
_NEOCC_SUPPORTED_SOLVE_FOR = frozenset(_NEOCC_UNIT_FACTORS)


def _upper_triangular_to_full_dimension(
    upper_triangular: npt.NDArray[np.float64],
    dimension: int,
) -> npt.NDArray[np.float64]:
    expected = dimension * (dimension + 1) // 2
    if len(upper_triangular) != expected:
        raise ValueError(
            f"Upper triangular matrix for dimension {dimension} should have {expected} elements, got {len(upper_triangular)}"
        )

    full = np.zeros((dimension, dimension), dtype=np.float64)
    full[np.triu_indices(dimension)] = upper_triangular
    full[np.tril_indices(dimension, -1)] = full.T[np.tril_indices(dimension, -1)]
    return full


def _full_covariance_dimension(n_elements: int) -> int:
    """
    Infer the square dimension D of a matrix stored as D*(D+1)/2 upper-triangular
    elements.
    """
    dimension = int((np.sqrt(8 * n_elements + 1) - 1) // 2)
    if dimension * (dimension + 1) // 2 != n_elements:
        raise ValueError(
            f"Covariance upper-triangular length {n_elements} is not a valid "
            "triangular number."
        )
    return dimension


def _solved_state_covariance_from_upper_triangular(
    upper_triangular: List[float],
    solved_dimension: int,
) -> npt.NDArray[np.float64]:
    """
    Build the solved-state covariance from an OEF upper-triangular matrix.

    The OEF covariance may append parameters beyond the solved orbital and
    non-gravitational state (most commonly absolute magnitude, see "2018 CW2").
    ``solved_dimension`` is the LSP-reported dimension (6 orbital + k non-grav),
    so the leading ``solved_dimension`` block is the solved-state covariance and
    any trailing rows/columns (e.g. magnitude) are dropped. This keeps a fitted
    non-grav parameter such as A2 (dimension 7, full 7x7) while still discarding
    a magnitude column appended to a 6D solution (dimension 6, full 7x7).
    """
    full = _upper_triangular_to_full_dimension(
        np.array(upper_triangular), _full_covariance_dimension(len(upper_triangular))
    )
    if full.shape[0] > solved_dimension:
        full = full[:solved_dimension, :solved_dimension]
    return full


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
    lines = data.strip().split("\n")
    result = {}

    # Parse header
    header = {}
    for line in lines:
        line = line.strip()
        if line == "END_OF_HEADER":
            break
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.split("!")[0].strip().strip("'")
    result["header"] = header

    # Find object ID
    for line in lines:
        if not line.startswith(
            ("!", " ", "format", "rectype", "refsys", "END_OF_HEADER")
        ):
            result["object_id"] = line.strip()
            break

    # Parse Keplerian elements
    for line in lines:
        if line.strip().startswith("KEP"):
            elements = line.split()[1:]
            result["elements"] = {
                "a": float(elements[0]),  # semi-major axis
                "e": float(elements[1]),  # eccentricity
                "i": float(elements[2]),  # inclination
                "node": float(elements[3]),  # longitude of ascending node
                "peri": float(elements[4]),  # argument of perihelion
                "M": float(elements[5]),  # mean anomaly
            }

    # Parse epoch
    for line in lines:
        if line.strip().startswith("MJD"):
            result["epoch"] = float(line.split()[1])
            result["time_system"] = line.split()[2]

    # Parse magnitude: OEF MAG line is (absolute magnitude H, slope parameter G), V-band.
    # Ref: ESA NEOCC Automated Data Access, https://neo.ssa.esa.int/computer-access
    for line in lines:
        if line.strip().startswith("MAG"):
            mag_data = line.split()[1:]
            result["magnitude"] = {
                "H": float(mag_data[0]),
                "G": float(mag_data[1]),
            }

    # Parse non-gravitational metadata. Built with setdefault/update so the
    # result does not depend on whether LSP or NGR appears first.
    for line in lines:
        if line.strip().startswith("LSP"):
            lsp = [int(value) for value in line.split()[1:]]
            result.setdefault("nongrav", {}).update(
                {
                    "model_used": lsp[0] if len(lsp) > 0 else None,
                    "parameter_count": lsp[1] if len(lsp) > 1 else None,
                    "dimension": lsp[2] if len(lsp) > 2 else 6,
                    "solve_for_parameter_codes": lsp[3:] if len(lsp) > 3 else [],
                }
            )
        if line.strip().startswith("NGR"):
            ngr = [float(value) for value in line.split()[1:]]
            result.setdefault("nongrav", {})["vector"] = ngr

    # Parse derived parameters (marked with !)
    derived = {}
    for line in lines:
        if line.strip().startswith("!") and len(line.split()) >= 3:
            key = line.split()[1].lower()
            try:
                value = float(line.split()[2])
                derived[key] = value
            except ValueError:
                derived[key] = line.split()[2]
    result["derived"] = derived

    # Parse covariance matrix
    cov_matrix = []
    for line in lines:
        if line.strip().startswith("COV"):
            cov_matrix.extend([float(x) for x in line.split()[1:]])
    if cov_matrix:
        result["covariance_full"] = _solved_state_covariance_from_upper_triangular(
            cov_matrix, result.get("nongrav", {}).get("dimension", 6)
        )
        result["covariance"] = result["covariance_full"][:6, :6]

    # Parse correlation matrix
    cor_matrix = []
    for line in lines:
        if line.strip().startswith("COR"):
            cor_matrix.extend([float(x) for x in line.split()[1:]])
    if cor_matrix:
        result["correlation_full"] = _solved_state_covariance_from_upper_triangular(
            cor_matrix, result.get("nongrav", {}).get("dimension", 6)
        )
        result["correlation"] = result["correlation_full"][:6, :6]

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


def _solve_for_codes_to_names(codes: list[int]) -> list[str]:
    mapping = {
        1: "AMRAT",
        2: "A2",
        3: "A1",
        4: "A3",
        5: "DT",
    }
    return [mapping[code] for code in codes if code in mapping]


def _neocc_solved_state_is_supported(data: Dict[str, Any]) -> bool:
    """
    Return True if the parsed OEF non-grav solution can be decoded with the
    Yarkovsky-model assumptions used below: every solve-for code is known and
    in {AMRAT, A2}, the solved dimension is consistent with the solve-for
    list, and the model is either absent (0) or Yarkovsky (1).
    """
    info = data.get("nongrav") or {}
    codes = info.get("solve_for_parameter_codes") or []
    solve_for = _solve_for_codes_to_names(codes)
    if len(solve_for) != len(codes):
        # Unknown solve-for codes were dropped by the mapping.
        return False
    dimension = info.get("dimension") or 6
    if dimension - 6 != len(solve_for):
        return False
    if info.get("model_used") not in (None, 0, 1):
        return False
    return all(name in _NEOCC_SUPPORTED_SOLVE_FOR for name in solve_for)


def _neocc_solved_sigmas(
    data: Dict[str, Any], solve_for: list[str]
) -> Dict[str, float]:
    """
    Extract per-parameter sigmas (in canonical units) from the diagonal of the
    raw OEF solved-state covariance.
    """
    covariance = data.get("covariance_full")
    if covariance is None:
        return {}
    sigmas = {}
    for offset, name in enumerate(solve_for):
        variance = float(covariance[6 + offset, 6 + offset])
        if variance >= 0.0:
            sigmas[name] = np.sqrt(variance) * _NEOCC_UNIT_FACTORS[name]
    return sigmas


def _scale_neocc_solved_covariance(
    covariance: npt.NDArray[np.float64], solve_for: list[str]
) -> npt.NDArray[np.float64]:
    """
    Convert the non-grav rows/columns of a raw OEF solved-state covariance to
    the canonical units declared by NonGravitationalParameters. Scaling both
    the row and the column squares the factor on the diagonal.
    """
    scaled = np.array(covariance, dtype=np.float64)
    for offset, name in enumerate(solve_for):
        factor = _NEOCC_UNIT_FACTORS[name]
        index = 6 + offset
        scaled[index, :] *= factor
        scaled[:, index] *= factor
    return scaled


def _non_gravitational_parameters_from_neocc(
    data: Dict[str, Any],
) -> NonGravitationalParameters:
    info = data.get("nongrav") or {}
    if not info:
        return NonGravitationalParameters.nulls(1)

    solve_for = _solve_for_codes_to_names(info.get("solve_for_parameter_codes") or [])
    model_used = info.get("model_used")
    model = None
    if model_used == 1:
        model = "yarkovsky"
    elif model_used not in (None, 0):
        model = f"neocc-model-{model_used}"

    vector = info.get("vector") or []
    amrat = None
    a2 = None
    sigmas: Dict[str, float] = {}
    if _neocc_solved_state_is_supported(data):
        # The Yarkovsky-model NGR vector is (AMRAT, A2) in OEF native units;
        # convert to the canonical units declared by NonGravitationalParameters.
        amrat = (
            float(vector[0]) * _NEOCC_UNIT_FACTORS["AMRAT"] if len(vector) > 0 else None
        )
        a2 = float(vector[1]) * _NEOCC_UNIT_FACTORS["A2"] if len(vector) > 1 else None
        sigmas = _neocc_solved_sigmas(data, solve_for)
    elif vector:
        logger.warning(
            "NEOCC non-grav solution for object %s uses an unsupported model or "
            "solve-for parameters (%s); the nominal values and solved-state "
            "covariance are left null.",
            data.get("object_id"),
            ",".join(solve_for) if solve_for else "unknown",
        )

    return NonGravitationalParameters.from_kwargs(
        source=["NEOCC"],
        model=[model],
        solution_dimension=[info.get("dimension")],
        parameter_count=[len(solve_for) if solve_for else None],
        estimated_parameter_names=[",".join(solve_for) if solve_for else None],
        A1=[None],
        A1_sigma=[None],
        A2=[a2],
        A2_sigma=[sigmas.get("A2")],
        A3=[None],
        A3_sigma=[None],
        DT=[None],
        DT_sigma=[None],
        R0=[None],
        R0_sigma=[None],
        ALN=[None],
        ALN_sigma=[None],
        NK=[None],
        NK_sigma=[None],
        NM=[None],
        NM_sigma=[None],
        NN=[None],
        NN_sigma=[None],
        AMRAT=[amrat],
        AMRAT_sigma=[sigmas.get("AMRAT")],
        RHO=[None],
        RHO_sigma=[None],
    )


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
    include_nongrav : bool, optional
        If True (default), populate the non-gravitational parameter and
        solved-state covariance columns from the NEOCC solution (converted to
        the canonical units declared by NonGravitationalParameters). If False,
        those columns are returned null.

    Returns
    -------
    orbits : `~adam_core.orbits.Orbits`
        Orbits object containing the orbital elements of the specified NEOs.
    """
    base_url = "https://neo.ssa.esa.int/PSDB-portlet/download"

    if orbit_type == "eq":
        raise NotImplementedError("Equinoctial elements are not supported yet.")

    if orbit_epoch == "middle":
        orbit_epoch = 0
    elif orbit_epoch == "present-day":
        orbit_epoch = 1
    else:
        raise ValueError(f"Invalid orbit epoch: {orbit_epoch}")

    orbits = Orbits.empty()

    for object_id in object_ids:

        # Clean object ID so that there are no spaces
        object_id = object_id.replace(" ", "")

        params = {"file": f"{object_id}.{orbit_type}{orbit_epoch}"}

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = _parse_oef(response.text)

        if orbit_type == "ke" and "time_system" in data:
            time_scale = data["time_system"]
            if time_scale == "TDT":
                time_scale = "tt"
            else:
                raise ValueError(f"Unsupported time scale: {time_scale}")

            if data["header"]["refsys"] != "ECLM J2000":
                raise ValueError(
                    f"Unsupported reference system: {data['header']['refsys']}"
                )

            phys = _physical_parameters_from_neocc(data)
            nongrav = _non_gravitational_parameters_from_neocc(data)
            solve_for = _solve_for_codes_to_names(
                (data.get("nongrav") or {}).get("solve_for_parameter_codes") or []
            )
            solved_covariance_native = data.get("covariance_full")
            if solved_covariance_native is not None:
                if _neocc_solved_state_is_supported(data):
                    # Convert the non-grav rows/columns to canonical units so
                    # the covariance is consistent with the nominal values.
                    solved_covariance_native = _scale_neocc_solved_covariance(
                        solved_covariance_native, solve_for
                    )
                elif len(solve_for) > 0 or solved_covariance_native.shape[0] > 6:
                    # _non_gravitational_parameters_from_neocc warned already.
                    solved_covariance_native = None

            keplerian_coordinates = KeplerianCoordinates.from_kwargs(
                a=[data["elements"]["a"]],
                e=[data["elements"]["e"]],
                i=[data["elements"]["i"]],
                raan=[data["elements"]["node"]],
                ap=[data["elements"]["peri"]],
                M=[data["elements"]["M"]],
                time=Timestamp.from_mjd([data["epoch"]], scale=time_scale),
                covariance=CoordinateCovariances.from_matrix(
                    data["covariance"].reshape(
                        1,
                        6,
                        6,
                    )
                ),
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"]),
            )
            solved_covariance_cartesian = transform_solved_state_covariances_jacobian(
                keplerian_coordinates.values,
                [solved_covariance_native],
                _keplerian_to_cartesian_a,
                in_axes=(0, 0, None, None),
                out_axes=0,
                mu=keplerian_coordinates.origin.mu(),
                max_iter=1000,
                tol=1e-15,
            )

            orbit = Orbits.from_kwargs(
                orbit_id=[data["object_id"]],
                object_id=[data["object_id"]],
                coordinates=keplerian_coordinates.to_cartesian(),
                physical_parameters=phys,
                non_gravitational_parameters=nongrav,
                solved_state_covariance=SolvedStateCovariances.from_matrix(
                    solved_covariance_cartesian,
                    (
                        [list(ORBITAL_PARAMETER_NAMES) + solve_for]
                        if solved_covariance_native is not None
                        else [None]
                    ),
                ),
            )
            orbits = qv.concatenate([orbits, orbit])

    return orbits if include_nongrav else orbits.without_non_gravitational_parameters()
