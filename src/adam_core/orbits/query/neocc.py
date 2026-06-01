from typing import Any, Dict, List, Literal, Union

import numpy as np
import numpy.typing as npt
import quivr as qv
import requests

from adam_core.coordinates.covariances import _upper_triangular_to_full

from ...coordinates import CoordinateCovariances, KeplerianCoordinates, Origin
from ..non_gravitational_parameters import NonGravitationalParameters
from ...orbits import Orbits
from ...time import Timestamp
from ..physical_parameters import PhysicalParameters


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

    # Parse non-gravitational metadata.
    for line in lines:
        if line.strip().startswith("LSP"):
            lsp = [int(value) for value in line.split()[1:]]
            result["nongrav"] = {
                "model_used": lsp[0] if len(lsp) > 0 else None,
                "parameter_count": lsp[1] if len(lsp) > 1 else None,
                "dimension": lsp[2] if len(lsp) > 2 else 6,
                "solve_for_parameter_codes": lsp[3:] if len(lsp) > 3 else [],
            }
        if line.strip().startswith("NGR"):
            ngr = [float(value) for value in line.split()[1:]]
            if "nongrav" not in result:
                result["nongrav"] = {}
            result["nongrav"]["vector"] = ngr

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
        dimension = result.get("nongrav", {}).get("dimension", 6)
        if dimension == 6:
            result["covariance_full"] = _upper_triangular_to_full(np.array(cov_matrix))
        else:
            result["covariance_full"] = _upper_triangular_to_full_dimension(
                np.array(cov_matrix), dimension
            )
        result["covariance"] = result["covariance_full"][:6, :6]

    # Parse correlation matrix
    cor_matrix = []
    for line in lines:
        if line.strip().startswith("COR"):
            cor_matrix.extend([float(x) for x in line.split()[1:]])
    if cor_matrix:
        dimension = result.get("nongrav", {}).get("dimension", 6)
        if dimension == 6:
            result["correlation_full"] = _upper_triangular_to_full(np.array(cor_matrix))
        else:
            result["correlation_full"] = _upper_triangular_to_full_dimension(
                np.array(cor_matrix), dimension
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


def _non_gravitational_parameters_from_neocc(
    data: Dict[str, Any],
) -> NonGravitationalParameters:
    info = data.get("nongrav") or {}
    if not info:
        return NonGravitationalParameters.nulls(1)

    vector = info.get("vector") or []
    amrat = float(vector[0]) if len(vector) > 0 else None
    # NEOCC OEF documents the Yarkovsky parameter in 1E-10 au/day^2.
    a2 = float(vector[1]) * 1e-10 if len(vector) > 1 else None

    solve_for = _solve_for_codes_to_names(info.get("solve_for_parameter_codes") or [])
    model_used = info.get("model_used")
    model = None
    if model_used == 1:
        model = "yarkovsky"
    elif model_used not in (None, 0):
        model = f"neocc-model-{model_used}"

    return NonGravitationalParameters.from_kwargs(
        source=["NEOCC"],
        model=[model],
        solution_dimension=[info.get("dimension")],
        parameter_count=[info.get("parameter_count")],
        estimated_parameter_names=[",".join(solve_for) if solve_for else None],
        A1=[None],
        A1_sigma=[None],
        A2=[a2],
        A2_sigma=[None],
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
        AMRAT_sigma=[None],
        RHO=[None],
        RHO_sigma=[None],
    )


def query_neocc(
    object_ids: Union[List, npt.ArrayLike],
    orbit_type: Literal["ke", "eq"] = "ke",
    orbit_epoch: Literal["middle", "present-day"] = "present-day",
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
        if orbit_type == "ke":

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

            orbit = Orbits.from_kwargs(
                orbit_id=[data["object_id"]],
                object_id=[data["object_id"]],
                coordinates=KeplerianCoordinates.from_kwargs(
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
                ).to_cartesian(),
                physical_parameters=phys,
                non_gravitational_parameters=nongrav,
            )
            orbits = qv.concatenate([orbits, orbit])

    return orbits
