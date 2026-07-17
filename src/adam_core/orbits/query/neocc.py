import json
from typing import Any, Dict, List, Literal, Union

import numpy as np
import numpy.typing as npt

from ...orbits import Orbits
from ...utils.http import _raise_compatible_http_error
from ..physical_parameters import PhysicalParameters


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
    for matrix_name in ("covariance", "correlation"):
        if matrix_name in result:
            result[matrix_name] = np.asarray(
                result[matrix_name], dtype=np.float64
            ).reshape(6, 6)
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
    from adam_core import _rust_native

    from ..._rust.arrow import table_from_record_batch

    try:
        batch = _rust_native.query_neocc_arrow(
            [str(value) for value in object_ids], orbit_type, orbit_epoch
        )
    except RuntimeError as error:
        _raise_compatible_http_error(error)
    return table_from_record_batch(Orbits, batch)
