import numpy.typing as npt
from astropy.time import Time

BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE62_MAP = {BASE62[i]: i for i in range(len(BASE62))}


def _unpack_mpc_date(epoch_pf: str) -> Time:
    # Taken from Lynne Jones' SSO TOOLS.
    # See https://minorplanetcenter.net/iau/info/PackedDates.html
    # for MPC documentation on packed dates.
    # Examples:
    #    1998 Jan. 18.73     = J981I73
    #    2001 Oct. 22.138303 = K01AM138303
    epoch_pf = str(epoch_pf)
    year = int(epoch_pf[0], base=32) * 100 + int(epoch_pf[1:3])
    month = int(epoch_pf[3], base=32)
    day = int(epoch_pf[4], base=32)
    isot_string = "{:d}-{:02d}-{:02d}".format(year, month, day)

    if len(epoch_pf) > 5:
        fractional_day = float("." + epoch_pf[5:])
        hours = int((24 * fractional_day))
        minutes = int(60 * ((24 * fractional_day) - hours))
        seconds = 3600 * (24 * fractional_day - hours - minutes / 60)
        isot_string += "T{:02d}:{:02d}:{:09.6f}".format(hours, minutes, seconds)

    return Time(isot_string, format="isot", scale="tt")


def convert_mpc_packed_dates(pf_tt: npt.ArrayLike) -> Time:
    """
    Convert MPC packed form dates (in the TT time scale) to
    MJDs in TT. See: https://minorplanetcenter.net/iau/info/PackedDates.html
    for details on the packed date format.

    Parameters
    ----------
    pf_tt : `~numpy.ndarray` (N)
        MPC-style packed form epochs in the TT time scale.

    Returns
    -------
    mjd_tt : `~astropy.time.core.Time` (N)
        Epochs in TT MJDs.
    """
    isot_tt = []
    for epoch in pf_tt:
        isot_tt.append(_unpack_mpc_date(epoch))

    return Time(isot_tt)


def pack_numbered_designation(designation: str) -> str:
    """
    Pack a numbered MPC designation.

    Runs in the Rust backend (legacy-exact port, W11 helper migration).

    Examples of numbered designations:
        Numbered      Packed
        3202          03202
        50000         50000
        100345        A0345
        360017        a0017
        203289        K3289
        620000        ~0000
        620061        ~000z
        3140113       ~AZaz
        15396335      ~zzzz

    Parameters
    ----------
    designation : str
        MPC numbered designation.

    Returns
    -------
    designation_pf : str
        MPC packed numbered designation.

    Raises
    ------
    ValueError : If the numbered designation cannot be packed.
        If the numbered designation is larger than 15396335.
    """
    from adam_core import _rust_native as _rn

    return _rn.pack_numbered_designation(designation)


def pack_provisional_designation(designation: str) -> str:
    """
    Pack a provisional MPC designation.

    Runs in the Rust backend (legacy-exact port, W11 helper migration).

    Examples of provisional designations:
        Provisional   Packed
        1995 XA       J95X00A
        1995 XL1      J95X01L
        1995 FB13     J95F13B
        1998 SQ108    J98SA8Q
        1998 SV127    J98SC7V
        1998 SS162    J98SG2S
        2099 AZ193    K99AJ3Z
        2008 AA360    K08Aa0A
        2007 TA418    K07Tf8A

    Parameters
    ----------
    designation : str
        MPC provisional designation.

    Returns
    -------
    designation_pf : str
        MPC packed provisional designation.

    Raises
    ------
    ValueError : If the provisional designation cannot be packed.
        The provisional designations is not at least 6 characters long.
        The first 4 characters of the provisional designation are not a year.
        The 5th character of the provisional designation is not a space.
        The provisional designation contains a hyphen.
        The half-month letter is I or Z.
    """
    from adam_core import _rust_native as _rn

    return _rn.pack_provisional_designation(designation)


def pack_survey_designation(designation: str) -> str:
    """
    Pack a survey MPC designation.

    Runs in the Rust backend (legacy-exact port, W11 helper migration).

    Examples of survey designations:
        Survey       Packed
        2040 P-L     PLS2040
        3138 T-1     T1S3138
        1010 T-2     T2S1010
        4101 T-3     T3S4101

    Parameters
    ----------
    designation : str
        MPC survey designation.

    Returns
    -------
    designation_pf : str
        MPC packed survey designation.

    Raises
    ------
    ValueError : If the survey designation cannot be packed.
        The survey designation does not start with P-L, T-1, T-2, or T-3.
    """
    from adam_core import _rust_native as _rn

    return _rn.pack_survey_designation(designation)


def pack_mpc_designation(designation: str) -> str:
    """
    Pack a unpacked MPC designation. For example, provisional
    designation 1998 SS162 will be packed to J98SG2S. Permanent
    designation 323 will be packed to 00323.

    Runs in the Rust backend (legacy-exact port, W11 helper migration).

    TODO: add support for comet and natural satellite designations

    Parameters
    ----------
    designation : str
        MPC unpacked designation.

    Returns
    -------
    designation_pf : str
        MPC packed form designation.

    Raises
    ------
    ValueError : If designation cannot be packed.
    """
    from adam_core import _rust_native as _rn

    return _rn.pack_mpc_designation(designation)


def unpack_numbered_designation(designation_pf: str) -> str:
    """
    Unpack a numbered MPC designation.

    Runs in the Rust backend (legacy-exact port, W11 helper migration).

    Examples of numbered designations:
        Numbered      Unpacked
        03202         3202
        50000         50000
        A0345         100345
        a0017         360017
        K3289         203289
        ~0000         620000
        ~000z         620061
        ~AZaz         3140113
        ~zzzz         15396335

    Parameters
    ----------
    designation_pf : str
        MPC packed numbered designation.

    Returns
    -------
    designation : str
        MPC unpacked numbered designation.

    Raises
    ------
    ValueError : If the numbered designation cannot be unpacked.
    """
    from adam_core import _rust_native as _rn

    return _rn.unpack_numbered_designation(designation_pf)


def unpack_provisional_designation(designation_pf: str) -> str:
    """
    Unpack a provisional MPC designation.

    Runs in the Rust backend (legacy-exact port, W11 helper migration).

    Examples of provisional designations:
        Provisional   Unpacked
        J95X00A       1995 XA
        J95X01L       1995 XL1
        J95F13B       1995 FB13
        J98SA8Q       1998 SQ108
        J98SC7V       1998 SV127
        J98SG2S       1998 SS162
        K99AJ3Z       2099 AZ193
        K08Aa0A       2008 AA360
        K07Tf8A       2007 TA418

    Parameters
    ----------
    designation_pf : str
        MPC packed provisional designation.

    Returns
    -------
    designation : str
        MPC unpacked provisional designation.

    Raises
    ------
    ValueError : If the provisional designation cannot be unpacked.
        The packed provisional designation is not 7 characters long.
        The packed provisional designation does not have a year.
    """
    from adam_core import _rust_native as _rn

    return _rn.unpack_provisional_designation(designation_pf)


def unpack_survey_designation(designation_pf: str) -> str:
    """
    Unpack a survey MPC designation.

    Runs in the Rust backend (legacy-exact port, W11 helper migration).

    Examples of survey designations:
        Survey       Packed
        PLS2040      2040 P-L
        T1S3138      3138 T-1
        T2S1010      1010 T-2
        T3S4101      4101 T-3

    Parameters
    ----------
    designation_pf : str
        MPC packed survey designation.

    Returns
    -------
    designation : str
        MPC unpacked survey designation.

    Raises
    ------
    ValueError : If the survey designation cannot be unpacked.
        The packed survey designation does not start with PLS, T1S, T2S, or T3S.
    """
    from adam_core import _rust_native as _rn

    return _rn.unpack_survey_designation(designation_pf)


def unpack_mpc_designation(designation_pf: str) -> str:
    """
    Unpack a packed MPC designation. For example, provisional
    designation J98SG2S will be unpacked to 1998 SS162. Permanent
    designation 00323 will be unpacked to 323.

    Runs in the Rust backend (legacy-exact port, W11 helper migration).

    TODO: add support for comet and natural satellite designations

    Parameters
    ----------
    designation_pf : str
        MPC packed form designation.

    Returns
    -------
    designation : str
        MPC unpacked designation.

    Raises
    ------
    ValueError : If designation_pf cannot be unpacked.
    """
    from adam_core import _rust_native as _rn

    return _rn.unpack_mpc_designation(designation_pf)
