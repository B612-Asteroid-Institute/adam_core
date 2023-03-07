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
    year = _lookup_mpc(epoch_pf[0]) * 100 + int(epoch_pf[1:3])
    month = _lookup_mpc(epoch_pf[3])
    day = _lookup_mpc(epoch_pf[4])
    isot_string = "{:d}-{:02d}-{:02d}".format(year, month, day)

    if len(epoch_pf) > 5:
        fractional_day = float("." + epoch_pf[5:])
        hours = int((24 * fractional_day))
        minutes = int(60 * ((24 * fractional_day) - hours))
        seconds = 3600 * (24 * fractional_day - hours - minutes / 60)
        isot_string += "T{:02d}:{:02d}:{:09.6f}".format(hours, minutes, seconds)

    return Time(isot_string, format="isot", scale="tt")


def _lookup_mpc(x: str) -> int:
    # Convert the single character dates into integers.
    try:
        x = int(x)
    except ValueError:
        x = ord(x) - 55
    if x < 0 or x > 31:
        raise ValueError
    return x


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
