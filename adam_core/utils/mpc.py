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


def pack_mpc_designation(designation: str) -> str:
    """
    Pack a unpacked MPC designation. For example, provisional
    designation 1998 SS162 will be packed to J98SG2S. Permanent
    designation 323 will be packed to 00323.

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
    is_numbered = True
    is_provisional = True
    is_survey = True
    is_packed = False

    # If the designation contains a dash it must be a
    # survey designation
    if "-" in designation:
        is_numbered = False
        is_provisional = False

    # Lets see if its a numbered object
    while is_numbered and not is_packed:
        try:
            number = int(designation)
            if number <= 99999:
                designation_pf = "{:05}".format(number)
            elif (number >= 100000) & (number <= 619999):
                ind = int(np.floor(number / 10000))
                designation_pf = "{}{:04}".format(BASE62[ind], number % 10000)
            else:
                x = number - 620000
                number_pf = []
                while x:
                    number_pf.append(BASE62[int(x % 62)])
                    x = int(x / 62)

                number_pf.reverse()
                designation_pf = "~{}".format("".join(number_pf).zfill(4))

            is_packed = True

        except ValueError:
            is_numbered = False

    # If its not numbered, maybe its a provisional designation
    while is_provisional and not is_packed:
        try:
            year = BASE62[int(designation[0:2])] + designation[2:4]
            letter1 = designation[5]
            letter2 = designation[6]
            cycle = designation[7:]

            cycle_pf = "00"
            if len(cycle) > 0:
                cycle = int(cycle)
                if cycle <= 99:
                    cycle_pf = str(cycle).zfill(2)
                else:
                    cycle_pf = BASE62[int(cycle / 10)] + str(cycle % 10)

            designation_pf = "{}{}{}{}".format(year, letter1, cycle_pf, letter2)
            is_packed = True

        except ValueError:
            is_provisional = False

    # If its a survey designation, deal with it
    while is_survey and not is_packed:
        try:
            number = designation[0:4]
            survey = designation[5:]

            if survey == "P-L":
                survey_pf = "PLS"

            if survey[0:2] == "T-":
                survey_pf = "T{}S".format(survey[2])

            designation_pf = "{}{}".format(survey_pf, number.zfill(4))
            is_packed = True

        except ValueError:
            is_survey = False

    # If at this point its not yet packed then something went wrong
    if not is_numbered and not is_provisional and not is_survey:
        err = (
            "Unpacked designation '{}' could not be packed.\n"
            "It could not be recognized as any of the following:\n"
            " - a numbered object (e.g. '3202', '203289', '3140113')\n"
            " - a provisional designation (e.g. '1998 SV127', '2008 AA360')\n"
            " - a survey designation (e.g. '2040 P-L', '3138 T-1')"
        )
        raise ValueError(err.format(designation))

    return designation_pf


def unpack_mpc_designation(designation_pf: str) -> str:
    """
    Unpack a packed MPC designation. For example, provisional
    designation J98SG2S will be unpacked to 1998 SS162. Permanent
    designation 00323 will be unpacked to 323.

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
    is_numbered = True
    is_provisional = True
    is_survey = True
    is_unpacked = False

    while is_numbered and not is_unpacked:
        try:
            # Numbered objects (1 - 99999)
            if designation_pf.isdecimal():
                number = int(designation_pf)

            # Numbered objects (620000+)
            elif designation_pf[0] == "~":
                number = 620000
                number_pf = designation_pf[1:]
                for i, c in enumerate(number_pf):
                    power = len(number_pf) - (i + 1)
                    number += BASE62_MAP[c] * (62**power)

            # Numbered objects (100000 - 619999)
            else:
                number = BASE62_MAP[designation_pf[0]] * 10000 + int(designation_pf[1:])

            designation = str(number)
            is_unpacked = True

        except ValueError:
            is_numbered = False

    while is_provisional and not is_unpacked:
        try:
            year = str(BASE62_MAP[designation_pf[0]] * 100 + int(designation_pf[1:3]))
            letter1 = designation_pf[3]
            letter2 = designation_pf[6]
            cycle1 = designation_pf[4]
            cycle2 = designation_pf[5]

            number = int(BASE62_MAP[cycle1]) * 10 + BASE62_MAP[cycle2]
            if number == 0:
                number = ""

            designation = "{} {}{}{}".format(year, letter1, letter2, number)
            is_unpacked = True

        except ValueError:
            is_provisional = False

    while is_survey and not is_unpacked:
        try:
            number = int(designation_pf[3:8])
            survey_pf = designation_pf[0:3]

            if survey_pf == "PLS":
                survey = "P-L"

            if survey_pf[0] == "T" and survey_pf[2] == "S":
                survey = "T-{}".format(survey_pf[1])

            designation = "{} {}".format(number, survey)
            is_unpacked = True

        except ValueError:
            is_survey = False

    if not is_numbered and not is_provisional and not is_survey:
        err = (
            "Packed form designation '{}' could not be unpacked.\n"
            "It could not be recognized as any of the following:\n"
            " - a numbered object (e.g. '03202', 'K3289', '~AZaz')\n"
            " - a provisional designation (e.g. 'J98SC7V', 'K08Aa0A')\n"
            " - a survey designation (e.g. 'PLS2040', 'T1S3138')"
        )
        raise ValueError(err.format(designation_pf))

    return designation
