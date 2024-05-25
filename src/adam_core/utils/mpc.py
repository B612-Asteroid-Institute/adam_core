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
    number = int(designation)
    if number > 15396335:
        raise ValueError(
            "Numbered designation is too large. Maximum supported is 15396335."
        )

    if number <= 99999:
        return "{:05}".format(number)
    elif (number >= 100000) and (number <= 619999):
        bigpart, remainder = divmod(number, 10000)
        return f"{BASE62[bigpart]}{remainder:04}"
    else:
        x = number - 620000
        number_pf = []
        while x:
            number_pf.append(BASE62[int(x % 62)])
            x //= 62

        number_pf.reverse()
        return "~{}".format("".join(number_pf).zfill(4))


def pack_provisional_designation(designation: str) -> str:
    """
    Pack a provisional MPC designation.

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
    if len(designation) < 6:
        raise ValueError(
            "Provisional designations should be at least 6 characters long."
        )
    if not designation[:3].isdecimal():
        raise ValueError(
            "Expected the first 4 characters of the provisional designation to be a year."
        )
    if designation[4] != " ":
        raise ValueError(
            "Expected the 5th character of the provisional designation to be a space."
        )
    if "-" in designation:
        raise ValueError("Provisional designations cannot contain a hyphen.")

    year = BASE62[int(designation[0:2])] + designation[2:4]
    letter1 = designation[5]
    letter2 = designation[6]
    cycle = designation[7:]

    if letter1 in {"I", "Z"}:
        raise ValueError("Half-month letters cannot be I or Z.")
    if letter1.isdecimal() or letter2.isdecimal():
        raise ValueError("Invalid provisional designation.")

    cycle_pf = "00"
    if len(cycle) > 0:
        cycle_int = int(cycle)
        if cycle_int <= 99:
            cycle_pf = str(cycle_int).zfill(2)
        else:
            cycle_pf = BASE62[cycle_int // 10] + str(cycle_int % 10)

    designation_pf = "{}{}{}{}".format(year, letter1, cycle_pf, letter2)
    return designation_pf


def pack_survey_designation(designation: str) -> str:
    """
    Pack a survey MPC designation.

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
    number = designation[0:4]
    survey = designation[5:]

    if survey == "P-L":
        survey_pf = "PLS"

    elif survey[0:2] == "T-" and survey[2] in {"1", "2", "3"}:
        survey_pf = "T{}S".format(survey[2])

    else:
        raise ValueError("Survey designations must start with P-L, T-1, T-2, T-3.")

    designation_pf = "{}{}".format(survey_pf, number.zfill(4))
    return designation_pf


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
    # Lets see if its a numbered object
    try:
        return pack_numbered_designation(designation)
    except ValueError:
        pass

    # If its not numbered, maybe its a provisional designation
    try:
        return pack_provisional_designation(designation)
    except ValueError:
        pass

    # If its a survey designation, deal with it
    try:
        return pack_survey_designation(designation)
    except ValueError:
        pass

    err = (
        "Unpacked designation '{}' could not be packed.\n"
        "It could not be recognized as any of the following:\n"
        " - a numbered object (e.g. '3202', '203289', '3140113')\n"
        " - a provisional designation (e.g. '1998 SV127', '2008 AA360')\n"
        " - a survey designation (e.g. '2040 P-L', '3138 T-1')"
    )
    raise ValueError(err.format(designation))


def unpack_numbered_designation(designation_pf: str) -> str:
    """
    Unpack a numbered MPC designation.

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
        The packed numbered designation is not at least 4 characters long.
    """
    number = None
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

    if number is None:
        raise ValueError("Packed numbered designation could not be unpacked.")
    else:
        designation = str(number)

    return designation


def unpack_provisional_designation(designation_pf: str) -> str:
    """
    Unpack a provisional MPC designation.

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
    if len(designation_pf) != 7:
        raise ValueError("Provisional designation must be 7 characters long.")
    if not designation_pf[1].isdecimal() or not designation_pf[2].isdecimal():
        raise ValueError("Provisional designation must have a year.")
    year = str(BASE62_MAP[designation_pf[0]] * 100 + int(designation_pf[1:3]))
    letter1 = designation_pf[3]
    letter2 = designation_pf[6]
    if letter1.isdecimal() or letter2.isdecimal():
        raise ValueError()
    cycle1 = designation_pf[4]
    cycle2 = designation_pf[5]

    number = int(BASE62_MAP[cycle1]) * 10 + BASE62_MAP[cycle2]
    if number == 0:
        number_str = ""
    else:
        number_str = str(number)

    designation = "{} {}{}{}".format(year, letter1, letter2, number_str)

    return designation


def unpack_survey_designation(designation_pf: str) -> str:
    """
    Unpack a survey MPC designation.

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
    number = int(designation_pf[3:8])
    survey_pf = designation_pf[0:3]
    if survey_pf not in {"PLS", "T1S", "T2S", "T3S"}:
        raise ValueError(
            "Packed survey designation must start with PLS, T1S, T2S, or T3S."
        )

    if survey_pf == "PLS":
        survey = "P-L"

    if survey_pf[0] == "T" and survey_pf[2] == "S":
        survey = "T-{}".format(survey_pf[1])

    designation = "{} {}".format(number, survey)
    return designation


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
    # Lets see if its a numbered object
    try:
        return unpack_numbered_designation(designation_pf)
    except ValueError:
        pass

    # Lets see if its a provisional designation
    try:
        return unpack_provisional_designation(designation_pf)
    except ValueError:
        pass

    # Lets see if its a survey designation
    try:
        return unpack_survey_designation(designation_pf)
    except ValueError:
        pass

    # At this point we haven't had any success so lets raise an error
    err = (
        "Packed form designation '{}' could not be unpacked.\n"
        "It could not be recognized as any of the following:\n"
        " - a numbered object (e.g. '03202', 'K3289', '~AZaz')\n"
        " - a provisional designation (e.g. 'J98SC7V', 'K08Aa0A')\n"
        " - a survey designation (e.g. 'PLS2040', 'T1S3138')"
    )
    raise ValueError(err.format(designation_pf))
