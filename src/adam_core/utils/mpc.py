from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy.typing as npt

if TYPE_CHECKING:
    from astropy.time import Time

BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE62_MAP = {BASE62[i]: i for i in range(len(BASE62))}


@dataclass(frozen=True)
class ADESDesignationParts:
    """Validated ADES identity fields derived from one submitted MPC label."""

    perm_id: str | None = None
    prov_id: str | None = None
    trk_sub: str | None = None


def _astropy_time_class():
    try:
        from astropy.time import Time
    except ModuleNotFoundError as error:
        raise ImportError(
            "Astropy is required for MPC packed-date Time compatibility; "
            "install adam-core[astropy]"
        ) from error
    return Time


def _unpack_mpc_date(epoch_pf: str) -> Time:
    # See https://minorplanetcenter.net/iau/info/PackedDates.html
    # for MPC documentation on packed dates.
    # Examples:
    #    1998 Jan. 18.73     = J981I73
    #    2001 Oct. 22.138303 = K01AM138303
    # The packed-date computation runs in the Rust backend (bead
    # personal-cmy.26); astropy wraps the resulting ISOT string.
    from adam_core import _rust_native as _rn

    return _astropy_time_class()(
        _rn.unpack_mpc_date_isot(str(epoch_pf)), format="isot", scale="tt"
    )


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
    from adam_core import _rust_native as _rn

    # One Rust crossing decodes the complete input batch; Astropy remains the
    # external time-object compatibility boundary.
    isot_tt = _rn.unpack_mpc_dates_isot([str(epoch) for epoch in pf_tt])
    return _astropy_time_class()(isot_tt, format="isot", scale="tt")


def _rust_designation_call(name: str, designation: str) -> str:
    """Call one strict scalar Rust designation codec."""
    from adam_core import _rust_native as _rn

    return getattr(_rn, name)(designation)


def pack_numbered_designation(designation: str) -> str:
    """Pack one canonical decimal minor-planet number."""
    return _rust_designation_call("pack_numbered_designation", designation)


def unpack_numbered_designation(designation_pf: str) -> str:
    """Unpack one canonical five-character numbered minor-planet identity."""
    return _rust_designation_call("unpack_numbered_designation", designation_pf)


def pack_provisional_designation(designation: str) -> str:
    """Pack a canonical ordinary, extended, or A-prefix minor-planet provisional."""
    return _rust_designation_call("pack_provisional_designation", designation)


def unpack_provisional_designation(designation_pf: str) -> str:
    """Unpack a canonical ordinary or underscore-extended minor-planet provisional."""
    return _rust_designation_call("unpack_provisional_designation", designation_pf)


def pack_survey_designation(designation: str) -> str:
    """Pack a canonical P-L or T-1/T-2/T-3 survey designation."""
    return _rust_designation_call("pack_survey_designation", designation)


def unpack_survey_designation(designation_pf: str) -> str:
    """Unpack a canonical PLS/T1S/T2S/T3S survey designation."""
    return _rust_designation_call("unpack_survey_designation", designation_pf)


def pack_numbered_comet_designation(designation: str) -> str:
    """Pack a canonical numbered comet/interstellar identity and optional fragment."""
    return _rust_designation_call("pack_numbered_comet_designation", designation)


def unpack_numbered_comet_designation(designation_pf: str) -> str:
    """Unpack a canonical numbered comet/interstellar identity."""
    return _rust_designation_call("unpack_numbered_comet_designation", designation_pf)


def pack_provisional_comet_designation(designation: str) -> str:
    """Pack a modern, ancient, BCE, or 12-character combined comet identity."""
    return _rust_designation_call("pack_provisional_comet_designation", designation)


def unpack_provisional_comet_designation(designation_pf: str) -> str:
    """Unpack a modern, ancient, BCE, or 12-character combined comet identity."""
    return _rust_designation_call(
        "unpack_provisional_comet_designation", designation_pf
    )


def pack_comet_designation(designation: str) -> str:
    """Pack any supported canonical comet or interstellar designation."""
    return _rust_designation_call("pack_comet_designation", designation)


def unpack_comet_designation(designation_pf: str) -> str:
    """Unpack any supported canonical comet or interstellar designation."""
    return _rust_designation_call("unpack_comet_designation", designation_pf)


def pack_permanent_satellite_designation(designation: str) -> str:
    """Pack a permanent natural-satellite identity written with a Roman numeral."""
    return _rust_designation_call("pack_permanent_satellite_designation", designation)


def unpack_permanent_satellite_designation(designation_pf: str) -> str:
    """Unpack a permanent natural-satellite identity to its official Roman numeral."""
    return _rust_designation_call(
        "unpack_permanent_satellite_designation", designation_pf
    )


def pack_provisional_satellite_designation(designation: str) -> str:
    """Pack a canonical provisional natural-satellite designation."""
    return _rust_designation_call("pack_provisional_satellite_designation", designation)


def unpack_provisional_satellite_designation(designation_pf: str) -> str:
    """Unpack a canonical provisional natural-satellite designation."""
    return _rust_designation_call(
        "unpack_provisional_satellite_designation", designation_pf
    )


def pack_satellite_designation(designation: str) -> str:
    """Pack a permanent or provisional natural-satellite designation."""
    return _rust_designation_call("pack_satellite_designation", designation)


def unpack_satellite_designation(designation_pf: str) -> str:
    """Unpack a permanent or provisional natural-satellite designation."""
    return _rust_designation_call("unpack_satellite_designation", designation_pf)


def pack_mpc_designation(designation: str) -> str:
    """Strictly dispatch and pack any supported canonical MPC designation."""
    return _rust_designation_call("pack_mpc_designation", designation)


def unpack_mpc_designation(designation_pf: str) -> str:
    """Strictly dispatch and unpack any supported canonical MPC designation."""
    return _rust_designation_call("unpack_mpc_designation", designation_pf)


def parse_ades_designation(designation: str) -> ADESDesignationParts:
    """Classify a canonical unpacked identity into ADES identity fields.

    Official packed identities, malformed/noncanonical designations, and
    ambiguous values are rejected before the bounded tracking-ID fallback.
    Submitted unpacked identity is retained; a combined numbered/provisional
    comet is represented by its two canonical ADES fields.
    """
    from adam_core import _rust_native as _rn

    perm_id, prov_id, trk_sub = _rn.parse_ades_designation(designation)
    return ADESDesignationParts(perm_id=perm_id, prov_id=prov_id, trk_sub=trk_sub)
