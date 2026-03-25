import logging
import re
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pyarrow.compute as pc
import quivr as qv
from astropy.time import Time

from ..time import Timestamp

STRING100 = 100
STRING25 = 25

logger = logging.getLogger(__name__)

ADESValidationGoal = Literal["spec_compliance", "mpc_submission"]
ADESValidationEnforcement = Literal["error", "warn", "autofix"]


@dataclass(frozen=True)
class ADESValidationProblem:
    code: str
    severity: Literal["error", "warning"]
    message: str
    line_number: int | None = None
    line: str | None = None


_IDENTIFICATION_GROUP_COLUMNS: tuple[str, ...] = (
    "permID",
    "provID",
    "artSat",
    "trkSub",
)
_KEYWORD_TOKEN_RE = re.compile(r"^[a-z]")
_KEYWORD_TOKEN_CANDIDATE_RE = re.compile(r"^[A-Za-z]")
_KEYWORD_REQUIRED_COLUMNS: tuple[str, ...] = ("obsTime",)


@dataclass
class ObservatoryObsContext:
    mpcCode: str
    name: Optional[str] = None

    def __post_init__(self):
        assert len(self.mpcCode) in [3, 4]
        if self.name is not None:
            assert len(self.name) <= STRING100


@dataclass
class SubmitterObsContext:
    name: str
    institution: Optional[str] = None

    def __post_init__(self):
        assert len(self.name) <= STRING100
        if self.institution is not None:
            assert len(self.institution) <= STRING100


@dataclass
class TelescopeObsContext:
    design: str
    aperture: float
    detector: str
    name: Optional[str] = None
    fRatio: Optional[float] = None
    filter: Optional[str] = None
    arraySize: Optional[str] = None
    pixelSize: Optional[float] = None

    def __post_init__(self):
        if self.name is not None:
            assert len(self.name) <= STRING100
        assert len(self.design) <= STRING25
        assert self.aperture > 0
        assert len(self.detector) <= STRING25
        if self.fRatio is not None:
            assert self.fRatio > 0
        if self.filter is not None:
            assert len(self.filter) <= STRING25
        if self.arraySize is not None:
            assert len(self.arraySize) <= STRING25
        if self.pixelSize is not None:
            assert self.pixelSize > 0


@dataclass
class SoftwareObsContext:
    astrometry: Optional[str] = None
    fitOrder: Optional[str] = None
    photometry: Optional[str] = None
    objectDetection: Optional[str] = None

    def __post_init__(self):
        if self.astrometry is not None:
            assert len(self.astrometry) <= STRING100
        if self.fitOrder is not None:
            assert len(self.fitOrder) <= STRING25
        if self.photometry is not None:
            assert len(self.photometry) <= STRING100
        if self.objectDetection is not None:
            assert len(self.objectDetection) <= STRING100


@dataclass
class ObsContext:
    observatory: ObservatoryObsContext
    submitter: SubmitterObsContext
    observers: list[str]
    measurers: list[str]
    telescope: TelescopeObsContext
    software: Optional[SoftwareObsContext] = None
    coinvestigators: Optional[list[str]] = None
    collaborators: Optional[list[str]] = None
    fundingSource: Optional[str] = None
    comments: Optional[list[str]] = None

    def __post_init__(self):
        assert len(self.observers) > 0
        for observer in self.observers:
            assert len(observer) <= STRING100
        assert len(self.measurers) > 0
        for measurer in self.measurers:
            assert len(measurer) <= STRING100
        if self.coinvestigators is not None:
            assert len(self.coinvestigators) > 0
            for coinvestigator in self.coinvestigators:
                assert len(coinvestigator) <= STRING100
        if self.collaborators is not None:
            assert len(self.collaborators) > 0
            for collaborator in self.collaborators:
                assert len(collaborator) <= STRING100
        if self.fundingSource is not None:
            assert len(self.fundingSource) <= STRING100
        if self.comments is not None:
            for comment in self.comments:
                assert len(comment) <= STRING100

    def to_string(self) -> str:
        lines = []
        for k, v in asdict(self).items():
            if isinstance(v, dict):
                lines.append(f"# {k}")
                for k2, v2 in v.items():
                    if v2 is not None:
                        lines.append(f"! {k2} {v2}")
            else:
                if v is not None:
                    if k in [
                        "observers",
                        "measurers",
                        "coinvestigators",
                        "collaborators",
                    ]:
                        lines.append(f"# {k}")
                        for name in v:
                            lines.append(f"! name {name}")
                    elif k == "fundingSource":
                        lines.append(f"# fundingSource {v}")
                    elif k == "comments":
                        if len(v) > 0:
                            lines.append("# comment")
                            for comment in v:
                                lines.append(f"! line {comment}")
        return "\n".join(lines) + "\n"


class ADESObservations(qv.Table):

    permID = qv.LargeStringColumn(nullable=True)
    provID = qv.LargeStringColumn(nullable=True)
    artSat = qv.LargeStringColumn(nullable=True)
    trkSub = qv.LargeStringColumn(nullable=True)
    obsID = qv.LargeStringColumn(nullable=True)
    obsSubID = qv.LargeStringColumn(nullable=True)
    trkID = qv.LargeStringColumn(nullable=True)
    trkMPC = qv.LargeStringColumn(nullable=True)
    obsTime = Timestamp.as_column()
    rmsTime = qv.Float64Column(nullable=True)
    # Optional ADES time precision/uncertainty fields
    uncTime = qv.LargeStringColumn(nullable=True)
    precTime = qv.Int64Column(nullable=True)
    ra = qv.Float64Column()
    dec = qv.Float64Column()
    # Optional ADES per-axis precision metadata
    precRA = qv.LargeStringColumn(nullable=True)
    precDec = qv.LargeStringColumn(nullable=True)
    # Optional ADES reference star and offset/geometry fields (optical)
    raStar = qv.Float64Column(nullable=True)
    decStar = qv.Float64Column(nullable=True)
    deltaRA = qv.Float64Column(nullable=True)
    deltaDec = qv.Float64Column(nullable=True)
    dist = qv.Float64Column(nullable=True)
    pa = qv.Float64Column(nullable=True)
    # ADES uses arcseconds for rmsRA and rmsDec
    # rmsRA is also multiplied by cos(dec)
    rmsRACosDec = qv.Float64Column(
        nullable=True,
        validator=qv.validators.and_(qv.validators.ge(10e-8), qv.validators.lt(1e2)),
    )
    rmsDec = qv.Float64Column(
        nullable=True,
        validator=qv.validators.and_(qv.validators.ge(10e-8), qv.validators.lt(1e2)),
    )
    rmsCorr = qv.Float64Column(nullable=True)
    # Optional ADES scatter for geometry/photometry
    rmsDist = qv.Float64Column(nullable=True)
    rmsPA = qv.Float64Column(nullable=True)
    mag = qv.Float64Column(nullable=True)
    rmsMag = qv.Float64Column(nullable=True)
    band = qv.LargeStringColumn(nullable=True)
    fltr = qv.LargeStringColumn(nullable=True)
    stn = qv.LargeStringColumn()
    mode = qv.LargeStringColumn()
    astCat = qv.LargeStringColumn()
    photCat = qv.LargeStringColumn(nullable=True)
    photAp = qv.LargeStringColumn(nullable=True)
    logSNR = qv.Float64Column(nullable=True)
    seeing = qv.Float64Column(nullable=True)
    exp = qv.Float64Column(nullable=True)
    rmsFit = qv.Float64Column(nullable=True)
    nucMag = qv.Float64Column(nullable=True)
    # Optional ADES generic position/velocity/covariance and coordinate type
    ctr = qv.Int64Column(nullable=True)
    pos1 = qv.Float64Column(nullable=True)
    pos2 = qv.Float64Column(nullable=True)
    pos3 = qv.Float64Column(nullable=True)
    poscov11 = qv.Float64Column(nullable=True)
    poscov12 = qv.Float64Column(nullable=True)
    poscov13 = qv.Float64Column(nullable=True)
    poscov22 = qv.Float64Column(nullable=True)
    poscov23 = qv.Float64Column(nullable=True)
    poscov33 = qv.Float64Column(nullable=True)
    vel1 = qv.Float64Column(nullable=True)
    vel2 = qv.Float64Column(nullable=True)
    vel3 = qv.Float64Column(nullable=True)
    # Optional ADES radar fields
    delay = qv.Float64Column(nullable=True)
    rmsDelay = qv.Float64Column(nullable=True)
    doppler = qv.Float64Column(nullable=True)
    rmsDoppler = qv.Float64Column(nullable=True)
    frq = qv.Float64Column(nullable=True)
    com = qv.Int64Column(nullable=True)
    nStars = qv.Int64Column(nullable=True)
    prog = qv.LargeStringColumn(nullable=True)
    ref = qv.LargeStringColumn(nullable=True)
    disc = qv.LargeStringColumn(nullable=True)
    subFrm = qv.LargeStringColumn(nullable=True)
    subFmt = qv.LargeStringColumn(nullable=True)
    notes = qv.LargeStringColumn(nullable=True)
    shapeOcc = qv.LargeStringColumn(nullable=True)
    trx = qv.LargeStringColumn(nullable=True)
    rcv = qv.LargeStringColumn(nullable=True)
    sys = qv.LargeStringColumn(nullable=True)
    # Optional ADES observatory context
    obsCenter = qv.LargeStringColumn(nullable=True)
    remarks = qv.LargeStringColumn(nullable=True)


def _keyword_record_tokens(line: str) -> list[str] | None:
    """
    Return PSV Keyword Record tokens if the line matches the ADES PSV rule.

    ADES_Description.pdf (Section 5, "ADES in PSV") defines a Keyword Record as a
    pipe-separated listing of element names where, after stripping, every token
    starts with a lower-case letter [a-z].

    Additionally, we require a "minimum viable" Keyword Record for robust parsing:
    - At least one Identification Group column is present
    - obsTime is present
    """
    if line.startswith("#") or line.startswith("!"):
        return None
    if "|" not in line:
        return None

    tokens = [t.strip() for t in line.split("|")]
    if not tokens:
        return None
    if any(t == "" for t in tokens):
        return None
    if not all(_KEYWORD_TOKEN_RE.match(t) for t in tokens):
        return None
    if not any(t in _IDENTIFICATION_GROUP_COLUMNS for t in tokens):
        return None
    if not all(req in tokens for req in _KEYWORD_REQUIRED_COLUMNS):
        return None
    return tokens


def _candidate_keyword_record_tokens(line: str) -> list[str] | None:
    """
    Heuristic: return pipe-separated tokens that *look* like a Keyword Record.

    This is intentionally looser than `_keyword_record_tokens()` so that we can
    surface helpful diagnostics (e.g. token casing) instead of silently treating
    a malformed header as data.
    """
    if line.startswith("#") or line.startswith("!"):
        return None
    if "|" not in line:
        return None

    tokens = [t.strip() for t in line.split("|")]
    if not tokens:
        return None
    if any(t == "" for t in tokens):
        return None
    if not all(_KEYWORD_TOKEN_CANDIDATE_RE.match(t) for t in tokens):
        return None
    return tokens


def find_ades_psv_problems(
    ades_string: str,
    *,
    goal: ADESValidationGoal = "spec_compliance",
) -> list[ADESValidationProblem]:
    """
    Find validation problems in an ADES PSV string without attempting to parse it.

    This is intended as a user-facing diagnostic to identify issues such as
    malformed Keyword Records (headers), unknown columns, and required ordering.
    """
    problems: list[ADESValidationProblem] = []
    known_columns = set(ADESObservations.empty().table.column_names)

    lines = [ln.rstrip("\n") for ln in ades_string.splitlines()]
    current_header: list[str] | None = None

    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("!"):
            continue

        tokens = _keyword_record_tokens(line)
        if tokens is not None:
            current_header = tokens

            # Spec: identification group elements must be first in the Keyword Record.
            id_cols = set(_IDENTIFICATION_GROUP_COLUMNS)
            seen_non_id = False
            for t in tokens:
                if t in id_cols:
                    if seen_non_id:
                        problems.append(
                            ADESValidationProblem(
                                code="keyword_record_identification_not_first",
                                severity="error",
                                message=(
                                    "Identification Group elements must be the first "
                                    "columns in a Keyword Record."
                                ),
                                line_number=line_number,
                                line=raw,
                            )
                        )
                        break
                else:
                    seen_non_id = True

            if not any(t in id_cols for t in tokens):
                problems.append(
                    ADESValidationProblem(
                        code="keyword_record_missing_identification",
                        severity="error",
                        message=(
                            "Keyword Record is missing Identification Group elements "
                            f"({', '.join(_IDENTIFICATION_GROUP_COLUMNS)})."
                        ),
                        line_number=line_number,
                        line=raw,
                    )
                )

            unknown = [t for t in tokens if t not in known_columns]
            if unknown:
                problems.append(
                    ADESValidationProblem(
                        code="keyword_record_unknown_columns",
                        severity="warning",
                        message=f"Unknown ADES columns will be ignored: {unknown}",
                        line_number=line_number,
                        line=raw,
                    )
                )
            continue

        candidate_tokens = _candidate_keyword_record_tokens(line)
        if candidate_tokens is not None:
            # Keyword Record tokens must start with a lowercase letter [a-z].
            non_lower = [t for t in candidate_tokens if not _KEYWORD_TOKEN_RE.match(t)]
            if non_lower:
                canon_by_casefold = {c.casefold(): c for c in known_columns}
                suggestions: list[str] = []
                for t in non_lower:
                    canon = canon_by_casefold.get(t.casefold())
                    if canon is not None and canon != t:
                        suggestions.append(f"{t} -> {canon}")

                msg = (
                    "Keyword Record tokens must start with a lowercase letter. "
                    f"Non-conforming tokens: {non_lower}."
                )
                if suggestions:
                    msg += f" Suggested fixes: {suggestions}."

                problems.append(
                    ADESValidationProblem(
                        code="keyword_record_tokens_not_lowercase",
                        severity="error",
                        message=msg,
                        line_number=line_number,
                        line=raw,
                    )
                )

            # Minimum viable requirements: ID + obsTime.
            lower = [t.casefold() for t in candidate_tokens]
            id_lower = {c.casefold() for c in _IDENTIFICATION_GROUP_COLUMNS}
            if not any(t in id_lower for t in lower):
                problems.append(
                    ADESValidationProblem(
                        code="keyword_record_missing_identification",
                        severity="error",
                        message=(
                            "Keyword Record must include at least one Identification Group "
                            f"column ({', '.join(_IDENTIFICATION_GROUP_COLUMNS)})."
                        ),
                        line_number=line_number,
                        line=raw,
                    )
                )
            if "obstime" not in lower:
                problems.append(
                    ADESValidationProblem(
                        code="keyword_record_missing_obstime",
                        severity="error",
                        message="Keyword Record must include obsTime.",
                        line_number=line_number,
                        line=raw,
                    )
                )

        # If we have no Keyword Record yet, we can't reliably interpret a data row.
        if current_header is None:
            problems.append(
                ADESValidationProblem(
                    code="data_record_before_keyword_record",
                    severity="error",
                    message="Found a data record before any Keyword Record.",
                    line_number=line_number,
                    line=raw,
                )
            )
            continue

        values = raw.split("|")
        if len(values) != len(current_header):
            problems.append(
                ADESValidationProblem(
                    code="data_record_wrong_field_count",
                    severity="warning",
                    message=(
                        f"Data record has {len(values)} fields but Keyword Record has "
                        f"{len(current_header)} columns."
                    ),
                    line_number=line_number,
                    line=raw,
                )
            )

    return problems


def _policy_seconds_precision_max(goal: ADESValidationGoal) -> int:
    # ADES_Description.pdf: obsTime may have no more than 6 digits after decimal.
    return 6


def _validate_writer_settings(
    *,
    goal: ADESValidationGoal,
    enforcement: ADESValidationEnforcement,
    seconds_precision: int,
) -> tuple[int, list[ADESValidationProblem]]:
    problems: list[ADESValidationProblem] = []
    max_seconds_precision = _policy_seconds_precision_max(goal)

    if seconds_precision > max_seconds_precision:
        if enforcement == "autofix":
            problems.append(
                ADESValidationProblem(
                    code="seconds_precision_clamped",
                    severity="warning",
                    message=(
                        f"seconds_precision={seconds_precision} exceeds the {goal} cap "
                        f"({max_seconds_precision}); clamping to {max_seconds_precision}."
                    ),
                )
            )
            seconds_precision = max_seconds_precision
        else:
            problems.append(
                ADESValidationProblem(
                    code="seconds_precision_exceeds_cap",
                    severity="error" if enforcement == "error" else "warning",
                    message=(
                        f"seconds_precision={seconds_precision} exceeds the {goal} cap "
                        f"({max_seconds_precision})."
                    ),
                )
            )

    return seconds_precision, problems


def validate_ades_observations_values(
    observations: ADESObservations,
    *,
    goal: ADESValidationGoal,
    enforcement: ADESValidationEnforcement,
) -> list[ADESValidationProblem]:
    problems: list[ADESValidationProblem] = []

    # Spec: mag restricted to [-5, +35].
    mag = observations.mag
    mag_is_value = pc.is_valid(mag)
    mag_is_finite = pc.is_finite(mag)
    mag_too_low = pc.less(mag, -5.0)
    mag_too_high = pc.greater(mag, 35.0)
    mag_bad = pc.and_(
        pc.and_(mag_is_value, mag_is_finite),
        pc.or_(mag_too_low, mag_too_high),
    )
    if pc.any(mag_bad).as_py():
        problems.append(
            ADESValidationProblem(
                code="mag_out_of_range",
                severity="error" if enforcement == "error" else "warning",
                message="Found mag values outside [-5, +35].",
            )
        )

    # Spec: presently only ctr=399 is allowed.
    ctr = observations.ctr
    ctr_bad = pc.and_(pc.is_valid(ctr), pc.not_equal(ctr, 399))
    if pc.any(ctr_bad).as_py():
        problems.append(
            ADESValidationProblem(
                code="ctr_not_399",
                severity="error" if enforcement == "error" else "warning",
                message="Found ctr values other than 399 (geocenter).",
            )
        )

    return problems


def ADES_to_string(
    observations: ADESObservations,
    obs_contexts: dict[str, ObsContext] | None = None,
    seconds_precision: int = 3,
    columns_precision: dict[str, int] = {
        # Astrometry
        "ra": 9,
        "dec": 9,
        "raStar": 9,
        "decStar": 9,
        "rmsRACosDec": 5,
        "rmsDec": 5,
        "rmsCorr": 8,
        # Optical geometry/offsets
        "deltaRA": 4,
        "deltaDec": 4,
        "dist": 4,
        "pa": 2,
        "rmsDist": 4,
        "rmsPA": 4,
        # Photometry
        "mag": 4,
        "rmsMag": 4,
        "rmsFit": 4,
        "nucMag": 1,
        # Exposure and SNR/seeing
        "exp": 2,
        "logSNR": 2,
        "seeing": 2,
        # Generic position/velocity and covariance
        "pos1": 6,
        "pos2": 6,
        "pos3": 6,
        "vel1": 6,
        "vel2": 6,
        "vel3": 6,
        "poscov11": 6,
        "poscov12": 6,
        "poscov13": 6,
        "poscov22": 6,
        "poscov23": 6,
        "poscov33": 6,
        # Radar
        "delay": 6,
        "rmsDelay": 6,
        "doppler": 5,
        "rmsDoppler": 5,
        "frq": 6,
    },
    *,
    goal: ADESValidationGoal = "spec_compliance",
    enforcement: ADESValidationEnforcement = "warn",
) -> str:
    """
    Write ADES observations to a string.

    Parameters
    ----------
    observations : ADESObservations
        The observations to write to a string.
    obs_contexts : dict[str, ObsContext] or None, optional
        A dictionary of observatory codes and their corresponding ObsContexts to use
        as the context headers for the different observatory codes in the observations.
        If None, the observatory context headers will be omitted from the output.
        Default is None.
    seconds_precision : int, optional
        The precision to use for the seconds in the obsTime field, by default 3.
    columns_precision : dict[str, int], optional
        A dictionary of column names and their corresponding precision to use when writing
        the observations to the file, by default {
            "ra": 8,
            "dec": 8,
            "rmsRACosDec": 4,
            "rmsDec": 4,
            "mag": 2,
            "rmsMag": 2,
        }
        The MPC enforces strict limits on these and submitters may need permission to send
        high-precision data.

    Returns
    -------
    ades_string : str
        The ADES observations as a string.
    """
    seconds_precision, setting_problems = _validate_writer_settings(
        goal=goal, enforcement=enforcement, seconds_precision=seconds_precision
    )
    value_problems = validate_ades_observations_values(
        observations, goal=goal, enforcement=enforcement
    )
    problems = setting_problems + value_problems
    if problems:
        errors = [p for p in problems if p.severity == "error"]
        if errors and enforcement == "error":
            msg = "\n".join(f"- {p.code}: {p.message}" for p in errors)
            raise ValueError(f"ADES validation failed:\n{msg}")
        for p in problems:
            if p.severity == "warning" or enforcement != "error":
                logger.warning("%s: %s", p.code, p.message)

    ades_string = "# version=2022\n"

    unique_observatories = observations.stn.unique().to_numpy(zero_copy_only=False)
    unique_observatories.sort()

    for obs in unique_observatories:
        if obs_contexts is not None and obs not in obs_contexts:
            raise ValueError(f"Observatory {obs} not found in obs_contexts")

        observations_obscode = observations.select("stn", obs)
        observations_obscode = observations_obscode.sort_by(
            [
                ("artSat", "ascending"),
                ("provID", "ascending"),
                ("permID", "ascending"),
                ("trkSub", "ascending"),
                ("obsTime.days", "ascending"),
                ("obsTime.nanos", "ascending"),
            ]
        )

        id_present = False
        if not pc.all(pc.is_null(observations_obscode.permID)).as_py():
            id_present = True
        if not pc.all(pc.is_null(observations_obscode.provID)).as_py():
            id_present = True
        if not pc.all(pc.is_null(observations_obscode.artSat)).as_py():
            id_present = True
        if not pc.all(pc.is_null(observations_obscode.trkSub)).as_py():
            id_present = True

        if not id_present:
            err = (
                "At least one of permID, provID, artSat, or trkSub should\n"
                "be present in observations."
            )
            raise ValueError(err)

        # Write the observatory context block (if provided)
        if obs_contexts is not None and obs in obs_contexts:
            obs_context = obs_contexts[obs]
            ades_string += obs_context.to_string()

        # Write the observations block (we first convert
        # to a pandas dataframe)
        ades = observations_obscode.to_dataframe()

        # Convert the timestamp to ISOT with the desired precision
        observation_times = Time(
            observations_obscode.obsTime.rescale("utc")
            .mjd()
            .to_numpy(zero_copy_only=False),
            format="mjd",
            precision=seconds_precision,
        )
        obs_time_idx = (
            int(ades.columns.get_loc("obsTime.days"))
            if "obsTime.days" in ades.columns
            else len(ades.columns)
        )
        ades.insert(
            obs_time_idx,
            "obsTime",
            np.array([i + "Z" for i in observation_times.utc.isot]),
        )
        ades.drop(columns=["obsTime.days", "obsTime.nanos"], inplace=True)

        ades.dropna(how="all", axis=1, inplace=True)

        # Change the precision of some of the columns to conform
        # to MPC standards
        for col, prec_col in columns_precision.items():
            if col in ades.columns:
                ades[col] = [
                    f"{i:.{prec_col}f}" if (i is not None and not np.isnan(i)) else ""
                    for i in ades[col]
                ]

        # Rename the columns to match the ADES format
        ades.rename(columns={"rmsRACosDec": "rmsRA"}, inplace=True)

        # Enforce specific ADES-required relative column ordering when present.
        def _enforce_group_order(df, group):
            present = [c for c in group if c in df.columns]
            if len(present) <= 1:
                return df
            present_set = set(present)
            new_cols = []
            inserted = False
            for col in df.columns:
                if col in present_set:
                    if not inserted:
                        new_cols.extend(present)
                        inserted = True
                    # skip subsequent present columns
                    continue
                new_cols.append(col)
            return df[new_cols]

        # Groups based on ADES spec conventions
        groups = [
            ["permID", "provID", "artSat", "trkSub"],
            ["obsTime", "rmsTime", "precTime", "uncTime"],
            ["ra", "dec", "rmsRA", "rmsDec", "rmsCorr"],
            ["raStar", "decStar"],
            ["deltaRA", "deltaDec"],
            ["dist", "pa", "rmsDist", "rmsPA"],
            [
                "sys",
                "ctr",
                "pos1",
                "pos2",
                "pos3",
                "vel1",
                "vel2",
                "vel3",
                "poscov11",
                "poscov12",
                "poscov13",
                "poscov22",
                "poscov23",
                "poscov33",
            ],
            ["trx", "rcv", "frq", "delay", "rmsDelay", "doppler", "rmsDoppler"],
            ["mag", "rmsMag", "band", "fltr", "photAp", "photCat", "nucMag", "rmsFit"],
            ["logSNR", "seeing", "exp"],
            ["nStars", "com"],
            ["prog", "ref", "disc", "subFrm", "subFmt"],
            ["stn", "mode", "astCat", "obsCenter"],
            ["notes", "remarks"],
        ]
        for group in groups:
            ades = _enforce_group_order(ades, group)

        ades_string += ades.to_csv(
            sep="|", header=True, index=False, float_format="%.16f", na_rep=""
        )

    return ades_string


def _data_dict_to_table(data_dict: dict[str, list[str]]) -> ADESObservations:
    if not data_dict:
        return ADESObservations.empty()

    # Get the set of known columns from ADESObservations
    known_columns = set(ADESObservations.empty().table.column_names)
    # Check for unknown columns
    unknown_columns = set(data_dict.keys()) - known_columns
    if unknown_columns:
        logger.warning(
            f"Found unknown ADES columns that will be ignored: {unknown_columns}"
        )

    # Convert every value that is empty string or whitespace to None
    for col in data_dict:
        data_dict[col] = [None if x == "" or x.isspace() else x for x in data_dict[col]]

    numeric_cols = [
        # Astrometry and errors
        "ra",
        "dec",
        "rmsRA",
        "rmsDec",
        "rmsCorr",
        "rmsTime",
        # Optical geometry/offsets
        "raStar",
        "decStar",
        "deltaRA",
        "deltaDec",
        "dist",
        "pa",
        "rmsDist",
        "rmsPA",
        # Photometry/exposure/SNR
        "mag",
        "rmsMag",
        "rmsFit",
        "nucMag",
        "logSNR",
        "seeing",
        "exp",
        # Generic pos/vel/covariance
        "pos1",
        "pos2",
        "pos3",
        "vel1",
        "vel2",
        "vel3",
        "poscov11",
        "poscov12",
        "poscov13",
        "poscov22",
        "poscov23",
        "poscov33",
        # Radar
        "delay",
        "rmsDelay",
        "doppler",
        "rmsDoppler",
        "frq",
    ]
    # Do all the data conversions and then initialize the new table and concatenate
    for col in numeric_cols:
        if col in data_dict:
            data_dict[col] = [
                float(x) if x is not None else None for x in data_dict[col]
            ]

    # Integer columns
    int_cols = [
        "precTime",
        "ctr",
        "com",
        "nStars",
    ]
    for col in int_cols:
        if col in data_dict:
            data_dict[col] = [int(x) if x is not None else None for x in data_dict[col]]

    # Some users are accustomed to having fixed-width columns, so we strip whitespace
    # from all the string columns with the exception of the 'remarks' column
    string_cols = [
        "provID",
        "permID",
        "artSat",
        "trkSub",
        "obsID",
        "obsSubID",
        "trkID",
        "trkMPC",
        "stn",
        "mode",
        "astCat",
        "photCat",
        "band",
        # Additional ADES strings
        "uncTime",
        "precRA",
        "precDec",
        "photAp",
        "fltr",
        "trx",
        "rcv",
        "sys",
        "obsCenter",
        "prog",
        "ref",
        "disc",
        "subFrm",
        "subFmt",
        "notes",
        "shapeOcc",
    ]
    for col in string_cols:
        if col in data_dict:
            data_dict[col] = [
                x.strip() if x is not None else None for x in data_dict[col]
            ]

    if "obsTime" in data_dict:
        # Remove 'Z' from timestamps and convert to MJD
        times = [t[:-1] if t is not None else None for t in data_dict["obsTime"]]
        data_dict["obsTime"] = Timestamp.from_iso8601(times, scale="utc")

    # Update the rmsRACosDec column name to be simply rmsRA
    if "rmsRA" in data_dict:
        data_dict["rmsRACosDec"] = data_dict["rmsRA"]
        data_dict.pop("rmsRA")

    # Only keep keys that are in ADESObservations
    data_dict = {k: v for k, v in data_dict.items() if k in known_columns}

    return ADESObservations.from_kwargs(**data_dict)


def ADES_string_to_tables(
    ades_string: str,
    *,
    goal: ADESValidationGoal = "spec_compliance",
    enforcement: ADESValidationEnforcement = "warn",
) -> Tuple[dict[str, ObsContext], ADESObservations]:
    """
    Parse an ADES format string into ObsContext and ADESObservations objects.

    Parameters
    ----------
    ades_string : str
        The ADES format string to parse.

    Returns
    -------
    tuple[dict[str, ObsContext], ADESObservations]
        A tuple containing:
        - A dictionary mapping observatory codes to their ObsContext objects
        - An ADESObservations table containing the observations
    """
    psv_problems = find_ades_psv_problems(ades_string, goal=goal)
    if psv_problems:
        errors = [p for p in psv_problems if p.severity == "error"]
        if errors and enforcement == "error":
            msg = "\n".join(
                f"- line {p.line_number}: {p.code}: {p.message}" for p in errors
            )
            raise ValueError(f"ADES PSV validation failed:\n{msg}")
        for p in psv_problems:
            logger.warning("%s: %s", p.code, p.message)

    # Split the string into lines and remove empty lines
    lines = [line.strip() for line in ades_string.split("\n") if line.strip()]

    # Start by parsing the data lines
    current_data = {}
    observations = ADESObservations.empty()
    for line in lines:
        # Skip over metadata lines
        if line.startswith("#") or line.startswith("!"):
            continue

        keyword_tokens = _keyword_record_tokens(line)
        if keyword_tokens is not None:
            observations = qv.concatenate(
                [observations, _data_dict_to_table(current_data)]
            )
            new_headers = keyword_tokens
            current_data = {header: [] for header in new_headers}
            continue

        # Add the data line to the current data dictionary
        if not current_data:
            # No keyword record found yet. The diagnostic helper will report this.
            continue
        data_line = line.split("|")
        for header, value in zip(current_data.keys(), data_line):
            current_data[header].append(value)

    # Add the last data dictionary to the observations table
    observations = qv.concatenate([observations, _data_dict_to_table(current_data)])

    obs_problems = validate_ades_observations_values(
        observations, goal=goal, enforcement=enforcement
    )
    if obs_problems:
        errors = [p for p in obs_problems if p.severity == "error"]
        if errors and enforcement == "error":
            msg = "\n".join(f"- {p.code}: {p.message}" for p in errors)
            raise ValueError(f"ADES validation failed:\n{msg}")
        for p in obs_problems:
            logger.warning("%s: %s", p.code, p.message)

    # Now we parse the metadata sections
    # Initialize variables
    obs_contexts = {}
    current_obs_context = {}
    current_section_key = None
    current_context_code = None

    for line in lines:
        if line.startswith("#"):
            line = line[1:].strip()
            if line.startswith("version"):
                continue
            current_section_key, *value = [x.strip() for x in line.split(" ")]
            if current_section_key == "observatory":
                # Only build obs context if current_obs_context is not empty
                if current_obs_context:
                    obs_contexts[current_context_code] = _build_obs_context(
                        current_obs_context
                    )
                current_obs_context = {}
                current_context_code = None

            # Some sections specify the value in the same line as the section key
            if len(value) > 0:
                current_obs_context[current_section_key] = " ".join(value)
            continue

        if line.startswith("!"):
            line = line[1:].strip()
            key, *value = [x.strip() for x in line.split(" ")]
            value = " ".join(value)
            if key == "mpcCode":
                current_context_code = value
            current_section = current_obs_context.setdefault(current_section_key, {})
            if current_section_key in ["observers", "measurers", "comment"]:
                current_key_values = current_section.setdefault(key, [])
                current_key_values.append(value)
            else:
                current_section[key] = value

    if current_obs_context:
        obs_contexts[current_context_code] = _build_obs_context(current_obs_context)

    return obs_contexts, observations


def _build_obs_context(context_dict: dict) -> ObsContext:
    """Helper function to build an ObsContext from parsed dictionary data."""
    # Extract observatory data
    observatory = ObservatoryObsContext(
        mpcCode=context_dict["observatory"]["mpcCode"],
        name=context_dict["observatory"].get("name"),
    )

    # Extract submitter data
    submitter = SubmitterObsContext(
        name=context_dict["submitter"]["name"],
        institution=context_dict["submitter"].get("institution"),
    )

    # Extract telescope data
    telescope_data = context_dict.get("telescope", {})
    telescope = TelescopeObsContext(
        name=telescope_data.get("name", None),
        design=telescope_data["design"],
        aperture=float(telescope_data["aperture"]),
        detector=telescope_data.get("detector"),
        fRatio=float(telescope_data["fRatio"]) if "fRatio" in telescope_data else None,
        filter=telescope_data.get("filter"),
        arraySize=telescope_data.get("arraySize"),
        pixelSize=(
            float(telescope_data["pixelSize"])
            if "pixelSize" in telescope_data
            else None
        ),
    )

    # Extract software data
    software_data = context_dict.get("software", {})
    software = None
    if software_data:
        software = SoftwareObsContext(
            astrometry=software_data.get("astrometry"),
            fitOrder=software_data.get("fitOrder"),
            photometry=software_data.get("photometry"),
            objectDetection=software_data.get("objectDetection"),
        )

    # Build ObsContext
    return ObsContext(
        observatory=observatory,
        submitter=submitter,
        observers=context_dict["observers"]["name"],
        measurers=context_dict["measurers"]["name"],
        telescope=telescope,
        software=software,
        fundingSource=context_dict.get("fundingSource"),
        comments=context_dict.get("comment", {}).get("line", []),
    )
