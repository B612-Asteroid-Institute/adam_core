import logging
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import quivr as qv

from ..time import Timestamp

STRING100 = 100
STRING25 = 25

logger = logging.getLogger(__name__)


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
    trkSub = qv.LargeStringColumn(nullable=True)
    obsSubID = qv.LargeStringColumn(nullable=True)
    obsTime = Timestamp.as_column()
    rmsTime = qv.Float64Column(nullable=True)
    ra = qv.Float64Column()
    dec = qv.Float64Column()
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
    mag = qv.Float64Column(nullable=True)
    rmsMag = qv.Float64Column(nullable=True)
    band = qv.LargeStringColumn(nullable=True)
    stn = qv.LargeStringColumn()
    mode = qv.LargeStringColumn()
    astCat = qv.LargeStringColumn()
    photCat = qv.LargeStringColumn(nullable=True)
    logSNR = qv.Float64Column(nullable=True)
    seeing = qv.Float64Column(nullable=True)
    exp = qv.Float64Column(nullable=True)
    remarks = qv.LargeStringColumn(nullable=True)


def ADES_to_string(
    observations: ADESObservations,
    obs_contexts: dict[str, ObsContext],
    seconds_precision: int = 3,
    columns_precision: dict[str, int] = {
        "ra": 9,
        "dec": 9,
        "rmsRACosDec": 5,
        "rmsDec": 5,
        "rmsCorr": 8,
        "mag": 4,
        "rmsMag": 4,
        "exp": 2,
        "logSNR": 2,
        "seeing": 2,
    },
) -> str:
    """
    Write ADES observations to a string.

    The observation blocks are rendered in the Rust backend (bead
    personal-cmy.20), byte-identically to the legacy Python/pandas writer
    (gated by the frozen legacy fixture in
    ``migration/artifacts/ades_parity_fixture_2026-07-05.json``); the
    ObsContext headers stay Python-rendered.

    Parameters
    ----------
    observations : ADESObservations
        The observations to write to a string.
    obs_contexts : dict[str, ObsContext]
        A dictionary of observatory codes and their corresponding ObsContexts to use
        as the context headers for the different observatory codes in the observations.
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
    from adam_core import _rust_native as _rn

    from .arrow_bridge import observations_to_ipc

    if len(observations) > 0 and observations.obsTime.scale != "utc":
        observations = observations.set_column(
            "obsTime", observations.obsTime.rescale("utc")
        )
    contexts_rendered = {
        code: context.to_string() for code, context in obs_contexts.items()
    }
    return _rn.ades_to_string_ipc(
        observations_to_ipc(observations),
        contexts_rendered,
        seconds_precision,
        {column: int(value) for column, value in columns_precision.items()},
    )


def ADES_string_to_tables(
    ades_string: str,
) -> Tuple[dict[str, ObsContext], ADESObservations]:
    """
    Parse an ADES format string into ObsContext and ADESObservations objects.

    The observation blocks are parsed in the Rust backend (bead
    personal-cmy.20); the ObsContext metadata sections are parsed by the same
    Python code the legacy parser used.

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
    from adam_core import _rust_native as _rn

    from .arrow_bridge import observations_from_ipc

    raw, unknown_columns = _rn.ades_string_to_observations_ipc(ades_string)
    if unknown_columns:
        logger.warning(
            f"Found unknown ADES columns that will be ignored: {set(unknown_columns)}"
        )
    observations = observations_from_ipc(raw, ADESObservations)

    # Split the string into lines and remove empty lines
    lines = [line.strip() for line in ades_string.split("\n") if line.strip()]
    obs_contexts = _parse_obs_contexts(lines)
    return obs_contexts, observations


def _parse_obs_contexts(lines: list[str]) -> dict[str, ObsContext]:
    """Parse the ObsContext metadata sections of an ADES file. This is the
    unchanged legacy metadata loop; only the observation-block parsing moved
    to Rust."""
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

    return obs_contexts


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
