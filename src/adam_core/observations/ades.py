from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyarrow.compute as pc
import quivr as qv
from astropy.time import Time

from ..time import Timestamp

STRING100 = 100
STRING25 = 25


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
    name: str
    design: str
    aperture: Optional[float] = None
    detector: Optional[str] = None
    fRatio: Optional[float] = None
    filter: Optional[str] = None
    arraySize: Optional[str] = None
    pixelSize: Optional[float] = None

    def __post_init__(self):
        assert len(self.name) <= STRING100
        assert len(self.design) <= STRING25
        if self.aperture is not None:
            assert self.aperture > 0
        if self.detector is not None:
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
            assert len(self.comments) > 0
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
    ra = qv.Float64Column()
    dec = qv.Float64Column()
    rmsRA = qv.Float64Column(nullable=True)
    rmsDec = qv.Float64Column(nullable=True)
    mag = qv.Float64Column(nullable=True)
    rmsMag = qv.Float64Column(nullable=True)
    band = qv.LargeStringColumn(nullable=True)
    stn = qv.LargeStringColumn()
    mode = qv.LargeStringColumn()
    astCat = qv.LargeStringColumn()
    remarks = qv.LargeStringColumn(nullable=True)


def ADES_to_string(
    observations: ADESObservations,
    obs_contexts: dict[str, ObsContext],
    seconds_precision: int = 3,
    columns_precision: dict[str, int] = {
        "ra": 8,
        "dec": 8,
        "rmsRA": 4,
        "rmsDec": 4,
        "mag": 2,
        "rmsMag": 2,
    },
) -> str:
    """
    Write ADES observations to a string.

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
            "rmsRA" : 4,
            "rmsDec" : 4,
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
    ades_string = "# version=2022\n"

    unique_observatories = observations.stn.unique().to_numpy(zero_copy_only=False)
    unique_observatories.sort()

    for obs in unique_observatories:
        if obs not in obs_contexts:
            raise ValueError(f"Observatory {obs} not found in obs_contexts")

        observations_obscode = observations.select("stn", obs)
        observations_obscode = observations_obscode.sort_by(
            [
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
        if not pc.all(pc.is_null(observations_obscode.trkSub)).as_py():
            id_present = True

        if not id_present:
            err = (
                "At least one of permID, provID, or trkSub should\n"
                "be present in observations."
            )
            raise ValueError(err)

        # Write the observatory context block
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
        ades.insert(
            4,
            "obsTime",
            np.array([i + "Z" for i in observation_times.utc.isot]),
        )
        ades.drop(columns=["obsTime.days", "obsTime.nanos"], inplace=True)

        # Multiply rmsRA by cos(dec) since ADES wants the random component in rmsRAcosDec
        ades.loc[:, "rmsRA"] *= np.cos(np.radians(ades["dec"]))

        # Convert rmsRA and rmsDec to arcseconds
        ades.loc[:, "rmsRA"] *= 3600
        ades.loc[:, "rmsDec"] *= 3600

        ades.dropna(how="all", axis=1, inplace=True)

        # Change the precision of some of the columns to conform
        # to MPC standards
        for col, prec_col in columns_precision.items():
            if col in ades.columns:
                ades[col] = [
                    f"{i:.{prec_col}f}" if i is not None or not np.isnan(i) else ""
                    for i in ades[col]
                ]

        ades_string += ades.to_csv(
            sep="|", header=True, index=False, float_format="%.16f"
        )

    return ades_string
