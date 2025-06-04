from dataclasses import dataclass
from typing import Optional, Union

from .lua import LuaDict


@dataclass(kw_only=True)
class Translation(LuaDict):
    type: str


@dataclass(kw_only=True)
class Transform(LuaDict):
    translation: Translation


@dataclass(kw_only=True)
class KeplerTranslation(Translation):
    ### See: https://docs.openspaceproject.com/latest/reference/asset-components/KeplerTranslation.html
    argument_of_periapsis: float
    ascending_node: float
    eccentricity: float
    epoch: str
    inclination: float
    mean_anomaly: float
    period: float
    semi_major_axis: float
    type: str = "KeplerTranslation"


@dataclass(kw_only=True)
class SpiceTranslation(Translation):
    ### See: https://docs.openspaceproject.com/latest/reference/asset-components/SpiceTranslation.html
    observer: Union[str, int]
    target: Union[str, int]
    fixed_date: Optional[str] = None
    frame: Optional[str] = None
    type: str = "SpiceTranslation"
