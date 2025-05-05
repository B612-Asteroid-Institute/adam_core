from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

from .lua import LuaDict
from .translation import Translation


class RenderBinMode(Enum):
    BACKGROUND = "Background"
    OPAQUE = "Opaque"
    PREDEFERREDTRANSPARENT = "PreDeferredTransparent"
    OVERLAY = "Overlay"
    POSTDEFERREDTRANSPARENT = "PostDeferredTransparent"
    STICKER = "Sticker"


class RenderableOrbitalKeplerFormat(Enum):
    TLE = "TLE"
    OMM = "OMM"
    SBDB = "SBDB"


class RenderableOrbitalKeplerRendering(Enum):
    TRAIL = "Trail"
    POINT = "Point"
    POINTS_TRAILS = "PointsTrails"


class RenderableTrailRendering(Enum):
    LINES = "Lines"
    POINTS = "Points"
    LINES_POINTS = "Lines+Points"


@dataclass(kw_only=True)
class Resource(LuaDict):
    """Custom class to point to local resource files"""

    path: str

    def to_string(self, indent: int = 0):
        return f'asset.resource("{self.path}")'


@dataclass(kw_only=True)
class Renderable(LuaDict):
    ### See:https://docs.openspaceproject.com/latest/reference/asset-components/Renderable.html
    type: str

    dim_in_atmosphere: Optional[bool] = None
    enabled: Optional[bool] = None
    opacity: Optional[float] = None
    render_bin_mode: Optional[RenderBinMode] = None
    tag: Optional[Union[str, List[str]]] = None


@dataclass(kw_only=True)
class RenderableOrbitalKepler(Renderable):
    ### See: https://docs.openspaceproject.com/latest/reference/asset-components/RenderableOrbitalKepler.html
    color: Tuple[float, float, float]
    format: RenderableOrbitalKeplerFormat
    path: Union[str, Resource]
    segment_quality: int
    type: str = "RenderableOrbitalKepler"

    contiguous_mode: Optional[bool] = None
    enable_max_size: Optional[bool] = None
    enable_outline: Optional[bool] = None
    max_size: Optional[float] = None
    outline_color: Optional[Tuple[float, float, float]] = None
    outline_width: Optional[float] = None
    point_size_exponent: Optional[float] = None
    rendering: Optional[RenderableOrbitalKeplerRendering] = None
    render_size: Optional[int] = None
    start_render_idx: Optional[int] = None
    trail_fade: Optional[float] = None
    trail_width: Optional[float] = None


@dataclass(kw_only=True)
class RenderableTrailOrbit(Renderable):
    ### See: https://docs.openspaceproject.com/latest/reference/asset-components/RenderableTrailOrbit.html
    color: Tuple[float, float, float]
    period: float  # In days
    resolution: int
    translation: Translation
    type: str = "RenderableTrailOrbit"

    enable_fade: Optional[bool] = None
    line_fade_amount: Optional[float] = None
    line_length: Optional[float] = None
    line_width: Optional[float] = None
    point_size: Optional[int] = None
    rendering: Optional[RenderableTrailRendering] = None


@dataclass(kw_only=True)
class RenderableTrailTrajectory(Renderable):
    ### See: https://docs.openspaceproject.com/latest/reference/asset-components/RenderableTrailTrajectory.html
    color: Tuple[float, float, float]
    end_time: str
    start_time: str
    translation: Translation
    type: str = "RenderableTrailTrajectory"

    accurate_trail_positions: Optional[int] = None
    enable_fade: Optional[bool] = None
    enable_sweep_chunking: Optional[int] = None
    line_fade_amount: Optional[float] = None
    line_length: Optional[float] = None
    line_width: Optional[float] = None
    point_size: Optional[int] = None
    rendering: Optional[RenderableTrailRendering] = None
    sample_interval: Optional[float] = None
    show_full_trail: Optional[bool] = None
    sweep_chunk_size: Optional[int] = None
    time_stamp_subsample_factor: Optional[int] = None
