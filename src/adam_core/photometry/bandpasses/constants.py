from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    weight_C: float
    weight_S: float
    citation: str


# Initial shipped templates. Mixes are defined as linear combinations of the base
# C and S reflectance templates.
TEMPLATE_SPECS: Final[tuple[TemplateSpec, ...]] = (
    TemplateSpec(
        template_id="C",
        weight_C=1.0,
        weight_S=0.0,
        citation="Bus–DeMeo taxonomy (C-type) — simplified reflectance template (vendored).",
    ),
    TemplateSpec(
        template_id="S",
        weight_C=0.0,
        weight_S=1.0,
        citation="Bus–DeMeo taxonomy (S-type) — simplified reflectance template (vendored).",
    ),
    TemplateSpec(
        template_id="NEO",
        weight_C=0.5,
        weight_S=0.5,
        citation="NEO population mix (assumed): 50% C / 50% S.",
    ),
    TemplateSpec(
        template_id="MBA",
        weight_C=0.7,
        weight_S=0.3,
        citation="Main-belt population mix (assumed): 70% C / 30% S.",
    ),
)


# MPC observatory codes to treat as ATLAS in v1.
ATLAS_MPC_CODES: Final[tuple[str, ...]] = ("T08", "T05", "M22", "W68")
