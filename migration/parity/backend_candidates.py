"""Backend/transport implementation candidates for parity/perf diagnostics.

Retired (bead personal-cmy.36.10): the temporary ``bridge.*`` Arrow-IPC
workflow candidates were removed once their coverage was absorbed by
canonical public-API lanes:

* ``bridge.propagate_orbits_2body`` -> ``dynamics.propagate_2body``
  (Arrow-native public facade, bead personal-cmy.36.4).
* ``bridge.rotate_orbits_frame`` -> ``coordinates.transform_coordinates``
  (Arrow-native public facade, bead personal-cmy.36.3).
* ``bridge.sample_orbit_variants`` -> ``orbits.VariantOrbits.create``
  (promoted to a canonical public lane; the Rust sampler is the public
  backend).
* ``bridge.evaluate_residuals_2body`` -> canonical residual/OD coverage
  (``coordinates.residuals.Residuals.calculate``, bead personal-cmy.36.9,
  plus ``dynamics.generate_ephemeris_2body``, bead personal-cmy.36.5).

Historical parity artifacts before process version
``rm-p1-023-canonical-variant-create-v1`` still contain ``bridge.*`` rows;
this module retains the machinery so those artifacts remain interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class BackendCandidate:
    """A non-public implementation candidate measured by the parity harness."""

    candidate_id: str
    canonical_name: str
    implementation_label: str
    boundary: str
    rust_module: str
    legacy_comparator: str
    note: str
    canonical_api_id: str | None = None

    def to_json(self) -> dict[str, str | None]:
        return {
            "candidate_id": self.candidate_id,
            "canonical_api_id": self.canonical_api_id,
            "canonical_name": self.canonical_name,
            "implementation_label": self.implementation_label,
            "boundary": self.boundary,
            "rust_module": self.rust_module,
            "legacy_comparator": self.legacy_comparator,
            "note": self.note,
        }


BACKEND_CANDIDATES: Final[tuple[BackendCandidate, ...]] = ()


BACKEND_CANDIDATES_BY_ID: Final[dict[str, BackendCandidate]] = {
    candidate.candidate_id: candidate for candidate in BACKEND_CANDIDATES
}


def get(api_id: str) -> BackendCandidate | None:
    return BACKEND_CANDIDATES_BY_ID.get(api_id)


def is_candidate(api_id: str) -> bool:
    return api_id in BACKEND_CANDIDATES_BY_ID
