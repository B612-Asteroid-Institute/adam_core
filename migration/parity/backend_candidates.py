"""Backend/transport implementation candidates for parity/perf diagnostics.

These entries are *not* public API migration rows. They identify temporary
implementation candidates (for example Arrow IPC workflow wrappers) that are
worth parity/speed tracking while we decide whether to promote them behind a
canonical public API name.
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


BACKEND_CANDIDATES: Final[tuple[BackendCandidate, ...]] = (
    BackendCandidate(
        candidate_id="bridge.propagate_orbits_2body",
        canonical_api_id="dynamics.propagate_2body",
        canonical_name="dynamics.propagate_2body",
        implementation_label="Arrow IPC OrbitBatch workflow",
        boundary="python+quivr+arrow-ipc",
        rust_module="adam_core.orbits.arrow_bridge.propagate_orbits_2body",
        legacy_comparator="adam_core.dynamics.propagate_2body",
        note=(
            "Diagnostic candidate for replacing/augmenting the canonical public "
            "propagate_2body path. It is not a separate public API identity."
        ),
    ),
    BackendCandidate(
        candidate_id="bridge.rotate_orbits_frame",
        canonical_api_id="coordinates.transform_coordinates",
        canonical_name="coordinates.transform_coordinates",
        implementation_label="Arrow IPC Orbits frame-rotation workflow",
        boundary="python+quivr+arrow-ipc",
        rust_module="adam_core.orbits.arrow_bridge._rotate_orbits_frame_ipc_candidate",
        legacy_comparator=(
            "adam_core.coordinates.transform.transform_coordinates(..., frame_out=...)"
        ),
        note=(
            "Diagnostic candidate only: this remains a private implementation "
            "detail benchmarked against transform_coordinates semantics, not a "
            "standalone public rotate_orbits_frame API."
        ),
    ),
    BackendCandidate(
        candidate_id="bridge.sample_orbit_variants",
        canonical_api_id=None,
        canonical_name="orbits.VariantOrbits.create",
        implementation_label="Arrow IPC covariance-variant sampler workflow",
        boundary="python+quivr+arrow-ipc",
        rust_module="adam_core.orbits.arrow_bridge.sample_orbit_variants",
        legacy_comparator="adam_core.orbits.variants.VariantOrbits.create",
        note=(
            "Diagnostic candidate for the Rust backend that should ultimately "
            "sit behind VariantOrbits.create if it wins semantics/performance."
        ),
    ),
    BackendCandidate(
        candidate_id="bridge.evaluate_residuals_2body",
        canonical_api_id=None,
        canonical_name="OD residual workflow (canonical public API TBD)",
        implementation_label="Arrow IPC 2-body ephemeris+residual workflow",
        boundary="python+quivr+arrow-ipc",
        rust_module="adam_core.orbits.arrow_bridge.evaluate_residuals_2body",
        legacy_comparator=(
            "adam_core.dynamics.generate_ephemeris_2body + "
            "adam_core.coordinates.residuals.Residuals.calculate"
        ),
        note=(
            "Diagnostic OD inner-loop candidate. Promotion must wait for the "
            "canonical OD public API/signature decision under personal-cmy.7."
        ),
    ),
)


BACKEND_CANDIDATES_BY_ID: Final[dict[str, BackendCandidate]] = {
    candidate.candidate_id: candidate for candidate in BACKEND_CANDIDATES
}


def get(api_id: str) -> BackendCandidate | None:
    return BACKEND_CANDIDATES_BY_ID.get(api_id)


def is_candidate(api_id: str) -> bool:
    return api_id in BACKEND_CANDIDATES_BY_ID
