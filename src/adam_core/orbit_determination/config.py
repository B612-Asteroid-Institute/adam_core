"""Complete declarative Core-owned orbit-determination configuration.

Every scientific declaration is serialized canonically and parsed, validated,
and re-emitted by Rust. Python classes provide immutable typed construction;
they do not implement validation or scientific defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, cast

from adam_core import _rust_native as _rust

from .observation_pipeline import ObservationPipeline

LossName = Literal["linear", "huber"]
JacobianName = Literal["auto", "analytic", "propagator", "central", "2-point"]
ObservationSelectionName = Literal[
    "combinations", "first+middle+last", "thirds", "spans"
]
VelocityMethodName = Literal["auto", "gibbs", "herrick+gibbs"]
AcceptanceName = Literal["prefit", "corrected"]
PrefitScopeName = Literal["local", "all"]

OdConfigurationError = _rust.OdConfigurationError
OdResourceValidationError = _rust.OdResourceValidationError


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


_PROFILE_IDENTITY_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class ProfileIdentity:
    """Immutable Rust-validated named-profile identity."""

    name: str
    version: int
    definition_sha256: str

    def __init__(
        self,
        name: str,
        version: int,
        definition_sha256: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _PROFILE_IDENTITY_TOKEN:
            raise TypeError("ProfileIdentity values are created only by Core Rust")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "definition_sha256", definition_sha256)


def _profile_identity_from_native(native: Any) -> ProfileIdentity | None:
    raw = native.profile_identity_json
    if raw is None:
        return None
    value = cast(dict[str, object], json.loads(raw))
    return ProfileIdentity(
        cast(str, value["name"]),
        cast(int, value["version"]),
        cast(str, value["definition_sha256"]),
        _token=_PROFILE_IDENTITY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class ResourceBindings:
    """Concrete local paths keyed by stable semantic resource slot."""

    paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        normalized: dict[str, Path] = {}
        for slot, value in self.paths.items():
            if not isinstance(slot, str):
                raise TypeError("resource slots must be strings")
            normalized[slot] = Path(os.fspath(value))
        object.__setattr__(self, "paths", MappingProxyType(normalized))

    def _native_paths(self) -> dict[str, str]:
        return {slot: os.fspath(path) for slot, path in self.paths.items()}


@dataclass(frozen=True, slots=True)
class TwoBodyFitConfig:
    propagation_max_iter: int
    propagation_tol: float
    lt_tol: float
    ephemeris_max_iter: int
    ephemeris_tol: float
    max_lt_iter: int


@dataclass(frozen=True, slots=True)
class WhitenedFitConfig:
    loss: LossName
    f_scale: float
    jacobian: JacobianName
    validate_covariance: bool
    xtol: float
    ftol: float
    gtol: float
    max_iterations: int
    max_nfev: int | None
    two_body: TwoBodyFitConfig


@dataclass(frozen=True, slots=True)
class SpanSearchConfig:
    target_span_days: float
    min_span_days: float
    max_span_days: float
    dense_window_days: tuple[float, float]
    edge_window_days: float
    max_triplets: int


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    max_corrections: int
    max_converged: int
    growth: float
    min_stage_obs: int
    max_stages: int
    stage_loss: LossName
    prefit_scope: PrefitScopeName


@dataclass(frozen=True, slots=True)
class RangingConfig:
    enabled: bool
    rho_min_au: float
    rho_max_au: float
    n_rho: int
    n_rho_dot: int
    max_corrections: int
    chunk_size: int
    max_arc_days: float


@dataclass(frozen=True, slots=True)
class IodSearchConfig:
    min_obs: int
    min_arc_length: float
    contamination_percentage: float
    rchi2_threshold: float
    observation_selection_method: ObservationSelectionName
    light_time: bool
    mu: float
    speed_of_light: float
    max_triplets: int | None
    max_pair_visits: int
    max_raw_pair_visits: int
    max_middle_visits: int
    velocity_method: VelocityMethodName
    spans: SpanSearchConfig
    acceptance: AcceptanceName
    correction: CorrectionConfig
    ranging: RangingConfig


@dataclass(frozen=True, slots=True)
class IodStageConfig:
    """Complete IOD/search/correction/ranging declaration."""

    search_correction_ranging: IodSearchConfig
    candidate_correction_fit: WhitenedFitConfig
    _native: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        declaration = {
            "search_correction_ranging": asdict(self.search_correction_ranging),
            "candidate_correction_fit": asdict(self.candidate_correction_fit),
        }
        native = _rust.iod_stage_config_from_json(_canonical_json(declaration))
        object.__setattr__(self, "_native", native)

    @classmethod
    def core_v1(cls) -> IodStageConfig:
        """Load the named Core-v1 values from Rust; not an implicit default."""

        return cls._from_native(_rust.IodStageConfig.core_v1())

    @classmethod
    def _from_native(cls, native: Any) -> IodStageConfig:
        raw = json.loads(native.canonical_json)
        search = raw["search_correction_ranging"]
        fit = raw["candidate_correction_fit"]
        value = object.__new__(cls)
        object.__setattr__(
            value, "search_correction_ranging", _iod_search_from_dict(search)
        )
        object.__setattr__(value, "candidate_correction_fit", _fit_from_dict(fit))
        object.__setattr__(value, "_native", native)
        return value


@dataclass(frozen=True, slots=True)
class IodRunConfig:
    """Complete bound pipeline/resource/profile and IOD-stage configuration."""

    observation_pipeline: ObservationPipeline
    iod: IodStageConfig
    resources: ResourceBindings
    profile_identity: ProfileIdentity | None = field(init=False, default=None)
    _native: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observation_pipeline, ObservationPipeline):
            raise TypeError("observation_pipeline must be ObservationPipeline")
        if not isinstance(self.iod, IodStageConfig):
            raise TypeError("iod must be IodStageConfig")
        if not isinstance(self.resources, ResourceBindings):
            raise TypeError("resources must be ResourceBindings")
        descriptors = self.observation_pipeline._descriptors()
        transforms_json = None if descriptors is None else _canonical_json(descriptors)
        object.__setattr__(
            self,
            "_native",
            _rust.explicit_iod_run_config(
                transforms_json, self.resources._native_paths(), self.iod._native
            ),
        )

    @classmethod
    def _from_native(
        cls,
        native: Any,
        *,
        observation_pipeline: ObservationPipeline,
        iod: IodStageConfig,
        resources: ResourceBindings,
    ) -> IodRunConfig:
        value = object.__new__(cls)
        object.__setattr__(value, "observation_pipeline", observation_pipeline)
        object.__setattr__(value, "iod", iod)
        object.__setattr__(value, "resources", resources)
        object.__setattr__(
            value, "profile_identity", _profile_identity_from_native(native)
        )
        object.__setattr__(value, "_native", native)
        return value

    @property
    def search_correction_ranging(self) -> IodSearchConfig:
        return self.iod.search_correction_ranging

    @property
    def candidate_correction_fit(self) -> WhitenedFitConfig:
        return self.iod.candidate_correction_fit

    @property
    def profile_name(self) -> str | None:
        return None if self.profile_identity is None else self.profile_identity.name

    @property
    def profile_definition_sha256(self) -> str | None:
        return (
            None
            if self.profile_identity is None
            else self.profile_identity.definition_sha256
        )

    @property
    def provenance(self) -> Mapping[str, object]:
        value = cast(dict[str, object], json.loads(self._native.provenance_json))
        return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class WorstResidualRefinement:
    method: Literal["worst_residual"]
    rchi2_threshold: float
    min_obs: int
    min_arc_length: float
    contamination_percentage: float
    fit: WhitenedFitConfig


@dataclass(frozen=True, slots=True)
class Cmc2003Refinement:
    method: Literal["cmc2003"]
    chi2_reject: float
    chi2_recover: float
    chi2_frac: float
    max_iterations: int
    max_rejected_fraction: float
    apparition_gap_days: float
    psd_floor_frac: float
    min_kept: int
    fit: WhitenedFitConfig


RefinementDeclaration = WorstResidualRefinement | Cmc2003Refinement


@dataclass(frozen=True, slots=True)
class RefinementRunConfig:
    """Complete fit/Jacobian/covariance/rejection declaration."""

    refinement: RefinementDeclaration
    _native: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        native = _rust.refinement_run_config_from_json(
            _canonical_json(asdict(self.refinement))
        )
        object.__setattr__(self, "_native", native)

    @classmethod
    def core_v1(cls) -> RefinementRunConfig:
        """Load the named Core-v1 values from Rust; not an implicit default."""

        return cls._from_native(_rust.RefinementRunConfig.core_v1())

    @classmethod
    def _from_native(cls, native: Any) -> RefinementRunConfig:
        raw = json.loads(native.canonical_json)
        method = raw["method"]
        fit = _fit_from_dict(raw.pop("fit"))
        if method == "worst_residual":
            declaration: RefinementDeclaration = WorstResidualRefinement(fit=fit, **raw)
        elif method == "cmc2003":
            declaration = Cmc2003Refinement(fit=fit, **raw)
        else:  # pragma: no cover - native/Python version skew guard
            raise RuntimeError(f"unknown native refinement method {method!r}")
        value = object.__new__(cls)
        object.__setattr__(value, "refinement", declaration)
        object.__setattr__(value, "_native", native)
        return value


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Complete covariance and provider-prediction evaluation configuration."""

    covariance_dimension: int
    prediction_fit: WhitenedFitConfig
    _native: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        declaration = {
            "covariance_dimension": self.covariance_dimension,
            "prediction_fit": asdict(self.prediction_fit),
        }
        object.__setattr__(
            self,
            "_native",
            _rust.evaluation_config_from_json(_canonical_json(declaration)),
        )

    @classmethod
    def core_v1(cls) -> EvaluationConfig:
        return cls._from_native(_rust.EvaluationConfig.core_v1())

    @classmethod
    def _from_native(cls, native: Any) -> EvaluationConfig:
        raw = json.loads(native.canonical_json)
        value = object.__new__(cls)
        object.__setattr__(
            value, "covariance_dimension", cast(int, raw["covariance_dimension"])
        )
        object.__setattr__(
            value, "prediction_fit", _fit_from_dict(raw["prediction_fit"])
        )
        object.__setattr__(value, "_native", native)
        return value


def _fit_from_dict(raw: Mapping[str, Any]) -> WhitenedFitConfig:
    values = dict(raw)
    values["two_body"] = TwoBodyFitConfig(**values["two_body"])
    return WhitenedFitConfig(**values)


def _iod_search_from_dict(raw: Mapping[str, Any]) -> IodSearchConfig:
    values = dict(raw)
    spans = dict(values["spans"])
    spans["dense_window_days"] = tuple(spans["dense_window_days"])
    values["spans"] = SpanSearchConfig(**spans)
    values["correction"] = CorrectionConfig(**values["correction"])
    values["ranging"] = RangingConfig(**values["ranging"])
    return IodSearchConfig(**values)


@dataclass(frozen=True, slots=True)
class OdConfig:
    """Complete bound IOD-plus-refinement configuration."""

    observation_pipeline: ObservationPipeline
    iod: IodStageConfig
    refinement: RefinementRunConfig
    resources: ResourceBindings
    profile_identity: ProfileIdentity | None = field(init=False, default=None)
    _native: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observation_pipeline, ObservationPipeline):
            raise TypeError("observation_pipeline must be ObservationPipeline")
        if not isinstance(self.iod, IodStageConfig):
            raise TypeError("iod must be IodStageConfig")
        if not isinstance(self.refinement, RefinementRunConfig):
            raise TypeError("refinement must be RefinementRunConfig")
        if not isinstance(self.resources, ResourceBindings):
            raise TypeError("resources must be ResourceBindings")
        descriptors = self.observation_pipeline._descriptors()
        transforms_json = None if descriptors is None else _canonical_json(descriptors)
        object.__setattr__(
            self,
            "_native",
            _rust.explicit_od_config(
                transforms_json,
                self.resources._native_paths(),
                self.iod._native,
                self.refinement._native,
            ),
        )

    @classmethod
    def explicit(
        cls,
        *,
        observation_pipeline: ObservationPipeline,
        iod: IodStageConfig,
        refinement: RefinementRunConfig,
        resources: ResourceBindings,
    ) -> OdConfig:
        return cls(observation_pipeline, iod, refinement, resources)

    @classmethod
    def _from_native(
        cls,
        native: Any,
        *,
        observation_pipeline: ObservationPipeline,
        iod: IodStageConfig,
        refinement: RefinementRunConfig,
        resources: ResourceBindings,
    ) -> OdConfig:
        value = object.__new__(cls)
        object.__setattr__(value, "observation_pipeline", observation_pipeline)
        object.__setattr__(value, "iod", iod)
        object.__setattr__(value, "refinement", refinement)
        object.__setattr__(value, "resources", resources)
        object.__setattr__(
            value, "profile_identity", _profile_identity_from_native(native)
        )
        object.__setattr__(value, "_native", native)
        return value

    def for_iod(self) -> IodRunConfig:
        return IodRunConfig._from_native(
            self._native.for_iod(),
            observation_pipeline=self.observation_pipeline,
            iod=self.iod,
            resources=self.resources,
        )

    def for_refinement(self) -> RefinementRunConfig:
        return RefinementRunConfig._from_native(self._native.for_refinement())

    def for_evaluation(self) -> EvaluationConfig:
        return EvaluationConfig._from_native(self._native.for_evaluation())

    @property
    def provenance(self) -> Mapping[str, object]:
        value = cast(dict[str, object], json.loads(self._native.provenance_json))
        return MappingProxyType(value)

    @property
    def profile_name(self) -> str | None:
        return None if self.profile_identity is None else self.profile_identity.name

    @property
    def profile_definition_sha256(self) -> str | None:
        return (
            None
            if self.profile_identity is None
            else self.profile_identity.definition_sha256
        )


__all__ = [
    "AcceptanceName",
    "Cmc2003Refinement",
    "CorrectionConfig",
    "EvaluationConfig",
    "IodRunConfig",
    "IodStageConfig",
    "IodSearchConfig",
    "JacobianName",
    "LossName",
    "OdConfig",
    "OdConfigurationError",
    "OdResourceValidationError",
    "ObservationSelectionName",
    "PrefitScopeName",
    "ProfileIdentity",
    "RangingConfig",
    "RefinementRunConfig",
    "ResourceBindings",
    "SpanSearchConfig",
    "TwoBodyFitConfig",
    "VelocityMethodName",
    "WhitenedFitConfig",
    "WorstResidualRefinement",
]
