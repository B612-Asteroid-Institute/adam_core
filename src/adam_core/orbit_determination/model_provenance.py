"""Explicit provenance contract for custom observation models.

adam-core's own observation models are identified by the exact native build
commit and behavior graph. This value can describe a provider declaration, but
it cannot authenticate live executable Python behavior and does not make a
custom model acceptable to ``run_od``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Immutable identity and configuration declared by a custom model.

    Parameters
    ----------
    package
        Distribution/project name asserted by the provider.
    version
        Version asserted by the provider.
    implementation_fingerprint
        Lowercase 64-hex SHA-256 (or equivalent content fingerprint) chosen by
        the provider. A science-affecting implementation change must use a new
        fingerprint.
    configuration
        Complete model configuration. adam-core canonicalizes supported values
        deterministically and rejects unsupported values before model work.
    """

    package: str
    version: str
    implementation_fingerprint: str
    configuration: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _FieldSchema:
    """Closed, non-executable vocabulary for one shipped instance field."""

    exact_types: tuple[type[Any], ...]
    item_schema: "_FieldSchema | None" = None
    mapping_keys: tuple[str, ...] = ()
    mapping_value_schema: "_FieldSchema | None" = None
    allowed_values: tuple[object, ...] = ()
    ndarray_dtypes: tuple[str, ...] = ()
    ndim: int | None = None
    shape: tuple[int, ...] = ()


def _field_schema(
    *exact_types: type[Any],
    item_schema: _FieldSchema | None = None,
    mapping_keys: tuple[str, ...] = (),
    mapping_value_schema: _FieldSchema | None = None,
    allowed_values: tuple[object, ...] = (),
    ndarray_dtypes: tuple[str, ...] = (),
    ndim: int | None = None,
    shape: tuple[int, ...] = (),
) -> _FieldSchema:
    return _FieldSchema(
        exact_types,
        item_schema,
        mapping_keys,
        mapping_value_schema,
        allowed_values,
        ndarray_dtypes,
        ndim,
        shape,
    )


@dataclass(frozen=True, slots=True)
class _BehaviorSpec:
    """Exact shipped class behavior used by built-in provenance checks."""

    model_type: type[Any]
    class_members: tuple[tuple[str, object], ...]
    exact_mro: tuple[type[Any], ...]
    class_surfaces: tuple[tuple[type[Any], tuple[tuple[str, object], ...]], ...]
    instance_fields: tuple[tuple[str, _FieldSchema], ...] = ()
    nested_model_fields: tuple[tuple[str, type[Any]], ...] = ()
    collaborators: tuple[tuple[str, "_BehaviorSpec"], ...] = ()
    module_delegates: tuple[tuple[Mapping[str, object], str, object], ...] = ()


def _capture_behavior_spec(
    model_type: type[Any],
    member_names: tuple[str, ...],
    *,
    instance_fields: tuple[tuple[str, _FieldSchema], ...] = (),
    nested_model_fields: tuple[tuple[str, type[Any]], ...] = (),
    collaborators: tuple[tuple[str, _BehaviorSpec], ...] = (),
    module_delegates: tuple[tuple[Mapping[str, object], str, object], ...] = (),
) -> _BehaviorSpec:
    """Capture explicitly named shipped members before callers can patch them."""
    members: list[tuple[str, object]] = []
    for name in member_names:
        for owner in model_type.__mro__:
            if name in vars(owner):
                members.append((name, vars(owner)[name]))
                break
        else:  # pragma: no cover - import-time registry programming error
            raise RuntimeError(
                f"registered behavior member {model_type.__qualname__}.{name} is missing"
            )
    field_names = tuple(name for name, _ in instance_fields)
    if len(set(field_names)) != len(field_names) or any(
        not schema.exact_types for _, schema in instance_fields
    ):
        raise RuntimeError(
            f"registered instance field schema for {model_type.__qualname__} is invalid"
        )
    recursive_names = {
        *(name for name, _ in nested_model_fields),
        *(name for name, _ in collaborators),
    }
    if not recursive_names <= set(field_names):
        raise RuntimeError(
            f"recursive fields for {model_type.__qualname__} lack a value schema"
        )
    exact_mro = type.__getattribute__(model_type, "__mro__")
    metaclass_mro = type.__getattribute__(type(model_type), "__mro__")
    # Inherited ABC behavior and metaclass dispatch are mutable Python state too.
    # Capture every owner, rather than assuming non-adam-core ancestors are inert.
    surface_owners = tuple(dict.fromkeys((*exact_mro, *metaclass_mro)))
    surfaces = tuple(
        (owner, tuple(type.__getattribute__(owner, "__dict__").items()))
        for owner in surface_owners
    )
    return _BehaviorSpec(
        model_type,
        tuple(members),
        exact_mro,
        surfaces,
        instance_fields,
        nested_model_fields,
        collaborators,
        module_delegates,
    )


@runtime_checkable
class ProvenancedModel(Protocol):
    """Structural provenance hook implemented by custom observation models.

    This protocol describes only a provider declaration. It does not
    authenticate executable Python behavior; ``run_od`` accepts only exact
    shipped Core models with declarative Core-Rust execution.
    """

    def model_provenance(self) -> ModelProvenance:
        """Return this model instance's explicit immutable provenance."""
