"""Real ABI-v2 capsule probes for documented Python workflow exceptions."""

from __future__ import annotations

import ctypes
from importlib.resources import files
from typing import Any, Callable

import pytest

from ...observations import Associations, SourceCatalog
from ...propagator.propagator import NativeBackend, Propagator
from .. import (
    FittedOrbitMembers,
    FittedOrbits,
    IodRunConfig,
    IodStageConfig,
    ObservationPipeline,
    OdConfig,
    RefinementRunConfig,
    ResourceBindings,
    evaluate_orbits,
    refine,
    run_iod,
    run_od,
)
from ..evaluate import OrbitDeterminationObservations
from .test_batch_workflows import _two_body_source

_PROPAGATE = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p)


class _ExternalPropagatorV2(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_size_t),
        ("capabilities", ctypes.c_uint64),
        ("integration_time_scale", ctypes.c_char_p),
        ("two_body_dynamics", ctypes.c_uint8),
        ("name", ctypes.c_char_p),
        ("version", ctypes.c_char_p),
        ("context", ctypes.c_void_p),
        ("propagate", _PROPAGATE),
        ("propagate_stm", ctypes.c_void_p),
        ("reserved", ctypes.c_size_t * 4),
        ("configuration_fingerprint", ctypes.c_char_p),
        ("data_fingerprint", ctypes.c_char_p),
    ]


assert ctypes.sizeof(_ExternalPropagatorV2) == 128
_PY_CAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
_PY_CAPSULE_NEW.restype = ctypes.py_object
_PY_CAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_CAPSULE_NAME = ctypes.create_string_buffer(b"adam_core.propagator.v2")
_KEEP_ALIVE: list[object] = [_CAPSULE_NAME]


def _table(
    *,
    abi: int = 2,
    size: int = 128,
    status: int = 1,
    fingerprints: bool = True,
    capabilities: int = 1,
    name: bytes = b"probe",
) -> _ExternalPropagatorV2:
    def callback(_context: object, _request: object) -> int:
        return status

    callback_pointer = _PROPAGATE(callback)
    context = ctypes.c_int(7)
    table = _ExternalPropagatorV2(
        abi_version=abi,
        struct_size=size,
        capabilities=capabilities,
        integration_time_scale=b"tdb",
        two_body_dynamics=0,
        name=name,
        version=b"1.0.0",
        context=ctypes.cast(ctypes.pointer(context), ctypes.c_void_p),
        propagate=callback_pointer,
        propagate_stm=None,
        configuration_fingerprint=b"a" * 64 if fingerprints else None,
        data_fingerprint=b"b" * 64 if fingerprints else None,
    )
    _KEEP_ALIVE.extend([callback_pointer, context, table])
    return table


def _provider(
    table: _ExternalPropagatorV2, *, declared_name: str = "probe"
) -> Propagator:
    class ProbePropagator(Propagator):
        native_backend = NativeBackend(
            name=declared_name,
            version="1.0.0",
            work_units=frozenset({"propagator_capsule"}),
        )

        def propagate_orbits(  # type: ignore[override]
            self, *args: Any, **kwargs: Any
        ) -> None:
            raise AssertionError("Python propagation must not run")

        def generate_ephemeris(  # type: ignore[override]
            self, *args: Any, **kwargs: Any
        ) -> None:
            raise AssertionError("Python ephemeris generation must not run")

        def propagator_capsule(self) -> object:
            return _PY_CAPSULE_NEW(ctypes.addressof(table), _CAPSULE_NAME, None)

    return ProbePropagator()


@pytest.fixture(scope="module")
def _source() -> SourceCatalog:
    data = files("adam_core.orbit_determination.tests.data")
    return _two_body_source(
        (
            FittedOrbits.from_parquet(str(data.joinpath("pure_iod_orbit.parquet"))),
            FittedOrbitMembers.from_parquet(
                str(data.joinpath("pure_iod_orbit_members.parquet"))
            ),
            OrbitDeterminationObservations.from_parquet(
                str(data.joinpath("pure_iod_orbit_observations.parquet"))
            ),
        )
    )


@pytest.fixture(scope="module")
def _associations(_source: SourceCatalog) -> Associations:
    return Associations.from_kwargs(
        detection_id=_source.id, object_id=["group"] * len(_source)
    )


@pytest.fixture(scope="module")
def _config() -> OdConfig:
    return OdConfig.explicit(
        observation_pipeline=ObservationPipeline.as_supplied(),
        iod=IodStageConfig.core_v1(),
        refinement=RefinementRunConfig.core_v1(),
        resources=ResourceBindings({}),
    )


@pytest.fixture(scope="module")
def _handoff(
    _source: SourceCatalog, _associations: Associations, _config: OdConfig
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    orbits, members, failures = run_iod(_source, _associations, None, _config.for_iod())
    assert len(orbits) == 1 and len(failures) == 0
    return orbits, members


def test_real_capsule_backend_panic_and_protocol_failures_are_runtime_errors(
    _source: SourceCatalog,
    _associations: Associations,
    _config: OdConfig,
    _handoff: tuple[FittedOrbits, FittedOrbitMembers],
) -> None:
    orbits, members = _handoff
    invocations: list[tuple[str, Callable[[], object]]] = [
        (
            "run_iod backend error",
            lambda: run_iod(
                _source,
                _associations,
                _provider(_table(status=1)),
                _config.for_iod(),
            ),
        ),
        (
            "run_od provider panic",
            lambda: run_od(
                _source, _associations, _provider(_table(status=2)), _config
            ),
        ),
        (
            "refine protocol violation",
            lambda: refine(
                orbits,
                members,
                _source,
                _provider(_table(status=3)),
                _config.for_refinement(),
            ),
        ),
        (
            "evaluate backend error",
            lambda: evaluate_orbits(
                orbits,
                members,
                _source,
                _provider(_table(status=1)),
                _config.for_evaluation(),
            ),
        ),
    ]
    for label, invoke in invocations:
        with pytest.raises(RuntimeError) as caught:
            invoke()
        assert str(caught.value), label


def test_real_capsule_invalid_identity_and_provenance_are_value_errors(
    _source: SourceCatalog,
    _associations: Associations,
    _config: OdConfig,
) -> None:
    invalid_providers = [
        _provider(_table(size=112, fingerprints=False)),
        _provider(_table(), declared_name="different"),
        _provider(_table(capabilities=1 | (1 << 63))),
    ]
    for provider in invalid_providers:
        with pytest.raises(ValueError):
            run_iod(_source, _associations, provider, _config.for_iod())


def test_real_capsule_abi_and_required_capability_protocol_errors_are_runtime_errors(
    _source: SourceCatalog,
    _associations: Associations,
    _config: OdConfig,
) -> None:
    for provider in [_provider(_table(abi=1)), _provider(_table(capabilities=0))]:
        with pytest.raises(RuntimeError):
            run_iod(_source, _associations, provider, _config.for_iod())


def test_wrong_python_provider_type_remains_type_error(
    _source: SourceCatalog,
    _associations: Associations,
    _config: OdConfig,
) -> None:
    with pytest.raises(TypeError):
        run_iod(_source, _associations, object(), _config.for_iod())  # type: ignore[arg-type]


def test_explicit_iod_config_is_still_constructible_without_resources() -> None:
    config = IodRunConfig(
        ObservationPipeline.as_supplied(),
        IodStageConfig.core_v1(),
        ResourceBindings({}),
    )
    assert config.profile_name is None
